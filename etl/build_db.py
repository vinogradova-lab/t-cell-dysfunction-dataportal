"""ETL: raw CSVs (in ./source/) -> normalized parquet tables + bulk-download bundle.

Run order for a fresh build:
    python scripts/sync_source.py     # stage raw inputs into ./source/
    python etl/build_db.py            # produce ./data/parquet + ./data/downloads

Everything is served as **log2 fold-change from D2**:
  * whole proteome & reactivity are percent-of-control (D2=100) -> log2(value/100)
  * RNA DESeq2 output is already log2FoldChange vs D2
Condition names are harmonized to the compact codes D2/D4A/D4C/D8A/D8C
(A=Acute, C=Chronic); the ATP add-back experiment adds the ±ATP variants.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from gene_index import build_gene_registry

PORTAL_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PORTAL_ROOT / "source"
PARQUET_DIR = PORTAL_ROOT / "data" / "parquet"
DOWNLOAD_DIR = PORTAL_ROOT / "data" / "downloads"


@lru_cache(maxsize=1)
def protein_coding_symbols() -> frozenset[str]:
    """Human protein-coding gene symbols (plus their aliases), from the NCBI
    Gene dump staged in ``source/genes_ncbi_9606_proteincoding.py``.

    Used to keep only protein-coding rows in the RNA-seq tables — the raw
    DESeq2 output covers the whole annotated transcriptome (lncRNAs, miRNAs,
    pseudogenes, …), which roughly triples the gene count. Aliases are included
    so coding genes whose RNA symbol differs from the current NCBI symbol are
    still retained.
    """
    path = SOURCE / "genes_ncbi_9606_proteincoding.py"
    spec = importlib.util.spec_from_file_location(
        "genes_ncbi_9606_proteincoding", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    syms: set[str] = set()
    for nt in mod.GENEID2NT.values():
        syms.add(nt.Symbol)
        syms.update(nt.Aliases)
    return frozenset(syms)


def _keep_protein_coding(df: pd.DataFrame) -> pd.DataFrame:
    """Filter an RNA table down to protein-coding symbols (+ aliases)."""
    return df[df["symbol"].isin(protein_coding_symbols())].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_s2_1() -> pd.DataFrame:
    """Supplementary sheet S2-1 (whole proteome): per-replicate census values,
    the volcano/significance block, and functional-group flags. Two title rows
    precede the header, so the real header is the third row (``header=2``)."""
    return pd.read_excel(
        SOURCE / "data_s2.xlsx",
        sheet_name="S2-1 Unenriched proteomics",
        header=2,
    )


@lru_cache(maxsize=1)
def load_s1_1() -> pd.DataFrame:
    """Supplementary sheet S1-1 (bulk RNA-seq): DESeq2 results vs D2, one wide
    block per comparison. Two title rows precede the header (``header=2``)."""
    return pd.read_excel(SOURCE / "data_s1.xlsx", sheet_name=0, header=2)


# whole-proteome / reactivity condition columns, in display order
FIVE_CONDITIONS = ["D2", "D4A", "D4C", "D8A", "D8C"]
# vs-D2 comparisons we surface, in display order (mirrors FIVE_CONDITIONS - D2)
FOUR_COMPARISONS = ["D4A", "D4C", "D8A", "D8C"]


def _pct_to_log2fc(pct: pd.Series) -> pd.Series:
    """percent-of-control (D2=100) -> log2 fold-change from D2.

    A percent_control of 0 (fully lost signal) -> log2(0) = -inf; we surface
    that as NaN rather than an infinity that would break plotting/parquet.
    """
    with np.errstate(divide="ignore"):
        lfc = np.log2(pct.astype(float) / 100.0)
    return lfc.replace([np.inf, -np.inf], np.nan)


def _first_residue_loc(residue: str) -> float:
    """'C379,C382' / 'C379; C382' -> 379.0 ; NaN if unparseable."""
    if not isinstance(residue, str):
        return np.nan
    for token in residue.replace(";", ",").split(","):
        token = token.strip().lstrip("Cc")
        if token.isdigit():
            return float(token)
    return np.nan


# --------------------------------------------------------------------------- #
# per-modality builders
# --------------------------------------------------------------------------- #
# whole-proteome biological replicates present in S2-1 (six per condition)
PROTEOME_REPS = [1, 2, 3, 4, 5, 6]


def _s2_1_chan_mean(df: pd.DataFrame, cond: str, rep: int) -> pd.Series:
    """Mean of the two technical sub-measurements (``_1``/``_2``) for one
    (condition x biological replicate) block of raw census values in S2-1."""
    cols = [f"{cond}_rep{rep}_{sub}_processed_census-out" for sub in (1, 2)]
    return df[cols].mean(axis=1)


def _s2_1_replicate_pct() -> pd.DataFrame:
    """Per-(protein x condition x biological replicate) percent-of-control from
    the S2-1 raw census values.

    Within each replicate the two technical sub-measurements are averaged, then
    every condition is scaled so that replicate's D2 mean = 100 — reproducing the
    manuscript's percent-of-control normalization.
    """
    df = load_s2_1()
    ids = df[["uniprot", "protein"]].rename(columns={"protein": "symbol"})
    frames = []
    for rep in PROTEOME_REPS:
        d2 = _s2_1_chan_mean(df, "D2", rep)
        for cond in FIVE_CONDITIONS:
            block = ids.copy()
            block["condition"] = cond
            block["rep"] = f"rep{rep}"
            pct = 100.0 * _s2_1_chan_mean(df, cond, rep) / d2
            block["percent_control"] = pct.values
            frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["percent_control"])


def build_proteome_replicates() -> pd.DataFrame:
    """Per-replicate whole-proteome percent-of-control -> long log2FC table.

    The portal bar for each condition is later drawn as the mean of these
    replicate values, so the overlaid points center on the bar.
    """
    out = _s2_1_replicate_pct()
    out["log2fc"] = _pct_to_log2fc(out["percent_control"])
    out = out.dropna(subset=["log2fc"])
    out["condition"] = pd.Categorical(
        out["condition"], categories=FIVE_CONDITIONS, ordered=True
    )
    out = out.sort_values(["symbol", "condition", "rep"])
    return out[["uniprot", "symbol", "condition", "rep", "percent_control", "log2fc"]]


def build_proteome() -> pd.DataFrame:
    """Aggregated whole-proteome table: mean percent-of-control across replicates,
    per (protein x condition). Derived from the same replicate values the portal
    overlays, so bars (mean of replicates) and the aggregate agree by construction.
    """
    reps = _s2_1_replicate_pct()
    agg = reps.groupby(
        ["uniprot", "symbol", "condition"], as_index=False
    )["percent_control"].mean()
    agg["log2fc"] = _pct_to_log2fc(agg["percent_control"])
    agg["condition"] = pd.Categorical(
        agg["condition"], categories=FIVE_CONDITIONS, ordered=True
    )
    agg = agg.sort_values(["symbol", "condition"])
    return agg[["uniprot", "symbol", "condition", "percent_control", "log2fc"]]


# sample column prefix (e.g. "Sample_D8.Chr.2_IGO_...") -> (condition, rep)
_RNA_SAMPLE_RE = re.compile(r"^Sample_(D2|D4|D8)(?:\.(Ac|Chr))?\.(\d+)_")


def _rna_sample_meta(col: str) -> tuple[str, str] | None:
    """'Sample_D8.Chr.2_IGO_13524_14' -> ('D8C', 'rep2'); D2 -> ('D2', ...)."""
    m = _RNA_SAMPLE_RE.match(col)
    if not m:
        return None
    day, ac, rep = m.group(1), m.group(2), m.group(3)
    cond = day if day == "D2" else day + ("A" if ac == "Ac" else "C")
    return cond, f"rep{rep}"


def build_rna_replicates() -> pd.DataFrame:
    """Per-sample VST-normalized counts -> long per-replicate log2FC from D2.

    ``normalized_counts.txt`` holds VST (log2-scale) counts, one column per
    sample. Per gene, each non-D2 sample's log2FC is its value minus the mean of
    that gene's D2 replicates. D2 itself is the reference (log2FC = 0) and is not
    emitted, matching the RNA figure's four displayed conditions.
    """
    counts = pd.read_csv(SOURCE / "rna_counts.txt", sep="\t", index_col=0)
    meta = {c: _rna_sample_meta(c) for c in counts.columns}
    meta = {c: m for c, m in meta.items() if m is not None}
    d2_cols = [c for c, (cond, _) in meta.items() if cond == "D2"]
    d2_mean = counts[d2_cols].mean(axis=1)

    order = ["D4A", "D4C", "D8A", "D8C"]
    rows = []
    for col, (cond, rep) in meta.items():
        if cond == "D2":
            continue
        lfc = counts[col] - d2_mean
        rows.append(
            pd.DataFrame(
                {"symbol": counts.index, "condition": cond, "rep": rep, "log2fc": lfc.values}
            )
        )
    out = pd.concat(rows, ignore_index=True).dropna(subset=["log2fc"])
    out = _keep_protein_coding(out)
    out["condition"] = pd.Categorical(out["condition"], categories=order, ordered=True)
    out = out.sort_values(["symbol", "condition", "rep"])
    return out[["symbol", "condition", "rep", "log2fc"]]


def build_rna() -> pd.DataFrame:
    """Bulk RNA-seq DESeq2 results vs D2 -> long table, one row per (gene x
    condition). Source is S1-1, wide: one block of ``baseMean_/log2FoldChange_/
    padj_<cond>_vs_D2`` columns per comparison.
    """
    df = load_s1_1().rename(columns={"gene_name": "symbol"})
    frames = []
    for cond in FOUR_COMPARISONS:
        block = pd.DataFrame(
            {
                "symbol": df["symbol"],
                "condition": cond,
                "log2fc": df[f"log2FoldChange_{cond}_vs_D2"],
                "padj": df[f"padj_{cond}_vs_D2"],
                "base_mean": df[f"baseMean_{cond}_vs_D2"],
            }
        )
        frames.append(block)
    rna = pd.concat(frames, ignore_index=True)
    # keep only protein-coding genes (drops lncRNA/miRNA/pseudogene rows)
    rna = _keep_protein_coding(rna)
    rna["condition"] = pd.Categorical(
        rna["condition"], categories=FOUR_COMPARISONS, ordered=True
    )
    return rna


def build_reactivity() -> pd.DataFrame:
    df = pd.read_csv(SOURCE / "reactivity_5cond.csv", index_col=0)
    df = df.rename(columns={"protein": "symbol", "LFC_tmt_abpp": "log2fc"})
    df["residue_loc"] = df["residue"].map(_first_residue_loc)
    # whole-proteome (protein-expression) log2FC vs D2, for the dot-plot's
    # expression triangles. Recompute from percent-of-control rather than trust
    # the precomputed LFC_WP column, which carries ±inf where signal was lost.
    df["wp_log2fc"] = _pct_to_log2fc(df["whole_proteome"])
    order = ["D4A", "D4C", "D8A", "D8C"]
    df["condition"] = pd.Categorical(
        df["condition"], categories=order, ordered=True
    )
    cols = [
        "uniprot",
        "symbol",
        "residue",
        "residue_loc",
        "sequence" if "sequence" in df.columns else None,
        "condition",
        "tmt_abpp",
        "log2fc",
        "reactivity_change",
        "wp_log2fc",
    ]
    cols = [c for c in cols if c is not None]
    out = df[cols].rename(columns={"tmt_abpp": "percent_control"})
    return out


# functional-group flags carried through from the S2-1 volcano block (global,
# i.e. not per-comparison).
VOLCANO_FLAGS = [
    "mitochondrial",
    "peroxisome",
    "redox_related",
    "cell_cycle",
    "nucleotide_metabolism",
    "endoplasmic_reticulum",
]
# vs-D2 comparisons we surface, in display order (mirrors FIVE_CONDITIONS − D2)
VOLCANO_COMPARISONS = FOUR_COMPARISONS


def _volcano_suffix_by_cond(df: pd.DataFrame) -> dict[str, str]:
    """Map each comparison code (e.g. "D4A") -> the exact S2-1 column suffix.

    S2-1 volcano columns are suffixed with a label like
    ``…_Exhaustion WP - D2 vs. D4A``. The values already carry the
    <cond>-vs-D2 sign (up in dysfunction = positive), despite the reversed
    label wording. We match only the ``D2 vs. …`` blocks, which selects the
    four vs-D2 comparisons and ignores the extra D8A-vs-D4A/D4C/D8C blocks.
    """
    suffix_by_cond: dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"log2_FC_(.* - D2 vs\. (D\w+) .*)", str(col))
        if m:
            suffix_by_cond[m.group(2)] = m.group(1)
    return suffix_by_cond


def build_volcano() -> pd.DataFrame:
    """Whole-proteome volcano data -> long table, one row per (protein × vs-D2
    comparison). Read from the S2-1 sheet's volcano block.
    """
    df = load_s2_1()
    suffix_by_cond = _volcano_suffix_by_cond(df)
    flags = [f for f in VOLCANO_FLAGS if f in df.columns]
    frames = []
    for cond in VOLCANO_COMPARISONS:
        suffix = suffix_by_cond.get(cond)
        if suffix is None:
            continue
        block = pd.DataFrame(
            {
                "uniprot": df["uniprot"],
                "symbol": df["protein"],
                "description": df["description"],
                "comparison": cond,
                "log2fc": df[f"log2_FC_{suffix}"],
                "p_value": df[f"p_value_{suffix}"],
                "neglog10_pval": df[f"-log10_pval_{suffix}"],
                "neglog10_padj": df[f"-log10_pval_adj_{suffix}"],
                "regulation": df[f"Regulation_{suffix}"],
            }
        )
        for f in flags:
            block[f] = df[f].astype("boolean")
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    # drop proteins with no measured FC in a comparison (all-NaN rows)
    out = out.dropna(subset=["log2fc", "neglog10_padj"]).reset_index(drop=True)
    out["comparison"] = pd.Categorical(
        out["comparison"], categories=VOLCANO_COMPARISONS, ordered=True
    )
    return out


def build_proteome_download() -> pd.DataFrame:
    """Single combined whole-proteome table for the bulk download, mirroring the
    layout of supplementary sheet S2-1: identity columns, per-biological-replicate
    percent-of-control values (technical pairs averaged, D2-normalized), the
    per-comparison volcano/significance columns, and the functional-group flags.
    """
    df = load_s2_1()
    out = df[["uniprot", "protein", "description"]].rename(
        columns={"protein": "symbol"}
    )
    # per-replicate percent-of-control: D{cond}_rep{N}, grouped by condition
    pct = {
        (cond, rep): 100.0
        * _s2_1_chan_mean(df, cond, rep)
        / _s2_1_chan_mean(df, "D2", rep)
        for rep in PROTEOME_REPS
        for cond in FIVE_CONDITIONS
    }
    for cond in FIVE_CONDITIONS:
        for rep in PROTEOME_REPS:
            out[f"{cond}_rep{rep}"] = pct[(cond, rep)].values
    # volcano/significance block, one group of columns per vs-D2 comparison
    suffix_by_cond = _volcano_suffix_by_cond(df)
    for cond in VOLCANO_COMPARISONS:
        suffix = suffix_by_cond.get(cond)
        if suffix is None:
            continue
        out[f"{cond}_log2fc"] = df[f"log2_FC_{suffix}"]
        out[f"{cond}_p_value"] = df[f"p_value_{suffix}"]
        out[f"{cond}_neglog10_pval"] = df[f"-log10_pval_{suffix}"]
        out[f"{cond}_neglog10_padj"] = df[f"-log10_pval_adj_{suffix}"]
        out[f"{cond}_regulation"] = df[f"Regulation_{suffix}"]
    # functional-group flags (global, not per-comparison)
    for f in VOLCANO_FLAGS:
        if f in df.columns:
            out[f] = df[f].astype("boolean")
    return out


def build_reactivity_atp() -> pd.DataFrame:
    df = pd.read_csv(
        SOURCE / "reactivity_atp.csv",
        usecols=[
            "uniprot",
            "protein",
            "residue",
            "residue_loc",
            "condition",
            "percent_control",
            "experiment",
            "technical_replicate",
        ],
    )
    df = df.rename(columns={"protein": "symbol"})
    df["lfc"] = _pct_to_log2fc(df["percent_control"])
    # replicate label: biological experiment + technical replicate, e.g. "d1.2"
    df["rep"] = (
        df["experiment"].astype(str)
        + "."
        + df["technical_replicate"].astype("Int64").astype(str)
    )
    order = ["D2", "D2-ATP", "D8A", "D8A-ATP", "D8C", "D8C-ATP"]
    df = df[df["condition"].isin(order)].copy()
    df["condition"] = pd.Categorical(
        df["condition"], categories=order, ordered=True
    )
    return df[
        ["uniprot", "symbol", "residue", "residue_loc", "condition",
         "rep", "percent_control", "lfc"]
    ]


# --------------------------------------------------------------------------- #
# bulk-download bundle
# --------------------------------------------------------------------------- #
DATA_DICTIONARY = """\
T cell dysfunction proteomics data portal — bulk download
=========================================================

All fold changes are expressed as log2 fold-change relative to the D2
(baseline / non-dysfunctional) condition.

Conditions (A = Acute, C = Chronic stimulation; number = day):
  D2   baseline / reference   D4A  day-4 acute    D4C  day-4 chronic
  D8A  day-8 acute            D8C  day-8 chronic
ATP add-back experiment adds a "+ATP" variant of D2/D8A/D8C.

Files
-----
whole_proteome.csv
    Whole-proteome protein expression + differential-expression significance,
    one row per protein (wide, mirroring supplementary sheet S2-1).
      * D{cond}_rep{N}: per-biological-replicate percent-of-control (D2=100),
        for cond in D2/D4A/D4C/D8A/D8C and N in 1..6 (technical pairs averaged,
        each replicate scaled so its D2 = 100).
      * {cond}_log2fc / _p_value / _neglog10_pval / _neglog10_padj / _regulation:
        volcano statistics for each cond-vs-D2 comparison (cond in D4A/D4C/D8A/D8C).
        regulation is the published significance call.
      * mitochondrial, peroxisome, redox_related, cell_cycle,
        nucleotide_metabolism, endoplasmic_reticulum: boolean functional-group
        flags (global, not per-comparison).
    columns: uniprot, symbol, description, D2_rep1..D8C_rep6, per-comparison
             volcano columns, functional-group flags

rna.csv
    Bulk RNA-seq differential expression vs D2 (DESeq2). One row per (gene x condition).
    Restricted to protein-coding genes (NCBI Gene, tax 9606); non-coding
    transcripts (lncRNA, miRNA, pseudogenes, …) are excluded.
    columns: symbol, condition, log2fc, padj, base_mean

rna_replicates.csv
    Per-replicate RNA log2FC from VST-normalized counts (value minus per-gene D2
    mean). One row per (gene x condition x replicate). Note these differ from the
    model-based DESeq2 log2fc in rna.csv. Protein-coding genes only (as rna.csv).
    columns: symbol, condition, rep, log2fc

reactivity_5cond.csv
    Cysteine reactivity across the 5 conditions. One row per (cysteine site x condition).
    wp_log2fc is the whole-proteome (protein-expression) log2FC vs D2, repeated
    per cysteine of a protein (used for the dot-plot expression triangles).
    columns: uniprot, symbol, residue, residue_loc, [sequence], condition,
             percent_control (D2=100), log2fc, reactivity_change, wp_log2fc

reactivity_atp.csv
    Cysteine reactivity ATP add-back, per-replicate. One row per measurement.
    rep is the biological experiment + technical replicate (e.g. "d1.2").
    columns: uniprot, symbol, residue, residue_loc, condition, rep,
             percent_control (D2=100), lfc

Provenance: whole proteome and bulk RNA-seq are read from the manuscript
supplementary workbooks (Data S1, Data S2); the remaining tables are derived
from the analysis repository t-cell-dysfunction-2026.
"""

# (parquet-table-name, download-basename) — tables written verbatim to the bundle
DOWNLOAD_TABLES = [
    ("rna", "rna"),
    ("rna_replicates", "rna_replicates"),
    ("reactivity", "reactivity_5cond"),
    ("reactivity_atp", "reactivity_atp"),
]
# download basenames written to the zip, in order (combined WP first)
DOWNLOAD_BASENAMES = ["whole_proteome"] + [b for _, b in DOWNLOAD_TABLES]


def write_downloads(tables: dict[str, pd.DataFrame]) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # combined whole-proteome (expression + replicates + significance), one file
    build_proteome_download().to_csv(
        DOWNLOAD_DIR / "whole_proteome.csv", index=False
    )
    for tbl_name, basename in DOWNLOAD_TABLES:
        df = tables[tbl_name]
        df.to_csv(DOWNLOAD_DIR / f"{basename}.csv", index=False)

    readme = DOWNLOAD_DIR / "README.txt"
    readme.write_text(DATA_DICTIONARY)

    zip_path = DOWNLOAD_DIR / "t_cell_dysfunction_proteomics.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme, "README.txt")
        for basename in DOWNLOAD_BASENAMES:
            zf.write(DOWNLOAD_DIR / f"{basename}.csv", f"{basename}.csv")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    if not SOURCE.exists() or not any(SOURCE.iterdir()):
        raise SystemExit(
            "source/ is empty — run `python scripts/sync_source.py` first."
        )

    # fresh output dirs
    for d in (PARQUET_DIR, DOWNLOAD_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("Building modality tables...")
    tables = {
        "proteome": build_proteome(),
        "proteome_replicates": build_proteome_replicates(),
        "rna": build_rna(),
        "rna_replicates": build_rna_replicates(),
        "reactivity": build_reactivity(),
        "reactivity_atp": build_reactivity_atp(),
        "volcano": build_volcano(),
    }

    # search registry: seed from proteome (uniprot+symbol+description) plus any
    # RNA-only symbols so gene-only hits are searchable too.
    seed = load_s2_1()[["uniprot", "protein", "description"]].rename(
        columns={"protein": "symbol"}
    )
    rna_only = pd.DataFrame(
        {"symbol": sorted(set(tables["rna"]["symbol"]) - set(seed["symbol"]))}
    )
    rna_only["uniprot"] = pd.NA
    rna_only["description"] = pd.NA
    seed = pd.concat([seed, rna_only], ignore_index=True)
    tables["genes"] = build_gene_registry(seed)

    print("Writing parquet tables...")
    for name, df in tables.items():
        # categoricals -> str for portable parquet/round-trip
        out = df.copy()
        for c in out.columns:
            if str(out[c].dtype) == "category":
                out[c] = out[c].astype(str)
        out.to_parquet(PARQUET_DIR / f"{name}.parquet", index=False)
        print(f"  {name:16s} {len(df):>8,d} rows")

    print("Writing bulk-download bundle...")
    write_downloads(tables)
    for f in sorted(DOWNLOAD_DIR.iterdir()):
        print(f"  {f.name:40s} {f.stat().st_size/1e6:7.2f} MB")

    print("\nETL complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
