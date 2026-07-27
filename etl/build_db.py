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


# quantile of a replicate's positive census values used as its limit of
# detection. The per-replicate *minimum* is not usable: rep5's is ~20x lower
# than every other replicate's, which would turn a below-detection channel into
# a ~-8 log2FC outlier that dominates the mean.
DETECTION_QUANTILE = 0.01


@lru_cache(maxsize=1)
def _detection_floors() -> dict[int, float]:
    """Per-biological-replicate limit of detection, as the 1st percentile of
    that replicate's positive census values across all five conditions.

    A census value of exactly 0 means "below detection in this TMT channel",
    not "absent" — the protein was identified in that replicate (it has a
    peptide count) but one channel produced no signal. Substituting the
    detection limit turns those into censored bounds instead of dropping them.
    """
    df = load_s2_1()
    floors = {}
    for rep in PROTEOME_REPS:
        vals = pd.concat([_s2_1_chan_mean(df, c, rep) for c in FIVE_CONDITIONS])
        vals = vals[vals.notna() & (vals > 0)]
        floors[rep] = float(vals.quantile(DETECTION_QUANTILE)) if len(vals) else np.nan
    return floors


def _censor(num: pd.Series, den: pd.Series, floor: float) -> tuple[pd.Series, pd.Series]:
    """percent-of-control with below-detection channels censored at ``floor``.

    Returns ``(percent_control, censored)``. See :func:`_s2_1_replicate_pct`
    for the rule table; cells carrying no information (both sides below
    detection) come back as NaN for the caller to drop.
    """
    both_zero = (num == 0) & (den == 0)
    pct = (100.0 * num.mask(num == 0, floor) / den.mask(den == 0, floor))
    return pct.mask(both_zero), ((num == 0) | (den == 0)) & ~both_zero


@lru_cache(maxsize=1)
def _s2_1_channel_pct() -> pd.DataFrame:
    """Per-(protein x condition x biological replicate x technical channel)
    percent-of-control from the S2-1 raw census values.

    Each technical channel is divided by its biological replicate's D2 *mean*,
    so the two channels of a replicate share a denominator and their spread
    reflects measurement noise in the numerator alone. This is the same
    convention the manuscript's reactivity tables use (see
    :func:`_wp_median_pct`, which reproduces the published column exactly).

    Censoring follows the same rules as :func:`_s2_1_replicate_pct`, applied
    per channel — so a single dead channel is floored rather than dragging its
    whole replicate down through the census mean.

    ``rep`` is the display label ("rep3.1"), mirroring the ATP figure's
    experiment.technical convention; ``bio_rep`` ("rep3") is what donor-level
    consumers group by.
    """
    df = load_s2_1()
    floors = _detection_floors()
    ids = df[["uniprot", "protein"]].rename(columns={"protein": "symbol"})
    frames = []
    for rep in PROTEOME_REPS:
        d2 = _s2_1_chan_mean(df, "D2", rep)
        for cond in FIVE_CONDITIONS:
            for sub in (1, 2):
                chan = df[f"{cond}_rep{rep}_{sub}_processed_census-out"]
                pct, censored = _censor(chan, d2, floors[rep])
                block = ids.copy()
                block["condition"] = cond
                block["rep"] = f"rep{rep}.{sub}"
                block["bio_rep"] = f"rep{rep}"
                block["percent_control"] = pct.values
                block["censored"] = censored.values
                frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["percent_control"])


@lru_cache(maxsize=1)
def _wp_median_pct() -> pd.DataFrame:
    """Whole-proteome percent-of-control under the *reactivity* convention:
    the median over all technical channels, each divided by its biological
    replicate's D2 mean.

    This is deliberately a different statistic from :func:`build_proteome`,
    which takes the mean over biological replicates (each already the mean of
    its two technical channels). The manuscript's reactivity tables use the
    median-of-channels form, and reproducing it here is what lets the dot
    plot's expression triangles stay on one consistent footing — verified to
    reproduce the upstream ``whole_proteome`` column exactly on all 18,643
    cells where that column is finite.

    Used only to fill the triangles the upstream column cannot express (a D2
    reference of zero, which it reports as inf).
    """
    out = _s2_1_channel_pct()
    med = out.groupby(["uniprot", "condition"], as_index=False).agg(
        percent_control=("percent_control", "median"),
        censored=("censored", "any"),
    )
    med["wp_log2fc"] = _pct_to_log2fc(med["percent_control"])
    return med[["uniprot", "condition", "wp_log2fc", "censored"]]


def _s2_1_replicate_pct() -> pd.DataFrame:
    """Per-(protein x condition x biological replicate) percent-of-control from
    the S2-1 raw census values.

    Within each replicate the two technical sub-measurements are averaged, then
    every condition is scaled so that replicate's D2 mean = 100 — reproducing the
    manuscript's percent-of-control normalization.

    Zero census values are *censored*, not dropped, by substituting that
    replicate's limit of detection (see :func:`_detection_floors`):

    ======  =========  ===========================  ==========================
    D2      condition  percent_control              meaning
    ======  =========  ===========================  ==========================
    NaN     any        dropped                      absent from this replicate
    > 0     > 0        ``100 * v / d2``             normal measurement
    > 0     == 0       ``100 * floor / d2``         upper bound on the loss
    == 0    > 0        ``100 * v / floor``          lower bound on the increase
    == 0    == 0       dropped                      no information either way
    ======  =========  ===========================  ==========================

    The last row matters: flooring *both* sides would yield floor/floor = 100%,
    injecting a spurious "no change" into conditions where the protein was
    simply never seen. ``censored`` flags the two middle cases so downstream
    consumers can present them as bounds rather than point estimates.

    This is the single choke point feeding both :func:`build_proteome` and
    :func:`build_proteome_replicates`, so the aggregate and the per-replicate
    overlay are computed from an identical set of values by construction.
    """
    df = load_s2_1()
    ids = df[["uniprot", "protein"]].rename(columns={"protein": "symbol"})
    floors = _detection_floors()
    frames = []
    for rep in PROTEOME_REPS:
        d2 = _s2_1_chan_mean(df, "D2", rep)
        floor = floors[rep]
        for cond in FIVE_CONDITIONS:
            pct, censored = _censor(_s2_1_chan_mean(df, cond, rep), d2, floor)
            block = ids.copy()
            block["condition"] = cond
            block["rep"] = f"rep{rep}"
            block["percent_control"] = pct.values
            block["censored"] = censored.values
            frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["percent_control"])


def build_proteome_replicates() -> pd.DataFrame:
    """Per-technical-channel whole-proteome percent-of-control -> long log2FC
    table. This is what the portal overlays as dots on the abundance bars.

    Channels, not biological-replicate means: each donor contributes both of its
    technical measurements, so the overlay shows the measurement spread the
    donor mean would hide. The bar itself is drawn as the mean of these points
    (``figures._replicate_means``), so bar and dots stay consistent.

    ``bio_rep`` is retained so donor-level consumers — notably the bulk-download
    reconstruction in ``etl/prerender.py`` — can recover per-donor values by
    averaging a channel pair.
    """
    out = _s2_1_channel_pct().copy()  # cached frame; don't mutate in place
    out["log2fc"] = _pct_to_log2fc(out["percent_control"])
    out = out.dropna(subset=["log2fc"])
    out["condition"] = pd.Categorical(
        out["condition"], categories=FIVE_CONDITIONS, ordered=True
    )
    out = out.sort_values(["symbol", "condition", "rep"])
    return out[
        ["uniprot", "symbol", "condition", "rep", "bio_rep", "percent_control",
         "log2fc", "censored"]
    ]


@lru_cache(maxsize=1)
def _d2_below_detection() -> pd.DataFrame:
    """(uniprot, condition) pairs whose D2 reference was below detection in a
    replicate that contributes to that condition.

    Percent-of-control is undefined without a D2 reference: we censor it at the
    limit of detection, which makes the ratio a *lower bound* rather than a
    measurement. The published S2-1 statistics instead average the raw zero into
    the D2 denominator, which halves it and inflates the fold change — for
    AFAP1L2 D8C by ~1.19 log2, enough to make it the most extreme point in the
    D8C volcano. Those comparisons are dropped from the volcano (see
    :func:`build_volcano`) and flagged here for the per-gene view.

    Derived from :func:`_s2_1_replicate_pct` so "contributes" keeps exactly one
    definition (not absent, not below detection on both sides).
    """
    reps = _s2_1_replicate_pct()
    df = load_s2_1()
    d2_zero = pd.concat(
        [
            pd.DataFrame(
                {
                    "uniprot": df["uniprot"],
                    "rep": f"rep{rep}",
                    "d2_zero": (_s2_1_chan_mean(df, "D2", rep) == 0).values,
                }
            )
            for rep in PROTEOME_REPS
        ],
        ignore_index=True,
    )
    hit = reps.merge(d2_zero[d2_zero["d2_zero"]], on=["uniprot", "rep"], how="inner")
    return (
        hit[["uniprot", "condition"]]
        .drop_duplicates()
        .assign(d2_below_detection=True)
        .reset_index(drop=True)
    )


def build_proteome() -> pd.DataFrame:
    """Aggregated whole-proteome table: mean percent-of-control across replicates,
    per (protein x condition). Derived from the same replicate values the portal
    overlays, so bars (mean of replicates) and the aggregate agree by construction.

    Deliberately aggregated over *biological* replicates, unlike the
    channel-level overlay in :func:`build_proteome_replicates` — ``n_reps``
    counts donors, which is the meaningful confidence signal.

    ``censored`` marks conditions where at least one donor was a
    below-detection bound rather than a measurement, and
    ``d2_below_detection`` the stronger case where the D2 reference itself was
    missing (which also removes the comparison from the volcano). Both are
    thin-evidence signals worth surfacing: one protein, AFAP1L2, rests on a
    single usable donor.
    """
    reps = _s2_1_replicate_pct()
    agg = reps.groupby(["uniprot", "symbol", "condition"], as_index=False).agg(
        percent_control=("percent_control", "mean"),
        n_reps=("percent_control", "size"),
        censored=("censored", "any"),
    )
    agg["log2fc"] = _pct_to_log2fc(agg["percent_control"])
    agg = agg.merge(_d2_below_detection(), on=["uniprot", "condition"], how="left")
    agg["d2_below_detection"] = agg["d2_below_detection"].notna()
    agg["condition"] = pd.Categorical(
        agg["condition"], categories=FIVE_CONDITIONS, ordered=True
    )
    agg = agg.sort_values(["symbol", "condition"])
    return agg[
        ["uniprot", "symbol", "condition", "percent_control", "log2fc",
         "n_reps", "censored", "d2_below_detection"]
    ]


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
    # expression triangles. Keep the manuscript's own aggregate (the
    # whole_proteome percent-of-control column) so the triangles stay tied to
    # the published values. Note this is a *median over technical channels*,
    # not the mean over biological replicates that build_proteome() reports —
    # the two differ by ~0.035 log2 for most proteins.
    df["wp_log2fc"] = _pct_to_log2fc(df["whole_proteome"])
    # ...except where that column is non-finite: it carries ±inf wherever the
    # upstream aggregation divided by a D2 of zero, which drops the triangle
    # entirely (AFAP1L2 D8C is the only such cell). There we substitute
    # _wp_median_pct(), which computes the same median-of-channels statistic
    # but censors the empty D2 at the limit of detection instead of dividing by
    # zero — so the patched triangle stays comparable to its neighbours.
    # Only ±inf is patched — a *missing* whole_proteome value means the protein
    # was not quantified upstream, and must stay an absent triangle.
    wp = (
        _wp_median_pct()[["uniprot", "condition", "wp_log2fc"]]
        .drop_duplicates(subset=["uniprot", "condition"])
        .rename(columns={"wp_log2fc": "_wp_fallback"})
    )
    # merge resets the index (the source CSV is read with index_col=0), so
    # reset first to keep the mask aligned with the merged frame.
    df = df.reset_index(drop=True).merge(
        wp, on=["uniprot", "condition"], how="left"
    )
    non_finite = np.isinf(pd.to_numeric(df["whole_proteome"], errors="coerce"))
    df["wp_log2fc"] = df["wp_log2fc"].where(~non_finite, df["_wp_fallback"])
    df = df.drop(columns="_wp_fallback")
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

    # Drop comparisons whose D2 reference was below detection in a contributing
    # replicate. The published log2FC divides by a D2 mean that averaged in a
    # raw zero, inflating the fold change (AFAP1L2 D8C by ~1.19 log2, making it
    # the most extreme point in that volcano) — and its p-value cannot be
    # recomputed here, so correcting the x-position in place would pair a value
    # with a statistic never computed from it. The protein keeps its per-gene
    # view, where the censored value is shown with a low-confidence note.
    excl = _d2_below_detection().rename(columns={"condition": "comparison"})
    before = len(out)
    out = (
        out.merge(excl, on=["uniprot", "comparison"], how="left")
        .query("d2_below_detection.isna()")
        .drop(columns="d2_below_detection")
        .reset_index(drop=True)
    )
    if before != len(out):
        print(f"  volcano: dropped {before - len(out)} row(s) with a "
              f"below-detection D2 reference")

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
        A census value of exactly 0 means "below detection in that TMT channel",
        not "absent". Such channels are censored at that replicate's limit of
        detection (the 1st percentile of its positive census values) rather than
        discarded, so the value is a bound: an upper bound on the loss when the
        condition channel was empty, a lower bound on the increase when the D2
        reference was. Where both channels were empty the ratio carries no
        information and the replicate is omitted for that condition.

      NOTE the portal's volcano plot omits any (protein x comparison) whose D2
      reference was below detection in a contributing replicate. The published
      log2FC there divides by a D2 mean that averaged in a raw zero, which
      halves the denominator and inflates the fold change (for AFAP1L2 D8C by
      ~1.19 log2, making it the most extreme point in that comparison). The
      statistics are still reported in this file as published; only the plotted
      point is withheld, and the protein keeps its per-gene view.
      * {cond}_log2fc / _p_value / _neglog10_pval / _neglog10_padj / _regulation:
        volcano statistics for each cond-vs-D2 comparison (cond in D4A/D4C/D8A/D8C).
        regulation is the published significance call. These are reproduced here
        as published, including comparisons the portal's volcano plot omits (see
        below) — so this file remains a faithful copy of the source statistics.
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
    per cysteine of a protein (used for the dot-plot expression triangles). It is
    the manuscript's own whole-proteome aggregate: the MEDIAN over technical
    channels, each divided by its biological replicate's D2 mean. Note this is a
    different statistic from whole_proteome.csv above, which reports the MEAN
    over biological replicates — expect the two to differ by ~0.035 log2 for most
    proteins. The sole exception is where the upstream median is non-finite (a D2
    reference of zero): there wp_log2fc is recomputed with that channel censored
    at the limit of detection, which would otherwise leave the triangle missing.
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
