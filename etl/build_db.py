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
def load_s2_2() -> pd.DataFrame:
    """Supplementary sheet S2-2 (bulk RNA vs whole proteome): per-comparison RNA
    DESeq2 statistics beside the matched WP fold change, over the ~6.1k genes
    measured by both. Two title rows precede the header (``header=2``).

    Read for one thing only: the ``D8C vs D8A`` RNA block, which is the single
    non-vs-D2 transcriptome comparison the manuscript publishes (S1-1 carries
    only the four vs-D2 contrasts).

    NOTE this sheet labels its comparisons **numerator-first** — "D8C vs D8A"
    really is log2(D8C/D8A) — which is the *opposite* of S2-1, where
    "D8A vs. D8C" also means log2(D8C/D8A). Verified: this sheet's
    ``D8C vs D2 RNA_log2FoldChange`` reproduces S1-1's
    ``log2FoldChange_D8C_vs_D2``.
    """
    return pd.read_excel(
        SOURCE / "data_s2.xlsx",
        sheet_name="Data S2-2 Bulk RNA vs WP",
        header=2,
    )


@lru_cache(maxsize=1)
def load_s1_1() -> pd.DataFrame:
    """Supplementary sheet S1-1 (bulk RNA-seq): DESeq2 results vs D2, one wide
    block per comparison. Two title rows precede the header (``header=2``)."""
    return pd.read_excel(SOURCE / "data_s1.xlsx", sheet_name=0, header=2)


@lru_cache(maxsize=1)
def load_s3_1() -> pd.DataFrame:
    """Supplementary sheet S3-1 (polar metabolomics): compound annotations,
    per-sample channel ratios and raw intensities, and the differential
    block for six pairwise comparisons. Two title rows precede the header
    (``header=2``), as in S1-1/S2-1."""
    return pd.read_excel(
        SOURCE / "data_s3.xlsx",
        sheet_name="S3-1 Polar metabolites",
        header=2,
    )


# whole-proteome / reactivity condition columns, in display order
FIVE_CONDITIONS = ["D2", "D4A", "D4C", "D8A", "D8C"]
# vs-D2 comparisons we surface, in display order (mirrors FIVE_CONDITIONS - D2).
# A bare condition code means "that condition vs D2" — the implicit reference
# every per-gene view uses.
FOUR_COMPARISONS = ["D4A", "D4C", "D8A", "D8C"]
# comparisons that are *not* against D2, written "<numerator>_vs_<denominator>"
# so the reference is explicit wherever a bare code would be ambiguous.
EXTRA_COMPARISONS = ["D8C_vs_D8A"]


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

    The per-channel ``censored`` flag is used internally but not published: it
    fires on too few cells to be worth a column in every consumer.
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
         "log2fc"]
    ]


@lru_cache(maxsize=None)
def _ref_below_detection(ref: str = "D2") -> pd.DataFrame:
    """(uniprot, condition) pairs whose ``ref`` denominator was below detection
    in a replicate that contributes to that condition.

    A fold change is undefined without a reference channel: we censor a zero one
    at the limit of detection, which makes the ratio a *lower bound* rather than
    a measurement. The published S2-1 statistics instead average the raw zero
    into the denominator, which halves it and inflates the fold change. Those
    comparisons are dropped from the volcano (see :func:`build_volcano`) and
    flagged for the per-gene view.

    ``ref`` is parameterized because the volcano carries two kinds of
    comparison. With ``ref="D2"`` it inflates AFAP1L2's D8C fold change by
    ~1.19 log2, enough to make it the most extreme point in that panel. With
    ``ref="D8A"`` — the denominator of the D8C-vs-D8A comparison — the same
    rep5 dropout inflates AFAP1L2 to +4.68 and TNFRSF4 to +2.87 log2.

    "Contributes" keeps exactly one definition, the same one :func:`_censor`
    applies: the reference channel is empty while the numerator is not. A
    replicate empty on *both* sides carries no information either way and is
    already excluded upstream, so it is not a hit here.
    """
    df = load_s2_1()
    frames = []
    for rep in PROTEOME_REPS:
        den = _s2_1_chan_mean(df, ref, rep)
        for cond in FIVE_CONDITIONS:
            if cond == ref:
                continue
            num = _s2_1_chan_mean(df, cond, rep)
            hit = (den == 0) & num.notna() & (num != 0)
            if not hit.any():
                continue
            frames.append(
                pd.DataFrame(
                    {"uniprot": df.loc[hit, "uniprot"].values, "condition": cond}
                )
            )
    cols = ["uniprot", "condition"]
    hits = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return (
        hits[cols]
        .drop_duplicates()
        .assign(below_detection=True)
        .reset_index(drop=True)
    )


def _d2_below_detection() -> pd.DataFrame:
    """The vs-D2 case of :func:`_ref_below_detection`, under the column name
    :func:`build_proteome` publishes."""
    return _ref_below_detection("D2").rename(
        columns={"below_detection": "d2_below_detection"}
    )


def build_proteome() -> pd.DataFrame:
    """Aggregated whole-proteome table: mean percent-of-control across replicates,
    per (protein x condition). Derived from the same replicate values the portal
    overlays, so bars (mean of replicates) and the aggregate agree by construction.

    Deliberately aggregated over *biological* replicates, unlike the
    channel-level overlay in :func:`build_proteome_replicates` — ``n_reps``
    counts donors, which is the meaningful confidence signal.

    ``d2_below_detection`` marks the conditions where the D2 reference itself
    was missing, which makes the fold change a lower bound and removes the
    comparison from the volcano. Together with ``n_reps`` these are the
    thin-evidence signals worth surfacing: one protein, AFAP1L2, rests on a
    single usable donor.
    """
    reps = _s2_1_replicate_pct()
    agg = reps.groupby(["uniprot", "symbol", "condition"], as_index=False).agg(
        percent_control=("percent_control", "mean"),
        n_reps=("percent_control", "size"),
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
         "n_reps", "d2_below_detection"]
    ]


# Sample column -> (condition, rep). The raw count matrix names its columns after
# the bam paths ("sorted_bam_files/Sample_D4-Ac-1_IGO_13524_4_sorted.bam"), so this
# has to be searched rather than matched, and it accepts either separator: the
# featureCounts output hyphenates where the older normalized matrix used dots.
_RNA_SAMPLE_RE = re.compile(r"Sample_(D2|D4|D8)[.-]?(?:(Ac|Chr)[.-])?(\d+)_IGO")


def _rna_sample_meta(col: str) -> tuple[str, str] | None:
    """'…/Sample_D8-Chr-2_IGO_13524_14_sorted.bam' -> ('D8C', 'rep2'); D2 ->
    ('D2', …). Returns None for the annotation columns (Chr/Start/End/…)."""
    m = _RNA_SAMPLE_RE.search(col)
    if not m:
        return None
    day, ac, rep = m.group(1), m.group(2), m.group(3)
    cond = day if day == "D2" else day + ("A" if ac == "Ac" else "C")
    return cond, f"rep{rep}"


def _size_factors(counts: pd.DataFrame) -> pd.Series:
    """Library-size factors, one per sample column, by median of ratios.

    This reimplements DESeq2's ``estimateSizeFactors`` here rather than calling
    it — the ETL is pure Python and takes no R dependency. Only this step is
    reproduced: there is no GLM, no dispersion estimation and no shrinkage, which
    is why the replicate points it feeds are supporting evidence and the bar
    itself stays S1-1's published DESeq2 estimate.

    The reference is the per-gene geometric mean taken over genes counted in
    **every** sample — DESeq2's own rule, since a single zero sends the geometric
    mean to zero and makes the gene carry no information about library size. Each
    sample's factor is then the median of its ratios to that reference.
    """
    positive = counts[(counts > 0).all(axis=1)]
    log_counts = np.log(positive)
    return np.exp(log_counts.sub(log_counts.mean(axis=1), axis=0).median(axis=0))


def build_rna_replicates() -> pd.DataFrame:
    """Raw per-sample counts -> long per-replicate log2FC from D2.

    ``rna_counts_raw.txt`` is the analysis repo's featureCounts matrix: gene
    annotation columns (Chr/Start/End/Strand/Length) followed by one raw-count
    column per bam. We normalize it here with DESeq2 median-of-ratios size
    factors and take log2(count + 0.5) — the pseudocount ``DESeq2::plotCounts``
    uses, which only bites on the ~11% of displayed cells whose count is zero.
    Per gene, each non-D2 sample's log2FC is its value minus the mean of that
    gene's D2 replicates. D2 itself is the reference (log2FC = 0) and is not
    emitted, matching the RNA figure's four displayed conditions.

    Doing the normalization here rather than reading the analysis repo's
    ``normalized_counts.txt`` is deliberate. Differences within that matrix are
    **not** log2 ratios: regressed against S1-1's DESeq2 ``log2FoldChange`` they
    come out compressed by a near-constant factor (slope 1.488, R^2 0.996 for
    baseMean > 1000, and worse at low counts), which had the bar chart
    understating every fold change against an axis labelled log2FC. Normalizing
    the raw counts instead reproduces the DESeq2 estimates to a median of 0.002
    for baseMean > 100, so the per-gene bars and the volcano agree.
    """
    raw = pd.read_csv(SOURCE / "rna_counts_raw.txt", sep="\t", index_col=0)
    meta = {c: _rna_sample_meta(c) for c in raw.columns}
    meta = {c: m for c, m in meta.items() if m is not None}
    counts = raw[list(meta)].astype(float)

    lognorm = np.log2(counts.div(_size_factors(counts), axis=1) + 0.5)
    d2_cols = [c for c, (cond, _) in meta.items() if cond == "D2"]
    d2_mean = lognorm[d2_cols].mean(axis=1)

    order = ["D4A", "D4C", "D8A", "D8C"]
    rows = []
    for col, (cond, rep) in meta.items():
        if cond == "D2":
            continue
        lfc = lognorm[col] - d2_mean
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
                # the GLM's own standard error for that log2FC — what the bar
                # chart draws as its error bar, since the bar is the log2FC.
                # Verified to pair with it: stat == log2FoldChange / lfcSE
                # exactly, so these are unshrunken Wald results.
                "lfc_se": df[f"lfcSE_{cond}_vs_D2"],
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
# comparisons we surface, in display order: the four vs-D2 panels, then the
# chronic-vs-acute contrast at day 8.
VOLCANO_COMPARISONS = FOUR_COMPARISONS + EXTRA_COMPARISONS

# comparison id -> the S2-1 block label fragment that carries it.
#
# S2-1's labels are REVERSED — they read "<denominator> vs. <numerator>" but the
# values carry numerator/denominator. So "D2 vs. D4A" is log2(D4A/D2), and
# "D8A vs. D8C" is log2(D8C/D8A). (Verified: log2_FC "D2 vs. D8A" is the exact
# negation of log2_FC "D8A vs. D2", and "D8A vs. D8C" reproduces
# log2(mean(D8C)/mean(D8A)) from the expression table at corr 0.97.)
#
# Spelled out rather than matched by regex because the sheet ships eight blocks
# and we want five: a pattern like "(\w+) vs\. (\w+)" would also pick up
# D8A-vs-D4A / D8A-vs-D4C / D8A-vs-D2, which the portal does not plot. Each
# fragment below is unique among the eight, so substring matching is safe.
_S2_1_VOLCANO_BLOCK = {
    "D4A": "D2 vs. D4A",
    "D4C": "D2 vs. D4C",
    "D8A": "D2 vs. D8A",
    "D8C": "D2 vs. D8C",
    "D8C_vs_D8A": "D8A vs. D8C",
}

# denominator condition per comparison, for the below-detection exclusion
VOLCANO_REFERENCE = {c: "D2" for c in FOUR_COMPARISONS} | {"D8C_vs_D8A": "D8A"}

# the five statistics each comparison contributes, in emit order. Shared with
# etl/prerender.py, which rebuilds the same download columns from parquet.
VOLCANO_STATS = ["log2fc", "p_value", "neglog10_pval", "neglog10_padj",
                 "regulation"]


def _volcano_suffix_by_cond(df: pd.DataFrame) -> dict[str, str]:
    """Map each comparison id (e.g. "D4A", "D8C_vs_D8A") -> the exact S2-1
    column suffix, e.g. ``Exhaustion WP - D2 vs. D4A (6315 Proteins)``.

    The sheet stamps the protein count into every suffix, so it cannot be
    hardcoded; we find the one ``log2_FC_`` column whose suffix contains the
    block fragment from :data:`_S2_1_VOLCANO_BLOCK`.
    """
    suffix_by_cond: dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"log2_FC_(.+)", str(col))
        if not m:
            continue
        suffix = m.group(1)
        for cid, fragment in _S2_1_VOLCANO_BLOCK.items():
            if fragment in suffix:
                suffix_by_cond[cid] = suffix
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

    # Drop comparisons whose reference channel was below detection in a
    # contributing replicate. The published log2FC divides by a denominator mean
    # that averaged in a raw zero, inflating the fold change (AFAP1L2 D8C by
    # ~1.19 log2 against D2, and by far more against D8A) — and its p-value
    # cannot be recomputed here, so correcting the x-position in place would
    # pair a value with a statistic never computed from it. The protein keeps
    # its per-gene view, where the censored value carries a low-confidence note.
    #
    # The reference differs by comparison: D2 for the four vs-D2 panels, D8A for
    # D8C-vs-D8A. _ref_below_detection is keyed by the *numerator* condition, so
    # each reference's table is filtered to the comparisons that use it.
    excl = pd.concat(
        [
            _ref_below_detection(ref)
            .assign(comparison=cid)
            .loc[lambda d: d["condition"] == cid.split("_vs_")[0]]
            for cid, ref in VOLCANO_REFERENCE.items()
        ],
        ignore_index=True,
    )[["uniprot", "comparison", "below_detection"]]
    before = len(out)
    out = (
        out.merge(excl, on=["uniprot", "comparison"], how="left")
        .query("below_detection.isna()")
        .drop(columns="below_detection")
        .reset_index(drop=True)
    )
    if before != len(out):
        print(f"  volcano: dropped {before - len(out)} row(s) with a "
              f"below-detection reference channel")

    out["comparison"] = pd.Categorical(
        out["comparison"], categories=VOLCANO_COMPARISONS, ordered=True
    )
    return out


# RNA volcano significance rule. Deliberately *not* the proteome's rule: the
# published S2-1 call is raw p < 0.05 with |log2FC| >= log2(1.5), while DESeq2
# gives a multiplicity-corrected padj for every gene, so the RNA volcano uses
# padj with the stricter 2-fold cutoff. The per-gene RNA bar chart draws its
# dashed guides at the same RNA_FC_CUTOFF (see figures.rna_figure).
RNA_PADJ_CUTOFF = 0.05
RNA_FC_CUTOFF = 1.0  # log2(2)


def _rna_regulation(log2fc: pd.Series, padj: pd.Series) -> pd.Series:
    """Volcano regulation call, reusing the S2-1 category vocabulary so both
    volcanoes share one legend, palette, and draw order."""
    sig = padj < RNA_PADJ_CUTOFF
    big = log2fc.abs() >= RNA_FC_CUTOFF
    up = log2fc > 0
    return pd.Series(
        np.select(
            [sig & big & up, sig & big & ~up, sig & ~big, ~sig & big & up,
             ~sig & big & ~up],
            ["Significant Up", "Significant Down", "Significant but <2 FC",
             "Not Significant Up", "Not Significant Down"],
            default="Not Significant",
        ),
        index=log2fc.index,
    )


def _rna_d8c_vs_d8a() -> pd.DataFrame:
    """The D8C-vs-D8A transcriptome block, from supplementary sheet S2-2.

    S1-1 publishes only the four vs-D2 DESeq2 contrasts, so this one comparison
    comes from S2-2 instead. Both are the same transcriptome-wide fit — S2-2's
    ``D8C vs D2`` column reproduces S1-1's, and its ``D8C vs D8A`` column
    reproduces the analysis repo's DESeq2 output exactly.

    Read as labelled: S2-2 is numerator-first (see :func:`load_s2_2`), the
    opposite of S2-1's reversed convention, so no sign flip here.

    S2-2's own ``regulation_protein_rna`` column is deliberately ignored — it is
    a plot colour code ("dark grey", "purple", …) for a different figure, not a
    significance call. :func:`_rna_regulation` supplies the call so all five
    panels share one rule, one legend, and one palette.
    """
    s2 = load_s2_2().drop_duplicates(subset=["protein"])
    return pd.DataFrame(
        {
            "symbol": s2["protein"].values,
            "condition": "D8C_vs_D8A",
            "log2fc": s2["D8C vs D8A RNA_log2FoldChange"].values,
            "padj": s2["D8C vs D8A RNA_padj"].values,
            # S2-2 carries no baseMean; the column exists for the vs-D2 rows
            "base_mean": np.nan,
        }
    )


def build_rna_volcano() -> pd.DataFrame:
    """Transcriptome volcano data -> long table, one row per (gene x
    comparison), **restricted to genes with matched whole-proteome data**.

    Same shape as :func:`build_volcano` so the two share a figure builder. The
    matched-gene restriction is the point of the plot: it puts the RNA and
    protein volcanoes on one gene universe (~6.2k), so a reader can compare them
    without the transcriptome's extra ~11k unmeasured-by-MS genes changing the
    shape of the cloud.

    Covers the four vs-D2 comparisons (S1-1, via :func:`build_rna`) plus
    D8C-vs-D8A (S2-2, via :func:`_rna_d8c_vs_d8a`). The two sources are
    concatenated *before* the shared tail below so every comparison goes through
    one gene restriction, one null drop, one padj floor and one regulation call.

    ``padj`` comes from the transcriptome-wide DESeq2 fit and is **not**
    recomputed on the subset — re-running the multiplicity correction over 6.2k
    genes would produce numbers that disagree with rna.csv and with the
    manuscript. The figure annotates this.
    """
    sources = [build_rna(), _rna_d8c_vs_d8a()]
    rna = pd.concat([s for s in sources if not s.empty], ignore_index=True)
    matched = set(build_proteome()["symbol"])
    out = rna[rna["symbol"].isin(matched)].copy()
    # DESeq2 leaves padj null wherever independent filtering removed the gene
    # from the multiplicity correction; those genes have no y-position.
    out = out.dropna(subset=["log2fc", "padj"]).reset_index(drop=True)

    # A padj of exactly 0 (underflow) would plot at +inf, which parquet and
    # Plotly both choke on — pin it just above the smallest positive padj.
    # The floor is pooled across comparisons, so adding one can lower it and
    # shift the handful of already-clamped points; that is intended, and a
    # changed y-maximum is not a bug.
    positive = out.loc[out["padj"] > 0, "padj"]
    floor = float(positive.min()) if len(positive) else np.nan
    out["neglog10_padj"] = -np.log10(out["padj"].mask(out["padj"] <= 0, floor))
    out["regulation"] = _rna_regulation(out["log2fc"], out["padj"])

    desc = (
        load_s2_1()[["protein", "description"]]
        .rename(columns={"protein": "symbol"})
        .drop_duplicates(subset=["symbol"])
    )
    out = out.merge(desc, on="symbol", how="left")
    out = out.rename(columns={"condition": "comparison"})
    out["comparison"] = pd.Categorical(
        out["comparison"], categories=VOLCANO_COMPARISONS, ordered=True
    )
    out = out.sort_values(["comparison", "symbol"])
    return out[
        ["symbol", "description", "comparison", "log2fc", "padj",
         "neglog10_padj", "base_mean", "regulation"]
    ].reset_index(drop=True)


def build_proteome_download() -> pd.DataFrame:
    """Single combined whole-proteome table for the bulk download, mirroring the
    layout of supplementary sheet S2-1: identity columns, per-biological-replicate
    percent-of-control values (technical pairs averaged, D2-normalized), the
    per-comparison volcano/significance columns, and the functional-group flags.

    Comparisons whose reference channel was below detection are blanked, exactly
    as :func:`build_volcano` withholds their plotted point. This file has two
    build paths — here from the S2-1 xlsx, and in ``etl/prerender.py`` from the
    committed parquet, which is what CI ships — and the parquet no longer
    carries those rows. Blanking here is what keeps the two byte-identical.
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
        excl = _ref_below_detection(VOLCANO_REFERENCE[cond])
        withheld = out["uniprot"].isin(
            excl.loc[excl["condition"] == cond.split("_vs_")[0], "uniprot"]
        )
        out.loc[withheld, [f"{cond}_{s}" for s in VOLCANO_STATS]] = np.nan
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
# polar metabolomics (Data S3-1)
# --------------------------------------------------------------------------- #
# The portal ships metabolomics as a bulk download only — no per-gene view — so
# this table *is* the download: the parquet round-trips straight to CSV in both
# etl/build_db.py and etl/prerender.py.

# S3-1 annotation columns -> snake_case. Mapped explicitly rather than derived:
# the workbook's "Super Pathway" header is truncated mid-parenthesis, so any
# rule that rewrote the header text would have to encode that typo anyway.
METABOLITE_ANNOTATION_COLS = {
    "Compound": "compound",
    "HMDB": "hmdb",
    "KEGG": "kegg",
    "Alias": "alias",
    "Pathway": "pathway",
    "Metabolite Class (Vardhana lab)": "metabolite_class",
    "Super Pathway (MSK metabolomics Core dMRM method": "super_pathway",
    "Chemical Taxonomy, Super Class (HMDB)": "hmdb_super_class",
    "Chemical Taxonomy, Sub Class (HMDB)": "hmdb_sub_class",
}

# per-sample value blocks: workbook column suffix -> ours, in emit order
METABOLITE_VALUE_SUFFIXES = {
    "cell-volume-normalized-QRILC-imputation": "channel_ratio",
    "raw-signal-intensity": "raw_intensity",
}

# statistic name in the S3-1 differential block -> our column suffix
METABOLITE_STATS = {
    "p_value": "p_value",
    "log2_FC": "log2fc",
    "-log10_pval": "neglog10_pval",
    "-log10_pval_adj": "neglog10_padj",
    "Regulation": "regulation",
}

# metabolomics comparisons as (numerator, denominator), in display order: the
# four vs-D2 contrasts, then the acute-vs-chronic pair at each timepoint.
METABOLITE_COMPARISONS = [
    ("D4A", "D2"),
    ("D4C", "D2"),
    ("D8A", "D2"),
    ("D8C", "D2"),
    ("D4C", "D4A"),
    ("D8C", "D8A"),
]

# "<stat>_metabolomics - <den> vs. <num> (174 Metabolites)". Note the *reversed*
# wording: as in S2-1, the values carry the second-named condition over the
# first (verified — "D2 vs. D8C" reproduces log2(mean(D8C)/mean(D2)) exactly).
# The trailing group catches pandas' ".1" rename of the duplicate block.
_METAB_STAT_RE = re.compile(
    r"^(?P<stat>.+?)_metabolomics - (?P<den>\S+) vs\. (?P<num>\S+) \(.*?\)"
    r"(?P<dup>\.\d+)?$"
)

# "D8C_ZC2_S58_<suffix>" -> condition / donor / sample id
_METAB_SAMPLE_RE = re.compile(
    r"^(?P<cond>D2|D4A|D4C|D8A|D8C)_(?P<donor>[^_]+)_(?P<sample>[^_]+)_"
    r"(?P<suffix>.+)$"
)


def _metabolite_sample_cols(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(source column, renamed column) for the per-sample value blocks.

    Re-sorted into channel-ratio-then-raw, donor, condition, sample order. The
    workbook's own column order interleaves the donors and puts NL57's D8C block
    ahead of its D2 block, which makes the CSV awkward to read side by side.
    """
    cond_rank = {c: i for i, c in enumerate(FIVE_CONDITIONS)}
    suffix_rank = {s: i for i, s in enumerate(METABOLITE_VALUE_SUFFIXES)}
    found = []
    for col in df.columns:
        m = _METAB_SAMPLE_RE.match(str(col))
        if m is None or m.group("suffix") not in METABOLITE_VALUE_SUFFIXES:
            continue
        cond, donor, sample, suffix = m.group("cond", "donor", "sample", "suffix")
        found.append(
            (
                (suffix_rank[suffix], donor, cond_rank[cond], sample),
                col,
                f"{cond}_{donor}_{sample}_{METABOLITE_VALUE_SUFFIXES[suffix]}",
            )
        )
    return [(col, new) for _, col, new in sorted(found)]


def _metabolite_stat_cols(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    """(numerator, denominator) -> {statistic: source column}.

    S3-1 carries every differential block **twice**; pandas renames the second
    copy with a ``.1`` suffix. The copies are identical, so we keep the first
    and verify the duplicate agrees rather than trusting the layout — a future
    revision that made them differ would otherwise be silently halved.
    """
    blocks: dict[tuple[str, str], dict[str, str]] = {}
    for col in df.columns:
        m = _METAB_STAT_RE.match(str(col))
        if m is None or m.group("stat") not in METABOLITE_STATS:
            continue
        key = (m.group("num"), m.group("den"))
        stat = m.group("stat")
        if m.group("dup"):
            first = blocks.get(key, {}).get(stat)
            if first is not None and not df[first].equals(df[col]):
                raise ValueError(
                    f"S3-1 duplicate block disagrees with the original: "
                    f"{col!r} != {first!r}"
                )
            continue
        blocks.setdefault(key, {})[stat] = col
    return blocks


def build_metabolomics() -> pd.DataFrame:
    """Polar metabolomics -> one row per compound, wide, mirroring S3-1.

    Annotation columns, then per-sample channel ratios and raw signal
    intensities (4 donors x 5 conditions x 3 injections), then the differential
    block for each comparison. Comparison columns are named numerator-first
    (``D8C_vs_D2_log2fc``) because the workbook labels them the other way round
    while carrying the numerator-first sign — and because metabolomics, unlike
    the whole proteome, also contrasts conditions that are not D2.
    """
    df = load_s3_1()
    missing = [c for c in METABOLITE_ANNOTATION_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"S3-1 is missing expected annotation column(s): {missing}")

    # source column -> output name, in emit order; sliced in one pass at the end
    # (160-odd sequential assignments would fragment the frame).
    picked: list[tuple[str, str]] = [
        (col, new) for col, new in METABOLITE_ANNOTATION_COLS.items()
    ]
    picked += _metabolite_sample_cols(df)

    blocks = _metabolite_stat_cols(df)
    for num, den in METABOLITE_COMPARISONS:
        block = blocks.get((num, den))
        if block is None:
            continue
        picked += [
            (block[stat], f"{num}_vs_{den}_{suffix}")
            for stat, suffix in METABOLITE_STATS.items()
            if stat in block
        ]

    return (
        df[[col for col, _ in picked]]
        .set_axis([new for _, new in picked], axis=1)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# bulk-download bundle
# --------------------------------------------------------------------------- #
DATA_DICTIONARY = """\
T cell dysfunction proteomics data portal — bulk download
=========================================================

Fold changes are expressed as log2 fold-change relative to the D2 (baseline /
non-dysfunctional) condition, except where a column name gives the reference
explicitly: a column named "A_vs_B" is log2(A/B), numerator first.

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

      NOTE the portal's volcano plot omits any (protein x comparison) whose
      reference channel was below detection in a contributing replicate. The
      published log2FC there divides by a denominator mean that averaged in a
      raw zero, which halves it and inflates the fold change. The reference is
      D2 for the vs-D2 comparisons (AFAP1L2 D8C is inflated by ~1.19 log2,
      making it the most extreme point in that panel) and D8A for D8C_vs_D8A
      (AFAP1L2 +4.68, TNFRSF4 +2.87). The protein keeps its per-gene view.
      Those three (protein x comparison) cells are blank in the statistics
      columns below, since this file is regenerated from the same table the
      plot reads. Everything else is reproduced as published.
      * {cond}_log2fc / _p_value / _neglog10_pval / _neglog10_padj / _regulation:
        volcano statistics for each cond-vs-D2 comparison (cond in D4A/D4C/D8A/D8C).
        regulation is the published significance call. These are reproduced here
        as published, including comparisons the portal's volcano plot omits (see
        above) — so this file remains a faithful copy of the source statistics.
      * D8C_vs_D8A_log2fc / _p_value / _neglog10_pval / _neglog10_padj /
        _regulation: the same five statistics for day-8 chronic over day-8
        acute. Named numerator-first because, unlike the bare-condition columns
        above, this comparison is not against D2 — the value is
        log2(D8C/D8A), positive meaning higher under chronic stimulation.
        (Source: sheet S2-1's "D8A vs. D8C" block, whose label is reversed
        relative to its values, as all of that sheet's labels are.)
      * mitochondrial, peroxisome, redox_related, cell_cycle,
        nucleotide_metabolism, endoplasmic_reticulum: boolean functional-group
        flags (global, not per-comparison).
    columns: uniprot, symbol, description, D2_rep1..D8C_rep6, per-comparison
             volcano columns, functional-group flags

rna.csv
    Bulk RNA-seq differential expression vs D2 (DESeq2). One row per (gene x condition).
    Restricted to protein-coding genes (NCBI Gene, tax 9606); non-coding
    transcripts (lncRNA, miRNA, pseudogenes, …) are excluded.
    lfc_se is DESeq2's standard error for log2fc, from the negative-binomial GLM
    with its dispersion shrunk toward the transcriptome-wide mean-dispersion
    trend. It does not shrink to zero with sequencing depth: at high counts it is
    dominated by biological overdispersion, not counting noise.
    columns: symbol, condition, log2fc, lfc_se, padj, base_mean

rna_replicates.csv
    Per-replicate RNA log2FC. One row per (gene x condition x replicate). Raw
    counts are normalized with DESeq2 median-of-ratios size factors, taken as
    log2(count + 0.5), and expressed relative to the per-gene D2 mean. These
    agree closely with the model-based DESeq2 log2fc in rna.csv (median absolute
    difference 0.002 for genes with baseMean > 100); expect them to diverge for
    very low-count genes, where the GLM estimate and a raw count ratio legitimately
    differ. Protein-coding genes only (as rna.csv).
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

metabolomics.csv
    Polar metabolite abundance (LC-MS/MS), one row per compound (wide,
    mirroring supplementary sheet S3-1). Data are from n = 4 donors (LSP75,
    NL57, ZC1, ZC2), three injections each per condition. 174 of the 203
    metabolites reported by the MSK metabolomics core passed the signal
    filter (at least one condition averaging above 5000 raw intensity);
    missing values were imputed by quantile regression imputation of
    left-censored data (QRILC).
      * compound, hmdb, kegg, alias, pathway, metabolite_class,
        super_pathway, hmdb_super_class, hmdb_sub_class: compound identity and
        annotation. metabolite_class was curated by the Vardhana lab.
      * {cond}_{donor}_{sample}_channel_ratio: cell-volume-normalized,
        QRILC-imputed channel ratio (signal intensity divided by the sum of
        that metabolite's intensities). This is the value all analyses use.
      * {cond}_{donor}_{sample}_raw_intensity: the raw signal intensity the
        ratio was computed from.
      * {num}_vs_{den}_log2fc / _p_value / _neglog10_pval / _neglog10_padj /
        _regulation: differential abundance for six comparisons — each of
        D4A/D4C/D8A/D8C against D2, plus D4C vs D4A and D8C vs D8A.
        p-values are two-sample t-tests; regulation is the published call.
        NOTE these columns are named NUMERATOR FIRST: D8C_vs_D2_log2fc is
        log2(D8C/D2). The source workbook labels the same block "D2 vs. D8C"
        while carrying the D8C/D2 sign, so the names here are rewritten to
        match the values rather than the label.
    columns: compound + 8 annotation columns, 60 channel_ratio columns,
             60 raw_intensity columns, 6 x 5 comparison columns

Provenance: whole proteome, bulk RNA-seq, and polar metabolomics are read from
the manuscript supplementary workbooks (Data S1, Data S2, Data S3); the
remaining tables are derived from the analysis repository
t-cell-dysfunction-2026.
"""

# (parquet-table-name, download-basename) — tables written verbatim to the bundle
DOWNLOAD_TABLES = [
    ("rna", "rna"),
    ("rna_replicates", "rna_replicates"),
    ("reactivity", "reactivity_5cond"),
    ("reactivity_atp", "reactivity_atp"),
    ("metabolomics", "metabolomics"),
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
        "metabolomics": build_metabolomics(),
        "volcano": build_volcano(),
        "rna_volcano": build_rna_volcano(),
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
