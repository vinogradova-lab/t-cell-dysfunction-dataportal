"""Plotly figure builders — one per modality.

Each returns a plotly ``Figure``; the Flask layer serializes with
``fig.to_json()`` and the browser renders it with Plotly.js. All figures use the
manuscript palette (``palette.py``) so the portal matches the published look.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import polars as pl

from palette import (
    ATP_FILLS,
    ATP_LEGEND_ORDER,
    ATP_ORDER,
    ATP_OUTLINE,
    ATP_SYMBOLS,
    EXHAUSTION_COLS,
    FC_CUTOFF,
    FONT_FAMILY,
    FOUR_CONDITION_ORDER,
    PVAL_CUTOFF,
    REACTIVITY_GUIDE,
    REGULATION_COLORS,
    RNA_FC_CUTOFF,
    VOLCANO_HIGHLIGHT,
    VOLCANO_REG_COLORS,
    VOLCANO_REG_ORDER,
)

# Titles are rendered as HTML headings in the card (see app.js), so the plots
# themselves carry no title — this avoids the title/legend collision and keeps
# the top legend clean.
_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT_FAMILY, size=13, color="#111"),
    margin=dict(l=64, r=20, t=16, b=50),
    hovermode="closest",
    showlegend=False,
)
# a bit more top room when a horizontal legend sits above the plot
_LEGEND_TOP_MARGIN = dict(l=64, r=20, t=34, b=50)

# SEM error bars on the abundance bars. Plotly sizes both of these in px, so
# the cap can't track bar width the way ggplot's category-relative
# ``width = 0.4`` does; this reads proportionate at the portal's chart sizes.
ERROR_BAR_LINE_WIDTH = 0.25
ERROR_BAR_CAP_WIDTH = 8

# shared by every vs-D2 categorical x-axis (proteome, RNA, reactivity) so the
# three panels of a gene view label their axis identically
CONDITION_AXIS_TITLE = "condition (vs D2)"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_BASE_LAYOUT)
    fig.add_annotation(
        text=msg, showarrow=False, font=dict(size=14, color="#888"),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _order(df: pl.DataFrame, order: list[str]) -> pl.DataFrame:
    """Sort by a fixed condition order."""
    idx = {c: i for i, c in enumerate(order)}
    return (
        df.with_columns(
            pl.col("condition").replace_strict(idx, default=999).alias("_o")
        )
        .sort("_o")
        .drop("_o")
    )


def _has_reps(reps_df: pl.DataFrame | None) -> bool:
    return reps_df is not None and not reps_df.is_empty()


def _replicate_sem(reps_df: pl.DataFrame, order: list[str]) -> dict[str, float]:
    """condition -> standard error of the mean replicate log2FC (sd/sqrt(n)).

    Mirrors the manuscript's ``geom_errorbar`` in whole_proteome_visualization.Rmd
    and rna_visualization.Rmd, which pool every replicate column for a condition
    (6 donors x 2 technical channels for the proteome) and take sd/sqrt(n).

    Read it as the spread of the overlaid points, **not** as the uncertainty of
    the bar: the bar is the published log2FC, and no published standard error
    exists for it (the S2-1 volcano block carries only p-values, and
    back-calculating an SE from one would require assuming a test and its df).
    Pooling channels also treats a donor's two technical measurements as
    independent, so the interval is narrower than a donor-level SEM by at least
    √2 — it measures channel reproducibility, not donor variability. It is
    retained because it reproduces the manuscript's own figure and because the
    points are per-donor ratios (each donor's D2 scaled to 100 in
    ``build_db._s2_1_replicate_pct``), so the reference sits inside every point
    rather than being treated as error-free.

    Computed on log2FC rather than the manuscript's percent-of-control because
    that is the axis here; an error bar has to match the scale it is drawn on.
    Conditions with n < 2 have no defined sd and are omitted.
    """
    agg = reps_df.group_by("condition").agg(
        pl.col("log2fc").std().alias("sd"), pl.len().alias("n")
    )
    return {
        r["condition"]: r["sd"] / math.sqrt(r["n"])
        for r in agg.iter_rows(named=True)
        if r["condition"] in order and r["n"] > 1 and r["sd"] is not None
    }


def _error_y(errors: dict[str, float], conditions: list[str]) -> dict:
    """Symmetric error bars for a bar trace, from condition -> half-width.

    A missing or non-finite entry leaves that condition with no bar rather than
    a misleading zero-length one — single-replicate proteome conditions have no
    defined SEM, and DESeq2 leaves lfcSE null wherever it fit no model.
    """
    def _finite(c):
        v = errors.get(c)
        return v if v is not None and math.isfinite(v) else None

    return dict(
        type="data",
        array=[_finite(c) for c in conditions],
        visible=True,
        color="#333",
        thickness=ERROR_BAR_LINE_WIDTH,
        width=ERROR_BAR_CAP_WIDTH,
    )


def _add_replicate_points(fig: go.Figure, reps_df: pl.DataFrame, order: list[str]) -> None:
    """Overlay jittered per-replicate dots on a categorical bar chart.

    Reuses the ATP-figure trick: a transparent Box carrying ``boxpoints='all'``
    renders only the jittered points (no visible box) over the bars.
    """
    sub = _order(reps_df.filter(pl.col("condition").is_in(order)), order)
    if sub.is_empty():
        return
    transparent = "rgba(0,0,0,0)"
    # One Box per condition so each dot's fill matches its bar; a black outline
    # keeps the points readable where they overlap the bars.
    for cond in order:
        cond_sub = sub.filter(pl.col("condition") == cond)
        if cond_sub.is_empty():
            continue
        fig.add_trace(
            go.Box(
                x=cond_sub["condition"].to_list(),
                y=cond_sub["log2fc"].to_list(),
                customdata=cond_sub["rep"].to_list(),
                marker=dict(
                    color=EXHAUSTION_COLS.get(cond, "#888"),
                    size=5,
                    line=dict(color="#000", width=0.6),
                ),
                line=dict(color=transparent),
                fillcolor=transparent,
                boxpoints="all",
                jitter=0.6,
                pointpos=0,
                hoveron="points",
                hovertemplate="%{customdata}: log2FC = %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )


# --------------------------------------------------------------------------- #
# abundance bar charts (proteome + rna share a shape)
# --------------------------------------------------------------------------- #
def _abundance_bar(
    df: pl.DataFrame, order: list[str], yaxis: str,
    hover_extra: str = "", reps_df: pl.DataFrame | None = None,
    note: str | None = None,
) -> go.Figure:
    df = _order(df, order)
    conditions = df["condition"].to_list()
    rows = df.to_dicts()
    # The bar is whatever ``log2fc`` the caller put on the row — a published
    # estimate, not a summary of the overlaid points. A null stays null so the
    # bar is a gap rather than a zero.
    values = [row["log2fc"] for row in rows]
    colors = [EXHAUSTION_COLS.get(c, "#888") for c in conditions]
    error_y = (
        _error_y(_replicate_sem(reps_df, order), conditions)
        if _has_reps(reps_df) else None
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=values,
            marker_color=colors,
            marker_line_color="#333",
            marker_line_width=0.5,
            width=0.5,
            error_y=error_y,
            customdata=rows,
            hovertemplate="<b>%{x}</b><br>log2FC = %{y:.2f}" + hover_extra
            + "<extra></extra>",
        )
    )
    # ±log2(1.5) reference guides
    for y in (FC_CUTOFF, -FC_CUTOFF):
        fig.add_hline(y=y, line_dash="dash", line_color="#bbb", line_width=1)
    fig.add_hline(y=0, line_color="#666", line_width=1)
    if _has_reps(reps_df):
        _add_replicate_points(fig, reps_df, order)
    layout = dict(_BASE_LAYOUT)
    if note:
        # caveat sits above the plot, so it can't be missed by a reader who
        # only looks at the bars
        layout["margin"] = _LEGEND_TOP_MARGIN
    fig.update_layout(**layout)
    if note:
        fig.add_annotation(
            text=note, xref="paper", yref="paper", x=0, y=1.0,
            xanchor="left", yanchor="bottom", showarrow=False,
            font=dict(size=11, color="#a15c00"),
        )
    fig.update_yaxes(title=yaxis, zeroline=False)
    fig.update_xaxes(
        title=CONDITION_AXIS_TITLE, categoryorder="array", categoryarray=order
    )
    return fig


def _evidence_note(
    df: pl.DataFrame, reps_df: pl.DataFrame | None = None
) -> str | None:
    """Caveat for whole-proteome bars resting on thin evidence.

    Silent for the ordinary case. Names the volcano omission when a D2
    reference was missing, so the per-gene view and the dataset-wide view don't
    appear to disagree for no reason.

    Also names the conditions where a donor fell below detection, because those
    replicate points are floored bounds rather than measurements — they cannot
    fall below the limit of detection while the published bar can, which is the
    whole reason the dots sit off the bar on F13A1, HBB and TNFRSF4. Keyed off
    the per-channel ``censored`` flag rather than a low ``n_reps`` threshold,
    which covers the cases badly: ``< 3`` fires on 6.2% of cells and still
    misses TNFRSF4.
    """
    parts: list[str] = []
    no_ref: list[str] = []
    if "d2_below_detection" in df.columns:
        no_ref = df.filter(pl.col("d2_below_detection"))["condition"].to_list()
        if no_ref:
            parts.append(
                f"D2 reference below detection in {', '.join(no_ref)} — "
                "fold change is a lower bound; omitted from the volcano"
            )
    if _has_reps(reps_df) and "censored" in reps_df.columns:
        # conditions the clause above already covers are skipped: a missing D2
        # is why those channels were censored, so naming them twice adds words
        # and no information.
        conds = [
            c
            for c in reps_df.filter(pl.col("censored"))["condition"]
            .unique(maintain_order=True)
            .to_list()
            if c not in no_ref
        ]
        if conds:
            parts.append(
                f"donor below detection in {', '.join(conds)} — those replicate "
                "points are bounds, not measurements"
            )
    if "n_reps" in df.columns:
        thin = df.filter(pl.col("n_reps") < 2)["condition"].to_list()
        if thin:
            parts.append(f"single donor in {', '.join(thin)}")
    return "⚠ " + "; ".join(parts) if parts else None


def proteome_figure(
    df: pl.DataFrame, symbol: str, reps_df: pl.DataFrame | None = None
) -> go.Figure:
    if df.is_empty():
        return _empty("No whole-proteome data")
    # Drop the D2 bar: D2 is the normalization reference (percent-of-control -> log2FC
    # ≈ 0), so its flat bar carries no information. D2 stays in the underlying data
    # (it's the denominator); we only omit it from the rendered bars/points here.
    # FOUR_CONDITION_ORDER is exactly the proteome conditions minus D2.
    df = df.filter(pl.col("condition") != "D2")
    if reps_df is not None:
        reps_df = reps_df.filter(pl.col("condition") != "D2")
    # Bar = the S2-1 volcano block's own log2FC, which is exactly what the
    # proteome volcano plots on its x axis and what whole_proteome.csv reports —
    # so a protein's bar and its volcano point are the same number by
    # construction. (Deliberately NOT the mean of the replicates drawn over it:
    # that mean tracked the published value closely but not exactly, and parted
    # from it by up to 1.3 log2 wherever a donor was censored at the limit of
    # detection.) Selected into `log2fc` here rather than inside _abundance_bar,
    # which stays generic; a slice from a parquet without the column falls back
    # to the aggregate it used to draw.
    if "published_log2fc" in df.columns:
        df = df.with_columns(pl.col("published_log2fc").alias("log2fc"))
    return _abundance_bar(
        df, FOUR_CONDITION_ORDER,
        yaxis="protein abundance, log₂(FC from D2)",
        hover_extra="<br>%{customdata.n_reps} donor(s)",
        reps_df=reps_df,
        note=_evidence_note(df, reps_df),
    )


def rna_figure(
    df: pl.DataFrame, symbol: str, reps_df: pl.DataFrame | None = None
) -> go.Figure:
    if df.is_empty():
        return _empty("No RNA-seq data")
    df = _order(df, FOUR_CONDITION_ORDER)
    conditions = df["condition"].to_list()
    deseq = df["log2fc"].to_list()
    padj = df["padj"].to_list()
    colors = [EXHAUSTION_COLS.get(c, "#888") for c in conditions]

    # Error bar = DESeq2's lfcSE, the GLM's own standard error for the log2FC the
    # bar draws — so bar and interval describe one published quantity.
    # Deliberately NOT the SEM of the replicate points below: that SEM takes the
    # spread of the three treatment samples about their own mean and treats the
    # D2 reference as error-free, dropping the reference group's contribution
    # entirely. It runs ~1.9x under lfcSE as a result. The proteome bars do draw
    # a replicate SEM, but there the points are already per-donor ratios with
    # the reference inside them, and no published SE exists to use instead.
    lfc_se = (
        df["lfc_se"].to_list() if "lfc_se" in df.columns else [None] * len(deseq)
    )

    # Bar = the DESeq2 model estimate, which is exactly what the transcriptome
    # volcano plots on its x axis and what rna.csv reports — so a gene's bar and
    # its volcano point are the same number by construction, and cannot drift.
    # (Deliberately NOT the mean of the replicates drawn over it; the proteome
    # bars follow the same rule, drawing their own published log2FC.) The
    # replicates are normalized onto the same log2 scale in build_rna_replicates,
    # so they still centre on the bar for any reasonably expressed gene.
    has_reps = _has_reps(reps_df)

    hovertemplate = (
        "<b>%{x}</b><br>DESeq2 log2FC = %{y:.2f} ± %{customdata[1]:.2f}"
        "<br>padj = %{customdata[0]:.2g}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=deseq,
            marker_color=colors,
            marker_line_color="#333",
            marker_line_width=0.5,
            width=0.5,
            error_y=_error_y(dict(zip(conditions, lfc_se)), conditions),
            customdata=[[p, s] for p, s in zip(padj, lfc_se)],
            hovertemplate=hovertemplate,
        )
    )
    # ±log2(2) guides, not the proteome's ±log2(1.5): the RNA views are held to
    # the stricter cutoff the transcriptome volcano calls significance on.
    for y in (RNA_FC_CUTOFF, -RNA_FC_CUTOFF):
        fig.add_hline(y=y, line_dash="dash", line_color="#bbb", line_width=1)
    fig.add_hline(y=0, line_color="#666", line_width=1)
    if has_reps:
        _add_replicate_points(fig, reps_df, FOUR_CONDITION_ORDER)
    fig.update_layout(**_BASE_LAYOUT)
    fig.update_yaxes(title="mRNA, log₂(FC from D2)", zeroline=False)
    fig.update_xaxes(
        title=CONDITION_AXIS_TITLE, categoryorder="array",
        categoryarray=FOUR_CONDITION_ORDER,
    )
    return fig


# --------------------------------------------------------------------------- #
# reactivity — per-condition cysteine dot plot (x = vs-D2 comparisons)
# --------------------------------------------------------------------------- #
def _is_change(value) -> bool:
    """reactivity_change flags significant cysteines (bool ``True``, or the
    string ``"True"`` if the column ever arrives untyped); null otherwise."""
    if value is None:
        return False
    if isinstance(value, str):
        return value == "True"
    return bool(value)


def reactivity_figure(
    df: pl.DataFrame, symbol: str, reps_df: pl.DataFrame | None = None
) -> go.Figure:
    """Per-condition cysteine-reactivity dot plot for a single protein.

    Each cysteine is a dot placed in its condition column (x = the four vs-D2
    comparisons), colored by whether it is a significant reactivity change —
    ``Higher`` / ``Lower`` — or ``Unchanged``; significant cysteines are labeled
    with their residue id. A green triangle per condition marks the
    whole-proteome (protein-expression) log2FC. Mirrors the manuscript facetted
    dot plot (reactivity_visualization.Rmd) for one protein.
    """
    if df.is_empty():
        return _empty("No cysteine-reactivity data")

    order = FOUR_CONDITION_ORDER
    xpos = {c: i for i, c in enumerate(order)}

    # per-cysteine points grouped by regulation category
    cats: dict[str, list[dict]] = {"Unchanged": [], "Lower": [], "Higher": []}
    wp: dict[str, float] = {}
    for row in df.iter_rows(named=True):
        cond = row["condition"]
        if cond not in xpos:
            continue
        wp_val = row["wp_log2fc"]
        if wp_val is not None and cond not in wp:
            wp[cond] = wp_val
        y = row["log2fc"]
        if y is None:
            continue
        if _is_change(row["reactivity_change"]):
            cat = "Higher" if y > 0 else "Lower"
        else:
            cat = "Unchanged"
        cats[cat].append(
            {
                "x": xpos[cond],
                "y": y,
                "residue": row["residue"],
                "cond": cond,
            }
        )

    fig = go.Figure()
    for y in (REACTIVITY_GUIDE, -REACTIVITY_GUIDE):
        fig.add_hline(y=y, line_dash="dash", line_color="#bbb", line_width=1)
    fig.add_hline(y=0, line_color="#666", line_width=1)

    # draw Unchanged first so significant (labeled) points sit on top; the
    # legend doubles as a per-category show/hide toggle.
    label_pos = {"Higher": "top center", "Lower": "bottom center"}
    for cat in ("Unchanged", "Lower", "Higher"):
        pts = cats[cat]
        if not pts:
            continue
        labeled = cat in label_pos
        fig.add_trace(
            go.Scatter(
                name=cat,
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                mode="markers+text" if labeled else "markers",
                text=[p["residue"] for p in pts] if labeled else None,
                textposition=label_pos.get(cat),
                textfont=dict(size=10, color=REGULATION_COLORS[cat]),
                marker=dict(
                    color=REGULATION_COLORS[cat],
                    line=dict(color="#333", width=0.8),
                    size=9,
                    symbol="circle",
                ),
                customdata=[[p["residue"], p["cond"]] for p in pts],
                hovertemplate="<b>%{customdata[0]}</b> — %{customdata[1]}"
                "<br>log2FC = %{y:.2f}"
                f"<br>{cat}<extra></extra>",
            )
        )

    # whole-proteome (protein-expression) reference: one triangle per condition
    wp_conds = [c for c in order if c in wp]
    if wp_conds:
        fig.add_trace(
            go.Scatter(
                name="Protein expression",
                x=[xpos[c] for c in wp_conds],
                y=[wp[c] for c in wp_conds],
                mode="markers",
                marker=dict(
                    color=REGULATION_COLORS["Protein expression"],
                    line=dict(color="black", width=0.8),
                    size=11,
                    symbol="triangle-up",
                ),
                customdata=[[c] for c in wp_conds],
                hovertemplate="<b>Protein expression</b> — %{customdata[0]}"
                "<br>log2FC = %{y:.2f}<extra></extra>",
            )
        )

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = _LEGEND_TOP_MARGIN
    layout["showlegend"] = True
    fig.update_layout(
        **layout,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_yaxes(title="cysteine reactivity, log₂(FC from D2)", zeroline=False)
    fig.update_xaxes(
        title=CONDITION_AXIS_TITLE,
        tickmode="array",
        tickvals=[xpos[c] for c in order],
        ticktext=list(order),
        range=[-0.5, len(order) - 0.5],
    )
    return fig


# --------------------------------------------------------------------------- #
# reactivity — ATP add-back (the MAP2K4-style box + jitter)
# --------------------------------------------------------------------------- #
def reactivity_atp_figure(
    df: pl.DataFrame, symbol: str, reps_df: pl.DataFrame | None = None
) -> go.Figure:
    if df.is_empty():
        return _empty("No ATP add-back data")
    residues = (
        df.select(["residue", "residue_loc"])
        .unique()
        .sort("residue_loc", nulls_last=True)["residue"]
        .to_list()
    )
    fig = go.Figure()
    for cond in ATP_ORDER:
        sub = df.filter(pl.col("condition") == cond)
        if sub.is_empty():
            continue
        # one grouped box trace per condition across residues (x = residue).
        # customdata carries the replicate label so each jittered point names
        # the replicate it came from (matching the whole-proteome hover).
        fig.add_trace(
            go.Box(
                name=cond,
                legendrank=ATP_LEGEND_ORDER.index(cond) + 1,
                x=sub["residue"].to_list(),
                y=sub["lfc"].to_list(),
                customdata=sub["rep"].to_list(),
                marker=dict(
                    color=ATP_FILLS.get(cond, "#888"),
                    line=dict(color=ATP_OUTLINE, width=0.8),
                    symbol=ATP_SYMBOLS.get(cond, "circle"),
                    size=5,
                ),
                line=dict(color=ATP_OUTLINE, width=1),
                fillcolor=ATP_FILLS.get(cond, "#888"),
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                opacity=0.9,
                hoveron="points",
                hovertemplate=f"<b>{cond}</b> — %{{x}}<br>rep %{{customdata}}"
                "<br>log2FC = %{y:.2f}<extra></extra>",
            )
        )
    for y in (REACTIVITY_GUIDE, -REACTIVITY_GUIDE):
        fig.add_hline(y=y, line_dash="dash", line_color="#bbb", line_width=1)
    fig.add_hline(y=0, line_color="#666", line_width=1)
    layout = dict(_BASE_LAYOUT)
    layout["margin"] = _LEGEND_TOP_MARGIN
    layout["showlegend"] = True
    fig.update_layout(
        **layout,
        boxmode="group",
        # entrywidth pins three entries per row so the 3x2 grid holds at any
        # card width, instead of letting Plotly pick the wrap point.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            x=0,
            entrywidthmode="fraction",
            entrywidth=1 / 3,
        ),
    )
    fig.update_yaxes(title="cysteine reactivity, log₂(FC from D2)", zeroline=False)
    fig.update_xaxes(title="cysteine site", categoryorder="array", categoryarray=residues)
    return fig


# --------------------------------------------------------------------------- #
# whole-proteome volcano — dataset-wide (all proteins for one comparison)
# --------------------------------------------------------------------------- #
def _volcano_x_title(comparison: str) -> str:
    """x-axis title naming the comparison's reference condition.

    A bare comparison id ("D8C") is against D2, the implicit reference the whole
    portal uses. An id written "<numerator>_vs_<denominator>" names both. This
    axis is the only place the reference appears, which is what lets the
    comparison picker's labels stay short (see store.VOLCANO_COMPARISONS).
    """
    if "_vs_" in comparison:
        num, den = comparison.split("_vs_", 1)
        return f"log₂(FC, {num} / {den})"
    return "log₂(FC from D2)"


def volcano_figure(
    df: pl.DataFrame,
    comparison: str,
    highlight: str | None = None,
    *,
    y_col: str = "neglog10_pval",
    p_col: str = "p_value",
    p_label: str = "p",
    y_title: str = "−log₁₀(p)",
    fc_cutoff: float = FC_CUTOFF,
    note: str | None = None,
) -> go.Figure:
    """Volcano for one comparison: x = log2FC, y = a −log10 significance.

    One WebGL scatter trace per regulation category (so the legend doubles as a
    show/hide toggle and significant points draw on top). ``highlight`` pins a
    single protein with a labelled marker — used to locate the searched gene.

    Points are identified by ``label`` (``Store.volcano_slice`` supplies it): the
    bare symbol, or ``"TMPO (P42166)"`` where a symbol was measured under several
    accessions. The whole-proteome volcano has one point per *accession*, so
    matching on symbol pinned an arbitrary one of a split symbol's two points and
    click-through could not tell which protein was meant.

    The keyword arguments exist so the transcriptome volcano
    (:func:`rna_volcano_figure`) can reuse this body: it plots DESeq2's adjusted
    p against a 2-fold cutoff, where the whole proteome plots the manuscript's
    raw p against 1.5-fold. Everything else — the WebGL layering, the pin, the
    annotation that has to sit above the gl canvas — is identical and must not
    be duplicated.
    """
    if df.is_empty():
        return _empty("No volcano data")

    # tolerate a frame built without Store.volcano_slice (tests, ad-hoc calls)
    if "label" not in df.columns:
        df = df.with_columns(pl.col("symbol").alias("label"))

    fig = go.Figure()
    for reg in VOLCANO_REG_ORDER:
        sub = df.filter(pl.col("regulation") == reg)
        if sub.is_empty():
            continue
        fig.add_trace(
            go.Scattergl(
                name=reg,
                x=sub["log2fc"].to_list(),
                y=sub[y_col].to_list(),
                mode="markers",
                marker=dict(
                    color=VOLCANO_REG_COLORS.get(reg, "#888"),
                    size=5,
                    line=dict(width=0),
                    opacity=0.75,
                ),
                # customdata[0] = entry label; drives click-through to the page
                # for that exact protein, and is what the pin matches on
                customdata=[[s, p] for s, p in zip(
                    sub["label"].to_list(), sub[p_col].to_list()
                )],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>log2FC = %{x:.2f}"
                    f"<br>{p_label} = %{{customdata[1]:.2g}}"
                    f"<br>{reg}<extra></extra>"
                ),
            )
        )

    # significance guides: vertical at the fold-change cutoff, horizontal at the
    # p cutoff (0.05 on whichever p the dataset calls significance with)
    for x in (fc_cutoff, -fc_cutoff):
        fig.add_vline(x=x, line_dash="dash", line_color="#bbb", line_width=1)
    fig.add_hline(
        y=-np.log10(PVAL_CUTOFF), line_dash="dash", line_color="#bbb",
        line_width=1,
    )
    fig.add_vline(x=0, line_color="#666", line_width=1)

    # Pin the searched protein, if it's in this comparison. Matched on label, so
    # a split symbol pins the accession the reader actually opened rather than
    # whichever row came first. Kept in step with app.js pinVolcano().
    if highlight:
        pin = df.filter(pl.col("label") == highlight)
        if not pin.is_empty():
            r = pin.row(0, named=True)
            hx, hy = r["log2fc"], r[y_col]
            # Marker: a Scattergl trace added *last*, so it draws on top of the
            # WebGL point cloud (SVG traces can't — the gl canvas sits above the
            # SVG layer). A filled dot with a white halo reads clearly on top.
            fig.add_trace(
                go.Scattergl(
                    name=highlight,
                    x=[hx],
                    y=[hy],
                    mode="markers",
                    marker=dict(
                        color=VOLCANO_HIGHLIGHT,
                        size=12,
                        symbol="circle",
                        line=dict(width=2.5, color="#ffffff"),
                    ),
                    hovertemplate=(
                        f"<b>{highlight}</b><br>log2FC = %{{x:.2f}}"
                        f"<br>{p_label} = {r[p_col]:.2g}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            # Label: an annotation renders in the top SVG infolayer, i.e. above
            # the WebGL canvas — so it's never occluded by points.
            fig.add_annotation(
                x=hx,
                y=hy,
                text=highlight,
                showarrow=False,
                yshift=14,
                font=dict(size=12, color=VOLCANO_HIGHLIGHT, family=FONT_FAMILY),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=VOLCANO_HIGHLIGHT,
                borderwidth=1,
                borderpad=2,
            )

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = _LEGEND_TOP_MARGIN
    layout["showlegend"] = True
    fig.update_layout(
        **layout,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    if note:
        # sits under the legend row, above the plotting area
        fig.add_annotation(
            text=note, xref="paper", yref="paper", x=1, y=1.0,
            xanchor="right", yanchor="bottom", showarrow=False,
            font=dict(size=11, color="#6b7280"),
        )
    fig.update_xaxes(title=_volcano_x_title(comparison), zeroline=False)
    fig.update_yaxes(title=y_title, zeroline=False)
    return fig


def rna_volcano_figure(
    df: pl.DataFrame, comparison: str, highlight: str | None = None
) -> go.Figure:
    """Transcriptome volcano, restricted to genes with matched proteomic data.

    Same body as the whole-proteome volcano on a different significance basis:
    DESeq2's adjusted p and a 2-fold cutoff, rather than the manuscript's raw p
    and 1.5-fold. The note records the two things a reader can't see from the
    cloud — that the gene universe is the matched ~6.2k rather than the whole
    transcriptome, and that padj still comes from the transcriptome-wide fit.
    """
    return volcano_figure(
        df,
        comparison,
        highlight,
        y_col="neglog10_padj",
        p_col="padj",
        p_label="padj",
        y_title="−log₁₀(adj. p)",
        fc_cutoff=RNA_FC_CUTOFF,
        note="genes with matched whole-proteome data; "
             "adj. p from the transcriptome-wide fit",
    )


VOLCANO_BUILDERS = {
    "proteome": volcano_figure,
    "rna": rna_volcano_figure,
}


BUILDERS = {
    "proteome": proteome_figure,
    "rna": rna_figure,
    "reactivity": reactivity_figure,
    "reactivity_atp": reactivity_atp_figure,
}
