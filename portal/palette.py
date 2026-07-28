"""Colors, condition ordering, and thresholds — ported verbatim from the
manuscript figure code so the portal matches the published figures.

Sources:
  * bin/figure_utils.R                     (EXHAUSTION_COLS, regulation colors)
  * notebooks/03_reactivity/
        reactivity_visualization.Rmd       (add_back fills / shapes)
"""

# 5-condition exhaustion palette (figure_utils.R `cols` / EXHAUSTION_COLS)
EXHAUSTION_COLS = {
    "D2": "#A1A1A1",
    "D4A": "#ABA2D6",
    "D8A": "#603EA6",
    "D4C": "#FBB31B",
    "D8C": "#E27753",
}

# ordering used for the 5-condition views
FIVE_CONDITION_ORDER = ["D2", "D4A", "D4C", "D8A", "D8C"]
# RNA / reactivity have no D2 row (D2 is the reference, log2FC = 0)
FOUR_CONDITION_ORDER = ["D4A", "D4C", "D8A", "D8C"]

# regulation palette (figure_utils.R)
REGULATION_COLORS = {
    "Higher": "#EE6363",       # indianred2
    "Lower": "#5CACEE",        # steelblue2
    "Unchanged": "#D3D3D3",    # lightgrey
    "Protein expression": "#54A868",
}

# ATP add-back fills / outline / shapes (reactivity_visualization.Rmd 1171-1202)
ATP_ORDER = ["D2", "D2-ATP", "D8A", "D8A-ATP", "D8C", "D8C-ATP"]
ATP_FILLS = {
    "D2": "#A9A9A9",       # darkgrey
    "D2-ATP": "#FCFCFC",   # grey99
    "D8A": "#603EA6",
    "D8A-ATP": "#D13B95",
    "D8C": "#E27753",
    "D8C-ATP": "#EF3D3B",
}
# base condition = filled circle, "-ATP" variant = diamond
ATP_SYMBOLS = {
    "D2": "circle",
    "D2-ATP": "diamond",
    "D8A": "circle",
    "D8A-ATP": "diamond",
    "D8C": "circle",
    "D8C-ATP": "diamond",
}
ATP_OUTLINE = "black"

# volcano regulation categories (whole-proteome DE calls) -> colors.
# Up/Down (significant + |log2FC| >= log2(1.5)) reuse the manuscript regulation
# reds/blues; "significant but small FC" is a muted tint; everything else grey.
# The two volcanoes share this vocabulary but not their cutoffs: the proteome
# uses the manuscript's log2(1.5), the transcriptome log2(2), so each has its own
# "significant but small fold change" bucket.
VOLCANO_REG_COLORS = {
    "Significant Up": "#EE6363",        # indianred2 (Higher)
    "Significant Down": "#5CACEE",      # steelblue2 (Lower)
    "Significant but <1.5 FC": "#B0B0B0",
    "Significant but <2 FC": "#B0B0B0",
    "Not Significant": "#D3D3D3",       # lightgrey (Unchanged)
    "Not Significant Up": "#D3D3D3",
    "Not Significant Down": "#D3D3D3",
}
# legend/draw order (least to most salient, so significant points end up on top).
# One list covers both datasets — categories a dataset never emits are skipped.
VOLCANO_REG_ORDER = [
    "Not Significant",
    "Not Significant Up",
    "Not Significant Down",
    "Significant but <1.5 FC",
    "Significant but <2 FC",
    "Significant Down",
    "Significant Up",
]
# highlight color for a searched protein pinned onto the volcano
VOLCANO_HIGHLIGHT = "#111111"

# thresholds
FC_CUTOFF = 0.5849625007211562  # log2(1.5) — the manuscript's proteome cutoff
# The RNA views use a stricter 2-fold cutoff against DESeq2's adjusted p, rather
# than the proteome's 1.5-fold against a raw p. Applies to both the per-gene RNA
# bar chart's dashed guides and the transcriptome volcano, so the two agree.
RNA_FC_CUTOFF = 1.0             # log2(2)
PVAL_CUTOFF = 0.05              # significance threshold used for the volcano guide
REACTIVITY_GUIDE = 1.0          # dashed guides at ±1 on reactivity boxplots

FONT_FAMILY = "Arial, Helvetica, sans-serif"
