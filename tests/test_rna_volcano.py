"""Significance calls and gene filtering for the transcriptome volcano.

The RNA volcano deliberately does *not* reuse the whole proteome's rule: it
calls significance on DESeq2's adjusted p with a 2-fold cutoff, where the
proteome reproduces the manuscript's raw-p / 1.5-fold ``Regulation`` column.
Mixing the two up would relabel thousands of genes without any visible error, so
each row of the rule table is pinned here, along with the two ways a gene drops
out of the plot entirely (no matched protein, or no padj).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

import build_db  # noqa: E402


def _rna_frame(rows: list[dict]) -> pd.DataFrame:
    """Long RNA table in the shape ``build_rna`` returns."""
    return pd.DataFrame(
        [
            {
                "symbol": r["symbol"],
                "condition": r.get("condition", "D8C"),
                "log2fc": r["log2fc"],
                "padj": r["padj"],
                "base_mean": r.get("base_mean", 100.0),
            }
            for r in rows
        ]
    )


_S2_2_COLS = [
    "protein",
    "D8C vs D8A RNA_log2FoldChange",
    "D8C vs D8A RNA_pvalue",
    "D8C vs D8A RNA_padj",
    # a plot colour code, not a significance call — must be ignored
    "D8C vs D8A regulation_protein_rna",
]


def _s2_2_frame(rows: list[dict]) -> pd.DataFrame:
    """S2-2 block in the shape the sheet ships it, numerator-first."""
    return pd.DataFrame(
        [
            {
                "protein": r["symbol"],
                "D8C vs D8A RNA_log2FoldChange": r["log2fc"],
                "D8C vs D8A RNA_pvalue": r.get("pvalue", 0.001),
                "D8C vs D8A RNA_padj": r["padj"],
                "D8C vs D8A regulation_protein_rna": "purple",
            }
            for r in rows
        ],
        columns=_S2_2_COLS,
    )


@pytest.fixture
def stub(monkeypatch):
    """Drive ``build_rna_volcano`` off synthetic RNA + proteome tables."""

    def _apply(
        rows: list[dict],
        matched: list[str] | None = None,
        s2_2: list[dict] | None = None,
    ) -> pd.DataFrame:
        matched = [r["symbol"] for r in rows] if matched is None else matched
        monkeypatch.setattr(build_db, "build_rna", lambda: _rna_frame(rows))
        monkeypatch.setattr(
            build_db, "load_s2_2", lambda: _s2_2_frame(s2_2 or [])
        )
        monkeypatch.setattr(
            build_db,
            "build_proteome",
            lambda: pd.DataFrame({"symbol": matched}),
        )
        monkeypatch.setattr(
            build_db,
            "load_s2_1",
            lambda: pd.DataFrame(
                {"protein": matched, "description": [f"{s} desc" for s in matched]}
            ),
        )
        return build_db.build_rna_volcano()

    return _apply


def _reg(out: pd.DataFrame, symbol: str) -> str:
    return out.loc[out["symbol"] == symbol, "regulation"].iloc[0]


# padj < 0.05 and |log2FC| >= log2(2) -> significant; each combination of the
# two conditions gets its own category, mirroring the S2-1 vocabulary.
@pytest.mark.parametrize(
    "log2fc, padj, expected",
    [
        (1.5, 0.01, "Significant Up"),
        (-1.5, 0.01, "Significant Down"),
        (0.5, 0.01, "Significant but <2 FC"),
        (-0.5, 0.01, "Significant but <2 FC"),
        (1.5, 0.20, "Not Significant Up"),
        (-1.5, 0.20, "Not Significant Down"),
        (0.5, 0.20, "Not Significant"),
    ],
)
def test_regulation_rule(stub, log2fc, padj, expected):
    out = stub([{"symbol": "G1", "log2fc": log2fc, "padj": padj}])
    assert _reg(out, "G1") == expected


def test_cutoffs_are_inclusive_at_the_boundary(stub):
    """log2FC exactly at log2(2) counts as a 2-fold change."""
    out = stub([{"symbol": "G1", "log2fc": build_db.RNA_FC_CUTOFF, "padj": 0.01}])
    assert _reg(out, "G1") == "Significant Up"


def test_genes_without_matched_proteomics_are_dropped(stub):
    """The whole point of the plot: one gene universe with the proteome."""
    out = stub(
        [
            {"symbol": "KEEP", "log2fc": 1.5, "padj": 0.01},
            {"symbol": "RNAONLY", "log2fc": 1.5, "padj": 0.01},
        ],
        matched=["KEEP"],
    )
    assert out["symbol"].tolist() == ["KEEP"]


def test_genes_without_padj_are_dropped(stub):
    """DESeq2 independent filtering leaves padj null — no y-position."""
    out = stub(
        [
            {"symbol": "KEEP", "log2fc": 1.5, "padj": 0.01},
            {"symbol": "FILTERED", "log2fc": 1.5, "padj": np.nan},
        ]
    )
    assert out["symbol"].tolist() == ["KEEP"]


def test_padj_of_zero_does_not_plot_at_infinity(stub):
    """An underflowed padj would otherwise break parquet and Plotly both."""
    out = stub(
        [
            {"symbol": "UNDERFLOW", "log2fc": 3.0, "padj": 0.0},
            {"symbol": "SMALLEST", "log2fc": 2.0, "padj": 1e-300},
        ]
    )
    assert np.isfinite(out["neglog10_padj"]).all()
    pinned = out.loc[out["symbol"] == "UNDERFLOW", "neglog10_padj"].iloc[0]
    assert pinned == pytest.approx(300.0)


def test_carries_description_for_the_hover(stub):
    out = stub([{"symbol": "G1", "log2fc": 1.5, "padj": 0.01}])
    assert out.loc[0, "description"] == "G1 desc"


# ---- D8C vs D8A (supplementary sheet S2-2) ------------------------------- #
# The one non-vs-D2 transcriptome comparison the manuscript publishes. S1-1
# carries only the four vs-D2 contrasts, so this block comes from S2-2 — which
# labels its comparisons numerator-first, the *opposite* of S2-1's reversed
# convention. Taking it as reversed would mirror the whole panel about x = 0.


def _row(out: pd.DataFrame, symbol: str, comparison: str) -> pd.Series:
    m = (out["symbol"] == symbol) & (out["comparison"] == comparison)
    return out.loc[m].iloc[0]


def test_s2_2_block_is_taken_numerator_first(stub):
    """"D8C vs D8A" in S2-2 really is log2(D8C/D8A) — no sign flip."""
    out = stub(
        [{"symbol": "G1", "log2fc": 0.4, "padj": 0.01}],
        s2_2=[{"symbol": "G1", "log2fc": 1.5, "padj": 0.01}],
    )
    assert _row(out, "G1", "D8C_vs_D8A")["log2fc"] == pytest.approx(1.5)
    # ...and the vs-D2 row for the same gene is untouched
    assert _row(out, "G1", "D8C")["log2fc"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "log2fc, padj, expected",
    [
        (1.5, 0.01, "Significant Up"),
        (-1.5, 0.01, "Significant Down"),
        (0.5, 0.01, "Significant but <2 FC"),
        (1.5, 0.20, "Not Significant Up"),
        (0.5, 0.20, "Not Significant"),
    ],
)
def test_d8c_vs_d8a_uses_the_same_regulation_rule(stub, log2fc, padj, expected):
    """S2-2 ships its own ``regulation_protein_rna`` column, but it is a plot
    colour code ("purple", "dark grey", …) for a different figure. All five
    panels must share one rule so they share one legend and palette.
    """
    out = stub(
        [{"symbol": "G1", "log2fc": 0.1, "padj": 0.9}],
        s2_2=[{"symbol": "G1", "log2fc": log2fc, "padj": padj}],
    )
    assert _row(out, "G1", "D8C_vs_D8A")["regulation"] == expected


def test_d8c_vs_d8a_is_restricted_to_matched_genes(stub):
    """S2-2 covers ~6.1k genes, a slightly different set from the vs-D2 rows.
    One gene universe means the same restriction applies to both.
    """
    out = stub(
        [{"symbol": "KEEP", "log2fc": 1.5, "padj": 0.01}],
        matched=["KEEP"],
        s2_2=[
            {"symbol": "KEEP", "log2fc": 1.5, "padj": 0.01},
            {"symbol": "S2ONLY", "log2fc": 1.5, "padj": 0.01},
        ],
    )
    assert out.loc[out["comparison"] == "D8C_vs_D8A", "symbol"].tolist() == ["KEEP"]


def test_d8c_vs_d8a_shares_the_null_padj_and_underflow_handling(stub):
    """The shared tail, not a parallel copy of it: a null padj drops and a
    zero padj is floored, exactly as on the vs-D2 rows.
    """
    out = stub(
        [{"symbol": "KEEP", "log2fc": 1.5, "padj": 1e-300}],
        matched=["KEEP", "FILTERED", "UNDERFLOW"],
        s2_2=[
            {"symbol": "KEEP", "log2fc": 1.5, "padj": 0.01},
            {"symbol": "FILTERED", "log2fc": 1.5, "padj": np.nan},
            {"symbol": "UNDERFLOW", "log2fc": 3.0, "padj": 0.0},
        ],
    )
    extra = out[out["comparison"] == "D8C_vs_D8A"]
    assert sorted(extra["symbol"]) == ["KEEP", "UNDERFLOW"]
    assert np.isfinite(out["neglog10_padj"]).all()
    # floored against the smallest positive padj pooled across comparisons
    assert _row(out, "UNDERFLOW", "D8C_vs_D8A")["neglog10_padj"] == pytest.approx(300.0)


def test_comparison_ordering_puts_the_new_panel_last(stub):
    """Display order drives the picker; chronic-vs-acute reads as a follow-up
    to the four vs-D2 panels, not as one of them.
    """
    out = stub(
        [{"symbol": "G1", "condition": c, "log2fc": 0.4, "padj": 0.01}
         for c in ("D4A", "D4C", "D8A", "D8C")],
        s2_2=[{"symbol": "G1", "log2fc": 1.5, "padj": 0.01}],
    )
    assert out["comparison"].tolist() == build_db.VOLCANO_COMPARISONS
