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


@pytest.fixture
def stub(monkeypatch):
    """Drive ``build_rna_volcano`` off synthetic RNA + proteome tables."""

    def _apply(rows: list[dict], matched: list[str] | None = None) -> pd.DataFrame:
        matched = [r["symbol"] for r in rows] if matched is None else matched
        monkeypatch.setattr(build_db, "build_rna", lambda: _rna_frame(rows))
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
