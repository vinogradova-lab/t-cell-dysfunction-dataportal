"""The per-gene RNA bar must plot the same number as the transcriptome volcano.

Both read ``rna.log2fc`` — DESeq2's ``log2FoldChange`` from S1-1 — so a gene's
bar height and its volcano x-position are one value and cannot drift apart. The
bar used to be the mean of the overlaid replicate points instead, which is how
GBE1 came to draw -0.86 at D4A while the volcano plotted -1.30.

The replicate points are supporting evidence layered on top; they are now on the
same log2 scale (see tests/test_rna_replicates.py) but they are not what sets the
bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portal"))

from figures import rna_figure  # noqa: E402


ORDER = ["D4A", "D4C", "D8A", "D8C"]


def _rna(log2fc: dict[str, float], lfc_se: dict[str, float] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["TEST"] * len(log2fc),
            "condition": list(log2fc),
            "log2fc": list(log2fc.values()),
            "lfc_se": [(lfc_se or {}).get(c, 0.1) for c in log2fc],
            "padj": [1e-6] * len(log2fc),
            "base_mean": [1000.0] * len(log2fc),
        }
    )


def _reps(values: dict[str, list[float]]) -> pl.DataFrame:
    rows = [
        {"symbol": "TEST", "condition": cond, "rep": f"rep{i}", "log2fc": v}
        for cond, vals in values.items()
        for i, v in enumerate(vals, start=1)
    ]
    return pl.DataFrame(rows)


def _bar(fig):
    """The bar trace's (condition -> y), ignoring the replicate scatter."""
    trace = next(t for t in fig.data if t.type == "bar")
    return dict(zip(trace.x, trace.y))


def test_bar_is_the_deseq2_estimate_not_the_replicate_mean():
    """The regression guard: replicates deliberately disagree with the model
    estimate, and the bar must follow the model."""
    deseq = {"D4A": -1.303, "D4C": -0.292, "D8A": -0.546, "D8C": 0.469}
    fig = rna_figure(
        _rna(deseq), "TEST",
        _reps({c: [0.0, 0.0, 0.0] for c in ORDER}),   # nowhere near `deseq`
    )
    assert _bar(fig) == pytest.approx(deseq)


def test_bar_is_unchanged_when_replicates_are_absent():
    """Genes with no per-replicate counts plotted the DESeq2 value before this
    change too — that path must stay identical."""
    deseq = {"D4A": -1.303, "D4C": -0.292, "D8A": -0.546, "D8C": 0.469}
    assert _bar(rna_figure(_rna(deseq), "TEST", None)) == pytest.approx(deseq)
    assert _bar(
        rna_figure(_rna(deseq), "TEST", _reps({}))
    ) == pytest.approx(deseq)


def test_replicate_points_are_still_drawn_over_the_bar():
    """The bar not being their mean must not mean they were dropped."""
    fig = rna_figure(
        _rna({c: 1.0 for c in ORDER}),
        "TEST",
        _reps({c: [0.9, 1.0, 1.1] for c in ORDER}),
    )
    plotted = [y for t in fig.data if t.type != "bar" for y in (t.y or [])]
    assert sorted(set(plotted)) == pytest.approx([0.9, 1.0, 1.1])


def test_error_bars_are_lfcse_not_the_replicate_sem():
    """The interval must describe the quantity the bar draws. The replicates
    here have SEM 0.0577, which is what we used to draw and must not come back.
    """
    se = {"D4A": 0.098, "D4C": 0.097, "D8A": 0.097, "D8C": 0.096}
    fig = rna_figure(
        _rna({c: 1.0 for c in ORDER}, se),
        "TEST",
        _reps({c: [0.9, 1.0, 1.1] for c in ORDER}),
    )
    trace = next(t for t in fig.data if t.type == "bar")
    assert list(trace.error_y.array) == pytest.approx([se[c] for c in ORDER])


def test_error_bars_survive_without_replicates():
    """lfcSE comes from the model, not the points, so a gene with no
    per-replicate counts still gets an interval — it never could before."""
    fig = rna_figure(_rna({c: 1.0 for c in ORDER}, {c: 0.2 for c in ORDER}), "TEST", None)
    trace = next(t for t in fig.data if t.type == "bar")
    assert list(trace.error_y.array) == pytest.approx([0.2] * 4)


def test_missing_lfcse_leaves_no_bar():
    """DESeq2 leaves lfcSE null where it fit no model; a null must not render as
    a zero-length interval implying perfect precision."""
    df = _rna({c: 1.0 for c in ORDER}).with_columns(
        pl.Series("lfc_se", [None, 0.2, float("nan"), 0.3])
    )
    trace = next(t for t in rna_figure(df, "TEST", None).data if t.type == "bar")
    assert list(trace.error_y.array) == [None, pytest.approx(0.2), None, pytest.approx(0.3)]


def test_hover_reports_the_deseq2_value():
    """The hover used to name two numbers because they differed; it now names
    the one the bar draws, and must not silently go back to 'mean of
    replicates'."""
    fig = rna_figure(
        _rna({c: 1.0 for c in ORDER}), "TEST",
        _reps({c: [0.9, 1.0, 1.1] for c in ORDER}),
    )
    template = next(t for t in fig.data if t.type == "bar").hovertemplate
    assert "DESeq2 log2FC" in template
    assert "mean of replicates" not in template
