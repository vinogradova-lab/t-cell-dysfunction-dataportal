"""The per-gene proteome bar must plot the same number as the proteome volcano.

Both read the S2-1 volcano block's ``log2_FC`` — the bar via
``proteome.published_log2fc``, the volcano via ``volcano.log2fc`` — so a
protein's bar height and its volcano x-position are one value and cannot drift
apart. The bar used to be the mean of the overlaid replicate points instead,
which is how F13A1 came to draw -2.78 at D8A while the volcano plotted -4.08.

That gap is not an error in either number. F13A1 was undetected in donor rep5 at
D8A, so that donor's channels are floored at its limit of detection: the points
are an upper bound on the loss and cannot fall below it, while the published
estimate used the raw zero and is free to. The ``censored`` flag is what the
figure uses to say so.

The replicate points are supporting evidence layered on top; they are not what
sets the bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portal"))

from figures import proteome_figure  # noqa: E402


ORDER = ["D4A", "D4C", "D8A", "D8C"]


def _proteome(
    published: dict[str, float | None],
    aggregate: dict[str, float] | None = None,
    d2_below_detection: dict[str, bool] | None = None,
) -> pl.DataFrame:
    """A `proteome` slice for one protein. D2 is included because the real slice
    carries it and `proteome_figure` is responsible for dropping it."""
    conds = ["D2"] + list(published)
    agg = aggregate or {}
    flag = d2_below_detection or {}
    return pl.DataFrame(
        {
            "uniprot": ["P00000"] * len(conds),
            "symbol": ["TEST"] * len(conds),
            "condition": conds,
            "percent_control": [100.0] * len(conds),
            "log2fc": [0.0] + [agg.get(c, 0.0) for c in published],
            "published_log2fc": [0.0] + list(published.values()),
            "n_reps": [6] * len(conds),
            "d2_below_detection": [False] + [flag.get(c, False) for c in published],
        },
        schema_overrides={"published_log2fc": pl.Float64},
    )


def _reps(
    values: dict[str, list[float]], censored: dict[str, bool] | None = None
) -> pl.DataFrame:
    """Per-channel replicate rows. Two channels per donor, as the real table."""
    flag = censored or {}
    rows = [
        {
            "uniprot": "P00000",
            "symbol": "TEST",
            "condition": cond,
            "rep": f"rep{i // 2 + 1}.{i % 2 + 1}",
            "bio_rep": f"rep{i // 2 + 1}",
            "percent_control": 100.0,
            "log2fc": v,
            "censored": flag.get(cond, False),
        }
        for cond, vals in values.items()
        for i, v in enumerate(vals)
    ]
    return pl.DataFrame(rows)


def _bar(fig):
    """The bar trace's (condition -> y), ignoring the replicate scatter."""
    trace = next(t for t in fig.data if t.type == "bar")
    return dict(zip(trace.x, trace.y))


def _note(fig) -> str:
    return " ".join(a.text for a in (fig.layout.annotations or []))


def test_bar_is_the_published_value_not_the_replicate_mean():
    """The regression guard, on the case that motivated the change. F13A1 D8A's
    four channels span -2.93..-2.63 while the published log2FC is -4.075; the
    bar must follow the published value."""
    published = {"D4A": -1.844, "D4C": -2.107, "D8A": -4.075, "D8C": -3.969}
    fig = proteome_figure(
        _proteome(published),
        "TEST",
        _reps(
            {
                "D4A": [-1.8, -1.9],
                "D4C": [-2.1, -2.1],
                # F13A1's real channels. rep5 is bit-identical across both of
                # its channels because both are the same floored constant.
                "D8A": [-2.633085, -2.642663, -2.923101, -2.923101],
                "D8C": [-2.510484, -2.552211, -2.923101, -2.923101],
            }
        ),
    )
    assert _bar(fig) == pytest.approx(published)


def test_bar_ignores_the_parquet_aggregate():
    """`proteome.log2fc` is log2 of the *arithmetic* mean of the donor ratios —
    biased upward, in no published artifact, and not what the bar draws."""
    published = {c: -1.0 for c in ORDER}
    fig = proteome_figure(
        _proteome(published, aggregate={c: 0.5 for c in ORDER}),
        "TEST",
        _reps({c: [-1.0, -1.0] for c in ORDER}),
    )
    assert _bar(fig) == pytest.approx(published)


def test_bar_falls_back_when_no_published_value():
    """A slice from a parquet built before this change has no
    `published_log2fc`; it must still draw the aggregate rather than crash."""
    df = _proteome({c: -1.0 for c in ORDER}, aggregate={c: 0.5 for c in ORDER}).drop(
        "published_log2fc"
    )
    fig = proteome_figure(df, "TEST", _reps({c: [-1.0, -1.0] for c in ORDER}))
    assert _bar(fig) == pytest.approx({c: 0.5 for c in ORDER})


def test_null_published_value_is_a_gap_not_a_zero():
    """AFAP1L2 D8C is dropped from the volcano (its D2 reference was below
    detection), so it has no published value. A null must not render as a bar
    of height zero, which reads as 'no change'."""
    fig = proteome_figure(
        _proteome(
            {"D4A": -0.5, "D4C": -0.4, "D8A": -0.3, "D8C": None},
            d2_below_detection={"D8C": True},
        ),
        "TEST",
        _reps({c: [-0.4, -0.4] for c in ORDER}),
    )
    assert _bar(fig)["D8C"] is None


def test_replicate_points_are_still_drawn_over_the_bar():
    """The bar not being their mean must not mean they were dropped."""
    fig = proteome_figure(
        _proteome({c: -4.0 for c in ORDER}),
        "TEST",
        _reps({c: [-2.6, -2.9] for c in ORDER}),
    )
    plotted = [y for t in fig.data if t.type != "bar" for y in (t.y or [])]
    assert sorted(set(plotted)) == pytest.approx([-2.9, -2.6])


def test_error_bars_are_the_replicate_sem():
    """No published SE exists for these bars, so the interval stays the SEM of
    the overlaid points — sd/sqrt(n) over channels, F13A1 D8A's 0.082."""
    fig = proteome_figure(
        _proteome({c: -4.0 for c in ORDER}),
        "TEST",
        _reps({c: [-2.633085, -2.642663, -2.923101, -2.923101] for c in ORDER}),
    )
    trace = next(t for t in fig.data if t.type == "bar")
    assert list(trace.error_y.array) == pytest.approx([0.0824] * 4, abs=5e-5)


def test_no_error_bar_below_two_replicates():
    """One channel has no defined sd; a null must not render as a zero-length
    interval implying perfect precision."""
    fig = proteome_figure(
        _proteome({c: -1.0 for c in ORDER}),
        "TEST",
        _reps({"D4A": [-1.0], "D4C": [-1.0, -1.2], "D8A": [-1.0], "D8C": [-1.0, -1.2]}),
    )
    trace = next(t for t in fig.data if t.type == "bar")
    assert [e is None for e in trace.error_y.array] == [True, False, True, False]


def test_hover_does_not_claim_a_replicate_mean():
    """The hover reports the bar's own value and a donor count. It must not
    regress to describing the bar as a mean of the plotted channels, which is
    what it no longer is."""
    fig = proteome_figure(
        _proteome({c: -1.0 for c in ORDER}),
        "TEST",
        _reps({c: [-1.0, -1.2] for c in ORDER}),
    )
    template = next(t for t in fig.data if t.type == "bar").hovertemplate
    assert "donor(s)" in template
    assert "mean" not in template.lower()


def test_censored_donor_raises_the_evidence_note():
    """The four detached bars are all censored cells; the figure must say so
    rather than leave the gap unexplained."""
    fig = proteome_figure(
        _proteome({"D4A": -1.844, "D4C": -2.107, "D8A": -4.075, "D8C": -3.969}),
        "TEST",
        _reps({c: [-2.6, -2.9] for c in ORDER}, censored={"D8A": True}),
    )
    note = _note(fig)
    assert "below detection in D8A" in note
    assert "bounds, not measurements" in note


def test_no_note_for_an_ordinary_protein():
    """The caveat has to stay silent on the ordinary case or it reads as noise."""
    fig = proteome_figure(
        _proteome({c: -1.0 for c in ORDER}),
        "TEST",
        _reps({c: [-1.0, -1.2] for c in ORDER}),
    )
    assert _note(fig) == ""


def test_missing_d2_reference_is_named_once():
    """A censored donor caused by a missing D2 reference is already explained by
    the first clause; naming the condition twice adds words and no information."""
    fig = proteome_figure(
        _proteome(
            {"D4A": -0.5, "D4C": -0.4, "D8A": -0.3, "D8C": None},
            d2_below_detection={"D8C": True},
        ),
        "TEST",
        _reps({c: [-0.4, -0.4] for c in ORDER}, censored={"D8C": True}),
    )
    note = _note(fig)
    assert "omitted from the volcano" in note
    assert "bounds, not measurements" not in note
