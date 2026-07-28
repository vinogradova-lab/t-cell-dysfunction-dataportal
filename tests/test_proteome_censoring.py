"""Censoring rules for below-detection whole-proteome channels.

A census value of exactly 0 means "below detection in that TMT channel", not
"absent". Getting this wrong is what made AFAP1L2's D8C expression triangle
disappear and pushed F13A1's a full log2 unit off: the aggregate averaged
percent-of-control *before* the log2 conversion (so 0 and inf survived into it)
while the per-replicate table converted first (so they were dropped), leaving
the two tables disagreeing.

These tests drive ``_s2_1_replicate_pct`` through a synthetic S2-1 frame so each
row of the rule table is pinned independently of the real workbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

import build_db  # noqa: E402


FLOOR = 0.05


def _s2_1_frame(values: dict[str, float | tuple[float, float]]) -> pd.DataFrame:
    """One-protein S2-1 frame. ``values`` maps condition -> census value for
    rep1, either a scalar (both technical channels alike) or a ``(ch1, ch2)``
    pair. Every other replicate is left NaN (absent from that run).
    """
    row: dict[str, object] = {"uniprot": "U1", "protein": "TEST"}
    for rep in build_db.PROTEOME_REPS:
        for cond in build_db.FIVE_CONDITIONS:
            v = values.get(cond, np.nan) if rep == 1 else np.nan
            chans = v if isinstance(v, tuple) else (v, v)
            for sub, cv in zip((1, 2), chans):
                row[f"{cond}_rep{rep}_{sub}_processed_census-out"] = cv
    return pd.DataFrame([row])


# every cached helper that reads the workbook; stale entries would leak the
# real S2-1 between tests
_CACHED = ("_detection_floors", "_s2_1_channel_pct", "_wp_median_pct",
           "_ref_below_detection")


@pytest.fixture
def stub(monkeypatch):
    """Patch the workbook loader + detection floors, clearing the lru_caches."""

    def _clear() -> None:
        for name in _CACHED:
            # _detection_floors is monkeypatched to a plain lambda during a
            # test, and this fixture tears down before monkeypatch undoes that
            clear = getattr(getattr(build_db, name), "cache_clear", None)
            if clear is not None:
                clear()

    def _apply(values: dict[str, float | tuple[float, float]]) -> pd.DataFrame:
        _clear()
        monkeypatch.setattr(build_db, "load_s2_1", lambda: _s2_1_frame(values))
        monkeypatch.setattr(
            build_db,
            "_detection_floors",
            lambda: {rep: FLOOR for rep in build_db.PROTEOME_REPS},
        )
        return build_db._s2_1_replicate_pct()

    yield _apply
    _clear()


def _cell(out: pd.DataFrame, cond: str) -> pd.DataFrame:
    return out[(out["condition"] == cond) & (out["rep"] == "rep1")]


def test_normal_measurement_is_a_plain_ratio(stub):
    out = stub({"D2": 0.20, "D8C": 0.10})
    row = _cell(out, "D8C").iloc[0]
    assert row["percent_control"] == pytest.approx(50.0)
    assert not row["censored"]


def test_zero_condition_is_floored_as_an_upper_bound_on_loss(stub):
    """Signal lost by day 8 (F13A1 / HBB / TNFRSF4)."""
    out = stub({"D2": 0.20, "D8C": 0.0})
    row = _cell(out, "D8C").iloc[0]
    assert row["percent_control"] == pytest.approx(100 * FLOOR / 0.20)
    assert row["censored"]


def test_zero_d2_is_floored_as_a_lower_bound_on_increase(stub):
    """AFAP1L2: the protein is seen only at D8C, so D2 has no reference.

    Previously this divided by zero, yielding inf -> NaN -> no triangle.
    """
    out = stub({"D2": 0.0, "D8C": 0.40})
    row = _cell(out, "D8C").iloc[0]
    assert row["percent_control"] == pytest.approx(100 * 0.40 / FLOOR)
    assert row["censored"]
    assert np.isfinite(row["percent_control"])


def test_both_channels_zero_is_dropped(stub):
    """The trap: flooring both sides would give floor/floor = 100%, injecting a
    spurious "no change" into conditions where the protein was never seen.
    """
    out = stub({"D2": 0.0, "D8A": 0.0, "D8C": 0.40})
    assert _cell(out, "D8A").empty
    # the informative condition in the same replicate still survives
    assert not _cell(out, "D8C").empty


def test_absent_replicate_is_dropped(stub):
    out = stub({"D2": 0.20, "D8C": 0.10})
    assert _cell(out, "D4A").empty
    assert out["rep"].unique().tolist() == ["rep1"]


def test_no_zero_or_infinite_percent_control_survives(stub):
    out = stub({"D2": 0.0, "D4A": 0.0, "D4C": 0.10, "D8A": 0.0, "D8C": 0.40})
    pct = out["percent_control"]
    assert np.isfinite(pct).all()
    assert (pct > 0).all()


def test_aggregate_and_replicates_agree_on_which_replicates_count(stub):
    """The original defect: build_proteome kept replicates that
    build_proteome_replicates dropped. Both now read the same censored values.
    """
    stub({"D2": 0.20, "D4C": 0.10, "D8C": 0.0})
    agg = build_db.build_proteome()
    reps = build_db.build_proteome_replicates()

    # the overlay is per channel, so compare donors to donors
    donors = reps.groupby(["uniprot", "condition"], observed=True)["bio_rep"].nunique()
    for _, r in agg.iterrows():
        assert r["n_reps"] == donors.get((r["uniprot"], r["condition"]), 0)
        assert np.isfinite(r["log2fc"])

    # the flag drives the censoring internally but is not a published column
    assert "censored" not in agg.columns
    assert "censored" not in reps.columns


def test_overlay_splits_each_donor_into_two_channels(stub):
    """The bar overlay plots technical channels, not donor means, so a donor
    contributes both of its measurements and the visible spread is real.
    """
    stub({"D2": 0.20, "D8C": (0.10, 0.30)})
    reps = build_db.build_proteome_replicates()

    d8c = reps[reps["condition"] == "D8C"]
    assert sorted(d8c["rep"]) == ["rep1.1", "rep1.2"]
    assert d8c["bio_rep"].unique().tolist() == ["rep1"]
    # each channel divided by the donor's D2 mean (0.20), not by its own channel
    assert sorted(d8c["percent_control"]) == pytest.approx([50.0, 150.0])


def test_channel_pair_averages_back_to_the_donor_value(stub):
    """Both channels share a denominator, so averaging the pair recovers the
    donor percent exactly. etl/prerender.py relies on this to rebuild the
    download's per-donor columns from the channel-level parquet.
    """
    stub({"D2": 0.20, "D8C": (0.10, 0.30)})
    reps = build_db.build_proteome_replicates()
    donor = build_db._s2_1_replicate_pct()

    got = reps[reps["condition"] == "D8C"]["percent_control"].mean()
    want = donor[donor["condition"] == "D8C"]["percent_control"].iloc[0]
    assert got == pytest.approx(want)


def test_aggregate_counts_donors_not_channels(stub):
    """n_reps is the confidence signal, so it must not double when the overlay
    switches to channels.
    """
    stub({"D2": 0.20, "D8C": (0.10, 0.30)})
    agg = build_db.build_proteome()
    assert agg[agg["condition"] == "D8C"]["n_reps"].iloc[0] == 1


def test_missing_d2_reference_is_flagged_for_volcano_exclusion(stub):
    """A zero D2 makes the published fold change divide by a halved denominator.
    The rule keys on D2 detection, not on a gene name.
    """
    stub({"D2": 0.0, "D8C": 0.40})
    flagged = build_db._d2_below_detection()
    assert set(zip(flagged["uniprot"], flagged["condition"])) == {("U1", "D8C")}
    assert build_db.build_proteome()["d2_below_detection"].any()


def test_measured_d2_is_not_flagged(stub):
    stub({"D2": 0.20, "D8A": 0.0, "D8C": 0.40})
    assert build_db._d2_below_detection().empty
    assert not build_db.build_proteome()["d2_below_detection"].any()


# The volcano's D8C-vs-D8A panel divides by D8A, so the same inflation applies
# with a different denominator. Real hits: AFAP1L2 (+4.68) and TNFRSF4 (+2.87),
# both from a dead D8A channel in rep5.
def test_the_reference_condition_is_parameterized(stub):
    """A dead D8A channel inflates every ratio taken against D8A, and none of
    the ratios taken against a measured D2. The two must not be conflated.
    """
    stub({"D2": 0.20, "D8A": 0.0, "D8C": 0.40})
    flagged = build_db._ref_below_detection("D8A")
    assert set(zip(flagged["uniprot"], flagged["condition"])) == {
        ("U1", "D2"), ("U1", "D8C")
    }
    assert build_db._ref_below_detection("D2").empty


def test_reference_and_numerator_both_zero_is_not_a_hit(stub):
    """Both channels empty carries no information either way, so the replicate
    is already excluded upstream rather than treated as an inflated ratio.
    """
    stub({"D2": 0.20, "D8A": 0.0, "D8C": 0.0})
    assert "D8C" not in build_db._ref_below_detection("D8A")["condition"].tolist()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "source" / "data_s2.xlsx").exists(),
    reason="needs source/ staged via scripts/sync_source.py",
)
def test_wp_median_reproduces_the_upstream_reactivity_column():
    """The expression triangles use a median over technical channels, not the
    mean over biological replicates that build_proteome() reports. Pinning the
    exact statistic keeps the one patched cell (AFAP1L2 D8C) comparable to its
    neighbours instead of silently mixing conventions.
    """
    root = Path(__file__).resolve().parents[1]
    src = pd.read_csv(root / "source" / "reactivity_5cond.csv", index_col=0)
    upstream = src[["uniprot", "condition", "whole_proteome"]].drop_duplicates(
        ["uniprot", "condition"]
    )
    upstream["expected"] = np.log2(
        pd.to_numeric(upstream["whole_proteome"], errors="coerce") / 100.0
    )
    upstream = upstream[np.isfinite(upstream["expected"])]

    got = build_db._wp_median_pct()
    m = upstream.merge(got, on=["uniprot", "condition"], how="inner")
    assert len(m) == len(upstream)

    # Uncensored cells must reproduce the published value bit-for-bit — that is
    # what proves we compute the same statistic upstream does.
    plain = m[~m["censored"]]
    assert len(plain) > 18_000
    assert np.allclose(plain["wp_log2fc"], plain["expected"], rtol=1e-6)

    # The only departures are cells with a below-detection channel, where
    # upstream averaged a raw 0 and we substitute the detection limit.
    assert not np.isclose(
        m[m["censored"]]["wp_log2fc"], m[m["censored"]]["expected"], rtol=1e-6
    ).any()


def test_detection_floor_uses_percentile_not_minimum():
    """Guards the floor choice: rep5's minimum positive census value is ~20x
    lower than every other replicate's, which as a floor would turn a
    below-detection channel into a ~-8 log2FC outlier that dominates the mean.
    """
    assert build_db.DETECTION_QUANTILE == 0.01
