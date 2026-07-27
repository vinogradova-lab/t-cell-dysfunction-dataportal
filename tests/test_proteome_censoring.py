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


def _s2_1_frame(values: dict[str, float]) -> pd.DataFrame:
    """One-protein S2-1 frame. ``values`` maps condition -> census value for
    rep1; both technical sub-measurements get the same value, so their mean is
    that value. Every other replicate is left NaN (absent from that run).
    """
    row: dict[str, object] = {"uniprot": "U1", "protein": "TEST"}
    for rep in build_db.PROTEOME_REPS:
        for cond in build_db.FIVE_CONDITIONS:
            v = values.get(cond, np.nan) if rep == 1 else np.nan
            for sub in (1, 2):
                row[f"{cond}_rep{rep}_{sub}_processed_census-out"] = v
    return pd.DataFrame([row])


@pytest.fixture
def stub(monkeypatch):
    """Patch the workbook loader + detection floors, clearing the lru_caches."""

    def _apply(values: dict[str, float]) -> pd.DataFrame:
        monkeypatch.setattr(build_db, "load_s2_1", lambda: _s2_1_frame(values))
        monkeypatch.setattr(
            build_db,
            "_detection_floors",
            lambda: {rep: FLOOR for rep in build_db.PROTEOME_REPS},
        )
        return build_db._s2_1_replicate_pct()

    return _apply


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

    counts = reps.groupby(["uniprot", "condition"], observed=True).size()
    for _, r in agg.iterrows():
        assert r["n_reps"] == counts.get((r["uniprot"], r["condition"]), 0)
        assert np.isfinite(r["log2fc"])

    d8c = agg[agg["condition"] == "D8C"].iloc[0]
    assert d8c["censored"]
    assert not agg[agg["condition"] == "D4C"].iloc[0]["censored"]


def test_detection_floor_uses_percentile_not_minimum():
    """Guards the floor choice: rep5's minimum positive census value is ~20x
    lower than every other replicate's, which as a floor would turn a
    below-detection channel into a ~-8 log2FC outlier that dominates the mean.
    """
    assert build_db.DETECTION_QUANTILE == 0.01
