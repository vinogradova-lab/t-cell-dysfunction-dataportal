"""Which S2-1 block feeds which volcano comparison, and which points are withheld.

Two things here are easy to get backwards and produce a plausible-looking plot:

  * S2-1 labels its blocks ``"<denominator> vs. <numerator>"`` while the values
    carry numerator/denominator — so ``"D8A vs. D8C"`` is log2(D8C/D8A), and
    reading the label at face value would mirror the whole panel about x = 0;
    and
  * the sheet ships eight blocks where the portal plots five, so a looser match
    would silently pull in ``D8A vs. D4A`` / ``D8A vs. D4C`` / ``D8A vs. D2``.

The exclusion of points whose *reference* channel was below detection also has a
per-comparison reference now (D2 for the vs-D2 panels, D8A for D8C-vs-D8A), so
each panel's rule is pinned against a synthetic S2-1 frame rather than the real
1.5 MB workbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

import build_db  # noqa: E402


# every block S2-1 actually ships, label as written in the workbook, mapped to
# the log2FC we plant in it. The three the portal does not plot get distinctive
# values so a loose match shows up as a wrong number rather than a missing key.
_BLOCKS = {
    "D2 vs. D4A": 0.1,
    "D2 vs. D4C": 0.2,
    "D2 vs. D8A": 0.3,
    "D2 vs. D8C": 0.4,
    "D8A vs. D8C": 1.5,
    "D8A vs. D4A": -99.0,
    "D8A vs. D4C": -98.0,
    "D8A vs. D2": -97.0,
}

_CACHED = ("_detection_floors", "_s2_1_channel_pct", "_wp_median_pct",
           "_ref_below_detection")


def _s2_1_frame(census: dict[str, float] | None = None) -> pd.DataFrame:
    """One-protein S2-1 frame carrying both the census columns and all eight
    volcano blocks, with the workbook's real suffix shape."""
    row: dict[str, object] = {
        "uniprot": "U1", "protein": "TEST", "description": "test protein",
    }
    census = census or {c: 0.20 for c in build_db.FIVE_CONDITIONS}
    for rep in build_db.PROTEOME_REPS:
        for cond in build_db.FIVE_CONDITIONS:
            v = census.get(cond, np.nan) if rep == 1 else np.nan
            for sub in (1, 2):
                row[f"{cond}_rep{rep}_{sub}_processed_census-out"] = v
    for label, lfc in _BLOCKS.items():
        suffix = f"Exhaustion WP - {label} (6315 Proteins)"
        row[f"log2_FC_{suffix}"] = lfc
        row[f"p_value_{suffix}"] = 0.001
        row[f"-log10_pval_{suffix}"] = 3.0
        row[f"-log10_pval_adj_{suffix}"] = 2.0
        row[f"Regulation_{suffix}"] = "Significant Up"
    for flag in build_db.VOLCANO_FLAGS:
        row[flag] = False
    return pd.DataFrame([row])


@pytest.fixture
def stub(monkeypatch):
    """Patch the S2-1 loader + detection floors, clearing the lru_caches."""

    def _clear() -> None:
        for name in _CACHED:
            # these are plain lambdas mid-test, and this fixture tears down
            # before monkeypatch restores the cached originals
            clear = getattr(getattr(build_db, name), "cache_clear", None)
            if clear is not None:
                clear()

    def _apply(census: dict[str, float] | None = None) -> pd.DataFrame:
        _clear()
        monkeypatch.setattr(build_db, "load_s2_1", lambda: _s2_1_frame(census))
        monkeypatch.setattr(
            build_db,
            "_detection_floors",
            lambda: {rep: 0.05 for rep in build_db.PROTEOME_REPS},
        )
        return build_db.build_volcano()

    yield _apply
    _clear()


def _lfc(out: pd.DataFrame, comparison: str) -> float:
    return float(out.loc[out["comparison"] == comparison, "log2fc"].iloc[0])


def test_reversed_s2_1_label_maps_to_a_numerator_first_id(stub):
    """"D8A vs. D8C" carries log2(D8C/D8A), so it becomes D8C_vs_D8A unnegated."""
    out = stub()
    assert _lfc(out, "D8C_vs_D8A") == pytest.approx(1.5)
    assert "D8A_vs_D8C" not in set(out["comparison"].astype(str))


def test_vs_d2_blocks_are_unaffected_by_the_new_mapping(stub):
    out = stub()
    for cond, want in [("D4A", 0.1), ("D4C", 0.2), ("D8A", 0.3), ("D8C", 0.4)]:
        assert _lfc(out, cond) == pytest.approx(want)


def test_the_three_unplotted_blocks_are_ignored(stub):
    """S2-1 ships eight blocks; the portal plots five. The D8A-vs-D4A/D4C/D2
    blocks carry sentinel values, so a looser match shows up as a bad number.
    """
    out = stub()
    assert sorted(set(out["comparison"].astype(str))) == sorted(
        build_db.VOLCANO_COMPARISONS
    )
    assert not (out["log2fc"] < -90).any()


def test_suffix_lookup_covers_every_plotted_comparison():
    """A renamed block must fail loudly at build time, not drop a panel."""
    got = build_db._volcano_suffix_by_cond(_s2_1_frame())
    assert sorted(got) == sorted(build_db.VOLCANO_COMPARISONS)


def test_below_detection_d2_is_withheld_from_its_vs_d2_panel_only(stub):
    """A dead D2 channel inflates the vs-D2 fold changes; it says nothing about
    D8C-vs-D8A, which never divides by D2.
    """
    out = stub({"D2": 0.0, "D4A": 0.20, "D4C": 0.20, "D8A": 0.20, "D8C": 0.40})
    present = set(out["comparison"].astype(str))
    assert present == {"D8C_vs_D8A"}


def test_below_detection_d8a_is_withheld_from_the_chronic_vs_acute_panel(stub):
    """The AFAP1L2 / TNFRSF4 case: D8A dropped out, so log2(D8C/D8A) divides by
    a halved denominator. The vs-D2 panels are untouched.
    """
    out = stub({"D2": 0.20, "D4A": 0.20, "D4C": 0.20, "D8A": 0.0, "D8C": 0.40})
    present = set(out["comparison"].astype(str))
    assert "D8C_vs_D8A" not in present
    assert {"D4A", "D4C", "D8C"} <= present


def test_a_fully_measured_protein_keeps_every_panel(stub):
    out = stub()
    assert set(out["comparison"].astype(str)) == set(build_db.VOLCANO_COMPARISONS)
