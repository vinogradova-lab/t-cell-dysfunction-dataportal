"""Column handling for the polar-metabolomics download (Data S3-1).

Two things about S3-1 will silently produce a wrong CSV if they regress:

  * every differential block appears **twice** in the sheet, so a naive read
    doubles the statistics columns (pandas renames the copy with a ``.1``
    suffix, which then leaks into the download's header); and
  * the blocks are labelled ``<denominator> vs. <numerator>`` while carrying the
    numerator-over-denominator sign — so "D2 vs. D8C" is log2(D8C/D2). Reading
    the label at face value would flip the sign of every fold change.

These tests drive the builder through a synthetic S3-1 frame so both are pinned
without the 1.3 MB workbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

import build_db  # noqa: E402


# one channel-ratio and one raw-intensity sample per condition, enough to
# exercise the renaming without writing out all 120 sample columns
_DONOR = "LSP75"


def _s3_1_frame(*, duplicate_blocks: bool = True, d8c_log2fc: float = 1.5):
    """One-compound S3-1 frame with the sheet's real column naming."""
    row: dict[str, object] = {
        col: f"<{col}>" for col in build_db.METABOLITE_ANNOTATION_COLS
    }
    row["Compound"] = "Citric acid"
    for i, cond in enumerate(build_db.FIVE_CONDITIONS, start=1):
        for suffix in build_db.METABOLITE_VALUE_SUFFIXES:
            row[f"{cond}_{_DONOR}_S{i:02d}_{suffix}"] = float(i)

    def _block(den: str, num: str, log2fc: float) -> dict[str, object]:
        tag = f"metabolomics - {den} vs. {num} (174 Metabolites)"
        return {
            f"p_value_{tag}": 0.01,
            f"log2_FC_{tag}": log2fc,
            f"-log10_pval_{tag}": 2.0,
            f"-log10_pval_adj_{tag}": 1.5,
            f"Regulation_{tag}": "Significant Up",
        }

    blocks = {}
    for den, num, lfc in [
        ("D2", "D4A", 0.1),
        ("D2", "D4C", 0.2),
        ("D2", "D8A", 0.3),
        ("D2", "D8C", d8c_log2fc),
        ("D4A", "D4C", 0.4),
        ("D8A", "D8C", 0.5),
    ]:
        blocks.update(_block(den, num, lfc))
    row.update(blocks)

    df = pd.DataFrame([row])
    if duplicate_blocks:
        # the workbook repeats the whole differential section verbatim; pandas
        # dedups the header, so reproduce that rename here
        dup = pd.DataFrame([{f"{k}.1": v for k, v in blocks.items()}])
        df = pd.concat([df, dup], axis=1)
    return df


@pytest.fixture
def stub(monkeypatch):
    """Patch the S3-1 loader, clearing its lru_cache around the test."""

    def _clear() -> None:
        # during a test load_s3_1 is a plain lambda, and this fixture tears down
        # before monkeypatch restores the cached original
        clear = getattr(build_db.load_s3_1, "cache_clear", None)
        if clear is not None:
            clear()

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        _clear()
        monkeypatch.setattr(build_db, "load_s3_1", lambda: df)
        return build_db.build_metabolomics()

    yield _apply
    _clear()


def test_duplicate_stat_blocks_are_dropped(stub):
    out = stub(_s3_1_frame(duplicate_blocks=True))
    assert [c for c in out.columns if c.endswith(".1")] == []
    # one column per (comparison x statistic), not two
    assert sum(c.endswith("_log2fc") for c in out.columns) == len(
        build_db.METABOLITE_COMPARISONS
    )


def test_duplicate_block_that_disagrees_is_an_error(stub):
    """A future revision where the copies diverge must not be halved silently."""
    df = _s3_1_frame(duplicate_blocks=True)
    df["log2_FC_metabolomics - D2 vs. D8C (174 Metabolites).1"] = -99.0
    with pytest.raises(ValueError, match="duplicate block disagrees"):
        stub(df)


def test_comparison_columns_are_named_numerator_first(stub):
    """"D2 vs. D8C" carries log2(D8C/D2), so it becomes D8C_vs_D2."""
    out = stub(_s3_1_frame(d8c_log2fc=1.5))
    assert out.loc[0, "D8C_vs_D2_log2fc"] == pytest.approx(1.5)
    assert "D2_vs_D8C_log2fc" not in out.columns
    # ...including the comparisons that are not against D2
    assert out.loc[0, "D8C_vs_D8A_log2fc"] == pytest.approx(0.5)
    assert out.loc[0, "D4C_vs_D4A_log2fc"] == pytest.approx(0.4)


def test_sample_columns_are_renamed_and_kept(stub):
    out = stub(_s3_1_frame())
    assert out.loc[0, "D2_LSP75_S01_channel_ratio"] == pytest.approx(1.0)
    assert out.loc[0, "D8C_LSP75_S05_raw_intensity"] == pytest.approx(5.0)
    # the long workbook suffixes are gone
    assert not any("QRILC" in c or "raw-signal" in c for c in out.columns)


def test_annotation_columns_are_snake_cased(stub):
    out = stub(_s3_1_frame())
    assert list(out.columns[:9]) == list(
        build_db.METABOLITE_ANNOTATION_COLS.values()
    )
    assert out.loc[0, "compound"] == "Citric acid"


def test_missing_annotation_column_is_an_error(stub):
    df = _s3_1_frame().drop(columns=["HMDB"])
    with pytest.raises(KeyError, match="HMDB"):
        stub(df)
