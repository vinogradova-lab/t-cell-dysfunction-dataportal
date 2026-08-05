"""Scale of the per-replicate RNA log2FCs behind the bar chart's points.

These numbers used to come from the analysis repo's ``normalized_counts.txt``,
whose differences are *not* log2 ratios: against S1-1's DESeq2
``log2FoldChange`` they came out compressed by a near-constant ~1.49x, so the
bar chart understated every RNA fold change against an axis labelled log2FC
(GBE1 at D4A drew -0.86 where the volcano plotted -1.30). They are now derived
here from raw counts with DESeq2 median-of-ratios size factors.

The tests drive ``build_rna_replicates`` through a synthetic featureCounts frame
so the normalization is pinned independently of the 47 MB real matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

import build_db  # noqa: E402


# featureCounts annotation columns, which precede the per-bam count columns and
# must be dropped before any arithmetic
_ANNOTATION = ["Chr", "Start", "End", "Strand", "Length"]

# (column name, condition, rep) for the 15 samples, in the file's own order
_SAMPLES = [
    (f"sorted_bam_files/Sample_{tag}_IGO_13524_{i}_sorted.bam", cond, f"rep{rep}")
    for i, (tag, cond, rep) in enumerate(
        [
            ("D2-1", "D2", 1), ("D2-2", "D2", 2), ("D2-3", "D2", 3),
            ("D4-Ac-1", "D4A", 1), ("D4-Ac-2", "D4A", 2), ("D4-Ac-3", "D4A", 3),
            ("D4-Chr-1", "D4C", 1), ("D4-Chr-2", "D4C", 2), ("D4-Chr-3", "D4C", 3),
            ("D8-Ac-1", "D8A", 1), ("D8-Ac-2", "D8A", 2), ("D8-Ac-3", "D8A", 3),
            ("D8-Chr-1", "D8C", 1), ("D8-Chr-2", "D8C", 2), ("D8-Chr-3", "D8C", 3),
        ],
        start=1,
    )
]
_SAMPLE_COLS = [c for c, _, _ in _SAMPLES]


def _counts_frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    """featureCounts-shaped frame: ``rows`` maps gene symbol -> 15 raw counts,
    in ``_SAMPLES`` order. Annotation columns are filled with plausible junk so
    a build that forgets to drop them fails loudly rather than silently."""
    frame = pd.DataFrame(rows, index=_SAMPLE_COLS).T
    frame.index.name = "Geneid"
    for col in _ANNOTATION:
        frame.insert(0, col, "chr1;chr1" if col == "Chr" else 1000)
    return frame


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Point the loader at a synthetic counts file and let every gene through
    the protein-coding filter (which otherwise reads the 12 MB NCBI table).

    ``unquantified_rna_symbols`` is stubbed empty for the same reason: it reads
    S1-1 out of ``SOURCE``, which these tests have redirected to a tmp dir
    holding nothing but the counts file. The normalization under test here is
    independent of that filter — see test_rna_unquantified.py for the filter.
    """

    def _install(rows: dict[str, list[float]]) -> None:
        path = tmp_path / "rna_counts_raw.txt"
        _counts_frame(rows).to_csv(path, sep="\t")
        monkeypatch.setattr(build_db, "SOURCE", tmp_path)
        monkeypatch.setattr(
            build_db, "protein_coding_symbols", lambda: frozenset(rows)
        )
        monkeypatch.setattr(
            build_db, "unquantified_rna_symbols", lambda: frozenset()
        )

    return _install


def _mean_by_condition(out: pd.DataFrame, symbol: str) -> dict[str, float]:
    sub = out[out["symbol"] == symbol]
    return sub.groupby("condition", observed=True)["log2fc"].mean().to_dict()


def _background(n: int = 50) -> dict[str, list[float]]:
    """Genes held flat across all 15 samples.

    Median-of-ratios assumes most genes are unchanged, and reads any shift
    shared by *every* gene as a library-size difference to be normalized away.
    So a fixture testing a gene's fold change needs a flat background for the
    size factors to land on 1 — otherwise the effect under test is absorbed.
    Varying levels between genes keeps the frame from being degenerate.
    """
    return {f"BG{i}": [1000.0 + 10 * i] * 15 for i in range(n)}


# --------------------------------------------------------------------------- #
# column parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("col, expected", [(c, (k, r)) for c, k, r in _SAMPLES])
def test_sample_columns_are_parsed_from_bam_paths(col, expected):
    """The raw matrix names columns after bam paths and hyphenates where the
    old normalized matrix used dots — both the prefix and the separator would
    have defeated the previous anchored, dot-only regex."""
    assert build_db._rna_sample_meta(col) == expected


@pytest.mark.parametrize("col", _ANNOTATION)
def test_annotation_columns_are_not_samples(col):
    assert build_db._rna_sample_meta(col) is None


def test_dotted_sample_columns_still_parse():
    """The older normalized matrix's naming still resolves, so the regex change
    is a widening rather than a swap."""
    assert build_db._rna_sample_meta(
        "Sample_D8.Chr.2_IGO_13524_14"
    ) == ("D8C", "rep2")


# --------------------------------------------------------------------------- #
# size factors
# --------------------------------------------------------------------------- #
def test_size_factors_are_median_of_ratios():
    """A sample sequenced twice as deep gets twice the factor, so dividing by it
    puts the two libraries back on one scale."""
    counts = pd.DataFrame(
        {"a": [100.0, 200.0, 400.0], "b": [200.0, 400.0, 800.0]}
    )
    sf = build_db._size_factors(counts)
    assert sf["b"] / sf["a"] == pytest.approx(2.0)


def test_size_factor_reference_skips_genes_with_a_zero():
    """A gene missing from one library says nothing about library size, and its
    zero would send the geometric-mean reference to zero. DESeq2 drops it; so do
    we — the factors must match a frame that never contained the gene."""
    kept = {"a": [100.0, 200.0, 400.0], "b": [200.0, 400.0, 800.0]}
    with_zero = {k: v + [0.0 if k == "a" else 999999.0] for k, v in kept.items()}
    assert build_db._size_factors(
        pd.DataFrame(with_zero)
    ).to_dict() == pytest.approx(build_db._size_factors(pd.DataFrame(kept)).to_dict())


# --------------------------------------------------------------------------- #
# log2FC derivation
# --------------------------------------------------------------------------- #
def test_log2fc_is_relative_to_the_d2_mean(stub):
    """A gene held at 4x its D2 level lands at +2, on every condition."""
    stub(_background() | {"UP": [1000.0] * 3 + [4000.0] * 12})
    means = _mean_by_condition(build_db.build_rna_replicates(), "UP")
    assert set(means) == {"D4A", "D4C", "D8A", "D8C"}
    for cond, value in means.items():
        assert value == pytest.approx(2.0, abs=0.01), cond


def test_d2_is_not_emitted(stub):
    """D2 is the reference (log2FC = 0 by construction); the figure draws four
    bars, not five."""
    stub({"FLAT": [1000.0] * 15})
    out = build_db.build_rna_replicates()
    assert sorted(out["condition"].unique()) == ["D4A", "D4C", "D8A", "D8C"]
    assert list(out["condition"].cat.categories) == ["D4A", "D4C", "D8A", "D8C"]


def test_library_depth_is_normalized_away(stub):
    """A gene that holds a constant *share* of a library twice as deep must read
    as unchanged, not as +1."""
    deep = [2000.0] * 3 + [1000.0] * 12       # D2 sequenced twice as deep
    stub({f"G{i}": deep for i in range(50)})
    means = _mean_by_condition(build_db.build_rna_replicates(), "G0")
    for cond, value in means.items():
        assert value == pytest.approx(0.0, abs=0.01), cond


def test_replicate_mean_recovers_the_true_log2_ratio(stub):
    """The regression guard for the bug this file exists for.

    GBE1's raw counts imply D4A/D2 = -1.30 and DESeq2 reports -1.303; the old
    VST-matrix derivation returned -0.855. Anything that reintroduces a
    compressed scale fails here.
    """
    d2, d4a = 3600.0, 3600.0 * 2 ** -1.3
    stub(_background() | {"GBE1ISH": [d2] * 3 + [d4a] * 3 + [d2] * 9})
    means = _mean_by_condition(build_db.build_rna_replicates(), "GBE1ISH")
    assert means["D4A"] == pytest.approx(-1.3, abs=0.01)
    # the value the compressed VST matrix used to return for this gene
    assert means["D4A"] != pytest.approx(-0.855, abs=0.05)


def test_zero_counts_survive_as_finite_values(stub):
    """log2(0) is -inf, which parquet and Plotly both choke on; the +0.5
    pseudocount keeps a genuinely-absent replicate on the chart."""
    stub(_background() | {"LOST": [1000.0] * 3 + [0.0] * 3 + [1000.0] * 9})
    out = build_db.build_rna_replicates()
    d4a = out[(out["symbol"] == "LOST") & (out["condition"] == "D4A")]["log2fc"]
    assert len(d4a) == 3
    assert np.isfinite(d4a).all()
    assert (d4a < -10).all()
