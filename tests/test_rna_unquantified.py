"""Genes the RNA-seq covered but never quantified, and where they go.

DESeq2's independent filtering removes low-count genes from the multiplicity
correction, leaving ``padj`` null while still reporting a ``log2FoldChange``.
That fold change is unconstrained — in the real data those genes carry a median
lfcSE of 4.73, a ±27-fold interval, and a third of their rows are a log2FC of
exactly 0 at p exactly 1 — so drawing it as a bar states an effect the data
cannot support. ``build_rna`` drops them dataset-wide.

The rule is pinned here because it is load-bearing in four places at once: the
bar chart, the volcano, the ``rna.csv`` download, and the gene registry, where a
dropped gene with no other modality stops being a page at all. It also has to
stay all-or-nothing per gene — a per-row drop would leave half-populated bar
charts — and it must key on DESeq2's verdict rather than a baseMean threshold,
since the quantified and unquantified populations overlap in baseMean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

import build_db  # noqa: E402


def _s1_1_frame(rows: list[dict]) -> pd.DataFrame:
    """S1-1 in the wide shape the sheet ships it: one block of
    ``baseMean_/log2FoldChange_/lfcSE_/padj_<cond>_vs_D2`` columns per
    comparison. Each row maps a symbol to a per-condition padj dict; anything
    unspecified gets a plausible quantified value.
    """
    out: dict[str, list] = {"gene_name": [r["symbol"] for r in rows]}
    for cond in build_db.FOUR_COMPARISONS:
        out[f"log2FoldChange_{cond}_vs_D2"] = [
            r.get("log2fc", 1.5) for r in rows
        ]
        out[f"lfcSE_{cond}_vs_D2"] = [r.get("lfc_se", 0.1) for r in rows]
        out[f"padj_{cond}_vs_D2"] = [r["padj"].get(cond, 0.01) for r in rows]
        out[f"baseMean_{cond}_vs_D2"] = [
            r.get("base_mean", 1000.0) for r in rows
        ]
    return pd.DataFrame(out)


@pytest.fixture
def stub(monkeypatch):
    """Drive the RNA builders off a synthetic S1-1.

    ``unquantified_rna_symbols`` is lru_cached (main() consults it from three
    builders and the sheet read is expensive), so the cache is cleared on both
    sides of the test — a stale entry would leak the previous case's symbols
    into this one, and a real entry would leak out.
    """

    def _install(rows: list[dict]) -> None:
        build_db.unquantified_rna_symbols.cache_clear()
        monkeypatch.setattr(build_db, "load_s1_1", lambda: _s1_1_frame(rows))
        monkeypatch.setattr(
            build_db,
            "protein_coding_symbols",
            lambda: frozenset(r["symbol"] for r in rows),
        )

    yield _install
    build_db.unquantified_rna_symbols.cache_clear()


_NULL = {c: np.nan for c in build_db.FOUR_COMPARISONS}


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_gene_null_in_every_comparison_is_unquantified(stub):
    """The real case: DESeq2 filtered the gene out of the fit entirely."""
    stub([{"symbol": "KEEP", "padj": {}}, {"symbol": "DROP", "padj": _NULL}])
    assert build_db.unquantified_rna_symbols() == frozenset({"DROP"})


def test_gene_with_a_padj_anywhere_is_kept(stub):
    """All-or-nothing, not any-null.

    No gene in the real data is null in only some comparisons — the four share
    one fit — but the distinction decides whether a surviving gene can reach the
    bar chart with a hole in it, so it is pinned rather than left to chance.
    """
    stub([{"symbol": "PARTIAL", "padj": {**_NULL, "D8C": 0.01}}])
    assert build_db.unquantified_rna_symbols() == frozenset()
    out = build_db.build_rna()
    assert set(out["symbol"]) == {"PARTIAL"}
    assert len(out) == len(build_db.FOUR_COMPARISONS)


def test_low_base_mean_alone_does_not_drop_a_gene(stub):
    """The filter is DESeq2's verdict, not a baseMean threshold.

    The two populations overlap — in the real data the unquantified genes reach
    baseMean 19 while the quantified ones start at 0.89 — so a cutoff would both
    keep noise and discard measured genes.
    """
    stub([{"symbol": "LOW", "padj": {}, "base_mean": 0.2}])
    assert build_db.unquantified_rna_symbols() == frozenset()
    assert "LOW" in set(build_db.build_rna()["symbol"])


# --------------------------------------------------------------------------- #
# what the rule feeds
# --------------------------------------------------------------------------- #
def test_build_rna_drops_every_row_of_an_unquantified_gene(stub):
    """Per gene, not per row: no half-populated bar chart."""
    stub([{"symbol": "KEEP", "padj": {}}, {"symbol": "DROP", "padj": _NULL}])
    out = build_db.build_rna()
    assert set(out["symbol"]) == {"KEEP"}
    assert out["padj"].notna().all()


def test_kept_and_unquantified_partition_the_input(stub):
    """Between them the two account for every gene S1-1 reported.

    The registry's ``rna_unquantified`` column is the complement of what reaches
    ``rna.parquet``, and the portal reads it to tell "not quantified" apart from
    "not in the study". A gene falling through both would silently lose its
    explanation and render as a missing card again.
    """
    stub(
        [
            {"symbol": "KEEP", "padj": {}},
            {"symbol": "DROP_A", "padj": _NULL},
            {"symbol": "DROP_B", "padj": _NULL},
        ]
    )
    kept = set(build_db.build_rna()["symbol"])
    dropped = set(build_db.unquantified_rna_symbols())
    assert kept == {"KEEP"}
    assert dropped == {"DROP_A", "DROP_B"}
    assert not (kept & dropped)
    assert kept | dropped == {"KEEP", "DROP_A", "DROP_B"}


def test_volcano_never_sees_an_unquantified_gene(stub, monkeypatch):
    """build_rna_volcano's own dropna is now a guard for the S2-2 block only;
    the vs-D2 comparisons arrive already filtered."""
    stub([{"symbol": "KEEP", "padj": {}}, {"symbol": "DROP", "padj": _NULL}])
    monkeypatch.setattr(
        build_db,
        "build_proteome",
        lambda: pd.DataFrame({"symbol": ["KEEP", "DROP"]}),
    )
    # empty, but column-shaped: _rna_d8c_vs_d8a selects out of it by name
    monkeypatch.setattr(
        build_db,
        "load_s2_2",
        lambda: pd.DataFrame(
            columns=[
                "protein",
                "D8C vs D8A RNA_log2FoldChange",
                "D8C vs D8A RNA_pvalue",
                "D8C vs D8A RNA_padj",
            ]
        ),
    )
    monkeypatch.setattr(
        build_db,
        "load_s2_1",
        lambda: pd.DataFrame(
            {"protein": ["KEEP", "DROP"], "description": ["k", "d"]}
        ),
    )
    out = build_db.build_rna_volcano()
    assert set(out["symbol"]) == {"KEEP"}
