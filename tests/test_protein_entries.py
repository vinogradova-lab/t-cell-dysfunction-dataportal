"""One portal entry per protein, not per gene symbol.

A gene symbol is not a unique protein id. Every measurement table is keyed on
UniProt accession, and five symbols were measured under more than one:

    TMPO    P42166 / P42167   splice isoforms (LAP2alpha, LAP2beta/gamma)
    MOCS2   O96007 / O96033   bicistronic; two distinct subunits
    MIEF1   L0R8F8 / Q9NQG6   altORF -- different reading frames, no shared
                              sequence (AltMIEF1 in the proteome, MiD51 in the
                              cysteine assay)
    CDKN2A  P42771 / Q8N726   p16INK4a vs p14ARF, likewise
    POLR1D  P0DPB6 / P0DPB5   isoforms

Keyed on symbol, that broke in two ways. Where both accessions sat in one table,
the figure builders pooled them: TMPO's D8C bar drew mean(0.772, 0.332) = 0.531,
a value belonging to neither protein. Where each table held a *different*
accession, nothing was averaged but the page put one protein's abundance beside
another's cysteines -- CDKN2A rendered p16INK4a's bars next to p14ARF's dots.

These assertions run against the committed parquet rather than a fixture: the
defect was in how real accessions distribute across real tables, and no
synthetic frame in this suite has ever had a duplicate symbol (every synthetic
S2-1 protein is given a unique accession), which is exactly why the whole class
was invisible to it.

The single-accession assertions matter as much as the split ones -- 17,543 of
17,553 entries take that path and must behave as they always did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "portal"))
sys.path.insert(0, str(ROOT / "etl"))

from figures import BUILDERS, volcano_figure  # noqa: E402
from store import MODALITIES, Store  # noqa: E402

# symbol -> the accessions it was measured under, and which assays saw each
SPLIT = {
    "TMPO": ["P42166", "P42167"],
    "MOCS2": ["O96007", "O96033"],
    "MIEF1": ["L0R8F8", "Q9NQG6"],
    "CDKN2A": ["P42771", "Q8N726"],
    "POLR1D": ["P0DPB5", "P0DPB6"],
}


@pytest.fixture(scope="module")
def store() -> Store:
    return Store()


# --------------------------------------------------------------------------- #
# the entry universe
# --------------------------------------------------------------------------- #
def test_split_symbols_get_one_entry_per_accession(store):
    by_symbol: dict[str, list[dict]] = {}
    for e in store.entries():
        by_symbol.setdefault(e["symbol"], []).append(e)
    multi = {s: v for s, v in by_symbol.items() if len(v) > 1}
    assert set(multi) == set(SPLIT), "the set of split symbols changed"
    for symbol, accessions in SPLIT.items():
        assert sorted(e["uniprot"] for e in multi[symbol]) == sorted(accessions)


def test_split_entries_are_identified_by_accession(store):
    a, b = (store.entry(f"TMPO.{u}") for u in ("P42166", "P42167"))
    assert (a["id"], a["label"]) == ("TMPO.P42166", "TMPO (P42166)")
    assert (b["id"], b["label"]) == ("TMPO.P42167", "TMPO (P42167)")
    # each carries its own protein name, not the symbol's one-size-fits-all
    assert a["description"] != b["description"]
    assert "alpha" in a["description"]


def test_single_accession_entries_are_bare_symbols(store):
    """The 17,543-entry path: id and label are just the symbol."""
    e = store.entry("MAP2K4")
    assert e["id"] == "MAP2K4"
    assert e["label"] == "MAP2K4"


def test_every_entry_resolves_and_none_is_empty(store):
    for e in store.entries():
        assert store.entry(e["id"]) is e
        assert e["modalities"], f"{e['id']} has no data in any modality"


def test_entry_accepts_id_label_or_bare_symbol(store):
    target = store.entry("TMPO.P42167")
    assert store.entry("TMPO (P42167)") is target
    # a bare symbol resolves to the lowest accession, deterministically
    assert store.entry("TMPO")["id"] == "TMPO.P42166"
    assert store.entry("no-such-gene") is None


# --------------------------------------------------------------------------- #
# slices are one protein
# --------------------------------------------------------------------------- #
def test_no_slice_ever_holds_two_proteins(store):
    """The invariant the whole design rests on — it lets every figure builder
    stay single-protein, with no accession handling of its own."""
    for e in store.entries():
        for m in MODALITIES:
            df = store.slice(m, e["symbol"], e["uniprot"])
            if "uniprot" in df.columns and not df.is_empty():
                assert df["uniprot"].n_unique() == 1, f"{e['id']} / {m}"


def test_the_bar_is_one_protein_not_the_mean_of_two(store):
    """The regression guard, asserted on what is actually drawn.

    Keyed on symbol, the slices for "TMPO" held both proteins at once: the
    replicate overlay drew all 24 channel rows, and the bar — then the mean of
    those rows — drew their pooled 0.531, a value belonging to neither protein.
    Each entry must now draw its own row and its own 12 channels.

    The bar is the published log2FC rather than that mean now (see
    tests/test_proteome_figure.py), so the pooling would no longer reach it
    through the replicates — but the same symbol-keyed slice would still hand
    the figure two rows, which is what these values pin.
    """
    d8c = pl.col("condition") == "D8C"
    pooled = store.tables["proteome_replicates"].filter(
        (pl.col("symbol") == "TMPO") & d8c
    )
    assert pooled.height == 24
    assert round(pooled["log2fc"].mean(), 3) == 0.531  # the old, fabricated bar

    drawn = {}
    for uniprot in ("P42166", "P42167"):
        reps = store.replicates("proteome", "TMPO", uniprot)
        assert reps.filter(d8c).height == 12, "replicates must be one protein's"
        slice_ = store.slice("proteome", "TMPO", uniprot)
        assert slice_.height == 5, "one protein's rows, one per condition"
        fig = BUILDERS["proteome"](slice_, uniprot, reps)
        bar = next(t for t in fig.data if t.type == "bar")
        drawn[uniprot] = dict(zip(bar.x, bar.y))["D8C"]

    assert round(drawn["P42166"], 3) == 0.741
    assert round(drawn["P42167"], 3) == 0.328
    for value in drawn.values():
        assert round(value, 3) != 0.531


def test_modalities_follow_the_accession_not_the_symbol(store):
    """MIEF1's two proteins were seen by different assays, so neither entry may
    claim the other's data. AltMIEF1 has no cysteine data; MiD51 no abundance."""
    assert store.entry("MIEF1.L0R8F8")["modalities"] == ["proteome", "rna"]
    assert store.entry("MIEF1.Q9NQG6")["modalities"] == ["rna", "reactivity"]
    # p16INK4a vs p14ARF, same shape
    assert "reactivity" not in store.entry("CDKN2A.P42771")["modalities"]
    assert "proteome" not in store.entry("CDKN2A.Q8N726")["modalities"]


def test_rna_is_shared_by_every_entry_of_a_symbol(store):
    """RNA-seq is transcript-level: one measurement serves every protein the
    transcript encodes, so both TMPO entries legitimately show the same card."""
    a = store.slice("rna", "TMPO", "P42166")
    b = store.slice("rna", "TMPO", "P42167")
    assert not a.is_empty()
    assert a.equals(b)


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def test_searching_a_split_symbol_offers_both_proteins(store):
    hits = [r["id"] for r in store.search("TMPO")]
    assert hits[:2] == ["TMPO.P42166", "TMPO.P42167"]


def test_searching_an_accession_offers_only_that_protein(store):
    assert [r["id"] for r in store.search("P42167")] == ["TMPO.P42167"]


def test_cysteine_only_proteins_are_reachable(store):
    """Seeding the registry from the proteome alone left proteins that only the
    cysteine assays saw out of the index entirely — they held real data no
    search could reach."""
    hits = [r["id"] for r in store.search("BLTP1")]
    assert hits, "cysteine-only protein missing from the search index"
    assert store.entry(hits[0])["modalities"]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry_id", ["TMPO.P42166", "TMPO.P42167", "MAP2K4"])
def test_figures_keep_the_plain_condition_axis(store, entry_id):
    """An earlier fix split the x axis instead, giving one card eight bars with
    "D8C<br>P42166" categories. Entries replaced it; no axis markup may return."""
    e = store.entry(entry_id)
    for m in MODALITIES:
        df = store.slice(m, e["symbol"], e["uniprot"])
        if df.is_empty():
            continue
        fig = BUILDERS[m](df, e["label"], store.replicates(m, e["symbol"], e["uniprot"]))
        for trace in fig.data:
            for x in list(getattr(trace, "x", None) or []):
                assert "<br>" not in str(x)


def test_proteome_bar_has_one_bar_per_condition(store):
    e = store.entry("TMPO.P42166")
    df = store.slice("proteome", "TMPO", "P42166")
    fig = BUILDERS["proteome"](df, e["label"], store.replicates("proteome", "TMPO", "P42166"))
    bar = next(t for t in fig.data if t.type == "bar")
    assert list(bar.x) == ["D4A", "D4C", "D8A", "D8C"]


# --------------------------------------------------------------------------- #
# volcano
# --------------------------------------------------------------------------- #
def test_volcano_labels_split_symbols_by_accession(store):
    df = store.volcano_slice("D8C", "proteome")
    labels = dict(
        zip(*df.filter(pl.col("symbol") == "TMPO")
            .select(["uniprot", "label"])
            .to_dict(as_series=False)
            .values())
    )
    assert labels == {"P42166": "TMPO (P42166)", "P42167": "TMPO (P42167)"}
    # everything else keeps its bare symbol, so the JSON does not grow
    one = df.filter(pl.col("symbol") == "MAP2K4")
    assert one["label"].to_list() == ["MAP2K4"]


def test_transcriptome_volcano_labels_by_symbol(store):
    """It is transcript-level and has no accession to qualify with."""
    df = store.volcano_slice("D8C", "rna")
    assert df.filter(pl.col("symbol") == "TMPO")["label"].to_list() == ["TMPO"]


def test_volcano_pins_the_protein_that_was_opened(store):
    """``pin.row(0)`` matched on symbol and pinned an arbitrary one of TMPO's
    two points. Matching on label pins exactly the one the reader opened."""
    df = store.volcano_slice("D8C", "proteome")
    expected = {
        u: df.filter(pl.col("uniprot") == u)["log2fc"][0] for u in ("P42166", "P42167")
    }
    assert expected["P42166"] != expected["P42167"]
    for uniprot, log2fc in expected.items():
        fig = volcano_figure(df, "D8C", f"TMPO ({uniprot})")
        pins = [t for t in fig.data if t.showlegend is False and t.name.startswith("TMPO")]
        assert len(pins) == 1
        assert pins[0].x[0] == log2fc
        assert len(fig.layout.annotations) == 1


def test_volcano_click_through_carries_the_entry_label(store):
    """customdata[0] is what app.js hands to selectGene."""
    df = store.volcano_slice("D8C", "proteome")
    fig = volcano_figure(df, "D8C")
    names = {row[0] for trace in fig.data for row in (trace.customdata or [])}
    assert "TMPO (P42166)" in names
    assert "TMPO" not in names  # never the ambiguous bare symbol
    assert "MAP2K4" in names


# --------------------------------------------------------------------------- #
# the entry id has to survive being a filename and a URL
# --------------------------------------------------------------------------- #
def test_entry_ids_need_no_escaping(store):
    from prerender import _safe_key

    assert _safe_key("TMPO.P42166") == "TMPO.P42166"
    for e in store.entries():
        if e["id"] != e["symbol"]:
            assert _safe_key(e["id"]) == e["id"]
