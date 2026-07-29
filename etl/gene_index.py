"""Build the gene search registry.

The registry powers autocomplete: users type a gene symbol, an alias, or a
UniProt accession and get back the canonical ``(symbol, uniprot, description)``.

Canonical genes come from the datasets themselves (whichever proteins/genes we
actually have data for). We enrich each with aliases + a human-readable
description from the NCBI ``GENEID2NT`` table, which ``scripts/sync_source.py``
stages into ``source/genes_ncbi_9606_proteincoding.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PORTAL_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PORTAL_ROOT / "source"
# uniprot -> function summary, built offline by scripts/build_uniprot_functions.py
# (workbook + UniProt API) and committed so CI stays offline.
UNIPROT_FUNCTIONS_CSV = PORTAL_ROOT / "data" / "uniprot_functions.csv"


def _clean_description(desc: str) -> str:
    """Strip UniProt header cruft ('... OS=Homo sapiens OX=9606 GN=... SV=1')
    down to the human-readable protein name."""
    if not isinstance(desc, str):
        return ""
    return desc.split(" OS=")[0].strip()


def _load_geneid2nt() -> dict:
    """Import GENEID2NT from the staged NCBI gene module in source/."""
    module_path = SOURCE_DIR / "genes_ncbi_9606_proteincoding.py"
    if not module_path.exists():
        # Aliases are a nice-to-have; degrade gracefully if the table is absent.
        return {}
    spec = importlib.util.spec_from_file_location(
        "genes_ncbi_9606_proteincoding", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return getattr(module, "GENEID2NT", {})


def _load_uniprot_functions() -> dict[str, str]:
    """Return uniprot -> function summary from the committed CSV.

    Empty-string rows (accessions UniProt has no FUNCTION for) are dropped so a
    missing key uniformly means "no function to show". Absent CSV degrades
    gracefully to an empty map — the function line is simply omitted."""
    if not UNIPROT_FUNCTIONS_CSV.exists():
        return {}
    df = pd.read_csv(UNIPROT_FUNCTIONS_CSV, keep_default_na=False)
    return {
        str(u): str(f)
        for u, f in zip(df["uniprot"], df["uniprot_function"])
        if str(f).strip()
    }


def _symbol_lookup(geneid2nt: dict) -> dict:
    """Return symbol -> {aliases, description}."""
    by_symbol: dict[str, dict] = {}
    for nt in geneid2nt.values():
        aliases = list(nt.Aliases) if nt.Aliases else []
        by_symbol[nt.Symbol] = {"aliases": aliases, "description": nt.description}
    return by_symbol


def build_gene_registry(seed: pd.DataFrame) -> pd.DataFrame:
    """Build the registry frame.

    ``seed`` must have columns ``uniprot``, ``symbol``, ``description`` — one row
    per protein/gene we have data for (dedup handled here).

    Deduped on ``(uniprot, symbol)``, so a symbol that maps to several UniProt
    entries keeps one row per protein. Deduping on ``symbol`` alone kept whichever
    row pandas happened to see first and silently discarded the rest — that is
    what let a single "TMPO" page pool P42166 and P42167, two proteins with
    different fold changes, and put p16INK4a's abundance beside p14ARF's
    cysteines under one "CDKN2A" heading.
    """
    by_symbol = _symbol_lookup(_load_geneid2nt())
    functions = _load_uniprot_functions()

    seed = (
        seed.dropna(subset=["symbol"])
        .drop_duplicates(subset=["uniprot", "symbol"])
        .reset_index(drop=True)
    )

    rows = []
    for _, r in seed.iterrows():
        symbol = str(r["symbol"])
        info = by_symbol.get(symbol, {})
        aliases = info.get("aliases", [])
        # Prefer the dataset's own description (full protein name); fall back to
        # the NCBI short description.
        description = _clean_description(r.get("description"))
        if not description:
            description = info.get("description", "")
        uniprot = r.get("uniprot")
        rows.append(
            {
                "uniprot": uniprot,
                "symbol": symbol,
                "description": description,
                # pipe-delimited, searchable
                "aliases": "|".join(aliases),
                # UniProt function summary (empty when unavailable)
                "uniprot_function": functions.get(str(uniprot), ""),
            }
        )
    return pd.DataFrame(rows)
