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
    """
    by_symbol = _symbol_lookup(_load_geneid2nt())

    seed = (
        seed.dropna(subset=["symbol"])
        .drop_duplicates(subset=["symbol"])
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
        rows.append(
            {
                "uniprot": r.get("uniprot"),
                "symbol": symbol,
                "description": description,
                # pipe-delimited, searchable
                "aliases": "|".join(aliases),
            }
        )
    return pd.DataFrame(rows)
