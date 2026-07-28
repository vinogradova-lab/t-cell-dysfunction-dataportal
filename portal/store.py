"""In-memory data store.

At import time we load the parquet tables produced by ``etl/build_db.py`` into
polars DataFrames and build a small search index. Everything is read-only, so a
single process-wide store is shared across requests (gunicorn workers each hold
their own copy — the tables are small).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import polars as pl

PORTAL_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = Path(os.environ.get("PARQUET_DIR", PORTAL_ROOT / "data" / "parquet"))
DOWNLOAD_DIR = Path(
    os.environ.get("DOWNLOAD_DIR", PORTAL_ROOT / "data" / "downloads")
)

# modality -> parquet filename
_TABLE_FILES = {
    "genes": "genes.parquet",
    "proteome": "proteome.parquet",
    "proteome_replicates": "proteome_replicates.parquet",
    "rna": "rna.parquet",
    "rna_replicates": "rna_replicates.parquet",
    "reactivity": "reactivity.parquet",
    "reactivity_atp": "reactivity_atp.parquet",
    # download-only: no per-gene view, so not a modality
    "metabolomics": "metabolomics.parquet",
    "volcano": "volcano.parquet",
    "rna_volcano": "rna_volcano.parquet",
}

MODALITIES = ["proteome", "rna", "reactivity", "reactivity_atp"]

# volcano comparisons, in display order, with human labels. The first four are
# against D2 — left unqualified because the figure's x-axis title names the
# reference — and the last is chronic over acute at day 8.
VOLCANO_COMPARISONS = [
    ("D4A", "Day 4, Acute"),
    ("D4C", "Day 4, Chronic"),
    ("D8A", "Day 8, Acute"),
    ("D8C", "Day 8, Chronic"),
    ("D8C_vs_D8A", "Day 8, Chronic vs Acute"),
]

# volcano dataset id -> (parquet table, picker label). Both datasets cover the
# same comparisons; the RNA one is restricted to genes that also have
# whole-proteome data, so the two plot the same gene universe.
VOLCANO_DATASETS = [
    ("proteome", "volcano", "Whole proteome"),
    ("rna", "rna_volcano", "Transcriptome (matched genes)"),
]
_VOLCANO_TABLE = {ds: table for ds, table, _ in VOLCANO_DATASETS}
DEFAULT_VOLCANO_DATASET = VOLCANO_DATASETS[0][0]


class Store:
    def __init__(self, parquet_dir: Path = PARQUET_DIR):
        self.tables: dict[str, pl.DataFrame] = {}
        for name, fname in _TABLE_FILES.items():
            path = parquet_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"missing parquet table {path}. Run the ETL first "
                    "(scripts/sync_source.py then etl/build_db.py)."
                )
            self.tables[name] = pl.read_parquet(path)

        # search index: one row per gene with a lowercased haystack of
        # symbol + uniprot + aliases for fast prefix/substring matching.
        genes = self.tables["genes"]
        self._genes = genes.with_columns(
            pl.col("symbol").str.to_lowercase().alias("_symbol_lc"),
            pl.col("uniprot").fill_null("").str.to_lowercase().alias("_uniprot_lc"),
            pl.col("aliases").fill_null("").str.to_lowercase().alias("_aliases_lc"),
        )
        # symbol -> uniprot / description / function, for gene-page metadata.
        # uniprot_function is optional (older parquet may predate the column).
        meta_cols = ["symbol", "uniprot", "description"]
        if "uniprot_function" in genes.columns:
            meta_cols.append("uniprot_function")
        self._by_symbol = {
            r["symbol"]: r
            for r in genes.select(meta_cols).iter_rows(named=True)
        }
        # sets of symbols present in each modality, for "has data" flags
        self._symbols_in = {
            m: set(self.tables[m]["symbol"].unique().to_list()) for m in MODALITIES
        }

    # ---- search -------------------------------------------------------- #
    def search(self, query: str, limit: int = 15) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        g = self._genes
        # rank: exact symbol > symbol prefix > uniprot match > alias/substring
        exact = g.filter(pl.col("_symbol_lc") == q)
        prefix = g.filter(
            pl.col("_symbol_lc").str.starts_with(q) & (pl.col("_symbol_lc") != q)
        )
        uni = g.filter(pl.col("_uniprot_lc").str.starts_with(q))
        alias = g.filter(
            pl.col("_aliases_lc").str.contains(q, literal=True)
            | (pl.col("_symbol_lc").str.contains(q, literal=True))
        )
        seen: set[str] = set()
        out: list[dict] = []
        for frame in (exact, prefix, uni, alias):
            for r in frame.select(
                ["symbol", "uniprot", "description"]
            ).iter_rows(named=True):
                if r["symbol"] in seen:
                    continue
                seen.add(r["symbol"])
                out.append(
                    {
                        "symbol": r["symbol"],
                        "uniprot": r["uniprot"],
                        "description": r["description"],
                        "modalities": self.modalities_for(r["symbol"]),
                    }
                )
                if len(out) >= limit:
                    return out
        return out

    # ---- gene metadata ------------------------------------------------- #
    def gene_meta(self, symbol: str) -> dict | None:
        r = self._by_symbol.get(symbol)
        if r is None:
            return None
        return {
            "symbol": r["symbol"],
            "uniprot": r["uniprot"],
            "description": r["description"],
            "uniprot_function": r.get("uniprot_function") or "",
            "modalities": self.modalities_for(symbol),
        }

    def modalities_for(self, symbol: str) -> list[str]:
        return [m for m in MODALITIES if symbol in self._symbols_in[m]]

    # ---- per-gene data slices ------------------------------------------ #
    def slice(self, modality: str, symbol: str) -> pl.DataFrame:
        return self.tables[modality].filter(pl.col("symbol") == symbol)

    def replicates(self, modality: str, symbol: str) -> pl.DataFrame:
        """Per-replicate rows for a gene, or an empty frame when unavailable."""
        table = self.tables.get(f"{modality}_replicates")
        if table is None:
            return pl.DataFrame()
        return table.filter(pl.col("symbol") == symbol)

    # ---- volcano (dataset-wide) ---------------------------------------- #
    def volcano_datasets(self) -> list[dict]:
        """Volcano datasets that have a table loaded, for the dataset picker."""
        return [
            {"id": ds, "label": label}
            for ds, table, label in VOLCANO_DATASETS
            if table in self.tables
        ]

    def volcano_comparisons(
        self, dataset: str = DEFAULT_VOLCANO_DATASET
    ) -> list[dict]:
        """Available comparisons that actually have volcano data."""
        table = _VOLCANO_TABLE.get(dataset)
        if table is None or table not in self.tables:
            return []
        present = set(self.tables[table]["comparison"].unique().to_list())
        return [
            {"id": cid, "label": label}
            for cid, label in VOLCANO_COMPARISONS
            if cid in present
        ]

    def volcano_slice(
        self, comparison: str, dataset: str = DEFAULT_VOLCANO_DATASET
    ) -> pl.DataFrame:
        """All points for one (dataset, comparison) — empty frame if unknown."""
        table = _VOLCANO_TABLE.get(dataset)
        if table is None or table not in self.tables:
            return pl.DataFrame()
        return self.tables[table].filter(pl.col("comparison") == comparison)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
