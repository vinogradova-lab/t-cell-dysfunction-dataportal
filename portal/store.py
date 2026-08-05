"""In-memory data store.

At import time we load the parquet tables produced by ``etl/build_db.py`` into
polars DataFrames and build a small search index. Everything is read-only, so a
single process-wide store is shared across requests (gunicorn workers each hold
their own copy — the tables are small).

The unit of a search result, a URL and a page is one **protein**, not one gene
symbol — see :class:`Store`.
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

# The id is also the bundle filename and the ``?gene=`` value, so it sticks to
# characters ``prerender._safe_key`` passes through unescaped.
_ID_SEP = "."


def _split_id(symbol: str, uniprot: str) -> str:
    return f"{symbol}{_ID_SEP}{uniprot}"


def _split_label(symbol: str, uniprot: str) -> str:
    return f"{symbol} ({uniprot})"


class Store:
    """Parquet tables plus a protein-level entry index.

    A gene symbol is not a unique protein id: the tables are keyed on UniProt
    accession, and five symbols were measured under more than one (TMPO, MOCS2,
    MIEF1, CDKN2A, POLR1D). Keyed on symbol their pages pooled two proteins —
    TMPO's D8C bar drew mean(0.772, 0.332) = 0.531, belonging to neither. So one
    **entry** = one registry row = one protein:

        1 accession   ->  id = "ACTB"          label = "ACTB"
        several       ->  id = "TMPO.P42166"   label = "TMPO (P42166)"

    15,763 of 15,773 entries take the first branch, unchanged from the old
    symbol-keyed behaviour. The rule lives here alone; prerender, figures and
    app.js all consume what this computes.
    """

    def __init__(self, parquet_dir: Path = PARQUET_DIR):
        self.tables: dict[str, pl.DataFrame] = {}
        for name, fname in _TABLE_FILES.items():
            path = parquet_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"missing parquet table {path}. Run the ETL first "
                    "(scripts/sync_source.py then etl/build_db.py)."
                )
            # parquet is LFS-tracked; a clone without LFS leaves pointer files
            # here, and polars' parse error for one is opaque
            with path.open("rb") as fh:
                if fh.read(4) != b"PAR1":
                    raise RuntimeError(
                        f"{path} is a Git LFS pointer, not a parquet table. "
                        "Run `git lfs install && git lfs pull`."
                    )
            self.tables[name] = pl.read_parquet(path)

        genes = self.tables["genes"]
        # symbols holding several accessions — the ones that get qualified ids.
        # Derived, not hardcoded, so a source revision needs no code change.
        multi = (
            genes.filter(pl.col("uniprot").is_not_null())
            .group_by("symbol")
            .agg(pl.col("uniprot").n_unique().alias("n"))
            .filter(pl.col("n") > 1)["symbol"]
        )
        self._multi_symbols = set(multi.to_list())

        # Which modalities key on accession. RNA does not — it is
        # transcript-level, so both TMPO entries share one measurement.
        self._by_uniprot = {
            m: "uniprot" in self.tables[m].columns for m in MODALITIES
        }
        # membership keys, per that distinction
        self._keys_in: dict[str, set] = {}
        for m in MODALITIES:
            t = self.tables[m]
            self._keys_in[m] = (
                set(zip(t["symbol"].to_list(), t["uniprot"].to_list()))
                if self._by_uniprot[m]
                else set(t["symbol"].to_list())
            )

        # one entry per registry row, in registry order
        meta_cols = ["symbol", "uniprot", "description", "aliases"]
        for optional in ("uniprot_function", "rna_unquantified"):
            if optional in genes.columns:
                meta_cols.append(optional)
        self._entries: list[dict] = []
        for r in genes.select(meta_cols).iter_rows(named=True):
            symbol, uniprot = r["symbol"], r["uniprot"]
            split = uniprot is not None and symbol in self._multi_symbols
            self._entries.append(
                {
                    "id": _split_id(symbol, uniprot) if split else symbol,
                    "label": _split_label(symbol, uniprot) if split else symbol,
                    "symbol": symbol,
                    "uniprot": uniprot,
                    "description": r["description"] or "",
                    "aliases": r["aliases"] or "",
                    "uniprot_function": r.get("uniprot_function") or "",
                    "modalities": self.modalities_for(symbol, uniprot),
                    # sequenced but never quantified, so "rna" is absent from
                    # modalities for a reason the page can state
                    "rna_unquantified": bool(r.get("rna_unquantified")),
                }
            )
        self._by_entry = {e["id"]: e for e in self._entries}
        # A bare symbol (an old link, an RNA volcano click) resolves to the
        # lowest accession — deterministic, and matched by app.js.
        self._primary: dict[str, str] = {}
        for e in sorted(self._entries, key=lambda e: (e["symbol"], e["uniprot"] or "")):
            self._primary.setdefault(e["symbol"], e["id"])

        # accession -> volcano point label, split symbols only; everything else
        # keeps its bare symbol, so the volcano JSON does not grow
        self._label_by_uniprot = {
            e["uniprot"]: e["label"]
            for e in self._entries
            if e["uniprot"] is not None and e["symbol"] in self._multi_symbols
        }

        # search index: one row per entry with a lowercased haystack of
        # symbol + uniprot + aliases for fast prefix/substring matching.
        self._genes = genes.with_columns(
            pl.col("symbol").str.to_lowercase().alias("_symbol_lc"),
            pl.col("uniprot").fill_null("").str.to_lowercase().alias("_uniprot_lc"),
            pl.col("aliases").fill_null("").str.to_lowercase().alias("_aliases_lc"),
            pl.Series("_id", [e["id"] for e in self._entries]),
        )
        # symbol -> rows, per table, built on first use. Filtering by symbol is a
        # full scan; the prerender does ~140k of them, so one partition pass per
        # table turns the whole thing into dict lookups. Lazy so the web app
        # doesn't pay for tables it never slices.
        self._partitions: dict[str, dict[str, pl.DataFrame]] = {}

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
        # dedup on entry id: "TMPO" returns both its proteins, "P42167" one
        seen: set[str] = set()
        out: list[dict] = []
        for frame in (exact, prefix, uni, alias):
            for eid in frame["_id"].to_list():
                if eid in seen:
                    continue
                seen.add(eid)
                out.append(self._by_entry[eid])
                if len(out) >= limit:
                    return out
        return out

    # ---- entries (one per protein) ------------------------------------- #
    def entries(self) -> list[dict]:
        """Every entry, in registry order — the portal's full gene universe."""
        return self._entries

    def entry(self, entry_id: str) -> dict | None:
        """One entry by id, bare symbol, or display label.

        All three, so old ``?gene=TMPO`` links keep working and a volcano click
        can pass ``customdata[0]`` straight through.
        """
        hit = self._by_entry.get(entry_id)
        if hit is not None:
            return hit
        primary = self._primary.get(entry_id)
        if primary is not None:
            return self._by_entry[primary]
        # display label, e.g. "TMPO (P42166)"
        if entry_id.endswith(")") and " (" in entry_id:
            symbol, _, rest = entry_id.partition(" (")
            return self._by_entry.get(_split_id(symbol, rest[:-1]))
        return None

    def modalities_for(self, symbol: str, uniprot: str | None = None) -> list[str]:
        """Modalities holding data for this protein.

        Matching accession-keyed tables on ``(symbol, uniprot)`` is what gives
        ``MIEF1 (Q9NQG6)`` a reactivity card and no proteome card.
        """
        def has(m: str) -> bool:
            if self._by_uniprot[m]:
                return (symbol, uniprot) in self._keys_in[m]
            return symbol in self._keys_in[m]

        return [m for m in MODALITIES if has(m)]

    # ---- per-gene data slices ------------------------------------------ #
    def _by_symbol_index(self, table: str) -> dict[str, pl.DataFrame]:
        index = self._partitions.get(table)
        if index is None:
            # as_dict keys are one-element tuples (the partition key)
            index = {
                key[0]: frame
                for key, frame in self.tables[table]
                .partition_by("symbol", as_dict=True)
                .items()
            }
            self._partitions[table] = index
        return index

    def _slice(self, table: str, symbol: str, uniprot: str | None) -> pl.DataFrame:
        # Symbol lookup first (the fast path), then narrow the handful of rows
        # to one protein. .clear() keeps the schema on a miss, so callers can
        # select columns off it the same way they would off a hit.
        frame = self._by_symbol_index(table).get(symbol, self.tables[table].clear())
        if uniprot is not None and "uniprot" in frame.columns:
            frame = frame.filter(pl.col("uniprot") == uniprot)
        return frame

    def slice(
        self, modality: str, symbol: str, uniprot: str | None = None
    ) -> pl.DataFrame:
        return self._slice(modality, symbol, uniprot)

    def replicates(
        self, modality: str, symbol: str, uniprot: str | None = None
    ) -> pl.DataFrame:
        """Per-replicate rows for a gene, or an empty frame when unavailable."""
        name = f"{modality}_replicates"
        if name not in self.tables:
            return pl.DataFrame()
        return self._slice(name, symbol, uniprot)

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
        """All points for one (dataset, comparison) — empty frame if unknown.

        Carries a ``label`` column — entry label for a split symbol, bare symbol
        otherwise — which the hover shows, click-through resolves, and the pin
        matches. The proteome volcano has one point per *accession*, so matching
        on symbol pinned an arbitrary one of a symbol's two. Derived here so
        figures.py and app.js share one rule.
        """
        table = _VOLCANO_TABLE.get(dataset)
        if table is None or table not in self.tables:
            return pl.DataFrame()
        df = self.tables[table].filter(pl.col("comparison") == comparison)
        if "uniprot" in df.columns and self._label_by_uniprot:
            label = (
                pl.col("uniprot")
                .replace_strict(self._label_by_uniprot, default=None)
                .fill_null(pl.col("symbol"))
            )
        else:
            # the transcriptome volcano is symbol-keyed; nothing to qualify
            label = pl.col("symbol")
        return df.with_columns(label.alias("label"))


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
