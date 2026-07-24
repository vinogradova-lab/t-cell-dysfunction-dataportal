"""Build ``data/uniprot_functions.csv`` — the committed ``uniprot -> function`` map.

The per-gene page header shows a one-line UniProt function summary. That text is
sourced in two passes:

  1. The published supplementary workbook (``source/data_s2.xlsx``) already
     carries a ``uniprot_function`` column on its "Low-input WP" sheets — use it
     verbatim so the portal matches the paper.
  2. Accessions not covered there are fetched from the UniProt REST API, reusing
     the FUNCTION-comment parser from the analysis repo
     (``t-cell-dysfunction-2026`` notebooks/05_low_input_proteomics/low_input.py).

This is a LOCAL step: it needs the private ``source/`` workbooks and network
access. CI never runs it — the ETL ([etl/gene_index.py]) reads the committed CSV
this writes, so the GitHub Pages build stays fully offline.

    python scripts/build_uniprot_functions.py

The CSV doubles as an API cache: it is re-read on each run and only accessions
absent from both the workbook and the existing CSV are fetched. Accessions that
UniProt returns *no* FUNCTION for are written back as empty rows (a negative
cache), so re-running with an unchanged protein list makes **zero** API calls.
Delete the CSV to force a full refetch.

``unipressed`` is required locally (already used by the analysis repo); it is
intentionally NOT in requirements.txt since CI does not need it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# UniProtKB accession syntax (excludes contaminant/std pseudo-ids in the sheets).
_ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)

PORTAL_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PORTAL_ROOT / "source"
OUT_CSV = PORTAL_ROOT / "data" / "uniprot_functions.csv"

# Supplementary sheets (Data S2) whose header row (0-indexed row 2) exposes a
# `uniprot_function` column keyed by `uniprot`.
WORKBOOK = SOURCE / "data_s2.xlsx"
FUNCTION_SHEETS = [
    "S2-6 Low-input WP, mouse",
    "S2-7 Low-input WP, TILs",
    "S2-8 Low-input RC, inhibitors",
    "S2-9 Low-input WP, inhibitors",
]


def target_accessions() -> list[str]:
    """The accessions we need a function for: every UniProt id in the gene
    registry (built from the whole-proteome seed)."""
    genes = pd.read_parquet(PORTAL_ROOT / "data" / "parquet" / "genes.parquet")
    acc = genes["uniprot"].dropna().astype(str)
    return sorted({a for a in acc if _ACCESSION_RE.match(a)})


def workbook_functions() -> dict[str, str]:
    """uniprot -> function, harvested from the published supplementary sheets."""
    out: dict[str, str] = {}
    for sheet in FUNCTION_SHEETS:
        df = pd.read_excel(WORKBOOK, sheet_name=sheet, header=2)
        if "uniprot" not in df.columns or "uniprot_function" not in df.columns:
            continue
        sub = df[["uniprot", "uniprot_function"]].dropna()
        for u, f in zip(sub["uniprot"].astype(str), sub["uniprot_function"].astype(str)):
            f = f.strip()
            if u and f and u not in out:
                out[u] = f
    return out


def _get_function(entry: dict) -> str:
    """Concatenate an entry's FUNCTION comment texts, matching the analysis
    repo's ``get_function`` (pipe-delimited, deduped)."""
    texts: list[str] = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") != "FUNCTION":
            continue
        for t in comment.get("texts", []):
            v = (t.get("value") or "").strip()
            if v and v not in texts:
                texts.append(v)
    return "|".join(texts)


def api_functions(accessions: list[str]) -> dict[str, str]:
    """Fetch FUNCTION comments from UniProt for the given accessions."""
    if not accessions:
        return {}
    from unipressed import UniprotkbClient  # local-only dependency

    out: dict[str, str] = {}
    chunk_size = 100  # the /accessions endpoint accepts at most 100 ids per call
    chunks = [
        accessions[i : i + chunk_size] for i in range(0, len(accessions), chunk_size)
    ]
    for i, chunk in enumerate(chunks, 1):
        print(f"  UniProt API: chunk {i}/{len(chunks)} ({len(chunk)} accessions)")
        try:
            entries = list(UniprotkbClient.fetch_many(chunk))
        except Exception as e:  # noqa: BLE001 - a bad chunk shouldn't sink the run
            print(f"    chunk {i} failed ({e}); skipping")
            continue
        for entry in entries:
            acc = entry.get("primaryAccession")
            fn = _get_function(entry)
            if acc and fn:
                out[acc] = fn
    return out


def load_cache() -> dict[str, str]:
    """Prior run's results (accession -> function), including empty-string
    negatives. Empty when the CSV does not exist yet."""
    if not OUT_CSV.exists():
        return {}
    df = pd.read_csv(OUT_CSV, keep_default_na=False)  # keep "" as "", not NaN
    return dict(zip(df["uniprot"].astype(str), df["uniprot_function"].astype(str)))


def main() -> int:
    targets = target_accessions()
    print(f"Target accessions (from genes.parquet): {len(targets):,d}")

    wb = workbook_functions()
    wb_hits = {u: wb[u] for u in targets if u in wb}
    print(f"  from workbook: {len(wb_hits):,d}")

    cache = load_cache()
    cached_hits = sum(1 for u in targets if u not in wb_hits and cache.get(u))
    print(f"  from cache ({OUT_CSV.name}): {cached_hits:,d}")

    # Only accessions we have never resolved (no workbook text, not in the cache
    # at all — present-but-empty counts as "already tried") hit the network.
    need = [u for u in targets if u not in wb_hits and u not in cache]
    print(f"  new -> UniProt API: {len(need):,d}")
    api = api_functions(need)
    print(f"  from API: {sum(1 for v in api.values() if v):,d}")

    # Persist negatives (queried, no FUNCTION) as empty rows so we don't refetch.
    for u in need:
        cache[u] = api.get(u, "")

    # Precedence: published workbook > cached/API text. Write every target,
    # keeping empty rows as the negative cache.
    rows = [(u, wb_hits[u] if u in wb_hits else cache.get(u, "")) for u in targets]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["uniprot", "uniprot_function"]).to_csv(
        OUT_CSV, index=False
    )
    covered = sum(1 for _, f in rows if f)
    print(
        f"\nWrote {OUT_CSV.relative_to(PORTAL_ROOT)}: {covered:,d}/{len(targets):,d} "
        f"accessions ({100 * covered / len(targets):.1f}%) have a function "
        f"({len(rows) - covered:,d} cached negatives)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
