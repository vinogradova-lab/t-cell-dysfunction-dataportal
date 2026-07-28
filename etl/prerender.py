"""Pre-render the portal into a static ``site/`` tree for GitHub Pages.

The live Flask app ([portal/app.py]) is a pure function of the parquet tables
produced by [etl/build_db.py] — every endpoint is deterministic given the
parquet. This script drives the same ``Store`` and figure builders off the
committed parquet and emits a fully static site: a search index, one figure
bundle per gene, the volcano figures, the bulk-download files, and the SPA
shell with asset paths rewritten to be **relative** (so it works both at a
project-page subpath and, later, at a custom domain — no code change).

Usage:
    python etl/prerender.py [--limit N] [--out site]

``--limit`` renders only the first N genes (for a quick smoke build); omit it
for the full ~17.5k-gene site.

Runs off the committed parquet only — it does **not** need the ``source/``
workbooks. The combined ``whole_proteome.csv`` download is reconstructed from
the ``proteome_replicates`` + ``volcano`` parquet tables (same columns as the
ETL's xlsx-based ``build_proteome_download``).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

PORTAL_ROOT = Path(__file__).resolve().parents[1]
# import the live portal modules + ETL constants (no side effects at import)
sys.path.insert(0, str(PORTAL_ROOT / "etl"))
sys.path.insert(0, str(PORTAL_ROOT / "portal"))

import build_db  # noqa: E402  (DATA_DICTIONARY, DOWNLOAD_TABLES, constants)
from app import _DOWNLOAD_ITEMS  # noqa: E402  (download labels, DRY with the API)
from figures import BUILDERS, VOLCANO_BUILDERS  # noqa: E402
from store import MODALITIES, Store  # noqa: E402

STATIC_SUBDIRS = ["css", "js", "vendor", "img"]


def _safe_key(symbol: str) -> str:
    """Filesystem-/URL-safe gene key: keep ``[A-Za-z0-9._-]``, percent-encode
    the rest. The common case is ``key == symbol``; the client stores the key
    per gene, so no server-side URL decoding is ever needed."""
    return "".join(
        ch if (ch.isalnum() or ch in "._-") else f"%{ord(ch):02X}"
        for ch in symbol
    )


def _fig_json(fig) -> dict:
    """Plotly figure -> plain dict (parsed once so json.dump controls encoding)."""
    return json.loads(fig.to_json())


# --------------------------------------------------------------------------- #
# search index + per-gene figure bundles
# --------------------------------------------------------------------------- #
def write_genes_index(store: Store, api_dir: Path) -> list[dict]:
    genes = store.tables["genes"]
    records = []
    for r in genes.select(
        ["symbol", "uniprot", "description", "aliases"]
    ).iter_rows(named=True):
        records.append(
            {
                "symbol": r["symbol"],
                "uniprot": r["uniprot"],
                "description": r["description"],
                "aliases": r["aliases"],
                "modalities": store.modalities_for(r["symbol"]),
                "key": _safe_key(r["symbol"]),
            }
        )
    (api_dir / "genes.json").write_text(json.dumps(records, separators=(",", ":")))
    return records


def write_gene_bundles(
    store: Store, records: list[dict], api_dir: Path, limit: int | None
) -> int:
    gene_dir = api_dir / "gene"
    gene_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    todo = [r for r in records if r["modalities"]]
    if limit is not None:
        todo = todo[:limit]
    total = len(todo)
    for i, r in enumerate(todo, 1):
        symbol = r["symbol"]
        bundle: dict[str, dict | str | None] = {}
        for m in MODALITIES:
            df = store.slice(m, symbol)
            if df.is_empty():
                bundle[m] = None
            else:
                fig = BUILDERS[m](df, symbol, store.replicates(m, symbol))
                bundle[m] = _fig_json(fig)
        # per-gene metadata carried alongside the figures (not a modality, so the
        # frontend's modality loop ignores it); kept out of the always-loaded
        # genes.json index because the function text is large.
        meta = store.gene_meta(symbol)
        bundle["uniprot_function"] = (meta or {}).get("uniprot_function", "")
        (gene_dir / f"{r['key']}.json").write_text(
            json.dumps(bundle, separators=(",", ":"))
        )
        written += 1
        if i % 1000 == 0 or i == total:
            print(f"  gene bundles {i:>6,d}/{total:,d}")
    return written


# --------------------------------------------------------------------------- #
# volcano (base figures; highlight is client-side)
# --------------------------------------------------------------------------- #
def write_volcano(store: Store, api_dir: Path) -> None:
    vdir = api_dir / "volcano"
    vdir.mkdir(parents=True, exist_ok=True)
    datasets = store.volcano_datasets()
    (vdir / "datasets.json").write_text(json.dumps(datasets))
    for ds in datasets:
        dsdir = vdir / ds["id"]
        dsdir.mkdir(parents=True, exist_ok=True)
        builder = VOLCANO_BUILDERS[ds["id"]]
        comparisons = store.volcano_comparisons(ds["id"])
        (dsdir / "comparisons.json").write_text(json.dumps(comparisons))
        for c in comparisons:
            df = store.volcano_slice(c["id"], ds["id"])
            if df.is_empty():
                continue
            (dsdir / f"{c['id']}.json").write_text(
                json.dumps(_fig_json(builder(df, c["id"])), separators=(",", ":"))
            )


# --------------------------------------------------------------------------- #
# bulk-download bundle (regenerated from the committed parquet)
# --------------------------------------------------------------------------- #
def _proteome_download_from_parquet(store: Store) -> pd.DataFrame:
    """Reconstruct the combined whole-proteome download from parquet, matching
    the column layout of ``build_db.build_proteome_download`` (which reads the
    S2-1 xlsx, absent in CI)."""
    reps = store.tables["proteome_replicates"].to_pandas()
    # The replicate table is per technical channel, but this download's
    # D{cond}_rep{N} columns are per *donor* (as build_proteome_download emits
    # them from the xlsx). Averaging a channel pair recovers the donor value
    # exactly, not approximately: both channels are divided by the same D2 mean,
    # so mean(100*c1/d2, 100*c2/d2) == 100*mean(c1,c2)/d2.
    wide = reps.pivot_table(
        index=["uniprot", "symbol"],
        columns=["condition", "bio_rep"],
        values="percent_control",
        aggfunc="mean",
    )
    wide.columns = [f"{cond}_{rep}" for cond, rep in wide.columns]
    wide = wide.reset_index()
    rep_cols = [
        f"{cond}_rep{n}"
        for cond in build_db.FIVE_CONDITIONS
        for n in build_db.PROTEOME_REPS
        if f"{cond}_rep{n}" in wide.columns
    ]

    v = store.tables["volcano"].to_pandas()
    stat_cols = ["log2fc", "p_value", "neglog10_pval", "neglog10_padj", "regulation"]
    flags = [f for f in build_db.VOLCANO_FLAGS if f in v.columns]
    vol_cols: list[str] = []
    out = wide
    # description from the gene registry (S2-1-seeded, covers every protein)
    genes = store.tables["genes"].to_pandas()[["symbol", "description"]]
    out = out.merge(genes, on="symbol", how="left")
    for cmp in build_db.VOLCANO_COMPARISONS:
        sub = v[v["comparison"] == cmp][["uniprot"] + stat_cols].rename(
            columns={c: f"{cmp}_{c}" for c in stat_cols}
        )
        out = out.merge(sub, on="uniprot", how="left")
        vol_cols += [f"{cmp}_{c}" for c in stat_cols]
    flag_tbl = v.drop_duplicates("uniprot")[["uniprot"] + flags]
    out = out.merge(flag_tbl, on="uniprot", how="left")

    cols = ["uniprot", "symbol", "description"] + rep_cols + vol_cols + flags
    return out[[c for c in cols if c in out.columns]]


def write_downloads(store: Store, site: Path, api_dir: Path) -> None:
    dl = site / "downloads"
    dl.mkdir(parents=True, exist_ok=True)

    # combined whole proteome (expression + replicates + significance)
    _proteome_download_from_parquet(store).to_csv(dl / "whole_proteome.csv", index=False)
    # the remaining tables map 1:1 to parquet
    for tbl_name, basename in build_db.DOWNLOAD_TABLES:
        store.tables[tbl_name].write_csv(dl / f"{basename}.csv")

    readme = dl / "README.txt"
    readme.write_text(build_db.DATA_DICTIONARY)

    zip_path = dl / "t_cell_dysfunction_proteomics.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme, "README.txt")
        for basename in build_db.DOWNLOAD_BASENAMES:
            zf.write(dl / f"{basename}.csv", f"{basename}.csv")

    # manifest with RELATIVE hrefs (mirrors /api/downloads)
    manifest = [
        {
            "label": "All data (combined)",
            "format": "zip",
            "href": "downloads/t_cell_dysfunction_proteomics.zip",
            "bytes": zip_path.stat().st_size,
        }
    ]
    for label, base in _DOWNLOAD_ITEMS:
        path = dl / f"{base}.csv"
        if path.exists():
            manifest.append(
                {
                    "label": label,
                    "format": "csv",
                    "href": f"downloads/{base}.csv",
                    "bytes": path.stat().st_size,
                }
            )
    (api_dir / "downloads.json").write_text(json.dumps(manifest))


# --------------------------------------------------------------------------- #
# static shell (index.html + assets)
# --------------------------------------------------------------------------- #
_URLFOR_RE = re.compile(r"\{\{\s*url_for\('static',\s*filename='([^']+)'\)\s*\}\}")


def write_shell(site: Path) -> None:
    # copy static asset dirs, dropping the ``static/`` prefix
    static_root = PORTAL_ROOT / "portal" / "static"
    for sub in STATIC_SUBDIRS:
        src = static_root / sub
        if src.exists():
            shutil.copytree(src, site / sub, dirs_exist_ok=True)
    # render index.html with a url_for shim -> relative asset paths
    template = (PORTAL_ROOT / "portal" / "templates" / "index.html").read_text()
    html = _URLFOR_RE.sub(r"\1", template)
    (site / "index.html").write_text(html)
    # bypass Jekyll so _-prefixed paths (and .nojekyll itself) are served
    (site / ".nojekyll").write_text("")
    # optional custom domain: write CNAME when the env var is set
    import os

    domain = os.environ.get("PAGES_CNAME")
    if domain:
        (site / "CNAME").write_text(domain.strip() + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PORTAL_ROOT / "site")
    ap.add_argument(
        "--limit", type=int, default=None,
        help="render only the first N genes (quick smoke build)",
    )
    args = ap.parse_args()

    store = Store()  # loads the committed parquet tables
    site: Path = args.out
    if site.exists():
        shutil.rmtree(site)
    api_dir = site / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    print("Writing search index...")
    records = write_genes_index(store, api_dir)
    print(f"  {len(records):,d} genes")

    print("Writing per-gene figure bundles...")
    n = write_gene_bundles(store, records, api_dir, args.limit)
    print(f"  {n:,d} bundles")

    print("Writing volcano figures...")
    write_volcano(store, api_dir)

    print("Writing bulk-download bundle...")
    write_downloads(store, site, api_dir)

    print("Writing static shell...")
    write_shell(site)

    print(f"\nStatic site written to {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
