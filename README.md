# T cell Dysfunction Proteomics Data Portal

A public web portal for the proteomic data in the T cell
dysfunction manuscript. Search a gene/protein to view its **transcriptomic**,
**whole-proteome**, and **cysteine-reactivity** changes (including the ATP
add-back per-cysteine boxplots) across the dysfunction conditions, plus a
**bulk download** of all processed data.

Flask + gunicorn API, interactive **Plotly.js** charts (manuscript palette),
nginx reverse proxy. Data is precomputed into columnar **parquet** at build time
and loaded in-memory (polars); the app is fully read-only.

## Data

Most of the portal is built directly from the manuscript's **published
supplementary workbooks** . The remaining tables come from the sibling analysis repo
[`t-cell-dysfunction-2026`](../t-cell-dysfunction-2026). This repo tracks none of
these inputs; `sync_source.py` stages them into `source/` on demand.

| Modality | Value shown | Source |
|---|---|---|
| Whole proteome | log₂FC vs D2 (+ per-replicate, volcano, p-values) | Data S2, sheet `S2-1` |
| Bulk RNA-seq | log₂FC vs D2 + padj | Data S1, sheet `S1-1` |
| RNA replicate overlays | VST-normalized counts | analysis repo |
| Reactivity (5 cond) | log₂FC vs D2 per cysteine | analysis repo |
| Reactivity ATP add-back | log₂FC vs D2, per replicate | analysis repo |

All fold changes are **log₂ fold-change relative to D2**. The date-stamped
`Data S1*.xlsx` / `Data S2*.xlsx` filenames change between revisions, so
`sync_source.py` globs for the newest.

Two different aggregates of protein expression are in play, and they are not
interchangeable: the whole-proteome tab reports the **mean over biological
replicates**, while the reactivity dot plot's expression triangles carry the
manuscript's **median over technical channels**. They differ by ~0.035 log₂ for
most proteins.

In the whole proteome, a TMT channel reading exactly 0 means *below detection*,
not *absent*: those channels are censored at the replicate's limit of detection
(1st percentile of its positive census values), so the value is a bound rather
than a point estimate. Channels where both the condition
and its D2 reference were empty carry no information and are dropped. `n_reps`
records how many donors backed each aggregate — one protein (AFAP1L2) rests
on a single usable donor.

The abundance bars overlay **per technical channel** (two per donor), so the
dots show measurement spread rather than a donor mean; the bar is the mean of
the dots shown. The volcano additionally **omits** any comparison whose D2
reference was below detection — the published fold change there divides by a
D2 mean that averaged in a raw zero, inflating it (AFAP1L2 D8C by ~1.19 log₂).
Those proteins keep their per-gene view, annotated with the caveat.

## Build & run

```bash
# 1. Stage inputs (once, and whenever upstream data changes)
python scripts/sync_source.py                 # from ../t-cell-dysfunction-2026
python scripts/sync_source.py --source-repo /path/to/repo   # or elsewhere

# 2a. Local (dev)
pip install -r requirements.txt
python etl/build_db.py                         # source/ -> data/parquet + data/downloads
python portal/app.py                           # http://127.0.0.1:5000

# 2b. Docker (production-like) — ETL runs in the image build, so run step 1 first
docker compose up --build                      # http://localhost:8080
```

Under Docker, `web` runs gunicorn (read-only); `nginx` reverse-proxies, gzips,
and fronts static assets and `/downloads/`.

## API

| Route | Purpose |
|---|---|
| `GET /api/search?q=<query>` | autocomplete over symbol / UniProt / alias |
| `GET /api/gene/<symbol>` | metadata + which modalities have data |
| `GET /api/gene/<symbol>/<modality>` | Plotly figure JSON (`204` if no data) |
| `GET /api/volcano/comparisons` | whole-proteome volcano comparisons (vs D2) |
| `GET /api/volcano/<comparison>` | volcano figure JSON (`?highlight=<symbol>` to pin a protein) |
| `GET /api/downloads` | bulk-download manifest |
| `GET /downloads/<file>` | processed data files (CSV / ZIP) |

`<modality>` ∈ `proteome`, `rna`, `reactivity`, `reactivity_atp`.
`<comparison>` ∈ `D4A`, `D4C`, `D8A`, `D8C` (each vs D2). Unlike the per-gene
routes, the volcano is **dataset-wide** — all proteins for one comparison, each
point clickable through to that protein's per-gene view.
Deep links: `/?gene=MAP2K4` opens straight to a gene (shareable / citable).

## Layout

```
scripts/sync_source.py   stage supplementary workbooks + analysis inputs into source/
etl/build_db.py          source workbooks/CSVs -> parquet tables + bulk-download bundle
etl/gene_index.py        search registry (symbols/uniprot/aliases)
portal/app.py            Flask routes
portal/store.py          in-memory parquet store + queries
portal/palette.py        colors/ordering ported from the manuscript figure code
portal/figures.py        Plotly figure builders (one per modality)
portal/templates,static  single-page frontend (+ vendored plotly.min.js)
Dockerfile, docker-compose.yml, nginx/  deployment
```

`source/`, `data/parquet/`, and `data/downloads/` are git-ignored build
artifacts — regenerate them with the steps above.
