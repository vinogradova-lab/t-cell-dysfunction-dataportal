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
| RNA D8C vs D8A | log₂FC + padj (volcano only) | Data S2, sheet `S2-2` |
| RNA replicate overlays | VST-normalized counts | analysis repo |
| Reactivity (5 cond) | log₂FC vs D2 per cysteine | analysis repo |
| Reactivity ATP add-back | log₂FC vs D2, per replicate | analysis repo |
| Polar metabolomics | channel ratios + per-comparison DE (download only) | Data S3, sheet `S3-1` |

Fold changes are **log₂ fold-change relative to D2** unless a column or
comparison names its own reference (`D8C_vs_D8A`, and the metabolomics
`A_vs_B` columns, are numerator-first). The date-stamped
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

The deployed portal is **static** — the frontend in `portal/static/js/app.js`
talks only to the pre-rendered JSON under `site/api/`, never to a live server.
So previewing it locally means building that tree and serving it as files.

```bash
# 1. Stage inputs (once, and whenever upstream data changes)
python scripts/sync_source.py                 # from ../t-cell-dysfunction-2026
python scripts/sync_source.py --source-repo /path/to/repo   # or elsewhere

# 2. ETL: source workbooks -> parquet + bulk downloads
pip install -r requirements.txt
python etl/build_db.py                         # source/ -> data/parquet + data/downloads

# 3. Pre-render the site, then serve it
python etl/prerender.py --out site --limit 200  # drop --limit for the full ~17.5k genes
cd site && python -m http.server 8099           # http://127.0.0.1:8099
```

`--limit N` renders only the first N per-gene figure bundles, turning a
~10-minute full build into ~13 seconds. Everything else is still complete — the
volcanoes, the full search index, and the bulk downloads all work. Searching a
gene outside the first N finds it (the index covers all ~17.5k) but its charts
fail with "Could not load data for this gene", since only its bundle is missing.
Re-run `prerender.py` to pick up any change to `figures.py`, `store.py`, or the
parquet: the served tree is a build artifact, not live code.

`python portal/app.py` (http://127.0.0.1:5000) still runs the Flask JSON API
below, but **it does not serve a working UI**: the SPA it hands out requests
`api/genes.json`, `api/downloads.json` and friends, which are prerender outputs
with no Flask route behind them, so the page loads empty. Use it to exercise the
API with `curl`, not to look at the portal. The Docker stack (`docker compose up
--build`, http://localhost:8080) runs that same Flask app behind gunicorn/nginx
and so has the same limitation; GitHub Pages, not Docker, is what actually
serves this portal.

## API

Consumed by `curl`, not by the frontend — the browser reads the pre-rendered
equivalents under `site/api/` (see Build & run).

| Route | Purpose |
|---|---|
| `GET /api/search?q=<query>` | autocomplete over symbol / UniProt / alias |
| `GET /api/gene/<symbol>` | metadata + which modalities have data |
| `GET /api/gene/<symbol>/<modality>` | Plotly figure JSON (`204` if no data) |
| `GET /api/volcano/datasets` | volcano datasets (`proteome`, `rna`) |
| `GET /api/volcano/<dataset>/comparisons` | that dataset's comparisons |
| `GET /api/volcano/<dataset>/<comparison>` | volcano figure JSON (`?highlight=<symbol>` to pin a gene) |
| `GET /api/downloads` | bulk-download manifest |
| `GET /downloads/<file>` | processed data files (CSV / ZIP) |

`<modality>` ∈ `proteome`, `rna`, `reactivity`, `reactivity_atp`.
`<comparison>` ∈ `D4A`, `D4C`, `D8A`, `D8C` (a bare code is that condition vs
D2) plus `D8C_vs_D8A`, day-8 chronic over day-8 acute. The figure's x-axis title
names the reference, so the picker's labels don't have to. Unlike the per-gene
routes, the volcanoes are **dataset-wide** — every gene for one comparison, each
point clickable through to that gene's per-gene view.
Deep links: `/?gene=MAP2K4` opens straight to a gene (shareable / citable).

The two volcanoes call significance differently, on purpose. The whole-proteome
one reproduces the manuscript's published `Regulation` column: raw p < 0.05 with
|log₂FC| ≥ log₂(1.5). The transcriptome one uses DESeq2's adjusted p < 0.05 with
|log₂FC| ≥ log₂(2) — the same 2-fold guides the per-gene RNA bar chart draws —
and is restricted to the ~6.2k genes that also have whole-proteome data, so the
two plots cover one gene universe. That `padj` comes from the transcriptome-wide
fit; it is not recomputed over the matched subset.

The `D8C_vs_D8A` panel is published data too, but from different sheets on the
two sides, and they label comparisons in **opposite directions** — the single
easiest thing to get backwards here. The proteome's comes from S2-1's
`D8A vs. D8C` block, whose labels are reversed relative to its values (that
block is log₂(D8C/D8A), the same convention that makes `D2 vs. D8C` the D8C
panel). The transcriptome's comes from S2-2 `D8C vs D8A RNA_*`, which is
numerator-first and read as written; S1-1 publishes only the four vs-D2
contrasts, so it cannot supply this one.

## Layout

```
scripts/sync_source.py   stage supplementary workbooks + analysis inputs into source/
etl/build_db.py          source workbooks/CSVs -> parquet tables + bulk-download bundle
etl/gene_index.py        search registry (symbols/uniprot/aliases)
etl/prerender.py         parquet -> static site/ tree (what CI deploys, and what
                         the frontend actually reads)
portal/app.py            Flask JSON API (curl-only; does not serve a working UI)
portal/store.py          in-memory parquet store + queries
portal/palette.py        colors/ordering ported from the manuscript figure code
portal/figures.py        Plotly figure builders (one per modality + the volcanoes)
portal/templates,static  single-page frontend (+ vendored plotly.min.js)
.github/workflows/pages.yml  prerender + deploy to GitHub Pages
Dockerfile, docker-compose.yml, nginx/  legacy Flask deployment (see Build & run)
```

`source/`, `data/parquet/`, and `data/downloads/` are git-ignored build
artifacts — regenerate them with the steps above.
