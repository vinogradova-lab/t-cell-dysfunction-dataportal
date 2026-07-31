# T cell Dysfunction Proteomics Data Portal

A public web portal for the proteomic data in the T cell dysfunction manuscript.
Search a gene/protein to view its **transcriptomic**, **whole-proteome**, and
**cysteine-reactivity** changes (including the ATP add-back per-cysteine
boxplots), plus a **bulk download** of all processed data.

The deployed portal is **static**: `build_db.py` precomputes columnar **parquet**,
`prerender.py` renders every figure to JSON, and GitHub Pages serves the result
as files. Charts are **Plotly.js** in the manuscript palette. A read-only Flask
API (polars-backed) exposes the same data for `curl` but serves no working UI.

## Data

Most of the portal is built from the manuscript's **published supplementary
workbooks**; the rest comes from the sibling analysis repo
[`t-cell-dysfunction-2026`](../t-cell-dysfunction-2026). This repo tracks none of
these inputs — `sync_source.py` stages them into `source/` on demand, globbing
for the newest `Data S*.xlsx` since the date-stamped names change between
revisions.

| Modality | Value shown | Source |
|---|---|---|
| Whole proteome | log₂FC vs D2 (+ per-replicate, volcano, p-values) | Data S2, sheet `S2-1` |
| Bulk RNA-seq | log₂FC vs D2 + lfcSE + padj | Data S1, sheet `S1-1` |
| RNA D8C vs D8A | log₂FC + padj (volcano only) | Data S2, sheet `S2-2` |
| RNA replicate overlays | raw counts + median-of-ratios size factors | analysis repo |
| Reactivity (5 cond) | log₂FC vs D2 per cysteine | analysis repo |
| Reactivity ATP add-back | log₂FC vs D2, per replicate | analysis repo |
| Polar metabolomics | channel ratios + per-comparison DE (download only) | Data S3, sheet `S3-1` |

### Fold-change conventions

Fold changes are **log₂ relative to D2** unless a column or comparison names its
own reference (`D8C_vs_D8A` and the metabolomics `A_vs_B` columns are
numerator-first).

`D8C_vs_D8A` is the easiest thing here to get backwards: its two sides come from
sheets that label comparisons in **opposite directions**. The proteome's is
S2-1's `D8A vs. D8C` block, whose labels are reversed relative to its values (it
holds log₂(D8C/D8A), the same convention that makes `D2 vs. D8C` the D8C panel).
The transcriptome's is S2-2 `D8C vs D8A RNA_*`, numerator-first and read as
written; S1-1 has only the four vs-D2 contrasts.

## Statistics

### Protein abundance

A protein's abundance bar *is* its volcano x-position: both read S2-1's published
`log2_FC` from one column (`proteome.published_log2fc` = `volcano.log2fc`), so the
two cannot drift apart. AFAP1L2 D8C, the one cell with no published value, draws a
**gap**, not a zero.

Two other proteome fold changes exist and are **not** interchangeable with it.
`proteome.log2fc` — log₂ of the mean donor ratio — is portal-internal, appears in
no published artifact, and averages ratios outside the log, which biases it upward
by Jensen's inequality on every cell, increasingly with effect size (+0.33 log₂ at
the largest). The reactivity dot plot's expression triangles are the manuscript's
own aggregate, the **median over technical channels**; they land ~0.035 log₂ from
the bar, and up to 1.8 (HBB D8C).

The replicate dots over the bar are supporting evidence, not what sets it: **per
technical channel**, two per donor, each over its own donor's D2 mean, so they show
measurement spread rather than donor means. Their mean tracks the published value
to a median of 0.005 log₂.

A TMT channel reading exactly 0 means *below detection*, not *absent*: it is
censored at the replicate's limit of detection (1st percentile of its positive
census values), so the value is a bound — upper on the loss when the condition
channel was empty, lower on the increase when the D2 reference was. Channels empty
on both sides are dropped, and `n_reps` records the surviving donors (AFAP1L2 rests
on one). Censoring applies at whichever level does the collapsing, so a channel is
floored on its own in the median form but first diluted through its donor mean in
the other.

This is why a few bars sit off their own dots: a floored point cannot fall below
the limit of detection, while the published estimate used the raw zero and can.
F13A1 D8A draws −4.08 over channels spanning −2.93…−2.63; F13A1, HBB and TNFRSF4
are the whole list, annotated from the `censored` flag on `proteome_replicates`.

The volcano further **omits** any comparison whose reference channel was below
detection, since dividing by a mean that averaged in a raw zero inflates the
published fold change (AFAP1L2 D8C by ~1.19 log₂); those proteins keep their
per-gene view with the caveat annotated.

### Volcano significance

The two volcanoes call significance differently, on purpose. The whole-proteome
one reproduces the manuscript's `Regulation` column (raw p < 0.05, |log₂FC| ≥
log₂(1.5)). The transcriptome one uses DESeq2's adjusted p < 0.05 with |log₂FC| ≥
log₂(2) — the guides the per-gene RNA bars draw — over the ~6.1k genes that also
have whole-proteome data, so both cover one gene universe. That `padj` is the
transcriptome-wide fit, not recomputed over the subset.

### RNA bars vs. replicate points

A gene's RNA bar height *is* its volcano x-position (both S1-1's
`log2FoldChange`) and its error bar is that same coefficient's `lfcSE`, so bar
and interval are one published quantity. `lfcSE` comes from the negative-binomial
GLM with dispersion shrunk toward a transcriptome-wide trend; it does **not**
fall to zero with sequencing depth, being dominated at high counts by biological
overdispersion.

The replicate points over the bar are separate supporting evidence, from raw
counts normalized with median-of-ratios size factors (`build_db._size_factors` —
the ETL takes no R dependency and reproduces only that step). They track the bar
to a median of 0.002 log₂ above baseMean 100. **Do not read their spread as the
error bar**: the SEM of three treatment samples about their own mean ignores the
D2 reference's own uncertainty and runs ~1.9× under `lfcSE`.

Whole-proteome bars *do* carry a replicate SEM, S2-1 publishing no standard error
to use instead. It avoids the problem above — those points are already per-donor
ratios, so the reference sits inside every one — but read it as the spread of the
plotted channels, **not** as the bar's uncertainty: pooling a donor's two
technical channels as independent makes it narrower than a donor-level SEM by at
least √2.

## Build & run

`data/parquet/` is tracked with **Git LFS**, so install it before cloning
(`brew install git-lfs && git lfs install`) — otherwise the checkout leaves
pointer files where the tables should be and `Store()` refuses to start. In an
existing clone, `git lfs install --local && git lfs pull` is enough.

```bash
# 1. Stage inputs (once, and whenever upstream data changes)
python scripts/sync_source.py                 # from ../t-cell-dysfunction-2026
python scripts/sync_source.py --source-repo /path/to/repo   # or elsewhere

# 2. ETL: source workbooks -> parquet + bulk downloads
pip install -r requirements.txt
python etl/build_db.py                         # source/ -> data/parquet + data/downloads

# 3. Pre-render the site, then serve it
python etl/prerender.py --out site --limit 200  # drop --limit for all ~17.5k genes
cd site && python -m http.server 8099           # http://127.0.0.1:8099
```

Building and serving that tree is the **only** way to preview the UI: the
frontend fetches pre-rendered JSON under `site/api/` and never talks to a live
server. `python portal/app.py` and `docker compose up --build` run the JSON API
alone — the SPA they hand out requests `api/genes.json` and friends, which have
no Flask route behind them, so the page loads empty. Use them with `curl`;
GitHub Pages, not Docker, is the real deployment.

`--limit N` renders only the first N per-gene bundles (~13 s versus ~10 min).
The volcanoes, search index and downloads stay complete, but a gene outside the
first N is found by search and then fails to load its charts. Re-run
`prerender.py` after any change to `figures.py`, `store.py` or the parquet — the
served tree is a build artifact, not live code.

## API

| Route | Purpose |
|---|---|
| `GET /api/search?q=<query>` | autocomplete over symbol / UniProt / alias |
| `GET /api/gene/<symbol>` | metadata + which modalities have data |
| `GET /api/gene/<symbol>/<modality>` | Plotly figure JSON (`204` if no data) |
| `GET /api/volcano/datasets` | volcano datasets (`proteome`, `rna`) |
| `GET /api/volcano/<dataset>/comparisons` | that dataset's comparisons |
| `GET /api/volcano/<dataset>/<comparison>` | volcano figure JSON (`?highlight=<symbol>` to pin a gene) |
| `GET /api/downloads` | bulk-download manifest |
| `GET /downloads/<file1>` | processed data files (CSV / ZIP) |

`<modality>` ∈ `proteome`, `rna`, `reactivity`, `reactivity_atp`.
`<comparison>` ∈ `D4A`, `D4C`, `D8A`, `D8C` (a bare code is that condition vs D2)
plus `D8C_vs_D8A`, day-8 chronic over day-8 acute; the figure's x-axis title
names the reference, so the picker's labels don't have to. Unlike the per-gene
routes the volcanoes are **dataset-wide** — every gene for one comparison, each
point clickable through to its per-gene view. `/?gene=MAP2K4` deep-links a gene.

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

`source/`, `data/downloads/` and `site/` are git-ignored — regenerate them with
the steps above. **`data/parquet/` is committed**: CI has no `source/`, so
`prerender.py` builds the site from the tracked parquet. Any ETL change must land
its rebuilt parquet in the same commit, or Pages will keep serving the old data.
