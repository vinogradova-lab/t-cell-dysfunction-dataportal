"use strict";

// Static build: this SPA is a thin Plotly renderer over pre-rendered JSON
// emitted by etl/prerender.py. All fetches are RELATIVE (api/...), so the site
// works at a project-page subpath now and at a custom domain later with no code
// change. Search, gene metadata, and volcano highlighting are all client-side.

const MODALITY_ORDER = ["rna", "proteome", "reactivity", "reactivity_atp"];
const MODALITY_LABEL = {
  rna: "Transcriptomics (bulk RNA-seq)",
  proteome: "Whole proteome",
  reactivity: "Cysteine reactivity",
  reactivity_atp: "Cysteine reactivity — ATP add-back",
};

// volcano highlight styling — mirrors portal/palette.py (VOLCANO_HIGHLIGHT,
// FONT_FAMILY) so the client-side pin matches the server oracle.
const VOLCANO_HIGHLIGHT = "#111111";
const FONT_FAMILY = "Arial, Helvetica, sans-serif";

const PLOT_OPTS = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

// per-dataset headline + caption for the volcano section. Kept next to the
// figure code rather than in index.html so the stated cutoffs can't drift away
// from the ones figures.py actually draws.
const VOLCANO_COPY = {
  proteome: {
    title: "Whole-proteome volcano",
    blurb:
      "Differential protein abundance across the proteome, for the comparison " +
      "named on the x axis. Each point is a protein; click one to open its " +
      "per-gene view. Dashed guides mark " +
      "±1.5-fold and p = 0.05.",
  },
  rna: {
    title: "Transcriptome volcano",
    blurb:
      "Differential mRNA abundance for the comparison named on the x axis, " +
      "restricted to the ~6.2k genes that also have whole-proteome data. " +
      "Each point is a gene; click one to open its per-gene view. " +
      "Dashed guides mark ±2-fold and adjusted p = 0.05.",
  },
};

const searchInput = document.getElementById("search");
const suggestionsEl = document.getElementById("suggestions");
const geneView = document.getElementById("gene-view");
const geneHeader = document.getElementById("gene-header");
const chartsEl = document.getElementById("charts");
const volcanoDatasetSelect = document.getElementById("volcano-dataset");
const volcanoSelect = document.getElementById("volcano-comparison");
const volcanoPlot = document.getElementById("volcano-plot");
const volcanoTitle = document.getElementById("volcano-title");
const volcanoBlurb = document.getElementById("volcano-blurb");

let currentEntry = null; // the resolved record for the open page
let currentComparison = null;
let currentDataset = null;

let activeIndex = -1;
let currentSuggestions = [];

// ---- search index (loaded once) -------------------------------------- //
// One record per PROTEIN, not per gene symbol. Five symbols were measured under
// more than one UniProt accession (TMPO, MOCS2, MIEF1, CDKN2A, POLR1D) and get
// one record each, with id "TMPO.P42166" and label "TMPO (P42166)". prerender
// omits id/label wherever they equal the symbol, so both default here.
let GENES = [];
let GENE_BY_KEY = new Map(); // lowercased id AND label -> record
let GENE_BY_SYMBOL = new Map(); // lowercased symbol -> that symbol's primary record
let GENE_SIBLINGS = new Map(); // lowercased symbol -> records, only where > 1

async function loadGenes() {
  const res = await fetch("api/genes.json");
  const raw = await res.json();
  GENES = raw.map((g) => ({
    ...g,
    id: g.id || g.symbol,
    label: g.label || g.symbol,
    _s: g.symbol.toLowerCase(),
    _u: (g.uniprot || "").toLowerCase(),
    _a: (g.aliases || "").toLowerCase(),
  }));
  GENE_BY_KEY = new Map();
  GENE_BY_SYMBOL = new Map();
  const bySymbol = new Map();
  for (const g of GENES) {
    GENE_BY_KEY.set(g.id.toLowerCase(), g);
    GENE_BY_KEY.set(g.label.toLowerCase(), g);
    // lowest accession wins, matching store.Store._primary — so a bare symbol
    // resolves the same way on both sides
    const prev = GENE_BY_SYMBOL.get(g._s);
    if (!prev || (g.uniprot || "") < (prev.uniprot || "")) {
      GENE_BY_SYMBOL.set(g._s, g);
    }
    if (!bySymbol.has(g._s)) bySymbol.set(g._s, []);
    bySymbol.get(g._s).push(g);
  }
  GENE_SIBLINGS = new Map(
    [...bySymbol].filter(([, list]) => list.length > 1)
  );
}

// Resolve anything that names a protein: an entry id, a display label, or a
// bare symbol (an old ?gene= link, or a click on the transcript-level RNA
// volcano, whose points are symbols).
function resolveGene(name) {
  if (!name) return null;
  const k = String(name).toLowerCase();
  return GENE_BY_KEY.get(k) || GENE_BY_SYMBOL.get(k) || null;
}

// port of store.py Store.search: exact > prefix > uniprot-prefix > alias/substring
function searchGenes(query, limit = 15) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const seen = new Set();
  const out = [];
  // dedup on entry id, not symbol: "TMPO" must offer both of its proteins,
  // while "P42167" offers only that one
  const push = (g) => {
    if (seen.has(g.id)) return;
    seen.add(g.id);
    out.push(g);
  };
  const exact = GENES.filter((g) => g._s === q);
  const prefix = GENES.filter((g) => g._s.startsWith(q) && g._s !== q);
  const uni = GENES.filter((g) => g._u.startsWith(q));
  const alias = GENES.filter((g) => g._a.includes(q) || g._s.includes(q));
  for (const frame of [exact, prefix, uni, alias]) {
    for (const g of frame) {
      push(g);
      if (out.length >= limit) return out;
    }
  }
  return out;
}

// ---- search / autocomplete ------------------------------------------- //
searchInput.addEventListener("input", () => {
  const q = searchInput.value.trim();
  if (!q) return hideSuggestions();
  currentSuggestions = searchGenes(q);
  activeIndex = currentSuggestions.length ? 0 : -1;
  renderSuggestions();
});

searchInput.addEventListener("keydown", (e) => {
  if (suggestionsEl.hidden) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeIndex = Math.min(activeIndex + 1, currentSuggestions.length - 1);
    renderSuggestions();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeIndex = Math.max(activeIndex - 1, 0);
    renderSuggestions();
  } else if (e.key === "Enter") {
    e.preventDefault();
    const pick = currentSuggestions[activeIndex] || currentSuggestions[0];
    if (pick) selectGene(pick.id);
  } else if (e.key === "Escape") {
    hideSuggestions();
  }
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-box")) hideSuggestions();
});

document.querySelectorAll(".example").forEach((btn) =>
  btn.addEventListener("click", () => {
    searchInput.value = btn.dataset.gene;
    selectGene(btn.dataset.gene);
  })
);

function renderSuggestions() {
  if (!currentSuggestions.length) return hideSuggestions();
  suggestionsEl.innerHTML = "";
  currentSuggestions.forEach((s, i) => {
    const li = document.createElement("li");
    li.className = "suggestion" + (i === activeIndex ? " active" : "");
    const mods = s.modalities.length;
    li.innerHTML =
      `<span class="s-symbol">${escapeHtml(s.label)}</span>` +
      `<span class="s-desc">${escapeHtml(s.description || "")}</span>` +
      `<span class="s-badges">${mods} dataset${mods === 1 ? "" : "s"}</span>`;
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      selectGene(s.id);
    });
    suggestionsEl.appendChild(li);
  });
  suggestionsEl.hidden = false;
}

function hideSuggestions() {
  suggestionsEl.hidden = true;
  suggestionsEl.innerHTML = "";
  currentSuggestions = [];
  activeIndex = -1;
}

// ---- gene view ------------------------------------------------------- //
// Where a symbol was measured under several accessions, each protein has its own
// page — so each one points at the others. Without this the sibling is only
// reachable by searching the symbol again, and for pairs that are really two
// halves of one story (TMPO's LAP2 isoforms, POLR1D's) that is easy to miss.
function siblingsHtml(meta) {
  const siblings = (GENE_SIBLINGS.get(meta._s) || []).filter(
    (g) => g.id !== meta.id
  );
  if (!siblings.length) return "";
  const links = siblings
    .map(
      (g) =>
        `<button class="sibling" data-gene="${escapeHtml(g.id)}">` +
        `${escapeHtml(g.label)}</button>` +
        (g.description ? ` — ${escapeHtml(g.description)}` : "")
    )
    .join("; ");
  return `<p class="muted gene-siblings">Also measured as ${links}</p>`;
}

// delegated so it survives the header being rewritten on every selection
geneHeader.addEventListener("click", (e) => {
  const btn = e.target.closest(".sibling");
  if (btn) selectGene(btn.dataset.gene);
});

async function selectGene(name, updateUrl = true) {
  hideSuggestions();
  // `name` may be an entry id, a display label, or a bare symbol
  const meta = resolveGene(name);
  const shown = meta ? meta.label : String(name);
  searchInput.value = shown;
  if (!meta || meta.id !== (currentEntry && currentEntry.id)) {
    currentEntry = meta;
    refreshVolcanoHighlight();
  }
  if (updateUrl) {
    // shareable / journal-linkable deep link, e.g. ?gene=MAP2K4
    const url = new URL(window.location);
    url.searchParams.set("gene", meta ? meta.id : name);
    history.replaceState(null, "", url);
  }
  geneView.hidden = false;
  geneHeader.innerHTML = `<h2>${escapeHtml(shown)}</h2><p class="muted">Loading…</p>`;
  chartsEl.innerHTML = "";
  geneView.scrollIntoView({ behavior: "smooth", block: "start" });

  if (!meta) {
    geneHeader.innerHTML =
      `<h2>${escapeHtml(shown)}</h2><p class="error">No data found for “${escapeHtml(shown)}”.</p>`;
    return;
  }

  const uni = meta.uniprot
    ? ` · UniProt <a href="https://www.uniprot.org/uniprotkb/${meta.uniprot}" target="_blank" rel="noopener">${meta.uniprot}</a>`
    : "";
  geneHeader.innerHTML =
    `<h2>${escapeHtml(meta.label)}</h2>` +
    `<p class="gene-desc">${escapeHtml(meta.description || "")}</p>` +
    `<p class="muted">detected in ${meta.modalities.length} dataset(s)${uni}</p>` +
    siblingsHtml(meta);

  const available = MODALITY_ORDER.filter((m) => meta.modalities.includes(m));
  if (!available.length) {
    chartsEl.innerHTML = `<p class="muted">No plottable data for this gene.</p>`;
    return;
  }

  // one fetch per gene: the bundle carries every modality (null where absent)
  let bundle;
  try {
    const res = await fetch(`api/gene/${meta.key}.json`);
    if (!res.ok) throw new Error("bundle error");
    bundle = await res.json();
  } catch (err) {
    chartsEl.innerHTML = `<p class="error">Could not load data for this gene.</p>`;
    return;
  }

  // UniProt function summary rides in the bundle (kept out of the always-loaded
  // search index because it's large). Pipe-delimited multi-entry text -> spaces.
  const fn = (bundle.uniprot_function || "").replace(/\|/g, " ").trim();
  if (fn) {
    const p = document.createElement("p");
    p.className = "gene-function";
    p.innerHTML = `<span class="fn-label">Function</span>`;
    const text = document.createElement("span");
    text.className = "fn-text clamped";
    text.textContent = fn;
    p.appendChild(text);
    geneHeader.appendChild(p);
    // only offer a toggle when the text actually overflows the clamp
    if (text.scrollHeight - text.clientHeight > 4) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "fn-toggle";
      btn.textContent = "Show more";
      btn.addEventListener("click", () => {
        const clamped = text.classList.toggle("clamped");
        btn.textContent = clamped ? "Show more" : "Show less";
      });
      p.appendChild(btn);
    }
  }

  // Two passes on purpose: build & append EVERY card first, then render. The
  // charts grid is `repeat(auto-fit, minmax(440px, 1fr))`, so a lone card fills
  // the whole row — rendering a plot inline (before its siblings exist) sizes it
  // to the full-width row, and Plotly's `responsive` only re-fits on window
  // resize, so the first plot stayed ~2x its final card and its right-hand bars
  // spilled behind the next card. Appending all cards first means each plot
  // measures its final multi-column track width. Do NOT collapse back into one loop.
  const pending = [];
  for (const m of available) {
    const fig = bundle[m];
    if (!fig) continue; // modality present in index but empty slice
    const card = document.createElement("div");
    card.className = "chart-card";
    const title = document.createElement("h3");
    title.className = "chart-title";
    title.textContent = `${MODALITY_LABEL[m]} — ${meta.label}`;
    card.appendChild(title);
    const plot = document.createElement("div");
    plot.className = "plot";
    plot.id = `plot-${m}`;
    card.appendChild(plot);
    chartsEl.appendChild(card);
    pending.push({ plot, fig, m });
  }
  for (const { plot, fig, m } of pending) {
    Plotly.newPlot(plot, fig.data, fig.layout, {
      ...PLOT_OPTS,
      toImageButtonOptions: { format: "svg", filename: `${meta.id}_${m}` },
    });
  }
}

// ---- volcanoes (dataset-wide) ---------------------------------------- //
const volcanoCache = {}; // "dataset:comparison" -> {data, layout}

async function initVolcano() {
  if (!volcanoSelect || !volcanoDatasetSelect || !volcanoPlot) return;
  let datasets;
  try {
    const res = await fetch("api/volcano/datasets.json");
    datasets = await res.json();
  } catch (err) {
    datasets = [];
  }
  if (!datasets.length) {
    document.getElementById("volcano-section")?.remove();
    return;
  }
  volcanoDatasetSelect.innerHTML = "";
  datasets.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d.id;
    opt.textContent = d.label;
    volcanoDatasetSelect.appendChild(opt);
  });
  // a single dataset needs no picker, but the section still works
  volcanoDatasetSelect.closest(".volcano-picker").hidden = datasets.length < 2;
  volcanoDatasetSelect.addEventListener("change", () =>
    selectVolcanoDataset(volcanoDatasetSelect.value)
  );
  volcanoSelect.addEventListener("change", () =>
    loadVolcano(currentDataset, volcanoSelect.value)
  );
  await selectVolcanoDataset(datasets[0].id);
}

// swap datasets: refill the comparison picker, keeping the current comparison
// selected when the new dataset also offers it (both cover the same four).
async function selectVolcanoDataset(dataset) {
  let comparisons;
  try {
    const res = await fetch(`api/volcano/${dataset}/comparisons.json`);
    if (!res.ok) throw new Error("comparisons error");
    comparisons = await res.json();
  } catch (err) {
    comparisons = [];
  }
  if (!comparisons.length) {
    volcanoPlot.innerHTML = `<p class="error">Could not load the volcano plot.</p>`;
    return;
  }
  currentDataset = dataset;
  const copy = VOLCANO_COPY[dataset];
  if (copy) {
    if (volcanoTitle) volcanoTitle.textContent = copy.title;
    if (volcanoBlurb) volcanoBlurb.textContent = copy.blurb;
  }
  const keep = comparisons.some((c) => c.id === currentComparison)
    ? currentComparison
    : null;
  volcanoSelect.innerHTML = "";
  comparisons.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    // a bare id ("D8C") is that condition vs D2; an id written
    // "<numerator>_vs_<denominator>" names its own reference, so appending
    // "vs D2" to it would read "…Chronic vs Acute (D8C_vs_D8A) vs D2".
    opt.textContent = c.id.includes("_vs_")
      ? `${c.label} (${c.id.replace("_vs_", " vs ")})`
      : `${c.label} (${c.id}) vs D2`;
    volcanoSelect.appendChild(opt);
  });
  // default to the most dysfunctional comparison if present, else the first
  const preferred =
    keep || (comparisons.find((c) => c.id === "D8C") || comparisons[0]).id;
  volcanoSelect.value = preferred;
  await loadVolcano(dataset, preferred);
}

let volcanoLoadSeq = 0;

async function loadVolcano(dataset, comparison) {
  currentDataset = dataset;
  currentComparison = comparison;
  const key = `${dataset}:${comparison}`;
  const seq = ++volcanoLoadSeq;
  let fig = volcanoCache[key];
  if (!fig) {
    try {
      const res = await fetch(`api/volcano/${dataset}/${comparison}.json`);
      if (!res.ok) throw new Error("volcano error");
      fig = await res.json();
      volcanoCache[key] = fig;
    } catch (err) {
      volcanoPlot.innerHTML = `<p class="error">Could not load the volcano plot.</p>`;
      return;
    }
  }
  // a newer load superseded this one — drop the stale render
  if (seq !== volcanoLoadSeq) return;
  volcanoPlot.innerHTML = "";
  // clone the cached figure so Plotly's in-place mutations don't corrupt it
  const data = JSON.parse(JSON.stringify(fig.data));
  const layout = JSON.parse(JSON.stringify(fig.layout));
  await Plotly.newPlot(volcanoPlot, data, layout, {
    ...PLOT_OPTS,
    toImageButtonOptions: {
      format: "svg",
      filename: `volcano_${dataset}_${comparison}`,
    },
  });
  if (currentEntry) pinVolcano(fig, currentEntry);
  // Click a point -> open that protein's page. customdata[0] is the entry label
  // on the proteome volcano (one point per accession) and the bare symbol on the
  // transcriptome one; selectGene resolves either.
  volcanoPlot.on("plotly_click", (ev) => {
    const pt = ev.points && ev.points[0];
    const name = pt && pt.customdata && pt.customdata[0];
    if (name) selectGene(name);
  });
}

// Client-side port of the highlight block in figures.py volcano_figure: find the
// protein in the cached point cloud, then draw a pinned marker + label.
//
// Matched on whichever key that dataset's points carry. The whole-proteome
// volcano has one point per UniProt accession and labels them "TMPO (P42166)",
// so an entry pins its own protein rather than whichever of the two came first.
// The transcriptome volcano is transcript-level and labels by symbol, where both
// of a symbol's entries legitimately share one point.
function pinVolcano(fig, entry) {
  const name = currentDataset === "rna" ? entry.symbol : entry.label;
  let hx, hy, hp;
  for (const tr of fig.data) {
    if (!tr.customdata || !tr.x) continue;
    for (let j = 0; j < tr.customdata.length; j++) {
      if (tr.customdata[j] && tr.customdata[j][0] === name) {
        hx = tr.x[j];
        hy = tr.y[j];
        hp = tr.customdata[j][1];
        break;
      }
    }
    if (hx !== undefined) break;
  }
  if (hx === undefined) return; // protein not in this comparison
  // customdata[1] is whichever p the dataset called significance with
  const pLabel = currentDataset === "rna" ? "padj" : "p";
  Plotly.addTraces(volcanoPlot, {
    type: "scattergl",
    name,
    x: [hx],
    y: [hy],
    mode: "markers",
    marker: {
      color: VOLCANO_HIGHLIGHT,
      size: 12,
      symbol: "circle",
      line: { width: 2.5, color: "#ffffff" },
    },
    hovertemplate:
      `<b>${name}</b><br>log2FC = %{x:.2f}` +
      `<br>${pLabel} = ${fmtSig(hp)}<extra></extra>`,
    showlegend: false,
  });
  // append, don't replace: relayout swaps the whole annotations array, and the
  // RNA volcano ships one from the server (the matched-genes caveat).
  Plotly.relayout(volcanoPlot, {
    annotations: [
      ...(fig.layout.annotations || []),
      {
        x: hx,
        y: hy,
        text: name,
        showarrow: false,
        yshift: 14,
        font: { size: 12, color: VOLCANO_HIGHLIGHT, family: FONT_FAMILY },
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: VOLCANO_HIGHLIGHT,
        borderwidth: 1,
        borderpad: 2,
      },
    ],
  });
}

// re-pin the volcano on the currently active gene without a full page change
function refreshVolcanoHighlight() {
  if (currentDataset && currentComparison) {
    loadVolcano(currentDataset, currentComparison);
  }
}

// ---- downloads panel ------------------------------------------------- //
async function loadDownloads() {
  try {
    const res = await fetch("api/downloads.json");
    const items = await res.json();
    const list = document.getElementById("downloads-list");
    list.innerHTML = "";
    items.forEach((it) => {
      const a = document.createElement("a");
      a.className = "download-item" + (it.format === "zip" ? " combined" : "");
      a.href = it.href;
      a.innerHTML =
        `<span class="d-label">${it.label}</span>` +
        `<span class="d-meta">${it.format.toUpperCase()} · ${fmtBytes(it.bytes)}</span>`;
      list.appendChild(a);
    });
  } catch (err) {
    /* leave panel empty on failure */
  }
}

// ---- utils ----------------------------------------------------------- //
function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// approximate Python's "%.2g" for the hover p-value
function fmtSig(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
  return Number(x).toPrecision(2).replace(/\.?0+(e|$)/, "$1");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---- startup --------------------------------------------------------- //
async function main() {
  try {
    await loadGenes();
  } catch (err) {
    geneHeader &&
      (geneHeader.innerHTML = `<p class="error">Could not load the search index.</p>`);
  }
  loadDownloads();
  initVolcano();
  // Deep link support: ?gene=MAP2K4 auto-selects on load. selectGene resolves
  // entry ids (?gene=TMPO.P42166), display labels and bare symbols alike, so
  // links predating the per-protein split still land somewhere sensible.
  const initialGene = new URLSearchParams(window.location.search).get("gene");
  if (initialGene) selectGene(initialGene, false);
}

main();
