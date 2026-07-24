"use strict";

// Static build: this SPA is a thin Plotly renderer over pre-rendered JSON
// emitted by etl/prerender.py. All fetches are RELATIVE (api/...), so the site
// works at a project-page subpath now and at a custom domain later with no code
// change. Search, gene metadata, and volcano highlighting are all client-side.

const MODALITY_ORDER = ["rna", "proteome", "reactivity", "reactivity_atp"];
const MODALITY_LABEL = {
  rna: "Transcriptomics (bulk RNA-seq)",
  proteome: "Whole proteome",
  reactivity: "Cysteine reactivity (vs D2)",
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

const searchInput = document.getElementById("search");
const suggestionsEl = document.getElementById("suggestions");
const geneView = document.getElementById("gene-view");
const geneHeader = document.getElementById("gene-header");
const chartsEl = document.getElementById("charts");
const volcanoSelect = document.getElementById("volcano-comparison");
const volcanoPlot = document.getElementById("volcano-plot");

let currentGene = null;
let currentComparison = null;

let activeIndex = -1;
let currentSuggestions = [];

// ---- search index (loaded once) -------------------------------------- //
let GENES = []; // [{symbol, uniprot, description, aliases, modalities, key, _s, _u, _a}]
let GENE_BY_SYMBOL = new Map(); // lowercased symbol -> record

async function loadGenes() {
  const res = await fetch("api/genes.json");
  const raw = await res.json();
  GENES = raw.map((g) => ({
    ...g,
    _s: g.symbol.toLowerCase(),
    _u: (g.uniprot || "").toLowerCase(),
    _a: (g.aliases || "").toLowerCase(),
  }));
  GENE_BY_SYMBOL = new Map(GENES.map((g) => [g._s, g]));
}

// port of store.py Store.search: exact > prefix > uniprot-prefix > alias/substring
function searchGenes(query, limit = 15) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const seen = new Set();
  const out = [];
  const push = (g) => {
    if (seen.has(g.symbol)) return;
    seen.add(g.symbol);
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
    if (pick) selectGene(pick.symbol);
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
      `<span class="s-symbol">${s.symbol}</span>` +
      `<span class="s-desc">${escapeHtml(s.description || "")}</span>` +
      `<span class="s-badges">${mods} dataset${mods === 1 ? "" : "s"}</span>`;
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      selectGene(s.symbol);
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
async function selectGene(symbol, updateUrl = true) {
  hideSuggestions();
  searchInput.value = symbol;
  if (symbol !== currentGene) {
    currentGene = symbol;
    refreshVolcanoHighlight();
  }
  if (updateUrl) {
    // shareable / journal-linkable deep link, e.g. ?gene=MAP2K4
    const url = new URL(window.location);
    url.searchParams.set("gene", symbol);
    history.replaceState(null, "", url);
  }
  geneView.hidden = false;
  geneHeader.innerHTML = `<h2>${escapeHtml(symbol)}</h2><p class="muted">Loading…</p>`;
  chartsEl.innerHTML = "";
  geneView.scrollIntoView({ behavior: "smooth", block: "start" });

  const meta = GENE_BY_SYMBOL.get(symbol.toLowerCase());
  if (!meta) {
    geneHeader.innerHTML =
      `<h2>${escapeHtml(symbol)}</h2><p class="error">No data found for “${escapeHtml(symbol)}”.</p>`;
    return;
  }

  const uni = meta.uniprot
    ? ` · UniProt <a href="https://www.uniprot.org/uniprotkb/${meta.uniprot}" target="_blank" rel="noopener">${meta.uniprot}</a>`
    : "";
  geneHeader.innerHTML =
    `<h2>${escapeHtml(meta.symbol)}</h2>` +
    `<p class="gene-desc">${escapeHtml(meta.description || "")}</p>` +
    `<p class="muted">${meta.modalities.length} dataset(s) available${uni}</p>`;

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
    title.textContent = `${MODALITY_LABEL[m]} — ${symbol}`;
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
      toImageButtonOptions: { format: "svg", filename: `${symbol}_${m}` },
    });
  }
}

// ---- whole-proteome volcano (dataset-wide) --------------------------- //
const volcanoCache = {}; // comparison -> {data, layout}

async function initVolcano() {
  if (!volcanoSelect || !volcanoPlot) return;
  let comparisons;
  try {
    const res = await fetch("api/volcano/comparisons.json");
    comparisons = await res.json();
  } catch (err) {
    comparisons = [];
  }
  if (!comparisons.length) {
    document.getElementById("volcano-section")?.remove();
    return;
  }
  volcanoSelect.innerHTML = "";
  comparisons.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.label} (${c.id}) vs D2`;
    volcanoSelect.appendChild(opt);
  });
  volcanoSelect.addEventListener("change", () => loadVolcano(volcanoSelect.value));
  // default to the most dysfunctional comparison if present, else the first
  const preferred = comparisons.find((c) => c.id === "D8C") || comparisons[0];
  volcanoSelect.value = preferred.id;
  loadVolcano(preferred.id);
}

let volcanoLoadSeq = 0;

async function loadVolcano(comparison) {
  currentComparison = comparison;
  const seq = ++volcanoLoadSeq;
  let fig = volcanoCache[comparison];
  if (!fig) {
    try {
      const res = await fetch(`api/volcano/${comparison}.json`);
      if (!res.ok) throw new Error("volcano error");
      fig = await res.json();
      volcanoCache[comparison] = fig;
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
    toImageButtonOptions: { format: "svg", filename: `volcano_${comparison}` },
  });
  if (currentGene) pinVolcano(fig, currentGene);
  // click a point -> open that protein's per-gene view
  volcanoPlot.on("plotly_click", (ev) => {
    const pt = ev.points && ev.points[0];
    const sym = pt && pt.customdata && pt.customdata[0];
    if (sym) selectGene(sym);
  });
}

// client-side port of the highlight block in figures.py volcano_figure:
// find the gene in the cached point cloud, then draw a pinned marker + label.
function pinVolcano(fig, symbol) {
  let hx, hy, hp;
  for (const tr of fig.data) {
    if (!tr.customdata || !tr.x) continue;
    for (let j = 0; j < tr.customdata.length; j++) {
      if (tr.customdata[j] && tr.customdata[j][0] === symbol) {
        hx = tr.x[j];
        hy = tr.y[j];
        hp = tr.customdata[j][1];
        break;
      }
    }
    if (hx !== undefined) break;
  }
  if (hx === undefined) return; // gene not in this comparison
  Plotly.addTraces(volcanoPlot, {
    type: "scattergl",
    name: symbol,
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
      `<b>${symbol}</b><br>log2FC = %{x:.2f}<br>p = ${fmtSig(hp)}<extra></extra>`,
    showlegend: false,
  });
  Plotly.relayout(volcanoPlot, {
    annotations: [
      {
        x: hx,
        y: hy,
        text: symbol,
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
  if (currentComparison) loadVolcano(currentComparison);
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
  // deep link support: ?gene=MAP2K4 auto-selects on load
  const initialGene = new URLSearchParams(window.location.search).get("gene");
  if (initialGene) selectGene(initialGene, false);
}

main();
