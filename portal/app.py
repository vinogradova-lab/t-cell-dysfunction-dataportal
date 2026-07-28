"""Flask app for the T cell dysfunction proteomics data portal.

Read-only JSON API + a single-page frontend. Figures are built server-side with
Plotly and shipped as JSON for the browser to render.

Routes
------
GET /                                   the single-page app
GET /api/search?q=<query>               autocomplete over symbol/uniprot/alias
GET /api/gene/<symbol>                  metadata + which modalities have data
GET /api/gene/<symbol>/<modality>       Plotly figure JSON (204 if no data)
GET /api/volcano/datasets               available volcano datasets (proteome / rna)
GET /api/volcano/<dataset>/comparisons  available comparisons for a dataset
GET /api/volcano/<dataset>/<comparison> volcano figure JSON (?highlight=<symbol>)
GET /api/downloads                      bulk-download manifest
GET /downloads/<file>                   bulk-download files (nginx serves these
                                        in production; this is a dev fallback)
"""

from __future__ import annotations

from functools import lru_cache

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from flask import render_template

from figures import BUILDERS, VOLCANO_BUILDERS
from store import DOWNLOAD_DIR, MODALITIES, get_store

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify(get_store().search(q))


@app.route("/api/gene/<symbol>")
def api_gene(symbol: str):
    meta = get_store().gene_meta(symbol)
    if meta is None:
        abort(404, description=f"unknown gene/protein: {symbol}")
    return jsonify(meta)


@app.route("/api/gene/<symbol>/<modality>")
def api_gene_modality(symbol: str, modality: str):
    if modality not in MODALITIES:
        abort(404, description=f"unknown modality: {modality}")
    store = get_store()
    if store.gene_meta(symbol) is None:
        abort(404, description=f"unknown gene/protein: {symbol}")
    df = store.slice(modality, symbol)
    if df.is_empty():
        # no data for this gene in this modality — frontend hides the section
        return Response(status=204)
    fig = BUILDERS[modality](df, symbol, store.replicates(modality, symbol))
    return Response(fig.to_json(), mimetype="application/json")


# ---- volcanoes (dataset-wide) ----------------------------------------- #
@app.route("/api/volcano/datasets")
def api_volcano_datasets():
    return jsonify(get_store().volcano_datasets())


@app.route("/api/volcano/<dataset>/comparisons")
def api_volcano_comparisons(dataset: str):
    comparisons = get_store().volcano_comparisons(dataset)
    if not comparisons:
        abort(404, description=f"unknown volcano dataset: {dataset}")
    return jsonify(comparisons)


@lru_cache(maxsize=16)
def _volcano_json(dataset: str, comparison: str) -> str | None:
    """Base (un-highlighted) volcano figure JSON, cached — it's static."""
    builder = VOLCANO_BUILDERS.get(dataset)
    if builder is None:
        return None
    df = get_store().volcano_slice(comparison, dataset)
    if df.is_empty():
        return None
    return builder(df, comparison).to_json()


@app.route("/api/volcano/<dataset>/<comparison>")
def api_volcano(dataset: str, comparison: str):
    store = get_store()
    builder = VOLCANO_BUILDERS.get(dataset)
    highlight = request.args.get("highlight")
    if builder is not None and highlight:
        # highlighted variant is cheap and gene-specific, so build fresh
        df = store.volcano_slice(comparison, dataset)
        if not df.is_empty():
            fig = builder(df, comparison, highlight=highlight)
            return Response(fig.to_json(), mimetype="application/json")
        payload = None
    else:
        payload = _volcano_json(dataset, comparison)
    if payload is None:
        abort(404, description=f"unknown volcano: {dataset}/{comparison}")
    return Response(payload, mimetype="application/json")


# ---- bulk download ---------------------------------------------------- #
_DOWNLOAD_ITEMS = [
    ("Whole proteome (expression, replicates + significance)", "whole_proteome"),
    ("Bulk RNA-seq", "rna"),
    ("Cysteine reactivity (5 conditions)", "reactivity_5cond"),
    ("Cysteine reactivity, ATP add-back", "reactivity_atp"),
    ("Polar metabolomics", "metabolomics"),
]


@app.route("/api/downloads")
def api_downloads():
    manifest = []
    # combined bundle first
    zip_path = DOWNLOAD_DIR / "t_cell_dysfunction_proteomics.zip"
    if zip_path.exists():
        manifest.append(
            {
                "label": "All data (combined)",
                "format": "zip",
                "href": "/downloads/t_cell_dysfunction_proteomics.zip",
                "bytes": zip_path.stat().st_size,
            }
        )
    for label, base in _DOWNLOAD_ITEMS:
        path = DOWNLOAD_DIR / f"{base}.csv"
        if path.exists():
            manifest.append(
                {
                    "label": label,
                    "format": "csv",
                    "href": f"/downloads/{base}.csv",
                    "bytes": path.stat().st_size,
                }
            )
    return jsonify(manifest)


@app.route("/downloads/<path:filename>")
def downloads(filename: str):
    # In production nginx serves /downloads/ directly; this is a dev fallback.
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)


@app.errorhandler(404)
def not_found(err):
    return jsonify(error=str(getattr(err, "description", "not found"))), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
