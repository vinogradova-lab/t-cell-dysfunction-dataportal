# T cell dysfunction proteomics portal.
#
# Multi-stage build:
#   1) "builder" runs the ETL (source CSVs -> parquet + downloads) so the raw
#      68 MB inputs never ship in the final image.
#   2) the runtime image carries only the app + the compact parquet/downloads.
#
# The build context must already contain ./source/ (populated by
# `python scripts/sync_source.py`). See README.md.

# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder
WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY etl ./etl
COPY scripts ./scripts
COPY source ./source
RUN python etl/build_db.py

# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PARQUET_DIR=/app/data/parquet \
    DOWNLOAD_DIR=/app/data/downloads

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY portal ./portal
# baked-in, read-only data artifacts from the builder stage
COPY --from=builder /build/data/parquet ./data/parquet
COPY --from=builder /build/data/downloads ./data/downloads

# refresh the vendored Plotly.js from the *installed* plotly package so the
# browser bundle always matches the version that renders the figure JSON
RUN python - <<'PY'
import os, shutil, plotly
src = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
dst = "portal/static/vendor/plotly.min.js"
shutil.copyfile(src, dst)
print("vendored", src, "->", dst)
PY

EXPOSE 8000
# gunicorn serves the Flask app; workers each hold a read-only copy of the data
CMD ["gunicorn", "--chdir", "portal", "--bind", "0.0.0.0:8000", \
     "--workers", "3", "--timeout", "60", "app:app"]
