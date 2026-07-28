"""Copy the raw inputs this portal needs from the analysis repo into ./source/.

The canonical data lives in the sibling analysis repo
(``t-cell-dysfunction-2026``). This portal is a separate repo, so before an ETL
run (local or Docker build) we stage just the handful of files the ETL reads
into ``source/`` — a self-contained, gitignored build context.

Usage:
    python scripts/sync_source.py [--source-repo /path/to/t-cell-dysfunction-2026]

Defaults to the sibling directory ``../t-cell-dysfunction-2026``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PORTAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPO = PORTAL_ROOT.parent / "t-cell-dysfunction-2026"

# Files the ETL reads. Keys are paths relative to the analysis repo root;
# values are the destination path relative to ./source/ in this repo.
#
# The whole-proteome (expression + per-replicate + volcano) and bulk-RNA DESeq2
# tables are now read straight from the published supplementary workbooks in
# ``supp_data/`` — the manuscript's source of truth — rather than from analysis
# intermediates. The date-stamped filenames change between manuscript revisions,
# so we glob for the current ``Data S1*.xlsx`` / ``Data S2*.xlsx`` below.
SOURCE_FILES = {
    # RNA per-sample VST-normalized counts (for replicate overlays; not in the
    # supplementary workbooks, so still staged from the analysis repo)
    "data/rna/counts/normalized_counts.txt": "rna_counts.txt",
    # cysteine reactivity, 5-condition long format (LFC precomputed)
    "data/reactivity/reactivity_changes/output/03_reactivity_vs_wp/"
    "rc_df_long_format.csv": "reactivity_5cond.csv",
    # cysteine reactivity ATP add-back, per-replicate long format
    "data/reactivity/atp_add_back/06_results/boxplots/"
    "formatted_reactivity_data.csv": "reactivity_atp.csv",
    # NCBI gene table for alias/description enrichment of the search index
    "bin/genes_ncbi_9606_proteincoding.py": "genes_ncbi_9606_proteincoding.py",
}

# Supplementary workbooks (glob pattern relative to the analysis repo -> dest
# name). Data S1 sheet "S1-1" holds the bulk-RNA DESeq2 results vs D2; Data S2
# sheet "S2-1" holds the whole-proteome per-replicate values + volcano/
# significance columns + functional-group flags, and sheet "S2-2" supplies the
# one RNA comparison S1-1 omits (D8C vs D8A); Data S3 sheet "S3-1" holds the polar
# metabolomics (the portal ships it as a bulk download only, so the workbook's
# other sheets — lipidomics and isotope tracing — are not read). Update the
# patterns if the manuscript renames the files.
SOURCE_GLOBS = {
    "supp_data/Data S1*.xlsx": "data_s1.xlsx",
    "supp_data/Data S2*.xlsx": "data_s2.xlsx",
    "supp_data/Data S3*.xlsx": "data_s3.xlsx",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-repo",
        type=Path,
        default=DEFAULT_SOURCE_REPO,
        help="Path to the t-cell-dysfunction-2026 analysis repo "
        f"(default: {DEFAULT_SOURCE_REPO}).",
    )
    args = ap.parse_args()

    src_repo: Path = args.source_repo.resolve()
    dest_dir = PORTAL_ROOT / "source"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not src_repo.exists():
        print(f"ERROR: source repo not found: {src_repo}", file=sys.stderr)
        return 1

    missing = []
    to_copy: list[tuple[Path, str]] = [
        (src_repo / rel_src, rel_dst) for rel_src, rel_dst in SOURCE_FILES.items()
    ]
    # resolve the supplementary workbooks by glob (date-stamped filenames)
    for pattern, rel_dst in SOURCE_GLOBS.items():
        matches = sorted(src_repo.glob(pattern))
        if matches:
            to_copy.append((matches[-1], rel_dst))  # newest by name
        else:
            missing.append(str(src_repo / pattern))

    for src, rel_dst in to_copy:
        dst = dest_dir / rel_dst
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  copied {rel_dst:28s}  ({size_mb:6.1f} MB)")

    if missing:
        print("\nERROR: these expected inputs were not found:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"\nSynced {len(to_copy)} files into {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
