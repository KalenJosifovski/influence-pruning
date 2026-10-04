"""Build the molab deployment artifacts.

Two layouts are produced from the same minimal file set:

* ``--layout zip``    -> ``dist/molab_bundle.zip``, a self-extracting upload bundle;
* ``--layout upload`` -> ``dist/molab_upload/``, the same tree unzipped for manual upload
  through molab's Files panel.

The set is minimal for an unchanged notebook: ``verify_run`` requires all sixteen artifacts
per run, so nothing inside ``results/boostin_pruning/<run>/`` can be dropped. The corrected
effects carry only ranked-comparison resample draws.

Run with ``pixi run -e dev molab-bundle`` or ``pixi run -e dev molab-upload``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

from influence_pruning.artifacts import verify_run
from influence_pruning.errors import ArtifactError

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST = REPO_ROOT / "dist"
BUNDLE = DIST / "molab_bundle"
ARCHIVE = DIST / "molab_bundle.zip"
UPLOAD_DIR = DIST / "molab_upload"
NOTEBOOK = REPO_ROOT / "notebooks" / "04_pooling_surgery.py"
PACKAGE = REPO_ROOT / "src" / "influence_pruning"
RUN_ROOT = REPO_ROOT / "results" / "boostin_pruning"
EFFECTS_ROOT = REPO_ROOT / "analysis" / "effects_recomputed"
RANKED_KINDS = ("ranked_vs_random_median", "ranked_vs_matched_median")

INSTRUCTIONS = """\
MoleculeGrid / Pooling Under the Knife: molab deployment files
==============================================================

Route A, self-extracting zip
  1. On molab, create a new notebook from 04_pooling_surgery.py.
  2. Upload molab_bundle.zip through the Files panel. The notebook extracts it on first run.

Route B, manual tree (this folder)
  1. On molab, create a new notebook from 04_pooling_surgery.py.
  2. Upload the folders src/, results/, and analysis/ through the Files panel, preserving
     the paths exactly. If the panel cannot upload folders, create the folders first and
     upload the files into them.
  3. Run the notebook. It finds the package and artifacts relative to the notebook.

Either route needs `anywidget` (molab installs it on first import).

Contents:
  src/influence_pruning/           local package (including molgrid.py)
  results/boostin_pruning/<run>/   complete verified run directories (16 files each)
  analysis/effects_recomputed/     corrected effects + ranked-comparison draws
"""

UPLOAD_HEADER = """\
Manual molab upload tree
========================
Upload everything below into the molab notebook's file tree, preserving the
paths exactly as listed. The notebook loads them relative to its own directory.
`04_pooling_surgery.py` is the notebook to create/upload; the three folders are
its runtime data. Nothing else is required.
"""


def real_runs() -> list[Path]:
    """Return completed, schema-valid, non-dry-run directories."""
    runs = []
    for facts_path in sorted(RUN_ROOT.glob("*/run_facts.json")):
        directory = facts_path.parent
        if not (directory / "COMPLETED").is_file():
            continue
        try:
            facts = verify_run(directory)
        except ArtifactError:
            continue
        if facts.get("dry_run", False):
            continue
        runs.append(directory)
    if not runs:
        raise SystemExit("no completed scientific runs found under results/boostin_pruning")
    return runs


def stage(destination: Path) -> None:
    """Populate a deployment directory with the minimal package and artifact set."""
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "src").mkdir(parents=True)
    shutil.copytree(
        PACKAGE,
        destination / "src" / PACKAGE.name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for directory in real_runs():
        shutil.copytree(directory, destination / "results" / "boostin_pruning" / directory.name)
        effects_path = EFFECTS_ROOT / directory.name / "paired_effects.parquet"
        draws_path = EFFECTS_ROOT / directory.name / "bootstrap_draws.parquet"
        if not effects_path.is_file() or not draws_path.is_file():
            raise SystemExit(
                f"missing corrected effects for {directory.name}; run notebook 04 first"
            )
        target = destination / "analysis" / "effects_recomputed" / directory.name
        target.mkdir(parents=True)
        shutil.copy2(effects_path, target / "paired_effects.parquet")
        effects = pd.read_parquet(effects_path)
        ranked = set(effects.loc[effects["comparison_kind"].isin(RANKED_KINDS), "comparison_id"])
        draws = pd.read_parquet(draws_path)
        draws.loc[draws["comparison_id"].isin(ranked)].to_parquet(
            target / "bootstrap_draws.parquet"
        )
    (destination / "MOLAB_UPLOAD.txt").write_text(INSTRUCTIONS)


def build_zip() -> Path:
    """Build the self-extracting archive; the notebook is deliberately not included."""
    stage(BUNDLE)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(BUNDLE))
    return ARCHIVE


def build_upload() -> Path:
    """Build the unzipped manual-upload tree, notebook included for convenience."""
    stage(UPLOAD_DIR)
    shutil.copy2(NOTEBOOK, UPLOAD_DIR / NOTEBOOK.name)
    lines = [UPLOAD_HEADER]
    for path in sorted(UPLOAD_DIR.rglob("*")):
        if path.is_file():
            relative = path.relative_to(UPLOAD_DIR)
            lines.append(f"  {relative}  ({path.stat().st_size / 1e6:.2f} MB)")
    (UPLOAD_DIR / "UPLOAD_TREE.txt").write_text("\n".join(lines) + "\n")
    return UPLOAD_DIR


def total_size(folder: Path) -> float:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", choices=("zip", "upload", "both"), default="both", help="what to build"
    )
    args = parser.parse_args()
    if args.layout in ("zip", "both"):
        archive = build_zip()
        print(f"zip:    {archive.relative_to(REPO_ROOT)} ({archive.stat().st_size / 1e6:.1f} MB)")
    if args.layout in ("upload", "both"):
        folder = build_upload()
        print(
            f"upload: {folder.relative_to(REPO_ROOT)}/ ({total_size(folder):.1f} MB, "
            f"notebook + src/ + results/ + analysis/)"
        )


if __name__ == "__main__":
    sys.exit(main())
