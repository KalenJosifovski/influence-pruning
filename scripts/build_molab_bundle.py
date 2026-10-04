"""Build the self-extracting molab upload bundle.

The bundle contains everything ``notebooks/04_pooling_surgery.py`` needs to run outside the
repository checkout: the notebook itself, the ``influence_pruning`` package, the complete
real-run artifact directories that ``verify_run`` checks, and the corrected-effects cache
with ranked-comparison resample draws only. ``dist/molab_bundle.zip`` can be uploaded to a
molab notebook's storage panel; the notebook extracts it automatically on first run.

Run with ``pixi run -e dev molab-bundle``.
"""

from __future__ import annotations

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
NOTEBOOK = REPO_ROOT / "notebooks" / "04_pooling_surgery.py"
PACKAGE = REPO_ROOT / "src" / "influence_pruning"
RUN_ROOT = REPO_ROOT / "results" / "boostin_pruning"
EFFECTS_ROOT = REPO_ROOT / "analysis" / "effects_recomputed"
RANKED_KINDS = ("ranked_vs_random_median", "ranked_vs_matched_median")

INSTRUCTIONS = """\
MoleculeGrid / Pooling Under the Knife: self-extracting molab bundle
====================================================================

1. On molab, create a new notebook and paste or upload
   `04_pooling_surgery.py`.
2. Open the Files panel (folder icon in the sidebar) and upload
   `molab_bundle.zip`. The notebook extracts it into the session on first
   run; no manual unzipping is needed.
3. Install `anywidget` from the package manager panel if the notebook asks
   (molab also installs packages on first import).
4. Run the notebook. Everything else (the `influence_pruning` package,
   run artifacts, corrected effects) is inside the bundle.

Bundle contents:
  src/influence_pruning/           local package (including molgrid.py)
  results/boostin_pruning/<run>/   complete verified run directories
  analysis/effects_recomputed/     corrected effects + ranked draws
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


def build() -> Path:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "src").mkdir(parents=True)
    shutil.copy2(NOTEBOOK, BUNDLE / NOTEBOOK.name)
    shutil.copytree(
        PACKAGE,
        BUNDLE / "src" / PACKAGE.name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    runs = real_runs()
    for directory in runs:
        shutil.copytree(directory, BUNDLE / "results" / "boostin_pruning" / directory.name)
        effects_path = EFFECTS_ROOT / directory.name / "paired_effects.parquet"
        draws_path = EFFECTS_ROOT / directory.name / "bootstrap_draws.parquet"
        if not effects_path.is_file() or not draws_path.is_file():
            raise SystemExit(
                f"missing corrected effects for {directory.name}; run notebook 04 first"
            )
        target = BUNDLE / "analysis" / "effects_recomputed" / directory.name
        target.mkdir(parents=True)
        shutil.copy2(effects_path, target / "paired_effects.parquet")
        effects = pd.read_parquet(effects_path)
        ranked = set(effects.loc[effects["comparison_kind"].isin(RANKED_KINDS), "comparison_id"])
        draws = pd.read_parquet(draws_path)
        draws.loc[draws["comparison_id"].isin(ranked)].to_parquet(
            target / "bootstrap_draws.parquet"
        )
    (BUNDLE / "MOLAB_UPLOAD.txt").write_text(INSTRUCTIONS)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(BUNDLE))
    return ARCHIVE


def main() -> None:
    archive = build()
    print(f"notebook:  {NOTEBOOK.relative_to(REPO_ROOT)}")
    print(f"runs:      {', '.join(directory.name for directory in real_runs())}")
    print(f"archive:   {archive.relative_to(REPO_ROOT)} ({archive.stat().st_size / 1e6:.1f} MB)")
    print("upload the notebook and dist/molab_bundle.zip to the molab notebook storage panel")


if __name__ == "__main__":
    sys.exit(main())
