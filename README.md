# Influence-Guided Data Pruning

Conditional attribution-pruning for pooled molecular-property models. The study asks whether a
data-attribution method can identify molecules to retain or remove when fitting a pooled model:
after fitting on the full pooled dataset, does the attribution ranking predict the effect of
removing groups of training molecules?

This repository implements **Phase 1** of `PLAN.md`/`IMPLEMENTATION.md`: BoostIn with XGBoost on
HLM CLint with ExpansionRx as target and Biogen plus Polaris as donors, with full support for
rotating any source into the target role. Phase 2 (TracIn with a neural molecular model, and the
curated CYP setting) is deliberately out of scope; the `TrainingAttributor` interface, the
`PruningSplit`/`DeletionArm` containers, the evaluation code, and the artifact schema are the
hand-off boundary for it.

## Scientific design in one paragraph

For a pooled training set \(T = L \cup B \cup P\), a full model \(f_{\mathrm{full}}\) is fitted,
every training observation is scored with BoostIn against a distinct target-like influence
surface \(S\), and batches are deleted from \(T\). Each pruned model \(f_{-A}\) is compared with
\(f_{\mathrm{full}}\) on a held-out target-domain evaluation surface \(E\):
\(\Delta_A = \mathrm{RMSE}(E, f_{\mathrm{full}}) - \mathrm{RMSE}(E, f_{-A})\), so \(\Delta_A > 0\)
means pruning improved evaluation RMSE. Pools (`donor_only`, `native_only`, `mixed_pool`), batch
sizes, uniform-random controls, and label/source-stratified controls are declared in
configuration. Stratified controls preserve source composition and within-source label
quantile bins; they are not exact continuous-label matches, so their achieved label
distributions can differ from the reference batch (reported as Wasserstein diagnostics).
Attribution never sees \(E\); evaluation never influences scoring or arm construction. Sign
orientation is validated only through conditional deletion: the full pooled model is scored
against \(S\), high-score and low-score batches are deleted from that same pool, and both
pruned models are compared on \(E\). No addition-utility calibration is used or reported.

The ExpansionRx official test identifiers are time-ordered and split at their midpoint: the
earlier half is \(S\), the later half is \(E\). Because the supplied test has informed prior
exploration, results are **temporally ordered exploratory** analyses, not prospective
confirmation. For a Biogen or Polaris target rotation, the available target test partition is
split into whole-Butina-cluster selection and evaluation surfaces.

## Environments

Phase 1 runs in an isolated environment because `tree-influence` requires NumPy 1.x and a pinned
XGBoost release, which cannot join the NumPy 2.x default solve.

| Environment | Purpose | Command |
| :--- | :--- | :--- |
| `dev` | Tests, linting, notebook development | `pixi run -e dev …` |
| `influence` | BoostIn scoring and every XGBoost refit | `pixi run -e influence ipp …` |

```bash
pixi run -e influence python -c "import numpy, xgboost, rdkit, tree_influence; print(numpy.__version__, xgboost.__version__)"
pixi run -e dev pytest
pixi run -e dev ruff check src tests
pixi run -e influence test-influence
```

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml` and point it at the local dataset
catalogue. That file is Gitignored and must never refer to or scan
`~/datasets/nuclear_receptor_hts/`.

## Commands

```bash
# Data standardisation, contracts, and partitions only; no model fits.
pixi run -e influence ipp audit --config configs/boostin_expansionrx_hlm.yaml

# Capped end-to-end run that exercises the complete artifact contract (marked dry_run).
pixi run -e influence ipp dry-run --config configs/dryrun_boostin_expansionrx_hlm.yaml

# One scoring fit, one full fit, one pruned fit, and the projected run budget.
pixi run -e influence ipp benchmark --config configs/boostin_expansionrx_hlm.yaml

# The real episode (not yet authorised; review the dry-run artifacts and benchmark first).
pixi run -e influence ipp boostin-prune --config configs/boostin_expansionrx_hlm.yaml

# Read-only integrity verification of any completed run.
pixi run -e influence ipp verify --run results/boostin_pruning/<run-id>
```

Target rotations use the same code path:

```bash
pixi run -e influence ipp boostin-prune --config configs/boostin_biogen_hlm.yaml
pixi run -e influence ipp boostin-prune --config configs/boostin_polaris_hlm.yaml
```

Polaris HLM is unresolved by default and fails closed; every HLM configuration carries the
documented IVIVE override (scale factor 0.9, values below 10 µL/min/mg censored) with its
source and rationale retained in `contract_overrides`.

## Artifact contract

Each completed run is self-contained under
`results/<kind>/<timestamp>_<endpoint>_<target>_<config-hash>/`:

| Artifact | Purpose |
| :--- | :--- |
| `config.source.yaml`, `config.resolved.yaml` | Original and fully resolved configuration, with `config_hash` |
| `environment.json`, `run_facts.json`, `run.log`, `COMPLETED` | Versions, machine, fit counts, timings, completion state |
| `data_audit.parquet`, `exclusions.parquet` | Counts, units, transforms, and every excluded molecule |
| `partition_manifest.parquet` | One row per observation: role, provenance, chronology/cluster assignment, label visibility |
| `boostin_scores.parquet` | One row per training molecule × score seed plus aggregate ranks |
| `arm_manifest.parquet`, `arm_summary.parquet` | Exact deletion membership and arm-level matching diagnostics |
| `predictions.parquet`, `seed_metrics.parquet` | Per-molecule predictions and RMSE/MAE per condition × seed |
| `paired_effects.parquet`, `bootstrap_draws.parquet` | Paired effects with cluster-bootstrap intervals and every resample |

`COMPLETED` is written only after every required artifact exists and passes schema validation.
`ipp verify` is read-only. Infeasible stratified-control groups are recorded in
`arm_summary.parquet` with an explicit diagnostic and are never silently replaced by random
selection. `paired_effects.parquet` compares each ranked arm against the median of *all*
recorded random draws and the median of all recorded stratified-control draws
(`ranked_vs_random_median`, `ranked_vs_matched_median`), listing the control policy, draw
count, and draw IDs in `control_policy`, `n_control_draws`, and `control_arm_ids`.

## Notebooks

`notebooks/01_data_and_partition_audit.py` and `notebooks/02_boostin_pruning_analysis.py` are
read-only with respect to raw run artifacts and model computation. They load a completed run by
path, call `verify`, and refuse to run if artifacts are missing or the configuration hash
disagrees. They never fit models, recompute attribution, regenerate partitions, or recompute
bootstrap resamples. They may export derived figures to `analysis/<run-id>/`.

```bash
pixi run -e dev notebooks             # view all notebooks
pixi run -e dev pooling-surgery       # run notebook 04 as an app
pixi run -e dev pooling-surgery-edit  # edit notebook 04
```

`notebooks/04_pooling_surgery.py` ("Pooling Under the Knife") is the cross-target story: it
compares pooling against local-only training across the three rotations, validates the BoostIn
deletion direction with paired resample draws, explains the label confound, and closes with
calibration and chemical-space panels. It ships a custom anywidget molecule explorer
(`src/influence_pruning/molgrid.py`) with search, sorting, click selection, scaffold
highlighting, and box/lasso brushing from the PCA map. The widget degrades to a static RDKit
grid when `anywidget` is unavailable.

### Running on molab

Build the self-extracting upload bundle:

```bash
pixi run -e dev molab-bundle     # writes dist/molab_bundle.zip (~53 MB)
```

Then, on molab:

1. Create a new notebook and paste (or upload) `notebooks/04_pooling_surgery.py`.
2. Open the Files panel (folder icon in the sidebar) and upload `dist/molab_bundle.zip`.
3. Run the notebook. On first run it extracts the bundle into the session, adds `src/` to
   the import path, and loads the three verified runs — no unzipping or path setup.
4. If prompted, install `anywidget` from the package manager panel (molab also installs
   packages on first import).

Locally the extraction step is a no-op, so the same notebook file serves both environments.
The bundle unpacks to the package, the three run directories that `verify_run` checks, and
the corrected-effects cache. For reference, the uncompressed sizes are:

| Bundle | Contents | All three rotations |
| :--- | :--- | ---: |
| Run directories | every file `verify_run` checks, including `predictions.parquet` and the results-level `bootstrap_draws.parquet` | 56 MB |
| Corrected effects | `analysis/effects_recomputed/<run>/paired_effects.parquet` | 0.2 MB |
| Optional resample draws | `analysis/effects_recomputed/<run>/bootstrap_draws.parquet`; omit to let the notebook recompute in ~40 s | 6.8 MB ranked-only (32 MB full) |

Nothing is too large for molab: the built zip is ~53 MB, comfortably within the notebook
storage panel, and the notebook recomputes the draw cache in about 40 s if it is omitted.
The bundle builder skips dry-run and stale-schema runs automatically.

## Repository layout

```text
src/influence_pruning/   config, contracts, standardize, similarity, data, partitions,
                         model, attribution, pruning_arms, evaluate, bootstrap, artifacts,
                         run, cli
configs/                 one YAML per target rotation plus the capped dry-run config
notebooks/               read-only Marimo analysis layers
tests/                   synthetic-frame and temporary-directory tests only
results/                 immutable script-generated artifacts, Gitignored
analysis/                notebook-exported derived figures, Gitignored
```

## Status

- The complete Phase 1 implementation, tests, dry run, and benchmark are in place.
- The dry run (real chemistry, capped limits) produces and verifies every required artifact.
- The registered ExpansionRx HLM benchmark projects **600 arms / 3,015 fits / ≈12 h** on an
  M1-class 8-core machine; the real run is deferred until that budget is reviewed (Myriad CPU
  or GPU execution can be decided then).
- No real-data pruning result is claimed yet.
