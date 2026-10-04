# Phase 1 BoostIn Pruning Implementation Plan

## Status and Scope

This document specifies the first implementation in the Influence-Guided Data Pruning project. It is a plan, not an instruction to run models or create result artifacts. The implementation evaluates **conditional batch pruning**: whether BoostIn scores computed in a full pooled XGBoost model predict the effect of removing batches from that same pooled training set.

The initial episode is HLM clearance with ExpansionRx as target and Biogen plus Polaris as donors. The code must support rotation, so Biogen or Polaris can subsequently be the target and the other two sources become donors. Phase 2 TracIn is explicitly out of scope here.

This project is deliberately independent of the preceding [towards-global-models](../towards-global-models/) repository. Its logic may be adapted from that work, but it must not import source modules or read result directories from it at runtime. The old repository remains an auditable historical record of addition-based S1/S2 experiments.

## Scientific Contract

### Question

For a pooled training dataset \(T\), where \(T=L\cup B\cup P\) contains native target-programme observations \(L\), Biogen observations \(B\), and Polaris observations \(P\), does a BoostIn ranking identify batches whose deletion changes target-domain loss as predicted?

The full model is:

\[
f_{\mathrm{full}} = \operatorname{fit}(T).
\]

For a target-like selection surface \(S\), BoostIn produces local influences \(I_{ij}\) for training observation \(i\) and selection molecule \(j\). The score for a training observation is:

\[
s_i=\frac{1}{|S|}\sum_{j\in S} I_{ij}.
\]

The implementation must calibrate and record the sign convention. With the convention used by `tree_influence`, a positive local influence decreases the loss of the selected test example. Therefore a high-score removal batch is expected to harm performance, whereas removal of a low-score batch may improve it.

For a deletion arm \(A\subset T\), fit:

\[
f_{-A}=\operatorname{fit}(T\setminus A).
\]

Evaluate both models on a distinct target-domain evaluation surface \(E\). The primary paired effect is:

\[
\Delta_A = \operatorname{RMSE}(E,f_{\mathrm{full}})-
           \operatorname{RMSE}(E,f_{-A}).
\]

Thus \(\Delta_A>0\) means that pruning arm \(A\) improved evaluation RMSE; \(\Delta_A<0\) means pruning harmed it.

### What this does and does not test

| Claim | Tested here? | Design element |
| :--- | :---: | :--- |
| Conditional data attribution predicts deletion utility | Yes | Score in the full pooled model and delete from that same pool. |
| Adding a selected donor subset improves a local model | No | This is the earlier addition-based question, not a BoostIn deletion validation. |
| A donor-only subset is sufficient to predict target chemistry | No | Donor-only models are a separate coverage/assay-alignment experiment. |
| Assay harmonisation creates a global model | No | Endpoint contracts are retained, but harmonisation is not changed in Phase 1. |

### Evaluation discipline

The ranking may use only \(S\); pruning outcomes may use only \(E\). The two roles must be structurally distinct in the API, not merely separated by convention.

For ExpansionRx HLM:

```text
Official target train + eligible Biogen + eligible Polaris  -> full pooled training T
Earlier half of temporally ordered official test             -> influence selection S
Later half of temporally ordered official test               -> evaluation E
```

The official ExpansionRx test identifiers are time-ordered. Split at the identifier midpoint after standardisation and target-specific overlap exclusion. The earlier test half is \(S\); the later half is \(E\). This leaves approximately 10% of the full dataset for final evaluation while retaining all official training data for \(T\).

The supplied ExpansionRx test has informed previous exploratory work. Results from this episode must be labelled **temporally ordered exploratory**, not prospective or newly sealed.

For a Biogen or Polaris target rotation, divide the available target test partition into two disjoint groups of whole Butina clusters: one selection surface and one evaluation surface. The configured seed, fingerprint specification, distance threshold, fraction, and cluster membership must be saved.

## Experiment Matrix

### Pools and provenance classes

The first episode runs all three score-and-delete pools. Scores are always calculated once in the same model \(f_{\mathrm{full}}\); a pool only determines which rows are eligible for ranking and control construction.

| Pool | Eligible deletions | Primary purpose |
| :--- | :--- | :--- |
| `donor_only` | Biogen and Polaris rows, individually and combined | Test whether public-data utility is source/domain-specific. |
| `native_only` | ExpansionRx training rows | Test whether the estimator works on the target programme’s own distribution. |
| `mixed_pool` | Every row in \(T\) | Test the practical policy: remove the worst observations irrespective of origin. |

For `donor_only`, report Biogen-only, Polaris-only, and combined-donor eligibility as separate strata. For `mixed_pool`, report the source composition of every selected batch: it must never be summarised as source-agnostic if it is almost entirely one source.

### Arms

At each batch size \(k\in\{25,50,100,150\}\), and within each eligible pool, construct:

| Policy | Batch definition | Expected directional result if score is valid |
| :--- | :--- | :--- |
| `boostin_high` | The \(k\) largest aggregated BoostIn scores | Negative \(\Delta\): removal harms. |
| `boostin_low` | The \(k\) smallest aggregated BoostIn scores | Positive or least-negative \(\Delta\): removal helps or costs least. |
| `random` | Repeated uniform batches, without replacement within a draw | Reference distribution. |
| `label_source_matched` | Repeated source-stratified, quantile-bin-matched batches: the scored batch’s source composition and within-source label quantile-bin counts are preserved, but selected without score and without exact continuous-label equality | Tests whether score adds information beyond label level and source. |

Use 10 draws for `random` and 10 draws for `label_source_matched` at each `pool × direction × k` combination. Random draws use deterministic, recorded seeds. The exact number of model fits should be calculated by `benchmark` before any real run.

`local_only`—all target training data and no external donors—will be fitted and reported once per seed as a practical reference. It is not the deletion baseline. The primary deletion baseline is always `full_pooled`.

### Label/source-stratified control algorithm

The control is intentionally difficult. It should not be quietly weakened when a high- or low-score batch is also label-extreme, and it is not an exact continuous-label match: the required counts come from quantile bins, so achieved label values within a bin can differ from the reference batch.

1. For the scored reference batch, retain its `source × label-bin` composition (bin counts, not exact values).
2. Stratify values within each source into configured quantile bins, initially deciles, based only on the candidate pool for that arm.
3. Draw without replacement from the same eligible pool, excluding the scored batch, with the same count in each `source × bin` stratum.
4. If a stratum cannot supply enough rows, deterministically retry using successively coarser bins: deciles, quintiles, then terciles.
5. If construction remains infeasible, record that arm at planning time with an explicit diagnostic. Do not silently fall back to random selection.
6. Persist both the intended and achieved label summaries. The run must report an earth-mover or quantile-distance diagnostic (`label_wasserstein_overall`, `label_wasserstein_max_source`) plus `matching_bins_used`, `matching_plan`, and `matching_diagnostic` so the notebook can show how close the stratification came. No hard matching threshold is imposed; non-zero distances are expected and reported.

Some large deletion batches, especially from Polaris, may make a non-overlapping stratified control impossible. That is a scientific limitation of that arm, not a reason to relax the contract. A future alternative can permit overlapping controls, but it should be a separately named policy, not substituted here.

### Comparisons and uncertainty

For every arm, retain predictions for every evaluation molecule and every matched model seed. Cluster the evaluation molecules with the fixed Butina threshold and calculate paired cluster-bootstrap intervals by resampling evaluation clusters.

The analysis reports:

- Full-pool versus pruned effect \(\Delta_A\), for every arm.
- Ranked-arm versus the median random-removal policy at the same pool, direction, and batch size: the median is taken over every recorded random draw’s seed-averaged RMSE inside each cluster-bootstrap resample and identically on the full evaluation set. No representative draw is selected by its observed effect.
- Ranked-arm versus the median label/source-stratified control policy, constructed the same way over every recorded control draw of the same direction.
- Per-seed effects, all individual draw effects, and bootstrap intervals. Policy comparisons additionally record `control_policy`, `n_control_draws`, and `control_arm_ids`.

No `delta_min`, binary decision label, or p-value threshold is required. This is exploratory: interpret effect magnitude, interval width, draw-to-draw variation, and consistency across the three provenance pools.

## Repository Layout

Create the following layout in this repository. Names below are the intended architecture, not a requirement to duplicate the older project’s module boundaries exactly.

```text
influence-pruning/
├── PLAN.md
├── IMPLEMENTATION.md
├── README.md
├── pyproject.toml
├── pixi.toml
├── pixi.lock
├── .gitignore
├── configs/
│   ├── paths.example.yaml
│   ├── paths.local.yaml                 # Gitignored
│   ├── boostin_expansionrx_hlm.yaml
│   ├── boostin_biogen_hlm.yaml
│   ├── boostin_polaris_hlm.yaml
│   └── dryrun_boostin_expansionrx_hlm.yaml
├── src/influence_pruning/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── contracts.py
│   ├── data.py
│   ├── standardize.py
│   ├── similarity.py
│   ├── partitions.py
│   ├── model.py
│   ├── attribution.py
│   ├── pruning_arms.py
│   ├── evaluate.py
│   ├── bootstrap.py
│   ├── artifacts.py
│   └── run.py
├── notebooks/
│   ├── 01_data_and_partition_audit.py
│   └── 02_boostin_pruning_analysis.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_partitions.py
│   ├── test_attribution.py
│   ├── test_pruning_arms.py
│   ├── test_evaluate.py
│   ├── test_artifacts.py
│   └── test_end_to_end.py
└── results/
    └── boostin_pruning/<timestamp>_<endpoint>_<target>_<config-hash>/
```

`results/` is generated and Gitignored. Do not copy legacy S1/S2 results into this project. `paths.local.yaml` is Gitignored; it may point to the existing local dataset catalogue but must never refer to or scan `~/datasets/nuclear_receptor_hts/`.

## Pixi Environment Plan

### Principles

The Phase 1 model-fitting environment must be compatible with `tree-influence`, which currently needs NumPy 1.x and a compatible XGBoost release. Keep it isolated rather than weakening a general analysis environment. The previous project’s [`pixi.toml`](../towards-global-models/pixi.toml) demonstrates this separation in its `influence` feature.

Use Python 3.12 across initial environments. Do not install packages with global `pip`, `uv pip`, or an ad hoc virtual environment.

### Proposed environments

| Environment | Purpose | Key dependencies |
| :--- | :--- | :--- |
| `base` | Shared package imports and artifact reading | Python 3.12, Pandas, PyArrow, PyYAML, RDKit, Matplotlib, Seaborn. |
| `dev` | Tests, linting, and Marimo notebook development | `base` plus Pytest, Ruff, Marimo. |
| `influence` | All Phase 1 BoostIn scoring and XGBoost refits | Python 3.12, NumPy 1.26, XGBoost 3.4, RDKit, Pandas, PyArrow, `tree-influence` pinned to a commit. |

The `influence` environment should use `no-default-feature = true`, as in the older project, to avoid an impossible NumPy 1.x / NumPy 2.x solve. It must include every runtime dependency used by the Phase 1 CLI; no script should require switching environments mid-run.

### Initial `pixi.toml` shape

The final pin should be resolved and committed by Pixi, but the intended structure is:

```toml
[workspace]
name = "influence-pruning"
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64"]

[feature.dev.dependencies]
pytest = ">=8.3,<9"
ruff = ">=0.11,<1"
marimo = ">=0.14,<1"

[feature.influence.dependencies]
python = "3.12.*"
numpy = "1.26.*"
xgboost = "3.4.*"
pandas = ">=2.2,<3"
pyarrow = ">=17,<23"
pyyaml = ">=6,<7"
rdkit = ">=2024.9,<2027"
matplotlib = ">=3.9,<4"
seaborn = ">=0.13,<1"

[feature.influence.pypi-dependencies]
tree-influence = { git = "https://github.com/jjbrophy47/tree_influence", rev = "<verified-commit>" }

[environments]
dev = { features = ["dev"] }
influence = { features = ["influence"], no-default-feature = true }

[feature.influence.tasks]
ipp = { cmd = "python -m influence_pruning", env = { PYTHONPATH = "src", PYTHONUNBUFFERED = "1" } }
test-influence = { cmd = "python -m pytest", env = { PYTHONPATH = "src", PYTHONUNBUFFERED = "1" } }
```

Before the first real run, verify imports and print versions with:

```bash
pixi run -e influence python -c "import numpy, xgboost, rdkit, tree_influence; print(numpy.__version__, xgboost.__version__)"
pixi run -e influence test-influence
```

The implementation must not train a model as an environment check.

## Reuse Map From the Earlier Project

Adapt logic after reading the listed implementation and tests. Do not runtime-import from the older project; copy only the minimal tested code and rename it under `influence_pruning`.

| New module | Adapt from | Required change |
| :--- | :--- | :--- |
| `standardize.py` | [`standardize.py`](../towards-global-models/src/tgm/standardize.py) | Preserve canonical SMILES, InChIKey, fingerprints, duplicate aggregation, and observation provenance. |
| `contracts.py` | [`contracts.py`](../towards-global-models/src/tgm/contracts.py) and [`config.py`](../towards-global-models/src/tgm/config.py) | Retain explicit endpoint transforms and Polaris HLM scale-factor override; fail closed on unresolved units. |
| `data.py` | [`data.py`](../towards-global-models/src/tgm/data.py) | Generalise target rotation; return target train, target test, eligible donors, exclusions, and an audit table. |
| `similarity.py` | [`similarity.py`](../towards-global-models/src/tgm/similarity.py) | Reuse Morgan fingerprints, maximum Tanimoto, and Butina cluster labels. |
| `partitions.py` | [`study_splits.py`](../towards-global-models/src/tgm/study_splits.py) | Replace `local_base/influence/development/outer` with full training, selection, and evaluation roles. |
| `model.py` | [`model.py`](../towards-global-models/src/tgm/model.py) | Preserve the frozen XGBoost parameter path and explicit `base_score` only for tree-influence parsing. |
| `attribution.py` | [`boostin.py`](../towards-global-models/src/tgm/boostin.py) | Score every row of the one pooled model, not each donor in a separate local-plus-donor model. |
| `bootstrap.py` | [`bootstrap.py`](../towards-global-models/src/tgm/bootstrap.py) | Preserve paired cluster resampling; additionally persist the bootstrap distribution. |
| `artifacts.py` | [`study_runs.py`](../towards-global-models/src/tgm/study_runs.py) | Use a new study root and a stricter artifact-completion contract. |
| `cli.py` | [`cli.py`](../towards-global-models/src/tgm/cli.py) | Add only pruning-specific commands; do not carry S1/C4/C6 commands into this project. |

## Configuration Schema

Use one YAML file per target rotation. All paths and all scientifically relevant values must be resolved into the run directory before data loading or fitting begins.

```yaml
study:
  kind: boostin_pruning
  phase: exploratory

endpoint: hlm_clint
target: expansionrx
donor_sources: [biogen, polaris]
paths_file: configs/paths.local.yaml
results_root: results

split:
  expansionrx_test_mode: temporal_halves
  target_test_selection_fraction: 0.50
  butina_distance_threshold: 0.40
  butina_seed: 0

model:
  radius: 2
  n_bits: 2048
  chirality: false
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  min_child_weight: 1.0
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 1.0
  tree_method: hist
  n_jobs: -1

boostin:
  score_seeds: [0, 1, 2, 3, 4]
  outcome_seeds: [0, 1, 2, 3, 4]
  batch_sizes: [25, 50, 100, 150]
  random_draws: 10
  matching_draws: 10
  label_match_bins: [10, 5, 3]

bootstrap:
  n_resamples: 2000
  cluster_distance_threshold: 0.40
  seed: 0

contract_overrides:
  # Existing verified Polaris HLM override, with source and rationale retained.
```

The parser must reject:

- A target included in `donor_sources`.
- Unknown source, endpoint, or target test mode.
- Empty or duplicate seed sets.
- Non-increasing or non-positive batch sizes.
- Batch sizes exceeding the eligible pool for any required arm.
- A temporal test mode for a target whose IDs cannot be parsed and verified as ordered.
- A Butina split with no molecules or no clusters in either surface.
- Missing, invalid, or non-positive scale-factor contracts.

## Core Implementation

### Data and partitions

Implement immutable data containers analogous to the older project’s `LabelledPartition`, but use names that reflect this study:

```python
@dataclass(frozen=True, slots=True)
class PruningSplit:
    full_training: LabelledPartition
    influence_selection: LabelledPartition
    evaluation: LabelledPartition
    partition_manifest: pd.DataFrame
```

`full_training` contains all eligible target-training observations and all eligible rows from each donor source. It does **not** include either test half. Every row must carry:

```text
candidate_id, source, origin_class, original_id, inchikey,
canonical_smiles, model_target, fingerprint position
```

`candidate_id` must be source-qualified, for example `biogen:<original_id>:<inchikey>`. Validate unique candidate IDs and disjoint InChIKeys across full training, selection, and evaluation. Remove target–donor structure overlap before creating the split, using the existing project’s exclusion convention.

The module must expose two constructors:

```python
def make_temporal_test_halves(bundle: ObservationBundle) -> tuple[LabelledPartition, LabelledPartition]: ...
def make_butina_test_halves(frame: pd.DataFrame, settings: SplitSettings) -> tuple[LabelledPartition, LabelledPartition]: ...
```

The temporal constructor verifies that parsed IDs are unique and strictly increasing, then records the numeric midpoint, counts, and ID ranges. The Butina constructor allocates whole clusters reproducibly and records every cluster assignment.

### BoostIn attribution

`attribution.py` should expose an estimator-neutral interface, even though BoostIn is the sole Phase 1 implementation:

```python
class TrainingAttributor(Protocol):
    def score(
        self,
        training: LabelledPartition,
        selection: LabelledPartition,
        model_settings: ModelSettings,
        seeds: tuple[int, ...],
    ) -> pd.DataFrame: ...
```

`BoostInAttributor.score()` should:

1. Fit `XGBRegressor` on all `full_training` rows for each `score_seed`.
2. Use an explicit numerical `base_score` only in the model supplied to `tree-influence`; this follows the compatibility handling in the older [`fit_regressor`](../towards-global-models/src/tgm/model.py).
3. Call `BoostIn().fit(model, train_x, train_y)`.
4. Call `get_local_influence(selection_x, selection_y)` and require a matrix of shape `(n_training, n_selection)`.
5. Reduce over selection molecules to `raw_score` for every training row.
6. Create a global within-seed rank across **all** training candidates. This global rank is necessary for the `mixed_pool` policy; per-pool ranks are derived only when constructing an arm.
7. Aggregate per-seed ranks by mean rank and record raw-score mean, standard deviation, rank standard deviation, seed count, source, origin class, labels, and identifiers.

Do not score Biogen and Polaris independently, as the old [`score_one_donor`](../towards-global-models/src/tgm/boostin.py) does. The central change is that every score must be conditional on the same full pooled model from which pruning will occur.

Implement a small synthetic sign-calibration test. It must construct a pooled toy problem containing deliberately harmful shifted-label observations, score against a separate selection set, and verify the expected **deletion** orientation—not the old addition orientation. This establishes software sign handling but is not a scientific validation of the real-data ranking.

### Arm planning

`pruning_arms.py` turns score rows into immutable deletion manifests. It must never fit a model.

```python
@dataclass(frozen=True, slots=True)
class DeletionArm:
    arm_id: str
    eligible_pool: str
    policy: str
    direction: str | None
    batch_size: int
    draw_number: int | None
    removed_candidate_ids: tuple[str, ...]
```

For every eligible pool, rank only its eligible candidates by the global aggregated score. The `boostin_high` and `boostin_low` batches are deterministic. Random and stratified controls are deterministic given their draw seed.

The arm manifest must have one row per `arm_id × candidate_id`, and include score, rank within pool, source, origin class, label, label-matching stratum, and selection reason. A second `arm_summary.parquet` has one row per arm with requested/achieved size, source counts, label summaries, matching diagnostics, and exact training size after pruning.

Arm IDs must be human-readable and stable, for example:

```text
donor_only:boostin_high:k100
native_only:random:k50:draw03
mixed_pool:label_source_matched_low:k150:draw08
```

The planner must fail if any deletion batch contains a duplicate candidate, a candidate outside its eligible pool, or a candidate absent from full training. It must also prove that `n_train_pruned = n_train_full - batch_size` before any fit is launched.

### Fitting and evaluation

`evaluate.py` owns the only model refits. It receives a `PruningSplit`, the arm manifests, settings, and output seeds.

Pseudocode:

```python
for seed in outcome_seeds:
    full_prediction = fit_predict(full_training, evaluation, seed)
    local_only_prediction = fit_predict(native_training_only, evaluation, seed)

    for arm in deletion_arms:
        pruned = full_training.exclude(arm.removed_candidate_ids)
        prediction = fit_predict(pruned, evaluation, seed)
        persist_prediction_rows(seed, arm, prediction)

for arm in deletion_arms:
    bootstrap full_prediction vs arm_prediction on identical evaluation clusters
    bootstrap arm_prediction vs the random and stratified-control policies
    (median control RMSE inside every resample; every recorded draw enters)
```

Use the same XGBoost parameters and matched outcome seeds for every condition. Do not add a hyperparameter-search phase. Cluster evaluation chemistry only, using the fixed Butina threshold; cluster labels must be written into the prediction artifact.

Refactor the old [`cluster_bootstrap_delta`](../towards-global-models/src/tgm/bootstrap.py) concept so it returns both a compact interval record and the resampled effects. The latter are an artifact, enabling the Marimo notebook to plot distributions without rerunning bootstrap calculations.

### Commands

The CLI should remain small:

```text
ipp audit --config CONFIG       # Standardisation, contracts, partitions, no model fits
ipp dry-run --config CONFIG     # Synthetic or capped data, artifact-contract exercise
ipp benchmark --config CONFIG   # One representative full/pruned fit, projected budget
ipp boostin-prune --config CONFIG
ipp verify --run RUN_DIRECTORY  # Read-only integrity and artifact validation
```

`audit` may load data but must not import `tree_influence` or fit a model. `dry-run` must operate on explicitly capped configuration limits and be visibly marked `dry_run: true` in every artifact. `verify` must never write or refit.

## Artifact Contract

Each completed run is self-contained under:

```text
results/boostin_pruning/<timestamp>_<endpoint>_<target>_<config-hash>/
```

Required artifacts:

| Artifact | Format | Purpose |
| :--- | :--- | :--- |
| `config.source.yaml`, `config.resolved.yaml` | YAML | Original and fully resolved configuration. |
| `environment.json` | JSON | Python, package, platform, and Git revision metadata. |
| `data_audit.parquet` | Parquet | Counts, exclusions, units, transform, and source summaries. |
| `partition_manifest.parquet` | Parquet | One row per observation with role, source, chronology/cluster assignment, and label visibility. |
| `boostin_scores.parquet` | Parquet | One row per training molecule × score seed plus aggregate score/rank information. |
| `arm_manifest.parquet` | Parquet | Exact removed membership for every arm. |
| `arm_summary.parquet` | Parquet | Arm-level provenance and matching diagnostics. |
| `predictions.parquet` | Parquet | One row per evaluation molecule × condition × seed. |
| `seed_metrics.parquet` | Parquet | RMSE and MAE per condition × seed. |
| `paired_effects.parquet` | Parquet | Full-versus-pruned and policy-versus-control point estimates and intervals. |
| `bootstrap_draws.parquet` | Parquet | All paired bootstrap resamples for every comparison. |
| `run_facts.json`, `run.log`, `COMPLETED` | JSON/text | Completion state, fit count, timings, and logs. |

The scripts may write a compact `RUN.md` summary, but no publication-style interpretation should be embedded in the runner. The Marimo notebooks are responsible for figures, descriptive tables, and narrative analysis. `COMPLETED` is created only after all required artifacts exist and pass schema validation.

## Marimo Analysis Notebooks

The notebooks are read-only with respect to models and primary calculations. They must load a completed run by path, call `verify`, and fail if required artifacts are absent or their config hash disagrees.

### `01_data_and_partition_audit.py`

This notebook will:

- Display source counts before and after exclusions.
- Show endpoint distributions by source and partition.
- Audit temporal ID ranges for ExpansionRx or whole-cluster assignments for a rotated target.
- Show overlap/exclusion accounting and test that selection/evaluation structures are disjoint.
- Summarise full-pool source composition and label distributions.

### `02_boostin_pruning_analysis.py`

This notebook will:

- Plot influence-score distributions by source, origin class, and label.
- Quantify score association with labels, residuals, source identity, and target similarity without asserting causality.
- Plot pruning curves by pool, direction, and batch size against the random-draw distribution.
- Compare ranked removals with label/source-stratified control policies, displaying the matching-quality diagnostics (Wasserstein distances, bins used, intended composition) alongside every control comparison.
- Show per-seed effects, paired bootstrap distributions, and source composition of mixed batches.
- Render selected molecular structures only from saved canonical SMILES and identifiers; no model calls.
- Export figures to a notebook-selected `analysis/<run-id>/figures/` directory while leaving immutable raw run artifacts untouched; `results/` receives no notebook writes.

No notebook may call XGBoost, `tree_influence`, data-standardisation code, partition generation, or bootstrap recomputation. If an essential plot is impossible from the saved artifacts, extend the script artifact contract and rerun—do not repair the gap interactively in the notebook.

## Test Plan

Tests must use synthetic frames and temporary directories only. They must not scan `~/datasets`, use a real `paths.local.yaml`, or touch a pre-existing `results/` directory. The forbidden `nuclear_receptor_hts` directory must never be enumerated or read.

### Configuration and contracts

- Parse every committed configuration.
- Reject invalid target/donor combinations, duplicate seeds, impossible batch sizes, unknown split mode, and invalid label-match bins.
- Verify HLM contract resolution applies Polaris scale factor before log transform and fails closed without the documented override.
- Verify the frozen Morgan/XGBoost values appear identically in every HLM rotation config.

### Partitions and leakage

- Verify ExpansionRx temporal halves are non-empty, ordered, contiguous in rank, and disjoint by source-qualified ID and InChIKey.
- Verify the selection half is earlier than the evaluation half.
- Verify Butina halves preserve whole clusters and are deterministic under a fixed seed.
- Verify rotating the target changes the target source and donor sources correctly.
- Verify scoring functions cannot receive an evaluation partition by type/signature: `BoostInAttributor.score(training, selection, ...)` has no evaluation argument.
- Verify evaluation rows are inaccessible to arm planning and scoring code paths.

### Attribution and arm construction

- Verify `tree_influence` absence raises a targeted environment error.
- Verify the local-influence matrix has expected orientation and shape.
- Verify synthetic deletion calibration establishes the recorded high/low orientation.
- Verify every full-training candidate has one score per seed and one aggregate rank.
- Verify global score ranks are unique and stable under ties using candidate ID.
- Verify deterministic high/low batches have exact membership and size.
- Verify all random draws are reproducible and contain no duplicates.
- Verify each label/source-stratified control preserves source counts and source×bin composition, reports non-zero continuous-label Wasserstein distances rather than hiding them, never overlaps the ranked arm, and that an infeasible control is recorded with an explicit diagnostic rather than replaced by random selection.
- Verify `mixed_pool` reports its source composition and never assumes source balance.

### Fitting, metrics, and artifacts

- Verify every pruned training frame equals full training minus exactly the manifest membership.
- Verify all conditions use identical evaluation ordering and matched seeds.
- Verify effect sign manually on a tiny fixed prediction example: `RMSE_full - RMSE_pruned`.
- Verify paired bootstrap recomputes RMSE on repeated clusters rather than averaging precomputed molecule errors.
- Verify `bootstrap_draws.parquet` reproduces the stored point and percentile interval.
- Verify all required artifacts exist before `COMPLETED` is written.
- Verify `verify --run` rejects incomplete, hash-mismatched, or schema-inconsistent runs without writing files.
- Run a capped end-to-end dry run that creates all expected artifact types and passes `verify`.

### Commands to run during implementation

```bash
pixi run -e dev pytest
pixi run -e dev ruff check src tests
pixi run -e dev ruff format --check src tests
pixi run -e influence test-influence
```

Real-data commands are deferred until review of the dry-run artifacts:

```bash
pixi run -e influence ipp audit --config configs/boostin_expansionrx_hlm.yaml
pixi run -e influence ipp dry-run --config configs/dryrun_boostin_expansionrx_hlm.yaml
pixi run -e influence ipp benchmark --config configs/boostin_expansionrx_hlm.yaml
pixi run -e influence ipp boostin-prune --config configs/boostin_expansionrx_hlm.yaml
```

## Delivery Sequence

1. Initialise the package, Pixi environments, Gitignore, local-path example, and test scaffolding.
2. Implement endpoint contracts, standardisation, overlap exclusion, and target-rotation data loading; finish audit-only command and tests.
3. Implement temporal and Butina test-half partitions with complete manifests and leakage guards.
4. Implement the pooled BoostIn attributor and synthetic deletion-sign calibration.
5. Implement deterministic ranked, random, and label/source-stratified control pruning-arm planning.
6. Implement matched-seed refitting, paired cluster bootstrap with persisted draws, and immutable artifacts.
7. Implement `verify`, the capped dry run, benchmark, and artifact-schema tests.
8. Build the two Marimo notebooks only after a dry run produces the complete artifact contract.
9. Review dry-run artifacts and benchmark timing before authorising any real-data run.

## Acceptance Criteria

Phase 1 is ready for a real HLM run only when:

- The full test suite and Ruff checks pass in their appropriate environments.
- The dry run has produced and verified every required artifact.
- The score model demonstrably contains native and both donor sources, while neither test half appears in its training data.
- Every deletion arm is provably a removal from the same full pooled membership used for attribution.
- Selection and evaluation surfaces are structurally disjoint.
- Random and label/source-stratified control manifests are reproducible, and their matching-quality diagnostics are auditable.
- The Marimo notebooks load a completed dry run and generate their intended plots without fitting, scoring, splitting, or bootstrapping.
- A benchmark gives a recorded fit count and wall-time projection for the registered configuration.

## Phase 2 Interface Boundary

Do not implement TracIn now. The `TrainingAttributor` interface, common `PruningSplit`, `DeletionArm`, evaluation code, and artifact schema are the deliberate hand-off boundary. Phase 2 may supply a Chemprop/TracIn attributor that writes the same molecule-level score schema and reuses the same pruning and analysis machinery.

If curated CYP data become ready, design the multitask loss and task-specific attribution target before adding model code. In particular, missing task labels, target scaling, source-level assay offsets, checkpoint selection, and whether the influence score is per-task or aggregated must be specified in a separate TracIn implementation document.

## References

1. [Brophy, J.; Hammoudeh, Z.; Lowd, D. Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees. *Journal of Machine Learning Research* 24, 1–48 (2023)](http://jmlr.org/papers/v24/22-0449.html)
2. [TreeInfluence: Influence Estimation for Gradient-Boosted Decision Trees](https://github.com/jjbrophy47/tree_influence)
3. [Chemprop multitask model documentation](https://chemprop.readthedocs.io/en/main/multi_task.html)
4. [Captum influence-function and TracIn API documentation](https://captum.ai/api/influence.html)
