"""Matched-seed refitting, predictions, metrics, and paired comparison effects.

``evaluate`` owns the only model refits. Every condition is fitted with the same XGBoost
parameters and the same matched seed set: the full pooled baseline, the local-only practical
reference, and every planned deletion arm. Evaluation clustering uses the fixed Butina
threshold once per run, and every paired comparison is bootstrapped by resampling evaluation
clusters with all resampled effects persisted.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from influence_pruning.bootstrap import (
    PairedBootstrapResult,
    cluster_bootstrap_delta,
    cluster_bootstrap_median_control,
)
from influence_pruning.config import BootstrapSettings, ModelSettings
from influence_pruning.model import fit_predict
from influence_pruning.partitions import LabelledPartition, PruningSplit, partition
from influence_pruning.pruning_arms import POLICY_HIGH, POLICY_LOW, POLICY_RANDOM, DeletionArm
from influence_pruning.similarity import cluster_labels

FULL_POOLED = "full_pooled"
LOCAL_ONLY = "local_only"

PREDICTION_COLUMNS = (
    "condition",
    "condition_kind",
    "arm_id",
    "eligible_pool",
    "stratum",
    "policy",
    "direction",
    "batch_size",
    "draw_number",
    "seed",
    "evaluation_id",
    "original_id",
    "inchikey",
    "cluster_id",
    "observed",
    "predicted",
)
METRIC_COLUMNS = (
    "condition",
    "condition_kind",
    "arm_id",
    "eligible_pool",
    "stratum",
    "policy",
    "direction",
    "batch_size",
    "draw_number",
    "seed",
    "n_eval",
    "rmse",
    "mae",
)
EFFECT_COLUMNS = (
    "comparison_id",
    "comparison_kind",
    "condition_a",
    "condition_b",
    "eligible_pool",
    "stratum",
    "policy",
    "direction",
    "batch_size",
    "draw_number",
    "control_policy",
    "n_control_draws",
    "control_arm_ids",
    "effect",
    "effect_definition",
    "ci_low",
    "ci_high",
    "n_resamples",
    "n_molecules",
    "n_clusters",
    "settings",
)
BOOTSTRAP_DRAW_COLUMNS = ("comparison_id", "resample_index", "effect")
KIND_FULL_VS_PRUNED = "full_vs_pruned"
KIND_BASELINE = "baseline"
KIND_RANKED_VS_RANDOM_MEDIAN = "ranked_vs_random_median"
KIND_RANKED_VS_MATCHED_MEDIAN = "ranked_vs_matched_median"


@dataclass(frozen=True, slots=True)
class _ControlComparison:
    """One policy-level comparison of a ranked arm against every control draw.

    :param comparison_id: stable comparison identifier.
    :param kind: ``ranked_vs_random_median`` or ``ranked_vs_matched_median``.
    :param ranked_arm: the ranked deletion arm being compared.
    :param control_policy: policy identifier shared by every control draw.
    :param controls: every recorded control draw, ordered by draw number and arm ID.
    """

    comparison_id: str
    kind: str
    ranked_arm: DeletionArm
    control_policy: str
    controls: tuple[DeletionArm, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Predictions, metrics, paired effects, and every bootstrap resample.

    :param predictions: one row per evaluation molecule × condition × seed.
    :param metrics: RMSE and MAE per condition × seed.
    :param effects: point estimates and percentile intervals per paired comparison.
    :param bootstrap_draws: one row per comparison × resample.
    """

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    effects: pd.DataFrame
    bootstrap_draws: pd.DataFrame


def evaluate_split(
    split: PruningSplit,
    arms: tuple[DeletionArm, ...],
    model_settings: ModelSettings,
    outcome_seeds: tuple[int, ...],
    bootstrap_settings: BootstrapSettings,
) -> EvaluationResult:
    """Refit every condition and evaluate on the held-out target-domain surface.

    :param split: pruning split.
    :param arms: planned deletion arms.
    :param model_settings: model parameters shared by every condition.
    :param outcome_seeds: matched model seeds used for every refit.
    :param bootstrap_settings: cluster-bootstrap settings.
    :returns: complete evaluation result.
    """
    evaluation = split.evaluation
    observed = evaluation.frame["model_target"].to_numpy(dtype=float)
    evaluation_order = evaluation.frame["candidate_id"].astype(str).to_numpy()
    clusters = cluster_labels(
        evaluation.fingerprints, bootstrap_settings.cluster_distance_threshold
    )
    local_only = _local_only_partition(split)
    conditions: list[tuple[str, str, LabelledPartition, DeletionArm | None]] = [
        (FULL_POOLED, "full_pooled", split.full_training, None),
        (LOCAL_ONLY, "local_only", local_only, None),
    ]
    conditions.extend((arm.arm_id, "pruned", _pruned_partition(split, arm), arm) for arm in arms)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for condition, kind, training, arm in conditions:
        for seed in outcome_seeds:
            predicted = fit_predict(
                training.fingerprints,
                training.frame["model_target"].to_numpy(dtype=float),
                evaluation.fingerprints,
                model_settings,
                seed,
            )
            prediction_frames.append(
                _prediction_frame(
                    evaluation, observed, clusters, condition, kind, arm, seed, predicted
                )
            )
            metric_rows.append(
                {
                    "condition": condition,
                    "condition_kind": kind,
                    "arm_id": None if arm is None else arm.arm_id,
                    "eligible_pool": None if arm is None else arm.eligible_pool,
                    "stratum": None if arm is None else arm.stratum,
                    "policy": None if arm is None else arm.policy,
                    "direction": None if arm is None else arm.direction,
                    "batch_size": None if arm is None else arm.batch_size,
                    "draw_number": None if arm is None else arm.draw_number,
                    "seed": seed,
                    "n_eval": int(len(evaluation.frame)),
                    "rmse": _rmse(observed, predicted),
                    "mae": _mae(observed, predicted),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True).loc[:, list(PREDICTION_COLUMNS)]
    metrics = pd.DataFrame(metric_rows, columns=list(METRIC_COLUMNS))
    effects, draws = _paired_effects(
        arms=arms,
        observed=observed,
        clusters=clusters,
        evaluation_order=evaluation_order,
        predictions=predictions,
        outcome_seeds=outcome_seeds,
        bootstrap_settings=bootstrap_settings,
    )
    return EvaluationResult(
        predictions=predictions, metrics=metrics, effects=effects, bootstrap_draws=draws
    )


def _local_only_partition(split: PruningSplit) -> LabelledPartition:
    """Return the native-target-only training partition as a practical reference."""
    frame = split.full_training.frame
    native = frame.loc[frame["origin_class"].astype(str).eq("native")].reset_index(drop=True)
    if native.empty:
        raise ValueError("local-only reference requires native target training rows")
    native["fingerprint_row"] = np.arange(len(native), dtype=int)
    return partition(LOCAL_ONLY, native)


def _pruned_partition(split: PruningSplit, arm: DeletionArm) -> LabelledPartition:
    """Return full training minus exactly the arm's manifest membership."""
    mask = ~split.full_training.frame["candidate_id"].astype(str).isin(
        set(arm.removed_candidate_ids)
    )
    pruned = split.full_training.frame.loc[mask].reset_index(drop=True)
    expected = split.full_training.size - arm.batch_size
    if len(pruned) != expected:
        raise ValueError(
            f"pruned training for {arm.arm_id} has {len(pruned)} rows, expected {expected}"
        )
    return partition(f"pruned:{arm.arm_id}", pruned)


def _prediction_frame(
    evaluation: LabelledPartition,
    observed: np.ndarray,
    clusters: np.ndarray,
    condition: str,
    kind: str,
    arm: DeletionArm | None,
    seed: int,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Build the prediction rows for one condition and seed."""
    frame = evaluation.frame
    return pd.DataFrame(
        {
            "condition": condition,
            "condition_kind": kind,
            "arm_id": None if arm is None else arm.arm_id,
            "eligible_pool": None if arm is None else arm.eligible_pool,
            "stratum": None if arm is None else arm.stratum,
            "policy": None if arm is None else arm.policy,
            "direction": None if arm is None else arm.direction,
            "batch_size": None if arm is None else arm.batch_size,
            "draw_number": None if arm is None else arm.draw_number,
            "seed": seed,
            "evaluation_id": frame["candidate_id"].astype(str).to_numpy(),
            "original_id": frame["original_id"].astype(str).to_numpy(),
            "inchikey": frame["inchikey"].astype(str).to_numpy(),
            "cluster_id": clusters,
            "observed": observed,
            "predicted": predicted,
        }
    )


def _paired_effects(
    *,
    arms: tuple[DeletionArm, ...],
    observed: np.ndarray,
    clusters: np.ndarray,
    evaluation_order: np.ndarray,
    predictions: pd.DataFrame,
    outcome_seeds: tuple[int, ...],
    bootstrap_settings: BootstrapSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build every paired comparison and its persisted bootstrap distribution.

    Individual ``full_pooled_vs_<arm>`` effects cover every arm. Policy-level comparisons put
    each ranked arm against the median of *all* recorded random draws, and against the median
    of all recorded label/source-stratified control draws of the same direction. No control is
    selected by its observed effect. Prediction matrices are reindexed to the evaluation frame
    order so `observed` and `clusters` remain aligned with every condition's rows.
    """
    matrix = _condition_matrix(predictions, outcome_seeds, evaluation_order)

    settings_label = (
        f"cluster_butina={bootstrap_settings.cluster_distance_threshold};"
        f"n_resamples={bootstrap_settings.n_resamples};seed={bootstrap_settings.seed}"
    )
    effect_rows: list[dict[str, Any]] = []
    draw_rows: list[pd.DataFrame] = []

    def record(result: PairedBootstrapResult, **fields: Any) -> None:
        effect_rows.append(_effect_row(result, **fields))
        draw_rows.append(
            pd.DataFrame(
                {
                    "comparison_id": fields["comparison_id"],
                    "resample_index": np.arange(bootstrap_settings.n_resamples, dtype=int),
                    "effect": result.draws,
                }
            )
        )

    record(
        cluster_bootstrap_delta(
            observed,
            matrix[FULL_POOLED],
            matrix[LOCAL_ONLY],
            clusters,
            comparison_id=f"full_pooled_vs_{LOCAL_ONLY}",
            n_resamples=bootstrap_settings.n_resamples,
            seed=bootstrap_settings.seed,
            settings=settings_label,
            effect_definition=_effect_definition(KIND_BASELINE, FULL_POOLED, LOCAL_ONLY),
        ),
        comparison_id=f"full_pooled_vs_{LOCAL_ONLY}",
        kind=KIND_BASELINE,
        condition_a=FULL_POOLED,
        condition_b=LOCAL_ONLY,
        arm=None,
    )
    for arm in arms:
        record(
            cluster_bootstrap_delta(
                observed,
                matrix[FULL_POOLED],
                matrix[arm.arm_id],
                clusters,
                comparison_id=f"full_pooled_vs_{arm.arm_id}",
                n_resamples=bootstrap_settings.n_resamples,
                seed=bootstrap_settings.seed,
                settings=settings_label,
                effect_definition=_effect_definition(KIND_FULL_VS_PRUNED, FULL_POOLED, arm.arm_id),
            ),
            comparison_id=f"full_pooled_vs_{arm.arm_id}",
            kind=KIND_FULL_VS_PRUNED,
            condition_a=FULL_POOLED,
            condition_b=arm.arm_id,
            arm=arm,
        )

    for comparison in _control_comparisons(arms):
        controls = np.stack([matrix[control.arm_id] for control in comparison.controls], axis=0)
        record(
            cluster_bootstrap_median_control(
                observed,
                matrix[comparison.ranked_arm.arm_id],
                controls,
                clusters,
                comparison_id=comparison.comparison_id,
                n_resamples=bootstrap_settings.n_resamples,
                seed=bootstrap_settings.seed,
                settings=settings_label,
                effect_definition=_effect_definition(
                    comparison.kind, comparison.ranked_arm.arm_id, comparison.comparison_id
                ),
            ),
            comparison_id=comparison.comparison_id,
            kind=comparison.kind,
            condition_a=f"median_control[{comparison.control_policy}]",
            condition_b=comparison.ranked_arm.arm_id,
            arm=comparison.ranked_arm,
            control_policy=comparison.control_policy,
            control_arms=comparison.controls,
        )

    effects = pd.DataFrame(effect_rows, columns=list(EFFECT_COLUMNS))
    draws = (
        pd.concat(draw_rows, ignore_index=True).loc[:, list(BOOTSTRAP_DRAW_COLUMNS)]
        if draw_rows
        else pd.DataFrame(columns=list(BOOTSTRAP_DRAW_COLUMNS))
    )
    return effects, draws


def _control_comparisons(arms: tuple[DeletionArm, ...]) -> list[_ControlComparison]:
    """Group every recorded control draw for each ranked arm's policy-level comparisons.

    Every random draw of the same ``eligible_pool × stratum × batch_size`` enters the random
    comparison, and every stratified-control draw of the same group and direction enters the
    matched comparison. Draws that could not be constructed are recorded as infeasible in the
    arm plan and are absent here, so a ranked arm with no feasible controls produces no
    comparison for that policy rather than a malformed one.
    """
    random_by_group: dict[tuple[str, int], list[DeletionArm]] = {}
    matched_by_group: dict[tuple[str, int, str], list[DeletionArm]] = {}
    for arm in arms:
        group = (f"{arm.eligible_pool}:{arm.stratum}", arm.batch_size)
        if arm.policy == POLICY_RANDOM:
            random_by_group.setdefault(group, []).append(arm)
        elif arm.policy in ("label_source_matched_high", "label_source_matched_low"):
            matched_by_group.setdefault((group[0], group[1], str(arm.direction)), []).append(arm)

    comparisons: list[_ControlComparison] = []
    for arm in arms:
        if arm.policy not in (POLICY_HIGH, POLICY_LOW):
            continue
        group = (f"{arm.eligible_pool}:{arm.stratum}", arm.batch_size)
        random_controls = _ordered_controls(random_by_group.get(group, []))
        if random_controls:
            comparisons.append(
                _ControlComparison(
                    comparison_id=f"{arm.arm_id}_vs_random_median",
                    kind=KIND_RANKED_VS_RANDOM_MEDIAN,
                    ranked_arm=arm,
                    control_policy=POLICY_RANDOM,
                    controls=random_controls,
                )
            )
        matched_controls = _ordered_controls(
            matched_by_group.get((group[0], group[1], str(arm.direction)), [])
        )
        if matched_controls:
            comparisons.append(
                _ControlComparison(
                    comparison_id=f"{arm.arm_id}_vs_matched_median",
                    kind=KIND_RANKED_VS_MATCHED_MEDIAN,
                    ranked_arm=arm,
                    control_policy=matched_controls[0].policy,
                    controls=matched_controls,
                )
            )
    return comparisons


def _ordered_controls(arms: list[DeletionArm]) -> tuple[DeletionArm, ...]:
    """Return control draws in stable draw order for the artifact."""
    return tuple(
        sorted(
            arms,
            key=lambda arm: (
                arm.draw_number if arm.draw_number is not None else -1,
                arm.arm_id,
            ),
        )
    )


def _condition_matrix(
    predictions: pd.DataFrame,
    outcome_seeds: tuple[int, ...],
    evaluation_order: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return condition → (n_eval, n_seeds) matrices in the evaluation frame's row order.

    :param predictions: one row per condition × seed × evaluation molecule.
    :param outcome_seeds: matched model seeds present in every condition.
    :param evaluation_order: evaluation candidate identifiers in frame order; prediction rows
        are realigned to this order so paired statistics never mix molecules.
    :raises ValueError: a condition is missing a seed or an evaluation molecule.
    """
    order = np.asarray(evaluation_order, dtype=str)
    matrices: dict[str, np.ndarray] = {}
    for condition, group in predictions.groupby("condition", sort=False):
        wide = group.pivot(index="evaluation_id", columns="seed", values="predicted")
        missing = [seed for seed in outcome_seeds if seed not in wide.columns]
        if missing:
            raise ValueError(f"condition {condition} is missing seeds {missing}")
        absent = [molecule for molecule in order if molecule not in wide.index]
        if absent:
            raise ValueError(f"condition {condition} is missing evaluation molecules {absent[:3]}")
        matrices[str(condition)] = wide.loc[order, list(outcome_seeds)].to_numpy(dtype=float)
    return matrices


def _effect_definition(kind: str, condition_a: str, condition_b: str) -> str:
    """Return the human-readable paired statistic for one comparison."""
    if kind == KIND_FULL_VS_PRUNED:
        return f"rmse({condition_a}) - rmse({condition_b}); positive means pruning improved"
    if kind == KIND_RANKED_VS_RANDOM_MEDIAN:
        return (
            "median over all random draws of rmse(random draw) - rmse(ranked); positive means "
            "the ranked arm beats the median random-removal policy"
        )
    if kind == KIND_RANKED_VS_MATCHED_MEDIAN:
        return (
            "median over all label/source-stratified control draws of rmse(control draw) - "
            "rmse(ranked); positive means the ranked arm beats the median stratified-control "
            "policy"
        )
    return f"rmse({condition_a}) - rmse({condition_b})"


def _effect_row(
    result: PairedBootstrapResult,
    *,
    comparison_id: str,
    kind: str,
    condition_a: str,
    condition_b: str,
    arm: DeletionArm | None,
    control_policy: str | None = None,
    control_arms: tuple[DeletionArm, ...] = (),
) -> dict[str, Any]:
    """Build one effects-table row."""
    interval = result.interval
    return {
        "comparison_id": comparison_id,
        "comparison_kind": kind,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "eligible_pool": None if arm is None else arm.eligible_pool,
        "stratum": None if arm is None else arm.stratum,
        "policy": None if arm is None else arm.policy,
        "direction": None if arm is None else arm.direction,
        "batch_size": None if arm is None else arm.batch_size,
        "draw_number": None if arm is None else arm.draw_number,
        "control_policy": control_policy,
        "n_control_draws": len(control_arms) if control_arms else None,
        "control_arm_ids": (
            json.dumps([control.arm_id for control in control_arms]) if control_arms else None
        ),
        "effect": interval.effect,
        "effect_definition": interval.effect_definition,
        "ci_low": interval.ci_low,
        "ci_high": interval.ci_high,
        "n_resamples": interval.n_resamples,
        "n_molecules": interval.n_molecules,
        "n_clusters": interval.n_clusters,
        "settings": interval.settings,
    }


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def _mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(observed - predicted)))
