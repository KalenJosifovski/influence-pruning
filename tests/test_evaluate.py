"""Evaluation, metric, and bootstrap recomputation tests."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from influence_pruning.bootstrap import (
    PairedBootstrapResult,
    cluster_bootstrap_delta,
    cluster_bootstrap_median_control,
    cluster_bootstrap_statistic,
)
from influence_pruning.config import BootstrapSettings, ModelSettings, RunConfig
from influence_pruning.data import load_observations
from influence_pruning.evaluate import (
    FULL_POOLED,
    KIND_RANKED_VS_MATCHED_MEDIAN,
    KIND_RANKED_VS_RANDOM_MEDIAN,
    LOCAL_ONLY,
    _control_comparisons,
    _pruned_partition,
    evaluate_split,
)
from influence_pruning.partitions import build_pruning_split, partition
from influence_pruning.pruning_arms import DeletionArm, plan_arms


def _fabricated_scores(split) -> pd.DataFrame:
    """Build deterministic aggregate scores without a model fit."""
    frame = split.full_training.frame.loc[
        :, ["candidate_id", "source", "origin_class", "model_target"]
    ].copy()
    frame["rank_mean"] = -frame["model_target"].astype(float)
    frame = frame.sort_values(["rank_mean", "candidate_id"], kind="stable").reset_index(drop=True)
    frame["aggregate_rank"] = np.arange(1, len(frame) + 1, dtype=int)
    frame["raw_score_mean"] = -frame["rank_mean"]
    return frame


@pytest.fixture()
def evaluated(expansionrx_config: RunConfig):
    """A complete small evaluation over the synthetic target split."""
    bundle = load_observations(expansionrx_config)
    split = build_pruning_split(bundle)
    settings = expansionrx_config.boostin.__class__(
        score_seeds=(0,),
        outcome_seeds=(0,),
        batch_sizes=(5,),
        random_draws=2,
        matching_draws=2,
        label_match_bins=(10, 5, 3),
    )
    plan = plan_arms(split, _fabricated_scores(split), settings, expansionrx_config.donor_sources)
    bootstrap = BootstrapSettings(n_resamples=25, cluster_distance_threshold=0.4, seed=0)
    model = ModelSettings(n_estimators=5, n_jobs=1)
    result = evaluate_split(split, plan.arms, model, (0,), bootstrap)
    return split, plan, result


def test_pruned_training_equals_full_training_minus_manifest(evaluated) -> None:
    """Every pruned condition equals full training minus exactly its manifest membership."""
    split, plan, _ = evaluated
    for arm in plan.arms[:10]:
        pruned = _pruned_partition(split, arm)
        remaining = set(split.full_training.frame["candidate_id"]) - set(arm.removed_candidate_ids)
        assert set(pruned.frame["candidate_id"]) == remaining
        assert pruned.size == split.full_training.size - arm.batch_size


def test_predictions_share_one_evaluation_order(evaluated) -> None:
    """All conditions predict the same molecules in the same order with matched seeds."""
    split, _, result = evaluated
    expected = list(split.evaluation.frame["candidate_id"])
    assert set(result.predictions["condition_kind"]) == {"full_pooled", "local_only", "pruned"}
    for _, group in result.predictions.groupby("condition", sort=False):
        assert list(group["evaluation_id"]) == expected
        assert group["seed"].nunique() == 1
    assert set(result.predictions["observed"]) == set(split.evaluation.frame["model_target"])


def test_metrics_match_manual_rmse_and_mae(evaluated) -> None:
    """Seed metrics equal the manually recomputed RMSE and MAE."""
    _, _, result = evaluated
    predictions = result.predictions
    observed = predictions.loc[predictions["condition"].eq(FULL_POOLED), "observed"].to_numpy()
    predicted = predictions.loc[predictions["condition"].eq(FULL_POOLED), "predicted"].to_numpy()
    row = result.metrics.loc[
        result.metrics["condition"].eq(FULL_POOLED) & result.metrics["seed"].eq(0)
    ].iloc[0]
    assert row["rmse"] == approx(float(np.sqrt(np.mean((observed - predicted) ** 2))))
    assert row["mae"] == approx(float(np.mean(np.abs(observed - predicted))))


def test_full_vs_pruned_effect_sign_and_definition(evaluated) -> None:
    """full_pooled minus arm is positive exactly when the pruned model has lower RMSE."""
    _, plan, result = evaluated
    effects = result.effects.set_index("comparison_id")
    metrics = result.metrics.set_index(["condition", "seed"])["rmse"]
    for arm in plan.arms[:5]:
        full = metrics.loc[(FULL_POOLED, 0)]
        pruned = metrics.loc[(arm.arm_id, 0)]
        comparison = effects.loc[f"full_pooled_vs_{arm.arm_id}"]
        assert comparison["effect"] == approx(full - pruned)
        assert "positive means pruning improved" in comparison["effect_definition"]
    baseline = effects.loc[f"full_pooled_vs_{LOCAL_ONLY}"]
    assert baseline["effect"] == approx(
        metrics.loc[(FULL_POOLED, 0)] - metrics.loc[(LOCAL_ONLY, 0)]
    )


def test_effect_points_are_invariant_to_evaluation_row_order(
    expansionrx_config: RunConfig,
) -> None:
    """Shuffling the evaluation frame must not change any paired point estimate.

    Interval endpoints may move when tied synthetic fingerprints cluster differently under
    permutation; the alignment contract under test is that observed targets, clusters, and
    prediction rows stay matched, which preserves every point estimate exactly.
    """
    bundle = load_observations(expansionrx_config)
    split = build_pruning_split(bundle)
    settings = expansionrx_config.boostin.__class__(
        score_seeds=(0,),
        outcome_seeds=(0,),
        batch_sizes=(5,),
        random_draws=1,
        matching_draws=1,
        label_match_bins=(10, 5),
    )
    plan = plan_arms(split, _fabricated_scores(split), settings, expansionrx_config.donor_sources)
    bootstrap = BootstrapSettings(n_resamples=25, cluster_distance_threshold=0.4, seed=0)
    model = ModelSettings(n_estimators=5, n_jobs=1)
    original = evaluate_split(split, plan.arms, model, (0,), bootstrap)
    permutation = np.random.default_rng(3).permutation(len(split.evaluation.frame))
    shuffled = replace(
        split,
        evaluation=partition(
            "evaluation", split.evaluation.frame.iloc[permutation].reset_index(drop=True)
        ),
    )
    second = evaluate_split(shuffled, plan.arms, model, (0,), bootstrap)
    left = original.effects.set_index("comparison_id").sort_index()
    right = second.effects.set_index("comparison_id").sort_index()
    assert list(left.index) == list(right.index)
    assert left["effect"].to_numpy() == approx(right["effect"].to_numpy())
    assert left["n_molecules"].tolist() == right["n_molecules"].tolist()


def test_bootstrap_draws_reproduce_the_stored_interval(evaluated) -> None:
    """Persisted bootstrap draws reproduce every stored percentile interval."""
    _, _, result = evaluated
    for comparison_id, group in result.bootstrap_draws.groupby("comparison_id", sort=False):
        row = result.effects.loc[result.effects["comparison_id"].eq(comparison_id)].iloc[0]
        low, high = np.percentile(group["effect"].to_numpy(), [2.5, 97.5])
        assert row["ci_low"] == approx(float(low))
        assert row["ci_high"] == approx(float(high))
        assert row["n_resamples"] == len(group)


def test_bootstrap_recomputes_rmse_on_resampled_clusters() -> None:
    """The paired statistic recomputes RMSE per resample rather than averaging errors."""
    rng = np.random.default_rng(0)
    observed = rng.standard_normal(40)
    predictions_a = observed[:, None] + 0.1 * rng.standard_normal((40, 1))
    predictions_b = observed[:, None] + 0.3 * rng.standard_normal((40, 1))
    clusters = np.repeat(np.arange(8), 5)

    def rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.sqrt(np.mean((targets - predictions) ** 2)))

    def statistic(index: np.ndarray) -> float:
        return rmse(observed[index], predictions_a[index, 0]) - rmse(
            observed[index], predictions_b[index, 0]
        )

    result = cluster_bootstrap_delta(
        observed,
        predictions_a,
        predictions_b,
        clusters,
        comparison_id="check",
        n_resamples=17,
        seed=3,
    )
    assert isinstance(result, PairedBootstrapResult)
    point, manual_draws = cluster_bootstrap_statistic(statistic, clusters, n_resamples=17, seed=3)
    assert result.interval.effect == approx(point)
    assert result.draws == approx(manual_draws)
    # A statistic that averaged precomputed molecule errors would not match.
    precomputed = (
        np.abs(observed - predictions_a[:, 0]).mean()
        - np.abs(observed - predictions_b[:, 0]).mean()
    )
    assert not np.isclose(result.interval.effect, precomputed)


def test_bootstrap_effect_positive_when_condition_b_is_better() -> None:
    """With A as the worse model, the paired effect is positive."""
    observed = np.zeros(30)
    worse = np.full((30, 1), 1.0)
    better = np.full((30, 1), 0.1)
    clusters = np.repeat(np.arange(10), 3)
    result = cluster_bootstrap_delta(
        observed,
        worse,
        better,
        clusters,
        comparison_id="a_vs_b",
        n_resamples=10,
        seed=0,
    )
    assert result.interval.effect == approx(0.9)
    assert result.interval.n_clusters == 10


def test_median_control_statistic_uses_every_draw() -> None:
    """The point estimate is the median over all controls, not any single draw's effect.

    Control RMSEs are 1, 2, 4, and 10 against a ranked RMSE of 0.5, so the median-control
    statistic is 2.5 while the individual control effects are 0.5, 1.5, 3.5, and 9.5. No
    individual draw reproduces the estimate.
    """
    observed = np.zeros(12)
    ranked = np.full((12, 1), 0.5)
    control_values = (1.0, 2.0, 4.0, 10.0)
    controls = np.stack([np.full((12, 1), value) for value in control_values], axis=0)
    clusters = np.repeat(np.arange(4), 3)
    result = cluster_bootstrap_median_control(
        observed,
        ranked,
        controls,
        clusters,
        comparison_id="median_check",
        n_resamples=10,
        seed=0,
    )
    median_control_rmse = float(np.median(control_values))
    expected = median_control_rmse - 0.5
    assert result.interval.effect == approx(expected)
    individual_effects = [value - 0.5 for value in control_values]
    assert all(abs(effect - expected) > 1e-9 for effect in individual_effects)
    assert all(abs(value - median_control_rmse) > 1e-9 for value in control_values)


def test_median_control_bootstrap_recomputes_inside_each_resample() -> None:
    """Every resample recomputes each control's RMSE and the median across controls."""
    rng = np.random.default_rng(5)
    observed = rng.standard_normal(24)
    clusters = np.repeat(np.arange(6), 4)
    ranked = observed[:, None] + 0.15 * rng.standard_normal((24, 1))
    controls = np.stack(
        [
            (observed + offset + 0.1 * rng.standard_normal(24))[:, None]
            for offset in (0.1, 0.4, 0.9)
        ],
        axis=0,
    )

    def manual_statistic(index: np.ndarray) -> float:
        ranked_rmse = np.sqrt(np.mean((observed[index, None] - ranked[index]) ** 2, axis=0)).mean()
        control_rmses = np.array(
            [
                np.sqrt(
                    np.mean((observed[index, None] - controls[draw, index, :]) ** 2, axis=0)
                ).mean()
                for draw in range(controls.shape[0])
            ]
        )
        return float(np.median(control_rmses) - ranked_rmse)

    point, manual_draws = cluster_bootstrap_statistic(
        manual_statistic, clusters, n_resamples=13, seed=2
    )
    result = cluster_bootstrap_median_control(
        observed,
        ranked,
        controls,
        clusters,
        comparison_id="recompute_check",
        n_resamples=13,
        seed=2,
    )
    assert result.interval.effect == approx(point)
    assert result.draws == approx(manual_draws)
    assert not np.allclose(result.draws, result.interval.effect)


def test_control_comparison_rows_list_every_draw_and_use_the_median(evaluated) -> None:
    """Stored policy comparisons cover every draw and use the median control RMSE."""
    _, plan, result = evaluated
    metrics = result.metrics.pivot_table(index="condition", columns="seed", values="rmse")
    effects = result.effects.set_index("comparison_id")
    for arm in plan.arms:
        if arm.policy not in ("boostin_high", "boostin_low"):
            continue
        key = (arm.eligible_pool, arm.stratum, arm.batch_size)
        random_controls = sorted(
            candidate.arm_id
            for candidate in plan.arms
            if candidate.policy == "random"
            and (candidate.eligible_pool, candidate.stratum, candidate.batch_size) == key
        )
        comparison = effects.loc[f"{arm.arm_id}_vs_random_median"]
        listed = json.loads(comparison["control_arm_ids"])
        assert comparison["comparison_kind"] == KIND_RANKED_VS_RANDOM_MEDIAN
        assert comparison["n_control_draws"] == len(random_controls)
        assert listed == random_controls
        expected = float(
            np.median([metrics.loc[control].mean() for control in listed])
            - metrics.loc[arm.arm_id].mean()
        )
        assert comparison["effect"] == approx(expected)

        matched_controls = sorted(
            candidate.arm_id
            for candidate in plan.arms
            if candidate.policy == f"label_source_matched_{arm.direction}"
            and (candidate.eligible_pool, candidate.stratum, candidate.batch_size) == key
        )
        if matched_controls:
            matched = effects.loc[f"{arm.arm_id}_vs_matched_median"]
            assert matched["comparison_kind"] == KIND_RANKED_VS_MATCHED_MEDIAN
            assert matched["n_control_draws"] == len(matched_controls)
            assert json.loads(matched["control_arm_ids"]) == matched_controls
            assert matched["control_policy"] == f"label_source_matched_{arm.direction}"
            expected_matched = float(
                np.median([metrics.loc[control].mean() for control in matched_controls])
                - metrics.loc[arm.arm_id].mean()
            )
            assert matched["effect"] == approx(expected_matched)
        else:
            assert f"{arm.arm_id}_vs_matched_median" not in effects.index


def test_ranked_arm_without_matched_controls_has_no_matched_comparison() -> None:
    """A ranked arm with no feasible controls produces no comparison for that policy."""

    def arm(
        arm_id: str,
        policy: str,
        direction: str | None = None,
        draw: int | None = None,
    ) -> DeletionArm:
        return DeletionArm(
            arm_id=arm_id,
            eligible_pool="mixed_pool",
            stratum="all",
            policy=policy,
            direction=direction,
            batch_size=5,
            draw_number=draw,
            removed_candidate_ids=tuple(f"c{index}" for index in range(5)),
        )

    ranked_high = arm("mixed_pool:all:boostin_high:k5", "boostin_high", "high")
    ranked_low = arm("mixed_pool:all:boostin_low:k5", "boostin_low", "low")
    random_arms = tuple(
        arm(f"mixed_pool:all:random:k5:draw{draw:02d}", "random", draw=draw) for draw in range(3)
    )
    comparisons = _control_comparisons((ranked_high, ranked_low, *random_arms))
    assert [comparison.kind for comparison in comparisons] == [
        KIND_RANKED_VS_RANDOM_MEDIAN,
        KIND_RANKED_VS_RANDOM_MEDIAN,
    ]
    assert [control.arm_id for control in comparisons[0].controls] == [
        f"mixed_pool:all:random:k5:draw{draw:02d}" for draw in range(3)
    ]

    matched_high = arm(
        "mixed_pool:all:label_source_matched_high:k5:draw00",
        "label_source_matched_high",
        "high",
        draw=0,
    )
    comparisons = _control_comparisons((ranked_high, ranked_low, *random_arms, matched_high))
    assert [comparison.kind for comparison in comparisons] == [
        KIND_RANKED_VS_RANDOM_MEDIAN,
        KIND_RANKED_VS_MATCHED_MEDIAN,
        KIND_RANKED_VS_RANDOM_MEDIAN,
    ]
    assert comparisons[1].ranked_arm.arm_id == ranked_high.arm_id
    assert comparisons[1].control_policy == "label_source_matched_high"
