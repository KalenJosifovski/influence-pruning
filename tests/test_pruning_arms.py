"""Deletion-arm planning tests: ranked, random, matched, and infeasible controls."""

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from influence_pruning.config import BoostInSettings
from influence_pruning.errors import ConfigError, RunError
from influence_pruning.partitions import LabelledPartition, PruningSplit
from influence_pruning.pruning_arms import (
    ArmPlan,
    DeletionArm,
    eligible_pools,
    plan_arms,
)
from influence_pruning.run import infeasible_matched_groups


def _split(frame: pd.DataFrame) -> PruningSplit:
    """Wrap one frame as a full-training split with dummy fingerprints."""
    n = len(frame)
    full = LabelledPartition(
        name="full_training",
        frame=frame.reset_index(drop=True),
        fingerprints=np.zeros((n, 4), dtype=np.uint8),
    )
    stub = LabelledPartition(
        name="stub",
        frame=frame.iloc[:1].reset_index(drop=True),
        fingerprints=np.zeros((1, 4), dtype=np.uint8),
    )
    return PruningSplit(
        full_training=full,
        influence_selection=stub,
        evaluation=stub,
        partition_manifest=pd.DataFrame(),
        split_record={},
    )


def _frame(labels: list[float], source: str = "biogen") -> pd.DataFrame:
    """Build a donor frame with the columns planning consumes."""
    return pd.DataFrame(
        {
            "candidate_id": [f"{source}:id{index}:KEY{index}" for index in range(len(labels))],
            "source": source,
            "origin_class": "donor",
            "original_id": [f"id{index}" for index in range(len(labels))],
            "inchikey": [f"KEY{index}" for index in range(len(labels))],
            "model_target": labels,
        }
    )


def _scores(frame: pd.DataFrame, rank_mean: np.ndarray | None = None) -> pd.DataFrame:
    """Build an aggregate score frame aligned with `frame`."""
    if rank_mean is None:
        rank_mean = np.arange(len(frame), dtype=float)
    return pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"].to_numpy(),
            "source": frame["source"].to_numpy(),
            "origin_class": frame["origin_class"].to_numpy(),
            "model_target": frame["model_target"].to_numpy(dtype=float),
            "aggregate_rank": np.arange(1, len(frame) + 1, dtype=int),
            "rank_mean": rank_mean,
            "raw_score_mean": -rank_mean,
        }
    )


def test_ranked_batches_have_exact_deterministic_membership() -> None:
    """High and low batches are the extreme pool ranks with exact sizes."""
    labels = [float(index % 8) for index in range(40)]
    frame = _frame(labels)
    scores = _scores(frame)
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=2, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    plan = plan_arms(_split(frame), scores, settings, ("biogen",))
    high = next(arm for arm in plan.arms if arm.arm_id == "donor_only:biogen:boostin_high:k5")
    low = next(arm for arm in plan.arms if arm.arm_id == "donor_only:biogen:boostin_low:k5")
    assert high.removed_candidate_ids == tuple(f"biogen:id{index}:KEY{index}" for index in range(5))
    assert low.removed_candidate_ids == tuple(
        f"biogen:id{index}:KEY{index}" for index in range(35, 40)
    )
    assert not (set(high.removed_candidate_ids) & set(low.removed_candidate_ids))
    assert plan.summary.loc[plan.summary["status"].eq("planned"), "achieved_size"].eq(5).all()


def test_random_draws_are_reproducible_without_duplicates() -> None:
    """Random batches are identical across planning runs and contain unique members."""
    frame = _frame([float(index % 5) for index in range(40)])
    scores = _scores(frame)
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=3, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    first = plan_arms(_split(frame), scores, settings, ("biogen",))
    second = plan_arms(_split(frame), scores, settings, ("biogen",))
    random_ids = [arm.arm_id for arm in first.arms if ":random:" in arm.arm_id]
    assert random_ids
    for arm_id in random_ids:
        first_arm = next(arm for arm in first.arms if arm.arm_id == arm_id)
        second_arm = next(arm for arm in second.arms if arm.arm_id == arm_id)
        assert first_arm.removed_candidate_ids == second_arm.removed_candidate_ids
        assert len(set(first_arm.removed_candidate_ids)) == first_arm.batch_size


def test_mixed_pool_matched_controls_preserve_source_and_label_composition() -> None:
    """Matched draws preserve the reference batch's source and label-bin composition."""
    labels: list[float] = []
    sources: list[str] = []
    for source in ("biogen", "polaris", "expansionrx"):
        labels.extend([0.0, 10.0] * 20)
        sources.extend([source] * 40)
    frame = pd.DataFrame(
        {
            "candidate_id": [
                f"{source}:id{index}:KEY{i}"
                for i, (source, index) in enumerate(zip(sources, range(120), strict=True))
            ],
            "source": sources,
            "origin_class": [
                "native" if source == "expansionrx" else "donor" for source in sources
            ],
            "original_id": [f"id{index}" for index in range(120)],
            "inchikey": [f"KEY{index}" for index in range(120)],
            "model_target": labels,
        }
    )
    # Rank the label-10 rows at the top so the high batch is label-extreme.
    rank = np.where(
        np.array(labels) > 5.0, np.arange(120, dtype=float) - 200.0, np.arange(120, dtype=float)
    )
    scores = _scores(frame, rank)
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=1, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    plan = plan_arms(_split(frame), scores, settings, ("biogen", "polaris", "expansionrx"))
    reference = plan.summary.loc[plan.summary["arm_id"].eq("mixed_pool:all:boostin_high:k5")].iloc[
        0
    ]
    matched = plan.summary.loc[
        plan.summary["arm_id"].str.startswith("mixed_pool:all:label_source_matched_high"),
    ]
    planned = matched.loc[matched["status"].eq("planned")]
    assert not planned.empty
    for _, row in planned.iterrows():
        assert row["source_biogen"] == reference["source_biogen"]
        assert row["source_polaris"] == reference["source_polaris"]
        assert row["source_expansionrx"] == reference["source_expansionrx"]
        assert row["label_wasserstein_overall"] == pytest.approx(0.0)
        assert row["matching_bins_used"] == 10
    manifest = plan.manifest
    reference_ids = set(
        manifest.loc[manifest["arm_id"].eq("mixed_pool:all:boostin_high:k5"), "candidate_id"]
    )
    for arm in plan.arms:
        if arm.arm_id.startswith("mixed_pool:all:label_source_matched_high"):
            assert not (set(arm.removed_candidate_ids) & reference_ids)


def test_infeasible_matched_arm_is_recorded_not_substituted() -> None:
    """A match that cannot avoid the scored batch is recorded with a diagnostic only.

    The pool has four separated label groups of ten rows each; a batch of 15 taken from the
    top two groups includes every top-group row, so no binning (quantile edges never collapse
    below three effective bins here) can supply the required top-group count without reusing
    the scored rows.
    """
    labels = [0.0] * 10 + [10.0] * 10 + [20.0] * 10 + [30.0] * 10
    frame = _frame(labels)
    rank = np.arange(40, dtype=float) - 100.0 * np.array(labels)
    scores = _scores(frame, rank)
    settings = BoostInSettings(
        batch_sizes=(15,), random_draws=1, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    plan = plan_arms(_split(frame), scores, settings, ("biogen",))
    infeasible = plan.summary.loc[plan.summary["status"].eq("infeasible")]
    assert not infeasible.empty
    assert infeasible["matching_diagnostic"].str.contains("no configured binning").all()
    infeasible_ids = set(infeasible["arm_id"])
    assert infeasible_ids
    assert all("label_source_matched" in arm_id for arm_id in infeasible_ids)
    assert not set(plan.manifest["arm_id"]) & infeasible_ids
    random_arms = [arm for arm in plan.arms if arm.policy == "random"]
    assert random_arms, "the random reference remains available for the other arms"


def test_capacity_rejects_batch_larger_than_disjoint_pool() -> None:
    """A batch size above floor(pool / 2) is a planning-time capacity failure."""
    frame = _frame([float(index % 4) for index in range(40)])
    settings = BoostInSettings(
        batch_sizes=(25,), random_draws=1, matching_draws=1, label_match_bins=(10, 5, 3)
    )
    with raises(ConfigError, match="batch_sizes exceed eligible-pool capacity"):
        plan_arms(_split(frame), _scores(frame), settings, ("biogen",))


def test_missing_scores_for_a_training_candidate_are_rejected() -> None:
    """Planning refuses to proceed when any full-training row lacks an aggregate score."""
    frame = _frame([float(index % 4) for index in range(40)])
    scores = _scores(frame).iloc[:-1]
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=1, matching_draws=1, label_match_bins=(10, 5, 3)
    )
    with raises(RunError, match="no aggregate score"):
        plan_arms(_split(frame), scores, settings, ("biogen",))


def test_pool_strata_cover_donor_native_and_mixed() -> None:
    """Eligible pools enumerate donor strata, the native pool, and the mixed pool."""
    frame = pd.concat(
        [
            _frame([1.0] * 10, source="biogen"),
            _frame([1.0] * 12, source="polaris"),
        ],
        ignore_index=True,
    )
    frame.loc[:9, "origin_class"] = "donor"
    native = frame.iloc[:5].copy()
    native["source"] = "expansionrx"
    native["origin_class"] = "native"
    combined = pd.concat([frame, native], ignore_index=True)
    split = _split(combined)
    pools = eligible_pools(split, ("biogen", "polaris"))
    keys = {pool.key for pool in pools}
    assert keys == {
        "donor_only:biogen",
        "donor_only:polaris",
        "donor_only:combined",
        "native_only:target",
        "mixed_pool:all",
    }
    donor_combined = next(pool for pool in pools if pool.key == "donor_only:combined")
    assert len(donor_combined.candidate_ids) == 22


def test_plan_manifest_matches_summary_composition() -> None:
    """Summary source counts equal the manifest composition for every planned arm."""
    frame = pd.concat(
        [
            _frame([float(index % 6) for index in range(30)], source="biogen"),
            _frame([float(index % 6) for index in range(30)], source="polaris"),
        ],
        ignore_index=True,
    )
    scores = _scores(frame)
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=1, matching_draws=1, label_match_bins=(10, 5, 3)
    )
    plan = plan_arms(_split(frame), scores, settings, ("biogen", "polaris"))
    for _, row in plan.summary.loc[plan.summary["status"].eq("planned")].iterrows():
        arm_manifest = plan.manifest.loc[plan.manifest["arm_id"].eq(row["arm_id"])]
        assert len(arm_manifest) == row["achieved_size"]
        assert int(arm_manifest["source"].eq("biogen").sum()) == row["source_biogen"]
        assert int(arm_manifest["source"].eq("polaris").sum()) == row["source_polaris"]
        assert row["n_train_pruned"] == row["n_train_full"] - row["batch_size"]


def test_arm_ids_are_unique() -> None:
    """Every planned arm has a unique identifier."""
    frame = _frame([float(index % 5) for index in range(40)])
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=2, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    plan: ArmPlan = plan_arms(_split(frame), _scores(frame), settings, ("biogen",))
    arm_ids = [arm.arm_id for arm in plan.arms]
    assert len(arm_ids) == len(set(arm_ids))
    assert all(isinstance(arm, DeletionArm) for arm in plan.arms)


def test_stratified_control_can_have_nonzero_label_distance_and_reports_it() -> None:
    """A planned source/bin-stratified control is not an exact continuous-label match.

    With distinct continuous labels the finest binnings are infeasible because the scored
    batch owns its label extremes; the control is constructed at a coarser quantile binning,
    so its achieved labels differ from the reference batch. The Wasserstein diagnostics must
    report that distance rather than hiding it.
    """
    labels = [float(index) for index in range(40)]
    frame = _frame(labels)
    scores = _scores(frame, -np.arange(40, dtype=float))
    settings = BoostInSettings(
        batch_sizes=(5,), random_draws=1, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    plan = plan_arms(_split(frame), scores, settings, ("biogen",))
    matched = plan.summary.loc[
        plan.summary["policy"].eq("label_source_matched_high")
        & plan.summary["status"].eq("planned")
    ]
    assert not matched.empty
    assert bool((matched["label_wasserstein_overall"] > 0.0).all())
    assert matched["matching_bins_used"].notna().all()
    assert matched["matching_plan"].notna().all()
    reference = plan.manifest.loc[
        plan.manifest["arm_id"].eq("donor_only:biogen:boostin_high:k5"), "candidate_id"
    ]
    for arm_id in matched["arm_id"]:
        control_members = plan.manifest.loc[plan.manifest["arm_id"].eq(arm_id), "candidate_id"]
        assert len(control_members) == 5
        assert not set(reference) & set(control_members)


def test_infeasible_matched_groups_are_exposed_for_benchmark() -> None:
    """The benchmark helper reports every infeasible stratified-control group explicitly."""
    labels = [0.0] * 10 + [10.0] * 10 + [20.0] * 10 + [30.0] * 10
    frame = _frame(labels)
    rank = np.arange(40, dtype=float) - 100.0 * np.array(labels)
    scores = _scores(frame, rank)
    settings = BoostInSettings(
        batch_sizes=(15,), random_draws=1, matching_draws=2, label_match_bins=(10, 5, 3)
    )
    plan = plan_arms(_split(frame), scores, settings, ("biogen",))
    groups = infeasible_matched_groups(plan)
    assert groups
    for group in groups:
        assert set(group) == {
            "eligible_pool",
            "stratum",
            "direction",
            "batch_size",
            "n_draws",
            "matching_diagnostic",
        }
        assert group["matching_diagnostic"]
        assert group["n_draws"] >= 1
    infeasible_rows = int(plan.summary["status"].eq("infeasible").sum())
    assert sum(group["n_draws"] for group in groups) == infeasible_rows
    assert int(plan.summary["status"].eq("planned").sum()) + infeasible_rows == len(plan.summary)
