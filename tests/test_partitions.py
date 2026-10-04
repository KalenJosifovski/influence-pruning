"""Partition construction, leakage guards, and target-rotation tests."""

import inspect
from dataclasses import replace

import pytest

from influence_pruning.attribution import BoostInAttributor, TrainingAttributor
from influence_pruning.config import RunConfig
from influence_pruning.data import load_observations
from influence_pruning.partitions import (
    build_pruning_split,
    make_butina_test_halves,
    make_temporal_test_halves,
)
from influence_pruning.pruning_arms import plan_arms


@pytest.fixture()
def expansionrx_split(expansionrx_config: RunConfig):
    """Loaded observations and pruning split for the synthetic ExpansionRx target."""
    bundle = load_observations(expansionrx_config)
    return bundle, build_pruning_split(bundle)


def test_temporal_halves_are_ordered_and_disjoint(expansionrx_split) -> None:
    """The earlier test half is the selection surface and the later half is evaluation."""
    _, split = expansionrx_split
    selection = split.influence_selection.frame
    evaluation = split.evaluation.frame
    assert len(selection) > 0 and len(evaluation) > 0
    assert len(selection) + len(evaluation) == 60
    assert set(selection["candidate_id"]) & set(evaluation["candidate_id"]) == set()
    assert set(selection["inchikey"]) & set(evaluation["inchikey"]) == set()
    selection_ranks = split.partition_manifest.loc[
        split.partition_manifest["role"].eq("influence_selection"), "chronology_rank"
    ]
    evaluation_ranks = split.partition_manifest.loc[
        split.partition_manifest["role"].eq("evaluation"), "chronology_rank"
    ]
    training_ranks = split.partition_manifest.loc[
        split.partition_manifest["role"].eq("full_training")
        & split.partition_manifest["origin_class"].eq("native"),
        "chronology_rank",
    ]
    assert selection_ranks.max() < evaluation_ranks.min()
    assert training_ranks.max() < selection_ranks.min()
    assert sorted(selection_ranks.tolist()) == list(
        range(int(selection_ranks.min()), int(selection_ranks.max()) + 1)
    )
    assert sorted(evaluation_ranks.tolist()) == list(
        range(int(evaluation_ranks.min()), int(evaluation_ranks.max()) + 1)
    )
    assert split.split_record["selection_earlier_than_evaluation"] is True
    assert split.split_record["midpoint_count"] == 30


def test_temporal_halves_reject_unparsable_ids(expansionrx_config: RunConfig) -> None:
    """Temporal splitting refuses IDs that cannot be parsed and ordered."""
    bundle = load_observations(expansionrx_config)
    broken = bundle.target_test.copy()
    broken.loc[broken.index[0], "original_id"] = "no-trailing-number"
    patched = replace(bundle, target_test=broken)
    with pytest.raises(Exception, match="parsable"):
        make_temporal_test_halves(patched)


def test_evaluation_is_absent_from_full_training(expansionrx_split) -> None:
    """Full training excludes both test halves; selection and evaluation are disjoint."""
    _, split = expansionrx_split
    training_ids = set(split.full_training.frame["candidate_id"])
    selection_ids = set(split.influence_selection.frame["candidate_id"])
    evaluation_ids = set(split.evaluation.frame["candidate_id"])
    assert not (evaluation_ids & training_ids)
    assert not (selection_ids & training_ids)
    assert not (selection_ids & evaluation_ids)
    assert split.full_training.size == len(training_ids)


def test_butina_halves_preserve_whole_clusters_and_are_deterministic(
    biogen_config: RunConfig,
) -> None:
    """Whole clusters never cross surfaces and allocation is seed-deterministic."""
    bundle = load_observations(biogen_config)
    first = make_butina_test_halves(bundle.target_test, biogen_config.split)
    second = make_butina_test_halves(bundle.target_test, biogen_config.split)
    selection, evaluation, record = first
    assert set(selection.frame["candidate_id"]) == {
        candidate for candidate in second[0].frame["candidate_id"]
    }
    assert record["n_selection"] + record["n_evaluation"] == record["n_test"]
    assert record["n_selection"] > 0 and record["n_evaluation"] > 0
    selection_clusters = set(selection.frame["cluster_id"].astype(int))
    evaluation_clusters = set(evaluation.frame["cluster_id"].astype(int))
    assert selection_clusters & evaluation_clusters == set()
    assert selection_clusters == set(record["selection_cluster_ids"])
    assert evaluation_clusters == set(record["evaluation_cluster_ids"])


def test_rotation_changes_target_and_donor_composition(biogen_config: RunConfig) -> None:
    """Rotating the target moves the configured source into the native role."""
    bundle = load_observations(biogen_config)
    split = build_pruning_split(bundle)
    training = split.full_training.frame
    native_sources = set(training.loc[training["origin_class"].eq("native"), "source"])
    donor_sources = set(training.loc[training["origin_class"].eq("donor"), "source"])
    assert native_sources == {"biogen"}
    assert donor_sources == {"expansionrx", "polaris"}
    assert set(bundle.donors) == {"expansionrx", "polaris"}
    assert split.full_training.size > 0


def test_target_donor_overlap_is_excluded(expansionrx_config: RunConfig) -> None:
    """A donor structure matching target chemistry never enters full training."""
    # The synthetic Biogen frame deliberately reuses two ExpansionRx structures.
    bundle = load_observations(expansionrx_config)
    biogen_keys = set(bundle.donors["biogen"]["inchikey"])
    target_keys = set(bundle.target_train["inchikey"]) | set(bundle.target_test["inchikey"])
    assert not (biogen_keys & target_keys)
    assert bool(bundle.exclusions["exclusion_reason"].eq("donor_overlap_with_target").any())


def test_scoring_interface_cannot_receive_an_evaluation_surface() -> None:
    """Attribution signatures structurally exclude evaluation partitions."""
    parameters = set(inspect.signature(BoostInAttributor.score).parameters)
    assert parameters == {"self", "training", "selection", "model_settings", "seeds"}
    assert not any("evaluation" in name for name in parameters)
    protocol_parameters = set(inspect.signature(TrainingAttributor.score).parameters)
    assert protocol_parameters == {"self", "training", "selection", "model_settings", "seeds"}
    planning_parameters = set(inspect.signature(plan_arms).parameters)
    assert not any("evaluation" in name for name in planning_parameters)


def test_manifest_label_visibility_is_stage_specific(expansionrx_split) -> None:
    """Evaluation labels are visible only to the evaluation stage."""
    _, split = expansionrx_split
    manifest = split.partition_manifest
    evaluation = manifest.loc[manifest["role"].eq("evaluation")]
    training = manifest.loc[manifest["role"].eq("full_training")]
    selection = manifest.loc[manifest["role"].eq("influence_selection")]
    assert not bool(evaluation["label_visible_to_scoring"].any())
    assert bool(evaluation["label_visible_to_evaluation"].all())
    assert bool(training["label_visible_to_scoring"].all())
    assert bool(selection["label_visible_to_scoring"].all())
    assert not bool(training["label_visible_to_evaluation"].any())
    assert not bool(selection["label_visible_to_evaluation"].any())


@pytest.mark.parametrize("target_fixture", ["biogen_config", "polaris_config"])
def test_rotated_split_is_structurally_valid(target_fixture: str, request) -> None:
    """Both rotated targets build complete, disjoint splits."""
    config = request.getfixturevalue(target_fixture)
    bundle = load_observations(config)
    split = build_pruning_split(bundle)
    assert split.influence_selection.size > 0
    assert split.evaluation.size > 0
    assert split.full_training.size > 0
    assert (
        set(split.influence_selection.frame["inchikey"]) & set(split.evaluation.frame["inchikey"])
        == set()
    )
    if config.target == "polaris":
        native = split.full_training.frame
        assert set(native.loc[native["origin_class"].eq("native"), "source"]) == {"polaris"}
