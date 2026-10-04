"""Full-training, influence-selection, and evaluation partitions.

The ranking may use only the influence-selection surface; pruning outcomes may use only the
evaluation surface. The two roles are structurally distinct: scoring code receives a
:class:`LabelledPartition` named ``influence_selection`` and never an evaluation partition,
and the evaluation partition is only materialised for :mod:`influence_pruning.evaluate`.

For ExpansionRx HLM the official test set is time-ordered and split at its identifier
midpoint: the earlier half is the influence and policy-selection surface, the later half is
the evaluation surface. For a Biogen or Polaris target the available target test partition is
split into two disjoint groups of whole Butina clusters.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from influence_pruning.config import SplitSettings
from influence_pruning.data import ObservationBundle
from influence_pruning.errors import ChronologyAuditError, RunError
from influence_pruning.similarity import butina_clusters, fingerprint_matrix

MANIFEST_COLUMNS = (
    "role",
    "partition_name",
    "source",
    "origin_class",
    "candidate_id",
    "original_id",
    "inchikey",
    "canonical_smiles",
    "model_target",
    "chronology_rank",
    "cluster_id",
    "fingerprint_row",
    "label_visible_to_scoring",
    "label_visible_to_evaluation",
)


@dataclass(frozen=True, slots=True)
class LabelledPartition:
    """A named partition whose targets are available to the owning stage.

    :param name: partition identifier recorded in the split manifest.
    :param frame: model-ready observations with ``model_target`` and ``candidate_id`` present.
    :param fingerprints: ``(n, n_bits)`` fingerprint matrix aligned with `frame`.
    """

    name: str
    frame: pd.DataFrame
    fingerprints: np.ndarray

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidate identifiers in frame order."""
        return tuple(self.frame["candidate_id"].astype(str))

    @property
    def size(self) -> int:
        """Number of observations."""
        return len(self.frame)


@dataclass(frozen=True, slots=True)
class PruningSplit:
    """The three surfaces of one conditional pruning experiment.

    :param full_training: every eligible target-training and donor observation; excludes both
        test halves.
    :param influence_selection: target-like selection surface used only for attribution and
        policy selection.
    :param evaluation: target-domain evaluation surface used only for outcome measurement.
    :param partition_manifest: fixed-schema membership table for all three surfaces.
    :param split_record: JSON-ready split diagnostics (midpoint, counts, cluster allocation).
    """

    full_training: LabelledPartition
    influence_selection: LabelledPartition
    evaluation: LabelledPartition
    partition_manifest: pd.DataFrame
    split_record: dict[str, Any]


def partition(name: str, frame: pd.DataFrame) -> LabelledPartition:
    """Build a labelled partition from an ordered frame.

    :param name: partition identifier.
    :param frame: model-ready observations.
    :returns: partition with its fingerprint matrix and aligned rows.
    :raises RunError: the frame is empty or lacks the model-ready provenance columns.
    """
    ordered = frame.reset_index(drop=True).copy()
    if ordered.empty:
        raise RunError(f"partition {name!r} is empty")
    missing = [
        column
        for column in ("candidate_id", "source", "origin_class", "inchikey", "model_target")
        if column not in ordered.columns
    ]
    if missing:
        raise RunError(f"partition {name!r} is missing model-ready columns {missing}")
    if ordered["candidate_id"].duplicated().any():
        duplicates = ordered.loc[ordered["candidate_id"].duplicated(), "candidate_id"].tolist()
        raise RunError(f"partition {name!r} repeats candidate IDs: {duplicates[:5]}")
    ordered["fingerprint_row"] = np.arange(len(ordered), dtype=int)
    return LabelledPartition(name=name, frame=ordered, fingerprints=fingerprint_matrix(ordered))


def _molecule_numbers(names: pd.Series) -> pd.Series:
    """Extract the trailing numeric component of molecule names."""
    extracted = names.astype(str).str.extract(r"(\d+)\s*$")[0]
    return pd.to_numeric(extracted, errors="coerce")


def make_temporal_test_halves(
    bundle: ObservationBundle,
) -> tuple[LabelledPartition, LabelledPartition, dict[str, Any]]:
    """Split the time-ordered target test partition into earlier and later halves.

    The parsed identifiers must be unique and strictly increasing. The split is made at the
    count midpoint of the identifier-ordered partition: the earlier half is the
    influence-selection surface, the later half is the evaluation surface.

    :param bundle: observation bundle whose target test IDs are time-ordered.
    :returns: ``(influence_selection, evaluation, record)``.
    :raises ChronologyAuditError: identifiers do not parse, repeat, or increase strictly, or a
        half would be empty.
    """
    frame = bundle.target_test.reset_index(drop=True)
    numbers = _molecule_numbers(frame["original_id"])
    if numbers.isna().any():
        raise ChronologyAuditError(
            "temporal test halves require parsable trailing identifiers; "
            f"{int(numbers.isna().sum())} rows do not parse"
        )
    if numbers.duplicated().any():
        raise ChronologyAuditError("temporal test halves require unique identifiers")
    order = np.argsort(numbers.to_numpy(dtype=np.int64), kind="stable")
    ordered_numbers = numbers.to_numpy(dtype=np.int64)[order]
    if ordered_numbers.size < 2 or not bool(np.all(np.diff(ordered_numbers) > 0)):
        raise ChronologyAuditError("temporal test halves require strictly increasing identifiers")
    midpoint = len(frame) // 2
    if midpoint < 1 or midpoint >= len(frame):
        raise ChronologyAuditError(
            f"temporal split leaves an empty half for {len(frame)} test molecules"
        )
    selection = frame.iloc[order[:midpoint]].reset_index(drop=True)
    evaluation = frame.iloc[order[midpoint:]].reset_index(drop=True)
    record: dict[str, Any] = {
        "test_mode": "temporal_halves",
        "n_test": int(len(frame)),
        "midpoint_count": int(midpoint),
        "n_selection": int(len(selection)),
        "n_evaluation": int(len(evaluation)),
        "selection_id_min": int(ordered_numbers[:midpoint].min()),
        "selection_id_max": int(ordered_numbers[:midpoint].max()),
        "evaluation_id_min": int(ordered_numbers[midpoint:].min()),
        "evaluation_id_max": int(ordered_numbers[midpoint:].max()),
        "selection_earlier_than_evaluation": bool(
            ordered_numbers[:midpoint].max() < ordered_numbers[midpoint:].min()
        ),
    }
    return (
        partition("influence_selection", selection),
        partition("evaluation", evaluation),
        record,
    )


def make_butina_test_halves(
    frame: pd.DataFrame, settings: SplitSettings
) -> tuple[LabelledPartition, LabelledPartition, dict[str, Any]]:
    """Split a target test partition into whole-cluster selection and evaluation surfaces.

    Whole Butina clusters are allocated deterministically under the configured seed. Cluster
    membership is recorded on every molecule so that close analogues can never fall into both
    surfaces.

    :param frame: target test-partition observations.
    :param settings: cluster threshold, seed, and selection fraction.
    :returns: ``(influence_selection, evaluation, record)``.
    :raises RunError: either surface would be empty of molecules or clusters.
    """
    ordered = frame.reset_index(drop=True).copy()
    fingerprints = fingerprint_matrix(ordered)
    clusters = butina_clusters(fingerprints, settings.butina_distance_threshold)
    if len(clusters) < 2:
        raise RunError(
            "Butina test halves require at least two clusters; "
            f"found {len(clusters)} for {len(ordered)} molecules"
        )
    order = np.random.default_rng(settings.butina_seed).permutation(len(clusters))
    target_count = settings.target_test_selection_fraction * len(ordered)
    selected: list[int] = []
    selected_clusters: list[int] = []
    count = 0
    for cluster_index in order:
        cluster = clusters[int(cluster_index)]
        if abs(count + len(cluster) - target_count) <= abs(count - target_count):
            selected.extend(cluster)
            selected_clusters.append(int(cluster_index))
            count += len(cluster)
    mask = np.zeros(len(ordered), dtype=bool)
    mask[selected] = True
    if not mask.any() or mask.all():
        raise RunError("Butina test halves produced an empty selection or evaluation surface")
    evaluation_clusters = [
        index for index in range(len(clusters)) if index not in set(selected_clusters)
    ]
    if not selected_clusters or not evaluation_clusters:
        raise RunError("Butina test halves require at least one cluster per surface")
    cluster_labels = np.zeros(len(ordered), dtype=int)
    for index, cluster in enumerate(clusters):
        for member in cluster:
            cluster_labels[member] = index
    selection = ordered.loc[mask].reset_index(drop=True)
    evaluation = ordered.loc[~mask].reset_index(drop=True)
    selection["cluster_id"] = cluster_labels[mask]
    evaluation["cluster_id"] = cluster_labels[~mask]
    record: dict[str, Any] = {
        "test_mode": "butina_halves",
        "n_test": int(len(ordered)),
        "butina_distance_threshold": settings.butina_distance_threshold,
        "butina_seed": settings.butina_seed,
        "target_test_selection_fraction": settings.target_test_selection_fraction,
        "n_clusters": len(clusters),
        "n_selection_clusters": len(selected_clusters),
        "n_evaluation_clusters": len(evaluation_clusters),
        "n_selection": int(mask.sum()),
        "n_evaluation": int((~mask).sum()),
        "achieved_selection_fraction": float(mask.mean()),
        "selection_cluster_ids": sorted(selected_clusters),
        "evaluation_cluster_ids": sorted(evaluation_clusters),
    }
    return (
        partition("influence_selection", selection),
        partition("evaluation", evaluation),
        record,
    )


def build_pruning_split(bundle: ObservationBundle) -> PruningSplit:
    """Assemble the full-training, influence-selection, and evaluation surfaces.

    :param bundle: observation bundle for the configured target rotation.
    :returns: validated split with its partition manifest.
    :raises RunError: full training is empty, identifiers collide, or a surface is empty.
    """
    config = bundle.config
    frames = [bundle.target_train] + [bundle.donors[source] for source in config.donor_sources]
    full_frame = pd.concat(frames, ignore_index=True)
    if full_frame["candidate_id"].duplicated().any():
        duplicates = full_frame.loc[
            full_frame["candidate_id"].duplicated(), "candidate_id"
        ].tolist()
        raise RunError(f"full training repeats candidate IDs: {duplicates[:5]}")
    full = partition("full_training", full_frame)

    if config.split.test_mode == "temporal_halves":
        selection, evaluation, record = make_temporal_test_halves(bundle)
    else:
        selection, evaluation, record = make_butina_test_halves(bundle.target_test, config.split)

    assert_disjoint_partitions(full, selection, evaluation)
    record.update(
        {
            "n_train_full": int(full.size),
            "n_influence_selection": int(selection.size),
            "n_evaluation": int(evaluation.size),
            "n_donor_sources": len(config.donor_sources),
        }
    )
    manifest = _partition_manifest(full, selection, evaluation)
    return PruningSplit(
        full_training=full,
        influence_selection=selection,
        evaluation=evaluation,
        partition_manifest=manifest,
        split_record=record,
    )


def assert_disjoint_partitions(
    full: LabelledPartition, selection: LabelledPartition, evaluation: LabelledPartition
) -> None:
    """Assert that no structure or candidate identifier is shared between surfaces.

    :param full: full training partition.
    :param selection: influence-selection partition.
    :param evaluation: evaluation partition.
    :raises RunError: a partition repeats an identifier, or two partitions share an InChIKey
        or a source-qualified candidate identifier.
    """
    groups = (
        (full.name, full.frame),
        (selection.name, selection.frame),
        (evaluation.name, evaluation.frame),
    )
    for name, frame in groups:
        repeated = frame["candidate_id"].astype(str).duplicated()
        if bool(repeated.any()):
            raise RunError(f"partition {name!r} repeats candidate identifiers")
    for index, (name_a, frame_a) in enumerate(groups):
        for name_b, frame_b in groups[index + 1 :]:
            shared_keys = set(frame_a["inchikey"].astype(str)) & set(
                frame_b["inchikey"].astype(str)
            )
            if shared_keys:
                raise RunError(
                    f"partitions {name_a!r} and {name_b!r} share InChIKeys: "
                    f"{sorted(shared_keys)[:5]}"
                )
            shared_ids = set(frame_a["candidate_id"].astype(str)) & set(
                frame_b["candidate_id"].astype(str)
            )
            if shared_ids:
                raise RunError(
                    f"partitions {name_a!r} and {name_b!r} share candidate identifiers: "
                    f"{sorted(shared_ids)[:5]}"
                )


def _partition_manifest(
    full: LabelledPartition,
    selection: LabelledPartition,
    evaluation: LabelledPartition,
) -> pd.DataFrame:
    """Build the fixed-schema membership and label-visibility manifest."""
    entries = (
        ("full_training", full, True, False),
        ("influence_selection", selection, True, False),
        ("evaluation", evaluation, False, True),
    )
    rows: list[dict[str, Any]] = []
    for role, labelled, visible_to_scoring, visible_to_evaluation in entries:
        frame = labelled.frame
        for position, record in enumerate(frame.itertuples(index=False)):
            rows.append(
                {
                    "role": role,
                    "partition_name": labelled.name,
                    "source": str(record.source),
                    "origin_class": str(record.origin_class),
                    "candidate_id": str(record.candidate_id),
                    "original_id": str(record.original_id),
                    "inchikey": str(record.inchikey),
                    "canonical_smiles": str(record.canonical_smiles),
                    "model_target": float(record.model_target),
                    "chronology_rank": getattr(record, "chronology_rank", None),
                    "cluster_id": getattr(record, "cluster_id", None),
                    "fingerprint_row": position,
                    "label_visible_to_scoring": visible_to_scoring,
                    "label_visible_to_evaluation": visible_to_evaluation,
                }
            )
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest["chronology_rank"] = manifest["chronology_rank"].astype("Int64")
    manifest["cluster_id"] = manifest["cluster_id"].astype("Int64")
    return manifest
