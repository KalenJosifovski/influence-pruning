"""Raw dataset loading and model-ready observation construction with target rotation.

The loader generalises the preceding pipeline's ExpansionRx-centric flow: any of the three
sources can be the target, the other two become eligible donors, target-donor structure
overlap is removed, and the target test partition is returned whole so that
:mod:`influence_pruning.partitions` can split it into influence-selection and evaluation
surfaces.

No module here fits a model or materialises an evaluation surface.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from influence_pruning.config import RunConfig
from influence_pruning.contracts import (
    SourceContract,
    apply_contract,
    lookup_contract,
    resolve_contract,
)
from influence_pruning.errors import DatasetError
from influence_pruning.similarity import butina_clusters, fingerprint_matrix
from influence_pruning.standardize import (
    FingerprintSettings,
    deduplicate_observations,
    standardize_frame,
)

DATASET_SPECS: dict[str, tuple[str, str, str]] = {
    "expansionrx_train": ("expansionrx", "Molecule Name", "SMILES"),
    "expansionrx_test": ("expansionrx", "Molecule Name", "SMILES"),
    "biogen": ("biogen", "Internal ID", "SMILES"),
    "polaris": ("polaris", "Molecule Name", "CXSMILES"),
}
SOURCES = ("expansionrx", "biogen", "polaris")
EXCLUSION_OVERLAP = "donor_overlap_with_target"
EXCLUSION_INVALID_STRUCTURE = "invalid_structure"
EXCLUSION_CROSS_SPLIT_OVERLAP = "target_cross_split_overlap"

MODEL_READY_COLUMNS = (
    "candidate_id",
    "source",
    "origin_class",
    "original_id",
    "inchikey",
    "canonical_smiles",
    "model_target",
)
EXCLUSION_COLUMNS = (
    "source",
    "partition",
    "original_id",
    "endpoint",
    "exclusion_reason",
    "detail",
)
AUDIT_COLUMNS = (
    "source",
    "role",
    "endpoint",
    "n_raw",
    "n_parsed",
    "n_invalid_structure",
    "n_endpoint_excluded",
    "n_eligible",
    "n_after_dedup",
    "n_after_overlap_exclusion",
    "n_used",
    "units_in",
    "units_out",
    "transform",
    "scale_factor",
    "minimum_quantification_limit",
    "notes",
)


@dataclass(frozen=True, slots=True)
class DonorAvailability:
    """Availability of one donor dataset for the configured endpoint.

    :param status: ``"available"``, ``"unresolved"``, or ``"endpoint-absent"``.
    :param reason: why the donor cannot be used when not available.
    """

    source: str
    status: str
    column: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """Model-ready observations for one endpoint and target rotation.

    :param raw: raw CSV frames keyed by dataset key.
    :param standardized: standardised frames keyed by dataset key.
    :param applied: endpoint-applied frames (pre-deduplication) keyed by dataset key.
    :param target_train: eligible, deduplicated, model-ready target training observations.
    :param target_test: eligible, deduplicated target test-partition observations; this is the
        whole test partition before the selection/evaluation split.
    :param donors: eligible, deduplicated, overlap-excluded donor frames keyed by source.
    :param exclusions: recorded exclusion reasons across all sources.
    :param contracts: resolved contracts actually applied.
    :param donor_availability: donor availability records.
    :param split_record: molecule counts and split diagnostics for the target.
    :param data_audit: per-source/role audit summary table.
    :param capped: whether dry-run row caps were applied.
    """

    config: RunConfig
    endpoint: str
    raw: dict[str, pd.DataFrame]
    standardized: dict[str, pd.DataFrame]
    applied: dict[str, pd.DataFrame]
    target_train: pd.DataFrame
    target_test: pd.DataFrame
    donors: dict[str, pd.DataFrame]
    exclusions: pd.DataFrame
    contracts: dict[str, SourceContract]
    donor_availability: dict[str, DonorAvailability]
    split_record: dict[str, int]
    data_audit: pd.DataFrame
    capped: bool


def _model_ready(frame: pd.DataFrame, *, origin_class: str) -> pd.DataFrame:
    """Add the model-ready provenance columns to a deduplicated frame.

    :param frame: deduplicated observations with ``model_target`` present.
    :param origin_class: ``"native"`` for target-programme rows, ``"donor"`` for external rows.
    :returns: frame with the model-ready provenance columns populated.
    """
    out = frame.reset_index(drop=True).copy()
    out["candidate_id"] = [
        f"{source}:{original_id}:{inchikey}"
        for source, original_id, inchikey in zip(
            out["source"].astype(str),
            out["original_id"].astype(str),
            out["inchikey"].astype(str),
            strict=True,
        )
    ]
    out["origin_class"] = origin_class
    return out


def load_observations(config: RunConfig) -> ObservationBundle:
    """Load raw datasets and build model-ready observations for the configured endpoint.

    :param config: resolved run configuration.
    :returns: observation bundle for partition construction.
    :raises DatasetError: a dataset is missing an expected column or the target has no
        eligible observations.
    :raises ContractUnresolvedError: a required endpoint contract is unresolved.
    """
    raw = _read_all(config)
    standardized = _standardize_all(raw, config)
    exclusions: list[pd.DataFrame] = []
    contracts: dict[str, SourceContract] = {}
    applied_frames: dict[str, pd.DataFrame] = {}
    eligible: dict[str, pd.DataFrame] = {}

    for source in SOURCES:
        contract = lookup_contract(config.endpoint, source, config.contract_overrides)
        if contract is None:
            continue
        resolved = resolve_contract(config.endpoint, source, config.contract_overrides)
        frame = _source_frame(standardized, source)
        applied = apply_contract(resolved, frame)
        applied_frames[source] = applied
        exclusions.append(_exclusion_rows(applied, source, config.endpoint))
        eligible[source] = _eligible(applied)
        contracts[source] = resolved

    if config.target not in eligible:
        raise DatasetError(
            f"target dataset {config.target} has no eligible {config.endpoint} observations"
        )

    target_train, target_test, split_record, overlap_ids = _target_partitions(
        standardized, eligible, config
    )
    if overlap_ids:
        exclusions.append(_cross_split_exclusion_rows(overlap_ids, config.target, config.endpoint))
    target_keys = set(target_train["inchikey"].astype(str)) | set(
        target_test["inchikey"].astype(str)
    )

    donors: dict[str, pd.DataFrame] = {}
    donor_availability: dict[str, DonorAvailability] = {}
    for source in config.donor_sources:
        frame = deduplicate_observations(eligible[source])
        overlap = frame["inchikey"].astype(str).isin(target_keys)
        if bool(overlap.any()):
            exclusions.append(_overlap_exclusion_rows(frame.loc[overlap], source, config.endpoint))
        donors[source] = frame.loc[~overlap].reset_index(drop=True)
        if donors[source].empty:
            raise DatasetError(
                f"donor source {source!r} has no eligible observations after target-overlap "
                "exclusion; check its endpoint contract and the target chemistry"
            )
        donor_availability[source] = DonorAvailability(
            source=source,
            status="available",
            column=contracts[source].column,
            reason=(
                f"{contracts[source].units_in} -> {contracts[source].units_out}; "
                f"{len(donors[source])} eligible after target-overlap exclusion"
            ),
        )

    capped = config.limits.any_cap
    train_model = _model_ready(
        _cap(target_train, config.limits.max_target_train), origin_class="native"
    )
    test_model = _model_ready(
        _cap(target_test, config.limits.max_target_selection), origin_class="native"
    )
    donor_models = {
        source: _model_ready(_cap(frame, config.limits.max_donor_rows), origin_class="donor")
        for source, frame in donors.items()
    }
    if train_model.empty or test_model.empty:
        raise DatasetError(
            "target train and test partitions must both be non-empty; "
            f"got train={len(train_model)}, test={len(test_model)}"
        )

    audit = _data_audit(
        config,
        raw=raw,
        eligible=eligible,
        applied_frames=applied_frames,
        contracts=contracts,
        target_train=train_model,
        target_test=test_model,
        donors=donor_models,
        donor_availability=donor_availability,
    )
    return ObservationBundle(
        config=config,
        endpoint=config.endpoint,
        raw=raw,
        standardized=standardized,
        applied=applied_frames,
        target_train=train_model,
        target_test=test_model,
        donors=donor_models,
        exclusions=_concat_exclusions(exclusions),
        contracts=contracts,
        donor_availability=donor_availability,
        split_record=split_record,
        data_audit=audit,
        capped=capped,
    )


def _read_all(config: RunConfig) -> dict[str, pd.DataFrame]:
    """Read every raw dataset, verifying expected columns."""
    frames: dict[str, pd.DataFrame] = {}
    for key, (_, id_col, smiles_col) in DATASET_SPECS.items():
        path = getattr(config.paths, key)
        frame = pd.read_csv(path)
        missing = [column for column in (id_col, smiles_col) if column not in frame.columns]
        if missing:
            raise DatasetError(f"{path.name} is missing expected columns {missing}")
        frames[key] = frame
    return frames


def _standardize_all(raw: dict[str, pd.DataFrame], config: RunConfig) -> dict[str, pd.DataFrame]:
    """Standardise every raw frame once per run."""
    settings = FingerprintSettings(
        radius=config.model.radius,
        n_bits=config.model.n_bits,
        chirality=config.model.chirality,
    )
    return {
        key: standardize_frame(
            frame,
            id_col=DATASET_SPECS[key][1],
            smiles_col=DATASET_SPECS[key][2],
            source=DATASET_SPECS[key][0],
            settings=settings,
        )
        for key, frame in raw.items()
    }


def _source_frame(standardized: dict[str, pd.DataFrame], source: str) -> pd.DataFrame:
    """Return the complete standardised frame for one source."""
    if source == "expansionrx":
        return pd.concat(
            [standardized["expansionrx_train"], standardized["expansionrx_test"]],
            ignore_index=True,
        )
    return standardized[source]


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows that passed structure parsing and endpoint inclusion."""
    return frame.loc[frame["parse_ok"] & frame["included"]]


def _target_partitions(
    standardized: dict[str, pd.DataFrame],
    eligible: dict[str, pd.DataFrame],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[str]]:
    """Construct the target train and test partitions according to the source protocol.

    :returns: ``(target_train, target_test, split_record, overlap_original_ids)`` where both
        frames are deduplicated eligible observations (not yet model-ready) and the overlap
        identifiers name structures dropped because they appeared in both partitions.
    """
    target = config.target
    if target == "expansionrx":
        train_ids = set(standardized["expansionrx_train"]["original_id"].astype(str))
        test_ids = set(standardized["expansionrx_test"]["original_id"].astype(str))
        source_eligible = eligible[target]
        original = source_eligible["original_id"].astype(str)
        train_rows = source_eligible.loc[original.isin(train_ids)]
        test_rows = source_eligible.loc[~original.isin(train_ids)]
        train = deduplicate_observations(train_rows)
        test = deduplicate_observations(test_rows)
        record = {
            "n_target_train": len(train),
            "n_target_test": len(test),
            "n_train_ids_in_file": len(train_ids),
            "n_test_ids_in_file": len(test_ids),
        }
    elif target == "polaris":
        membership = _polaris_membership(standardized, eligible[target])
        train_event = deduplicate_observations(eligible[target].loc[membership.eq("Train")])
        test_event = deduplicate_observations(eligible[target].loc[membership.eq("Test")])
        train, test = train_event, test_event
        if train.empty or test.empty:
            raise DatasetError("Polaris target requires non-empty native Train and Test partitions")
        record = {
            "n_target_train": len(train),
            "n_target_test": len(test),
            "n_train_ids_in_file": int(membership.eq("Train").sum()),
            "n_test_ids_in_file": int(membership.eq("Test").sum()),
        }
    else:
        frame = deduplicate_observations(eligible[target])
        train, test = butina_cluster_holdout(
            frame,
            config.split.biogen_target_train_fraction,
            config.split.butina_distance_threshold,
            config.split.butina_seed,
        )
        record = {
            "n_target_train": len(train),
            "n_target_test": len(test),
            "biogen_target_train_fraction": config.split.biogen_target_train_fraction,
        }

    cross_overlap_keys = set(train["inchikey"].astype(str)) & set(test["inchikey"].astype(str))
    overlap_ids: list[str] = []
    if cross_overlap_keys:
        train, test = _exclude_cross_split_overlap(train, test)
        overlap_ids = sorted(
            eligible[target]
            .loc[eligible[target]["inchikey"].astype(str).isin(cross_overlap_keys), "original_id"]
            .astype(str)
            .tolist()
        )
        record["n_cross_split_overlap_excluded"] = len(cross_overlap_keys)

    if target == "expansionrx":
        combined = pd.concat([train, test], ignore_index=True)
        train = train.assign(chronology_rank=_chronology_ranks(combined, train))
        test = test.assign(chronology_rank=_chronology_ranks(combined, test))
    return train, test, record, overlap_ids


def _polaris_membership(standardized: dict[str, pd.DataFrame], frame: pd.DataFrame) -> pd.Series:
    """Map each eligible Polaris row to its official Train/Test partition."""
    original = (
        standardized["polaris"].drop_duplicates("original_id").set_index("original_id")["Set"]
    )
    membership = frame["original_id"].astype(str).map(original)
    if membership.isna().any():
        missing = frame.loc[membership.isna(), "original_id"].astype(str).tolist()[:5]
        raise DatasetError(f"Polaris rows without a Train/Test membership: {missing}")
    return membership


def _chronology_ranks(combined: pd.DataFrame, subset: pd.DataFrame) -> np.ndarray:
    """Return the global numeric-ID rank of each subset row within `combined`."""
    numbers = _molecule_numbers(combined["original_id"])
    if numbers.isna().any():
        raise DatasetError("cannot rank chronology: unparsable molecule names present")
    order = np.argsort(numbers.to_numpy(dtype=np.int64), kind="stable")
    rank_by_key = {
        str(combined["inchikey"].iloc[position]): rank for rank, position in enumerate(order)
    }
    return subset["inchikey"].astype(str).map(rank_by_key).to_numpy(dtype=int)


def _molecule_numbers(names: pd.Series) -> pd.Series:
    """Extract the trailing numeric component of molecule names."""
    extracted = names.astype(str).str.extract(r"(\d+)\s*$")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _exclude_cross_split_overlap(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop structures present in both the target train and test partitions.

    A structure seen in training must not appear on the evaluation surface; the conservative
    convention of the preceding project is to drop the shared structure from both sides and
    record the exclusion.
    """
    shared = set(train["inchikey"].astype(str)) & set(test["inchikey"].astype(str))
    train = train.loc[~train["inchikey"].astype(str).isin(shared)].reset_index(drop=True)
    test = test.loc[~test["inchikey"].astype(str).isin(shared)].reset_index(drop=True)
    return train, test


def butina_cluster_holdout(
    frame: pd.DataFrame, train_fraction: float, distance_threshold: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole deterministic Butina clusters into fitting and held-out chemistry."""
    clusters = butina_clusters(fingerprint_matrix(frame), distance_threshold)
    if len(clusters) < 2:
        raise DatasetError("whole-cluster holdout requires at least two clusters")
    order = np.random.default_rng(seed).permutation(len(clusters))
    target_count = int(len(frame) * train_fraction)
    selected: list[int] = []
    count = 0
    for cluster_index in order:
        cluster = clusters[int(cluster_index)]
        # Keep whole clusters, but stop once another cluster would move the achieved
        # fitting count farther from the requested count than the current set.
        if abs(count + len(cluster) - target_count) <= abs(count - target_count):
            selected.extend(cluster)
            count += len(cluster)
    mask = np.zeros(len(frame), dtype=bool)
    mask[selected] = True
    if not mask.any() or mask.all():
        raise DatasetError("whole-cluster holdout produced an empty fitting or held-out partition")
    return frame.loc[mask].reset_index(drop=True), frame.loc[~mask].reset_index(drop=True)


def _cap(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Apply a dry-run row cap, keeping the earliest rows."""
    if limit is None or len(frame) <= limit:
        return frame
    return frame.iloc[:limit].reset_index(drop=True)


def _exclusion_rows(frame: pd.DataFrame, partition: str, endpoint: str) -> pd.DataFrame:
    """Build exclusion records for rows that cannot enter modelling."""
    invalid_structure = ~frame["parse_ok"].to_numpy()
    not_included = (frame["parse_ok"] & ~frame["included"]).to_numpy()
    mask = invalid_structure | not_included
    subset = frame.loc[mask]
    if subset.empty:
        return pd.DataFrame(columns=list(EXCLUSION_COLUMNS))
    structure_flag = ~subset["parse_ok"].to_numpy()
    return pd.DataFrame(
        {
            "source": subset["source"].to_numpy(),
            "partition": partition,
            "original_id": subset["original_id"].astype(str).to_numpy(),
            "endpoint": endpoint,
            "exclusion_reason": np.where(
                structure_flag,
                EXCLUSION_INVALID_STRUCTURE,
                subset["exclusion_reason"].to_numpy(),
            ),
            "detail": np.where(
                structure_flag,
                subset["parse_error"].fillna("").astype(str).to_numpy(),
                "",
            ),
        }
    )


def _overlap_exclusion_rows(frame: pd.DataFrame, source: str, endpoint: str) -> pd.DataFrame:
    """Build exclusion records for donor molecules overlapping the target chemistry."""
    return pd.DataFrame(
        {
            "source": source,
            "partition": source,
            "original_id": frame["original_id"].astype(str).to_numpy(),
            "endpoint": endpoint,
            "exclusion_reason": EXCLUSION_OVERLAP,
            "detail": "InChIKey present in a target train or test partition",
        }
    )


def _cross_split_exclusion_rows(
    original_ids: list[str], target: str, endpoint: str
) -> pd.DataFrame:
    """Build exclusion records for structures shared by the target's two partitions."""
    return pd.DataFrame(
        {
            "source": target,
            "partition": target,
            "original_id": original_ids,
            "endpoint": endpoint,
            "exclusion_reason": EXCLUSION_CROSS_SPLIT_OVERLAP,
            "detail": "InChIKey present in both target partitions",
        }
    )


def _concat_exclusions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate exclusion records into one stable table."""
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(EXCLUSION_COLUMNS))
    return pd.concat(non_empty, ignore_index=True)


def _data_audit(
    config: RunConfig,
    *,
    raw: dict[str, pd.DataFrame],
    eligible: dict[str, pd.DataFrame],
    applied_frames: dict[str, pd.DataFrame],
    contracts: dict[str, SourceContract],
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    donors: dict[str, pd.DataFrame],
    donor_availability: dict[str, DonorAvailability],
) -> pd.DataFrame:
    """Build the per-source/role audit summary table."""
    deduped = {source: deduplicate_observations(eligible[source]) for source in SOURCES}
    rows: list[dict[str, Any]] = []
    roles: list[tuple[str, str, pd.DataFrame]] = [
        (config.target, "target_train", target_train),
        (config.target, "target_test", target_test),
    ]
    roles.extend((source, f"donor:{source}", donors[source]) for source in config.donor_sources)

    for source, role, used in roles:
        keys = ["expansionrx_train", "expansionrx_test"] if source == "expansionrx" else [source]
        n_raw = int(sum(len(raw[key]) for key in keys))
        applied = applied_frames[source]
        n_parsed = int(applied["parse_ok"].sum())
        n_eligible = int(len(eligible[source]))
        contract = contracts[source]
        n_after_dedup = int(len(deduped[source]))
        n_after_overlap = n_after_dedup if source == config.target else int(len(donors[source]))
        availability = donor_availability.get(source)
        notes = ""
        if role == "target_train":
            notes = "target programme training observations"
        elif role == "target_test":
            notes = "whole target test partition before selection/evaluation split"
        elif availability is not None:
            notes = availability.reason
        rows.append(
            {
                "source": source,
                "role": role,
                "endpoint": config.endpoint,
                "n_raw": n_raw,
                "n_parsed": n_parsed,
                "n_invalid_structure": n_raw - n_parsed,
                "n_endpoint_excluded": n_parsed - n_eligible,
                "n_eligible": n_eligible,
                "n_after_dedup": n_after_dedup,
                "n_after_overlap_exclusion": n_after_overlap,
                "n_used": int(len(used)),
                "units_in": contract.units_in,
                "units_out": contract.units_out,
                "transform": contract.transform_name,
                "scale_factor": contract.scale_factor,
                "minimum_quantification_limit": contract.minimum_quantification_limit,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))
