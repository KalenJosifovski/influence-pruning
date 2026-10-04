"""Deterministic deletion-arm planning for conditional batch pruning.

``pruning_arms`` turns pooled attribution scores into immutable deletion manifests. It never
fits a model. Every arm is either a deterministic ranked batch, a recorded uniform-random
batch, or a label/source-stratified control drawn without replacement from the same eligible
pool, excluding the scored reference batch.

Stratified controls preserve the scored batch's ``source × label-quantile-bin`` composition;
they are not exact continuous-label matches, so their achieved labels can differ from the
reference batch (reported as Wasserstein diagnostics). Construction retries with successively
coarser quantile binnings before the control is recorded as infeasible with an explicit
diagnostic. An infeasible arm is never silently replaced by a random batch.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from influence_pruning.config import BoostInSettings
from influence_pruning.errors import ArmPlanningError, RunError
from influence_pruning.partitions import PruningSplit

RANDOM_DRAW_SEED = 0
MATCHING_DRAW_SEED = 0
POOL_DONOR = "donor_only"
POOL_NATIVE = "native_only"
POOL_MIXED = "mixed_pool"
POLICY_HIGH = "boostin_high"
POLICY_LOW = "boostin_low"
POLICY_RANDOM = "random"

MANIFEST_COLUMNS = (
    "arm_id",
    "eligible_pool",
    "stratum",
    "policy",
    "direction",
    "batch_size",
    "draw_number",
    "candidate_id",
    "source",
    "origin_class",
    "original_id",
    "inchikey",
    "model_target",
    "pool_rank",
    "aggregate_rank",
    "rank_mean",
    "raw_score_mean",
    "matching_bin",
    "selection_reason",
)
SUMMARY_COLUMNS = (
    "arm_id",
    "eligible_pool",
    "stratum",
    "policy",
    "direction",
    "batch_size",
    "draw_number",
    "status",
    "requested_size",
    "achieved_size",
    "pool_size",
    "n_train_full",
    "n_train_pruned",
    "source_biogen",
    "source_polaris",
    "source_expansionrx",
    "label_mean",
    "label_min",
    "label_max",
    "label_wasserstein_overall",
    "label_wasserstein_max_source",
    "matching_bins_used",
    "matching_plan",
    "matching_diagnostic",
)


@dataclass(frozen=True, slots=True)
class DeletionArm:
    """One immutable deletion batch.

    :param arm_id: stable human-readable identifier.
    :param eligible_pool: one of ``donor_only``, ``native_only``, ``mixed_pool``.
    :param stratum: pool stratum (donor source, ``combined``, ``target``, or ``all``).
    :param policy: one of the declared pruning policies.
    :param direction: ``"high"`` or ``"low"`` for ranked and matched arms, else ``None``.
    :param batch_size: number of removed training observations.
    :param draw_number: draw index for random and matched arms, else ``None``.
    :param removed_candidate_ids: exact removed membership, unique and sorted.
    """

    arm_id: str
    eligible_pool: str
    stratum: str
    policy: str
    direction: str | None
    batch_size: int
    draw_number: int | None
    removed_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """All planned arms with their audit tables.

    :param arms: planned deletion arms in deterministic order.
    :param manifest: one row per planned arm and removed candidate.
    :param summary: one row per arm attempt, including infeasible matched arms.
    :param pool_sizes: eligible row count per ``pool:stratum`` key.
    """

    arms: tuple[DeletionArm, ...]
    manifest: pd.DataFrame
    summary: pd.DataFrame
    pool_sizes: dict[str, int]


@dataclass(frozen=True, slots=True)
class _PoolSpec:
    """Internal eligible-pool specification."""

    pool: str
    stratum: str
    candidate_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        """Stable ``pool:stratum`` key."""
        return f"{self.pool}:{self.stratum}"


@dataclass(frozen=True, slots=True)
class _MatchResult:
    """Result of matching one direction."""

    arms: tuple[tuple[DeletionArm, pd.DataFrame], ...]
    summary_rows: tuple[dict[str, Any], ...]


def eligible_pools(split: PruningSplit, donor_sources: tuple[str, ...]) -> tuple[_PoolSpec, ...]:
    """Build the eligible pools for every score-and-delete stratum.

    Pools with no eligible rows are omitted: an empty pool has no deletion arms, and data
    loading already rejects configurations whose target or donor partitions are empty.

    :param split: pruning split.
    :param donor_sources: donor sources in configured order.
    :returns: pool specifications in deterministic order.
    """
    training = split.full_training.frame
    donor_mask = training["origin_class"].astype(str).eq("donor")
    native_mask = training["origin_class"].astype(str).eq("native")
    candidates: list[_PoolSpec] = []
    for source in donor_sources:
        mask = donor_mask & training["source"].astype(str).eq(source)
        candidates.append(
            _PoolSpec(
                pool=POOL_DONOR,
                stratum=source,
                candidate_ids=tuple(training.loc[mask, "candidate_id"].astype(str)),
            )
        )
    candidates.append(
        _PoolSpec(
            pool=POOL_DONOR,
            stratum="combined",
            candidate_ids=tuple(training.loc[donor_mask, "candidate_id"].astype(str)),
        )
    )
    candidates.append(
        _PoolSpec(
            pool=POOL_NATIVE,
            stratum="target",
            candidate_ids=tuple(training.loc[native_mask, "candidate_id"].astype(str)),
        )
    )
    candidates.append(
        _PoolSpec(
            pool=POOL_MIXED,
            stratum="all",
            candidate_ids=tuple(training["candidate_id"].astype(str)),
        )
    )
    return tuple(pool for pool in candidates if pool.candidate_ids)


def plan_arms(
    split: PruningSplit,
    scores: pd.DataFrame,
    settings: BoostInSettings,
    donor_sources: tuple[str, ...],
) -> ArmPlan:
    """Plan every ranked, random, and label/source-stratified control deletion arm.

    :param split: pruning split whose full training partition defines the deletion universe.
    :param scores: aggregated pooled attribution scores (one row per training molecule per seed).
    :param settings: batch sizes and draw counts.
    :param donor_sources: donor sources in configured order.
    :returns: immutable arm plan with manifest and summary audit tables.
    :raises ArmPlanningError: a planned batch violates membership, size, or disjointness.
    :raises ConfigError: a batch size exceeds an eligible pool's capacity.
    """
    training = split.full_training.frame
    training_ids = set(training["candidate_id"].astype(str))
    n_train_full = split.full_training.size
    pools = eligible_pools(split, donor_sources)
    pool_sizes = {pool.key: len(pool.candidate_ids) for pool in pools}
    settings.require_capacity(pool_sizes)

    score_table = _score_table(scores, training_ids)
    arms: list[DeletionArm] = []
    manifest_frames: list[pd.DataFrame] = []
    summary_frames: list[dict[str, Any]] = []

    for pool_index, pool in enumerate(pools):
        pool_frame = _pool_frame(training, score_table, pool)
        # Positional reset is deliberate: every downstream selection, binning, and manifest
        # slice uses 0..n-1 positions into this exact frame.
        pool_frame = pool_frame.sort_values(
            ["rank_mean", "candidate_id"], kind="stable"
        ).reset_index(drop=True)
        pool_frame = pool_frame.assign(pool_rank=np.arange(1, len(pool_frame) + 1, dtype=int))
        for size in settings.batch_sizes:
            if size > len(pool_frame) // 2:
                raise ArmPlanningError(
                    f"pool {pool.key} cannot form disjoint high/low batches at size {size}"
                )
            for policy, direction, reference in (
                (POLICY_HIGH, "high", pool_frame.iloc[:size]),
                (POLICY_LOW, "low", pool_frame.iloc[-size:]),
            ):
                arm_id = f"{pool.key}:{policy}:k{size}"
                arm, rows = _ranked_arm(arm_id, pool, policy, direction, size, reference)
                arms.append(arm)
                manifest_frames.append(rows)
                summary_frames.append(
                    _summary_row(
                        arm, pool, n_train_full, reference, f"{direction}est aggregate rank"
                    )
                )

            for arm, rows in _random_arms(pool, pool_index, pool_frame, size, settings):
                arms.append(arm)
                manifest_frames.append(rows)
                summary_frames.append(
                    _summary_row(
                        arm,
                        pool,
                        n_train_full,
                        pool_frame.loc[pool_frame["candidate_id"].isin(arm.removed_candidate_ids)],
                        f"uniform random draw {arm.draw_number}",
                    )
                )

            matched_arms, matched_rows = _matched_arms(
                pool, pool_index, pool_frame, size, settings, n_train_full
            )
            arms.extend(arm for arm, _ in matched_arms)
            manifest_frames.extend(rows for _, rows in matched_arms)
            summary_frames.extend(matched_rows)

    manifest = (
        pd.concat(manifest_frames, ignore_index=True)
        if manifest_frames
        else pd.DataFrame(columns=list(MANIFEST_COLUMNS))
    )
    summary = pd.DataFrame(summary_frames, columns=list(SUMMARY_COLUMNS))
    _validate_plan(arms, manifest, training, training_ids)
    return ArmPlan(arms=tuple(arms), manifest=manifest, summary=summary, pool_sizes=pool_sizes)


def _score_table(scores: pd.DataFrame, training_ids: set[str]) -> pd.DataFrame:
    """Return one aggregate score row per training candidate."""
    if "aggregate_rank" not in scores.columns:
        raise RunError("planning requires aggregated scores with an aggregate_rank column")
    columns = [
        "candidate_id",
        "aggregate_rank",
        "rank_mean",
        "raw_score_mean",
        "source",
        "origin_class",
        "model_target",
    ]
    per_molecule = scores.drop_duplicates("candidate_id").loc[:, columns].copy()
    missing = training_ids - set(per_molecule["candidate_id"].astype(str))
    if missing:
        raise RunError(
            f"{len(missing)} full-training candidates have no aggregate score; "
            f"first: {sorted(missing)[:5]}"
        )
    if per_molecule["candidate_id"].duplicated().any():
        raise RunError("aggregate scores repeat candidate identifiers")
    return per_molecule


def _pool_frame(training: pd.DataFrame, score_table: pd.DataFrame, pool: _PoolSpec) -> pd.DataFrame:
    """Return the pool's training rows joined with their aggregate scores."""
    eligible = training.loc[training["candidate_id"].astype(str).isin(pool.candidate_ids)]
    joined = eligible.merge(
        score_table,
        on=["candidate_id", "source", "origin_class", "model_target"],
        how="left",
        validate="one_to_one",
    )
    if joined["aggregate_rank"].isna().any():
        raise ArmPlanningError(f"pool {pool.key} rows are missing aggregate scores")
    return joined.reset_index(drop=True)


def _ranked_arm(
    arm_id: str,
    pool: _PoolSpec,
    policy: str,
    direction: str,
    size: int,
    selected: pd.DataFrame,
) -> tuple[DeletionArm, pd.DataFrame]:
    """Build one deterministic ranked arm and its manifest rows."""
    if selected["candidate_id"].nunique() != size:
        raise ArmPlanningError(f"arm {arm_id} does not have {size} unique members")
    removed = tuple(sorted(selected["candidate_id"].astype(str)))
    arm = DeletionArm(
        arm_id=arm_id,
        eligible_pool=pool.pool,
        stratum=pool.stratum,
        policy=policy,
        direction=direction,
        batch_size=size,
        draw_number=None,
        removed_candidate_ids=removed,
    )
    rows = _arm_rows(selected, arm, selection_reason=f"{direction}est aggregate mean rank")
    return arm, rows


def _random_arms(
    pool: _PoolSpec,
    pool_index: int,
    pool_frame: pd.DataFrame,
    size: int,
    settings: BoostInSettings,
) -> list[tuple[DeletionArm, pd.DataFrame]]:
    """Draw the deterministic uniform-random arms for one pool and size."""
    rng = np.random.default_rng([RANDOM_DRAW_SEED, pool_index, size])
    arms: list[tuple[DeletionArm, pd.DataFrame]] = []
    for draw in range(settings.random_draws):
        positions = np.sort(rng.choice(len(pool_frame), size=size, replace=False))
        arm_id = f"{pool.key}:{POLICY_RANDOM}:k{size}:draw{draw:02d}"
        selected = pool_frame.iloc[positions]
        arm = DeletionArm(
            arm_id=arm_id,
            eligible_pool=pool.pool,
            stratum=pool.stratum,
            policy=POLICY_RANDOM,
            direction=None,
            batch_size=size,
            draw_number=draw,
            removed_candidate_ids=tuple(sorted(selected["candidate_id"].astype(str))),
        )
        rows = _arm_rows(selected, arm, selection_reason=f"uniform random draw {draw}")
        arms.append((arm, rows))
    return arms


def _matched_arms(
    pool: _PoolSpec,
    pool_index: int,
    pool_frame: pd.DataFrame,
    size: int,
    settings: BoostInSettings,
    n_train_full: int,
) -> tuple[list[tuple[DeletionArm, pd.DataFrame]], list[dict[str, Any]]]:
    """Build label/source-stratified control arms for both ranked directions.

    :returns: ``(planned_arms, summary_rows)``; infeasible matches appear only as summary rows
        with an explicit diagnostic and are never replaced by random selection.
    """
    ranked = pool_frame.sort_values(["rank_mean", "candidate_id"], kind="stable")
    healthy: list[tuple[DeletionArm, pd.DataFrame]] = []
    diagnostics: list[dict[str, Any]] = []
    for direction, reference in (
        ("high", ranked.iloc[:size]),
        ("low", ranked.iloc[-size:]),
    ):
        match = _match_direction(
            pool, pool_index, pool_frame, reference, direction, size, settings, n_train_full
        )
        healthy.extend(match.arms)
        diagnostics.extend(match.summary_rows)
    return healthy, diagnostics


def _match_direction(
    pool: _PoolSpec,
    pool_index: int,
    pool_frame: pd.DataFrame,
    reference: pd.DataFrame,
    direction: str,
    size: int,
    settings: BoostInSettings,
    n_train_full: int,
) -> _MatchResult:
    """Attempt one direction's stratified-control draws, trying successively coarser bins."""
    reference_ids = set(reference["candidate_id"].astype(str))
    candidates = pool_frame.loc[~pool_frame["candidate_id"].astype(str).isin(reference_ids)]
    attempts: list[str] = []
    for bin_count in settings.label_match_bins:
        binned_pool = _assign_bins(pool_frame, bin_count)
        binned_reference = binned_pool.loc[reference.index]
        required = (
            binned_reference.groupby(["source", "matching_bin"], sort=False)
            .size()
            .rename("required")
        )
        available = (
            binned_pool.loc[candidates.index]
            .groupby(["source", "matching_bin"], sort=False)
            .size()
            .rename("available")
        )
        comparison = pd.concat([required, available], axis=1).fillna(0).astype(int)
        deficits = comparison["required"] - comparison["available"]
        if bool((deficits > 0).any()):
            worst = int(deficits.max())
            deficient = int((deficits > 0).sum())
            attempts.append(
                f"bins={bin_count}: {deficient} deficient strata, worst shortfall {worst}"
            )
            continue
        return _materialise_matched(
            pool,
            pool_index,
            pool_frame,
            candidates,
            binned_pool,
            binned_reference,
            direction,
            size,
            settings,
            bin_count,
            comparison,
            attempts,
            n_train_full,
        )

    diagnostic = (
        "no configured binning permits a non-overlapping stratified control for "
        f"direction {direction}; " + "; ".join(attempts)
    )
    summary_rows = tuple(
        _infeasible_summary_row(
            arm_id=f"{pool.key}:label_source_matched_{direction}:k{size}:draw{draw:02d}",
            pool=pool,
            direction=direction,
            batch_size=size,
            draw_number=draw,
            n_train_full=n_train_full,
            diagnostic=diagnostic,
        )
        for draw in range(settings.matching_draws)
    )
    return _MatchResult(arms=(), summary_rows=summary_rows)


def _assign_bins(pool_frame: pd.DataFrame, bin_count: int) -> pd.DataFrame:
    """Assign per-source quantile label bins derived from the full candidate pool."""
    bins = np.zeros(len(pool_frame), dtype=int)
    for _, group in pool_frame.groupby("source", sort=False):
        positions = group.index.to_numpy()
        values = group["model_target"].astype(float)
        if values.nunique() <= 1:
            bins[positions] = 0
            continue
        _, edges = pd.qcut(values, q=bin_count, retbins=True, duplicates="drop")
        assigned = pd.cut(values, bins=edges, labels=False, include_lowest=True)
        bins[positions] = assigned.astype(int).to_numpy()
    return pool_frame.assign(matching_bin=bins)


def _materialise_matched(
    pool: _PoolSpec,
    pool_index: int,
    pool_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    binned_pool: pd.DataFrame,
    binned_reference: pd.DataFrame,
    direction: str,
    size: int,
    settings: BoostInSettings,
    bin_count: int,
    comparison: pd.DataFrame,
    attempts: list[str],
    n_train_full: int,
) -> _MatchResult:
    """Draw the matched batches once a feasible binning is found."""
    required = (
        binned_reference.groupby(["source", "matching_bin"], sort=False)
        .size()
        .rename("required")
        .reset_index()
    )
    candidates_binned = binned_pool.loc[candidates.index]
    planned: list[tuple[DeletionArm, pd.DataFrame]] = []
    summary_rows: list[dict[str, Any]] = []
    for draw in range(settings.matching_draws):
        rng = np.random.default_rng(
            [MATCHING_DRAW_SEED, pool_index, size, 0 if direction == "high" else 1, draw]
        )
        chosen_index: list[int] = []
        for stratum in required.itertuples(index=False):
            members = candidates_binned.loc[
                candidates_binned["source"].eq(stratum.source)
                & candidates_binned["matching_bin"].eq(stratum.matching_bin)
            ]
            if len(members) < int(stratum.required):
                raise ArmPlanningError(
                    f"matched arm construction changed feasibility for {pool.key} "
                    f"({stratum.source}, bin {stratum.matching_bin})"
                )
            positions = rng.choice(len(members), size=int(stratum.required), replace=False)
            chosen_index.extend(members.iloc[np.sort(positions)].index.tolist())
        selected = pool_frame.loc[sorted(set(chosen_index))]
        if len(selected) != size:
            raise ArmPlanningError(
                f"matched arm for {pool.key} direction {direction} draw {draw} has "
                f"{len(selected)} members, expected {size}"
            )
        arm_id = f"{pool.key}:label_source_matched_{direction}:k{size}:draw{draw:02d}"
        arm = DeletionArm(
            arm_id=arm_id,
            eligible_pool=pool.pool,
            stratum=pool.stratum,
            policy=f"label_source_matched_{direction}",
            direction=direction,
            batch_size=size,
            draw_number=draw,
            removed_candidate_ids=tuple(sorted(selected["candidate_id"].astype(str))),
        )
        rows = _arm_rows(
            selected,
            arm,
            selection_reason=(
                f"source/bin-stratified control draw {draw} ({bin_count} bins, "
                f"direction {direction})"
            ),
            binned=binned_pool,
        )
        planned.append((arm, rows))
        summary_rows.append(
            _summary_row(
                arm,
                pool,
                n_train_full,
                selected,
                f"source/bin-stratified control draw {draw} ({bin_count} bins, "
                f"direction {direction})",
                matching_bins_used=bin_count,
                matching_plan=_matching_plan_json(required, comparison),
                binned_reference=binned_reference,
                binned_matched=binned_pool.loc[selected.index],
                attempts=attempts,
            )
        )
    return _MatchResult(arms=tuple(planned), summary_rows=tuple(summary_rows))


def _arm_rows(
    selected: pd.DataFrame,
    arm: DeletionArm,
    *,
    selection_reason: str,
    binned: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the fixed-column manifest rows for one arm."""
    frame = selected
    if binned is not None:
        bins = binned.loc[frame.index, "matching_bin"].astype("Int64")
    else:
        bins = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    return pd.DataFrame(
        {
            "arm_id": arm.arm_id,
            "eligible_pool": arm.eligible_pool,
            "stratum": arm.stratum,
            "policy": arm.policy,
            "direction": arm.direction,
            "batch_size": arm.batch_size,
            "draw_number": arm.draw_number,
            "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
            "source": frame["source"].astype(str).to_numpy(),
            "origin_class": frame["origin_class"].astype(str).to_numpy(),
            "original_id": frame["original_id"].astype(str).to_numpy(),
            "inchikey": frame["inchikey"].astype(str).to_numpy(),
            "model_target": frame["model_target"].astype(float).to_numpy(),
            "pool_rank": frame["pool_rank"].astype(int).to_numpy(),
            "aggregate_rank": frame["aggregate_rank"].astype(int).to_numpy(),
            "rank_mean": frame["rank_mean"].astype(float).to_numpy(),
            "raw_score_mean": frame["raw_score_mean"].astype(float).to_numpy(),
            "matching_bin": bins,
            "selection_reason": selection_reason,
        }
    )


def _summary_row(
    arm: DeletionArm,
    pool: _PoolSpec,
    n_train_full: int,
    selected: pd.DataFrame,
    reason: str,
    *,
    matching_bins_used: int | None = None,
    matching_plan: str | None = None,
    binned_reference: pd.DataFrame | None = None,
    binned_matched: pd.DataFrame | None = None,
    attempts: list[str] | None = None,
) -> dict[str, Any]:
    """Build one arm-summary row with source and label diagnostics."""
    source_counts = selected["source"].astype(str).value_counts()
    row: dict[str, Any] = {
        "arm_id": arm.arm_id,
        "eligible_pool": arm.eligible_pool,
        "stratum": arm.stratum,
        "policy": arm.policy,
        "direction": arm.direction,
        "batch_size": arm.batch_size,
        "draw_number": arm.draw_number,
        "status": "planned",
        "requested_size": arm.batch_size,
        "achieved_size": int(len(selected)),
        "pool_size": len(pool.candidate_ids),
        "n_train_full": n_train_full,
        "n_train_pruned": n_train_full - arm.batch_size,
        "source_biogen": int(source_counts.get("biogen", 0)),
        "source_polaris": int(source_counts.get("polaris", 0)),
        "source_expansionrx": int(source_counts.get("expansionrx", 0)),
        "label_mean": float(selected["model_target"].astype(float).mean()),
        "label_min": float(selected["model_target"].astype(float).min()),
        "label_max": float(selected["model_target"].astype(float).max()),
        "label_wasserstein_overall": None,
        "label_wasserstein_max_source": None,
        "matching_bins_used": matching_bins_used,
        "matching_plan": matching_plan,
        "matching_diagnostic": "",
    }
    if binned_reference is not None and binned_matched is not None:
        row["label_wasserstein_overall"] = _wasserstein(
            binned_reference["model_target"].astype(float),
            binned_matched["model_target"].astype(float),
        )
        per_source: list[float] = []
        for source in sorted(set(binned_reference["source"].astype(str))):
            reference_values = binned_reference.loc[
                binned_reference["source"].astype(str).eq(source), "model_target"
            ].astype(float)
            matched_values = binned_matched.loc[
                binned_matched["source"].astype(str).eq(source), "model_target"
            ].astype(float)
            if len(reference_values) and len(matched_values):
                per_source.append(_wasserstein(reference_values, matched_values))
        row["label_wasserstein_max_source"] = max(per_source) if per_source else None
    if attempts:
        row["matching_diagnostic"] = "; ".join(attempts)
    row["matching_diagnostic"] = (row["matching_diagnostic"] + " " + reason).strip()
    return row


def _infeasible_summary_row(
    *,
    arm_id: str,
    pool: _PoolSpec,
    direction: str,
    batch_size: int,
    draw_number: int,
    n_train_full: int,
    diagnostic: str,
) -> dict[str, Any]:
    """Build the explicit diagnostic row for an infeasible matched arm."""
    return {
        "arm_id": arm_id,
        "eligible_pool": pool.pool,
        "stratum": pool.stratum,
        "policy": f"label_source_matched_{direction}",
        "direction": direction,
        "batch_size": batch_size,
        "draw_number": draw_number,
        "status": "infeasible",
        "requested_size": batch_size,
        "achieved_size": 0,
        "pool_size": len(pool.candidate_ids),
        "n_train_full": n_train_full,
        "n_train_pruned": None,
        "source_biogen": 0,
        "source_polaris": 0,
        "source_expansionrx": 0,
        "label_mean": None,
        "label_min": None,
        "label_max": None,
        "label_wasserstein_overall": None,
        "label_wasserstein_max_source": None,
        "matching_bins_used": None,
        "matching_plan": None,
        "matching_diagnostic": diagnostic,
    }


def _matching_plan_json(required: pd.DataFrame, comparison: pd.DataFrame) -> str:
    """Serialize the intended source × bin composition and pool availability."""
    payload = {
        "intended": {
            f"{row.source}|bin{int(row.matching_bin)}": int(row.required)
            for row in required.itertuples(index=False)
        },
        "candidate_pool": {
            f"{source}|bin{int(bin_id)}": int(value)
            for (source, bin_id), value in comparison["available"].items()
        },
    }
    return json.dumps(payload, sort_keys=True)


def _wasserstein(reference: pd.Series, matched: pd.Series) -> float:
    """One-dimensional earth-mover distance between two equal-size label samples."""
    left = np.sort(reference.to_numpy(dtype=float))
    right = np.sort(matched.to_numpy(dtype=float))
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    return float(np.mean(np.abs(left[:length] - right[:length])))


def _validate_plan(
    arms: list[DeletionArm],
    manifest: pd.DataFrame,
    training: pd.DataFrame,
    training_ids: set[str],
) -> None:
    """Prove every planned batch is a legal removal from full training."""
    training_size = len(training)
    for arm in arms:
        members = list(arm.removed_candidate_ids)
        if len(members) != len(set(members)):
            raise ArmPlanningError(f"arm {arm.arm_id} contains duplicate candidates")
        if len(members) != arm.batch_size:
            raise ArmPlanningError(
                f"arm {arm.arm_id} has {len(members)} members, expected {arm.batch_size}"
            )
        outside = sorted(set(members) - training_ids)
        if outside:
            raise ArmPlanningError(
                f"arm {arm.arm_id} removes candidates absent from full training: {outside[:5]}"
            )
        if training_size - arm.batch_size <= 0:
            raise ArmPlanningError(
                f"arm {arm.arm_id} would remove the entire training set "
                f"({training_size} rows, batch {arm.batch_size})"
            )
    planned_ids = set(manifest["arm_id"].astype(str))
    if planned_ids != {arm.arm_id for arm in arms}:
        raise ArmPlanningError("arm manifest does not cover exactly the planned arms")
