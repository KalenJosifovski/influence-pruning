"""Command orchestration: audit, benchmark, dry run, full run, and verification.

The CLI commands map onto this module. ``audit`` loads data and builds partitions but never
imports tree-influence or fits a model. ``dry-run`` and ``boostin-prune`` share one pipeline;
the dry run is the same code path on capped configuration limits, visibly marked in every
artifact. ``verify`` never writes or refits.
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd

from influence_pruning.artifacts import (
    KIND_AUDIT,
    KIND_BENCHMARK,
    KIND_RUN,
    RunDirectory,
    verify_run,
)
from influence_pruning.config import RunConfig
from influence_pruning.data import load_observations
from influence_pruning.errors import ConfigError
from influence_pruning.evaluate import evaluate_split
from influence_pruning.partitions import build_pruning_split
from influence_pruning.pruning_arms import ArmPlan, plan_arms

AUDIT_TABLES = (
    ("data_audit.parquet", "data_audit"),
    ("exclusions.parquet", "exclusions"),
    ("partition_manifest.parquet", "partition_manifest"),
)
RUN_TABLES = (
    ("predictions.parquet", "predictions"),
    ("seed_metrics.parquet", "metrics"),
    ("paired_effects.parquet", "effects"),
    ("bootstrap_draws.parquet", "bootstrap_draws"),
)


def _donor_availability_record(config: RunConfig, bundle: Any) -> dict[str, Any]:
    """Return a JSON-ready donor availability summary."""
    return {
        source: {
            "status": record.status,
            "column": record.column,
            "reason": record.reason,
        }
        for source, record in bundle.donor_availability.items()
    }


def run_audit(config: RunConfig) -> Path:
    """Load data, build partitions, and write audit artifacts without fitting models.

    :param config: resolved run configuration.
    :returns: the completed audit directory.
    """
    run = RunDirectory.create(config, KIND_AUDIT, command="audit")
    started = time.monotonic()
    try:
        bundle = load_observations(config)
        split = build_pruning_split(bundle)
        run.write_table("data_audit.parquet", bundle.data_audit)
        run.write_table("exclusions.parquet", bundle.exclusions)
        run.write_table("partition_manifest.parquet", split.partition_manifest)
        run.finalize(
            "completed",
            runtime_seconds=time.monotonic() - started,
            n_fits=0,
            extra={
                "endpoint": config.endpoint,
                "target": config.target,
                "donor_sources": list(config.donor_sources),
                "split_record": {key: _jsonify(value) for key, value in split.split_record.items()},
                "source_split_record": {
                    key: _jsonify(value) for key, value in bundle.split_record.items()
                },
                "donor_availability": _donor_availability_record(config, bundle),
            },
        )
    except Exception as exc:
        run.finalize(
            "failed",
            runtime_seconds=time.monotonic() - started,
            n_fits=0,
            note=f"{type(exc).__name__}: {exc}",
        )
        raise
    return run.path


def run_study(config: RunConfig, *, command: str) -> Path:
    """Run the complete conditional pruning study on the configured limits.

    :param config: resolved run configuration.
    :param command: CLI command name recorded in ``run_facts.json``.
    :returns: the completed run directory.
    """
    if config.dry_run and not config.limits.any_cap:
        raise ConfigError("dry-run requires at least one capped limit")
    run = RunDirectory.create(config, KIND_RUN, command=command)
    started = time.monotonic()
    try:
        run.logger.info(
            "loading observations (endpoint=%s, target=%s, donors=%s)",
            config.endpoint,
            config.target,
            list(config.donor_sources),
        )
        bundle = load_observations(config)
        split = build_pruning_split(bundle)
        run.logger.info(
            "partitions: train=%d selection=%d evaluation=%d",
            split.full_training.size,
            split.influence_selection.size,
            split.evaluation.size,
        )
        run.write_table("data_audit.parquet", bundle.data_audit)
        run.write_table("exclusions.parquet", bundle.exclusions)
        run.write_table("partition_manifest.parquet", split.partition_manifest)

        from influence_pruning.attribution import BoostInAttributor

        run.logger.info(
            "scoring %d training rows with BoostIn over %d selection molecules (seeds=%s)",
            split.full_training.size,
            split.influence_selection.size,
            list(config.boostin.score_seeds),
        )
        attributor = BoostInAttributor()
        scores = attributor.score(
            split.full_training,
            split.influence_selection,
            config.model,
            config.boostin.score_seeds,
        )
        run.write_table("boostin_scores.parquet", scores)

        plan = plan_arms(split, scores, config.boostin, config.donor_sources)
        run.write_table("arm_manifest.parquet", plan.manifest)
        run.write_table("arm_summary.parquet", plan.summary)
        infeasible = int(plan.summary["status"].eq("infeasible").sum())
        run.logger.info(
            "planned %d arms across %d pools (%d infeasible matched arms recorded)",
            len(plan.arms),
            len(plan.pool_sizes),
            infeasible,
        )

        run.logger.info(
            "refitting every condition for outcome seeds %s",
            list(config.boostin.outcome_seeds),
        )
        result = evaluate_split(
            split,
            plan.arms,
            config.model,
            config.boostin.outcome_seeds,
            config.bootstrap,
        )
        for name, attribute in RUN_TABLES:
            run.write_table(name, getattr(result, attribute))

        n_score_fits = len(config.boostin.score_seeds)
        n_outcome_fits = len(config.boostin.outcome_seeds) * (2 + len(plan.arms))
        run.finalize(
            "completed",
            runtime_seconds=time.monotonic() - started,
            n_fits=n_score_fits + n_outcome_fits,
            extra={
                "endpoint": config.endpoint,
                "target": config.target,
                "donor_sources": list(config.donor_sources),
                "n_score_fits": n_score_fits,
                "n_outcome_fits": n_outcome_fits,
                "n_arms": len(plan.arms),
                "n_infeasible_matched_arms": infeasible,
                "pool_sizes": plan.pool_sizes,
                "split_record": {key: _jsonify(value) for key, value in split.split_record.items()},
                "source_split_record": {
                    key: _jsonify(value) for key, value in bundle.split_record.items()
                },
                "donor_availability": _donor_availability_record(config, bundle),
            },
        )
    except Exception as exc:
        run.finalize(
            "failed",
            runtime_seconds=time.monotonic() - started,
            n_fits=0,
            note=f"{type(exc).__name__}: {exc}",
        )
        raise
    return run.path


def run_dry_run(config: RunConfig) -> Path:
    """Run the capped, visibly marked dry run of the complete artifact contract."""
    if not config.dry_run:
        raise ConfigError("dry-run requires dry_run: true in the configuration")
    if not config.limits.any_cap:
        raise ConfigError("dry-run requires explicitly capped configuration limits")
    return run_study(config, command="dry-run")


def run_prune(config: RunConfig) -> Path:
    """Run the real-data conditional BoostIn pruning episode."""
    if config.dry_run:
        raise ConfigError("this configuration is marked dry_run: true; use ipp dry-run")
    return run_study(config, command="boostin-prune")


def run_benchmark(config: RunConfig) -> Path:
    """Time one scoring, one full, and one pruned fit, then project the run budget.

    The projection multiplies measured per-fit times by the exact fit counts of the registered
    configuration: one pooled scoring fit per score seed, and ``2 + n_arms`` outcome fits per
    outcome seed (full pooled, local only, and every planned deletion arm).

    :param config: resolved run configuration.
    :returns: the completed benchmark directory.
    """
    run = RunDirectory.create(config, KIND_BENCHMARK, command="benchmark")
    started = time.monotonic()
    try:
        bundle = load_observations(config)
        split = build_pruning_split(bundle)

        from influence_pruning.attribution import BoostInAttributor

        score_started = time.monotonic()
        attributor = BoostInAttributor()
        scores = attributor.score(
            split.full_training,
            split.influence_selection,
            config.model,
            (config.boostin.score_seeds[0],),
        )
        score_seconds = time.monotonic() - score_started

        plan = plan_arms(split, scores, config.boostin, config.donor_sources)
        benchmark_arm = _benchmark_arm(plan)
        from influence_pruning.model import fit_predict

        full_started = time.monotonic()
        fit_predict(
            split.full_training.fingerprints,
            split.full_training.frame["model_target"].to_numpy(dtype=float),
            split.evaluation.fingerprints,
            config.model,
            config.boostin.outcome_seeds[0],
        )
        full_fit_seconds = time.monotonic() - full_started

        pruned_mask = ~split.full_training.frame["candidate_id"].astype(str).isin(
            set(benchmark_arm.removed_candidate_ids)
        )
        pruned = split.full_training.frame.loc[pruned_mask]
        pruned_started = time.monotonic()
        fit_predict(
            split.full_training.fingerprints[pruned_mask.to_numpy()],
            pruned["model_target"].to_numpy(dtype=float),
            split.evaluation.fingerprints,
            config.model,
            config.boostin.outcome_seeds[0],
        )
        pruned_fit_seconds = time.monotonic() - pruned_started

        n_score_fits = len(config.boostin.score_seeds)
        n_outcome_conditions = 2 + len(plan.arms)
        n_outcome_fits = len(config.boostin.outcome_seeds) * n_outcome_conditions
        projected_seconds = (
            n_score_fits * score_seconds
            + n_outcome_fits * (full_fit_seconds + pruned_fit_seconds) / 2
        )
        infeasible_groups = infeasible_matched_groups(plan)
        n_infeasible = int(plan.summary["status"].eq("infeasible").sum())
        run.logger.info(
            "benchmark: %d planned arms; %d stratified-control arms infeasible",
            len(plan.arms),
            n_infeasible,
        )
        payload = {
            "endpoint": config.endpoint,
            "target": config.target,
            "dry_run": config.dry_run,
            "n_arms": len(plan.arms),
            "n_planned_arms": len(plan.arms),
            "n_infeasible_matched_arms": n_infeasible,
            "infeasible_matched_groups": infeasible_groups,
            "n_score_fits": n_score_fits,
            "n_outcome_conditions": n_outcome_conditions,
            "n_outcome_fits": n_outcome_fits,
            "n_total_fits": n_score_fits + n_outcome_fits,
            "pool_sizes": plan.pool_sizes,
            "score_fit_seconds": round(score_seconds, 3),
            "full_fit_seconds": round(full_fit_seconds, 3),
            "pruned_fit_seconds": round(pruned_fit_seconds, 3),
            "benchmark_arm_id": benchmark_arm.arm_id,
            "projected_run_seconds": round(projected_seconds, 1),
            "projected_run_hours": round(projected_seconds / 3600.0, 2),
            "note": (
                "Projection assumes the measured pruned-fit time applies to every arm; "
                "it excludes bootstrap and artifact I/O."
            ),
        }
        run.write_json("benchmark.json", payload)
        run.finalize(
            "completed",
            runtime_seconds=time.monotonic() - started,
            n_fits=1 + 1 + 1,
            extra=payload,
        )
    except Exception as exc:
        run.finalize(
            "failed",
            runtime_seconds=time.monotonic() - started,
            n_fits=0,
            note=f"{type(exc).__name__}: {exc}",
        )
        raise
    return run.path


def _benchmark_arm(plan: ArmPlan):
    """Return the deterministic midpoint arm for the timing fit."""
    if not plan.arms:
        raise ConfigError("benchmark requires at least one planned arm")
    midpoint = len(plan.arms) // 2
    return plan.arms[midpoint]


def infeasible_matched_groups(plan: ArmPlan) -> list[dict[str, Any]]:
    """Summarise infeasible stratified-control groups for the benchmark payload.

    One entry per ``eligible_pool × stratum × direction × batch_size`` group that could not be
    matched without reusing the scored batch, naming the number of failed draws and the
    planning diagnostic. The nominal arm grid minus these draws is the planned arm count.

    :param plan: arm plan with its summary table.
    :returns: JSON-ready group records, empty when every control was constructed.
    """
    infeasible = plan.summary.loc[plan.summary["status"].eq("infeasible")]
    if infeasible.empty:
        return []
    grouped = (
        infeasible.groupby(
            ["eligible_pool", "stratum", "direction", "batch_size"],
            as_index=False,
            sort=False,
        )
        .agg(n_draws=("arm_id", "size"), matching_diagnostic=("matching_diagnostic", "first"))
        .sort_values(["eligible_pool", "stratum", "direction", "batch_size"], kind="stable")
    )
    return [
        {
            "eligible_pool": str(row.eligible_pool),
            "stratum": str(row.stratum),
            "direction": str(row.direction),
            "batch_size": int(row.batch_size),
            "n_draws": int(row.n_draws),
            "matching_diagnostic": str(row.matching_diagnostic),
        }
        for row in grouped.itertuples(index=False)
    ]


def verify(path: str | Path) -> dict[str, Any]:
    """Read-only verification of a completed run directory."""
    return verify_run(path)


def _jsonify(value: Any) -> Any:
    """Return a JSON-ready scalar or list."""
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
