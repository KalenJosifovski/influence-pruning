"""Capped end-to-end dry run and command-level contract tests."""

import builtins
from pathlib import Path

import pandas as pd
import pytest
from conftest import write_config
from pytest import raises

from influence_pruning.config import RunConfig, load_config
from influence_pruning.errors import ConfigError
from influence_pruning.run import run_audit, run_dry_run, run_prune, verify


def _influence_available() -> bool:
    """Return whether the isolated influence dependency is importable."""
    from influence_pruning.attribution import require_boostin
    from influence_pruning.errors import InfluenceDependencyError

    try:
        require_boostin()
    except InfluenceDependencyError:
        return False
    return True


requires_influence = pytest.mark.skipif(
    not _influence_available(),
    reason="tree-influence is installed only in the isolated influence environment",
)


@requires_influence
def test_capped_dry_run_satisfies_the_whole_artifact_contract(
    dry_run_config: RunConfig,
) -> None:
    """A capped dry run produces every required artifact and passes verification."""
    run_path = run_dry_run(dry_run_config)
    facts = verify(run_path)
    assert facts["kind"] == "boostin_pruning"
    assert facts["dry_run"] is True
    assert facts["status"] == "completed"
    assert (run_path / "COMPLETED").is_file()
    for name in (
        "data_audit.parquet",
        "partition_manifest.parquet",
        "boostin_scores.parquet",
        "arm_manifest.parquet",
        "arm_summary.parquet",
        "predictions.parquet",
        "seed_metrics.parquet",
        "paired_effects.parquet",
        "bootstrap_draws.parquet",
    ):
        assert (run_path / name).is_file()
        frame = pd.read_parquet(run_path / name)
        if name.endswith(".parquet") and name != "exclusions.parquet":
            assert "dry_run" in frame.columns
            assert bool(frame["dry_run"].all()) if len(frame) else True

    scores = pd.read_parquet(run_path / "boostin_scores.parquet")
    manifest = pd.read_parquet(run_path / "partition_manifest.parquet")
    training_ids = set(manifest.loc[manifest["role"].eq("full_training"), "candidate_id"])
    assert set(scores["candidate_id"]) == training_ids
    evaluation_ids = set(manifest.loc[manifest["role"].eq("evaluation"), "candidate_id"])
    assert not (set(scores["candidate_id"]) & evaluation_ids)

    predictions = pd.read_parquet(run_path / "predictions.parquet")
    assert {"full_pooled", "local_only"} <= set(predictions["condition"])
    assert bool(predictions["condition_kind"].eq("pruned").any())
    assert predictions["seed"].nunique() == 1
    assert int(facts["n_arms"]) > 0
    assert facts["n_outcome_fits"] == len(dry_run_config.boostin.outcome_seeds) * (
        2 + int(facts["n_arms"])
    )

    effects = pd.read_parquet(run_path / "paired_effects.parquet")
    draws = pd.read_parquet(run_path / "bootstrap_draws.parquet")
    assert set(effects["comparison_id"]) == set(draws["comparison_id"])
    assert bool(effects["comparison_kind"].eq("full_vs_pruned").any())
    assert bool(effects["comparison_kind"].eq("ranked_vs_random_median").any())
    assert bool(effects["comparison_kind"].eq("ranked_vs_matched_median").any())
    policy_comparisons = effects.loc[
        effects["comparison_kind"].isin(["ranked_vs_random_median", "ranked_vs_matched_median"])
    ]
    assert policy_comparisons["control_policy"].notna().all()
    assert policy_comparisons["n_control_draws"].notna().all()
    assert policy_comparisons["control_arm_ids"].map(lambda value: isinstance(value, str)).all()


@requires_influence
def test_prune_command_refuses_a_dry_run_configuration(dry_run_config: RunConfig) -> None:
    """The real-run command refuses configurations marked dry_run."""
    with raises(ConfigError, match="dry_run: true"):
        run_prune(dry_run_config)


@requires_influence
def test_benchmark_exposes_planned_and_infeasible_arm_counts(
    tmp_path: Path, synthetic_paths: Path
) -> None:
    """The benchmark payload states the planned grid and any infeasible control groups."""
    import json

    from influence_pruning.run import run_benchmark

    config_path = write_config(
        tmp_path / "benchmark.yaml", synthetic_paths, tmp_path / "results-benchmark"
    )
    config = load_config(config_path)
    run_path = run_benchmark(config)
    facts = verify(run_path)
    payload = json.loads((run_path / "benchmark.json").read_text())
    for key in ("n_planned_arms", "n_infeasible_matched_arms", "infeasible_matched_groups"):
        assert key in payload
        assert key in facts
    assert payload["n_planned_arms"] == payload["n_arms"] == facts["n_planned_arms"]
    assert payload["n_infeasible_matched_arms"] == sum(
        group["n_draws"] for group in payload["infeasible_matched_groups"]
    )
    for group in payload["infeasible_matched_groups"]:
        for field in ("eligible_pool", "stratum", "direction", "batch_size", "matching_diagnostic"):
            assert field in group
    log_text = (run_path / "run.log").read_text()
    assert "planned arms" in log_text
    assert "stratified-control arms infeasible" in log_text


def test_dry_run_requires_explicit_caps(tmp_path: Path, synthetic_paths: Path) -> None:
    """A dry run without capped limits is rejected before any fitting."""
    config_path = write_config(
        tmp_path / "uncapped.yaml", synthetic_paths, tmp_path / "results", dry_run=True
    )
    config = load_config(config_path)
    with raises(ConfigError, match="capped"):
        run_dry_run(config)


def test_audit_never_imports_tree_influence(
    tmp_path: Path, synthetic_paths: Path, monkeypatch
) -> None:
    """The audit command works while tree-influence imports are hard-blocked."""
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object):
        if name.startswith("tree_influence"):
            raise ModuleNotFoundError("blocked for the audit guard test")
        return real_import(name, *args, **kwargs)

    config_path = write_config(tmp_path / "audit.yaml", synthetic_paths, tmp_path / "results-audit")
    config = load_config(config_path)
    monkeypatch.setattr(builtins, "__import__", blocked)
    run_path = run_audit(config)
    facts = verify(run_path)
    assert facts["kind"] == "audit"
    assert facts["n_fits"] == 0
    assert (run_path / "data_audit.parquet").is_file()
    assert (run_path / "partition_manifest.parquet").is_file()
