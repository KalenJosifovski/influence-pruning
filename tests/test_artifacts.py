"""Artifact-contract, completion-marker, and read-only verification tests."""

from pathlib import Path

import pandas as pd
import yaml
from conftest import write_config
from pytest import raises

from influence_pruning.artifacts import (
    ARTIFACT_SCHEMAS,
    KIND_BENCHMARK,
    KIND_RUN,
    RunDirectory,
    validate_artifact_set,
    verify_run,
)
from influence_pruning.config import load_config
from influence_pruning.errors import ArtifactError


def _config(tmp_path: Path, synthetic_paths: Path, **overrides):
    """Write and load a synthetic configuration."""
    config_path = write_config(
        tmp_path / "config.yaml", synthetic_paths, tmp_path / "results", **overrides
    )
    return load_config(config_path)


def _write_fake_tables(run: RunDirectory) -> None:
    """Write empty tables with the required artifact schemas."""
    for name, columns in ARTIFACT_SCHEMAS.items():
        frame = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
        run.write_table(name, frame)


def test_finalize_requires_every_required_artifact(tmp_path: Path, synthetic_paths: Path) -> None:
    """Completion is refused while any required artifact is missing."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    with raises(ArtifactError, match="missing required artifacts"):
        run.finalize("completed", runtime_seconds=1.0, n_fits=0)
    assert not (run.path / "COMPLETED").exists()


def test_complete_run_writes_marker_and_verifies(tmp_path: Path, synthetic_paths: Path) -> None:
    """A complete run writes COMPLETED and passes read-only verification."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    _write_fake_tables(run)
    facts = run.finalize("completed", runtime_seconds=2.0, n_fits=7)
    assert facts["status"] == "completed"
    assert (run.path / "COMPLETED").exists()
    verified = verify_run(run.path)
    assert verified["config_hash"] == config.hash
    assert verified["n_fits"] == 7


def test_verify_rejects_a_deleted_artifact(tmp_path: Path, synthetic_paths: Path) -> None:
    """Verification refuses a run whose predictions artifact was removed."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    _write_fake_tables(run)
    run.finalize("completed", runtime_seconds=1.0, n_fits=0)
    (run.path / "predictions.parquet").unlink()
    with raises(ArtifactError, match="missing required artifacts"):
        verify_run(run.path)


def test_verify_rejects_a_hash_mismatch(tmp_path: Path, synthetic_paths: Path) -> None:
    """A resolved-configuration hash that disagrees with run facts is rejected."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    _write_fake_tables(run)
    run.finalize("completed", runtime_seconds=1.0, n_fits=0)
    resolved_path = run.path / "config.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["config_hash"] = "deadbeefdeadbeef"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    with raises(ArtifactError, match="disagrees"):
        verify_run(run.path)


def test_verify_rejects_schema_inconsistent_tables(tmp_path: Path, synthetic_paths: Path) -> None:
    """A table missing required columns is rejected."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    _write_fake_tables(run)
    run.finalize("completed", runtime_seconds=1.0, n_fits=0)
    pd.DataFrame({"not_a_column": [1]}).to_parquet(run.path / "predictions.parquet", index=False)
    with raises(ArtifactError, match="missing columns"):
        verify_run(run.path)


def test_verify_is_read_only(tmp_path: Path, synthetic_paths: Path) -> None:
    """Verification never writes to the run directory."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    _write_fake_tables(run)
    run.finalize("completed", runtime_seconds=1.0, n_fits=0)

    def snapshot() -> dict[str, int]:
        return {
            str(path.relative_to(run.path)): path.stat().st_mtime_ns
            for path in sorted(run.path.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    verify_run(run.path)
    assert snapshot() == before


def test_dry_run_artifacts_carry_the_visible_marker(tmp_path: Path, synthetic_paths: Path) -> None:
    """Every dry-run table is marked dry_run, and unmarked tables are rejected."""
    config = _config(
        tmp_path,
        synthetic_paths,
        dry_run=True,
        limits={"max_target_train": 20, "max_target_selection": 20, "max_donor_rows": 20},
    )
    run = RunDirectory.create(config, KIND_RUN, command="dry-run")
    _write_fake_tables(run)
    from influence_pruning.artifacts import COMPLETION_MARKER

    for name in ARTIFACT_SCHEMAS:
        assert "dry_run" in pd.read_parquet(run.path / name).columns
    validate_artifact_set(
        run.path, KIND_RUN, dry_run=True, allow_missing=("run_facts.json", COMPLETION_MARKER)
    )
    # An unmarked table fails validation for a dry run.
    frame = pd.read_parquet(run.path / "predictions.parquet").drop(columns=["dry_run"])
    frame.to_parquet(run.path / "predictions.parquet", index=False)
    with raises(ArtifactError, match="not marked dry_run"):
        validate_artifact_set(
            run.path, KIND_RUN, dry_run=True, allow_missing=("run_facts.json", COMPLETION_MARKER)
        )


def test_benchmark_kind_requires_its_payload(tmp_path: Path, synthetic_paths: Path) -> None:
    """The benchmark contract requires benchmark.json."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_BENCHMARK, command="benchmark")
    run.write_json("benchmark.json", {"n_total_fits": 3})
    run.finalize("completed", runtime_seconds=1.0, n_fits=3)
    assert verify_run(run.path)["kind"] == KIND_BENCHMARK
    (run.path / "benchmark.json").unlink()
    with raises(ArtifactError, match="missing required artifacts"):
        verify_run(run.path)


def test_failed_run_writes_facts_without_marker(tmp_path: Path, synthetic_paths: Path) -> None:
    """A failed run records its failure and cannot be verified as complete."""
    config = _config(tmp_path, synthetic_paths)
    run = RunDirectory.create(config, KIND_RUN, command="test")
    facts = run.finalize("failed", runtime_seconds=1.0, n_fits=0, note="boom")
    assert facts["status"] == "failed"
    assert not (run.path / "COMPLETED").exists()
    with raises(ArtifactError, match="not completed"):
        verify_run(run.path)
