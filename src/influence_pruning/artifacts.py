"""Run directories, artifact contracts, and read-only integrity verification.

A completed run is self-contained under ``results/<kind>/<timestamp>_<endpoint>_<target>_<hash>``.
``COMPLETED`` is written only after every required artifact exists and passes schema validation.
``verify_run`` never writes, refits, or imports model dependencies.
"""

import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from influence_pruning.config import RunConfig
from influence_pruning.data import AUDIT_COLUMNS, EXCLUSION_COLUMNS
from influence_pruning.errors import ArtifactError
from influence_pruning.evaluate import (
    BOOTSTRAP_DRAW_COLUMNS,
    EFFECT_COLUMNS,
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
)
from influence_pruning.partitions import MANIFEST_COLUMNS as PARTITION_MANIFEST_COLUMNS
from influence_pruning.pruning_arms import (
    MANIFEST_COLUMNS as ARM_MANIFEST_COLUMNS,
)
from influence_pruning.pruning_arms import (
    SUMMARY_COLUMNS as ARM_SUMMARY_COLUMNS,
)

KIND_RUN = "boostin_pruning"
KIND_AUDIT = "audit"
KIND_BENCHMARK = "benchmark"
KINDS = (KIND_RUN, KIND_AUDIT, KIND_BENCHMARK)
COMPLETION_MARKER = "COMPLETED"
TRACKED_PACKAGES = (
    "xgboost",
    "rdkit",
    "numpy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "tree_influence",
)
ARTIFACT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "data_audit.parquet": tuple(AUDIT_COLUMNS),
    "exclusions.parquet": tuple(EXCLUSION_COLUMNS),
    "partition_manifest.parquet": tuple(PARTITION_MANIFEST_COLUMNS),
    "boostin_scores.parquet": (
        "source",
        "origin_class",
        "candidate_id",
        "original_id",
        "inchikey",
        "model_target",
        "model_seed",
        "raw_score",
        "within_seed_rank",
        "train_prediction",
        "train_residual",
        "n_train",
        "n_selection",
        "raw_score_mean",
        "raw_score_sd",
        "rank_mean",
        "rank_sd",
        "n_score_seeds",
        "aggregate_rank",
        "max_tanimoto_to_selection",
        "beneficial_first",
        "score_direction",
        "score_model",
    ),
    "arm_manifest.parquet": tuple(ARM_MANIFEST_COLUMNS),
    "arm_summary.parquet": tuple(ARM_SUMMARY_COLUMNS),
    "predictions.parquet": tuple(PREDICTION_COLUMNS),
    "seed_metrics.parquet": tuple(METRIC_COLUMNS),
    "paired_effects.parquet": tuple(EFFECT_COLUMNS),
    "bootstrap_draws.parquet": tuple(BOOTSTRAP_DRAW_COLUMNS),
}
REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    KIND_RUN: (
        "config.source.yaml",
        "config.resolved.yaml",
        "environment.json",
        "data_audit.parquet",
        "exclusions.parquet",
        "partition_manifest.parquet",
        "boostin_scores.parquet",
        "arm_manifest.parquet",
        "arm_summary.parquet",
        "predictions.parquet",
        "seed_metrics.parquet",
        "paired_effects.parquet",
        "bootstrap_draws.parquet",
        "run_facts.json",
        "run.log",
        COMPLETION_MARKER,
    ),
    KIND_AUDIT: (
        "config.source.yaml",
        "config.resolved.yaml",
        "environment.json",
        "data_audit.parquet",
        "exclusions.parquet",
        "partition_manifest.parquet",
        "run_facts.json",
        "run.log",
        COMPLETION_MARKER,
    ),
    KIND_BENCHMARK: (
        "config.source.yaml",
        "config.resolved.yaml",
        "environment.json",
        "benchmark.json",
        "run_facts.json",
        "run.log",
        COMPLETION_MARKER,
    ),
}

_TABLE_ARTIFACTS = tuple(ARTIFACT_SCHEMAS)


def package_versions() -> dict[str, str]:
    """Return versions of the packages recorded in run manifests."""
    versions = {"python": sys.version.split()[0]}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_payload() -> dict[str, Any]:
    """Collect the environment payload recorded for a run."""
    try:
        memory = platform.uname()
        machine = f"{memory.system} {memory.release} {memory.machine}"
    except Exception:  # pragma: no cover - platform introspection is best-effort
        machine = "unknown"
    return {
        "versions": package_versions(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": machine,
        "argv": list(sys.argv),
    }


def make_logger(run_id: str, log_path: Path) -> logging.Logger:
    """Create a run logger writing to both a file and stderr."""
    logger = logging.getLogger(f"influence_pruning.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@dataclass(slots=True)
class RunDirectory:
    """One run directory holding configuration, artifacts, and completion state.

    :param kind: one of :data:`KINDS`.
    :param run_id: directory name encoding timestamp, endpoint, target, and config hash.
    :param outputs: artifact file names written so far, recorded in ``run_facts.json``.
    """

    kind: str
    run_id: str
    path: Path
    config_hash: str
    dry_run: bool
    logger: logging.Logger
    environment: dict[str, Any]
    outputs: list[str] = field(default_factory=list)
    n_fits: int = 0

    @classmethod
    def create(cls, config: RunConfig, kind: str, *, command: str) -> "RunDirectory":
        """Create a new timestamped run directory.

        :param config: resolved run configuration.
        :param kind: artifact-contract kind.
        :param command: CLI command name recorded in ``run_facts.json``.
        :returns: the created directory handle.
        :raises ArtifactError: the kind is unknown.
        """
        if kind not in KINDS:
            raise ArtifactError(f"unknown run kind {kind!r}; expected one of {KINDS}")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_id = f"{timestamp}_{config.endpoint}_{config.target}_{config.hash[:8]}"
        run_id = base_id
        counter = 1
        path = config.results_root / kind / run_id
        while path.exists():
            counter += 1
            run_id = f"{base_id}-{counter}"
            path = config.results_root / kind / run_id
        path.mkdir(parents=True)
        environment = environment_payload()
        environment.update({"kind": kind, "command": command, "dry_run": bool(config.dry_run)})
        run = cls(
            kind=kind,
            run_id=run_id,
            path=path,
            config_hash=config.hash,
            dry_run=bool(config.dry_run),
            logger=make_logger(run_id, path / "run.log"),
            environment=environment,
        )
        run.write_environment()
        run.write_resolved_config(config)
        run.logger.info("run %s created (kind=%s, dry_run=%s)", run_id, kind, config.dry_run)
        return run

    def write_environment(self) -> None:
        """Persist the environment payload."""
        (self.path / "environment.json").write_text(
            json.dumps(self.environment, indent=2, sort_keys=True)
        )
        self.record_output("environment.json")

    def write_resolved_config(self, config: RunConfig) -> None:
        """Persist the resolved configuration and the original config file."""
        payload = config.as_mapping()
        payload["config_hash"] = config.hash
        (self.path / "config.resolved.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        self.record_output("config.resolved.yaml")
        if config.source_path is not None and config.source_path.is_file():
            (self.path / "config.source.yaml").write_text(config.source_path.read_text())
            self.record_output("config.source.yaml")

    def write_text(self, name: str, content: str) -> None:
        """Write a text artifact and record it in the completion manifest."""
        (self.path / name).write_text(content)
        self.record_output(name)

    def write_json(self, name: str, payload: Any) -> None:
        """Write a JSON artifact and record it in the completion manifest."""
        (self.path / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        self.record_output(name)

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        """Write a parquet table, marking dry runs visibly, and record the artifact."""
        table = frame.copy()
        if self.dry_run:
            table["dry_run"] = True
        table.to_parquet(self.path / name, index=False)
        self.record_output(name)

    def record_output(self, name: str) -> None:
        """Record an artifact file name once."""
        if name not in self.outputs:
            self.outputs.append(name)

    def finalize(
        self,
        status: str,
        *,
        runtime_seconds: float,
        n_fits: int,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate the artifact contract, write ``run_facts.json``, and mark completion.

        :param status: ``"completed"`` or ``"failed"``.
        :param runtime_seconds: wall time of the run.
        :param n_fits: number of model fits performed.
        :param note: optional failure detail or completion note.
        :param extra: additional JSON-ready facts.
        :returns: the facts payload written to disk.
        """
        self.n_fits = int(n_fits)
        facts: dict[str, Any] = {
            "status": status,
            "kind": self.kind,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "dry_run": self.dry_run,
            "runtime_seconds": round(float(runtime_seconds), 3),
            "n_fits": int(n_fits),
            "outputs": sorted(self.outputs),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "argv": list(sys.argv),
        }
        if note:
            facts["note"] = note
        if extra:
            facts.update(extra)
        if status == "completed":
            self.record_output("run_facts.json")
            facts["outputs"] = sorted(self.outputs)
            (self.path / "run_facts.json").write_text(
                json.dumps(facts, indent=2, sort_keys=True, default=str)
            )
            validate_artifact_set(
                self.path, self.kind, dry_run=self.dry_run, allow_missing=(COMPLETION_MARKER,)
            )
            (self.path / COMPLETION_MARKER).write_text(
                f"{self.run_id}\n{self.config_hash}\n{facts['finished_at']}\n"
            )
        else:
            (self.path / "run_facts.json").write_text(
                json.dumps(facts, indent=2, sort_keys=True, default=str)
            )
        return facts


def validate_artifact_set(
    path: Path, kind: str, *, dry_run: bool, allow_missing: tuple[str, ...] = ()
) -> None:
    """Validate that every required artifact for a kind exists and satisfies its schema.

    :param path: run directory.
    :param kind: artifact-contract kind.
    :param dry_run: whether artifacts must carry the visible dry-run marker.
    :param allow_missing: artifact names excused from the presence check (used while
        ``run_facts.json`` and the completion marker are still being written).
    :raises ArtifactError: the run is incomplete or schema-inconsistent.
    """
    if kind not in REQUIRED_ARTIFACTS:
        raise ArtifactError(f"unknown run kind {kind!r}")
    missing = [
        name
        for name in REQUIRED_ARTIFACTS[kind]
        if name not in allow_missing and not (path / name).is_file()
    ]
    if missing:
        raise ArtifactError(f"{path.name}: missing required artifacts {missing}")
    for name in REQUIRED_ARTIFACTS[kind]:
        if name not in ARTIFACT_SCHEMAS:
            continue
        frame = pd.read_parquet(path / name)
        required = ARTIFACT_SCHEMAS[name]
        absent = [column for column in required if column not in frame.columns]
        if absent:
            raise ArtifactError(f"{path.name}/{name}: missing columns {absent}")
        if dry_run and "dry_run" not in frame.columns:
            raise ArtifactError(f"{path.name}/{name}: dry run artifact is not marked dry_run")


def verify_run(path: str | Path) -> dict[str, Any]:
    """Read-only integrity verification of a completed run directory.

    :param path: run directory.
    :returns: parsed ``run_facts.json``.
    :raises ArtifactError: the directory is incomplete, hash-mismatched, or schema-inconsistent.
    """
    run_path = Path(path)
    if not run_path.is_dir():
        raise ArtifactError(f"run directory not found: {run_path}")
    facts_path = run_path / "run_facts.json"
    if not facts_path.is_file():
        raise ArtifactError(f"{run_path.name}: run_facts.json is missing")
    facts = json.loads(facts_path.read_text())
    if facts.get("status") != "completed":
        raise ArtifactError(f"{run_path.name}: status is {facts.get('status')!r}, not completed")
    kind = str(facts.get("kind", ""))
    if kind not in REQUIRED_ARTIFACTS:
        raise ArtifactError(f"{run_path.name}: unknown artifact kind {kind!r}")
    marker = run_path / COMPLETION_MARKER
    if not marker.is_file():
        raise ArtifactError(f"{run_path.name}: completion marker is missing")
    marker_lines = marker.read_text().splitlines()
    if len(marker_lines) < 2 or marker_lines[1].strip() != str(facts.get("config_hash")):
        raise ArtifactError(f"{run_path.name}: completion marker hash disagrees with run facts")
    resolved = yaml.safe_load((run_path / "config.resolved.yaml").read_text())
    resolved_hash = str(resolved.get("config_hash", ""))
    if resolved_hash != str(facts.get("config_hash")):
        raise ArtifactError(
            f"{run_path.name}: resolved configuration hash {resolved_hash!r} disagrees with "
            f"run facts {facts.get('config_hash')!r}"
        )
    if bool(resolved.get("dry_run", False)) != bool(facts.get("dry_run", False)):
        raise ArtifactError(f"{run_path.name}: dry-run markers disagree between artifacts")
    validate_artifact_set(run_path, kind, dry_run=bool(facts.get("dry_run", False)))
    return facts
