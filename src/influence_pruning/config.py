"""Typed run configuration for the conditional BoostIn pruning study.

One YAML file per target rotation. All paths and all scientifically relevant values are
resolved and validated before data loading or fitting begins. The parser fails closed on
invalid target/donor combinations, duplicate seeds, impossible batch sizes, unknown split
modes, invalid label-match bins, and unresolved endpoint contracts.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from influence_pruning.contracts import resolve_contract
from influence_pruning.errors import ConfigError

STUDY_KIND = "boostin_pruning"
STUDY_PHASES = ("exploratory",)
ENDPOINTS = ("hlm_clint",)
SOURCES = ("expansionrx", "biogen", "polaris")
TARGETS = SOURCES
EXPANSIONRX_TEST_MODES = ("temporal_halves",)
BUTINA_TEST_MODES = ("butina_halves",)
DATASET_KEYS = ("expansionrx_train", "expansionrx_test", "biogen", "polaris")
COMPARISON_TARGET_SCALES = ("log10 mL/min/kg",)


@dataclass(frozen=True, slots=True)
class PathSet:
    """Local dataset file paths, resolved at configuration time."""

    expansionrx_train: Path
    expansionrx_test: Path
    biogen: Path
    polaris: Path

    def as_mapping(self) -> dict[str, str]:
        """Return the resolved paths keyed by dataset key."""
        return {key: str(getattr(self, key)) for key in DATASET_KEYS}


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """Target test-partition settings.

    :param test_mode: how the target test partition becomes selection and evaluation
        surfaces (``temporal_halves`` for ExpansionRx, ``butina_halves`` for rotated targets).
    :param target_test_selection_fraction: fraction of the target test partition assigned to
        the influence-selection surface; the remainder forms the evaluation surface.
    :param biogen_target_train_fraction: fraction of Biogen molecules assigned to target
        training when Biogen is the rotated target; the remainder is the test partition.
    :param butina_distance_threshold: Tanimoto distance cut for whole-cluster allocations.
    :param butina_seed: deterministic seed for whole-cluster allocations.
    """

    test_mode: str
    target_test_selection_fraction: float = 0.50
    biogen_target_train_fraction: float = 0.80
    butina_distance_threshold: float = 0.40
    butina_seed: int = 0


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Fingerprint and XGBoost parameters shared by scoring and every refit."""

    radius: int = 2
    n_bits: int = 2048
    chirality: bool = False
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    n_jobs: int = -1

    def xgboost_params(self, seed: int) -> dict[str, Any]:
        """Return `XGBRegressor` keyword arguments for one matched seed.

        :param seed: model seed shared across every condition in a comparison.
        :returns: keyword arguments including the fixed parameter set.
        """
        return {
            "objective": "reg:squarederror",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "random_state": seed,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True, slots=True)
class BoostInSettings:
    """BoostIn scoring, pruning-arm, and control settings.

    :param score_seeds: model seeds used to score every pooled training observation.
    :param outcome_seeds: matched model seeds used for every refit condition.
    :param batch_sizes: deletion batch sizes, strictly increasing and positive.
    :param random_draws: deterministic uniform-random batches per pool and batch size.
    :param matching_draws: deterministic label/source-stratified control batches per direction;
        they preserve source composition and within-source label quantile bins, not exact
        continuous label values.
    :param label_match_bins: successive quantile binnings tried for stratified-control
        construction, from finest to coarsest (for example deciles, then quintiles, then
        terciles).
    """

    score_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    outcome_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    batch_sizes: tuple[int, ...] = (25, 50, 100, 150)
    random_draws: int = 10
    matching_draws: int = 10
    label_match_bins: tuple[int, ...] = (10, 5, 3)

    def require_capacity(self, pool_sizes: Mapping[str, int]) -> None:
        """Reject batch sizes that cannot form disjoint ranked and matched batches.

        The ``boostin_high`` and ``boostin_low`` batches at one size must be distinct
        experimental arms, and a stratified control must draw from the pool excluding the scored
        reference batch. Both requirements bound the batch size by ``floor(pool_size / 2)``.

        :param pool_sizes: eligible row count per planned pool identifier.
        :raises ConfigError: any configured batch size exceeds a pool's capacity.
        """
        violations: list[str] = []
        for pool in sorted(pool_sizes):
            size = int(pool_sizes[pool])
            capacity = size // 2
            invalid = [batch for batch in self.batch_sizes if batch > capacity]
            if invalid:
                violations.append(
                    f"{pool} has {size} eligible rows (largest disjoint batch {capacity}; "
                    f"requested {invalid})"
                )
        if violations:
            raise ConfigError(
                "boostin.batch_sizes exceed eligible-pool capacity; " + "; ".join(violations)
            )


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    """Cluster-bootstrap settings for paired evaluation-molecule uncertainty."""

    n_resamples: int = 2000
    cluster_distance_threshold: float = 0.40
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Limits:
    """Dry-run caps applied after eligibility filtering; `None` means no cap."""

    max_target_train: int | None = None
    max_target_selection: int | None = None
    max_donor_rows: int | None = None

    @property
    def any_cap(self) -> bool:
        """Whether at least one cap is configured."""
        return any(
            limit is not None
            for limit in (self.max_target_train, self.max_target_selection, self.max_donor_rows)
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Fully resolved configuration for one pipeline invocation."""

    phase: str
    endpoint: str
    target: str
    donor_sources: tuple[str, ...]
    paths: PathSet
    results_root: Path
    split: SplitSettings
    model: ModelSettings
    boostin: BoostInSettings
    bootstrap: BootstrapSettings
    limits: Limits
    contract_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    dry_run: bool = False
    source_path: Path | None = None
    study_kind: str = STUDY_KIND

    @property
    def hash(self) -> str:
        """Return a stable content hash of the resolved configuration."""
        payload = json.dumps(self.as_mapping(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_mapping(self) -> dict[str, Any]:
        """Return the resolved configuration as a nested mapping."""
        return {
            "study": {"kind": self.study_kind, "phase": self.phase},
            "endpoint": self.endpoint,
            "target": self.target,
            "donor_sources": list(self.donor_sources),
            "paths": self.paths.as_mapping(),
            "results_root": str(self.results_root),
            "split": {
                "test_mode": self.split.test_mode,
                "target_test_selection_fraction": self.split.target_test_selection_fraction,
                "biogen_target_train_fraction": self.split.biogen_target_train_fraction,
                "butina_distance_threshold": self.split.butina_distance_threshold,
                "butina_seed": self.split.butina_seed,
            },
            "model": {
                "radius": self.model.radius,
                "n_bits": self.model.n_bits,
                "chirality": self.model.chirality,
                "n_estimators": self.model.n_estimators,
                "learning_rate": self.model.learning_rate,
                "max_depth": self.model.max_depth,
                "min_child_weight": self.model.min_child_weight,
                "subsample": self.model.subsample,
                "colsample_bytree": self.model.colsample_bytree,
                "reg_lambda": self.model.reg_lambda,
                "tree_method": self.model.tree_method,
                "n_jobs": self.model.n_jobs,
            },
            "boostin": {
                "score_seeds": list(self.boostin.score_seeds),
                "outcome_seeds": list(self.boostin.outcome_seeds),
                "batch_sizes": list(self.boostin.batch_sizes),
                "random_draws": self.boostin.random_draws,
                "matching_draws": self.boostin.matching_draws,
                "label_match_bins": list(self.boostin.label_match_bins),
            },
            "bootstrap": {
                "n_resamples": self.bootstrap.n_resamples,
                "cluster_distance_threshold": self.bootstrap.cluster_distance_threshold,
                "seed": self.bootstrap.seed,
            },
            "limits": {
                "max_target_train": self.limits.max_target_train,
                "max_target_selection": self.limits.max_target_selection,
                "max_donor_rows": self.limits.max_donor_rows,
            },
            "contract_overrides": self.contract_overrides,
            "dry_run": self.dry_run,
        }


def _mapping(value: Any, where: str) -> dict[str, Any]:
    """Return `value` as a mapping, raising `ConfigError` otherwise."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one configuration section as a mapping."""
    return _mapping(raw.get(key), key)


def _sequence(values: Any, where: str) -> list[Any]:
    """Return `values` as a list, raising `ConfigError` otherwise."""
    if not isinstance(values, (list, tuple)):
        raise ConfigError(f"{where} must be a list")
    return list(values)


def _int_tuple(values: Any, where: str) -> tuple[int, ...]:
    """Parse a sequence of positive integers."""
    items = _sequence(values, where)
    parsed = tuple(int(item) for item in items)
    if not parsed or any(item <= 0 for item in parsed):
        raise ConfigError(f"{where} must be non-empty positive integers")
    return parsed


def _seed_tuple(values: Any, where: str) -> tuple[int, ...]:
    """Parse a non-empty sequence of non-negative integer seeds."""
    items = _sequence(values, where)
    parsed = tuple(int(item) for item in items)
    if not parsed or any(item < 0 for item in parsed):
        raise ConfigError(f"{where} must be non-empty non-negative integers")
    return parsed


def _string_tuple(values: Any, where: str) -> tuple[str, ...]:
    """Parse a non-empty sequence of strings."""
    items = _sequence(values, where)
    parsed = tuple(str(item) for item in items)
    if not parsed or any(not item for item in parsed):
        raise ConfigError(f"{where} must be a non-empty list of strings")
    return parsed


def _unique_tuple(values: tuple[Any, ...], where: str) -> tuple[Any, ...]:
    """Return a sequence unchanged when its items are unique.

    :param values: parsed sequence.
    :param where: dotted section path used in the error message.
    :raises ConfigError: the sequence repeats a value.
    """
    if len(set(values)) != len(values):
        raise ConfigError(f"{where} must not repeat values, got {list(values)}")
    return values


def _strictly_increasing(values: tuple[int, ...], where: str) -> tuple[int, ...]:
    """Return a strictly increasing integer sequence.

    :param values: parsed sequence.
    :param where: dotted section path used in the error message.
    :raises ConfigError: the sequence is not strictly increasing.
    """
    if any(later <= earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ConfigError(f"{where} must be strictly increasing, got {list(values)}")
    return values


def _strictly_decreasing(values: tuple[int, ...], where: str) -> tuple[int, ...]:
    """Return a strictly decreasing integer sequence.

    :param values: parsed sequence.
    :param where: dotted section path used in the error message.
    :raises ConfigError: the sequence is not strictly decreasing.
    """
    if any(later >= earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ConfigError(f"{where} must be strictly decreasing, got {list(values)}")
    return values


def _optional_positive_int(value: Any, where: str) -> int | None:
    """Parse an optional positive integer cap."""
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ConfigError(f"{where} must be a positive integer or null")
    return parsed


def _study_section(raw: dict[str, Any]) -> tuple[str, str]:
    """Parse and validate the study kind and phase.

    :param raw: top-level configuration mapping.
    :returns: ``(kind, phase)``.
    :raises ConfigError: the section is missing or names an unknown study or phase.
    """
    study = _section(raw, "study")
    kind = str(study.get("kind", ""))
    if kind != STUDY_KIND:
        raise ConfigError(f"study.kind must be {STUDY_KIND!r}, got {kind!r}")
    phase = str(study.get("phase", ""))
    if phase not in STUDY_PHASES:
        raise ConfigError(f"study.phase must be one of {STUDY_PHASES}, got {phase!r}")
    return kind, phase


def _load_split(raw: dict[str, Any], target: str) -> SplitSettings:
    """Parse, validate, and resolve the split section for one target rotation.

    :param raw: top-level configuration mapping.
    :param target: resolved target source.
    :returns: validated split settings.
    :raises ConfigError: the test mode is unknown or invalid for the target, or a fraction,
        threshold, or seed is out of range.
    """
    section = _section(raw, "split")
    if target == "expansionrx":
        mode = str(
            section.get("expansionrx_test_mode", section.get("test_mode", "temporal_halves"))
        )
        allowed = EXPANSIONRX_TEST_MODES
    else:
        mode = str(section.get("test_mode", section.get(f"{target}_test_mode", "butina_halves")))
        allowed = BUTINA_TEST_MODES
    if mode not in allowed:
        raise ConfigError(
            f"split test mode for target {target!r} must be one of {allowed}, got {mode!r}"
        )

    selection_fraction = float(section.get("target_test_selection_fraction", 0.50))
    if not 0.0 < selection_fraction < 1.0:
        raise ConfigError("split.target_test_selection_fraction must lie strictly between 0 and 1")
    biogen_fraction = float(section.get("biogen_target_train_fraction", 0.80))
    if not 0.0 < biogen_fraction < 1.0:
        raise ConfigError("split.biogen_target_train_fraction must lie strictly between 0 and 1")
    threshold = float(section.get("butina_distance_threshold", 0.40))
    if not 0.0 < threshold <= 1.0:
        raise ConfigError("split.butina_distance_threshold must lie in (0, 1]")
    seed = int(section.get("butina_seed", 0))
    if seed < 0:
        raise ConfigError("split.butina_seed must be non-negative")
    return SplitSettings(
        test_mode=mode,
        target_test_selection_fraction=selection_fraction,
        biogen_target_train_fraction=biogen_fraction,
        butina_distance_threshold=threshold,
        butina_seed=seed,
    )


def _load_model(raw: dict[str, Any]) -> ModelSettings:
    """Parse and validate the model section."""
    section = _section(raw, "model")
    model = ModelSettings(
        radius=int(section.get("radius", 2)),
        n_bits=int(section.get("n_bits", 2048)),
        chirality=bool(section.get("chirality", False)),
        n_estimators=int(section.get("n_estimators", 500)),
        learning_rate=float(section.get("learning_rate", 0.05)),
        max_depth=int(section.get("max_depth", 6)),
        min_child_weight=float(section.get("min_child_weight", 1.0)),
        subsample=float(section.get("subsample", 0.8)),
        colsample_bytree=float(section.get("colsample_bytree", 0.8)),
        reg_lambda=float(section.get("reg_lambda", 1.0)),
        tree_method=str(section.get("tree_method", "hist")),
        n_jobs=int(section.get("n_jobs", -1)),
    )
    if model.radius <= 0 or model.n_bits <= 0 or model.n_estimators <= 0:
        raise ConfigError("model.radius, model.n_bits, and model.n_estimators must be positive")
    if not 0.0 < model.learning_rate <= 1.0:
        raise ConfigError("model.learning_rate must lie in (0, 1]")
    if model.max_depth <= 0:
        raise ConfigError("model.max_depth must be positive")
    if not 0.0 <= model.subsample <= 1.0 or not 0.0 <= model.colsample_bytree <= 1.0:
        raise ConfigError("model.subsample and model.colsample_bytree must lie in [0, 1]")
    if model.reg_lambda < 0.0:
        raise ConfigError("model.reg_lambda must be non-negative")
    return model


def _load_boostin(raw: dict[str, Any]) -> BoostInSettings:
    """Parse and validate the BoostIn study section."""
    section = _section(raw, "boostin")
    if not section:
        raise ConfigError("study.kind: boostin_pruning requires a boostin configuration section")
    defaults = BoostInSettings()
    settings = BoostInSettings(
        score_seeds=_unique_tuple(
            _seed_tuple(
                section.get("score_seeds", list(defaults.score_seeds)), "boostin.score_seeds"
            ),
            "boostin.score_seeds",
        ),
        outcome_seeds=_unique_tuple(
            _seed_tuple(
                section.get("outcome_seeds", list(defaults.outcome_seeds)),
                "boostin.outcome_seeds",
            ),
            "boostin.outcome_seeds",
        ),
        batch_sizes=_unique_tuple(
            _strictly_increasing(
                _int_tuple(
                    section.get("batch_sizes", list(defaults.batch_sizes)), "boostin.batch_sizes"
                ),
                "boostin.batch_sizes",
            ),
            "boostin.batch_sizes",
        ),
        random_draws=int(section.get("random_draws", defaults.random_draws)),
        matching_draws=int(section.get("matching_draws", defaults.matching_draws)),
        label_match_bins=_unique_tuple(
            _strictly_decreasing(
                _int_tuple(
                    section.get("label_match_bins", list(defaults.label_match_bins)),
                    "boostin.label_match_bins",
                ),
                "boostin.label_match_bins",
            ),
            "boostin.label_match_bins",
        ),
    )
    if settings.random_draws <= 0:
        raise ConfigError("boostin.random_draws must be positive")
    if settings.matching_draws <= 0:
        raise ConfigError("boostin.matching_draws must be positive")
    if len(settings.label_match_bins) < 2:
        raise ConfigError(
            "boostin.label_match_bins must declare at least two coarse-grained fallbacks"
        )
    return settings


def _load_bootstrap(raw: dict[str, Any]) -> BootstrapSettings:
    """Parse and validate the bootstrap section."""
    section = _section(raw, "bootstrap")
    settings = BootstrapSettings(
        n_resamples=int(section.get("n_resamples", 2000)),
        cluster_distance_threshold=float(section.get("cluster_distance_threshold", 0.40)),
        seed=int(section.get("seed", 0)),
    )
    if settings.n_resamples <= 0:
        raise ConfigError("bootstrap.n_resamples must be positive")
    if not 0.0 < settings.cluster_distance_threshold <= 1.0:
        raise ConfigError("bootstrap.cluster_distance_threshold must lie in (0, 1]")
    if settings.seed < 0:
        raise ConfigError("bootstrap.seed must be non-negative")
    return settings


def _load_paths(raw: dict[str, Any], paths_file_override: str | Path | None) -> PathSet:
    """Resolve dataset paths from `paths_file` and inline overrides, then verify them."""
    merged: dict[str, Any] = {}
    paths_file = paths_file_override if paths_file_override is not None else raw.get("paths_file")
    if paths_file is not None:
        candidate = Path(str(paths_file))
        if not candidate.is_file():
            raise ConfigError(f"paths_file not found: {candidate} (run from the repository root)")
        merged.update(_mapping(yaml.safe_load(candidate.read_text()), str(candidate)))
    merged.update(_section(raw, "paths"))

    missing = [key for key in DATASET_KEYS if not merged.get(key)]
    if missing:
        raise ConfigError(f"missing dataset paths for {missing}; set them in paths_file or paths")

    resolved = {key: Path(str(merged[key])).expanduser() for key in DATASET_KEYS}
    absent = [str(path) for path in resolved.values() if not path.is_file()]
    if absent:
        raise ConfigError(f"dataset files not found: {absent}")
    return PathSet(**resolved)


def load_config(
    path: str | Path,
    *,
    paths_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load, resolve, and validate a run configuration.

    :param path: YAML configuration file.
    :param paths_file: optional override for the configuration's ``paths_file``; used by tests
        to point committed configurations at synthetic datasets.
    :param overrides: optional top-level mapping merged over the file contents before
        validation; used by tests to alter committed configurations without editing them.
    :returns: validated configuration with resolved dataset paths.
    :raises ConfigError: file missing, malformed, or containing invalid values.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    raw = _mapping(yaml.safe_load(config_path.read_text()), str(config_path))
    if overrides:
        raw.update(dict(overrides))

    _, phase = _study_section(raw)

    endpoint = str(raw.get("endpoint", ""))
    if endpoint not in ENDPOINTS:
        raise ConfigError(f"endpoint must be one of {ENDPOINTS}, got {endpoint!r}")

    target = str(raw.get("target", ""))
    if target not in TARGETS:
        raise ConfigError(f"target must be one of {TARGETS}, got {target!r}")

    donor_sources = _unique_tuple(
        _string_tuple(raw.get("donor_sources", []), "donor_sources"), "donor_sources"
    )
    unknown_donors = [source for source in donor_sources if source not in SOURCES]
    if unknown_donors:
        raise ConfigError(f"donor_sources must name known datasets, got {unknown_donors}")
    if target in donor_sources:
        raise ConfigError(
            f"target {target!r} must not appear in donor_sources {list(donor_sources)}"
        )

    overrides_raw = _mapping(raw.get("contract_overrides"), "contract_overrides")
    contract_overrides = {
        str(key): dict(_mapping(value, f"contract_overrides.{key}"))
        for key, value in overrides_raw.items()
    }
    for source in (target, *donor_sources):
        resolve_contract(endpoint, source, contract_overrides)

    limits_raw = _section(raw, "limits")
    limits = Limits(
        max_target_train=_optional_positive_int(
            limits_raw.get("max_target_train"), "limits.max_target_train"
        ),
        max_target_selection=_optional_positive_int(
            limits_raw.get("max_target_selection"), "limits.max_target_selection"
        ),
        max_donor_rows=_optional_positive_int(
            limits_raw.get("max_donor_rows"), "limits.max_donor_rows"
        ),
    )

    return RunConfig(
        phase=phase,
        endpoint=endpoint,
        target=target,
        donor_sources=donor_sources,
        paths=_load_paths(raw, paths_file),
        results_root=Path(str(raw.get("results_root", "results"))),
        split=_load_split(raw, target),
        model=_load_model(raw),
        boostin=_load_boostin(raw),
        bootstrap=_load_bootstrap(raw),
        limits=limits,
        contract_overrides=contract_overrides,
        dry_run=bool(raw.get("dry_run", False)),
        source_path=config_path,
    )
