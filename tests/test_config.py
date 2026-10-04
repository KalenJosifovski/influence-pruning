"""Configuration, contract, and capacity tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import DEFAULT_CONFIG, POLARIS_HLM_OVERRIDE, write_config

from influence_pruning.config import BoostInSettings, load_config
from influence_pruning.contracts import (
    CONTRACTS,
    apply_contract,
    lookup_contract,
    resolve_contract,
)
from influence_pruning.errors import (
    ConfigError,
    ContractUnresolvedError,
    EndpointContractError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CONFIGS = sorted((REPO_ROOT / "configs").glob("boostin_*.yaml"))
ROTATION_CONFIGS = sorted((REPO_ROOT / "configs").glob("boostin_*_hlm.yaml"))


def test_committed_configs_parse(synthetic_paths: Path) -> None:
    """Every committed configuration parses against synthetic paths."""
    assert COMMITTED_CONFIGS, "expected committed configurations"
    for config_path in COMMITTED_CONFIGS:
        config = load_config(config_path, paths_file=synthetic_paths)
        assert config.study_kind == "boostin_pruning"
        assert config.phase == "exploratory"


def test_frozen_model_and_boostin_values_identical(synthetic_paths: Path) -> None:
    """The frozen Morgan/XGBoost and BoostIn values agree across HLM rotation configs."""
    configs = [load_config(path, paths_file=synthetic_paths) for path in ROTATION_CONFIGS]
    assert len(configs) == 3
    for config in configs:
        assert config.model.radius == 2
        assert config.model.n_bits == 2048
        assert config.model.chirality is False
        assert config.model.n_estimators == 500
        assert config.model.learning_rate == pytest.approx(0.05)
        assert config.model.max_depth == 6
        assert config.model.min_child_weight == pytest.approx(1.0)
        assert config.model.subsample == pytest.approx(0.8)
        assert config.model.colsample_bytree == pytest.approx(0.8)
        assert config.model.reg_lambda == pytest.approx(1.0)
        assert config.model.tree_method == "hist"
        assert config.boostin.score_seeds == (0, 1, 2, 3, 4)
        assert config.boostin.outcome_seeds == (0, 1, 2, 3, 4)
        assert config.boostin.random_draws == 10
        assert config.boostin.matching_draws == 10
        assert config.boostin.label_match_bins == (10, 5, 3)
        assert config.bootstrap.n_resamples == 2000
    main = load_config(ROTATION_CONFIGS[0], paths_file=synthetic_paths)
    dry = load_config(
        REPO_ROOT / "configs" / "dryrun_boostin_expansionrx_hlm.yaml", paths_file=synthetic_paths
    )
    assert dry.model == main.model


def test_target_rotation_swaps_target_and_donors(synthetic_paths: Path) -> None:
    """Rotating the target rewrites the target source and donor set."""
    biogen = load_config(
        REPO_ROOT / "configs" / "boostin_biogen_hlm.yaml", paths_file=synthetic_paths
    )
    polaris = load_config(
        REPO_ROOT / "configs" / "boostin_polaris_hlm.yaml", paths_file=synthetic_paths
    )
    assert biogen.target == "biogen"
    assert biogen.donor_sources == ("expansionrx", "polaris")
    assert biogen.split.test_mode == "butina_halves"
    assert polaris.target == "polaris"
    assert polaris.donor_sources == ("expansionrx", "biogen")
    assert polaris.split.test_mode == "butina_halves"


def test_polaris_hlm_contract_fails_closed_then_resolves() -> None:
    """The unverified Polaris HLM convention fails closed until the IVIVE override lands."""
    template = lookup_contract("hlm_clint", "polaris")
    assert template is not None
    assert not template.resolved
    with pytest.raises(ContractUnresolvedError):
        resolve_contract("hlm_clint", "polaris")
    resolved = resolve_contract("hlm_clint", "polaris", POLARIS_HLM_OVERRIDE)
    assert resolved.resolved
    assert resolved.status == "declared-by-config"
    assert resolved.scale_factor == pytest.approx(0.9)

    frame = pd.DataFrame(
        {
            "HLM": [100.0, 5.0, np.nan],
            "canonical_smiles": ["C", "CC", "CCC"],
            "fingerprint": [np.zeros(4, dtype=np.uint8)] * 3,
            "mol_weight": [10.0, 20.0, 30.0],
        }
    )
    applied = apply_contract(resolved, frame)
    assert applied.loc[0, "model_target"] == pytest.approx(np.log10(100.0 * 0.9))
    assert not bool(applied.loc[1, "included"])
    assert applied.loc[1, "exclusion_reason"] == "below_quantification_limit"
    assert not bool(applied.loc[2, "included"])
    assert applied.loc[2, "exclusion_reason"] == "missing_raw_value"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target": "biogen", "donor_sources": ["biogen", "polaris"]}, "must not appear"),
        ({"donor_sources": ["biogen", "unknown"]}, "known datasets"),
        ({"target": "nuclear_receptor_hts"}, "target must be one of"),
        ({"endpoint": "ksol"}, "endpoint must be one of"),
        ({"study": {"kind": "other", "phase": "exploratory"}}, "study.kind"),
        ({"study": {"kind": "boostin_pruning", "phase": "final"}}, "study.phase"),
    ],
)
def test_invalid_identity_fields_rejected(
    tmp_path: Path, synthetic_paths: Path, overrides: dict, message: str
) -> None:
    """Unknown or contradictory identity fields are rejected."""
    config_path = write_config(tmp_path / "config.yaml", synthetic_paths, tmp_path / "results")
    with pytest.raises(ConfigError, match=message):
        load_config(config_path, overrides=overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"boostin": {**DEFAULT_CONFIG["boostin"], "score_seeds": [0, 0]}},
            "must not repeat",
        ),
        (
            {"boostin": {**DEFAULT_CONFIG["boostin"], "outcome_seeds": []}},
            "non-empty",
        ),
        (
            {"boostin": {**DEFAULT_CONFIG["boostin"], "batch_sizes": [10, 5]}},
            "strictly increasing",
        ),
        (
            {"boostin": {**DEFAULT_CONFIG["boostin"], "batch_sizes": [0, 5]}},
            "positive integers",
        ),
        (
            {"boostin": {**DEFAULT_CONFIG["boostin"], "label_match_bins": [3, 5, 10]}},
            "strictly decreasing",
        ),
        (
            {"boostin": {**DEFAULT_CONFIG["boostin"], "random_draws": 0}},
            "random_draws must be positive",
        ),
        (
            {"bootstrap": {"n_resamples": 0, "cluster_distance_threshold": 0.4, "seed": 0}},
            "n_resamples must be positive",
        ),
        (
            {"split": {**DEFAULT_CONFIG["split"], "target_test_selection_fraction": 1.5}},
            "strictly between 0 and 1",
        ),
    ],
)
def test_invalid_boostin_and_split_values_rejected(
    tmp_path: Path, synthetic_paths: Path, overrides: dict, message: str
) -> None:
    """Malformed seeds, batch sizes, bins, draw counts, and fractions are rejected."""
    config_path = write_config(tmp_path / "config.yaml", synthetic_paths, tmp_path / "results")
    with pytest.raises(ConfigError, match=message):
        load_config(config_path, overrides=overrides)


def test_unknown_split_mode_rejected(tmp_path: Path, synthetic_paths: Path) -> None:
    """An unknown test mode is rejected for the configured target."""
    config_path = write_config(tmp_path / "config.yaml", synthetic_paths, tmp_path / "results")
    with pytest.raises(ConfigError, match="test mode"):
        load_config(
            config_path,
            overrides={"split": {**DEFAULT_CONFIG["split"], "expansionrx_test_mode": "random"}},
        )
    with pytest.raises(ConfigError, match="test mode"):
        load_config(
            config_path,
            overrides={
                "target": "biogen",
                "donor_sources": ["expansionrx", "polaris"],
                "split": {"test_mode": "temporal_halves"},
            },
        )


def test_missing_documented_override_fails_closed(tmp_path: Path, synthetic_paths: Path) -> None:
    """HLM configurations without a resolved Polaris override are rejected."""
    config_path = write_config(tmp_path / "config.yaml", synthetic_paths, tmp_path / "results")
    with pytest.raises(ContractUnresolvedError):
        load_config(config_path, overrides={"contract_overrides": {}})


def test_nonpositive_scale_factor_override_rejected(tmp_path: Path, synthetic_paths: Path) -> None:
    """A non-positive scale factor is rejected before any data are read."""
    config_path = write_config(tmp_path / "config.yaml", synthetic_paths, tmp_path / "results")
    bad = {
        "hlm_clint:polaris": {
            **POLARIS_HLM_OVERRIDE["hlm_clint:polaris"],
            "scale_factor": 0.0,
        }
    }
    with pytest.raises(EndpointContractError, match="scale factor must be positive"):
        load_config(config_path, overrides={"contract_overrides": bad})


def test_batch_capacity_contract() -> None:
    """Batch sizes above a pool's disjoint capacity are rejected with the pool named."""
    settings = BoostInSettings(batch_sizes=(25, 50))
    settings.require_capacity({"donor_only:biogen": 200})
    with pytest.raises(ConfigError, match="donor_only:biogen"):
        settings.require_capacity({"donor_only:biogen": 60})


def test_hlm_contracts_are_declared_for_every_usable_source() -> None:
    """ExpansionRx and Biogen HLM conventions remain explicitly declared."""
    for source in ("expansionrx", "biogen"):
        contract = CONTRACTS[("hlm_clint", source)]
        assert contract.resolved
        assert contract.units_out == "log10 mL/min/kg"
