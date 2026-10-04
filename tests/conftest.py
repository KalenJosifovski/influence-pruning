"""Fixtures creating synthetic datasets and configurations for pipeline tests.

Every test uses synthetic frames and temporary directories only. No test scans a real dataset
catalogue, reads a real ``paths.local.yaml``, or touches a pre-existing ``results/`` directory.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pytest import fixture

from influence_pruning.config import RunConfig, load_config

POLARIS_HLM_OVERRIDE: dict[str, dict[str, Any]] = {
    "hlm_clint:polaris": {
        "units_in": "µL/min/mg microsomal protein (synthetic test override)",
        "units_out": "log10 mL/min/kg",
        "transform_name": "log10_positive",
        "positive_only": True,
        "scale_factor": 0.9,
        "minimum_quantification_limit": 10.0,
        "evidence": "Synthetic test override mirroring the verified Bienta IVIVE conversion.",
    }
}

DEFAULT_CONFIG: dict[str, Any] = {
    "study": {"kind": "boostin_pruning", "phase": "exploratory"},
    "endpoint": "hlm_clint",
    "target": "expansionrx",
    "donor_sources": ["biogen", "polaris"],
    "split": {
        "expansionrx_test_mode": "temporal_halves",
        "target_test_selection_fraction": 0.50,
        "butina_distance_threshold": 0.40,
        "butina_seed": 0,
    },
    "model": {"n_estimators": 10, "n_jobs": 1},
    "boostin": {
        "score_seeds": [0],
        "outcome_seeds": [0],
        "batch_sizes": [3, 5],
        "random_draws": 2,
        "matching_draws": 2,
        "label_match_bins": [10, 5, 3],
    },
    "bootstrap": {"n_resamples": 20, "seed": 0},
    "contract_overrides": POLARIS_HLM_OVERRIDE,
    "limits": {
        "max_target_train": None,
        "max_target_selection": None,
        "max_donor_rows": None,
    },
    "dry_run": False,
}


SCAFFOLDS = (
    "c1ccccc1",
    "c1ccncc1",
    "c1cccnc1",
    "c1cc[nH]c1",
    "c1ccsc1",
    "c1ccoc1",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOCC1",
    "c1ccc2ccccc2c1",
)
# Two disjoint substituent alphabets: the prefix and suffix always come from different sets,
# so swapping prefix and suffix can never encode the same molecule twice. Every substituent
# must be attachable to the scaffold at both ends (prefix and suffix).
PREFIX_SUBSTITUENTS = (
    "F",
    "Cl",
    "Br",
    "I",
    "C",
    "O",
    "N",
    "C=C",
    "C#C",
    "OC",
    "NC",
    "SC",
    "CO",
    "CN",
    "C(C)C",
    "CCO",
)
SUFFIX_SUBSTITUENTS = (
    "C(F)(F)C",
    "CCC",
    "CC(C)C",
    "C(C)(C)C",
    "CCOC",
    "CCN",
    "CCS",
    "CC(F)C",
    "CC#CC",
    "COCC",
    "CNC",
    "CSC",
    "CC(C)O",
    "CC(C)(C)O",
    "CC(F)(F)C",
    "C(N)C",
)
EXPANSIONRX_BASE = 1
BIOGEN_BASE = 1000
POLARIS_BASE = 2000


def synthetic_smiles(index: int) -> str:
    """Return a valid, structurally unique SMILES string for a synthetic index.

    The index is encoded in mixed radix (scaffold, prefix substituent, suffix substituent) with
    period 2560, so every molecule in the fixture ranges is structurally unique while related
    compounds still cluster under Butina. Dataset fixtures use disjoint index bases so
    cross-dataset structures are unique unless deliberately reused.
    """
    scaffold = SCAFFOLDS[index % len(SCAFFOLDS)]
    first = PREFIX_SUBSTITUENTS[(index // len(SCAFFOLDS)) % len(PREFIX_SUBSTITUENTS)]
    second = SUFFIX_SUBSTITUENTS[(index // 160) % len(SUFFIX_SUBSTITUENTS)]
    return f"{first}{scaffold}{second}"


def _expansionrx_train() -> pd.DataFrame:
    """Synthetic ExpansionRx train-era frame with deliberate edge cases."""
    indices = [EXPANSIONRX_BASE + offset for offset in range(150)]
    smiles = [synthetic_smiles(index) for index in indices]
    hlm: list[float] = [10.0 + index for index in indices]
    hlm[4] = 0.0
    hlm[5] = float("nan")
    smiles[6] = "not_a_smiles"
    return pd.DataFrame(
        {
            "Molecule Name": [f"E-{index:07d}" for index in indices],
            "SMILES": smiles,
            "LogD": [index * 0.02 for index in indices],
            "KSOL": [float(200 - index) for index in indices],
            "HLM CLint": hlm,
        }
    )


def _expansionrx_test() -> pd.DataFrame:
    """Synthetic ExpansionRx test frame with IDs contiguous after the train era."""
    indices = [EXPANSIONRX_BASE + offset for offset in range(150, 210)]
    return pd.DataFrame(
        {
            "Molecule Name": [f"E-{index:07d}" for index in indices],
            "SMILES": [synthetic_smiles(index) for index in indices],
            "LogD": [index * 0.02 for index in indices],
            "KSOL": [float(300 + index) for index in indices],
            "HLM CLint": [10.0 + index for index in indices],
        }
    )


def _biogen() -> pd.DataFrame:
    """Synthetic Biogen donor frame; structures deliberately overlap the target.

    Two structures overlap the target chemistry and one structure appears twice as a
    replicate pair, exercising overlap exclusion and duplicate aggregation.
    """
    indices = [BIOGEN_BASE + offset for offset in range(120)]
    smiles = [synthetic_smiles(index) for index in indices]
    # Two structures deliberately overlap the target train era; one appears twice.
    smiles[0] = synthetic_smiles(EXPANSIONRX_BASE + 9)
    smiles[1] = synthetic_smiles(EXPANSIONRX_BASE + 10)
    smiles[2] = smiles[3]
    return pd.DataFrame(
        {
            "Internal ID": [f"Mol{index}" for index in range(1, 121)],
            "Vendor ID": [f"V{index}" for index in range(1, 121)],
            "SMILES": smiles,
            "CollectionName": "synthetic",
            "LOG HLM_CLint (mL/min/kg)": [1.0 + 0.01 * index for index in range(120)],
            "LOG SOLUBILITY PH 6.8 (ug/mL)": [1.2] * 120,
        }
    )


def _polaris() -> pd.DataFrame:
    """Synthetic Polaris donor frame with the real column layout and official Set split."""
    indices = [POLARIS_BASE + offset for offset in range(100)]
    hlm = [50.0 + offset for offset in range(100)]
    hlm[1] = float("nan")
    hlm[2] = 5.0  # below the synthetic quantification limit of 10
    return pd.DataFrame(
        {
            "CXSMILES": [synthetic_smiles(index) for index in indices],
            "HLM": hlm,
            "KSOL": [300.0] * 100,
            "LogD": [2.0] * 100,
            "Molecule Name": [f"ASAP-{index:07d}" for index in indices],
            "Set": ["Train"] * 60 + ["Test"] * 40,
        }
    )


def write_synthetic_datasets(directory: Path) -> Path:
    """Write the synthetic dataset CSVs into `directory` and return the paths file."""
    data_dir = directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _expansionrx_train().to_csv(data_dir / "expansion_data_train.csv", index=False)
    _expansionrx_test().to_csv(data_dir / "expansion_data_test.csv", index=False)
    _biogen().to_csv(data_dir / "ADME_public_set_3521.csv", index=False)
    _polaris().to_csv(data_dir / "ADMET.csv", index=False)
    paths_file = directory / "paths.yaml"
    paths_file.write_text(
        yaml.safe_dump(
            {
                "expansionrx_train": str(data_dir / "expansion_data_train.csv"),
                "expansionrx_test": str(data_dir / "expansion_data_test.csv"),
                "biogen": str(data_dir / "ADME_public_set_3521.csv"),
                "polaris": str(data_dir / "ADMET.csv"),
            }
        )
    )
    return paths_file


def write_config(
    config_path: Path,
    paths_file: Path,
    results_root: Path,
    **overrides: Any,
) -> Path:
    """Write one synthetic study configuration, applying top-level overrides."""
    config = {key: value for key, value in DEFAULT_CONFIG.items()}
    config.update(
        {
            "paths_file": str(paths_file),
            "results_root": str(results_root),
        }
    )
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = {**config[key], **value}
            config[key] = merged
        else:
            config[key] = value
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@fixture()
def synthetic_paths(tmp_path: Path) -> Path:
    """Write the synthetic dataset CSVs and return the paths file."""
    return write_synthetic_datasets(tmp_path)


@fixture()
def expansionrx_config(tmp_path: Path, synthetic_paths: Path) -> RunConfig:
    """A small ExpansionRx-target configuration over the synthetic datasets."""
    config_path = write_config(tmp_path / "expansionrx.yaml", synthetic_paths, tmp_path / "results")
    return load_config(config_path)


@fixture()
def biogen_config(tmp_path: Path, synthetic_paths: Path) -> RunConfig:
    """A small Biogen-target rotation configuration over the synthetic datasets."""
    config_path = write_config(
        tmp_path / "biogen.yaml",
        synthetic_paths,
        tmp_path / "results-biogen",
        target="biogen",
        donor_sources=["expansionrx", "polaris"],
        split={"test_mode": "butina_halves"},
    )
    return load_config(config_path)


@fixture()
def polaris_config(tmp_path: Path, synthetic_paths: Path) -> RunConfig:
    """A small Polaris-target rotation configuration over the synthetic datasets."""
    config_path = write_config(
        tmp_path / "polaris.yaml",
        synthetic_paths,
        tmp_path / "results-polaris",
        target="polaris",
        donor_sources=["expansionrx", "biogen"],
        split={"test_mode": "butina_halves"},
    )
    return load_config(config_path)


@fixture()
def dry_run_config(tmp_path: Path, synthetic_paths: Path) -> RunConfig:
    """A capped dry-run configuration over the synthetic datasets."""
    config_path = write_config(
        tmp_path / "dryrun.yaml",
        synthetic_paths,
        tmp_path / "results-dryrun",
        dry_run=True,
        limits={
            "max_target_train": 100,
            "max_target_selection": 40,
            "max_donor_rows": 60,
        },
    )
    return load_config(config_path)


@fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]
