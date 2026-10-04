"""RDKit standardisation, Morgan fingerprints, and duplicate aggregation.

Adapted from the preceding towards-global-models pipeline: canonical SMILES, InChIKey,
fingerprints, duplicate aggregation, and replicate provenance are preserved.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

STANDARDISED_COLUMNS = (
    "source",
    "original_id",
    "input_smiles",
    "canonical_smiles",
    "inchikey",
    "mol_weight",
    "parse_ok",
    "parse_error",
    "fingerprint",
)


@dataclass(frozen=True, slots=True)
class FingerprintSettings:
    """Morgan fingerprint parameters shared by every condition.

    :param radius: Morgan radius.
    :param n_bits: fingerprint length in bits.
    :param chirality: whether to include chirality.
    """

    radius: int = 2
    n_bits: int = 2048
    chirality: bool = False


def make_generator(settings: FingerprintSettings) -> rdFingerprintGenerator.FingerprintGenerator64:
    """Build the RDKit Morgan fingerprint generator for the given settings."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=settings.radius, fpSize=settings.n_bits, includeChirality=settings.chirality
    )


def parse_smiles(smiles: str) -> tuple[Chem.Mol | None, str | None]:
    """Parse a SMILES string, allowing CXSMILES extensions.

    :param smiles: input structure string.
    :returns: ``(mol, None)`` on success, ``(None, error)`` otherwise.
    """
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = True
    try:
        mol = Chem.MolFromSmiles(smiles, params)
    except Exception as exc:  # RDKit raises on malformed CXSMILES extensions
        return None, f"parse_exception: {exc}"
    if mol is None:
        return None, "parse_error"
    return mol, None


def standardize_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    smiles_col: str,
    source: str,
    settings: FingerprintSettings | None = None,
) -> pd.DataFrame:
    """Standardise structures and compute fingerprints for one raw source frame.

    Adds the :data:`STANDARDISED_COLUMNS`; invalid structures are retained with
    ``parse_ok=False`` so the endpoint audit can count them.

    :param frame: raw dataset frame.
    :param id_col: column holding the original molecule identifier.
    :param smiles_col: column holding the input structure string.
    :param source: dataset key recorded on every row.
    :param settings: fingerprint settings; defaults to Morgan radius 2, 2048 bits.
    :returns: the frame with standardisation columns appended.
    :raises KeyError: `id_col` or `smiles_col` is missing from `frame`.
    """
    fingerprint_settings = settings or FingerprintSettings()
    generator = make_generator(fingerprint_settings)

    records: list[dict[str, Any]] = []
    for original_id, smiles in zip(
        frame[id_col].astype(str), frame[smiles_col].astype(str), strict=True
    ):
        mol, error = parse_smiles(smiles)
        record: dict[str, Any] = {
            "source": source,
            "original_id": original_id,
            "input_smiles": smiles,
            "canonical_smiles": None,
            "inchikey": None,
            "mol_weight": np.nan,
            "parse_ok": False,
            "parse_error": error,
            "fingerprint": None,
        }
        if mol is not None:
            try:
                inchikey = Chem.MolToInchiKey(mol)
            except Exception:  # InChI generation can fail on exotic valences
                inchikey = ""
            if inchikey:
                record.update(
                    canonical_smiles=Chem.MolToSmiles(mol),
                    inchikey=inchikey,
                    mol_weight=float(Descriptors.MolWt(mol)),
                    parse_ok=True,
                    parse_error=None,
                    fingerprint=np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.uint8),
                )
            else:
                record["parse_error"] = "inchikey_failed"
        records.append(record)

    meta = pd.DataFrame(records, index=frame.index)
    return pd.concat([frame, meta], axis=1)


def deduplicate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate structures (same InChIKey) after endpoint transformation.

    The retained target is the median across replicates; the replicate count and spread
    are recorded on the retained row.

    :param frame: endpoint-applied eligible rows with ``model_target`` present.
    :returns: one row per InChIKey.
    """
    rows: list[dict[str, Any]] = []
    for inchikey, group in frame.groupby("inchikey", sort=False):
        values = group["model_target"].to_numpy(dtype=float)
        first = group.iloc[0]
        raw_values = ";".join(f"{value:g}" for value in group["raw_value"].to_numpy(dtype=float))
        rows.append(
            {
                "inchikey": inchikey,
                "canonical_smiles": first["canonical_smiles"],
                "mol_weight": first["mol_weight"],
                "fingerprint": first["fingerprint"],
                "model_target": float(np.median(values)),
                "replicate_count": int(values.size),
                "replicate_spread": float(values.max() - values.min()) if values.size > 1 else 0.0,
                "original_id": ";".join(group["original_id"].astype(str)),
                "raw_value": float(first["raw_value"]),
                "raw_values": raw_values,
                "source": first["source"],
            }
        )
    return pd.DataFrame(rows)
