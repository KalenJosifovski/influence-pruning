"""Endpoint contracts: unit conventions, value transforms, and fail-closed resolution.

Copied and trimmed from the preceding towards-global-models pipeline. Only the HLM CLint
contracts used by the Phase 1 episode are registered here; other endpoints must not be added
without a documented audit of their units.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd

from influence_pruning.errors import ContractUnresolvedError, EndpointContractError

Transform = Callable[[pd.Series, pd.Series | None], pd.Series]

TARGET_SCALES: dict[str, str] = {
    "hlm_clint": "log10 CLint [mL/min/kg]",
}

EXCLUSION_MISSING = "missing_raw_value"
EXCLUSION_NONPOSITIVE = "nonpositive_raw_value"
EXCLUSION_BELOW_QUANTIFICATION_LIMIT = "below_quantification_limit"
EXCLUSION_MW = "invalid_molecular_weight"
EXCLUSION_TRANSFORM = "transform_failed"


def _log10_positive(values: pd.Series, mol_weight: pd.Series | None = None) -> pd.Series:
    """Log10-transform strictly positive values; missing or non-positive values stay NaN."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    positive = values > 0.0
    result.loc[positive] = np.log10(values.loc[positive].astype(float))
    return result


def _identity(values: pd.Series, mol_weight: pd.Series | None = None) -> pd.Series:
    """Return values unchanged (already on the model scale)."""
    return values.astype(float)


TRANSFORMS: dict[str, Transform] = {
    "log10_positive": _log10_positive,
    "identity": _identity,
}


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Unit convention and value transform for one (endpoint, source) pair.

    :param endpoint: endpoint key, e.g. ``"hlm_clint"``.
    :param source: dataset key, e.g. ``"biogen"``.
    :param column: raw dataset column holding the endpoint.
    :param units_in: convention of the raw column as declared by the source.
    :param units_out: model-target convention after transformation.
    :param transform_name: key into :data:`TRANSFORMS`.
    :param status: ``"declared"``, ``"declared-by-config"``, or ``"unresolved"``.
    :param evidence: where the convention is documented, or why it is unresolved.
    :param positive_only: whether non-positive raw values are excluded.
    :param requires_molecular_weight: whether the transform needs ``mol_weight``.
    :param scale_factor: multiplicative factor applied to the raw value before the transform,
        used to harmonise a source's in-vitro unit onto the shared donor convention. Must be
        positive and finite.
    :param minimum_quantification_limit: raw-value lower reporting limit. Values strictly below
        it are excluded rather than treated as exact measurements.
    """

    endpoint: str
    source: str
    column: str
    units_in: str
    units_out: str
    transform_name: str
    status: str
    evidence: str
    positive_only: bool = False
    requires_molecular_weight: bool = False
    scale_factor: float = 1.0
    minimum_quantification_limit: float | None = None

    @property
    def resolved(self) -> bool:
        """Whether the contract may be applied to raw values."""
        return self.status != "unresolved"


CONTRACTS: dict[tuple[str, str], SourceContract] = {
    ("hlm_clint", "expansionrx"): SourceContract(
        endpoint="hlm_clint",
        source="expansionrx",
        column="HLM CLint",
        units_in="mL/min/kg (linear)",
        units_out="log10 mL/min/kg",
        transform_name="log10_positive",
        status="declared",
        evidence=(
            "OpenADMET-ExpansionRx README endpoint list: "
            "'Human Liver Microsomal (HLM) Clint: mL/min/kg'."
        ),
        positive_only=True,
    ),
    ("hlm_clint", "biogen"): SourceContract(
        endpoint="hlm_clint",
        source="biogen",
        column="LOG HLM_CLint (mL/min/kg)",
        units_in="log10 mL/min/kg",
        units_out="log10 mL/min/kg",
        transform_name="identity",
        status="declared",
        evidence=(
            "Biogen ADME column header 'LOG HLM_CLint (mL/min/kg)' and README "
            "('the experimental log(properties) for six endpoints')."
        ),
    ),
    ("hlm_clint", "polaris"): SourceContract(
        endpoint="hlm_clint",
        source="polaris",
        column="HLM",
        units_in="unknown",
        units_out="log10 mL/min/kg (tentative)",
        transform_name="identity",
        status="unresolved",
        evidence=(
            "ASAP-Polaris ADMET.csv column header 'HLM' carries no units. The Bienta "
            "microsomal stability protocol (protocols.io 5qpvokdb9l4o) describes the assay "
            "without a reporting convention, and Bienta materials plus studies citing Bienta "
            "report Clint in µL/min/mg protein — not the ExpansionRx/Biogen log10 mL/min/kg "
            "convention. An explicit IVIVE conversion must be supplied via contract_overrides "
            "before Polaris HLM may be used; no convention may be inferred from value ranges."
        ),
    ),
}


def lookup_contract(
    endpoint: str,
    source: str,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> SourceContract | None:
    """Return the effective contract for an (endpoint, source) pair.

    :param endpoint: endpoint key.
    :param source: dataset key.
    :param overrides: optional mapping keyed ``"<endpoint>:<source>"`` whose values
        replace contract fields (for example after a protocol audit).
    :returns: the contract, or ``None`` when the source does not carry the endpoint.
    :raises EndpointContractError: an override refers to an unknown contract or field, or
        carries a non-positive scale factor or quantification limit.
    """
    template = CONTRACTS.get((endpoint, source))
    override = (overrides or {}).get(f"{endpoint}:{source}")
    if override is None:
        return template
    if template is None:
        raise EndpointContractError(
            f"cannot override {endpoint} on {source}: no contract template exists"
        )
    known = {field.name for field in fields(SourceContract)}
    unknown = set(override) - known
    if unknown:
        raise EndpointContractError(
            f"unknown contract override fields for {endpoint}:{source}: {sorted(unknown)}"
        )
    fields_override = {str(key): value for key, value in override.items()}
    fields_override.setdefault("status", "declared-by-config")
    merged = replace(template, **fields_override)
    if not np.isfinite(merged.scale_factor) or merged.scale_factor <= 0.0:
        raise EndpointContractError(
            f"scale factor must be positive and finite for {endpoint}:{source}: "
            f"{merged.scale_factor!r}"
        )
    if merged.minimum_quantification_limit is not None and (
        not np.isfinite(merged.minimum_quantification_limit)
        or merged.minimum_quantification_limit <= 0.0
    ):
        raise EndpointContractError(
            "minimum quantification limit must be positive and finite for "
            f"{endpoint}:{source}: {merged.minimum_quantification_limit!r}"
        )
    return merged


def resolve_contract(
    endpoint: str,
    source: str,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> SourceContract:
    """Resolve the contract that must be used for modelling, failing closed when unresolved.

    :param endpoint: endpoint key.
    :param source: dataset key.
    :param overrides: optional contract overrides (see :func:`lookup_contract`).
    :returns: a resolved contract.
    :raises EndpointContractError: the source does not carry the endpoint.
    :raises ContractUnresolvedError: the unit convention has not been agreed.
    """
    contract = lookup_contract(endpoint, source, overrides)
    if contract is None:
        raise EndpointContractError(f"{source} does not carry endpoint {endpoint}")
    if not contract.resolved:
        raise ContractUnresolvedError(
            f"unresolved unit convention for {endpoint} on {source}: {contract.evidence}"
        )
    return contract


def apply_contract(contract: SourceContract, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a resolved contract to a standardised frame.

    Adds ``raw_value``, ``model_target``, ``included``, and ``exclusion_reason`` columns.

    :param contract: resolved source contract.
    :param frame: standardised frame with the contract column present.
    :returns: augmented copy of `frame`.
    :raises ContractUnresolvedError: the contract is not resolved.
    :raises EndpointContractError: the required column is missing.
    """
    if not contract.resolved:
        raise ContractUnresolvedError(
            f"unresolved unit convention for {contract.endpoint} on {contract.source}: "
            f"{contract.evidence}"
        )
    if contract.column not in frame.columns:
        raise EndpointContractError(
            f"column {contract.column!r} missing from the {contract.source} frame"
        )

    out = frame.copy()
    raw = pd.to_numeric(out[contract.column], errors="coerce")
    out["raw_value"] = raw
    reasons = pd.Series("", index=out.index, dtype=object)

    if contract.requires_molecular_weight:
        if "mol_weight" not in out.columns:
            raise EndpointContractError(
                f"{contract.endpoint}:{contract.source} requires molecular weights"
            )
        mol_weight = pd.to_numeric(out["mol_weight"], errors="coerce")
        invalid = mol_weight.isna() | (mol_weight <= 0.0)
        reasons.loc[invalid] = EXCLUSION_MW
        mol_weight = mol_weight.where(~invalid)
    else:
        mol_weight = None

    reasons.loc[raw.isna() & (reasons == "")] = EXCLUSION_MISSING
    if contract.positive_only:
        nonpositive = raw.notna() & (raw <= 0.0)
        reasons.loc[nonpositive & (reasons == "")] = EXCLUSION_NONPOSITIVE
    if contract.minimum_quantification_limit is not None:
        below_limit = raw.notna() & (raw < contract.minimum_quantification_limit)
        reasons.loc[below_limit & (reasons == "")] = EXCLUSION_BELOW_QUANTIFICATION_LIMIT

    scaled = raw * contract.scale_factor if contract.scale_factor != 1.0 else raw
    values = TRANSFORMS[contract.transform_name](scaled, mol_weight)
    reasons.loc[values.isna() & (reasons == "")] = EXCLUSION_TRANSFORM

    out["model_target"] = values
    out["included"] = reasons == ""
    out["exclusion_reason"] = reasons
    return out
