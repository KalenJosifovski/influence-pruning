"""Domain exceptions for the influence-pruning pipeline."""


class InfluencePruningError(Exception):
    """Base class for project domain errors."""


class ConfigError(InfluencePruningError):
    """Raised when a run configuration is invalid or incomplete."""


class EndpointContractError(InfluencePruningError):
    """Raised when an endpoint contract cannot be applied or does not exist."""


class ContractUnresolvedError(EndpointContractError):
    """Raised when an endpoint/source unit convention has not been agreed."""


class ChronologyAuditError(InfluencePruningError):
    """Raised when temporal identifier ordering cannot be verified."""


class FrameGuardError(InfluencePruningError):
    """Raised when code attempts to read an evaluation surface it may not see."""


class DatasetError(InfluencePruningError):
    """Raised when a raw dataset is missing expected columns or content."""


class RunError(InfluencePruningError):
    """Raised when a study run violates its declared contract."""


class InfluenceDependencyError(InfluencePruningError):
    """Raised when the isolated tree-influence environment is unavailable or incompatible."""


class ArmPlanningError(InfluencePruningError):
    """Raised when a deletion arm cannot be constructed under its contract."""


class ArtifactError(InfluencePruningError):
    """Raised when a run directory is incomplete or fails integrity validation."""
