"""Project-specific exceptions."""


class VLAZooError(Exception):
    """Base exception for vla_zoo failures."""


class UnknownModelError(VLAZooError, KeyError):
    """Raised when a model name is not registered."""


class MissingDependencyError(VLAZooError, ImportError):
    """Raised when an optional adapter dependency is not installed."""


class AdapterError(VLAZooError):
    """Raised when an adapter fails at runtime."""


class ConfigurationError(VLAZooError, ValueError):
    """Raised when runtime configuration is invalid."""


class RemoteRuntimeError(VLAZooError):
    """Raised when a remote inference call fails."""
