"""Project-specific exceptions."""


class VLARos2Error(Exception):
    """Base exception for vla_ros2 failures."""


class UnknownModelError(VLARos2Error, KeyError):
    """Raised when a model name is not registered."""


class MissingDependencyError(VLARos2Error, ImportError):
    """Raised when an optional adapter dependency is not installed."""


class AdapterError(VLARos2Error):
    """Raised when an adapter fails at runtime."""


class ConfigurationError(VLARos2Error, ValueError):
    """Raised when runtime configuration is invalid."""


class RemoteRuntimeError(VLARos2Error):
    """Raised when a remote inference call fails."""
