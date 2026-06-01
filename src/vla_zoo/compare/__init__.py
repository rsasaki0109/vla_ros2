"""Comparison helpers for adapters, methods, and runtime reports."""

from vla_zoo.compare.compatibility import (
    AdapterCompatibility,
    CompatibilityIssue,
    RobotProfile,
    check_adapter_compatibility,
    compatibility_matrix,
    format_compatibility_markdown,
    get_robot_profile,
)
from vla_zoo.compare.profiles import (
    MethodProfile,
    format_method_profiles_markdown,
    method_profiles,
)
from vla_zoo.compare.suite import SuiteArtifact, format_suite_readme

__all__ = [
    "AdapterCompatibility",
    "CompatibilityIssue",
    "MethodProfile",
    "RobotProfile",
    "SuiteArtifact",
    "check_adapter_compatibility",
    "compatibility_matrix",
    "format_compatibility_markdown",
    "format_method_profiles_markdown",
    "format_suite_readme",
    "get_robot_profile",
    "method_profiles",
]
