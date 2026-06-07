from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from vla_ros2.adapters.smolvla import SmolVLAAdapter
from vla_ros2.core.errors import AdapterError, MissingDependencyError
from vla_ros2.core.model import VLAAdapter
from vla_ros2.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

DEFAULT_PI0_PRETRAINED = "lerobot/pi0_base"

# The pi0 weights file and the asset the processor's tokenizer step pulls from.
PI0_WEIGHTS_FILENAME = "model.safetensors"
PI0_PREPROCESSOR_FILENAME = "policy_preprocessor.json"

PI0_ACTION_SPEC = ActionSpec(
    action_space="custom",
    shape=(32,),
    description=(
        "LeRobot/OpenPI pi0 continuous action. Shape and semantics are checkpoint-specific; "
        "the version-matched lerobot/pi0_base exposes a 32D action, while the older "
        "lerobot/pi0 config schema is rejected by LeRobot 0.5.1. The runtime action spec is "
        "derived from the loaded checkpoint config."
    ),
)


def _import_pi0_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    except ImportError as exc:
        msg = (
            "pi0 adapter requires LeRobot pi0 dependencies. "
            'Install them with: pip install "vla_ros2[openpi]"'
        )
        raise MissingDependencyError(msg) from exc
    return torch, PI0Policy, make_pre_post_processors


def _pi0_tokenizer_repo(preprocessor: Mapping[str, Any]) -> str | None:
    """Extract the tokenizer repo the pi0 processor pulls from, from a decoded
    ``policy_preprocessor.json``. Returns None if not declared."""
    steps = preprocessor.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, Mapping) or step.get("registry_name") != "tokenizer_processor":
            continue
        config = step.get("config")
        if isinstance(config, Mapping):
            name = config.get("tokenizer_name")
            if isinstance(name, str) and name:
                return name
    return None


def _pi0_local_load_error(
    *,
    pretrained: str,
    weights_status: str,
    tokenizer_repo: str | None,
    tokenizer_status: str,
) -> str | None:
    """Pure: return an actionable error message when a local pi0 load would NOT
    produce a real, fully-weighted model, else None.

    ``*_status`` is one of ``"ok"`` / ``"gated"`` / ``"missing"``. This exists
    because ``PI0Policy.from_pretrained`` silently returns a randomly-initialized
    model when it cannot fetch the weights ("Returning model without loading
    pretrained weights"), and the processor build only fails later on the gated
    tokenizer — both hazards we turn into one explicit, documented failure.
    """
    if weights_status != "ok":
        return (
            f"pi0 local load aborted: the checkpoint weights for {pretrained} "
            f"({PI0_WEIGHTS_FILENAME}) could not be resolved (status={weights_status}). "
            "LeRobot's PI0Policy.from_pretrained silently returns a randomly-initialized "
            "model in this case, so vla_ros2 refuses rather than emit actions from "
            "un-trained weights. Make the checkpoint available (drop HF_HUB_OFFLINE / set "
            "local_files_only=False with network access, or point --pretrained at a cached "
            "checkpoint)."
        )
    if tokenizer_repo and tokenizer_status == "gated":
        return (
            f"pi0 local load aborted: the processor tokenizer '{tokenizer_repo}' is a gated "
            f"Hugging Face repo (access restricted). Accept its license at "
            f"https://huggingface.co/{tokenizer_repo} and run with an authorized HF token "
            "(huggingface-cli login, or set HF_TOKEN), then retry."
        )
    if tokenizer_repo and tokenizer_status == "missing":
        return (
            f"pi0 local load aborted: the processor tokenizer '{tokenizer_repo}' could not be "
            "resolved. Ensure it is cached or reachable (drop HF_HUB_OFFLINE with network "
            "access)."
        )
    return None


def _hf_asset_status(
    repo: str,
    filename: str,
    *,
    local_files_only: bool,
    cache_dir: str | None,
    revision: str | None,
    head_only: bool,
) -> str:
    """Best-effort availability probe: ``"ok"`` / ``"gated"`` / ``"missing"``.

    Never downloads large content: a cache hit short-circuits, and when online,
    large files (``head_only=True``) are checked with a metadata HEAD rather than a
    full download. Small descriptor files (``head_only=False``) are resolved so a
    gated repo raises ``GatedRepoError`` exactly as the real load would.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    # Cache-only check first — authoritative and never touches the network.
    try:
        hf_hub_download(
            repo, filename, local_files_only=True, cache_dir=cache_dir, revision=revision
        )
        return "ok"
    except Exception:  # noqa: BLE001 - not cached; fall through to the online path
        pass

    if local_files_only or os.environ.get("HF_HUB_OFFLINE"):
        return "missing"

    if head_only:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        try:
            get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
            return "ok"
        except GatedRepoError:
            return "gated"
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return "gated" if status in (401, 403) else "missing"
        except Exception:  # noqa: BLE001
            return "missing"

    try:
        hf_hub_download(repo, filename, cache_dir=cache_dir, revision=revision)
        return "ok"
    except GatedRepoError:
        return "gated"
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return "gated" if status in (401, 403) else "missing"
    except Exception:  # noqa: BLE001
        return "missing"


def run_pi0_local_preflight(
    pretrained: str,
    *,
    local_files_only: bool = False,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> None:
    """Abort with an actionable :class:`AdapterError` when a local pi0 load would
    silently produce a random-weight model or fail later on the gated tokenizer.

    This guard is intentionally **pi0-specific**. The hazard exists only because
    pi0's modeling overrides ``from_pretrained`` to catch a failed state-dict load
    and return an un-weighted model ("Returning model without loading pretrained
    weights"). SmolVLA — and any other LeRobot policy loaded through the shared
    ``SmolVLAAdapter`` base — uses LeRobot's base ``PreTrainedPolicy.from_pretrained``,
    which *raises* (``FileNotFoundError`` / ``LocalEntryNotFoundError``) when the
    weights are unavailable. Verified 2026-06-03: loading an uncached SmolVLA
    checkpoint offline raises ``LocalEntryNotFoundError`` rather than returning a
    random model, so no preflight is needed for that path.

    Best-effort: if huggingface_hub is unavailable the normal
    :class:`MissingDependencyError` path takes over; a probe that itself errors
    unexpectedly degrades to ``"missing"`` rather than masking the real load.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        return

    weights_status = _hf_asset_status(
        pretrained,
        PI0_WEIGHTS_FILENAME,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        revision=revision,
        head_only=True,
    )

    tokenizer_repo: str | None = None
    tokenizer_status = "ok"
    try:
        from huggingface_hub import hf_hub_download

        preprocessor_path = hf_hub_download(
            pretrained,
            PI0_PREPROCESSOR_FILENAME,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
        )
        with open(preprocessor_path, encoding="utf-8") as handle:
            tokenizer_repo = _pi0_tokenizer_repo(json.load(handle))
    except Exception:  # noqa: BLE001 - tokenizer repo is best-effort metadata
        tokenizer_repo = None

    if tokenizer_repo:
        tokenizer_status = _hf_asset_status(
            tokenizer_repo,
            "tokenizer_config.json",
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
            head_only=False,
        )

    message = _pi0_local_load_error(
        pretrained=pretrained,
        weights_status=weights_status,
        tokenizer_repo=tokenizer_repo,
        tokenizer_status=tokenizer_status,
    )
    if message:
        raise AdapterError(message)


class Pi0Adapter(SmolVLAAdapter):
    """Remote-first pi0 adapter with optional local LeRobot loading."""

    name = "pi0"
    model_id = DEFAULT_PI0_PRETRAINED
    action_spec = PI0_ACTION_SPEC
    default_pretrained = DEFAULT_PI0_PRETRAINED
    adapter_label = "LeRobot/OpenPI pi0"

    def __init__(
        self,
        *,
        enable_local: bool = False,
        pretrained: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        self._local_enabled = bool(enable_local or pretrained)
        if not self._local_enabled:
            VLAAdapter.__init__(
                self,
                name=self.name,
                model_id=self.model_id,
                action_spec=self.action_spec,
                metadata={
                    "remote_first": True,
                    "enable_local": False,
                    **kwargs,
                },
            )
            return

        resolved_pretrained = pretrained or self.default_pretrained
        # Fail loudly before the heavy load: LeRobot silently returns a
        # random-weight model when it cannot fetch the checkpoint, and the gated
        # PaliGemma tokenizer only trips later during processor construction.
        run_pi0_local_preflight(
            resolved_pretrained,
            local_files_only=bool(kwargs.get("local_files_only", False)),
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
        )
        super().__init__(
            pretrained=resolved_pretrained,
            strict=strict,
            action_spec=kwargs.pop("action_spec", None),
            **kwargs,
        )
        self._local_enabled = True

    @classmethod
    def from_config(cls, **kwargs: Any) -> Pi0Adapter:
        return cls(**kwargs)

    def _import_policy_dependencies(self) -> tuple[Any, Any, Any]:
        return _import_pi0_dependencies()

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        if not self._local_enabled:
            msg = (
                "Local pi0/openpi inference is disabled by default because pi0 checkpoints "
                "are heavy and checkpoint/config compatibility varies by LeRobot version. "
                "Call load_model('pi0', enable_local=True, pretrained='lerobot/pi0_base') "
                "in a dedicated GPU environment to enable it."
            )
            raise NotImplementedError(msg)
        return super().predict_observation(observation)
