from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any

import numpy as np

from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.model import VLAAdapter
from vla_zoo.core.types import ActionSpace, ActionSpec, VLAAction, VLAActionChunk, VLAObservation

DEFAULT_PRETRAINED = "lerobot/smolvla_base"

SMOLVLA_ACTION_SPEC = ActionSpec(
    action_space="custom",
    shape=(6,),
    control_hz=None,
    description=(
        "LeRobot SmolVLA continuous action. Shape and semantics are checkpoint-specific; "
        "lerobot/smolvla_base exposes a 6D action and predicts 50-step chunks internally."
    ),
)


def _import_smolvla_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:
        msg = (
            "SmolVLA adapter requires LeRobot SmolVLA dependencies. "
            'Install them with: pip install "vla_zoo[smolvla]"'
        )
        raise MissingDependencyError(msg) from exc
    return torch, SmolVLAPolicy, make_pre_post_processors


def _feature_kind(feature: Any) -> str:
    value = getattr(feature, "type", "")
    return str(getattr(value, "value", value)).upper()


def _feature_shape(feature: Any) -> tuple[int, ...]:
    shape = getattr(feature, "shape", ())
    return tuple(int(dim) for dim in shape)


def _prod(shape: tuple[int, ...]) -> int:
    return int(prod(shape)) if shape else 0


class SmolVLAAdapter(VLAAdapter):
    """Lazy local adapter for LeRobot SmolVLA policies."""

    name = "smolvla"
    model_id = DEFAULT_PRETRAINED
    action_spec = SMOLVLA_ACTION_SPEC
    default_pretrained = DEFAULT_PRETRAINED
    adapter_label = "LeRobot SmolVLA"

    def __init__(
        self,
        *,
        pretrained: str | None = None,
        device: str = "auto",
        action_space: ActionSpace = "custom",
        control_hz: float | None = None,
        return_action_chunk: bool = False,
        fill_missing_images: bool = True,
        zero_state_if_missing: bool = True,
        local_files_only: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        force_download: bool = False,
        strict: bool = False,
        preprocessor_overrides: Mapping[str, Any] | None = None,
        action_spec: ActionSpec | None = None,
        **kwargs: Any,
    ) -> None:
        pretrained = pretrained or self.default_pretrained
        torch, policy_cls, processor_factory = self._import_policy_dependencies()
        self._torch = torch
        self.pretrained = pretrained
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.return_action_chunk = bool(return_action_chunk)
        self.fill_missing_images = bool(fill_missing_images)
        self.zero_state_if_missing = bool(zero_state_if_missing)

        self.policy = policy_cls.from_pretrained(
            pretrained,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
        if hasattr(self.policy, "to"):
            self.policy = self.policy.to(self.device)
        if hasattr(self.policy, "eval"):
            self.policy.eval()

        overrides = dict(preprocessor_overrides or {})
        device_override = dict(overrides.get("device_processor", {}))
        device_override["device"] = str(self.device)
        overrides["device_processor"] = device_override
        self.preprocess, self.postprocess = processor_factory(
            self.policy.config,
            pretrained,
            preprocessor_overrides=overrides,
        )

        self.image_keys = self._config_feature_keys("VISUAL")
        self.state_keys = self._config_feature_keys("STATE")
        self.state_key = self.state_keys[0] if self.state_keys else "observation.state"
        self.state_shape = self._config_feature_shape(self.state_key, default=(0,))
        self.action_shape = self._config_action_shape()

        spec = action_spec or ActionSpec(
            action_space=action_space,
            shape=self.action_shape,
            control_hz=control_hz,
            description=(
                f"{self.adapter_label} action from {pretrained}; action semantics are "
                "checkpoint and robot specific."
            ),
        )
        super().__init__(
            name=self.name,
            model_id=pretrained,
            action_spec=spec,
            metadata={
                "pretrained": pretrained,
                "device": self.device,
                "image_keys": self.image_keys,
                "state_key": self.state_key,
                "chunk_size": getattr(self.policy.config, "chunk_size", None),
                "n_action_steps": getattr(self.policy.config, "n_action_steps", None),
            },
        )

    @classmethod
    def from_config(cls, **kwargs: Any) -> SmolVLAAdapter:
        return cls(**kwargs)

    def _import_policy_dependencies(self) -> tuple[Any, Any, Any]:
        return _import_smolvla_dependencies()

    def _config_feature_keys(self, feature_type: str) -> tuple[str, ...]:
        features = getattr(self.policy.config, "input_features", {})
        return tuple(
            key for key, feature in features.items() if _feature_kind(feature) == feature_type
        )

    def _config_feature_shape(self, key: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
        features = getattr(self.policy.config, "input_features", {})
        feature = features.get(key)
        return _feature_shape(feature) if feature is not None else default

    def _config_action_shape(self) -> tuple[int, ...]:
        action_feature = getattr(self.policy.config, "action_feature", None)
        if action_feature is not None:
            shape = _feature_shape(action_feature)
            if shape:
                return shape
        output_features = getattr(self.policy.config, "output_features", {})
        action = output_features.get("action")
        shape = _feature_shape(action)
        return shape or self.action_spec.shape

    def _default_image_tensor(self, key: str) -> Any:
        shape = self._config_feature_shape(key, default=(3, 256, 256))
        if len(shape) != 3:
            shape = (3, 256, 256)
        return self._torch.zeros(shape, dtype=self._torch.float32)

    def _image_candidates(
        self,
        key: str,
        observation: VLAObservation,
    ) -> tuple[Any | None, str | None]:
        suffix = key.rsplit(".", maxsplit=1)[-1]
        aliases = (
            key,
            suffix,
            suffix.replace("camera", "image"),
            "primary" if suffix in {"camera1", "image", "image1"} else "",
        )
        for alias in aliases:
            if alias and alias in observation.images and observation.images[alias] is not None:
                return observation.images[alias], alias
        for alias, value in observation.images.items():
            if value is not None:
                return value, str(alias)
        return None, None

    def _image_to_tensor(self, image: Any, *, key: str) -> Any:
        torch = self._torch
        if hasattr(image, "detach"):
            tensor = image.detach()
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 3:
                msg = (
                    f"{self.adapter_label} image {key!r} must be CHW or HWC, "
                    f"got shape {tuple(tensor.shape)}"
                )
                raise ValueError(msg)
            if tensor.shape[0] not in {1, 3}:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.to(dtype=torch.float32)
            if tensor.numel() and float(tensor.detach().max().cpu()) > 1.5:
                tensor = tensor / 255.0
        else:
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            array = np.array(image, copy=True)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            if array.ndim != 3:
                msg = (
                    f"{self.adapter_label} image {key!r} must be HWC or CHW, "
                    f"got shape {array.shape}"
                )
                raise ValueError(msg)
            tensor = torch.as_tensor(array)
            if tensor.shape[0] not in {1, 3}:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.to(dtype=torch.float32)
            if np.issubdtype(array.dtype, np.integer) or float(tensor.max().cpu()) > 1.5:
                tensor = tensor / 255.0
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] != 3:
            msg = (
                f"{self.adapter_label} image {key!r} must have 1 or 3 channels, "
                f"got {tensor.shape[0]}"
            )
            raise ValueError(msg)
        return tensor.contiguous()

    def _frame_images(self, observation: VLAObservation) -> tuple[dict[str, Any], list[str]]:
        frame: dict[str, Any] = {}
        filled: list[str] = []
        for key in self.image_keys:
            image, alias = self._image_candidates(key, observation)
            if image is None:
                if not self.fill_missing_images:
                    msg = f"{self.adapter_label} requires image key {key!r}"
                    raise ValueError(msg)
                frame[key] = self._default_image_tensor(key)
                filled.append(key)
                continue
            frame[key] = self._image_to_tensor(image, key=key)
            if alias != key:
                filled.append(f"{key}<={alias}")
        if not frame:
            msg = f"{self.adapter_label} checkpoint does not declare visual input features"
            raise ValueError(msg)
        return frame, filled

    def _state_tensor(self, observation: VLAObservation) -> tuple[Any, bool]:
        state_value = None
        state_sources: list[Mapping[str, Any]] = [observation.state]
        metadata_state = observation.metadata.get("state")
        if isinstance(metadata_state, Mapping):
            state_sources.append(metadata_state)
        elif metadata_state is not None:
            state_sources.append({"state": metadata_state})

        for source in state_sources:
            for key in (self.state_key, "state", "agent_pos", "proprio", "joint_positions"):
                if key in source:
                    state_value = source[key]
                    break
            if state_value is not None:
                break

        state_size = max(_prod(self.state_shape), 1)
        if state_value is None:
            if not self.zero_state_if_missing:
                msg = f"{self.adapter_label} requires state key {self.state_key!r}"
                raise ValueError(msg)
            return self._torch.zeros(state_size, dtype=self._torch.float32), True

        tensor = self._torch.as_tensor(state_value, dtype=self._torch.float32).reshape(-1)
        if tensor.numel() < state_size:
            padded = self._torch.zeros(state_size, dtype=self._torch.float32)
            padded[: tensor.numel()] = tensor
            return padded, True
        if tensor.numel() > state_size:
            return tensor[:state_size], True
        return tensor, False

    def _build_frame(self, observation: VLAObservation) -> tuple[dict[str, Any], dict[str, Any]]:
        frame, filled_images = self._frame_images(observation)
        state, state_filled = self._state_tensor(observation)
        frame[self.state_key] = state
        frame["task"] = observation.instruction
        return frame, {"filled_images": filled_images, "state_filled_or_resized": state_filled}

    def _postprocess_tensor(self, action: Any) -> Any:
        try:
            return self.postprocess(action)
        except Exception:
            return action

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        frame, conversion_metadata = self._build_frame(observation)
        batch = self.preprocess(frame)

        with self._torch.inference_mode():
            if self.return_action_chunk:
                action_tensor = self.policy.predict_action_chunk(batch)
            else:
                action_tensor = self.policy.select_action(batch)
        action_tensor = self._postprocess_tensor(action_tensor)

        data = action_tensor.detach().to(dtype=self._torch.float32).cpu().numpy()
        if self.return_action_chunk:
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            if data.ndim != 2:
                msg = (
                    f"{self.adapter_label} action chunk must be 2D after batch squeeze, "
                    f"got shape {data.shape}"
                )
                raise ValueError(msg)
            actions = [
                VLAAction(
                    data=np.asarray(row, dtype=np.float32).reshape(self.action_spec.shape),
                    spec=self.action_spec,
                    dt=1.0 / self.action_spec.control_hz if self.action_spec.control_hz else None,
                    chunk_index=index,
                    metadata={
                        "model": self.pretrained,
                        "adapter": type(self).__name__,
                        **conversion_metadata,
                    },
                )
                for index, row in enumerate(data)
            ]
            return VLAActionChunk(
                actions=actions,
                metadata={
                    "model": self.pretrained,
                    "adapter": type(self).__name__,
                    "chunk_size": len(actions),
                    **conversion_metadata,
                },
            )

        if data.ndim == 2 and data.shape[0] == 1:
            data = data[0]
        return VLAAction(
            data=np.asarray(data, dtype=np.float32).reshape(self.action_spec.shape),
            spec=self.action_spec,
            dt=1.0 / self.action_spec.control_hz if self.action_spec.control_hz else None,
            metadata={
                "model": self.pretrained,
                "adapter": type(self).__name__,
                "image_keys": self.image_keys,
                "state_key": self.state_key,
                **conversion_metadata,
            },
        )
