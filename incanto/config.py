"""Loads and validates config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
VCAM_BACKENDS = ("obs", "unitycapture")


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed, or has invalid values."""


@dataclass(frozen=True)
class VideoConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    perception_width: int = 640
    capture_mode: str = "threaded"    # inline | threaded


@dataclass(frozen=True)
class ModelsConfig:
    hand_landmarker: str = "models/hand_landmarker.task"
    selfie_segmenter: str = "models/selfie_segmenter_landscape.tflite"

    def resolve(self, name: str) -> Path:
        """Absolute path of a configured model (relative paths are from the project root)."""
        path = Path(getattr(self, name))
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class PerceptionConfig:
    num_hands: int = 2
    hand_min_detection_confidence: float = 0.5
    hand_min_presence_confidence: float = 0.5
    hand_min_tracking_confidence: float = 0.5
    mask_blur_px: int = 5
    mask_feather_low: float = 0.35
    mask_feather_high: float = 0.65
    mask_temporal_alpha: float = 0.6
    segment_every_n: int = 2
    hands_async: bool = True


@dataclass(frozen=True)
class PreviewConfig:
    every_n: int = 2


@dataclass(frozen=True)
class VCamConfig:
    backend: str = "obs"


@dataclass(frozen=True)
class TriggersConfig:
    global_cooldown_s: float = 3.0
    min_confidence: float = 0.8


@dataclass(frozen=True)
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    vcam: VCamConfig = field(default_factory=VCamConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    triggers: TriggersConfig = field(default_factory=TriggersConfig)
    mappings: dict[str, str] = field(default_factory=dict)
    gestures: dict[str, dict[str, Any]] = field(default_factory=dict)
    spells: dict[str, dict[str, Any]] = field(default_factory=dict)


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping, got {type(value).__name__}")
    return value


def _build(cls, name: str, values: dict):
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in '{name}': {', '.join(sorted(unknown))}")
    defaults = cls()
    for key, value in values.items():
        expected = type(getattr(defaults, key))
        is_bool = isinstance(value, bool)
        if expected is float and isinstance(value, (int, float)) and not is_bool:
            continue
        if not isinstance(value, expected) or (is_bool and expected is not bool):
            raise ConfigError(f"'{name}.{key}' must be {expected.__name__}, got {value!r}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    video = _build(VideoConfig, "video", _section(raw, "video"))
    vcam = _build(VCamConfig, "vcam", _section(raw, "vcam"))
    triggers = _build(TriggersConfig, "triggers", _section(raw, "triggers"))
    models = _build(ModelsConfig, "models", _section(raw, "models"))
    perception = _build(PerceptionConfig, "perception", _section(raw, "perception"))
    preview = _build(PreviewConfig, "preview", _section(raw, "preview"))

    if video.width <= 0 or video.height <= 0:
        raise ConfigError("video.width and video.height must be positive")
    if not 1 <= video.fps <= 120:
        raise ConfigError("video.fps must be between 1 and 120")
    if video.capture_mode not in ("inline", "threaded"):
        raise ConfigError("video.capture_mode must be 'inline' or 'threaded'")
    if not 0 < video.perception_width <= video.width:
        raise ConfigError("video.perception_width must be in (0, video.width]")
    if vcam.backend not in VCAM_BACKENDS:
        raise ConfigError(f"vcam.backend must be one of {VCAM_BACKENDS}, got '{vcam.backend}'")
    if triggers.global_cooldown_s < 0:
        raise ConfigError("triggers.global_cooldown_s must be >= 0")
    if not 0 <= triggers.min_confidence <= 1:
        raise ConfigError("triggers.min_confidence must be between 0 and 1")

    if not 1 <= perception.num_hands <= 2:
        raise ConfigError("perception.num_hands must be 1 or 2")
    for key in ("hand_min_detection_confidence", "hand_min_presence_confidence",
                "hand_min_tracking_confidence", "mask_feather_low", "mask_feather_high"):
        if not 0 <= getattr(perception, key) <= 1:
            raise ConfigError(f"perception.{key} must be between 0 and 1")
    if perception.mask_feather_low >= perception.mask_feather_high:
        raise ConfigError("perception.mask_feather_low must be less than mask_feather_high")
    if not 0 < perception.mask_temporal_alpha <= 1:
        raise ConfigError("perception.mask_temporal_alpha must be in (0, 1]")
    if perception.mask_blur_px < 0:
        raise ConfigError("perception.mask_blur_px must be >= 0")
    if perception.segment_every_n < 1:
        raise ConfigError("perception.segment_every_n must be >= 1")

    if preview.every_n < 1:
        raise ConfigError("preview.every_n must be >= 1")

    mappings = _section(raw, "mappings")
    for gesture, spell in mappings.items():
        if not isinstance(spell, str):
            raise ConfigError(f"mappings.{gesture} must be a spell name (string), got {spell!r}")

    return Config(
        video=video,
        vcam=vcam,
        models=models,
        perception=perception,
        preview=preview,
        triggers=triggers,
        mappings=dict(mappings),
        gestures=_section(raw, "gestures"),
        spells=_section(raw, "spells"),
    )
