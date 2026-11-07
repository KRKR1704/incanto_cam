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

    if video.width <= 0 or video.height <= 0:
        raise ConfigError("video.width and video.height must be positive")
    if not 1 <= video.fps <= 120:
        raise ConfigError("video.fps must be between 1 and 120")
    if not 0 < video.perception_width <= video.width:
        raise ConfigError("video.perception_width must be in (0, video.width]")
    if vcam.backend not in VCAM_BACKENDS:
        raise ConfigError(f"vcam.backend must be one of {VCAM_BACKENDS}, got '{vcam.backend}'")
    if triggers.global_cooldown_s < 0:
        raise ConfigError("triggers.global_cooldown_s must be >= 0")
    if not 0 <= triggers.min_confidence <= 1:
        raise ConfigError("triggers.min_confidence must be between 0 and 1")

    mappings = _section(raw, "mappings")
    for gesture, spell in mappings.items():
        if not isinstance(spell, str):
            raise ConfigError(f"mappings.{gesture} must be a spell name (string), got {spell!r}")

    return Config(
        video=video,
        vcam=vcam,
        triggers=triggers,
        mappings=dict(mappings),
        gestures=_section(raw, "gestures"),
        spells=_section(raw, "spells"),
    )
