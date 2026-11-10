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
    mask_temporal: str = "attack_release"  # attack_release | symmetric
    mask_release_decay: float = 0.85       # attack_release: mask = max(current, previous * decay)
    segment_every_n: int = 1
    hands_async: bool = True


@dataclass(frozen=True)
class PreviewConfig:
    every_n: int = 2


@dataclass(frozen=True)
class CameraConfig:
    exposure: float | None = None  # None = auto; else manual exposure in log2 seconds (e.g. -6 = 1/64 s)


VANISH_MASK_MODES = ("seg_only", "seg_hands", "full")
PLATE_MATCH_MODES = ("off", "gain", "gain_offset", "gain_map")
PLATE_DIFF_METRICS = ("gray", "max_channel", "lab")


@dataclass(frozen=True)
class VanishConfig:
    """Mask used when the person is replaced by the plate. Over-covering is free (the plate IS the
    background); under-covering reveals the person, so everything here biases toward covering.
    Pixel sizes are at 1280 px frame width and are scaled to the perception resolution."""
    mask_mode: str = "full"              # seg_only | seg_hands | full (seg + hands + plate difference)
    dilate_px: int = 15                  # grow the segmenter mask
    hand_hull_dilate_px: int = 25        # grow the filled convex hull of each hand (landmarks are ~50 ms old)
    core_threshold: float = 0.1          # a pixel is "covered" (alpha exactly 1) if any source exceeds this
    core_close_px: int = 15              # close pinholes in the solid core
    outer_feather_px: int = 12           # blend ring OUTSIDE the solid core only (the ring never overlaps you)
    plate_diff_metric: str = "max_channel"  # gray | max_channel | lab: how |live - matched plate| is measured
    plate_diff_threshold: int = 28       # levels of difference that count as "not background"
    plate_diff_open_px: int = 3          # morphological open: remove speckle
    plate_diff_close_px: int = 9         # morphological close: fill holes
    plate_diff_release_decay: float = 0.8  # fast attack / slow release for the difference mask and zone
    plate_diff_large_blob_fraction: float = 0.05  # body-sized differences count anywhere (0 = only near me)
    plate_diff_cell_guard: bool = True   # per-cell lighting guard replaces the old global kill switch
    plate_diff_wash_fraction: float = 0.85  # guard: a cell is a "wash" if this much of it differs...
    plate_diff_wash_residual: float = 12.0  # ...and one neutral gain explains it to within this many levels
    near_me_px: int = 70                 # the plate difference is only allowed this far around the mask
    plate_match: str = "gain_map"        # off | gain | gain_offset | gain_map: match the plate to live brightness/colour
    plate_match_cells_x: int = 8         # gain_map grid size
    plate_match_cells_y: int = 6
    plate_match_smoothing: float = 0.15  # weight of each frame's estimate in the running gain/offset
    match_margin_px: int = 30            # background used for matching is this far outside the mask
    plate_refresh_alpha: float = 0.02    # per frame: blend live pixels into the plate OUTSIDE the near-me zone (0 = off)
    plate_refresh_guard_px: int = 40     # the refresh stays this much further out than the near-me zone


@dataclass(frozen=True)
class PlateConfig:
    frames: int = 30                     # frames averaged into the clean plate
    countdown_s: float = 5.0             # countdown after pressing B (time to leave the frame)
    settle_s: float = 1.0                # extra wait for auto exposure to settle on the empty room
    max_person_coverage: float = 0.02    # abort if the mask covers more than this fraction of the frame
    dir: str = "plates"                  # local-only (gitignored): a photo of your room
    sounds: bool = True                  # beeps for countdown / capturing / captured / error (winsound, Windows only)
    # Plate health: the plate must match the visible background, else vanish is refused / switched off.
    health_max_raw_diff: float = 20.0    # |mean live background - mean plate background|, levels
    health_gain_min: float = 0.8         # global live/plate gain (each colour channel) must stay inside
    health_gain_max: float = 1.25        # ...this range
    health_max_cell_gain: float = 1.5    # no grid cell may need more gain than this
    health_check_s: float = 0.5          # how often the plate is checked against the room
    health_fail_s: float = 1.0           # while vanish is on: unhealthy for this long switches vanish off

    def resolve_dir(self) -> Path:
        path = Path(self.dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


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
    camera: CameraConfig = field(default_factory=CameraConfig)
    vcam: VCamConfig = field(default_factory=VCamConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    vanish: VanishConfig = field(default_factory=VanishConfig)
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


def _parse_camera(values: dict) -> CameraConfig:
    unknown = set(values) - {"exposure"}
    if unknown:
        raise ConfigError(f"unknown key(s) in 'camera': {', '.join(sorted(unknown))}")
    exposure = values.get("exposure", "auto")
    if exposure == "auto" or exposure is None:
        return CameraConfig(None)
    if isinstance(exposure, bool) or not isinstance(exposure, (int, float)):
        raise ConfigError(f"camera.exposure must be \"auto\" or a number (e.g. -6), got {exposure!r}")
    if not -13 <= exposure <= 0:
        raise ConfigError("camera.exposure must be between -13 and 0 (log2 seconds)")
    return CameraConfig(float(exposure))


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
    camera = _parse_camera(_section(raw, "camera"))
    triggers = _build(TriggersConfig, "triggers", _section(raw, "triggers"))
    models = _build(ModelsConfig, "models", _section(raw, "models"))
    perception = _build(PerceptionConfig, "perception", _section(raw, "perception"))
    preview = _build(PreviewConfig, "preview", _section(raw, "preview"))
    plate = _build(PlateConfig, "plate", _section(raw, "plate"))
    vanish = _build(VanishConfig, "vanish", _section(raw, "vanish"))

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
    if perception.mask_temporal not in ("attack_release", "symmetric"):
        raise ConfigError("perception.mask_temporal must be 'attack_release' or 'symmetric'")
    if not 0 <= perception.mask_release_decay < 1:
        raise ConfigError("perception.mask_release_decay must be in [0, 1)")
    if perception.segment_every_n < 1:
        raise ConfigError("perception.segment_every_n must be >= 1")

    if preview.every_n < 1:
        raise ConfigError("preview.every_n must be >= 1")

    if vanish.mask_mode not in VANISH_MASK_MODES:
        raise ConfigError(f"vanish.mask_mode must be one of {VANISH_MASK_MODES}, got '{vanish.mask_mode}'")
    if vanish.plate_match not in PLATE_MATCH_MODES:
        raise ConfigError(f"vanish.plate_match must be one of {PLATE_MATCH_MODES}, got '{vanish.plate_match}'")
    if vanish.plate_diff_metric not in PLATE_DIFF_METRICS:
        raise ConfigError(f"vanish.plate_diff_metric must be one of {PLATE_DIFF_METRICS}, got '{vanish.plate_diff_metric}'")
    for key in ("dilate_px", "hand_hull_dilate_px", "core_close_px", "outer_feather_px", "plate_diff_open_px",
                "plate_diff_close_px", "match_margin_px", "plate_diff_threshold", "near_me_px",
                "plate_diff_wash_residual", "plate_refresh_guard_px"):
        if getattr(vanish, key) < 0:
            raise ConfigError(f"vanish.{key} must be >= 0")
    if not (1 <= vanish.plate_match_cells_x <= 64 and 1 <= vanish.plate_match_cells_y <= 64):
        raise ConfigError("vanish.plate_match_cells_x and plate_match_cells_y must be between 1 and 64")
    if not 0 < vanish.plate_match_smoothing <= 1:
        raise ConfigError("vanish.plate_match_smoothing must be in (0, 1]")
    if not 0 <= vanish.core_threshold < 1:
        raise ConfigError("vanish.core_threshold must be in [0, 1)")
    if not 0 <= vanish.plate_diff_release_decay < 1:
        raise ConfigError("vanish.plate_diff_release_decay must be in [0, 1)")
    if not 0 <= vanish.plate_diff_large_blob_fraction <= 1:
        raise ConfigError("vanish.plate_diff_large_blob_fraction must be between 0 and 1")
    if not 0 < vanish.plate_diff_wash_fraction <= 1:
        raise ConfigError("vanish.plate_diff_wash_fraction must be in (0, 1]")
    if not 0 <= vanish.plate_refresh_alpha <= 0.5:
        raise ConfigError("vanish.plate_refresh_alpha must be between 0 (off) and 0.5")

    if plate.frames < 1:
        raise ConfigError("plate.frames must be >= 1")
    if plate.countdown_s < 0 or plate.settle_s < 0:
        raise ConfigError("plate.countdown_s and plate.settle_s must be >= 0")
    if not 0 <= plate.max_person_coverage <= 1:
        raise ConfigError("plate.max_person_coverage must be between 0 and 1")
    if plate.health_max_raw_diff <= 0 or plate.health_max_cell_gain <= 1:
        raise ConfigError("plate.health_max_raw_diff must be > 0 and plate.health_max_cell_gain > 1")
    if not 0 < plate.health_gain_min < 1 < plate.health_gain_max:
        raise ConfigError("plate.health_gain_min must be in (0, 1) and plate.health_gain_max above 1")
    if plate.health_check_s <= 0 or plate.health_fail_s < 0:
        raise ConfigError("plate.health_check_s must be > 0 and plate.health_fail_s >= 0")

    mappings = _section(raw, "mappings")
    for gesture, spell in mappings.items():
        if not isinstance(spell, str):
            raise ConfigError(f"mappings.{gesture} must be a spell name (string), got {spell!r}")

    return Config(
        video=video,
        camera=camera,
        vcam=vcam,
        models=models,
        perception=perception,
        preview=preview,
        plate=plate,
        vanish=vanish,
        triggers=triggers,
        mappings=dict(mappings),
        gestures=_section(raw, "gestures"),
        spells=_section(raw, "spells"),
    )
