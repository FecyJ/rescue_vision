"""版本化运行配置及几何对象装配。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector


SCHEMA_VERSION = 1


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{location} must be a mapping with string keys.")
    return value


def _reject_unknown(
    data: dict[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {location}: {unknown}.")


def _required(data: dict[str, Any], key: str, location: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key {location}.{key}.")
    return data[key]


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer, got {value!r}.")
    return value


def _finite_float(value: object, location: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number, got {value!r}.")
    converted = float(value)
    if not (converted >= minimum and converted < float("inf")):
        raise ValueError(
            f"{location} must be finite and >= {minimum}, got {value!r}."
        )
    return converted


def _path_or_none(value: object, base_dir: Path, location: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty path string or null.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


@dataclass(frozen=True, slots=True)
class CameraConfig:
    backend: str
    image_size: tuple[int, int]
    fps: int
    lens_position: float


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    enabled: bool
    intrinsics_path: Path | None
    ground_mapping_path: Path | None


@dataclass(frozen=True, slots=True)
class RecordingConfig:
    queue_capacity: int
    image_format: str


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    max_observation_age_ms: float


@dataclass(frozen=True, slots=True)
class RuntimeGeometry:
    camera_model: CameraModel
    ground_projector: GroundProjector


@dataclass(frozen=True, slots=True)
class AppConfig:
    schema_version: int
    camera: CameraConfig
    geometry: GeometryConfig
    recording: RecordingConfig
    processing: ProcessingConfig

    def build_geometry(self) -> RuntimeGeometry | None:
        """加载并交叉验证内参、运行分辨率和地面映射。"""

        if not self.geometry.enabled:
            return None
        assert self.geometry.intrinsics_path is not None
        assert self.geometry.ground_mapping_path is not None

        calibration = CameraCalibration.from_json(
            self.geometry.intrinsics_path,
            allow_unusable=False,
        )
        if calibration.image_size != self.camera.image_size:
            raise ValueError(
                f"Runtime camera image_size {self.camera.image_size} does not "
                f"match intrinsics {calibration.image_size}."
            )
        projector = GroundProjector.from_json(
            self.geometry.ground_mapping_path,
            camera_calibration=calibration,
        )
        return RuntimeGeometry(CameraModel(calibration), projector)


def load_runtime_config(path: str | Path) -> AppConfig:
    """从 YAML 加载 schema v1；缺项和未知字段均视为错误。"""

    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    _reject_unknown(
        root,
        {"schema_version", "camera", "geometry", "recording", "processing"},
        "root",
    )

    schema_version = _required(root, "schema_version", "root")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config schema_version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}."
        )

    camera_raw = _mapping(_required(root, "camera", "root"), "camera")
    _reject_unknown(
        camera_raw,
        {"backend", "image_size", "fps", "lens_position"},
        "camera",
    )
    backend = _required(camera_raw, "backend", "camera")
    if backend not in {"rpicam_vid", "picamera2"}:
        raise ValueError(
            "camera.backend must be 'rpicam_vid' or 'picamera2'."
        )
    image_size_raw = _required(camera_raw, "image_size", "camera")
    if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
        raise ValueError("camera.image_size must be [width, height].")
    image_size = (
        _positive_int(image_size_raw[0], "camera.image_size[0]"),
        _positive_int(image_size_raw[1], "camera.image_size[1]"),
    )
    camera = CameraConfig(
        backend=backend,
        image_size=image_size,
        fps=_positive_int(_required(camera_raw, "fps", "camera"), "camera.fps"),
        lens_position=_finite_float(
            _required(camera_raw, "lens_position", "camera"),
            "camera.lens_position",
            minimum=0.0,
        ),
    )

    geometry_raw = _mapping(_required(root, "geometry", "root"), "geometry")
    _reject_unknown(
        geometry_raw,
        {"enabled", "intrinsics_path", "ground_mapping_path"},
        "geometry",
    )
    enabled = _required(geometry_raw, "enabled", "geometry")
    if not isinstance(enabled, bool):
        raise ValueError("geometry.enabled must be a boolean.")
    base_dir = config_path.parent
    intrinsics_path = _path_or_none(
        geometry_raw.get("intrinsics_path"),
        base_dir,
        "geometry.intrinsics_path",
    )
    ground_mapping_path = _path_or_none(
        geometry_raw.get("ground_mapping_path"),
        base_dir,
        "geometry.ground_mapping_path",
    )
    if enabled and (intrinsics_path is None or ground_mapping_path is None):
        raise ValueError(
            "Enabled geometry requires intrinsics_path and ground_mapping_path."
        )
    geometry = GeometryConfig(enabled, intrinsics_path, ground_mapping_path)

    recording_raw = _mapping(_required(root, "recording", "root"), "recording")
    _reject_unknown(recording_raw, {"queue_capacity", "image_format"}, "recording")
    image_format = _required(recording_raw, "image_format", "recording")
    if image_format not in {"png", "jpg"}:
        raise ValueError("recording.image_format must be 'png' or 'jpg'.")
    recording = RecordingConfig(
        queue_capacity=_positive_int(
            _required(recording_raw, "queue_capacity", "recording"),
            "recording.queue_capacity",
        ),
        image_format=image_format,
    )

    processing_raw = _mapping(
        _required(root, "processing", "root"),
        "processing",
    )
    _reject_unknown(
        processing_raw,
        {"max_observation_age_ms"},
        "processing",
    )
    processing = ProcessingConfig(
        max_observation_age_ms=_finite_float(
            _required(
                processing_raw,
                "max_observation_age_ms",
                "processing",
            ),
            "processing.max_observation_age_ms",
            minimum=0.001,
        )
    )

    return AppConfig(SCHEMA_VERSION, camera, geometry, recording, processing)
