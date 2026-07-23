"""版本化运行配置及几何对象装配。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.perception.types import TargetClass


SCHEMA_VERSION = 3


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


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def _threshold(value: object, location: str) -> float:
    return _finite_float(value, location, minimum=0.0)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    backend: str
    image_size: tuple[int, int]
    fps: int
    lens_position: float


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    intrinsics_enabled: bool
    intrinsics_path: Path | None
    ground_mapping_enabled: bool
    ground_mapping_path: Path | None


@dataclass(frozen=True, slots=True)
class RecordingConfig:
    queue_capacity: int
    image_format: str


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    max_observation_age_ms: float


@dataclass(frozen=True, slots=True)
class HailoConfig:
    enabled: bool
    hef_path: Path | None
    postprocess_onnx_path: Path | None
    output_mapping_path: Path | None
    model_version: str | None
    hef_sha256: str | None
    raw_classes: tuple[str, ...]
    class_mapping: tuple[TargetClass, ...]
    detection_threshold: float
    semantic_threshold: float
    k0_threshold: float
    max_detections: int

    def build_backend(self):
        """延迟导入并创建 Hailo 后端；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        assert self.hef_path is not None
        assert self.postprocess_onnx_path is not None
        assert self.output_mapping_path is not None
        assert self.model_version is not None
        assert self.hef_sha256 is not None
        from rescue_vision.perception.hailo_yolo26_pose import HailoYolo26PoseBackend

        return HailoYolo26PoseBackend(
            hef_path=self.hef_path,
            postprocess_onnx_path=self.postprocess_onnx_path,
            output_mapping_path=self.output_mapping_path,
            model_version=self.model_version,
            model_sha256=self.hef_sha256,
            class_count=len(self.raw_classes),
            max_detections=self.max_detections,
            score_threshold=self.detection_threshold,
        )

    def model_class_mapping(self) -> dict[int, TargetClass]:
        return dict(enumerate(self.class_mapping))


@dataclass(frozen=True, slots=True)
class RuntimeGeometry:
    camera_model: CameraModel
    ground_projector: GroundProjector | None


@dataclass(frozen=True, slots=True)
class AppConfig:
    schema_version: int
    camera: CameraConfig
    geometry: GeometryConfig
    recording: RecordingConfig
    processing: ProcessingConfig
    hailo: HailoConfig

    def build_camera_model(self) -> CameraModel | None:
        """内参启用时加载并校验与运行分辨率一致的相机模型。"""

        if not self.geometry.intrinsics_enabled:
            return None
        assert self.geometry.intrinsics_path is not None

        calibration = CameraCalibration.from_json(
            self.geometry.intrinsics_path,
            allow_unusable=False,
        )
        if calibration.image_size != self.camera.image_size:
            raise ValueError(
                f"Runtime camera image_size {self.camera.image_size} does not "
                f"match intrinsics {calibration.image_size}."
            )
        return CameraModel(calibration)

    def build_geometry(self) -> RuntimeGeometry | None:
        """按独立开关装配相机模型，并可选装配地面映射。"""

        camera_model = self.build_camera_model()
        if camera_model is None:
            return None
        if not self.geometry.ground_mapping_enabled:
            return RuntimeGeometry(camera_model, None)
        assert self.geometry.ground_mapping_path is not None
        projector = GroundProjector.from_json(
            self.geometry.ground_mapping_path,
            camera_calibration=camera_model.calibration,
        )
        return RuntimeGeometry(camera_model, projector)


def load_runtime_config(path: str | Path) -> AppConfig:
    """从 YAML 加载 schema v3；缺项和未知字段均视为错误。"""

    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    _reject_unknown(
        root,
        {
            "schema_version",
            "camera",
            "geometry",
            "recording",
            "processing",
            "hailo",
        },
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
        {
            "intrinsics_enabled",
            "intrinsics_path",
            "ground_mapping_enabled",
            "ground_mapping_path",
        },
        "geometry",
    )
    intrinsics_enabled = _required(
        geometry_raw,
        "intrinsics_enabled",
        "geometry",
    )
    if not isinstance(intrinsics_enabled, bool):
        raise ValueError("geometry.intrinsics_enabled must be a boolean.")
    ground_mapping_enabled = _required(
        geometry_raw,
        "ground_mapping_enabled",
        "geometry",
    )
    if not isinstance(ground_mapping_enabled, bool):
        raise ValueError("geometry.ground_mapping_enabled must be a boolean.")
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
    if intrinsics_enabled and intrinsics_path is None:
        raise ValueError(
            "Enabled intrinsics requires geometry.intrinsics_path."
        )
    if ground_mapping_enabled and not intrinsics_enabled:
        raise ValueError(
            "Enabled ground mapping requires geometry.intrinsics_enabled=true."
        )
    if ground_mapping_enabled and ground_mapping_path is None:
        raise ValueError(
            "Enabled ground mapping requires geometry.ground_mapping_path."
        )
    geometry = GeometryConfig(
        intrinsics_enabled,
        intrinsics_path,
        ground_mapping_enabled,
        ground_mapping_path,
    )

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

    hailo_raw = _mapping(_required(root, "hailo", "root"), "hailo")
    _reject_unknown(
        hailo_raw,
        {
            "enabled",
            "hef_path",
            "postprocess_onnx_path",
            "output_mapping_path",
            "model_version",
            "hef_sha256",
            "raw_classes",
            "class_mapping",
            "detection_threshold",
            "semantic_threshold",
            "k0_threshold",
            "max_detections",
        },
        "hailo",
    )
    hailo_enabled = _required(hailo_raw, "enabled", "hailo")
    if not isinstance(hailo_enabled, bool):
        raise ValueError("hailo.enabled must be a boolean.")
    hef_path = _path_or_none(
        hailo_raw.get("hef_path"), base_dir, "hailo.hef_path"
    )
    postprocess_onnx_path = _path_or_none(
        hailo_raw.get("postprocess_onnx_path"),
        base_dir,
        "hailo.postprocess_onnx_path",
    )
    output_mapping_path = _path_or_none(
        hailo_raw.get("output_mapping_path"),
        base_dir,
        "hailo.output_mapping_path",
    )
    model_version_value = hailo_raw.get("model_version")
    model_version = (
        _string(model_version_value, "hailo.model_version")
        if model_version_value is not None
        else None
    )
    checksum_value = hailo_raw.get("hef_sha256")
    checksum = (
        _string(checksum_value, "hailo.hef_sha256").lower()
        if checksum_value is not None
        else None
    )
    if checksum is not None and (
        len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("hailo.hef_sha256 must be 64 hexadecimal characters.")

    raw_classes_value = hailo_raw.get("raw_classes", [])
    if not isinstance(raw_classes_value, list):
        raise ValueError("hailo.raw_classes must be a list.")
    raw_classes = tuple(
        _string(value, f"hailo.raw_classes[{index}]")
        for index, value in enumerate(raw_classes_value)
    )
    if len(set(raw_classes)) != len(raw_classes):
        raise ValueError("hailo.raw_classes must not contain duplicates.")

    mapping_value = hailo_raw.get("class_mapping", {})
    mapping_raw = _mapping(mapping_value, "hailo.class_mapping")
    if set(mapping_raw) != set(raw_classes):
        raise ValueError(
            "hailo.class_mapping keys must exactly match hailo.raw_classes."
        )
    try:
        class_mapping = tuple(
            TargetClass(_string(mapping_raw[name], f"hailo.class_mapping.{name}"))
            for name in raw_classes
        )
    except ValueError as exc:
        raise ValueError(
            "hailo.class_mapping values must be green_supply, black_core, "
            "orange_injured, blue_danger or unknown."
        ) from exc

    detection_threshold = _threshold(
        _required(hailo_raw, "detection_threshold", "hailo"),
        "hailo.detection_threshold",
    )
    semantic_threshold = _threshold(
        _required(hailo_raw, "semantic_threshold", "hailo"),
        "hailo.semantic_threshold",
    )
    k0_threshold = _threshold(
        _required(hailo_raw, "k0_threshold", "hailo"),
        "hailo.k0_threshold",
    )
    for location, value in (
        ("hailo.detection_threshold", detection_threshold),
        ("hailo.semantic_threshold", semantic_threshold),
        ("hailo.k0_threshold", k0_threshold),
    ):
        if value > 1.0:
            raise ValueError(f"{location} must be <= 1.0.")
    if semantic_threshold < detection_threshold:
        raise ValueError(
            "hailo.semantic_threshold must be >= hailo.detection_threshold."
        )
    max_detections = _positive_int(
        _required(hailo_raw, "max_detections", "hailo"),
        "hailo.max_detections",
    )

    required_assets = (
        hef_path,
        postprocess_onnx_path,
        output_mapping_path,
        model_version,
        checksum,
    )
    if hailo_enabled and (
        any(value is None for value in required_assets) or not raw_classes
    ):
        raise ValueError(
            "Enabled hailo requires all asset paths, model identity and raw_classes."
        )
    hailo = HailoConfig(
        enabled=hailo_enabled,
        hef_path=hef_path,
        postprocess_onnx_path=postprocess_onnx_path,
        output_mapping_path=output_mapping_path,
        model_version=model_version,
        hef_sha256=checksum,
        raw_classes=raw_classes,
        class_mapping=class_mapping,
        detection_threshold=detection_threshold,
        semantic_threshold=semantic_threshold,
        k0_threshold=k0_threshold,
        max_detections=max_detections,
    )

    return AppConfig(
        SCHEMA_VERSION,
        camera,
        geometry,
        recording,
        processing,
        hailo,
    )
