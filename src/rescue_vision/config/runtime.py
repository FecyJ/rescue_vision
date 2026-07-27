"""版本化运行配置及几何对象装配。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rescue_vision.communication.remote import RemoteAccessMode, RemoteRole
from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.mission import MissionConfig
from rescue_vision.perception.types import TargetClass
from rescue_vision.tracking import TrackingConfig
from rescue_vision.world import (
    RegionKind,
    StaticRegion,
    WorldModel,
    WorldModelConfig,
)

if TYPE_CHECKING:
    from rescue_vision.communication import (
        RemoteConnectionOptions,
        RemoteMessageConnection,
        RemoteTcpServer,
        UartLineChannel,
    )


SCHEMA_VERSION = 7


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
class UartConfig:
    enabled: bool
    device: str | None
    baudrate: int
    read_timeout_ms: float
    write_timeout_ms: float
    receive_queue_capacity: int
    max_line_bytes: int

    def build_channel(self) -> UartLineChannel | None:
        """按配置创建协议无关 UART 行通道；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        assert self.device is not None
        from rescue_vision.communication import UartLineChannel

        return UartLineChannel(
            device=self.device,
            baudrate=self.baudrate,
            read_timeout_s=self.read_timeout_ms / 1000.0,
            write_timeout_s=self.write_timeout_ms / 1000.0,
            receive_queue_capacity=self.receive_queue_capacity,
            max_line_bytes=self.max_line_bytes,
        )


@dataclass(frozen=True, slots=True)
class RemoteConfig:
    enabled: bool
    role: RemoteRole
    host: str
    port: int
    access_mode: RemoteAccessMode
    connect_timeout_ms: float
    io_timeout_ms: float
    control_queue_capacity: int
    observation_queue_capacity: int
    max_header_bytes: int
    max_payload_bytes: int

    def _connection_options(self) -> RemoteConnectionOptions:
        from rescue_vision.communication import RemoteConnectionOptions

        return RemoteConnectionOptions(
            io_timeout_s=self.io_timeout_ms / 1000.0,
            control_queue_capacity=self.control_queue_capacity,
            observation_queue_capacity=self.observation_queue_capacity,
            max_header_bytes=self.max_header_bytes,
            max_payload_bytes=self.max_payload_bytes,
        )

    def build_server(self) -> RemoteTcpServer | None:
        """创建但不打开树莓派 TCP 监听端点；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        if self.role is not RemoteRole.SERVER:
            raise RuntimeError(
                "remote.role must be 'server' to build a server endpoint."
            )
        from rescue_vision.communication import RemoteTcpServer

        return RemoteTcpServer(
            host=self.host,
            port=self.port,
            access_mode=self.access_mode,
            connection_options=self._connection_options(),
        )

    def connect_client(self) -> RemoteMessageConnection | None:
        """仓库参考客户端主动建立 TCP 连接；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        if self.role is not RemoteRole.CLIENT:
            raise RuntimeError(
                "remote.role must be 'client' to connect a client endpoint."
            )
        from rescue_vision.communication import connect_remote_client

        return connect_remote_client(
            host=self.host,
            port=self.port,
            connect_timeout_s=self.connect_timeout_ms / 1000.0,
            access_mode=self.access_mode,
            connection_options=self._connection_options(),
        )


@dataclass(frozen=True, slots=True)
class WorldRuntimeConfig:
    model: WorldModelConfig
    regions: tuple[StaticRegion, ...]

    def build_model(self) -> WorldModel:
        return WorldModel(self.model, self.regions)


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
    backend_score_threshold: float
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
            score_threshold=self.backend_score_threshold,
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
    uart: UartConfig
    remote: RemoteConfig
    tracking: TrackingConfig
    world: WorldRuntimeConfig
    mission: MissionConfig
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
    """从 YAML 加载 schema v7；缺项和未知字段均视为错误。"""

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
            "uart",
            "remote",
            "tracking",
            "world",
            "mission",
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

    uart_raw = _mapping(_required(root, "uart", "root"), "uart")
    _reject_unknown(
        uart_raw,
        {
            "enabled",
            "device",
            "baudrate",
            "read_timeout_ms",
            "write_timeout_ms",
            "receive_queue_capacity",
            "max_line_bytes",
        },
        "uart",
    )
    uart_enabled = _required(uart_raw, "enabled", "uart")
    if not isinstance(uart_enabled, bool):
        raise ValueError("uart.enabled must be a boolean.")
    uart_device_raw = _required(uart_raw, "device", "uart")
    uart_device = (
        None
        if uart_device_raw is None
        else _string(uart_device_raw, "uart.device")
    )
    if uart_enabled and uart_device is None:
        raise ValueError("Enabled UART requires uart.device.")
    uart = UartConfig(
        enabled=uart_enabled,
        device=uart_device,
        baudrate=_positive_int(
            _required(uart_raw, "baudrate", "uart"),
            "uart.baudrate",
        ),
        read_timeout_ms=_finite_float(
            _required(uart_raw, "read_timeout_ms", "uart"),
            "uart.read_timeout_ms",
            minimum=0.001,
        ),
        write_timeout_ms=_finite_float(
            _required(uart_raw, "write_timeout_ms", "uart"),
            "uart.write_timeout_ms",
            minimum=0.001,
        ),
        receive_queue_capacity=_positive_int(
            _required(uart_raw, "receive_queue_capacity", "uart"),
            "uart.receive_queue_capacity",
        ),
        max_line_bytes=_positive_int(
            _required(uart_raw, "max_line_bytes", "uart"),
            "uart.max_line_bytes",
        ),
    )

    remote_raw = _mapping(_required(root, "remote", "root"), "remote")
    _reject_unknown(
        remote_raw,
        {
            "enabled",
            "role",
            "host",
            "port",
            "access_mode",
            "connect_timeout_ms",
            "io_timeout_ms",
            "control_queue_capacity",
            "observation_queue_capacity",
            "max_header_bytes",
            "max_payload_bytes",
        },
        "remote",
    )
    remote_enabled = _required(remote_raw, "enabled", "remote")
    if not isinstance(remote_enabled, bool):
        raise ValueError("remote.enabled must be a boolean.")
    try:
        remote_role = RemoteRole(_required(remote_raw, "role", "remote"))
    except (TypeError, ValueError) as exc:
        raise ValueError("remote.role must be 'server' or 'client'.") from exc
    try:
        remote_access_mode = RemoteAccessMode(
            _required(remote_raw, "access_mode", "remote")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "remote.access_mode must be 'observe_only' or 'debug_control'."
        ) from exc
    remote_port = _positive_int(
        _required(remote_raw, "port", "remote"),
        "remote.port",
    )
    if remote_port > 65_535:
        raise ValueError("remote.port must be <= 65535.")
    remote = RemoteConfig(
        enabled=remote_enabled,
        role=remote_role,
        host=_string(_required(remote_raw, "host", "remote"), "remote.host"),
        port=remote_port,
        access_mode=remote_access_mode,
        connect_timeout_ms=_finite_float(
            _required(remote_raw, "connect_timeout_ms", "remote"),
            "remote.connect_timeout_ms",
            minimum=0.001,
        ),
        io_timeout_ms=_finite_float(
            _required(remote_raw, "io_timeout_ms", "remote"),
            "remote.io_timeout_ms",
            minimum=0.001,
        ),
        control_queue_capacity=_positive_int(
            _required(remote_raw, "control_queue_capacity", "remote"),
            "remote.control_queue_capacity",
        ),
        observation_queue_capacity=_positive_int(
            _required(remote_raw, "observation_queue_capacity", "remote"),
            "remote.observation_queue_capacity",
        ),
        max_header_bytes=_positive_int(
            _required(remote_raw, "max_header_bytes", "remote"),
            "remote.max_header_bytes",
        ),
        max_payload_bytes=_positive_int(
            _required(remote_raw, "max_payload_bytes", "remote"),
            "remote.max_payload_bytes",
        ),
    )

    tracking_raw = _mapping(
        _required(root, "tracking", "root"),
        "tracking",
    )
    _reject_unknown(
        tracking_raw,
        {
            "confirmation_hits",
            "max_association_ground_mm",
            "min_association_iou",
            "max_coast_ms",
            "confidence_decay_per_second",
            "min_confidence",
        },
        "tracking",
    )
    tracking = TrackingConfig(
        confirmation_hits=_positive_int(
            _required(tracking_raw, "confirmation_hits", "tracking"),
            "tracking.confirmation_hits",
        ),
        max_association_ground_mm=_finite_float(
            _required(
                tracking_raw,
                "max_association_ground_mm",
                "tracking",
            ),
            "tracking.max_association_ground_mm",
            minimum=0.001,
        ),
        min_association_iou=_threshold(
            _required(tracking_raw, "min_association_iou", "tracking"),
            "tracking.min_association_iou",
        ),
        max_coast_ms=_finite_float(
            _required(tracking_raw, "max_coast_ms", "tracking"),
            "tracking.max_coast_ms",
            minimum=0.001,
        ),
        confidence_decay_per_second=_finite_float(
            _required(
                tracking_raw,
                "confidence_decay_per_second",
                "tracking",
            ),
            "tracking.confidence_decay_per_second",
            minimum=0.001,
        ),
        min_confidence=_threshold(
            _required(tracking_raw, "min_confidence", "tracking"),
            "tracking.min_confidence",
        ),
    )

    world_raw = _mapping(_required(root, "world", "root"), "world")
    _reject_unknown(
        world_raw,
        {
            "max_visual_age_ms",
            "opponent_max_age_ms",
            "danger_confirm_threshold",
            "danger_suspect_threshold",
            "unknown_suspect_threshold",
            "regions",
        },
        "world",
    )
    world_model = WorldModelConfig(
        max_visual_age_ms=_finite_float(
            _required(world_raw, "max_visual_age_ms", "world"),
            "world.max_visual_age_ms",
            minimum=0.001,
        ),
        opponent_max_age_ms=_finite_float(
            _required(world_raw, "opponent_max_age_ms", "world"),
            "world.opponent_max_age_ms",
            minimum=0.001,
        ),
        danger_confirm_threshold=_threshold(
            _required(
                world_raw,
                "danger_confirm_threshold",
                "world",
            ),
            "world.danger_confirm_threshold",
        ),
        danger_suspect_threshold=_threshold(
            _required(
                world_raw,
                "danger_suspect_threshold",
                "world",
            ),
            "world.danger_suspect_threshold",
        ),
        unknown_suspect_threshold=_threshold(
            _required(
                world_raw,
                "unknown_suspect_threshold",
                "world",
            ),
            "world.unknown_suspect_threshold",
        ),
    )
    regions_value = _required(world_raw, "regions", "world")
    if not isinstance(regions_value, list):
        raise ValueError("world.regions must be a list.")
    regions: list[StaticRegion] = []
    for index, value in enumerate(regions_value):
        location = f"world.regions[{index}]"
        region_raw = _mapping(value, location)
        _reject_unknown(
            region_raw,
            {"region_id", "kind", "polygon_field_mm"},
            location,
        )
        try:
            kind = RegionKind(
                _string(
                    _required(region_raw, "kind", location),
                    f"{location}.kind",
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"{location}.kind must be field, own_material, own_injured "
                "or opponent_safe."
            ) from exc
        polygon_value = _required(
            region_raw,
            "polygon_field_mm",
            location,
        )
        if not isinstance(polygon_value, list):
            raise ValueError(f"{location}.polygon_field_mm must be a list.")
        polygon: list[FieldPoint] = []
        for point_index, point_value in enumerate(polygon_value):
            point_location = (
                f"{location}.polygon_field_mm[{point_index}]"
            )
            if not isinstance(point_value, list) or len(point_value) != 2:
                raise ValueError(f"{point_location} must be [x_mm, y_mm].")
            polygon.append(
                FieldPoint(
                    _finite_float(
                        point_value[0],
                        f"{point_location}[0]",
                        minimum=-float("inf"),
                    ),
                    _finite_float(
                        point_value[1],
                        f"{point_location}[1]",
                        minimum=-float("inf"),
                    ),
                )
            )
        regions.append(
            StaticRegion(
                region_id=_string(
                    _required(region_raw, "region_id", location),
                    f"{location}.region_id",
                ),
                kind=kind,
                polygon_field=tuple(polygon),
            )
        )
    world = WorldRuntimeConfig(world_model, tuple(regions))

    mission_raw = _mapping(
        _required(root, "mission", "root"),
        "mission",
    )
    _reject_unknown(
        mission_raw,
        {
            "match_duration_s",
            "no_motion_timeout_s",
            "opponent_contact_timeout_s",
            "danger_avoid_distance_mm",
            "target_priority",
        },
        "mission",
    )
    priority_value = _required(
        mission_raw,
        "target_priority",
        "mission",
    )
    if not isinstance(priority_value, list):
        raise ValueError("mission.target_priority must be a list.")
    try:
        target_priority = tuple(
            TargetClass(
                _string(
                    value,
                    f"mission.target_priority[{index}]",
                )
            )
            for index, value in enumerate(priority_value)
        )
    except ValueError as exc:
        raise ValueError(
            "mission.target_priority values must be green_supply, "
            "black_core or orange_injured."
        ) from exc
    mission = MissionConfig(
        match_duration_s=_finite_float(
            _required(mission_raw, "match_duration_s", "mission"),
            "mission.match_duration_s",
            minimum=0.001,
        ),
        no_motion_timeout_s=_finite_float(
            _required(
                mission_raw,
                "no_motion_timeout_s",
                "mission",
            ),
            "mission.no_motion_timeout_s",
            minimum=0.001,
        ),
        opponent_contact_timeout_s=_finite_float(
            _required(
                mission_raw,
                "opponent_contact_timeout_s",
                "mission",
            ),
            "mission.opponent_contact_timeout_s",
            minimum=0.001,
        ),
        danger_avoid_distance_mm=_finite_float(
            _required(
                mission_raw,
                "danger_avoid_distance_mm",
                "mission",
            ),
            "mission.danger_avoid_distance_mm",
            minimum=0.001,
        ),
        target_priority=target_priority,
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
            "backend_score_threshold",
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

    backend_score_threshold = _threshold(
        _required(hailo_raw, "backend_score_threshold", "hailo"),
        "hailo.backend_score_threshold",
    )
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
        ("hailo.backend_score_threshold", backend_score_threshold),
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
    if backend_score_threshold > detection_threshold:
        raise ValueError(
            "hailo.backend_score_threshold must be <= "
            f"hailo.detection_threshold, got {backend_score_threshold} > "
            f"{detection_threshold}."
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
        backend_score_threshold=backend_score_threshold,
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
        uart,
        remote,
        tracking,
        world,
        mission,
        hailo,
    )
