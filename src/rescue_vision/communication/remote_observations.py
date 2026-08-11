"""独立电脑端协议使用的严格观察消息契约。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from rescue_vision.communication.remote import (
    RemoteAccessMode,
    RemoteAttributeValue,
)
from rescue_vision.communication.remote_messages import (
    CaptureAction,
    HeadingReference,
    _decode_json_object,
    _finite_float,
    _identifier,
    _non_negative_int,
    _require_exact_keys,
)


class ImageCoordinateSystem(str, Enum):
    RAW_PIXEL = "raw_pixel"
    UNDISTORTED_PIXEL = "undistorted_pixel"


class TeamColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"


class VehicleMotionState(str, Enum):
    STOPPED = "stopped"
    MOVING = "moving"
    BRAKING = "braking"
    EMERGENCY_STOPPED = "emergency_stopped"
    UNKNOWN = "unknown"


class VehicleSafetyMode(str, Enum):
    FIRMWARE_WATCHDOG = "firmware_watchdog"
    SUPERVISED_PHYSICAL_STOP = "supervised_physical_stop"
    UNAVAILABLE = "unavailable"


class VehicleStopReason(str, Enum):
    NONE = "none"
    DEADMAN_RELEASE = "deadman_release"
    COMMAND_EXPIRED = "command_expired"
    REMOTE_DISCONNECTED = "remote_disconnected"
    UART_FAULT = "uart_fault"
    CONTROLLER_WATCHDOG = "controller_watchdog"
    EMERGENCY_STOP = "emergency_stop"
    CAMERA_FAULT = "camera_fault"
    APPLICATION_SHUTDOWN = "application_shutdown"
    UNKNOWN = "unknown"


class CaptureRecordingState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    FAULT = "fault"


class CaptureRequestResult(str, Enum):
    NONE = "none"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class CaptureStopReason(str, Enum):
    REQUESTED = "requested"
    DISK_FULL = "disk_full"
    WRITE_ERROR = "write_error"
    CAMERA_ERROR = "camera_error"
    APPLICATION_SHUTDOWN = "application_shutdown"
    UNKNOWN = "unknown"


def _positive_int(value: object, location: str) -> int:
    converted = _non_negative_int(value, location)
    if converted == 0:
        raise ValueError(f"{location} must be positive.")
    return converted


def _positive_float(value: object, location: str) -> float:
    converted = _finite_float(value, location)
    if converted <= 0.0:
        raise ValueError(f"{location} must be positive.")
    return converted


def _optional_positive_int(value: object, location: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, location)


def _optional_positive_float(value: object, location: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, location)


def _optional_finite_float(value: object, location: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, location)


def _optional_identifier(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, location)


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be a boolean.")
    return value


def _enum_value(enum_type: type[Enum], value: object, location: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown {location} {value!r}.") from exc


def _encode_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RemoteSessionStatus:
    """TCP 会话的权限、能力、限制和发布周期。"""

    session_id: str
    server_instance_id: str
    timestamp_ns: int
    access_mode: RemoteAccessMode
    motion_control_available: bool
    gripper_control_available: bool
    capture_control_available: bool
    video_stream_available: bool
    map_snapshot_available: bool
    vehicle_state_available: bool
    capture_status_available: bool
    target_heading_control_available: bool
    session_status_period_ms: int
    vehicle_state_period_ms: int | None
    map_snapshot_period_ms: int | None
    capture_status_period_ms: int | None
    video_nominal_fps: float | None
    max_linear_velocity_m_s: float | None
    max_angular_velocity_rad_s: float | None
    max_control_command_valid_for_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _identifier(self.session_id, "session_id"),
        )
        object.__setattr__(
            self,
            "server_instance_id",
            _identifier(self.server_instance_id, "server_instance_id"),
        )
        _non_negative_int(self.timestamp_ns, "timestamp_ns")
        if not isinstance(self.access_mode, RemoteAccessMode):
            raise ValueError("access_mode must be RemoteAccessMode.")
        for name in (
            "motion_control_available",
            "gripper_control_available",
            "capture_control_available",
            "video_stream_available",
            "map_snapshot_available",
            "vehicle_state_available",
            "capture_status_available",
            "target_heading_control_available",
        ):
            _boolean(getattr(self, name), name)
        _positive_int(self.session_status_period_ms, "session_status_period_ms")
        vehicle_period = _optional_positive_int(
            self.vehicle_state_period_ms,
            "vehicle_state_period_ms",
        )
        map_period = _optional_positive_int(
            self.map_snapshot_period_ms,
            "map_snapshot_period_ms",
        )
        capture_period = _optional_positive_int(
            self.capture_status_period_ms,
            "capture_status_period_ms",
        )
        video_fps = _optional_positive_float(
            self.video_nominal_fps,
            "video_nominal_fps",
        )
        linear_limit = _optional_positive_float(
            self.max_linear_velocity_m_s,
            "max_linear_velocity_m_s",
        )
        angular_limit = _optional_positive_float(
            self.max_angular_velocity_rad_s,
            "max_angular_velocity_rad_s",
        )
        if (
            isinstance(self.max_control_command_valid_for_ms, bool)
            or not isinstance(self.max_control_command_valid_for_ms, int)
            or not 1 <= self.max_control_command_valid_for_ms <= 5_000
        ):
            raise ValueError(
                "max_control_command_valid_for_ms must be in [1, 5000]."
            )
        for name, converted in (
            ("vehicle_state_period_ms", vehicle_period),
            ("map_snapshot_period_ms", map_period),
            ("capture_status_period_ms", capture_period),
            ("video_nominal_fps", video_fps),
            ("max_linear_velocity_m_s", linear_limit),
            ("max_angular_velocity_rad_s", angular_limit),
        ):
            object.__setattr__(self, name, converted)
        if self.access_mode is RemoteAccessMode.OBSERVE_ONLY and (
            self.motion_control_available
            or self.gripper_control_available
            or self.capture_control_available
        ):
            raise ValueError(
                "observe_only sessions cannot advertise control capabilities."
            )
        if self.target_heading_control_available and not self.motion_control_available:
            raise ValueError(
                "target heading requires motion_control_available."
            )
        if self.motion_control_available and (
            not self.video_stream_available
            or not self.vehicle_state_available
        ):
            raise ValueError(
                "motion control requires video and vehicle state."
            )
        if self.gripper_control_available and (
            not self.video_stream_available
            or not self.vehicle_state_available
        ):
            raise ValueError(
                "gripper control requires video and vehicle state."
            )
        if self.capture_control_available and (
            not self.video_stream_available
            or not self.capture_status_available
        ):
            raise ValueError(
                "capture control requires video and capture status."
            )
        if self.motion_control_available != (
            linear_limit is not None and angular_limit is not None
        ):
            raise ValueError(
                "motion limits must both be present exactly when motion control "
                "is available."
            )
        if self.vehicle_state_available != (vehicle_period is not None):
            raise ValueError(
                "vehicle_state_period_ms presence must match availability."
            )
        if self.map_snapshot_available != (map_period is not None):
            raise ValueError(
                "map_snapshot_period_ms presence must match availability."
            )
        if self.capture_status_available != (capture_period is not None):
            raise ValueError(
                "capture_status_period_ms presence must match availability."
            )
        if self.video_stream_available != (video_fps is not None):
            raise ValueError(
                "video_nominal_fps presence must match availability."
            )

    def to_payload(self) -> bytes:
        return _encode_json(
            {
                "session_id": self.session_id,
                "server_instance_id": self.server_instance_id,
                "timestamp_ns": self.timestamp_ns,
                "access_mode": self.access_mode.value,
                "capabilities": {
                    "motion_control": self.motion_control_available,
                    "gripper_control": self.gripper_control_available,
                    "capture_control": self.capture_control_available,
                    "video_stream": self.video_stream_available,
                    "map_snapshot": self.map_snapshot_available,
                    "vehicle_state": self.vehicle_state_available,
                    "capture_status": self.capture_status_available,
                    "target_heading_control": (
                        self.target_heading_control_available
                    ),
                },
                "periods": {
                    "session_status_ms": self.session_status_period_ms,
                    "vehicle_state_ms": self.vehicle_state_period_ms,
                    "map_snapshot_ms": self.map_snapshot_period_ms,
                    "capture_status_ms": self.capture_status_period_ms,
                    "video_nominal_fps": self.video_nominal_fps,
                },
                "limits": {
                    "max_linear_velocity_m_s": self.max_linear_velocity_m_s,
                    "max_angular_velocity_rad_s": self.max_angular_velocity_rad_s,
                    "max_control_command_valid_for_ms": (
                        self.max_control_command_valid_for_ms
                    ),
                },
            }
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> RemoteSessionStatus:
        document = _decode_json_object(payload, "RemoteSessionStatus")
        _require_exact_keys(
            document,
            {
                "session_id",
                "server_instance_id",
                "timestamp_ns",
                "access_mode",
                "capabilities",
                "periods",
                "limits",
            },
            "RemoteSessionStatus",
        )
        capabilities = document["capabilities"]
        periods = document["periods"]
        limits = document["limits"]
        if not isinstance(capabilities, dict):
            raise ValueError("capabilities must be an object.")
        if not isinstance(periods, dict):
            raise ValueError("periods must be an object.")
        if not isinstance(limits, dict):
            raise ValueError("limits must be an object.")
        _require_exact_keys(
            capabilities,
            {
                "motion_control",
                "gripper_control",
                "capture_control",
                "video_stream",
                "map_snapshot",
                "vehicle_state",
                "capture_status",
                "target_heading_control",
            },
            "RemoteSessionStatus.capabilities",
        )
        _require_exact_keys(
            periods,
            {
                "session_status_ms",
                "vehicle_state_ms",
                "map_snapshot_ms",
                "capture_status_ms",
                "video_nominal_fps",
            },
            "RemoteSessionStatus.periods",
        )
        _require_exact_keys(
            limits,
            {
                "max_linear_velocity_m_s",
                "max_angular_velocity_rad_s",
                "max_control_command_valid_for_ms",
            },
            "RemoteSessionStatus.limits",
        )
        return cls(
            session_id=document["session_id"],
            server_instance_id=document["server_instance_id"],
            timestamp_ns=document["timestamp_ns"],
            access_mode=_enum_value(
                RemoteAccessMode,
                document["access_mode"],
                "access_mode",
            ),
            motion_control_available=capabilities["motion_control"],
            gripper_control_available=capabilities["gripper_control"],
            capture_control_available=capabilities["capture_control"],
            video_stream_available=capabilities["video_stream"],
            map_snapshot_available=capabilities["map_snapshot"],
            vehicle_state_available=capabilities["vehicle_state"],
            capture_status_available=capabilities["capture_status"],
            target_heading_control_available=capabilities[
                "target_heading_control"
            ],
            session_status_period_ms=periods["session_status_ms"],
            vehicle_state_period_ms=periods["vehicle_state_ms"],
            map_snapshot_period_ms=periods["map_snapshot_ms"],
            capture_status_period_ms=periods["capture_status_ms"],
            video_nominal_fps=periods["video_nominal_fps"],
            max_linear_velocity_m_s=limits["max_linear_velocity_m_s"],
            max_angular_velocity_rad_s=limits["max_angular_velocity_rad_s"],
            max_control_command_valid_for_ms=limits[
                "max_control_command_valid_for_ms"
            ],
        )


@dataclass(frozen=True, slots=True)
class VideoFrameAttributes:
    """JPEG 视频帧的严格 header attributes。"""

    MAX_DIMENSION: ClassVar[int] = 16_384
    MAX_PIXELS: ClassVar[int] = 33_554_432

    frame_sequence: int
    timestamp_ns: int
    width: int
    height: int
    coordinate_system: ImageCoordinateSystem
    calibration_id: str | None

    def __post_init__(self) -> None:
        _non_negative_int(self.frame_sequence, "frame_sequence")
        _non_negative_int(self.timestamp_ns, "timestamp_ns")
        width = _positive_int(self.width, "width")
        height = _positive_int(self.height, "height")
        if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
            raise ValueError("video dimensions exceed 16384.")
        if width * height > self.MAX_PIXELS:
            raise ValueError("video pixel count exceeds 33554432.")
        if not isinstance(self.coordinate_system, ImageCoordinateSystem):
            raise ValueError("coordinate_system must be ImageCoordinateSystem.")
        calibration_id = self.calibration_id
        if self.coordinate_system is ImageCoordinateSystem.RAW_PIXEL:
            if calibration_id is not None:
                raise ValueError("raw_pixel video must not declare intrinsics.")
        elif not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError(
                "undistorted_pixel video requires a non-empty calibration_id."
            )

    def to_attributes(self) -> dict[str, RemoteAttributeValue]:
        return {
            "frame_sequence": self.frame_sequence,
            "timestamp_ns": self.timestamp_ns,
            "width": self.width,
            "height": self.height,
            "coordinate_system": self.coordinate_system.value,
            "calibration_id": self.calibration_id,
        }

    @classmethod
    def from_attributes(
        cls,
        attributes: Mapping[str, RemoteAttributeValue],
    ) -> VideoFrameAttributes:
        _require_exact_keys(
            attributes,
            {
                "frame_sequence",
                "timestamp_ns",
                "width",
                "height",
                "coordinate_system",
                "calibration_id",
            },
            "VideoFrameAttributes",
        )
        return cls(
            frame_sequence=attributes["frame_sequence"],
            timestamp_ns=attributes["timestamp_ns"],
            width=attributes["width"],
            height=attributes["height"],
            coordinate_system=_enum_value(
                ImageCoordinateSystem,
                attributes["coordinate_system"],
                "coordinate_system",
            ),
            calibration_id=attributes["calibration_id"],
        )


@dataclass(frozen=True, slots=True)
class MapSnapshotAttributes:
    """PNG 场地地图的像素到 FieldPoint 映射。"""

    MAX_DIMENSION: ClassVar[int] = 16_384
    MAX_PIXELS: ClassVar[int] = 33_554_432

    snapshot_sequence: int
    timestamp_ns: int
    width: int
    height: int
    field_min_x_mm: float
    field_max_x_mm: float
    field_min_y_mm: float
    field_max_y_mm: float
    team_color: TeamColor

    def __post_init__(self) -> None:
        _non_negative_int(self.snapshot_sequence, "snapshot_sequence")
        _non_negative_int(self.timestamp_ns, "timestamp_ns")
        width = _positive_int(self.width, "width")
        height = _positive_int(self.height, "height")
        if width < 2 or height < 2:
            raise ValueError("map width and height must both be at least 2.")
        if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
            raise ValueError("map dimensions exceed 16384.")
        if width * height > self.MAX_PIXELS:
            raise ValueError("map pixel count exceeds 33554432.")
        for name in (
            "field_min_x_mm",
            "field_max_x_mm",
            "field_min_y_mm",
            "field_max_y_mm",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.field_min_x_mm >= self.field_max_x_mm:
            raise ValueError("field_min_x_mm must be less than field_max_x_mm.")
        if self.field_min_y_mm >= self.field_max_y_mm:
            raise ValueError("field_min_y_mm must be less than field_max_y_mm.")
        if not isinstance(self.team_color, TeamColor):
            raise ValueError("team_color must be TeamColor.")

    def to_attributes(self) -> dict[str, RemoteAttributeValue]:
        return {
            "snapshot_sequence": self.snapshot_sequence,
            "timestamp_ns": self.timestamp_ns,
            "width": self.width,
            "height": self.height,
            "coordinate_system": "field_mm",
            "field_min_x_mm": self.field_min_x_mm,
            "field_max_x_mm": self.field_max_x_mm,
            "field_min_y_mm": self.field_min_y_mm,
            "field_max_y_mm": self.field_max_y_mm,
            "team_color": self.team_color.value,
        }

    @classmethod
    def from_attributes(
        cls,
        attributes: Mapping[str, RemoteAttributeValue],
    ) -> MapSnapshotAttributes:
        _require_exact_keys(
            attributes,
            {
                "snapshot_sequence",
                "timestamp_ns",
                "width",
                "height",
                "coordinate_system",
                "field_min_x_mm",
                "field_max_x_mm",
                "field_min_y_mm",
                "field_max_y_mm",
                "team_color",
            },
            "MapSnapshotAttributes",
        )
        if attributes["coordinate_system"] != "field_mm":
            raise ValueError("Map coordinate_system must be 'field_mm'.")
        return cls(
            snapshot_sequence=attributes["snapshot_sequence"],
            timestamp_ns=attributes["timestamp_ns"],
            width=attributes["width"],
            height=attributes["height"],
            field_min_x_mm=attributes["field_min_x_mm"],
            field_max_x_mm=attributes["field_max_x_mm"],
            field_min_y_mm=attributes["field_min_y_mm"],
            field_max_y_mm=attributes["field_max_y_mm"],
            team_color=_enum_value(
                TeamColor,
                attributes["team_color"],
                "team_color",
            ),
        )


@dataclass(frozen=True, slots=True)
class VehicleStateObservation:
    """车端 UART、运动、安全和朝向状态快照。"""

    state_sequence: int
    timestamp_ns: int
    control_ready: bool
    safety_mode: VehicleSafetyMode
    uart_connected: bool
    watchdog_armed: bool
    emergency_stop_latched: bool
    motion_state: VehicleMotionState
    stop_reason: VehicleStopReason
    controller_uptime_ms: int | None
    target_left_velocity_m_s: float | None
    target_right_velocity_m_s: float | None
    measured_left_velocity_m_s: float | None
    measured_right_velocity_m_s: float | None
    gripper_left_angle_deg: float | None
    gripper_right_angle_deg: float | None
    heading_rad: float | None
    heading_reference: HeadingReference | None
    last_received_motion_command_id: str | None
    last_applied_motion_command_id: str | None
    last_received_gripper_command_id: str | None
    last_applied_gripper_command_id: str | None

    def __post_init__(self) -> None:
        _non_negative_int(self.state_sequence, "state_sequence")
        _non_negative_int(self.timestamp_ns, "timestamp_ns")
        for name in (
            "control_ready",
            "uart_connected",
            "watchdog_armed",
            "emergency_stop_latched",
        ):
            _boolean(getattr(self, name), name)
        if not isinstance(self.safety_mode, VehicleSafetyMode):
            raise ValueError("safety_mode must be VehicleSafetyMode.")
        if not isinstance(self.motion_state, VehicleMotionState):
            raise ValueError("motion_state must be VehicleMotionState.")
        if not isinstance(self.stop_reason, VehicleStopReason):
            raise ValueError("stop_reason must be VehicleStopReason.")
        uptime = (
            None
            if self.controller_uptime_ms is None
            else _non_negative_int(
                self.controller_uptime_ms,
                "controller_uptime_ms",
            )
        )
        object.__setattr__(self, "controller_uptime_ms", uptime)
        for name in (
            "target_left_velocity_m_s",
            "target_right_velocity_m_s",
            "measured_left_velocity_m_s",
            "measured_right_velocity_m_s",
        ):
            object.__setattr__(
                self,
                name,
                _optional_finite_float(getattr(self, name), name),
            )
        for name in (
            "gripper_left_angle_deg",
            "gripper_right_angle_deg",
        ):
            angle = _optional_finite_float(getattr(self, name), name)
            if angle is not None and not 0.0 <= angle <= 180.0:
                raise ValueError(f"{name} must be in [0, 180].")
            object.__setattr__(self, name, angle)
        if (self.gripper_left_angle_deg is None) != (
            self.gripper_right_angle_deg is None
        ):
            raise ValueError(
                "gripper angle fields must both be present or null."
            )
        heading = _optional_finite_float(self.heading_rad, "heading_rad")
        if heading is not None and not -math.pi <= heading <= math.pi:
            raise ValueError("heading_rad must be in [-pi, pi].")
        object.__setattr__(self, "heading_rad", heading)
        if self.heading_reference is not None and not isinstance(
            self.heading_reference,
            HeadingReference,
        ):
            raise ValueError("heading_reference must be HeadingReference or None.")
        if (heading is None) != (self.heading_reference is None):
            raise ValueError(
                "heading_rad and heading_reference must both be present or null."
            )
        for name in (
            "last_received_motion_command_id",
            "last_applied_motion_command_id",
            "last_received_gripper_command_id",
            "last_applied_gripper_command_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_identifier(getattr(self, name), name),
            )
        if (
            self.safety_mode is VehicleSafetyMode.FIRMWARE_WATCHDOG
            and not self.watchdog_armed
        ):
            raise ValueError(
                "firmware_watchdog safety mode requires watchdog_armed=true."
            )
        if (
            self.safety_mode is not VehicleSafetyMode.FIRMWARE_WATCHDOG
            and self.watchdog_armed
        ):
            raise ValueError(
                "watchdog_armed=true requires firmware_watchdog safety mode."
            )
        if self.control_ready and (
            not self.uart_connected
            or self.safety_mode is VehicleSafetyMode.UNAVAILABLE
            or self.emergency_stop_latched
        ):
            raise ValueError(
                "control_ready requires UART, an available safety mode and no "
                "latched emergency stop."
            )
        if self.motion_state is VehicleMotionState.EMERGENCY_STOPPED and (
            not self.emergency_stop_latched
            or self.stop_reason is not VehicleStopReason.EMERGENCY_STOP
        ):
            raise ValueError(
                "emergency_stopped requires latched emergency stop reason."
            )
        if self.stop_reason is VehicleStopReason.NONE and self.motion_state in {
            VehicleMotionState.STOPPED,
            VehicleMotionState.EMERGENCY_STOPPED,
        }:
            raise ValueError("stopped states require an explicit stop_reason.")

    def to_payload(self) -> bytes:
        return _encode_json(
            {
                "state_sequence": self.state_sequence,
                "timestamp_ns": self.timestamp_ns,
                "control_ready": self.control_ready,
                "safety_mode": self.safety_mode.value,
                "uart_connected": self.uart_connected,
                "watchdog_armed": self.watchdog_armed,
                "emergency_stop_latched": self.emergency_stop_latched,
                "motion_state": self.motion_state.value,
                "stop_reason": self.stop_reason.value,
                "controller_uptime_ms": self.controller_uptime_ms,
                "target_left_velocity_m_s": self.target_left_velocity_m_s,
                "target_right_velocity_m_s": self.target_right_velocity_m_s,
                "measured_left_velocity_m_s": self.measured_left_velocity_m_s,
                "measured_right_velocity_m_s": self.measured_right_velocity_m_s,
                "gripper_left_angle_deg": self.gripper_left_angle_deg,
                "gripper_right_angle_deg": self.gripper_right_angle_deg,
                "heading_rad": self.heading_rad,
                "heading_reference": (
                    None
                    if self.heading_reference is None
                    else self.heading_reference.value
                ),
                "last_received_motion_command_id": (
                    self.last_received_motion_command_id
                ),
                "last_applied_motion_command_id": (
                    self.last_applied_motion_command_id
                ),
                "last_received_gripper_command_id": (
                    self.last_received_gripper_command_id
                ),
                "last_applied_gripper_command_id": (
                    self.last_applied_gripper_command_id
                ),
            }
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> VehicleStateObservation:
        document = _decode_json_object(payload, "VehicleStateObservation")
        expected = {
            "state_sequence",
            "timestamp_ns",
            "control_ready",
            "safety_mode",
            "uart_connected",
            "watchdog_armed",
            "emergency_stop_latched",
            "motion_state",
            "stop_reason",
            "controller_uptime_ms",
            "target_left_velocity_m_s",
            "target_right_velocity_m_s",
            "measured_left_velocity_m_s",
            "measured_right_velocity_m_s",
            "gripper_left_angle_deg",
            "gripper_right_angle_deg",
            "heading_rad",
            "heading_reference",
            "last_received_motion_command_id",
            "last_applied_motion_command_id",
            "last_received_gripper_command_id",
            "last_applied_gripper_command_id",
        }
        _require_exact_keys(document, expected, "VehicleStateObservation")
        heading_reference = document["heading_reference"]
        return cls(
            state_sequence=document["state_sequence"],
            timestamp_ns=document["timestamp_ns"],
            control_ready=document["control_ready"],
            safety_mode=_enum_value(
                VehicleSafetyMode,
                document["safety_mode"],
                "safety_mode",
            ),
            uart_connected=document["uart_connected"],
            watchdog_armed=document["watchdog_armed"],
            emergency_stop_latched=document["emergency_stop_latched"],
            motion_state=_enum_value(
                VehicleMotionState,
                document["motion_state"],
                "motion_state",
            ),
            stop_reason=_enum_value(
                VehicleStopReason,
                document["stop_reason"],
                "stop_reason",
            ),
            controller_uptime_ms=document["controller_uptime_ms"],
            target_left_velocity_m_s=document["target_left_velocity_m_s"],
            target_right_velocity_m_s=document["target_right_velocity_m_s"],
            measured_left_velocity_m_s=document["measured_left_velocity_m_s"],
            measured_right_velocity_m_s=document["measured_right_velocity_m_s"],
            gripper_left_angle_deg=document["gripper_left_angle_deg"],
            gripper_right_angle_deg=document["gripper_right_angle_deg"],
            heading_rad=document["heading_rad"],
            heading_reference=(
                None
                if heading_reference is None
                else _enum_value(
                    HeadingReference,
                    heading_reference,
                    "heading_reference",
                )
            ),
            last_received_motion_command_id=document[
                "last_received_motion_command_id"
            ],
            last_applied_motion_command_id=document[
                "last_applied_motion_command_id"
            ],
            last_received_gripper_command_id=document[
                "last_received_gripper_command_id"
            ],
            last_applied_gripper_command_id=document[
                "last_applied_gripper_command_id"
            ],
        )


@dataclass(frozen=True, slots=True)
class CaptureStatusObservation:
    """采集会话状态及最近请求结果。"""

    status_sequence: int
    timestamp_ns: int
    recording_state: CaptureRecordingState
    recording_id: str | None
    accepted_frames: int
    written_frames: int
    dropped_frames: int
    available_disk_bytes: int | None
    last_request_id: str | None
    last_request_action: CaptureAction | None
    last_request_result: CaptureRequestResult
    last_request_artifact_id: str | None
    stop_reason: CaptureStopReason | None
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        _non_negative_int(self.status_sequence, "status_sequence")
        _non_negative_int(self.timestamp_ns, "timestamp_ns")
        if not isinstance(self.recording_state, CaptureRecordingState):
            raise ValueError("recording_state must be CaptureRecordingState.")
        object.__setattr__(
            self,
            "recording_id",
            _optional_identifier(self.recording_id, "recording_id"),
        )
        for name in ("accepted_frames", "written_frames", "dropped_frames"):
            _non_negative_int(getattr(self, name), name)
        if self.written_frames > self.accepted_frames:
            raise ValueError("written_frames must not exceed accepted_frames.")
        disk = (
            None
            if self.available_disk_bytes is None
            else _non_negative_int(
                self.available_disk_bytes,
                "available_disk_bytes",
            )
        )
        object.__setattr__(self, "available_disk_bytes", disk)
        object.__setattr__(
            self,
            "last_request_id",
            _optional_identifier(self.last_request_id, "last_request_id"),
        )
        if self.last_request_action is not None and not isinstance(
            self.last_request_action,
            CaptureAction,
        ):
            raise ValueError("last_request_action must be CaptureAction or None.")
        if not isinstance(self.last_request_result, CaptureRequestResult):
            raise ValueError("last_request_result must be CaptureRequestResult.")
        object.__setattr__(
            self,
            "last_request_artifact_id",
            _optional_identifier(
                self.last_request_artifact_id,
                "last_request_artifact_id",
            ),
        )
        if self.stop_reason is not None and not isinstance(
            self.stop_reason,
            CaptureStopReason,
        ):
            raise ValueError("stop_reason must be CaptureStopReason or None.")
        for name in ("error_code", "error_message"):
            object.__setattr__(
                self,
                name,
                _optional_identifier(getattr(self, name), name),
            )
        request_fields_present = (
            self.last_request_id is not None
            and self.last_request_action is not None
        )
        if request_fields_present != (
            self.last_request_result is not CaptureRequestResult.NONE
        ):
            raise ValueError(
                "request ID/action must be present exactly when result is not none."
            )
        if (
            self.last_request_artifact_id is not None
            and self.last_request_result is not CaptureRequestResult.COMPLETED
        ):
            raise ValueError(
                "last_request_artifact_id requires a completed request."
            )
        if self.recording_state is CaptureRecordingState.RECORDING:
            if self.recording_id is None or self.stop_reason is not None:
                raise ValueError(
                    "recording state requires recording_id and no stop_reason."
                )
        elif self.recording_id is not None:
            raise ValueError("non-recording state must not declare recording_id.")
        requires_error = (
            self.recording_state is CaptureRecordingState.FAULT
            or self.last_request_result
            in {
                CaptureRequestResult.REJECTED,
                CaptureRequestResult.FAILED,
            }
        )
        has_error = self.error_code is not None and self.error_message is not None
        if requires_error != has_error:
            raise ValueError(
                "error code and message must both be present exactly for a "
                "fault or rejected/failed request."
            )

    def to_payload(self) -> bytes:
        return _encode_json(
            {
                "status_sequence": self.status_sequence,
                "timestamp_ns": self.timestamp_ns,
                "recording_state": self.recording_state.value,
                "recording_id": self.recording_id,
                "accepted_frames": self.accepted_frames,
                "written_frames": self.written_frames,
                "dropped_frames": self.dropped_frames,
                "available_disk_bytes": self.available_disk_bytes,
                "last_request_id": self.last_request_id,
                "last_request_action": (
                    None
                    if self.last_request_action is None
                    else self.last_request_action.value
                ),
                "last_request_result": self.last_request_result.value,
                "last_request_artifact_id": self.last_request_artifact_id,
                "stop_reason": (
                    None if self.stop_reason is None else self.stop_reason.value
                ),
                "error_code": self.error_code,
                "error_message": self.error_message,
            }
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> CaptureStatusObservation:
        document = _decode_json_object(payload, "CaptureStatusObservation")
        _require_exact_keys(
            document,
            {
                "status_sequence",
                "timestamp_ns",
                "recording_state",
                "recording_id",
                "accepted_frames",
                "written_frames",
                "dropped_frames",
                "available_disk_bytes",
                "last_request_id",
                "last_request_action",
                "last_request_result",
                "last_request_artifact_id",
                "stop_reason",
                "error_code",
                "error_message",
            },
            "CaptureStatusObservation",
        )
        stop_reason = document["stop_reason"]
        last_request_action = document["last_request_action"]
        return cls(
            status_sequence=document["status_sequence"],
            timestamp_ns=document["timestamp_ns"],
            recording_state=_enum_value(
                CaptureRecordingState,
                document["recording_state"],
                "recording_state",
            ),
            recording_id=document["recording_id"],
            accepted_frames=document["accepted_frames"],
            written_frames=document["written_frames"],
            dropped_frames=document["dropped_frames"],
            available_disk_bytes=document["available_disk_bytes"],
            last_request_id=document["last_request_id"],
            last_request_action=(
                None
                if last_request_action is None
                else _enum_value(
                    CaptureAction,
                    last_request_action,
                    "last_request_action",
                )
            ),
            last_request_result=_enum_value(
                CaptureRequestResult,
                document["last_request_result"],
                "last_request_result",
            ),
            last_request_artifact_id=document["last_request_artifact_id"],
            stop_reason=(
                None
                if stop_reason is None
                else _enum_value(
                    CaptureStopReason,
                    stop_reason,
                    "stop_reason",
                )
            ),
            error_code=document["error_code"],
            error_message=document["error_message"],
        )
