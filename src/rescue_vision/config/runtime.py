"""运行配置及几何对象装配。"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rescue_vision.perception.gripper_color import GripperColorConfig
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.config.match_cc import MatchCCRuntimeConfig, parse_match_cc_config

from rescue_vision.communication.remote import RemoteAccessMode, RemoteRole
from rescue_vision.communication.remote_messages import RemoteTopic
from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossLocalizerConfig,
    FieldPose2D,
    FusionConfig,
    ImuFrameCalibration,
    OdometryCalibration,
    SafeZoneCornerLocalizerConfig,
    StaticLandmarkTrackingConfig,
)
from rescue_vision.mission import MissionConfig
from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    HsvColorClassifierConfig,
    HsvRange,
    POSE_MODEL_CLASSES,
    TargetClass,
)
from rescue_vision.perception.detector import (
    CenterCrossRefinementConfig,
    SafeZoneColorConfig,
)
from rescue_vision.perception.target_ground_geometry import (
    BoxTargetGeometry,
    RegularTetrahedronTargetGeometry,
    TargetGeometry,
    TargetGeometryShape,
    TargetGroundGeometryConfig,
)
from rescue_vision.tracking import TrackingConfig
from rescue_vision.world import (
    CenterCrossRay,
    CenterCrossTerminal,
    CenterLineTerminalKind,
    PhysicalRegionKind,
    PhysicalStaticRegion,
    RegionKind,
    StaticCenterCross,
    StaticFieldMap,
    StaticSafeZoneLandmarks,
    StaticRegion,
    TeamColor,
    WorldModel,
    WorldModelConfig,
    default_static_field_map,
)

if TYPE_CHECKING:
    from rescue_vision.communication import (
        RemoteConnectionOptions,
        RemoteMessageConnection,
        RemoteTcpServer,
        UartFrameChannel,
    )
    from rescue_vision.motion import (
        GripperCalibration,
        MotionController,
        RemoteGripperExecutor,
        RemoteMotionExecutor,
    )
    from rescue_vision.localization import CenterCrossLocalizer, OdometryImuFusion
    from rescue_vision.perception import TargetPoseDetector


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


def _nonnegative_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{location} must be a non-negative integer, got {value!r}."
        )
    return value


def _signed_angular_velocity(value: object, location: str) -> float:
    """带符号角速度：左转为正，右转为负，符号即方向。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number, got {value!r}.")
    converted = float(value)
    if not math.isfinite(converted) or abs(converted) < 0.001:
        raise ValueError(
            f"{location} must be a finite angular velocity with |value| >= "
            f"0.001 rad/s (positive means left turn), got {value!r}."
        )
    return converted


def _finite_float(value: object, location: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number, got {value!r}.")
    converted = float(value)
    if not (converted >= minimum and converted < float("inf")):
        raise ValueError(
            f"{location} must be finite and >= {minimum}, got {value!r}."
        )
    return converted


def _optional_servo_angle(value: object, location: str) -> float | None:
    if value is None:
        return None
    angle = _finite_float(value, location, minimum=0.0)
    if angle > 180.0:
        raise ValueError(f"{location} must be <= 180, got {value!r}.")
    return angle


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


def _hsv_triplet(value: object, location: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(component, bool)
            or not isinstance(component, int)
            for component in value
        )
    ):
        raise ValueError(f"{location} must be an integer [H, S, V] list.")
    return (value[0], value[1], value[2])


def _float_vector3(value: object, location: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{location} must be a list of three finite numbers.")
    return tuple(
        _finite_float(item, f"{location}[{index}]", minimum=-float("inf"))
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _float_matrix3(
    value: object, location: str
) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{location} must be a 3x3 list of finite numbers.")
    rows = tuple(
        _float_vector3(row, f"{location}[{index}]")
        for index, row in enumerate(value)
    )
    return rows


def _hsv_ranges(value: object, location: str) -> tuple[HsvRange, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list.")
    ranges: list[HsvRange] = []
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        item_raw = _mapping(item, item_location)
        _reject_unknown(item_raw, {"lower", "upper"}, item_location)
        ranges.append(
            HsvRange(
                lower=_hsv_triplet(
                    _required(item_raw, "lower", item_location),
                    f"{item_location}.lower",
                ),
                upper=_hsv_triplet(
                    _required(item_raw, "upper", item_location),
                    f"{item_location}.upper",
                ),
            )
        )
    return tuple(ranges)


def _target_geometry(value: object, location: str) -> TargetGeometry:
    raw = _mapping(value, location)
    shape_value = _string(
        _required(raw, "shape", location),
        f"{location}.shape",
    )
    try:
        shape = TargetGeometryShape(shape_value)
    except ValueError as exc:
        raise ValueError(
            f"{location}.shape must be 'box' or "
            "'regular_tetrahedron'."
        ) from exc
    if shape is TargetGeometryShape.BOX:
        _reject_unknown(
            raw,
            {"shape", "length_mm", "width_mm", "height_mm"},
            location,
        )
        return BoxTargetGeometry(
            length_mm=_finite_float(
                _required(raw, "length_mm", location),
                f"{location}.length_mm",
                minimum=0.001,
            ),
            width_mm=_finite_float(
                _required(raw, "width_mm", location),
                f"{location}.width_mm",
                minimum=0.001,
            ),
            height_mm=_finite_float(
                _required(raw, "height_mm", location),
                f"{location}.height_mm",
                minimum=0.001,
            ),
        )
    _reject_unknown(raw, {"shape", "edge_mm"}, location)
    return RegularTetrahedronTargetGeometry(
        edge_mm=_finite_float(
            _required(raw, "edge_mm", location),
            f"{location}.edge_mm",
            minimum=0.001,
        )
    )


def _merge_defaults(
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """递归合并配置默认值；未知键保留给相邻严格校验报告。"""

    merged: dict[str, Any] = {}
    for key, default_value in defaults.items():
        if key not in overrides:
            merged[key] = default_value
            continue
        override_value = overrides[key]
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            merged[key] = _merge_defaults(default_value, override_value)
        else:
            merged[key] = override_value
    for key in overrides.keys() - defaults.keys():
        merged[key] = overrides[key]
    return merged


def _perception_defaults() -> dict[str, Any]:
    """返回与示例配置一致且默认关闭目标地面几何的感知参数。"""

    return {
        "gripper_color": {
            "enabled": True,
            "polygon_normalized": [[0.47, 0.85], [0.53, 0.85], [0.57, 0.97], [0.43, 0.97]],
            "min_component_fraction": 0.03,
            "black_min_thickness_fraction": 0.12,
            "shadow_min_value": 30,
            "orange_bbox_min_color_fraction": 0.15,
            "orange_distinct_max_bbox_iou": 0.20,
            "orange_distinct_min_k0_distance_px": 20.0,
        },
        "detection_threshold": 0.25,
        "k0_threshold": 0.50,
        "center_cross_refinement": {
            "enabled": True,
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 16,
            "min_line_length_px": 18.0,
            "max_line_gap_px": 12.0,
            "max_intersection_distance_px": 18.0,
            "min_axis_angle_deg": 65.0,
        },
        "safe_zone_color": {
            "enabled": False,
            "red_hsv_ranges": [],
            "blue_hsv_ranges": [],
            "min_fraction": 0.08,
            "min_margin": 0.03,
        },
        "color_classifier": {
            "ranges": {
                "green_supply": [
                    {"lower": [35, 70, 71], "upper": [84, 255, 255]},
                ],
                "black_core": [
                    {"lower": [0, 0, 0], "upper": [179, 255, 70]},
                ],
                "orange_injured": [
                    {"lower": [0, 90, 80], "upper": [20, 255, 255]},
                    {"lower": [170, 90, 80], "upper": [179, 255, 255]},
                ],
                "blue_danger": [
                    {"lower": [85, 50, 71], "upper": [110, 255, 255]},
                ],
            },
            "min_color_fraction": 0.15,
            "min_color_dominance": 0.70,
            "min_dominance_margin": 0.20,
            "morphology_kernel_size": 3,
            "open_iterations": 1,
            "close_iterations": 1,
            "min_component_area_fraction": 0.002,
        },
        "target_ground_geometry": {
            "enabled": False,
            "objects": {
                "green_supply": {
                    "shape": "box",
                    "length_mm": 40.0,
                    "width_mm": 40.0,
                    "height_mm": 40.0,
                },
                "black_core": {
                    "shape": "regular_tetrahedron",
                    "edge_mm": 40.0,
                },
                "orange_injured": {
                    "shape": "box",
                    "length_mm": 80.0,
                    "width_mm": 40.0,
                    "height_mm": 40.0,
                },
                "blue_danger": {
                    "shape": "box",
                    "length_mm": 40.0,
                    "width_mm": 40.0,
                    "height_mm": 40.0,
                },
            },
            "fitting": {
                "coarse_center_step_mm": 5.0,
                "coarse_yaw_step_deg": 10.0,
                "refine_center_step_mm": 1.0,
                "refine_center_radius_mm": 6.0,
                "refine_top_candidates": 3,
                "refine_yaw_step_deg": 2.0,
                "refine_yaw_radius_deg": 10.0,
                "search_radius_margin_mm": 8.0,
                "silhouette_weight": 0.65,
                "contour_weight": 0.25,
                "contact_weight": 0.10,
                "contour_distance_scale_px": 4.0,
                "contact_distance_scale_px": 6.0,
                "max_contact_residual_px": 12.0,
                "ambiguity_score_delta": 0.03,
                "max_center_uncertainty_mm": 12.0,
                "min_fit_score": 0.60,
                "min_silhouette_iou": 0.45,
            },
        },
    }


def _localization_defaults() -> dict[str, Any]:
    return {
        "enabled": False,
        "ray_min_forward_distance_mm": 500.0,
        "ray_max_forward_distance_mm": 1800.0,
        "ray_max_lateral_distance_mm": 120.0,
        "ray_angle_tolerance_deg": 10.0,
        "min_anchor_confidence": 0.25,
        "max_prior_heading_innovation_deg": 20.0,
        "position_uncertainty_floor_mm": 20.0,
        "heading_uncertainty_floor_deg": 3.0,
        "static_landmarks": {
            "max_track_age_ms": 250.0,
            "max_prior_position_uncertainty_mm": 300.0,
            "max_prior_heading_uncertainty_deg": 15.0,
            "cross_base_radius_mm": 180.0,
            "safe_zone_base_margin_mm": 120.0,
            "confirmation_hits": 2,
            "max_confirmation_age_ms": 150.0,
            "confirmation_ground_tolerance_mm": 120.0,
            "confirmation_axis_tolerance_deg": 20.0,
        },
        "safe_zone_corners": {
            "max_observation_age_ms": 250.0,
            "min_baseline_mm": 150.0,
            "max_k0_corner_distance_error_mm": 80.0,
            "max_fit_residual_mm": 80.0,
            "position_uncertainty_floor_mm": 30.0,
            "heading_uncertainty_floor_deg": 3.0,
        },
        "fusion": {
            "enabled": False,
            "initial_pose": {
                "x_mm": 0.0,
                "y_mm": 0.0,
                "heading_deg": 0.0,
                "position_uncertainty_mm": 100.0,
                "heading_uncertainty_deg": 10.0,
                "confidence": 0.5,
            },
            "imu_calibration": {
                "reference_temperature_c": 25.0,
                "gyro_bias_rad_s": [0.0, 0.0, 0.0],
                "gyro_bias_temperature_coefficient_rad_s_per_c": [0.0, 0.0, 0.0],
                "gyro_cross_axis_scale": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "accel_bias_mm_s2": [0.0, 0.0, 0.0],
                "accel_bias_temperature_coefficient_mm_s2_per_c": [
                    0.0,
                    0.0,
                    0.0,
                ],
                "accel_cross_axis_scale": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "sensor_to_robot_rotation": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "encoder_distance_noise_fraction": 0.02,
            "encoder_heading_noise_std_deg": 1.0,
            "gyro_noise_std_rad_s": 0.03,
            "gyro_bias_random_walk_std_rad_s_per_sqrt_s": 0.001,
            "stationary_gyro_noise_std_rad_s": 0.01,
            "stationary_encoder_delta_count": 0,
            "allow_wheel_only": True,
            "wheel_only_covariance_scale": 4.0,
            "dropped_sample_covariance_scale": 3.0,
            "max_interpolated_overrun_samples": 1,
            "max_sample_interval_ms": 50.0,
            "max_telemetry_age_ms": 100.0,
            "max_encoder_speed_mm_s": 1000.0,
            "max_visual_alignment_error_ms": 30.0,
            "visual_innovation_gate": 16.27,
            "history_duration_ms": 1000.0,
            "max_tilt_deg": 20.0,
            "impact_accel_threshold_mm_s2": 4000.0,
        },
    }


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
    read_timeout_ms: float
    write_timeout_ms: float
    receive_queue_capacity: int

    def build_channel(self) -> UartFrameChannel | None:
        """按配置创建协议无关 UART COBS 帧通道；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        assert self.device is not None
        from rescue_vision.communication import UartFrameChannel
        from rescue_vision.motion.protocol import (
            MAX_DECODED_FRAME_BYTES,
            STM32_UART_BAUDRATE,
        )

        return UartFrameChannel(
            device=self.device,
            baudrate=STM32_UART_BAUDRATE,
            read_timeout_s=self.read_timeout_ms / 1000.0,
            write_timeout_s=self.write_timeout_ms / 1000.0,
            receive_queue_capacity=self.receive_queue_capacity,
            max_frame_bytes=MAX_DECODED_FRAME_BYTES,
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
            non_actuating_control_topics=(RemoteTopic.VIDEO_MODE.value,),
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
class GripperRuntimeConfig:
    enabled: bool
    open_left_angle_deg: float | None
    open_right_angle_deg: float | None
    closed_left_angle_deg: float | None
    closed_right_angle_deg: float | None
    transport_left_angle_deg: float | None
    transport_right_angle_deg: float | None
    full_travel_time_s: float | None
    angle_sum_deg: float | None

    def build_calibration(self) -> GripperCalibration | None:
        """创建连续夹爪控制标定；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        assert self.open_left_angle_deg is not None
        assert self.open_right_angle_deg is not None
        assert self.closed_left_angle_deg is not None
        assert self.closed_right_angle_deg is not None
        assert self.transport_left_angle_deg is not None
        assert self.transport_right_angle_deg is not None
        assert self.full_travel_time_s is not None
        assert self.angle_sum_deg is not None
        from rescue_vision.motion import GripperCalibration

        return GripperCalibration(
            open_left_angle_deg=self.open_left_angle_deg,
            open_right_angle_deg=self.open_right_angle_deg,
            closed_left_angle_deg=self.closed_left_angle_deg,
            closed_right_angle_deg=self.closed_right_angle_deg,
            full_travel_time_s=self.full_travel_time_s,
            angle_sum_deg=self.angle_sum_deg,
            transport_left_angle_deg=self.transport_left_angle_deg,
            transport_right_angle_deg=self.transport_right_angle_deg,
        )


@dataclass(frozen=True, slots=True)
class OdometryRuntimeConfig:
    enabled: bool
    encoder_counts_per_revolution: int | None
    left_wheel_radius_mm: float | None
    right_wheel_radius_mm: float | None
    gyro_z_sign: int
    # 定距里程计允许的连续 SAMPLE_OVERRUN 帧数；None 表示不因此中止，仅保留计数。
    max_consecutive_overrun_samples: int | None = 1

    def build_calibration(self) -> OdometryCalibration | None:
        if not self.enabled:
            return None
        assert self.encoder_counts_per_revolution is not None
        assert self.left_wheel_radius_mm is not None
        assert self.right_wheel_radius_mm is not None
        return OdometryCalibration(
            self.encoder_counts_per_revolution,
            self.left_wheel_radius_mm,
            self.right_wheel_radius_mm,
            self.gyro_z_sign,
        )


@dataclass(frozen=True, slots=True)
class ClusterBreakupRuntimeConfig:
    """固定出发姿态下的中心目标团解团测试参数。"""

    enabled: bool
    departure_distance_m: float
    departure_speed_m_s: float
    search_angular_velocity_rad_s: float
    search_timeout_s: float
    cluster_min_detections: int
    cluster_group_gap_ratio: float
    center_tolerance_ratio: float
    center_confirm_frames: int
    center_kp_rad_s: float
    center_max_angular_velocity_rad_s: float
    approach_speed_m_s: float
    gripper_open_distance_mm: float
    breakup_speed_m_s: float
    breakup_distance_m: float
    gripper_open_retreat_distance_m: float
    retreat_speed_m_s: float
    retreat_distance_m: float
    scan_green_angular_velocity_rad_s: float
    green_confirm_frames: int
    target_loss_timeout_ms: float
    motion_phase_timeout_s: float
    post_breakup_retreat_enabled: bool = True
    # CENTER_CLUSTER 的横向误差 EMA 系数；较小值抑制单帧检测抖动。
    center_error_filter_alpha: float = 0.35
    # 反向转向相对居中门限的附加滞回区，单位为图像半幅比例。
    center_reverse_deadband_ratio: float = 0.03
    # 反向转向必须连续满足的观测帧数。
    center_reverse_confirm_frames: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if not isinstance(self.post_breakup_retreat_enabled, bool):
            raise ValueError("post_breakup_retreat_enabled must be a boolean.")
        for name in (
            "departure_distance_m",
            "departure_speed_m_s",
            "search_timeout_s",
            "center_tolerance_ratio",
            "center_kp_rad_s",
            "center_max_angular_velocity_rad_s",
            "approach_speed_m_s",
            "gripper_open_distance_mm",
            "breakup_speed_m_s",
            "breakup_distance_m",
            "gripper_open_retreat_distance_m",
            "retreat_speed_m_s",
            "retreat_distance_m",
            "target_loss_timeout_ms",
            "motion_phase_timeout_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "search_angular_velocity_rad_s",
            "scan_green_angular_velocity_rad_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or abs(value) < 0.001:
                raise ValueError(
                    f"{name} must be a finite angular velocity with "
                    f"|value| >= 0.001 rad/s (positive means left turn), "
                    f"got {value!r}."
                )
        gap_ratio = float(self.cluster_group_gap_ratio)
        if not math.isfinite(gap_ratio) or not 0.0 <= gap_ratio <= 1.0:
            raise ValueError(
                "cluster_group_gap_ratio must be finite and within [0, 1], "
                f"got {self.cluster_group_gap_ratio!r}."
            )
        if self.center_tolerance_ratio > 1.0:
            raise ValueError("center_tolerance_ratio must be <= 1.0.")
        alpha = float(self.center_error_filter_alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError(
                "center_error_filter_alpha must be finite and in (0, 1]."
            )
        reverse_deadband = float(self.center_reverse_deadband_ratio)
        if not math.isfinite(reverse_deadband) or reverse_deadband < 0.0:
            raise ValueError(
                "center_reverse_deadband_ratio must be finite and non-negative."
            )
        if self.center_tolerance_ratio + reverse_deadband > 1.0:
            raise ValueError(
                "center_tolerance_ratio + center_reverse_deadband_ratio must be <= 1.0."
            )
        for name in (
            "cluster_min_detections",
            "center_confirm_frames",
            "green_confirm_frames",
            "center_reverse_confirm_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")




@dataclass(frozen=True, slots=True)
class NBOpeningTurn:
    """无解团开场的相对原地转向动作。"""

    angle_rad: float
    angular_velocity_rad_s: float

    def __post_init__(self) -> None:
        angle = _finite_float(
            self.angle_rad,
            "nb_opening_actions[].angle_rad",
            minimum=-float("inf"),
        )
        if abs(angle) < 0.001:
            raise ValueError(
                "nb_opening_actions[].angle_rad must have absolute value >= 0.001."
            )
        speed = _finite_float(
            self.angular_velocity_rad_s,
            "nb_opening_actions[].angular_velocity_rad_s",
            minimum=0.001,
        )
        object.__setattr__(self, "angle_rad", angle)
        object.__setattr__(self, "angular_velocity_rad_s", speed)


@dataclass(frozen=True, slots=True)
class NBOpeningStraight:
    """无解团开场的相对直行动作；负距离表示倒车。"""

    distance_m: float
    speed_m_s: float

    def __post_init__(self) -> None:
        distance = _finite_float(
            self.distance_m,
            "nb_opening_actions[].distance_m",
            minimum=-float("inf"),
        )
        if abs(distance) < 0.001:
            raise ValueError(
                "nb_opening_actions[].distance_m must have absolute value >= 0.001."
            )
        speed = _finite_float(
            self.speed_m_s,
            "nb_opening_actions[].speed_m_s",
            minimum=0.001,
        )
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "speed_m_s", speed)


NBOpeningAction = NBOpeningTurn | NBOpeningStraight


_DEFAULT_NB_OPENING_ACTIONS: tuple[NBOpeningAction, ...] = (
    # Starting from (1350, 1350) with heading -90 degrees, these values
    # reproduce the previous (100, 800) -> (100, -900) -> (100, 0) route.
    NBOpeningTurn(-1.1562894522101108, 0.5),
    NBOpeningStraight(1.3656500283747664, 0.5),
    NBOpeningTurn(1.1562894522101108, 0.5),
    NBOpeningStraight(1.7, 0.5),
    NBOpeningStraight(-0.9, 0.5),
)


@dataclass(frozen=True, slots=True)
class MatchRuntimeConfig:
    """固定启动、解团循环和绿色物资转运的 正式策略参数。"""

    enabled: bool
    robot_footprint_radius_mm: float = 160.0
    safety_margin_mm: float = 80.0
    startup_turn_angle_rad: float = math.radians(45.0)
    startup_turn_angular_velocity_rad_s: float = -0.30
    startup_turn_settle_time_s: float = 0.5
    startup_forward_distance_m: float = 1.2
    startup_forward_speed_m_s: float = 0.12
    startup_forward_settle_time_s: float = 0.5
    startup_straight_pid_kp_rad_s_per_m_s: float = 0.0
    startup_straight_pid_ki_rad_s_per_m: float = 0.0
    startup_straight_pid_kd_rad_s_per_m_s2: float = 0.0
    startup_straight_pid_integral_limit: float = 0.10
    startup_straight_pid_deadband_m_s: float = 0.0
    startup_straight_pid_max_angular_velocity_rad_s: float = 0.12
    cluster_search_angular_velocity_rad_s: float = -0.30
    cluster_search_empty_angular_velocity_rad_s: float = -0.45
    cluster_search_sweep_angle_rad: float = 2.0 * math.pi
    cluster_min_detections: int = 2
    cluster_group_ground_mm: float = 250.0
    cluster_breakup_standoff_mm: float = 260.0
    cluster_approach_speed_m_s: float = 0.12
    cluster_align_tolerance_rad: float = 0.10
    cluster_align_hold_ms: float = 500.0
    cluster_align_kp_rad_s: float = 1.0
    cluster_align_max_angular_velocity_rad_s: float = 0.35
    cluster_relocate_distance_m: float = 0.30
    cluster_relocate_speed_m_s: float = 0.10
    # 正式流程 专用：搜索目标团时，可靠的单个绿色普通物资可以抢占解团流程。
    # 正式运行配置显式开启；纯逻辑调用方可按场景关闭。
    opportunistic_single_green_enabled: bool = False
    # 目标周围该半径内不能有其它新鲜轨迹，单位 mm；需结合目标尺寸和
    # GroundPoint 实测误差现场复核。
    opportunistic_single_green_clearance_mm: float = 120.0
    # 旧纯逻辑夹具的构造兼容字段；正式运行由 near_field_grasp.max_range_mm
    # 统一控制近场接管，YAML 不再接受该字段。
    opportunistic_single_green_realign_standoff_mm: float = 350.0
    # match 闭爪定距解团的接触与路径校验参数。
    breakup_min_penetration_mm: float = 10.0
    breakup_retreat_clearance_mm: float = 20.0
    breakup_push_margin_mm: float = 30.0
    breakup_braking_margin_mm: float = 20.0
    breakup_deceleration_m_s2: float = 0.8
    # 同一物理接触团允许的推进尝试次数，取值范围 [1, 4]；重复射线
    # 只有实际几何改变、可产生更深的接触时才允许重试。
    breakup_max_attempts: int = 2
    # 当前帧接触核心的连续有效感知帧数；正式解团不再使用近场抓取确认。
    breakup_confirmation_frames: int = 3
    # 停稳后仍选不出任何合法接触计划时的重观测窗口，单位 ms。重复观测只有
    # 在新一帧检出新的可接触成员时才可能改变结果，所以只等一到两帧，不占满
    # 按确认帧数计的完整确认预算。
    breakup_no_plan_reobserve_ms: float = 1_000.0
    # 误夹释放后原团短解团；速度和制动复用普通解团，航向相对释放姿态。
    misgrasp_breakup_forward_distance_m: float = 0.30
    misgrasp_breakup_backward_distance_m: float = 0.08
    misgrasp_breakup_max_heading_change_rad: float = math.radians(10.0)
    breakup_forward_distance_m: float = 0.5
    breakup_forward_speed_m_s: float = 0.40
    breakup_backward_distance_m: float = 0.3
    breakup_backward_speed_m_s: float = 0.08
    # 正式流程解团阶段的临时车体加减速度；None 分别继承 motion 对应值。
    breakup_max_linear_acceleration_m_s2: float | None = None
    breakup_max_linear_deceleration_m_s2: float | None = None
    breakup_max_angular_acceleration_rad_s2: float | None = None
    breakup_max_angular_deceleration_rad_s2: float | None = None
    close_gripper_spin_angular_velocity_rad_s: float = -0.30
    spin_angle_rad: float = 2.0 * math.pi
    green_path_half_width_mm: float = 50.0
    green_max_age_ms: float = 500.0
    green_align_hold_ms: float = 500.0
    green_alignment_tolerance_mm: float = 10.0
    green_alignment_hysteresis_mm: float = 5.0
    grasp_task_timeout_ms: float = 20_000.0
    green_alignment_timeout_ms: float = 8_000.0
    green_alignment_stable_frames: int = 3
    green_alignment_kp_rad_s: float = 1.0
    green_alignment_max_angular_velocity_rad_s: float = 0.35
    # 绿色目标及近场组精对准的单轮最低速度；普通动作继续使用 motion 下限。
    green_alignment_min_wheel_velocity_m_s: float = 0.01
    pickup_cruise_speed_scale: float = 1.0
    # 夹取精细接近末段的比例速度增益，单位 s^-1。
    pickup_terminal_speed_gain_s_inv: float = 1.0
    green_approach_speed_m_s: float = 0.08
    # 以下旧纯逻辑夹具字段同样不再由 YAML 解析，正式流程不读取。
    green_grab_offset_mm: float = 60.0
    green_preclose_recheck_range_mm: float = 200.0
    green_preclose_recheck_hold_ms: float = 500.0
    green_preclose_max_carried_blocks: int = 2
    transport_rotate_angular_velocity_rad_s: float = 0.30
    safe_zone_key_search_angular_velocity_rad_s: float = 0.45
    # 单侧裁剪时持续扫描至完整 bbox 入镜的固定角速度。
    safe_zone_bbox_turn_max_angular_velocity_rad_s: float = 0.25
    # 完整 bbox 四边和所选两点都须离开图像边缘该余量。
    safe_zone_bbox_edge_margin_px: float = 2.0
    # 安全区关键点缺失后的静止重观测窗口，单位 s。
    safe_zone_keypoint_reobserve_timeout_s: float = 0.80
    # 两侧/上下裁剪时的一次倒车参数，即使两点齐全也调整；单位分别为 m/s、m。
    safe_zone_keypoint_reverse_speed_m_s: float = 0.08
    safe_zone_keypoint_reverse_max_distance_m: float = 0.10
    transport_align_tolerance_mm: float = 30.0
    # 普通 正式流程的单段安全区运输速度，单位 m/s。
    # 正式流程 安全区三段直行速度，分别为夹取后→d1、d1→d2、d2→末段，单位 m/s。
    safe_zone_grab_to_d1_speed_m_s: float = 0.08
    safe_zone_d1_to_d2_speed_m_s: float = 0.08
    safe_zone_d2_to_final_speed_m_s: float = 0.08
    # 正式流程 单橙色伤员 d2→安全区末段的独立速度，单位 m/s。
    safe_zone_orange_d2_to_final_speed_m_s: float = 0.08
    # 正式流程 d2→安全区末端的临时车体加减速度；None 分别继承 motion 对应值。
    safe_zone_d2_to_final_max_linear_acceleration_m_s2: float | None = None
    safe_zone_d2_to_final_max_linear_deceleration_m_s2: float | None = None
    safe_zone_d2_to_final_max_angular_acceleration_rad_s2: float | None = None
    safe_zone_d2_to_final_max_angular_deceleration_rad_s2: float | None = None
    # 解团目标对准达到起始间距后，先完全停车再开始推散。
    breakup_settle_time_s: float = 0.5
    # 正式流程 每个转向/直行动作切换前的零速保持时间；纯逻辑默认 0，真机配置应
    # 显式设置为现场允许的停顿，例如 0.3 s。
    action_settle_time_s: float = 0.0
    # 解团地图的物理半幅和夹爪前端偏移；解团中点必须落在两者相减后的范围内。
    breakup_field_half_extent_mm: float = 1500.0
    breakup_gripper_offset_mm: float = 200.0
    # 安全区扫描失败后的 FieldPoint 导航目标；这是动作航点，不替代静态地图
    # 中记录的安全区几何地标。
    safe_zone_fallback_target_field: FieldPoint = FieldPoint(-165.0, 1440.0)
    # 伤员单独转运时的己方伤员区航点；普通/核心物资继续使用
    # safe_zone_fallback_target_field。
    safe_zone_injured_target_field: FieldPoint = FieldPoint(165.0, 1440.0)
    # 安全区一圈扫描失败后的静态地图角点接近；重新看到己方安全区即返回
    # TRANSPORT_ALIGN_RED_ZONE，未在最大距离内看到则安全停车。
    safe_zone_fallback_heading_tolerance_rad: float = 0.20
    safe_zone_fallback_heading_kp_rad_s: float = 1.0
    safe_zone_fallback_max_angular_velocity_rad_s: float = 0.30
    # D1 视觉校准距安全区终点的最小/最大偏移，单位 mm。
    safe_zone_calibration_min_offset_mm: float = 300.0
    safe_zone_calibration_start_offset_mm: float = 500.0
    safe_zone_open_offset_mm: float = 137.0
    # 正式流程 夹取后→d1、d1→d2 直线段的软刹车过冲，使用带符号的
    # FieldPoint 分量（mm）。正值表示刹车后仍向对应的场地正轴方向移动，
    # 策略会把停车目标向反方向提前。
    safe_zone_d2_braking_overrun_x_mm: float = 0.0
    safe_zone_d2_braking_overrun_y_mm: float = 0.0
    # 正式流程 d2→末段的固定刹车过冲提前量，正值表示沿己方安全区方向
    # （区域 2 为 +y、区域 3 为 -y）过冲，单位 mm。
    safe_zone_d2_to_final_braking_overrun_mm: float = 0.0
    # 正式流程 单橙色伤员 d2→末段的独立刹车过冲提前量，单位 mm。
    safe_zone_orange_d2_to_final_braking_overrun_mm: float = 0.0
    safe_zone_calibration_stop_speed_threshold_m_s: float = 0.02
    safe_zone_calibration_stop_confirm_time_s: float = 0.30
    # 正式流程 match flow: independent reverse distance after delivery before
    # re-arming cluster search, in metres.
    safe_zone_exit_distance_m: float = 0.30
    required_transports: int = 4
    return_backup_speed_m_s: float = 0.08
    # 无解团（no-breakup）变体开场：从当前起点按相对动作序列执行。
    # 转向角为带符号 rad（左正右负），直行距离为带符号 m（前进正倒车负），
    # 每个动作的速度均为正的幅值。正式 match 不读取这些字段。
    nb_opening_actions: tuple[NBOpeningAction, ...] = _DEFAULT_NB_OPENING_ACTIONS
    # 1-based 动作序号；完成该动作并停稳后张开夹爪，None 表示不自动张爪。
    nb_opening_gripper_after_action: int | None = 2
    nb_opening_gripper_left_deg: float = 50.0
    nb_opening_gripper_right_deg: float = 130.0
    # 转向停止误差、编码器直行停止误差、动作切换停稳时间和单次转向超时。
    nb_opening_turn_tolerance_rad: float = 0.08
    nb_opening_distance_tolerance_m: float = 0.03
    nb_opening_settle_time_s: float = 0.30
    nb_opening_turn_timeout_s: float = 8.0
    # NB 相对动作闭环参数。制动减速度来自 motion 配置；以下响应、低速和
    # 停稳门限应在真车上分别标定，不能把它们当作零误差保证。
    nb_opening_execution_response_s: float = 0.04
    # 真车测得的有效减速度；None 使用 NB 控制器的保守初值，不能把 motion
    # 的软件限幅直接当成真车制动能力。
    nb_opening_effective_linear_deceleration_m_s2: float | None = None
    nb_opening_effective_angular_deceleration_rad_s2: float | None = None
    nb_opening_max_telemetry_age_ms: float = 160.0
    nb_opening_fine_linear_speed_m_s: float = 0.05
    nb_opening_fine_angular_velocity_rad_s: float = 0.12
    nb_opening_stop_wheel_speed_m_s: float = 0.015
    nb_opening_stop_angular_velocity_rad_s: float = 0.06
    nb_opening_heading_tolerance_rad: float = 0.03
    nb_opening_heading_kp_rad_s: float = 2.0
    nb_opening_heading_max_angular_velocity_rad_s: float = 0.25
    nb_opening_correction_max_distance_m: float = 0.08
    nb_opening_correction_max_angle_rad: float = 0.12
    nb_opening_correction_timeout_s: float = 0.50

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if not isinstance(self.opportunistic_single_green_enabled, bool):
            raise ValueError(
                "opportunistic_single_green_enabled must be a boolean."
            )
        if not isinstance(self.safe_zone_fallback_target_field, FieldPoint):
            raise ValueError(
                "safe_zone_fallback_target_field must be a FieldPoint."
            )
        if not isinstance(self.safe_zone_injured_target_field, FieldPoint):
            raise ValueError(
                "safe_zone_injured_target_field must be a FieldPoint."
            )
        if not isinstance(self.nb_opening_actions, tuple) or not self.nb_opening_actions:
            raise ValueError("nb_opening_actions must be a non-empty tuple.")
        for index, action in enumerate(self.nb_opening_actions, start=1):
            if not isinstance(action, (NBOpeningTurn, NBOpeningStraight)):
                raise ValueError(
                    "nb_opening_actions must contain only NBOpeningTurn or "
                    f"NBOpeningStraight, got item {index}: {action!r}."
                )
        if self.nb_opening_gripper_after_action is not None:
            if (
                isinstance(self.nb_opening_gripper_after_action, bool)
                or not isinstance(self.nb_opening_gripper_after_action, int)
                or not 1 <= self.nb_opening_gripper_after_action <= len(self.nb_opening_actions)
            ):
                raise ValueError(
                    "nb_opening_gripper_after_action must be None or a 1-based "
                    "action index within nb_opening_actions."
                )
            if not isinstance(
                self.nb_opening_actions[self.nb_opening_gripper_after_action - 1],
                NBOpeningStraight,
            ):
                raise ValueError(
                    "nb_opening_gripper_after_action must identify a straight action."
                )
        for name in (
            "robot_footprint_radius_mm",
            "safety_margin_mm",
            "startup_turn_angle_rad",
            "startup_turn_settle_time_s",
            "startup_forward_distance_m",
            "startup_forward_speed_m_s",
            "startup_forward_settle_time_s",
            "startup_straight_pid_integral_limit",
            "startup_straight_pid_max_angular_velocity_rad_s",
            "cluster_group_ground_mm",
            "cluster_breakup_standoff_mm",
            "cluster_approach_speed_m_s",
            "cluster_align_tolerance_rad",
            "cluster_align_hold_ms",
            "cluster_align_kp_rad_s",
            "cluster_align_max_angular_velocity_rad_s",
            "cluster_relocate_distance_m",
            "cluster_relocate_speed_m_s",
            "cluster_search_sweep_angle_rad",
            "opportunistic_single_green_clearance_mm",
            "opportunistic_single_green_realign_standoff_mm",
            "misgrasp_breakup_forward_distance_m",
            "misgrasp_breakup_backward_distance_m",
            "misgrasp_breakup_max_heading_change_rad",
            "breakup_forward_distance_m",
            "breakup_forward_speed_m_s",
            "breakup_backward_distance_m",
            "breakup_backward_speed_m_s",
            "spin_angle_rad",
            "green_path_half_width_mm",
            "green_max_age_ms",
            "green_align_hold_ms",
            "green_alignment_tolerance_mm",
            "green_alignment_hysteresis_mm",
            "grasp_task_timeout_ms",
            "green_alignment_timeout_ms",
            "green_alignment_kp_rad_s",
            "green_alignment_max_angular_velocity_rad_s",
            "green_alignment_min_wheel_velocity_m_s",
            "pickup_cruise_speed_scale",
            "pickup_terminal_speed_gain_s_inv",
            "green_approach_speed_m_s",
            "green_grab_offset_mm",
            "green_preclose_recheck_range_mm",
            "green_preclose_recheck_hold_ms",
            "transport_rotate_angular_velocity_rad_s",
            "safe_zone_key_search_angular_velocity_rad_s",
            "safe_zone_bbox_turn_max_angular_velocity_rad_s",
            "transport_align_tolerance_mm",
            "safe_zone_grab_to_d1_speed_m_s",
            "safe_zone_d1_to_d2_speed_m_s",
            "safe_zone_d2_to_final_speed_m_s",
            "safe_zone_orange_d2_to_final_speed_m_s",
            "breakup_settle_time_s",
            "breakup_field_half_extent_mm",
            "breakup_gripper_offset_mm",
            "safe_zone_fallback_heading_tolerance_rad",
            "safe_zone_fallback_heading_kp_rad_s",
            "safe_zone_fallback_max_angular_velocity_rad_s",
            "safe_zone_calibration_min_offset_mm",
            "safe_zone_calibration_start_offset_mm",
            "safe_zone_open_offset_mm",
            "safe_zone_calibration_stop_speed_threshold_m_s",
            "safe_zone_calibration_stop_confirm_time_s",
            "safe_zone_exit_distance_m",
            "return_backup_speed_m_s",
            "safe_zone_keypoint_reobserve_timeout_s",
            "safe_zone_keypoint_reverse_speed_m_s",
            "safe_zone_keypoint_reverse_max_distance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "nb_opening_turn_tolerance_rad",
            "nb_opening_distance_tolerance_m",
            "nb_opening_turn_timeout_s",
            "nb_opening_execution_response_s",
            "nb_opening_max_telemetry_age_ms",
            "nb_opening_fine_linear_speed_m_s",
            "nb_opening_fine_angular_velocity_rad_s",
            "nb_opening_stop_wheel_speed_m_s",
            "nb_opening_stop_angular_velocity_rad_s",
            "nb_opening_heading_tolerance_rad",
            "nb_opening_heading_kp_rad_s",
            "nb_opening_heading_max_angular_velocity_rad_s",
            "nb_opening_correction_max_distance_m",
            "nb_opening_correction_max_angle_rad",
            "nb_opening_correction_timeout_s",
            # 容差为 0 会让动作只能依赖浮点数完全相等，因此必须为正。
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "nb_opening_effective_linear_deceleration_m_s2",
            "nb_opening_effective_angular_deceleration_rad_s2",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive, or None, got {value!r}."
                )
        for name in ("nb_opening_gripper_left_deg", "nb_opening_gripper_right_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 180.0:
                raise ValueError(f"{name} must be finite and in [0, 180].")
        # 允许 0：表示动作完成后不停顿直接进入下一段。
        if (
            not math.isfinite(float(self.nb_opening_settle_time_s))
            or float(self.nb_opening_settle_time_s) < 0.0
        ):
            raise ValueError(
                "nb_opening_settle_time_s must be finite and non-negative."
            )
        for name in (
            "safe_zone_d2_to_final_max_linear_acceleration_m_s2",
            "safe_zone_d2_to_final_max_linear_deceleration_m_s2",
            "safe_zone_d2_to_final_max_angular_acceleration_rad_s2",
            "safe_zone_d2_to_final_max_angular_deceleration_rad_s2",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive, or None, got {value!r}."
                )
        for name in ('breakup_min_penetration_mm', 'breakup_retreat_clearance_mm', 'breakup_push_margin_mm', 'breakup_braking_margin_mm', 'breakup_deceleration_m_s2', 'breakup_no_plan_reobserve_ms'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        if self.misgrasp_breakup_max_heading_change_rad > math.radians(30.0):
            raise ValueError(
                "misgrasp_breakup_max_heading_change_rad must be <= 30 degrees, "
                f"got {self.misgrasp_breakup_max_heading_change_rad!r}."
            )
        if isinstance(self.breakup_max_attempts, bool) or not isinstance(self.breakup_max_attempts, int) or not 1 <= self.breakup_max_attempts <= 4:
            raise ValueError(f"breakup_max_attempts must be an integer in [1, 4], got {self.breakup_max_attempts!r}.")
        # 允许 1：与近场 confirmation_frames 一致，接触核心只认一帧就冻结。
        # 现场感知慢时多帧确认会凑不齐而反复超时重选，需要能配成单帧。
        if (isinstance(self.breakup_confirmation_frames, bool)
                or not isinstance(self.breakup_confirmation_frames, int)
                or not 1 <= self.breakup_confirmation_frames <= 5):
            raise ValueError(
                "breakup_confirmation_frames must be an integer in [1, 5], "
                f"got {self.breakup_confirmation_frames!r}."
            )
        for name in (
            "breakup_max_linear_acceleration_m_s2",
            "breakup_max_linear_deceleration_m_s2",
            "breakup_max_angular_acceleration_rad_s2",
            "breakup_max_angular_deceleration_rad_s2",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive, or None, got {value!r}."
                )
        if (
            not math.isfinite(float(self.action_settle_time_s))
            or float(self.action_settle_time_s) < 0.0
        ):
            raise ValueError(
                "action_settle_time_s must be finite and non-negative."
            )
        if self.breakup_gripper_offset_mm >= self.breakup_field_half_extent_mm:
            raise ValueError(
                "breakup_gripper_offset_mm must be smaller than "
                "breakup_field_half_extent_mm."
            )
        if not (
            self.safe_zone_open_offset_mm
            <= self.safe_zone_calibration_min_offset_mm
            <= self.safe_zone_calibration_start_offset_mm
        ):
            raise ValueError(
                "safe-zone offsets must satisfy safe_zone_open_offset_mm <= "
                "safe_zone_calibration_min_offset_mm <= "
                "safe_zone_calibration_start_offset_mm, got "
                f"{self.safe_zone_open_offset_mm!r}, "
                f"{self.safe_zone_calibration_min_offset_mm!r}, "
                f"{self.safe_zone_calibration_start_offset_mm!r}."
            )
        if (
            isinstance(self.green_preclose_max_carried_blocks, bool)
            or not isinstance(self.green_preclose_max_carried_blocks, int)
            or self.green_preclose_max_carried_blocks <= 0
        ):
            raise ValueError(
                "green_preclose_max_carried_blocks must be a positive integer, "
                f"got {self.green_preclose_max_carried_blocks!r}."
            )
        for name in (
            "startup_turn_angular_velocity_rad_s",
            "cluster_search_angular_velocity_rad_s",
            "cluster_search_empty_angular_velocity_rad_s",
            "close_gripper_spin_angular_velocity_rad_s",
        ):
            value = float(getattr(self, name))
            if abs(value) < 0.001:
                raise ValueError(f"{name} must have |value| >= 0.001 rad/s.")
        if (
            self.cluster_search_angular_velocity_rad_s
            * self.cluster_search_empty_angular_velocity_rad_s
            <= 0.0
        ):
            raise ValueError(
                "cluster_search_empty_angular_velocity_rad_s must use the "
                "same rotation direction as cluster_search_angular_velocity_rad_s."
            )
        if abs(self.cluster_search_empty_angular_velocity_rad_s) < abs(
            self.cluster_search_angular_velocity_rad_s
        ):
            raise ValueError(
                "cluster_search_empty_angular_velocity_rad_s must be at least "
                "as fast as cluster_search_angular_velocity_rad_s."
            )
        for name in (
            "startup_straight_pid_kp_rad_s_per_m_s",
            "startup_straight_pid_ki_rad_s_per_m",
            "startup_straight_pid_kd_rad_s_per_m_s2",
            "startup_straight_pid_deadband_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "safe_zone_d2_braking_overrun_x_mm",
            "safe_zone_d2_braking_overrun_y_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        for name in (
            "safe_zone_d2_to_final_braking_overrun_mm",
            "safe_zone_orange_d2_to_final_braking_overrun_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )
        if self.startup_turn_angle_rad >= 2.0 * math.pi:
            raise ValueError("startup_turn_angle_rad must be smaller than 2*pi.")
        if self.spin_angle_rad > 2.0 * math.pi:
            raise ValueError("spin_angle_rad must be <= 2*pi.")
        if self.cluster_search_sweep_angle_rad > 2.0 * math.pi:
            raise ValueError("cluster_search_sweep_angle_rad must be <= 2*pi.")
        if (
            not math.isfinite(float(self.safe_zone_bbox_edge_margin_px))
            or float(self.safe_zone_bbox_edge_margin_px) < 0.0
        ):
            raise ValueError(
                "safe_zone_bbox_edge_margin_px must be finite and non-negative."
            )
        for name in (
            "startup_straight_pid_kp_rad_s_per_m_s",
            "startup_straight_pid_ki_rad_s_per_m",
            "startup_straight_pid_kd_rad_s_per_m_s2",
            "startup_straight_pid_integral_limit",
            "startup_straight_pid_deadband_m_s",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.cluster_align_tolerance_rad >= math.pi:
            raise ValueError("cluster_align_tolerance_rad must be less than pi.")
        if (
            self.opportunistic_single_green_realign_standoff_mm
            <= self.green_grab_offset_mm
        ):
            raise ValueError(
                "opportunistic_single_green_realign_standoff_mm must be greater "
                "than green_grab_offset_mm."
            )
        for name in (
            "cluster_min_detections",
            "required_transports",
            "green_alignment_stable_frames",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class GreenGrabRuntimeConfig:
    """无地面标定时，像素居中接近单个绿色物资并合爪的试验参数。"""

    enabled: bool
    search_angular_velocity_rad_s: float
    approach_speed_m_s: float
    align_tolerance_ratio: float
    align_kp_rad_s: float
    align_max_angular_velocity_rad_s: float
    engage_bottom_fraction: float
    confirm_frames: int
    target_loss_timeout_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        for name in (
            "approach_speed_m_s",
            "align_tolerance_ratio",
            "align_kp_rad_s",
            "align_max_angular_velocity_rad_s",
            "target_loss_timeout_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        search = float(self.search_angular_velocity_rad_s)
        if not math.isfinite(search) or abs(search) < 0.001:
            raise ValueError(
                "search_angular_velocity_rad_s must be a finite angular "
                f"velocity with |value| >= 0.001 rad/s (positive means left "
                f"turn), got {search!r}."
            )
        if self.align_tolerance_ratio >= 1.0:
            raise ValueError("align_tolerance_ratio must be < 1.0.")
        engage = float(self.engage_bottom_fraction)
        if not math.isfinite(engage) or not 0.0 < engage <= 1.0:
            raise ValueError(
                "engage_bottom_fraction must be finite and in (0, 1], "
                f"got {self.engage_bottom_fraction!r}."
            )
        if (
            isinstance(self.confirm_frames, bool)
            or not isinstance(self.confirm_frames, int)
            or self.confirm_frames <= 0
        ):
            raise ValueError("confirm_frames must be a positive integer.")


@dataclass(frozen=True, slots=True)
class MotionRuntimeConfig:
    enabled: bool
    wheel_track_m: float | None
    max_linear_velocity_m_s: float
    max_angular_velocity_rad_s: float
    max_wheel_velocity_m_s: float
    max_linear_acceleration_m_s2: float
    max_linear_deceleration_m_s2: float
    max_angular_acceleration_rad_s2: float
    max_angular_deceleration_rad_s2: float
    min_wheel_velocity_m_s: float
    left_wheel_speed_weight: float
    right_wheel_speed_weight: float
    max_remote_command_valid_for_ms: int
    synchronization_timeout_s: float
    stall_guard_enabled: bool
    stall_guard_timeout_ms: int
    stall_guard_min_command_speed_m_s: float
    stall_guard_stationary_encoder_delta_count: int
    gripper: GripperRuntimeConfig
    odometry: OdometryRuntimeConfig
    cluster_breakup: ClusterBreakupRuntimeConfig

    def build_controller(
        self,
        channel: UartFrameChannel | None,
    ) -> MotionController | None:
        """按配置创建运动控制器；不会创建或打开 UART。"""

        if not self.enabled:
            return None
        if channel is None:
            raise RuntimeError("Enabled motion requires an enabled UART channel.")
        assert self.wheel_track_m is not None
        from rescue_vision.motion import MotionController, MotionLimits

        return MotionController(
            channel,
            MotionLimits(
                wheel_track_m=self.wheel_track_m,
                max_linear_velocity_m_s=self.max_linear_velocity_m_s,
                max_angular_velocity_rad_s=self.max_angular_velocity_rad_s,
                max_wheel_velocity_m_s=self.max_wheel_velocity_m_s,
                max_linear_acceleration_m_s2=(
                    self.max_linear_acceleration_m_s2
                ),
                max_linear_deceleration_m_s2=(
                    self.max_linear_deceleration_m_s2
                ),
                max_angular_acceleration_rad_s2=(
                    self.max_angular_acceleration_rad_s2
                ),
                max_angular_deceleration_rad_s2=(
                    self.max_angular_deceleration_rad_s2
                ),
                min_wheel_velocity_m_s=self.min_wheel_velocity_m_s,
                left_wheel_speed_weight=self.left_wheel_speed_weight,
                right_wheel_speed_weight=self.right_wheel_speed_weight,
                max_remote_command_valid_for_ms=(
                    self.max_remote_command_valid_for_ms
                ),
                stall_guard_enabled=self.stall_guard_enabled,
                stall_guard_timeout_ms=self.stall_guard_timeout_ms,
                stall_guard_min_command_speed_m_s=(
                    self.stall_guard_min_command_speed_m_s
                ),
                stall_guard_stationary_encoder_delta_count=(
                    self.stall_guard_stationary_encoder_delta_count
                ),
            ),
        )

    def build_remote_executor(
        self,
        controller: MotionController | None,
    ) -> RemoteMotionExecutor | None:
        """为已装配的控制器创建远程调试执行器。"""

        if not self.enabled:
            return None
        if controller is None:
            raise RuntimeError("Enabled motion requires a motion controller.")
        from rescue_vision.motion import RemoteMotionExecutor

        return RemoteMotionExecutor(controller)

    def build_remote_gripper_executor(
        self,
        controller: MotionController | None,
    ) -> RemoteGripperExecutor | None:
        """为已装配的控制器创建远程夹爪执行器。"""

        if not self.enabled or not self.gripper.enabled:
            return None
        if controller is None:
            raise RuntimeError("Enabled motion requires a motion controller.")
        from rescue_vision.motion import RemoteGripperExecutor

        calibration = self.gripper.build_calibration()
        assert calibration is not None
        return RemoteGripperExecutor(controller, calibration)


@dataclass(frozen=True, slots=True)
class WorldRuntimeConfig:
    model: WorldModelConfig
    team_color: TeamColor
    static_map: StaticFieldMap

    def mission_regions(self) -> tuple[StaticRegion, ...]:
        """Map fixed physical regions to current team-relative semantics."""

        mapped: list[StaticRegion] = []
        own_material = (
            PhysicalRegionKind.RED_MATERIAL
            if self.team_color is TeamColor.RED
            else PhysicalRegionKind.BLUE_MATERIAL
        )
        own_injured = (
            PhysicalRegionKind.RED_INJURED
            if self.team_color is TeamColor.RED
            else PhysicalRegionKind.BLUE_INJURED
        )
        opponent_kinds = (
            {
                PhysicalRegionKind.BLUE_MATERIAL,
                PhysicalRegionKind.BLUE_INJURED,
            }
            if self.team_color is TeamColor.RED
            else {
                PhysicalRegionKind.RED_MATERIAL,
                PhysicalRegionKind.RED_INJURED,
            }
        )
        for region in self.static_map.regions:
            kind: RegionKind | None = None
            if region.kind is PhysicalRegionKind.FIELD:
                kind = RegionKind.FIELD
            elif self.team_color is not TeamColor.UNKNOWN:
                if region.kind is own_material:
                    kind = RegionKind.OWN_MATERIAL
                elif region.kind is own_injured:
                    kind = RegionKind.OWN_INJURED
                elif region.kind in opponent_kinds:
                    kind = RegionKind.OPPONENT_SAFE
            if kind is not None:
                mapped.append(
                    StaticRegion(region.region_id, kind, region.polygon_field)
                )
        return tuple(mapped)

    def build_model(self) -> WorldModel:
        return WorldModel(self.model, self.mission_regions())


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    detection_threshold: float
    k0_threshold: float
    color_classifier: HsvColorClassifierConfig
    target_ground_geometry: TargetGroundGeometryConfig
    center_cross_refinement: CenterCrossRefinementConfig
    safe_zone_color: SafeZoneColorConfig
    gripper_color: GripperColorConfig = GripperColorConfig()

    def build_target_ground_geometry_estimator(
        self,
        *,
        max_observation_age_ms: float,
        ground_projector: GroundProjector | None,
    ):
        """按配置创建目标地面几何估计器；禁用时返回 ``None``。"""

        if not self.target_ground_geometry.enabled:
            return None
        if ground_projector is None:
            raise RuntimeError(
                "Enabled target_ground_geometry requires ground mapping."
            )
        from rescue_vision.perception.target_ground_geometry import (
            TargetGroundGeometryEstimator,
        )

        return TargetGroundGeometryEstimator(
            self.target_ground_geometry,
            max_observation_age_ms=max_observation_age_ms,
            ground_projector=ground_projector,
        )


@dataclass(frozen=True, slots=True)
class HailoConfig:
    enabled: bool
    hef_path: Path | None
    postprocess_onnx_path: Path | None
    output_mapping_path: Path | None
    raw_classes: tuple[str, ...]
    backend_score_threshold: float
    max_detections: int

    def build_backend(self):
        """延迟导入并创建 Hailo 后端；禁用时返回 ``None``。"""

        if not self.enabled:
            return None
        assert self.hef_path is not None
        assert self.postprocess_onnx_path is not None
        assert self.output_mapping_path is not None
        from rescue_vision.perception.hailo_yolo26_pose import HailoYolo26PoseBackend

        return HailoYolo26PoseBackend(
            hef_path=self.hef_path,
            postprocess_onnx_path=self.postprocess_onnx_path,
            output_mapping_path=self.output_mapping_path,
            class_count=len(self.raw_classes),
            max_detections=self.max_detections,
            score_threshold=self.backend_score_threshold,
        )

@dataclass(frozen=True, slots=True)
class RuntimeGeometry:
    camera_model: CameraModel
    ground_projector: GroundProjector | None


@dataclass(frozen=True, slots=True)
class LocalizationRuntimeConfig:
    center_cross: CenterCrossLocalizerConfig
    static_landmarks: StaticLandmarkTrackingConfig
    safe_zone_corners: SafeZoneCornerLocalizerConfig
    fusion: FusionConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig
    geometry: GeometryConfig
    recording: RecordingConfig
    processing: ProcessingConfig
    uart: UartConfig
    remote: RemoteConfig
    motion: MotionRuntimeConfig
    match: MatchRuntimeConfig
    tracking: TrackingConfig
    world: WorldRuntimeConfig
    mission: MissionConfig
    perception: PerceptionConfig
    localization: LocalizationRuntimeConfig
    hailo: HailoConfig
    green_grab: GreenGrabRuntimeConfig
    near_field_grasp: NearFieldGraspConfig = NearFieldGraspConfig()
    match_cc: MatchCCRuntimeConfig = MatchCCRuntimeConfig()

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

    def build_target_pose_detector(
        self,
        *,
        ground_projector: GroundProjector | None = None,
    ) -> TargetPoseDetector | None:
        """按当前 Hailo、类别和 HSV 配置创建任务目标检测器。

        Hailo 关闭时返回 ``None``，不会导入或访问硬件。返回的 detector 接管
        backend 生命周期，调用方必须在退出前调用其 ``close()`` 或使用上下文。
        """

        backend = self.hailo.build_backend()
        if backend is None:
            return None
        from rescue_vision.perception import TargetPoseDetector

        return TargetPoseDetector(
            backend,
            detection_threshold=self.perception.detection_threshold,
            k0_threshold=self.perception.k0_threshold,
            color_classifier=self.perception.color_classifier,
            max_observation_age_ms=self.processing.max_observation_age_ms,
            ground_projector=ground_projector,
            center_cross_refinement=self.perception.center_cross_refinement,
            safe_zone_color=self.perception.safe_zone_color,
            gripper_color=self.perception.gripper_color,
        )

    def build_center_cross_localizer(
        self,
        *,
        ground_projector: GroundProjector | None,
    ) -> CenterCrossLocalizer | None:
        """Build center-cross localization when all geometric inputs exist."""

        if (
            not self.localization.center_cross.enabled
            or not self.geometry.ground_mapping_enabled
            or ground_projector is None
        ):
            return None
        from rescue_vision.localization import CenterCrossLocalizer

        return CenterCrossLocalizer(
            self.localization.center_cross,
            static_map=self.world.static_map,
            max_observation_age_ms=self.processing.max_observation_age_ms,
        )

    def build_static_field_landmark_tracker(self):
        """Build bounded map-prior tracking for the visual localization path."""

        from rescue_vision.localization import StaticFieldLandmarkTracker

        return StaticFieldLandmarkTracker(
            self.world.static_map,
            self.localization.static_landmarks,
            max_linear_velocity_m_s=self.motion.max_linear_velocity_m_s,
            max_angular_velocity_rad_s=self.motion.max_angular_velocity_rad_s,
        )

    def build_safe_zone_corner_localizer(self):
        """Build safe-zone corner absolute-pose fitting from the static map."""

        from rescue_vision.localization import SafeZoneCornerLocalizer

        return SafeZoneCornerLocalizer(
            self.world.static_map,
            self.localization.safe_zone_corners,
        )

    def build_odometry_imu_fusion(self) -> OdometryImuFusion | None:
        """Build continuous localization without opening UART or camera resources."""

        if not self.localization.fusion.enabled:
            return None
        if not self.motion.enabled or self.motion.wheel_track_m is None:
            raise RuntimeError("Enabled localization fusion requires motion.")
        calibration = self.motion.odometry.build_calibration()
        if calibration is None:
            raise RuntimeError(
                "Enabled localization fusion requires motion.odometry."
            )
        from rescue_vision.localization import OdometryImuFusion

        return OdometryImuFusion(
            self.localization.fusion,
            calibration,
            wheel_track_m=self.motion.wheel_track_m,
        )

    def build_visual_localization_pipeline(
        self,
        *,
        ground_projector: GroundProjector | None,
        fusion: OdometryImuFusion | None,
    ):
        """装配 v3 场地地标消费链；任一门禁关闭时返回 ``None``。"""

        if fusion is None or not self.hailo.enabled:
            return None
        center = self.build_center_cross_localizer(ground_projector=ground_projector)
        if center is None:
            return None
        from rescue_vision.localization import VisualLocalizationPipeline

        return VisualLocalizationPipeline(
            center_cross_localizer=center,
            safe_zone_localizer=self.build_safe_zone_corner_localizer(),
            landmark_tracker=self.build_static_field_landmark_tracker(),
            fusion=fusion,
        )

    def build_match_sequence(self) -> MatchSequence:
        """按本配置装配 正式动作策略纯逻辑流程。"""

        if not self.match.enabled:
            raise RuntimeError(
                "match.enabled must be true to build the flow."
            )
        from rescue_vision.app.match import MatchSequence

        return MatchSequence.from_app_config(self)

    def build_grab_transport_sequence(
        self,
    ) -> GrabTransportSequence:
        """装配不含解团的 正式流程 绿色夹取—运输测试流程。"""

        if not self.match.enabled:
            raise RuntimeError(
                "match.enabled must be true to build the flow."
            )
        from rescue_vision.app.grab_transport import (
            GrabTransportSequence,
        )

        return GrabTransportSequence.from_app_config(self)

    def build_match_nb_sequence(self) -> MatchNBSequence:
        """装配当前 match 的简化开场变体。"""

        if not self.match.enabled:
            raise RuntimeError(
                "match.enabled must be true to build the flow."
            )
        from rescue_vision.app.match_nb import MatchNBSequence

        return MatchNBSequence.from_app_config(self)


def load_runtime_config(path: str | Path) -> AppConfig:
    """从 YAML 加载运行配置；未知字段和无效值均视为错误。"""

    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    _reject_unknown(
        root,
        {
            "camera",
            "geometry",
            "recording",
            "processing",
            "uart",
            "remote",
            "motion",
            "match",
            "match_cc",
            "near_field_grasp",
            "tracking",
            "world",
            "mission",
            "perception",
            "localization",
            "hailo",
            "green_grab",
        },
        "root",
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

    geometry_raw = _mapping(root.get("geometry", {}), "geometry")
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
    intrinsics_enabled = geometry_raw.get("intrinsics_enabled", False)
    if not isinstance(intrinsics_enabled, bool):
        raise ValueError("geometry.intrinsics_enabled must be a boolean.")
    ground_mapping_enabled = geometry_raw.get("ground_mapping_enabled", False)
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

    recording_raw = _mapping(root.get("recording", {}), "recording")
    _reject_unknown(recording_raw, {"queue_capacity", "image_format"}, "recording")
    image_format = recording_raw.get("image_format", "jpg")
    if image_format not in {"png", "jpg"}:
        raise ValueError("recording.image_format must be 'png' or 'jpg'.")
    recording = RecordingConfig(
        queue_capacity=_positive_int(
            recording_raw.get("queue_capacity", 8),
            "recording.queue_capacity",
        ),
        image_format=image_format,
    )

    processing_raw = _mapping(
        root.get("processing", {}),
        "processing",
    )
    _reject_unknown(
        processing_raw,
        {"max_observation_age_ms"},
        "processing",
    )
    processing = ProcessingConfig(
        max_observation_age_ms=_finite_float(
            processing_raw.get("max_observation_age_ms", 150.0),
            "processing.max_observation_age_ms",
            minimum=0.001,
        )
    )

    uart_raw = _mapping(root.get("uart", {}), "uart")
    _reject_unknown(
        uart_raw,
        {
            "enabled",
            "device",
            "read_timeout_ms",
            "write_timeout_ms",
            "receive_queue_capacity",
        },
        "uart",
    )
    uart_enabled = uart_raw.get("enabled", False)
    if not isinstance(uart_enabled, bool):
        raise ValueError("uart.enabled must be a boolean.")
    uart_device_raw = uart_raw.get("device")
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
        read_timeout_ms=_finite_float(
            uart_raw.get("read_timeout_ms", 100.0),
            "uart.read_timeout_ms",
            minimum=0.001,
        ),
        write_timeout_ms=_finite_float(
            uart_raw.get("write_timeout_ms", 100.0),
            "uart.write_timeout_ms",
            minimum=0.001,
        ),
        receive_queue_capacity=_positive_int(
            uart_raw.get("receive_queue_capacity", 256),
            "uart.receive_queue_capacity",
        ),
    )

    remote_raw = _mapping(root.get("remote", {}), "remote")
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
    remote_enabled = remote_raw.get("enabled", False)
    if not isinstance(remote_enabled, bool):
        raise ValueError("remote.enabled must be a boolean.")
    try:
        remote_role = RemoteRole(remote_raw.get("role", "server"))
    except (TypeError, ValueError) as exc:
        raise ValueError("remote.role must be 'server' or 'client'.") from exc
    try:
        remote_access_mode = RemoteAccessMode(
            remote_raw.get("access_mode", "observe_only")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "remote.access_mode must be 'observe_only' or 'debug_control'."
        ) from exc
    remote_port = _positive_int(
        remote_raw.get("port", 8765),
        "remote.port",
    )
    if remote_port > 65_535:
        raise ValueError("remote.port must be <= 65535.")
    remote = RemoteConfig(
        enabled=remote_enabled,
        role=remote_role,
        host=_string(remote_raw.get("host", "0.0.0.0"), "remote.host"),
        port=remote_port,
        access_mode=remote_access_mode,
        connect_timeout_ms=_finite_float(
            remote_raw.get("connect_timeout_ms", 2000.0),
            "remote.connect_timeout_ms",
            minimum=0.001,
        ),
        io_timeout_ms=_finite_float(
            remote_raw.get("io_timeout_ms", 100.0),
            "remote.io_timeout_ms",
            minimum=0.001,
        ),
        control_queue_capacity=_positive_int(
            remote_raw.get("control_queue_capacity", 32),
            "remote.control_queue_capacity",
        ),
        observation_queue_capacity=_positive_int(
            remote_raw.get("observation_queue_capacity", 3),
            "remote.observation_queue_capacity",
        ),
        max_header_bytes=_positive_int(
            remote_raw.get("max_header_bytes", 4096),
            "remote.max_header_bytes",
        ),
        max_payload_bytes=_positive_int(
            remote_raw.get("max_payload_bytes", 2_097_152),
            "remote.max_payload_bytes",
        ),
    )

    motion_raw = _mapping(root.get("motion", {}), "motion")
    _reject_unknown(
        motion_raw,
        {
            "enabled",
            "wheel_track_m",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_wheel_velocity_m_s",
            "max_linear_acceleration_m_s2",
            "max_linear_deceleration_m_s2",
            "max_angular_acceleration_rad_s2",
            "max_angular_deceleration_rad_s2",
            "min_wheel_velocity_m_s",
            "left_wheel_speed_weight",
            "right_wheel_speed_weight",
            "max_remote_command_valid_for_ms",
            "synchronization_timeout_s",
            "stall_guard",
            "gripper",
            "odometry",
            "cluster_breakup",
        },
        "motion",
    )
    motion_enabled = motion_raw.get("enabled", False)
    if not isinstance(motion_enabled, bool):
        raise ValueError("motion.enabled must be a boolean.")
    wheel_track_raw = motion_raw.get("wheel_track_m")
    wheel_track_m = (
        None
        if wheel_track_raw is None
        else _finite_float(
            wheel_track_raw,
            "motion.wheel_track_m",
            minimum=0.001,
        )
    )
    if motion_enabled and wheel_track_m is None:
        raise ValueError("Enabled motion requires motion.wheel_track_m.")
    if motion_enabled and not uart.enabled:
        raise ValueError("Enabled motion requires uart.enabled=true.")
    max_remote_validity = _positive_int(
        motion_raw.get("max_remote_command_valid_for_ms", 500),
        "motion.max_remote_command_valid_for_ms",
    )
    if max_remote_validity > 5_000:
        raise ValueError(
            "motion.max_remote_command_valid_for_ms must be <= 5000."
        )
    synchronization_timeout_s = _finite_float(
        motion_raw.get("synchronization_timeout_s", 1.0),
        "motion.synchronization_timeout_s",
        minimum=0.001,
    )
    stall_guard_raw = _mapping(
        motion_raw.get("stall_guard", {}),
        "motion.stall_guard",
    )
    _reject_unknown(
        stall_guard_raw,
        {
            "enabled",
            "timeout_ms",
            "min_command_speed_m_s",
            "stationary_encoder_delta_count",
        },
        "motion.stall_guard",
    )
    stall_guard_enabled = stall_guard_raw.get("enabled", True)
    if not isinstance(stall_guard_enabled, bool):
        raise ValueError("motion.stall_guard.enabled must be a boolean.")
    stall_guard_timeout_ms = _positive_int(
        stall_guard_raw.get("timeout_ms", 300),
        "motion.stall_guard.timeout_ms",
    )
    if stall_guard_timeout_ms > 5_000:
        raise ValueError("motion.stall_guard.timeout_ms must be <= 5000.")
    stall_guard_min_command_speed_m_s = _finite_float(
        stall_guard_raw.get("min_command_speed_m_s", 0.02),
        "motion.stall_guard.min_command_speed_m_s",
        minimum=0.001,
    )
    stall_guard_stationary_encoder_delta_count = _nonnegative_int(
        stall_guard_raw.get("stationary_encoder_delta_count", 0),
        "motion.stall_guard.stationary_encoder_delta_count",
    )
    gripper_raw = _mapping(
        motion_raw.get("gripper", {}),
        "motion.gripper",
    )
    _reject_unknown(
        gripper_raw,
        {
            "enabled",
            "open_left_angle_deg",
            "open_right_angle_deg",
            "closed_left_angle_deg",
            "closed_right_angle_deg",
            "transport_left_angle_deg",
            "transport_right_angle_deg",
            "full_travel_time_s",
            "angle_sum_deg",
        },
        "motion.gripper",
    )
    gripper_enabled = gripper_raw.get("enabled", False)
    if not isinstance(gripper_enabled, bool):
        raise ValueError("motion.gripper.enabled must be a boolean.")
    gripper_values = {
        name: _optional_servo_angle(
            gripper_raw.get(name),
            f"motion.gripper.{name}",
        )
        for name in (
            "open_left_angle_deg",
            "open_right_angle_deg",
            "closed_left_angle_deg",
            "closed_right_angle_deg",
            "transport_left_angle_deg",
            "transport_right_angle_deg",
        )
    }
    travel_time_raw = gripper_raw.get("full_travel_time_s")
    full_travel_time_s = (
        None
        if travel_time_raw is None
        else _finite_float(
            travel_time_raw,
            "motion.gripper.full_travel_time_s",
            minimum=0.001,
        )
    )
    angle_sum_raw = gripper_raw.get("angle_sum_deg")
    angle_sum_deg = (
        None
        if angle_sum_raw is None
        else _finite_float(
            angle_sum_raw,
            "motion.gripper.angle_sum_deg",
            minimum=0.001,
        )
    )
    if angle_sum_deg is not None and angle_sum_deg > 360.0:
        raise ValueError(
            "motion.gripper.angle_sum_deg must be <= 360, "
            f"got {angle_sum_raw!r}."
        )
    if gripper_enabled and not motion_enabled:
        raise ValueError(
            "Enabled motion.gripper requires motion.enabled=true."
        )
    if gripper_enabled and (
        any(value is None for value in gripper_values.values())
        or full_travel_time_s is None
        or angle_sum_deg is None
    ):
        raise ValueError(
            "Enabled motion.gripper requires open, closed and transport "
            "angles plus full_travel_time_s and angle_sum_deg."
        )
    gripper = GripperRuntimeConfig(
        enabled=gripper_enabled,
        open_left_angle_deg=gripper_values["open_left_angle_deg"],
        open_right_angle_deg=gripper_values["open_right_angle_deg"],
        closed_left_angle_deg=gripper_values["closed_left_angle_deg"],
        closed_right_angle_deg=gripper_values["closed_right_angle_deg"],
        transport_left_angle_deg=gripper_values["transport_left_angle_deg"],
        transport_right_angle_deg=gripper_values["transport_right_angle_deg"],
        full_travel_time_s=full_travel_time_s,
        angle_sum_deg=angle_sum_deg,
    )
    if gripper.enabled:
        # Reuse the motion-layer validation for distinct endpoints.
        gripper.build_calibration()
    odometry_raw = _mapping(motion_raw.get("odometry", {}), "motion.odometry")
    _reject_unknown(
        odometry_raw,
        {
            "enabled",
            "encoder_counts_per_revolution",
            "left_wheel_radius_mm",
            "right_wheel_radius_mm",
            "gyro_z_sign",
            "max_consecutive_overrun_samples",
        },
        "motion.odometry",
    )
    odometry_enabled = odometry_raw.get("enabled", False)
    if not isinstance(odometry_enabled, bool):
        raise ValueError("motion.odometry.enabled must be a boolean.")
    encoder_counts_raw = odometry_raw.get("encoder_counts_per_revolution")
    encoder_counts = (
        None
        if encoder_counts_raw is None
        else _positive_int(
            encoder_counts_raw,
            "motion.odometry.encoder_counts_per_revolution",
        )
    )
    odometry_floats: dict[str, float | None] = {}
    for name, minimum in (
        ("left_wheel_radius_mm", 0.001),
        ("right_wheel_radius_mm", 0.001),
    ):
        raw_value = odometry_raw.get(name)
        odometry_floats[name] = (
            None
            if raw_value is None
            else _finite_float(
                raw_value,
                f"motion.odometry.{name}",
                minimum=minimum,
            )
        )
    gyro_z_sign = odometry_raw.get("gyro_z_sign", 1)
    if (
        isinstance(gyro_z_sign, bool)
        or not isinstance(gyro_z_sign, int)
        or gyro_z_sign not in {-1, 1}
    ):
        raise ValueError("motion.odometry.gyro_z_sign must be exactly -1 or 1.")
    overrun_budget_raw = odometry_raw.get("max_consecutive_overrun_samples", 1)
    max_consecutive_overrun_samples = (
        None
        if overrun_budget_raw is None
        else _nonnegative_int(
            overrun_budget_raw,
            "motion.odometry.max_consecutive_overrun_samples",
        )
    )
    if odometry_enabled and (
        not motion_enabled
        or encoder_counts is None
        or any(value is None for value in odometry_floats.values())
    ):
        raise ValueError(
            "Enabled motion.odometry requires enabled motion and all calibration values."
        )
    odometry = OdometryRuntimeConfig(
        enabled=odometry_enabled,
        encoder_counts_per_revolution=encoder_counts,
        left_wheel_radius_mm=odometry_floats["left_wheel_radius_mm"],
        right_wheel_radius_mm=odometry_floats["right_wheel_radius_mm"],
        gyro_z_sign=gyro_z_sign,
        max_consecutive_overrun_samples=max_consecutive_overrun_samples,
    )
    if odometry.enabled:
        odometry.build_calibration()
    breakup_raw = _mapping(
        motion_raw.get("cluster_breakup", {}),
        "motion.cluster_breakup",
    )
    breakup_keys = {
        "enabled",
        "departure_distance_m",
        "departure_speed_m_s",
        "search_angular_velocity_rad_s",
        "search_timeout_s",
        "cluster_min_detections",
        "cluster_group_gap_ratio",
        "center_tolerance_ratio",
        "center_confirm_frames",
        "center_kp_rad_s",
        "center_max_angular_velocity_rad_s",
        "center_error_filter_alpha",
        "center_reverse_deadband_ratio",
        "center_reverse_confirm_frames",
        "approach_speed_m_s",
        "gripper_open_distance_mm",
        "breakup_speed_m_s",
        "breakup_distance_m",
        "gripper_open_retreat_distance_m",
        "retreat_speed_m_s",
        "retreat_distance_m",
        "scan_green_angular_velocity_rad_s",
        "green_confirm_frames",
        "target_loss_timeout_ms",
        "motion_phase_timeout_s",
        "post_breakup_retreat_enabled",
    }
    _reject_unknown(breakup_raw, breakup_keys, "motion.cluster_breakup")
    breakup_enabled = breakup_raw.get("enabled", False)
    if not isinstance(breakup_enabled, bool):
        raise ValueError("motion.cluster_breakup.enabled must be a boolean.")

    def breakup_float(name: str, default: float) -> float:
        return _finite_float(
            breakup_raw.get(name, default),
            f"motion.cluster_breakup.{name}",
            minimum=0.001,
        )

    def breakup_signed_angular(name: str, default: float) -> float:
        return _signed_angular_velocity(
            breakup_raw.get(name, default),
            f"motion.cluster_breakup.{name}",
        )

    center_tolerance_ratio = breakup_float("center_tolerance_ratio", 0.08)
    if center_tolerance_ratio > 1.0:
        raise ValueError(
            "motion.cluster_breakup.center_tolerance_ratio must be <= 1.0."
        )
    cluster_group_gap_ratio = _finite_float(
        breakup_raw.get("cluster_group_gap_ratio", 0.10),
        "motion.cluster_breakup.cluster_group_gap_ratio",
        minimum=0.0,
    )
    if cluster_group_gap_ratio > 1.0:
        raise ValueError(
            "motion.cluster_breakup.cluster_group_gap_ratio must be <= 1.0."
        )
    cluster_breakup = ClusterBreakupRuntimeConfig(
        enabled=breakup_enabled,
        departure_distance_m=breakup_float("departure_distance_m", 0.55),
        departure_speed_m_s=breakup_float("departure_speed_m_s", 0.15),
        search_angular_velocity_rad_s=breakup_signed_angular(
            "search_angular_velocity_rad_s", 0.45
        ),
        search_timeout_s=breakup_float("search_timeout_s", 12.0),
        cluster_min_detections=_positive_int(
            breakup_raw.get("cluster_min_detections", 2),
            "motion.cluster_breakup.cluster_min_detections",
        ),
        cluster_group_gap_ratio=cluster_group_gap_ratio,
        center_tolerance_ratio=center_tolerance_ratio,
        center_confirm_frames=_positive_int(
            breakup_raw.get("center_confirm_frames", 3),
            "motion.cluster_breakup.center_confirm_frames",
        ),
        center_kp_rad_s=breakup_float("center_kp_rad_s", 1.2),
        center_max_angular_velocity_rad_s=breakup_float(
            "center_max_angular_velocity_rad_s", 0.55
        ),
        center_error_filter_alpha=breakup_float(
            "center_error_filter_alpha", 0.35
        ),
        center_reverse_deadband_ratio=_finite_float(
            breakup_raw.get("center_reverse_deadband_ratio", 0.03),
            "motion.cluster_breakup.center_reverse_deadband_ratio",
            minimum=0.0,
        ),
        center_reverse_confirm_frames=_positive_int(
            breakup_raw.get("center_reverse_confirm_frames", 2),
            "motion.cluster_breakup.center_reverse_confirm_frames",
        ),
        approach_speed_m_s=breakup_float("approach_speed_m_s", 0.12),
        gripper_open_distance_mm=breakup_float(
            "gripper_open_distance_mm", 260.0
        ),
        breakup_speed_m_s=breakup_float("breakup_speed_m_s", 0.25),
        breakup_distance_m=breakup_float("breakup_distance_m", 0.25),
        gripper_open_retreat_distance_m=breakup_float(
            "gripper_open_retreat_distance_m", 0.10
        ),
        retreat_speed_m_s=breakup_float("retreat_speed_m_s", 0.10),
        retreat_distance_m=breakup_float("retreat_distance_m", 0.10),
        scan_green_angular_velocity_rad_s=breakup_signed_angular(
            "scan_green_angular_velocity_rad_s", 0.35
        ),
        green_confirm_frames=_positive_int(
            breakup_raw.get("green_confirm_frames", 2),
            "motion.cluster_breakup.green_confirm_frames",
        ),
        target_loss_timeout_ms=breakup_float(
            "target_loss_timeout_ms", 500.0
        ),
        motion_phase_timeout_s=breakup_float("motion_phase_timeout_s", 10.0),
        post_breakup_retreat_enabled=breakup_raw.get(
            "post_breakup_retreat_enabled", True
        ),
    )
    max_wheel_velocity_m_s = _finite_float(
        motion_raw.get("max_wheel_velocity_m_s", 0.30),
        "motion.max_wheel_velocity_m_s",
        minimum=0.001,
    )
    min_wheel_velocity_m_s = _finite_float(
        motion_raw.get("min_wheel_velocity_m_s", 0.02),
        "motion.min_wheel_velocity_m_s",
        minimum=0.001,
    )
    if min_wheel_velocity_m_s > max_wheel_velocity_m_s:
        raise ValueError(
            "motion.min_wheel_velocity_m_s must not exceed "
            "motion.max_wheel_velocity_m_s, got "
            f"{min_wheel_velocity_m_s!r} > {max_wheel_velocity_m_s!r}."
        )
    motion = MotionRuntimeConfig(
        enabled=motion_enabled,
        wheel_track_m=wheel_track_m,
        max_linear_velocity_m_s=_finite_float(
            motion_raw.get("max_linear_velocity_m_s", 0.25),
            "motion.max_linear_velocity_m_s",
            minimum=0.001,
        ),
        max_angular_velocity_rad_s=_finite_float(
            motion_raw.get("max_angular_velocity_rad_s", 1.0),
            "motion.max_angular_velocity_rad_s",
            minimum=0.001,
        ),
        max_wheel_velocity_m_s=max_wheel_velocity_m_s,
        max_linear_acceleration_m_s2=_finite_float(
            motion_raw.get("max_linear_acceleration_m_s2", 0.50),
            "motion.max_linear_acceleration_m_s2",
            minimum=0.001,
        ),
        max_linear_deceleration_m_s2=_finite_float(
            motion_raw.get("max_linear_deceleration_m_s2", 0.50),
            "motion.max_linear_deceleration_m_s2",
            minimum=0.001,
        ),
        max_angular_acceleration_rad_s2=_finite_float(
            motion_raw.get("max_angular_acceleration_rad_s2", 4.0),
            "motion.max_angular_acceleration_rad_s2",
            minimum=0.001,
        ),
        max_angular_deceleration_rad_s2=_finite_float(
            motion_raw.get("max_angular_deceleration_rad_s2", 4.0),
            "motion.max_angular_deceleration_rad_s2",
            minimum=0.001,
        ),
        min_wheel_velocity_m_s=min_wheel_velocity_m_s,
        left_wheel_speed_weight=_finite_float(
            motion_raw.get("left_wheel_speed_weight", 1.0),
            "motion.left_wheel_speed_weight",
            minimum=0.001,
        ),
        right_wheel_speed_weight=_finite_float(
            motion_raw.get("right_wheel_speed_weight", 1.0),
            "motion.right_wheel_speed_weight",
            minimum=0.001,
        ),
        max_remote_command_valid_for_ms=max_remote_validity,
        synchronization_timeout_s=synchronization_timeout_s,
        stall_guard_enabled=stall_guard_enabled,
        stall_guard_timeout_ms=stall_guard_timeout_ms,
        stall_guard_min_command_speed_m_s=stall_guard_min_command_speed_m_s,
        stall_guard_stationary_encoder_delta_count=(
            stall_guard_stationary_encoder_delta_count
        ),
        gripper=gripper,
        odometry=odometry,
        cluster_breakup=cluster_breakup,
    )
    if cluster_breakup.enabled:
        if not motion.enabled or not motion.gripper.enabled or not odometry.enabled:
            raise ValueError(
                "Enabled motion.cluster_breakup requires motion, gripper and "
                "odometry to be enabled."
            )
        for name in (
            "departure_speed_m_s",
            "approach_speed_m_s",
            "breakup_speed_m_s",
            "retreat_speed_m_s",
        ):
            if getattr(cluster_breakup, name) > motion.max_linear_velocity_m_s:
                raise ValueError(
                    f"motion.cluster_breakup.{name} exceeds "
                    "motion.max_linear_velocity_m_s."
                )
        for name in (
            "search_angular_velocity_rad_s",
            "center_max_angular_velocity_rad_s",
            "scan_green_angular_velocity_rad_s",
        ):
            if getattr(cluster_breakup, name) > motion.max_angular_velocity_rad_s:
                raise ValueError(
                    f"motion.cluster_breakup.{name} exceeds "
                    "motion.max_angular_velocity_rad_s."
                )

    match_raw = _mapping(
        root.get("match", {}),
        "match",
    )
    match_keys = {
        "breakup_min_penetration_mm",
        "breakup_retreat_clearance_mm",
        "breakup_push_margin_mm",
        "breakup_braking_margin_mm",
        "breakup_deceleration_m_s2",
        "breakup_max_attempts",
        "breakup_confirmation_frames",
        "breakup_no_plan_reobserve_ms",

        "robot_footprint_radius_mm",
        "safety_margin_mm",
        "enabled",
        "startup_turn_angle_rad",
        "startup_turn_angular_velocity_rad_s",
        "startup_turn_settle_time_s",
        "startup_forward_distance_m",
        "startup_forward_speed_m_s",
        "startup_forward_settle_time_s",
        "startup_straight_pid_kp_rad_s_per_m_s",
        "startup_straight_pid_ki_rad_s_per_m",
        "startup_straight_pid_kd_rad_s_per_m_s2",
        "startup_straight_pid_integral_limit",
        "startup_straight_pid_deadband_m_s",
        "startup_straight_pid_max_angular_velocity_rad_s",
        "cluster_search_angular_velocity_rad_s",
        "cluster_search_empty_angular_velocity_rad_s",
        "cluster_search_sweep_angle_rad",
        "cluster_min_detections",
        "cluster_group_ground_mm",
        "cluster_breakup_standoff_mm",
        "cluster_approach_speed_m_s",
        "cluster_align_tolerance_rad",
        "cluster_align_hold_ms",
        "cluster_align_kp_rad_s",
        "cluster_align_max_angular_velocity_rad_s",
        "cluster_relocate_distance_m",
        "cluster_relocate_speed_m_s",
        "opportunistic_single_green_enabled",
        "opportunistic_single_green_clearance_mm",
        "misgrasp_breakup_forward_distance_m",
        "misgrasp_breakup_backward_distance_m",
        "misgrasp_breakup_max_heading_change_rad",
        "breakup_forward_distance_m",
        "breakup_forward_speed_m_s",
        "breakup_backward_distance_m",
        "breakup_backward_speed_m_s",
        "breakup_max_linear_acceleration_m_s2",
        "breakup_max_linear_deceleration_m_s2",
        "breakup_max_angular_acceleration_rad_s2",
        "breakup_max_angular_deceleration_rad_s2",
        "close_gripper_spin_angular_velocity_rad_s",
        "spin_angle_rad",
        "green_path_half_width_mm",
        "green_max_age_ms",
        "green_align_hold_ms",
        "green_alignment_tolerance_mm",
        "green_alignment_hysteresis_mm",
        "grasp_task_timeout_ms",
        "green_alignment_timeout_ms",
        "green_alignment_stable_frames",
        "green_alignment_kp_rad_s",
        "green_alignment_max_angular_velocity_rad_s",
        "green_alignment_min_wheel_velocity_m_s",
        "pickup_cruise_speed_scale",
        "pickup_terminal_speed_gain_s_inv",
        "green_approach_speed_m_s",
        "transport_rotate_angular_velocity_rad_s",
        "safe_zone_key_search_angular_velocity_rad_s",
        "safe_zone_bbox_turn_max_angular_velocity_rad_s",
        "safe_zone_bbox_edge_margin_px",
        "safe_zone_keypoint_reobserve_timeout_s",
        "safe_zone_keypoint_reverse_speed_m_s",
        "safe_zone_keypoint_reverse_max_distance_m",
        "transport_align_tolerance_mm",
        "safe_zone_grab_to_d1_speed_m_s",
        "safe_zone_d1_to_d2_speed_m_s",
        "safe_zone_d2_to_final_speed_m_s",
        "safe_zone_orange_d2_to_final_speed_m_s",
        "safe_zone_d2_to_final_max_linear_acceleration_m_s2",
        "safe_zone_d2_to_final_max_linear_deceleration_m_s2",
        "safe_zone_d2_to_final_max_angular_acceleration_rad_s2",
        "safe_zone_d2_to_final_max_angular_deceleration_rad_s2",
        "breakup_settle_time_s",
        "action_settle_time_s",
        "breakup_field_half_extent_mm",
        "breakup_gripper_offset_mm",
        "safe_zone_fallback_target_field_mm",
        "safe_zone_injured_target_field_mm",
        "safe_zone_fallback_heading_tolerance_rad",
        "safe_zone_fallback_heading_kp_rad_s",
        "safe_zone_fallback_max_angular_velocity_rad_s",
        "safe_zone_calibration_min_offset_mm",
        "safe_zone_calibration_start_offset_mm",
        "safe_zone_open_offset_mm",
        "safe_zone_d2_braking_overrun_x_mm",
        "safe_zone_d2_braking_overrun_y_mm",
        "safe_zone_d2_to_final_braking_overrun_mm",
        "safe_zone_orange_d2_to_final_braking_overrun_mm",
        "safe_zone_calibration_stop_speed_threshold_m_s",
        "safe_zone_calibration_stop_confirm_time_s",
        "safe_zone_exit_distance_m",
        "required_transports",
        "return_backup_speed_m_s",
        "nb_opening_actions",
        "nb_opening_gripper_after_action",
        "nb_opening_gripper_left_deg",
        "nb_opening_gripper_right_deg",
        "nb_opening_turn_tolerance_rad",
        "nb_opening_distance_tolerance_m",
        "nb_opening_settle_time_s",
        "nb_opening_turn_timeout_s",
        "nb_opening_execution_response_s",
        "nb_opening_effective_linear_deceleration_m_s2",
        "nb_opening_effective_angular_deceleration_rad_s2",
        "nb_opening_max_telemetry_age_ms",
        "nb_opening_fine_linear_speed_m_s",
        "nb_opening_fine_angular_velocity_rad_s",
        "nb_opening_stop_wheel_speed_m_s",
        "nb_opening_stop_angular_velocity_rad_s",
        "nb_opening_heading_tolerance_rad",
        "nb_opening_heading_kp_rad_s",
        "nb_opening_heading_max_angular_velocity_rad_s",
        "nb_opening_correction_max_distance_m",
        "nb_opening_correction_max_angle_rad",
        "nb_opening_correction_timeout_s",
    }
    _reject_unknown(match_raw, match_keys, "match")
    match_enabled = match_raw.get("enabled", False)
    if not isinstance(match_enabled, bool):
        raise ValueError("match.enabled must be a boolean.")
    opportunistic_single_green_enabled = match_raw.get(
        "opportunistic_single_green_enabled", False
    )
    if not isinstance(opportunistic_single_green_enabled, bool):
        raise ValueError(
            "match.opportunistic_single_green_enabled must be "
            "a boolean."
        )

    def match_float(name: str, default: float) -> float:
        return _finite_float(
            match_raw.get(name, default),
            f"match.{name}",
            minimum=0.001,
        )

    def match_nonnegative_float(name: str, default: float) -> float:
        return _finite_float(
            match_raw.get(name, default),
            f"match.{name}",
            minimum=0.0,
        )

    def match_optional_positive_float(
        name: str,
    ) -> float | None:
        value = match_raw.get(name)
        if value is None:
            return None
        return _finite_float(
            value,
            f"match.{name}",
            minimum=0.001,
        )

    fallback_target_value = match_raw.get(
        "safe_zone_fallback_target_field_mm", [-165.0, 1440.0]
    )
    if (
        not isinstance(fallback_target_value, list)
        or len(fallback_target_value) != 2
    ):
        raise ValueError(
            "match.safe_zone_fallback_target_field_mm must be "
            "[x_mm, y_mm]."
        )
    fallback_target_field = FieldPoint(
        _finite_float(
            fallback_target_value[0],
            "match.safe_zone_fallback_target_field_mm[0]",
            minimum=-float("inf"),
        ),
        _finite_float(
            fallback_target_value[1],
            "match.safe_zone_fallback_target_field_mm[1]",
            minimum=-float("inf"),
        ),
    )
    injured_target_value = match_raw.get(
        "safe_zone_injured_target_field_mm", [165.0, 1440.0]
    )
    if not isinstance(injured_target_value, list) or len(injured_target_value) != 2:
        raise ValueError(
            "match.safe_zone_injured_target_field_mm must be [x_mm, y_mm]."
        )
    injured_target_field = FieldPoint(
        _finite_float(
            injured_target_value[0],
            "match.safe_zone_injured_target_field_mm[0]",
            minimum=-float("inf"),
        ),
        _finite_float(
            injured_target_value[1],
            "match.safe_zone_injured_target_field_mm[1]",
            minimum=-float("inf"),
        ),
    )

    def parse_nb_opening_actions() -> tuple[NBOpeningAction, ...]:
        if "nb_opening_actions" not in match_raw:
            return _DEFAULT_NB_OPENING_ACTIONS
        raw_actions = match_raw["nb_opening_actions"]
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError(
                "match.nb_opening_actions must be a non-empty list."
            )
        actions: list[NBOpeningAction] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            location = f"match.nb_opening_actions[{index - 1}]"
            action = _mapping(raw_action, location)
            action_type = _required(action, "type", location)
            if action_type == "turn":
                _reject_unknown(
                    action,
                    {"type", "angle_rad", "angular_velocity_rad_s"},
                    location,
                )
                angle = _finite_float(
                    _required(action, "angle_rad", location),
                    f"{location}.angle_rad",
                    minimum=-float("inf"),
                )
                speed = _finite_float(
                    _required(action, "angular_velocity_rad_s", location),
                    f"{location}.angular_velocity_rad_s",
                    minimum=0.001,
                )
                actions.append(NBOpeningTurn(angle, speed))
            elif action_type == "straight":
                _reject_unknown(
                    action,
                    {"type", "distance_m", "speed_m_s"},
                    location,
                )
                distance = _finite_float(
                    _required(action, "distance_m", location),
                    f"{location}.distance_m",
                    minimum=-float("inf"),
                )
                speed = _finite_float(
                    _required(action, "speed_m_s", location),
                    f"{location}.speed_m_s",
                    minimum=0.001,
                )
                actions.append(NBOpeningStraight(distance, speed))
            else:
                raise ValueError(
                    f"{location}.type must be 'turn' or 'straight', "
                    f"got {action_type!r}."
                )
        return tuple(actions)

    nb_opening_actions = parse_nb_opening_actions()
    gripper_after_action_raw = match_raw.get(
        "nb_opening_gripper_after_action", 2
    )
    nb_opening_gripper_after_action = (
        None
        if gripper_after_action_raw is None
        else _positive_int(
            gripper_after_action_raw,
            "match.nb_opening_gripper_after_action",
        )
    )

    match = MatchRuntimeConfig(
        breakup_min_penetration_mm=match_float("breakup_min_penetration_mm", 10.0),
        breakup_retreat_clearance_mm=match_float("breakup_retreat_clearance_mm", 20.0),
        breakup_push_margin_mm=match_float("breakup_push_margin_mm", 30.0),
        breakup_braking_margin_mm=match_float("breakup_braking_margin_mm", 20.0),
        breakup_deceleration_m_s2=match_float("breakup_deceleration_m_s2", 0.8),
        breakup_max_attempts=_positive_int(match_raw.get("breakup_max_attempts", 2), "match.breakup_max_attempts"),
        breakup_confirmation_frames=_positive_int(
            match_raw.get("breakup_confirmation_frames", 3),
            "match.breakup_confirmation_frames",
        ),
        breakup_no_plan_reobserve_ms=match_float(
            "breakup_no_plan_reobserve_ms", 1_000.0
        ),
        robot_footprint_radius_mm=match_float("robot_footprint_radius_mm", 160.0),
        safety_margin_mm=match_float("safety_margin_mm", 80.0),
        enabled=match_enabled,
        startup_turn_angle_rad=match_float(
            "startup_turn_angle_rad", math.radians(45.0)
        ),
        startup_turn_angular_velocity_rad_s=_signed_angular_velocity(
            match_raw.get("startup_turn_angular_velocity_rad_s", -0.30),
            "match.startup_turn_angular_velocity_rad_s",
        ),
        startup_turn_settle_time_s=match_float(
            "startup_turn_settle_time_s", 0.5
        ),
        startup_forward_distance_m=match_float(
            "startup_forward_distance_m", 1.2
        ),
        startup_forward_speed_m_s=match_float(
            "startup_forward_speed_m_s", 0.12
        ),
        startup_forward_settle_time_s=match_float(
            "startup_forward_settle_time_s", 0.5
        ),
        startup_straight_pid_kp_rad_s_per_m_s=match_nonnegative_float(
            "startup_straight_pid_kp_rad_s_per_m_s", 0.0
        ),
        startup_straight_pid_ki_rad_s_per_m=match_nonnegative_float(
            "startup_straight_pid_ki_rad_s_per_m", 0.0
        ),
        startup_straight_pid_kd_rad_s_per_m_s2=match_nonnegative_float(
            "startup_straight_pid_kd_rad_s_per_m_s2", 0.0
        ),
        startup_straight_pid_integral_limit=match_float(
            "startup_straight_pid_integral_limit", 0.10
        ),
        startup_straight_pid_deadband_m_s=match_nonnegative_float(
            "startup_straight_pid_deadband_m_s", 0.0
        ),
        startup_straight_pid_max_angular_velocity_rad_s=match_float(
            "startup_straight_pid_max_angular_velocity_rad_s", 0.12
        ),
        cluster_search_angular_velocity_rad_s=_signed_angular_velocity(
            match_raw.get("cluster_search_angular_velocity_rad_s", -0.30),
            "match.cluster_search_angular_velocity_rad_s",
        ),
        cluster_search_empty_angular_velocity_rad_s=_signed_angular_velocity(
            match_raw.get("cluster_search_empty_angular_velocity_rad_s", -0.45),
            "match.cluster_search_empty_angular_velocity_rad_s",
        ),
        cluster_search_sweep_angle_rad=match_float(
            "cluster_search_sweep_angle_rad", 2.0 * math.pi
        ),
        cluster_min_detections=_positive_int(
            match_raw.get("cluster_min_detections", 2),
            "match.cluster_min_detections",
        ),
        cluster_group_ground_mm=match_float(
            "cluster_group_ground_mm", 250.0
        ),
        cluster_breakup_standoff_mm=match_float(
            "cluster_breakup_standoff_mm", 260.0
        ),
        cluster_approach_speed_m_s=match_float(
            "cluster_approach_speed_m_s", 0.12
        ),
        cluster_align_tolerance_rad=match_float(
            "cluster_align_tolerance_rad", 0.10
        ),
        cluster_align_hold_ms=match_float(
            "cluster_align_hold_ms", 500.0
        ),
        cluster_align_kp_rad_s=match_float(
            "cluster_align_kp_rad_s", 1.0
        ),
        cluster_align_max_angular_velocity_rad_s=match_float(
            "cluster_align_max_angular_velocity_rad_s", 0.35
        ),
        cluster_relocate_distance_m=match_float(
            "cluster_relocate_distance_m", 0.30
        ),
        cluster_relocate_speed_m_s=match_float(
            "cluster_relocate_speed_m_s", 0.10
        ),
        opportunistic_single_green_enabled=opportunistic_single_green_enabled,
        opportunistic_single_green_clearance_mm=match_float(
            "opportunistic_single_green_clearance_mm", 120.0
        ),
        misgrasp_breakup_forward_distance_m=match_float("misgrasp_breakup_forward_distance_m", 0.30),
        misgrasp_breakup_backward_distance_m=match_float("misgrasp_breakup_backward_distance_m", 0.08),
        misgrasp_breakup_max_heading_change_rad=match_float("misgrasp_breakup_max_heading_change_rad", math.radians(10.0)),
        breakup_forward_distance_m=match_float(
            "breakup_forward_distance_m", 0.5
        ),
        breakup_forward_speed_m_s=match_float(
            "breakup_forward_speed_m_s", 0.40
        ),
        breakup_backward_distance_m=match_float(
            "breakup_backward_distance_m", 0.3
        ),
        breakup_backward_speed_m_s=match_float(
            "breakup_backward_speed_m_s", 0.08
        ),
        breakup_max_linear_acceleration_m_s2=match_optional_positive_float(
            "breakup_max_linear_acceleration_m_s2"
        ),
        breakup_max_linear_deceleration_m_s2=match_optional_positive_float(
            "breakup_max_linear_deceleration_m_s2"
        ),
        breakup_max_angular_acceleration_rad_s2=match_optional_positive_float(
            "breakup_max_angular_acceleration_rad_s2"
        ),
        breakup_max_angular_deceleration_rad_s2=match_optional_positive_float(
            "breakup_max_angular_deceleration_rad_s2"
        ),
        close_gripper_spin_angular_velocity_rad_s=_signed_angular_velocity(
            match_raw.get("close_gripper_spin_angular_velocity_rad_s", -0.30),
            "match.close_gripper_spin_angular_velocity_rad_s",
        ),
        spin_angle_rad=match_float("spin_angle_rad", 2.0 * math.pi),
        green_path_half_width_mm=match_float(
            "green_path_half_width_mm", 50.0
        ),
        green_max_age_ms=match_float("green_max_age_ms", 500.0),
        green_align_hold_ms=match_float("green_align_hold_ms", 500.0),
        green_alignment_tolerance_mm=match_float(
            "green_alignment_tolerance_mm", 10.0
        ),
        green_alignment_hysteresis_mm=match_float(
            "green_alignment_hysteresis_mm", 5.0
        ),
        grasp_task_timeout_ms=match_float("grasp_task_timeout_ms", 20_000.0),
        green_alignment_timeout_ms=match_float(
            "green_alignment_timeout_ms", 8_000.0
        ),
        green_alignment_stable_frames=_positive_int(
            match_raw.get("green_alignment_stable_frames", 3),
            "match.green_alignment_stable_frames",
        ),
        green_alignment_kp_rad_s=match_float(
            "green_alignment_kp_rad_s", 1.0
        ),
        green_alignment_max_angular_velocity_rad_s=match_float(
            "green_alignment_max_angular_velocity_rad_s", 0.35
        ),
        green_alignment_min_wheel_velocity_m_s=match_float(
            "green_alignment_min_wheel_velocity_m_s", 0.01
        ),
        pickup_cruise_speed_scale=match_float("pickup_cruise_speed_scale", 1.0),
        pickup_terminal_speed_gain_s_inv=match_float(
            "pickup_terminal_speed_gain_s_inv", 1.0
        ),
        green_approach_speed_m_s=match_float(
            "green_approach_speed_m_s", 0.08
        ),
        transport_rotate_angular_velocity_rad_s=match_float(
            "transport_rotate_angular_velocity_rad_s", 0.30
        ),
        safe_zone_key_search_angular_velocity_rad_s=match_float(
            "safe_zone_key_search_angular_velocity_rad_s", 0.45
        ),
        safe_zone_bbox_turn_max_angular_velocity_rad_s=match_float(
            "safe_zone_bbox_turn_max_angular_velocity_rad_s", 0.25
        ),
        safe_zone_bbox_edge_margin_px=match_nonnegative_float(
            "safe_zone_bbox_edge_margin_px", 2.0
        ),
        safe_zone_keypoint_reobserve_timeout_s=match_float(
            "safe_zone_keypoint_reobserve_timeout_s", 0.80
        ),
        safe_zone_keypoint_reverse_speed_m_s=match_float(
            "safe_zone_keypoint_reverse_speed_m_s", 0.08
        ),
        safe_zone_keypoint_reverse_max_distance_m=match_float(
            "safe_zone_keypoint_reverse_max_distance_m", 0.10
        ),
        transport_align_tolerance_mm=match_float(
            "transport_align_tolerance_mm", 30.0
        ),
        safe_zone_grab_to_d1_speed_m_s=match_float(
            "safe_zone_grab_to_d1_speed_m_s", 0.08
        ),
        safe_zone_d1_to_d2_speed_m_s=match_float(
            "safe_zone_d1_to_d2_speed_m_s", 0.08
        ),
        safe_zone_d2_to_final_speed_m_s=match_float(
            "safe_zone_d2_to_final_speed_m_s", 0.08
        ),
        safe_zone_orange_d2_to_final_speed_m_s=match_float(
            "safe_zone_orange_d2_to_final_speed_m_s", 0.08
        ),
        safe_zone_d2_to_final_max_linear_acceleration_m_s2=match_optional_positive_float(
            "safe_zone_d2_to_final_max_linear_acceleration_m_s2"
        ),
        safe_zone_d2_to_final_max_linear_deceleration_m_s2=match_optional_positive_float(
            "safe_zone_d2_to_final_max_linear_deceleration_m_s2"
        ),
        safe_zone_d2_to_final_max_angular_acceleration_rad_s2=match_optional_positive_float(
            "safe_zone_d2_to_final_max_angular_acceleration_rad_s2"
        ),
        safe_zone_d2_to_final_max_angular_deceleration_rad_s2=match_optional_positive_float(
            "safe_zone_d2_to_final_max_angular_deceleration_rad_s2"
        ),
        breakup_settle_time_s=match_float(
            "breakup_settle_time_s", 0.5
        ),
        action_settle_time_s=match_nonnegative_float(
            "action_settle_time_s", 0.0
        ),
        breakup_field_half_extent_mm=match_float(
            "breakup_field_half_extent_mm", 1500.0
        ),
        breakup_gripper_offset_mm=match_float(
            "breakup_gripper_offset_mm", 200.0
        ),
        safe_zone_fallback_target_field=fallback_target_field,
        safe_zone_injured_target_field=injured_target_field,
        safe_zone_fallback_heading_tolerance_rad=match_float(
            "safe_zone_fallback_heading_tolerance_rad", 0.20
        ),
        safe_zone_fallback_heading_kp_rad_s=match_float(
            "safe_zone_fallback_heading_kp_rad_s", 1.0
        ),
        safe_zone_fallback_max_angular_velocity_rad_s=match_float(
            "safe_zone_fallback_max_angular_velocity_rad_s", 0.30
        ),
        safe_zone_calibration_min_offset_mm=match_float(
            "safe_zone_calibration_min_offset_mm", 300.0
        ),
        safe_zone_calibration_start_offset_mm=match_float(
            "safe_zone_calibration_start_offset_mm", 500.0
        ),
        safe_zone_open_offset_mm=match_float(
            "safe_zone_open_offset_mm", 137.0
        ),
        safe_zone_d2_braking_overrun_x_mm=_finite_float(
            match_raw.get("safe_zone_d2_braking_overrun_x_mm", 0.0),
            "match.safe_zone_d2_braking_overrun_x_mm",
            minimum=-float("inf"),
        ),
        safe_zone_d2_braking_overrun_y_mm=_finite_float(
            match_raw.get("safe_zone_d2_braking_overrun_y_mm", 0.0),
            "match.safe_zone_d2_braking_overrun_y_mm",
            minimum=-float("inf"),
        ),
        safe_zone_d2_to_final_braking_overrun_mm=_finite_float(
            match_raw.get("safe_zone_d2_to_final_braking_overrun_mm", 0.0),
            "match.safe_zone_d2_to_final_braking_overrun_mm",
            minimum=0.0,
        ),
        safe_zone_orange_d2_to_final_braking_overrun_mm=_finite_float(
            match_raw.get(
                "safe_zone_orange_d2_to_final_braking_overrun_mm", 0.0
            ),
            "match.safe_zone_orange_d2_to_final_braking_overrun_mm",
            minimum=0.0,
        ),
        safe_zone_calibration_stop_speed_threshold_m_s=match_float(
            "safe_zone_calibration_stop_speed_threshold_m_s", 0.02
        ),
        safe_zone_calibration_stop_confirm_time_s=match_float(
            "safe_zone_calibration_stop_confirm_time_s", 0.30
        ),
        safe_zone_exit_distance_m=match_float(
            "safe_zone_exit_distance_m", 0.30
        ),
        required_transports=_positive_int(
            match_raw.get("required_transports", 4),
            "match.required_transports",
        ),
        return_backup_speed_m_s=match_float(
            "return_backup_speed_m_s", 0.08
        ),
        nb_opening_actions=nb_opening_actions,
        nb_opening_gripper_after_action=nb_opening_gripper_after_action,
        nb_opening_gripper_left_deg=match_float("nb_opening_gripper_left_deg", 50.0),
        nb_opening_gripper_right_deg=match_float("nb_opening_gripper_right_deg", 130.0),
        nb_opening_turn_tolerance_rad=match_float(
            "nb_opening_turn_tolerance_rad", 0.08
        ),
        nb_opening_distance_tolerance_m=match_float(
            "nb_opening_distance_tolerance_m", 0.03
        ),
        nb_opening_settle_time_s=match_nonnegative_float(
            "nb_opening_settle_time_s", 0.30
        ),
        nb_opening_turn_timeout_s=match_float("nb_opening_turn_timeout_s", 8.0),
        nb_opening_execution_response_s=match_float(
            "nb_opening_execution_response_s", 0.04
        ),
        nb_opening_effective_linear_deceleration_m_s2=match_optional_positive_float(
            "nb_opening_effective_linear_deceleration_m_s2"
        ),
        nb_opening_effective_angular_deceleration_rad_s2=match_optional_positive_float(
            "nb_opening_effective_angular_deceleration_rad_s2"
        ),
        nb_opening_max_telemetry_age_ms=match_float(
            "nb_opening_max_telemetry_age_ms", 160.0
        ),
        nb_opening_fine_linear_speed_m_s=match_float(
            "nb_opening_fine_linear_speed_m_s", 0.05
        ),
        nb_opening_fine_angular_velocity_rad_s=match_float(
            "nb_opening_fine_angular_velocity_rad_s", 0.12
        ),
        nb_opening_stop_wheel_speed_m_s=match_float(
            "nb_opening_stop_wheel_speed_m_s", 0.015
        ),
        nb_opening_stop_angular_velocity_rad_s=match_float(
            "nb_opening_stop_angular_velocity_rad_s", 0.06
        ),
        nb_opening_heading_tolerance_rad=match_float(
            "nb_opening_heading_tolerance_rad", 0.03
        ),
        nb_opening_heading_kp_rad_s=match_float(
            "nb_opening_heading_kp_rad_s", 2.0
        ),
        nb_opening_heading_max_angular_velocity_rad_s=match_float(
            "nb_opening_heading_max_angular_velocity_rad_s", 0.25
        ),
        nb_opening_correction_max_distance_m=match_float(
            "nb_opening_correction_max_distance_m", 0.08
        ),
        nb_opening_correction_max_angle_rad=match_float(
            "nb_opening_correction_max_angle_rad", 0.12
        ),
        nb_opening_correction_timeout_s=match_float(
            "nb_opening_correction_timeout_s", 0.50
        ),
    )
    match_cc = parse_match_cc_config(root.get("match_cc", {}))
    if match_cc.enabled and not match.enabled:
        raise ValueError("Enabled match_cc requires match.enabled=true for shared transport.")
    if match_cc.enabled:
        for name in (
            "target_approach_speed_m_s",
        ):
            if getattr(match_cc, name) > motion.max_linear_velocity_m_s:
                raise ValueError(f"match_cc.{name} exceeds motion.max_linear_velocity_m_s.")
        for name in (
            "target_search_angular_velocity_rad_s",
            "alignment_max_angular_velocity_rad_s",
        ):
            if abs(getattr(match_cc, name)) > motion.max_angular_velocity_rad_s:
                raise ValueError(f"match_cc.{name} exceeds motion.max_angular_velocity_rad_s.")
    if match.enabled:
        for index, action in enumerate(match.nb_opening_actions, start=1):
            if isinstance(action, NBOpeningTurn):
                if action.angular_velocity_rad_s > motion.max_angular_velocity_rad_s:
                    raise ValueError(
                        f"match.nb_opening_actions[{index - 1}] angular speed "
                        "exceeds motion.max_angular_velocity_rad_s."
                    )
                wheel_speed = (
                    action.angular_velocity_rad_s
                    * motion.wheel_track_m
                    / 2.0
                    * max(
                        motion.left_wheel_speed_weight,
                        motion.right_wheel_speed_weight,
                    )
                )
            else:
                if action.speed_m_s > motion.max_linear_velocity_m_s:
                    raise ValueError(
                        f"match.nb_opening_actions[{index - 1}] linear speed "
                        "exceeds motion.max_linear_velocity_m_s."
                    )
                wheel_speed = action.speed_m_s * max(
                    motion.left_wheel_speed_weight,
                    motion.right_wheel_speed_weight,
                )
            if wheel_speed > motion.max_wheel_velocity_m_s:
                raise ValueError(
                    f"match.nb_opening_actions[{index - 1}] requires wheel speed "
                    f"{wheel_speed:g} m/s, above motion.max_wheel_velocity_m_s."
                )
        for name in ("green_approach_speed_m_s", "safe_zone_grab_to_d1_speed_m_s"):
            scaled = getattr(match, name) * match.pickup_cruise_speed_scale
            if scaled > min(motion.max_linear_velocity_m_s, motion.max_wheel_velocity_m_s):
                raise ValueError(f"match.{name} scaled speed {scaled!r} exceeds motion limits.")
        for name in (
            "startup_forward_speed_m_s",
            "breakup_forward_speed_m_s",
            "breakup_backward_speed_m_s",
            "cluster_approach_speed_m_s",
            "cluster_relocate_speed_m_s",
            "green_approach_speed_m_s",
            "safe_zone_grab_to_d1_speed_m_s",
            "safe_zone_d1_to_d2_speed_m_s",
            "safe_zone_d2_to_final_speed_m_s",
            "safe_zone_orange_d2_to_final_speed_m_s",
        ):
            if getattr(match, name) > motion.max_linear_velocity_m_s:
                raise ValueError(
                    f"match.{name} exceeds "
                    "motion.max_linear_velocity_m_s."
                )
        for name in (
            "startup_turn_angular_velocity_rad_s",
            "cluster_search_angular_velocity_rad_s",
            "cluster_search_empty_angular_velocity_rad_s",
            "close_gripper_spin_angular_velocity_rad_s",
            "startup_straight_pid_max_angular_velocity_rad_s",
            "cluster_align_max_angular_velocity_rad_s",
            "green_alignment_max_angular_velocity_rad_s",
            "transport_rotate_angular_velocity_rad_s",
            "safe_zone_key_search_angular_velocity_rad_s",
            "safe_zone_bbox_turn_max_angular_velocity_rad_s",
            "safe_zone_fallback_max_angular_velocity_rad_s",
        ):
            if abs(getattr(match, name)) > motion.max_angular_velocity_rad_s:
                raise ValueError(
                    f"match.{name} exceeds "
                    "motion.max_angular_velocity_rad_s."
                )

    green_grab_raw = _mapping(root.get("green_grab", {}), "green_grab")
    _reject_unknown(
        green_grab_raw,
        {
            "enabled",
            "search_angular_velocity_rad_s",
            "approach_speed_m_s",
            "align_tolerance_ratio",
            "align_kp_rad_s",
            "align_max_angular_velocity_rad_s",
            "engage_bottom_fraction",
            "confirm_frames",
            "target_loss_timeout_ms",
        },
        "green_grab",
    )
    green_grab_enabled = green_grab_raw.get("enabled", False)
    if not isinstance(green_grab_enabled, bool):
        raise ValueError("green_grab.enabled must be a boolean.")

    def green_grab_float(name: str, default: float) -> float:
        return _finite_float(
            green_grab_raw.get(name, default),
            f"green_grab.{name}",
            minimum=0.001,
        )

    green_grab = GreenGrabRuntimeConfig(
        enabled=green_grab_enabled,
        search_angular_velocity_rad_s=_signed_angular_velocity(
            green_grab_raw.get("search_angular_velocity_rad_s", 0.30),
            "green_grab.search_angular_velocity_rad_s",
        ),
        approach_speed_m_s=green_grab_float("approach_speed_m_s", 0.08),
        align_tolerance_ratio=green_grab_float("align_tolerance_ratio", 0.06),
        align_kp_rad_s=green_grab_float("align_kp_rad_s", 1.2),
        align_max_angular_velocity_rad_s=green_grab_float(
            "align_max_angular_velocity_rad_s", 0.35
        ),
        engage_bottom_fraction=green_grab_float(
            "engage_bottom_fraction", 0.85
        ),
        confirm_frames=_positive_int(
            green_grab_raw.get("confirm_frames", 3),
            "green_grab.confirm_frames",
        ),
        target_loss_timeout_ms=green_grab_float(
            "target_loss_timeout_ms", 500.0
        ),
    )
    if green_grab.enabled:
        if green_grab.approach_speed_m_s > motion.max_linear_velocity_m_s:
            raise ValueError(
                "green_grab.approach_speed_m_s exceeds "
                "motion.max_linear_velocity_m_s."
            )
        for name in (
            "search_angular_velocity_rad_s",
            "align_max_angular_velocity_rad_s",
        ):
            if getattr(green_grab, name) > motion.max_angular_velocity_rad_s:
                raise ValueError(
                    f"green_grab.{name} exceeds "
                    "motion.max_angular_velocity_rad_s."
                )

    tracking_raw = _mapping(
        root.get("tracking", {}),
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
            tracking_raw.get("confirmation_hits", 2),
            "tracking.confirmation_hits",
        ),
        max_association_ground_mm=_finite_float(
            tracking_raw.get("max_association_ground_mm", 250.0),
            "tracking.max_association_ground_mm",
            minimum=0.001,
        ),
        min_association_iou=_threshold(
            tracking_raw.get("min_association_iou", 0.10),
            "tracking.min_association_iou",
        ),
        max_coast_ms=_finite_float(
            tracking_raw.get("max_coast_ms", 600.0),
            "tracking.max_coast_ms",
            minimum=0.001,
        ),
        confidence_decay_per_second=_finite_float(
            tracking_raw.get("confidence_decay_per_second", 0.8),
            "tracking.confidence_decay_per_second",
            minimum=0.001,
        ),
        min_confidence=_threshold(
            tracking_raw.get("min_confidence", 0.15),
            "tracking.min_confidence",
        ),
    )

    world_raw = _mapping(root.get("world", {}), "world")
    _reject_unknown(
        world_raw,
        {
            "max_visual_age_ms",
            "opponent_max_age_ms",
            "danger_confirm_threshold",
            "danger_suspect_threshold",
            "team_color",
            "static_map",
        },
        "world",
    )
    world_model = WorldModelConfig(
        max_visual_age_ms=_finite_float(
            world_raw.get("max_visual_age_ms", 250.0),
            "world.max_visual_age_ms",
            minimum=0.001,
        ),
        opponent_max_age_ms=_finite_float(
            world_raw.get("opponent_max_age_ms", 500.0),
            "world.opponent_max_age_ms",
            minimum=0.001,
        ),
        danger_confirm_threshold=_threshold(
            world_raw.get("danger_confirm_threshold", 0.60),
            "world.danger_confirm_threshold",
        ),
        danger_suspect_threshold=_threshold(
            world_raw.get("danger_suspect_threshold", 0.15),
            "world.danger_suspect_threshold",
        ),
    )
    try:
        team_color = TeamColor(
            _string(world_raw.get("team_color", "unknown"), "world.team_color")
        )
    except ValueError as exc:
        raise ValueError(
            "world.team_color must be red, blue or unknown."
        ) from exc

    static_map_raw = _mapping(world_raw.get("static_map", {}), "world.static_map")
    _reject_unknown(
        static_map_raw,
        {"center_cross", "safe_zone_landmarks", "regions"},
        "world.static_map",
    )
    default_map = default_static_field_map()
    center_raw = _mapping(
        static_map_raw.get("center_cross", {}),
        "world.static_map.center_cross",
    )
    _reject_unknown(
        center_raw,
        {"intersection_field_mm", "terminals"},
        "world.static_map.center_cross",
    )
    intersection_value = center_raw.get("intersection_field_mm", [0.0, 0.0])
    if not isinstance(intersection_value, list) or len(intersection_value) != 2:
        raise ValueError(
            "world.static_map.center_cross.intersection_field_mm must be "
            "[x_mm, y_mm]."
        )
    intersection_field = FieldPoint(
        _finite_float(
            intersection_value[0],
            "world.static_map.center_cross.intersection_field_mm[0]",
            minimum=-float("inf"),
        ),
        _finite_float(
            intersection_value[1],
            "world.static_map.center_cross.intersection_field_mm[1]",
            minimum=-float("inf"),
        ),
    )
    default_terminals = {
        item.ray.value: item.kind.value
        for item in default_map.center_cross.terminals
    }
    terminals_raw = _mapping(
        center_raw.get("terminals", default_terminals),
        "world.static_map.center_cross.terminals",
    )
    expected_rays = {ray.value for ray in CenterCrossRay}
    if set(terminals_raw) != expected_rays:
        raise ValueError(
            "world.static_map.center_cross.terminals keys must exactly be "
            f"{sorted(expected_rays)!r}."
        )
    terminals: list[CenterCrossTerminal] = []
    for ray in CenterCrossRay:
        location = f"world.static_map.center_cross.terminals.{ray.value}"
        try:
            terminal_kind = CenterLineTerminalKind(
                _string(terminals_raw[ray.value], location)
            )
        except ValueError as exc:
            raise ValueError(
                f"{location} must be red_safe_zone, blue_safe_zone, "
                "plain_boundary or unknown."
            ) from exc
        terminals.append(CenterCrossTerminal(ray, terminal_kind))
    center_cross_map = StaticCenterCross(
        intersection_field=intersection_field,
        terminals=tuple(terminals),
    )

    safe_landmarks_value = static_map_raw.get("safe_zone_landmarks", [])
    if not isinstance(safe_landmarks_value, list):
        raise ValueError("world.static_map.safe_zone_landmarks must be a list.")
    safe_zone_landmarks: list[StaticSafeZoneLandmarks] = []

    def parse_field_point(value: object, location: str) -> FieldPoint:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{location} must be [x_mm, y_mm].")
        return FieldPoint(
            _finite_float(value[0], f"{location}[0]", minimum=-float("inf")),
            _finite_float(value[1], f"{location}[1]", minimum=-float("inf")),
        )

    for index, value in enumerate(safe_landmarks_value):
        location = f"world.static_map.safe_zone_landmarks[{index}]"
        item = _mapping(value, location)
        _reject_unknown(
            item,
            {"color", "ground_anchor_field_mm", "near_field_corners_mm", "measured", "usable"},
            location,
        )
        try:
            color = TeamColor(_string(_required(item, "color", location), f"{location}.color"))
        except ValueError as exc:
            raise ValueError(f"{location}.color must be red or blue.") from exc
        corners = _required(item, "near_field_corners_mm", location)
        if not isinstance(corners, list) or len(corners) != 2:
            raise ValueError(f"{location}.near_field_corners_mm must contain two points.")
        measured = _required(item, "measured", location)
        usable = _required(item, "usable", location)
        if not isinstance(measured, bool) or not isinstance(usable, bool):
            raise ValueError(f"{location}.measured and usable must be booleans.")
        safe_zone_landmarks.append(StaticSafeZoneLandmarks(
            color,
            parse_field_point(_required(item, "ground_anchor_field_mm", location), f"{location}.ground_anchor_field_mm"),
            parse_field_point(corners[0], f"{location}.near_field_corners_mm[0]"),
            parse_field_point(corners[1], f"{location}.near_field_corners_mm[1]"),
            measured,
            usable,
        ))

    regions_value = static_map_raw.get("regions", [])
    if not isinstance(regions_value, list):
        raise ValueError("world.static_map.regions must be a list.")
    physical_regions: list[PhysicalStaticRegion] = []
    for index, value in enumerate(regions_value):
        location = f"world.static_map.regions[{index}]"
        region_raw = _mapping(value, location)
        _reject_unknown(
            region_raw,
            {"region_id", "kind", "polygon_field_mm"},
            location,
        )
        try:
            kind = PhysicalRegionKind(
                _string(
                    _required(region_raw, "kind", location),
                    f"{location}.kind",
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"{location}.kind must be field, red_material, red_injured, "
                "blue_material, blue_injured or start_zone."
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
        physical_regions.append(
            PhysicalStaticRegion(
                region_id=_string(
                    _required(region_raw, "region_id", location),
                    f"{location}.region_id",
                ),
                kind=kind,
                polygon_field=tuple(polygon),
            )
        )
    world = WorldRuntimeConfig(
        world_model,
        team_color,
        StaticFieldMap(
            center_cross_map,
            tuple(physical_regions),
            tuple(safe_zone_landmarks),
        ),
    )

    mission_raw = _mapping(
        root.get("mission", {}),
        "mission",
    )
    _reject_unknown(
        mission_raw,
        {
            "match_duration_s",
            "no_motion_timeout_s",
            "opponent_contact_timeout_s",
            "target_priority",
        },
        "mission",
    )
    priority_value = mission_raw.get(
        "target_priority",
        ["orange_injured", "black_core", "green_supply"],
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
            mission_raw.get("match_duration_s", 180.0),
            "mission.match_duration_s",
            minimum=0.001,
        ),
        no_motion_timeout_s=_finite_float(
            mission_raw.get("no_motion_timeout_s", 15.0),
            "mission.no_motion_timeout_s",
            minimum=0.001,
        ),
        opponent_contact_timeout_s=_finite_float(
            mission_raw.get("opponent_contact_timeout_s", 10.0),
            "mission.opponent_contact_timeout_s",
            minimum=0.001,
        ),
        target_priority=target_priority,
    )

    perception_overrides = _mapping(
        root.get("perception", {}),
        "perception",
    )
    perception_raw = _merge_defaults(
        _perception_defaults(),
        perception_overrides,
    )
    _reject_unknown(
        perception_raw,
        {
            "detection_threshold",
            "k0_threshold",
            "center_cross_refinement",
            "safe_zone_color",
            "color_classifier",
            "target_ground_geometry",
            "gripper_color",
        },
        "perception",
    )
    detection_threshold = _threshold(
        _required(
            perception_raw,
            "detection_threshold",
            "perception",
        ),
        "perception.detection_threshold",
    )
    k0_threshold = _threshold(
        _required(perception_raw, "k0_threshold", "perception"),
        "perception.k0_threshold",
    )
    for location, value in (
        ("perception.detection_threshold", detection_threshold),
        ("perception.k0_threshold", k0_threshold),
    ):
        if value > 1.0:
            raise ValueError(f"{location} must be <= 1.0.")

    cross_refinement_raw = _mapping(
        perception_raw["center_cross_refinement"],
        "perception.center_cross_refinement",
    )
    _reject_unknown(
        cross_refinement_raw,
        {
            "enabled", "canny_low", "canny_high", "hough_threshold",
            "min_line_length_px", "max_line_gap_px",
            "max_intersection_distance_px", "min_axis_angle_deg",
        },
        "perception.center_cross_refinement",
    )
    refinement_enabled = cross_refinement_raw["enabled"]
    if not isinstance(refinement_enabled, bool):
        raise ValueError("perception.center_cross_refinement.enabled must be a boolean.")
    canny_low = _nonnegative_int(cross_refinement_raw["canny_low"], "perception.center_cross_refinement.canny_low")
    canny_high = _positive_int(cross_refinement_raw["canny_high"], "perception.center_cross_refinement.canny_high")
    if canny_low >= canny_high or canny_high > 255:
        raise ValueError("center-cross Canny thresholds must satisfy 0 <= low < high <= 255.")
    center_cross_refinement = CenterCrossRefinementConfig(
        enabled=refinement_enabled,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=_positive_int(cross_refinement_raw["hough_threshold"], "perception.center_cross_refinement.hough_threshold"),
        min_line_length_px=_finite_float(cross_refinement_raw["min_line_length_px"], "perception.center_cross_refinement.min_line_length_px", minimum=0.001),
        max_line_gap_px=_finite_float(cross_refinement_raw["max_line_gap_px"], "perception.center_cross_refinement.max_line_gap_px", minimum=0.0),
        max_intersection_distance_px=_finite_float(cross_refinement_raw["max_intersection_distance_px"], "perception.center_cross_refinement.max_intersection_distance_px", minimum=0.001),
        min_axis_angle_deg=_finite_float(cross_refinement_raw["min_axis_angle_deg"], "perception.center_cross_refinement.min_axis_angle_deg", minimum=0.001),
    )
    if center_cross_refinement.min_axis_angle_deg > 90.0:
        raise ValueError("perception.center_cross_refinement.min_axis_angle_deg must be <= 90.")

    safe_color_raw = _mapping(perception_raw["safe_zone_color"], "perception.safe_zone_color")
    _reject_unknown(safe_color_raw, {"enabled", "red_hsv_ranges", "blue_hsv_ranges", "min_fraction", "min_margin"}, "perception.safe_zone_color")
    safe_color_enabled = safe_color_raw["enabled"]
    if not isinstance(safe_color_enabled, bool):
        raise ValueError("perception.safe_zone_color.enabled must be a boolean.")

    def parse_safe_ranges(name: str) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
        values = safe_color_raw[name]
        if not isinstance(values, list):
            raise ValueError(f"perception.safe_zone_color.{name} must be a list.")
        parsed = []
        for index, value in enumerate(values):
            item = _mapping(value, f"perception.safe_zone_color.{name}[{index}]")
            _reject_unknown(item, {"lower", "upper"}, f"perception.safe_zone_color.{name}[{index}]")
            hsv_range = HsvRange(
                _hsv_triplet(item.get("lower"), f"perception.safe_zone_color.{name}[{index}].lower"),
                _hsv_triplet(item.get("upper"), f"perception.safe_zone_color.{name}[{index}].upper"),
            )
            parsed.append((hsv_range.lower, hsv_range.upper))
        return tuple(parsed)

    red_safe_ranges = parse_safe_ranges("red_hsv_ranges")
    blue_safe_ranges = parse_safe_ranges("blue_hsv_ranges")
    if safe_color_enabled and (not red_safe_ranges or not blue_safe_ranges):
        raise ValueError("enabled safe-zone color classification requires red and blue HSV ranges.")
    safe_zone_color = SafeZoneColorConfig(
        enabled=safe_color_enabled,
        red_hsv_ranges=red_safe_ranges,
        blue_hsv_ranges=blue_safe_ranges,
        min_fraction=_threshold(safe_color_raw["min_fraction"], "perception.safe_zone_color.min_fraction"),
        min_margin=_threshold(safe_color_raw["min_margin"], "perception.safe_zone_color.min_margin"),
    )

    color_raw = _mapping(
        _required(
            perception_raw,
            "color_classifier",
            "perception",
        ),
        "perception.color_classifier",
    )
    _reject_unknown(
        color_raw,
        {
            "ranges",
            "min_color_fraction",
            "min_color_dominance",
            "min_dominance_margin",
            "morphology_kernel_size",
            "open_iterations",
            "close_iterations",
            "min_component_area_fraction",
        },
        "perception.color_classifier",
    )
    ranges_raw = _mapping(
        _required(
            color_raw,
            "ranges",
            "perception.color_classifier",
        ),
        "perception.color_classifier.ranges",
    )
    expected_color_names = {
        target_class.value for target_class in COLOR_TARGET_CLASSES
    }
    if set(ranges_raw) != expected_color_names:
        raise ValueError(
            "perception.color_classifier.ranges keys must exactly be "
            f"{sorted(expected_color_names)!r}."
        )
    parsed_ranges: dict[TargetClass, tuple[HsvRange, ...]] = {}
    for target_class in COLOR_TARGET_CLASSES:
        range_values = ranges_raw[target_class.value]
        range_location = (
            "perception.color_classifier.ranges."
            f"{target_class.value}"
        )
        parsed_ranges[target_class] = _hsv_ranges(range_values, range_location)

    color_classifier = HsvColorClassifierConfig(
        green_supply=parsed_ranges[TargetClass.GREEN_SUPPLY],
        black_core=parsed_ranges[TargetClass.BLACK_CORE],
        orange_injured=parsed_ranges[TargetClass.ORANGE_INJURED],
        blue_danger=parsed_ranges[TargetClass.BLUE_DANGER],
        min_color_fraction=_threshold(
            _required(
                color_raw,
                "min_color_fraction",
                "perception.color_classifier",
            ),
            "perception.color_classifier.min_color_fraction",
        ),
        min_color_dominance=_threshold(
            _required(
                color_raw,
                "min_color_dominance",
                "perception.color_classifier",
            ),
            "perception.color_classifier.min_color_dominance",
        ),
        min_dominance_margin=_threshold(
            _required(
                color_raw,
                "min_dominance_margin",
                "perception.color_classifier",
            ),
            "perception.color_classifier.min_dominance_margin",
        ),
        morphology_kernel_size=_positive_int(
            _required(
                color_raw,
                "morphology_kernel_size",
                "perception.color_classifier",
            ),
            "perception.color_classifier.morphology_kernel_size",
        ),
        open_iterations=_nonnegative_int(
            _required(
                color_raw,
                "open_iterations",
                "perception.color_classifier",
            ),
            "perception.color_classifier.open_iterations",
        ),
        close_iterations=_nonnegative_int(
            _required(
                color_raw,
                "close_iterations",
                "perception.color_classifier",
            ),
            "perception.color_classifier.close_iterations",
        ),
        min_component_area_fraction=_threshold(
            _required(
                color_raw,
                "min_component_area_fraction",
                "perception.color_classifier",
            ),
            "perception.color_classifier.min_component_area_fraction",
        ),
    )
    target_geometry_raw = _mapping(
        _required(
            perception_raw,
            "target_ground_geometry",
            "perception",
        ),
        "perception.target_ground_geometry",
    )
    _reject_unknown(
        target_geometry_raw,
        {"enabled", "objects", "fitting"},
        "perception.target_ground_geometry",
    )
    target_geometry_enabled = _required(
        target_geometry_raw,
        "enabled",
        "perception.target_ground_geometry",
    )
    if not isinstance(target_geometry_enabled, bool):
        raise ValueError(
            "perception.target_ground_geometry.enabled must be a boolean."
        )
    target_objects_raw = _mapping(
        _required(
            target_geometry_raw,
            "objects",
            "perception.target_ground_geometry",
        ),
        "perception.target_ground_geometry.objects",
    )
    expected_target_geometry_names = {
        item.value for item in COLOR_TARGET_CLASSES
    }
    if set(target_objects_raw) != expected_target_geometry_names:
        raise ValueError(
            "perception.target_ground_geometry.objects keys must exactly "
            f"be {sorted(expected_target_geometry_names)!r}."
        )
    target_geometries = {
        target_class: _target_geometry(
            target_objects_raw[target_class.value],
            "perception.target_ground_geometry.objects."
            f"{target_class.value}",
        )
        for target_class in COLOR_TARGET_CLASSES
    }
    target_fitting_raw = _mapping(
        _required(
            target_geometry_raw,
            "fitting",
            "perception.target_ground_geometry",
        ),
        "perception.target_ground_geometry.fitting",
    )
    target_fitting_names = {
        "coarse_center_step_mm",
        "coarse_yaw_step_deg",
        "refine_center_step_mm",
        "refine_center_radius_mm",
        "refine_top_candidates",
        "refine_yaw_step_deg",
        "refine_yaw_radius_deg",
        "search_radius_margin_mm",
        "silhouette_weight",
        "contour_weight",
        "contact_weight",
        "contour_distance_scale_px",
        "contact_distance_scale_px",
        "max_contact_residual_px",
        "ambiguity_score_delta",
        "max_center_uncertainty_mm",
        "min_fit_score",
        "min_silhouette_iou",
    }
    _reject_unknown(
        target_fitting_raw,
        target_fitting_names,
        "perception.target_ground_geometry.fitting",
    )
    target_ground_geometry = TargetGroundGeometryConfig(
        enabled=target_geometry_enabled,
        green_supply=target_geometries[TargetClass.GREEN_SUPPLY],
        black_core=target_geometries[TargetClass.BLACK_CORE],
        orange_injured=target_geometries[TargetClass.ORANGE_INJURED],
        blue_danger=target_geometries[TargetClass.BLUE_DANGER],
        coarse_center_step_mm=_finite_float(
            _required(
                target_fitting_raw,
                "coarse_center_step_mm",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "coarse_center_step_mm",
            minimum=0.001,
        ),
        coarse_yaw_step_deg=_finite_float(
            _required(
                target_fitting_raw,
                "coarse_yaw_step_deg",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "coarse_yaw_step_deg",
            minimum=0.001,
        ),
        refine_center_step_mm=_finite_float(
            _required(
                target_fitting_raw,
                "refine_center_step_mm",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "refine_center_step_mm",
            minimum=0.001,
        ),
        refine_center_radius_mm=_finite_float(
            _required(
                target_fitting_raw,
                "refine_center_radius_mm",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "refine_center_radius_mm",
            minimum=0.001,
        ),
        refine_top_candidates=_positive_int(
            _required(
                target_fitting_raw,
                "refine_top_candidates",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "refine_top_candidates",
        ),
        refine_yaw_step_deg=_finite_float(
            _required(
                target_fitting_raw,
                "refine_yaw_step_deg",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "refine_yaw_step_deg",
            minimum=0.001,
        ),
        refine_yaw_radius_deg=_finite_float(
            _required(
                target_fitting_raw,
                "refine_yaw_radius_deg",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "refine_yaw_radius_deg",
            minimum=0.001,
        ),
        search_radius_margin_mm=_finite_float(
            _required(
                target_fitting_raw,
                "search_radius_margin_mm",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "search_radius_margin_mm",
            minimum=0.001,
        ),
        silhouette_weight=_threshold(
            _required(
                target_fitting_raw,
                "silhouette_weight",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "silhouette_weight",
        ),
        contour_weight=_threshold(
            _required(
                target_fitting_raw,
                "contour_weight",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting.contour_weight",
        ),
        contact_weight=_threshold(
            _required(
                target_fitting_raw,
                "contact_weight",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting.contact_weight",
        ),
        contour_distance_scale_px=_finite_float(
            _required(
                target_fitting_raw,
                "contour_distance_scale_px",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "contour_distance_scale_px",
            minimum=0.001,
        ),
        contact_distance_scale_px=_finite_float(
            _required(
                target_fitting_raw,
                "contact_distance_scale_px",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "contact_distance_scale_px",
            minimum=0.001,
        ),
        max_contact_residual_px=_finite_float(
            _required(
                target_fitting_raw,
                "max_contact_residual_px",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "max_contact_residual_px",
            minimum=0.001,
        ),
        ambiguity_score_delta=_threshold(
            _required(
                target_fitting_raw,
                "ambiguity_score_delta",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "ambiguity_score_delta",
        ),
        max_center_uncertainty_mm=_finite_float(
            _required(
                target_fitting_raw,
                "max_center_uncertainty_mm",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "max_center_uncertainty_mm",
            minimum=0.001,
        ),
        min_fit_score=_threshold(
            _required(
                target_fitting_raw,
                "min_fit_score",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting.min_fit_score",
        ),
        min_silhouette_iou=_threshold(
            _required(
                target_fitting_raw,
                "min_silhouette_iou",
                "perception.target_ground_geometry.fitting",
            ),
            "perception.target_ground_geometry.fitting."
            "min_silhouette_iou",
        ),
    )
    gripper_color_raw = _mapping(perception_raw["gripper_color"], "perception.gripper_color")
    _reject_unknown(gripper_color_raw, {
        "enabled", "polygon_normalized", "min_component_fraction",
        "black_min_thickness_fraction", "shadow_min_value",
        "orange_bbox_min_color_fraction",
        "orange_distinct_max_bbox_iou", "orange_distinct_min_k0_distance_px",
    }, "perception.gripper_color")
    polygon = gripper_color_raw["polygon_normalized"]
    if not isinstance(polygon, list) or any(not isinstance(point, list) for point in polygon):
        raise ValueError(f"perception.gripper_color.polygon_normalized must be a list of points, got {polygon!r}")
    gripper_color = GripperColorConfig(
        enabled=gripper_color_raw["enabled"],
        shadow_min_value=gripper_color_raw["shadow_min_value"],
        polygon_normalized=tuple(tuple(point) for point in polygon),
        min_component_fraction=_finite_float(gripper_color_raw["min_component_fraction"], "perception.gripper_color.min_component_fraction", minimum=0.000001),
        black_min_thickness_fraction=_finite_float(gripper_color_raw["black_min_thickness_fraction"], "perception.gripper_color.black_min_thickness_fraction", minimum=0.000001),
        orange_bbox_min_color_fraction=_finite_float(
            gripper_color_raw["orange_bbox_min_color_fraction"],
            "perception.gripper_color.orange_bbox_min_color_fraction",
            minimum=0.000001,
        ),
        orange_distinct_max_bbox_iou=_finite_float(
            gripper_color_raw["orange_distinct_max_bbox_iou"],
            "perception.gripper_color.orange_distinct_max_bbox_iou",
            minimum=0.0,
        ),
        orange_distinct_min_k0_distance_px=_finite_float(
            gripper_color_raw["orange_distinct_min_k0_distance_px"],
            "perception.gripper_color.orange_distinct_min_k0_distance_px",
            minimum=0.000001,
        ),
    )
    perception = PerceptionConfig(
        detection_threshold=detection_threshold,
        k0_threshold=k0_threshold,
        color_classifier=color_classifier,
        target_ground_geometry=target_ground_geometry,
        center_cross_refinement=center_cross_refinement,
        safe_zone_color=safe_zone_color,
        gripper_color=gripper_color,
    )

    localization_raw = _merge_defaults(
        _localization_defaults(),
        _mapping(root.get("localization", {}), "localization"),
    )
    _reject_unknown(
        localization_raw,
        {
            "enabled",
            "ray_min_forward_distance_mm",
            "ray_max_forward_distance_mm",
            "ray_max_lateral_distance_mm",
            "ray_angle_tolerance_deg",
            "min_anchor_confidence",
            "max_prior_heading_innovation_deg",
            "position_uncertainty_floor_mm",
            "heading_uncertainty_floor_deg",
            "static_landmarks",
            "safe_zone_corners",
            "fusion",
        },
        "localization",
    )
    localization_enabled = _required(localization_raw, "enabled", "localization")
    if not isinstance(localization_enabled, bool):
        raise ValueError("localization.enabled must be a boolean.")
    min_anchor_confidence = _threshold(
        _required(
            localization_raw,
            "min_anchor_confidence",
            "localization",
        ),
        "localization.min_anchor_confidence",
    )
    if min_anchor_confidence > 1.0:
        raise ValueError("localization.min_anchor_confidence must be <= 1.0.")
    center_cross_localization = CenterCrossLocalizerConfig(
        enabled=localization_enabled,
        ray_min_forward_distance_mm=_finite_float(
            _required(
                localization_raw,
                "ray_min_forward_distance_mm",
                "localization",
            ),
            "localization.ray_min_forward_distance_mm",
            minimum=0.0,
        ),
        ray_max_forward_distance_mm=_finite_float(
            _required(
                localization_raw,
                "ray_max_forward_distance_mm",
                "localization",
            ),
            "localization.ray_max_forward_distance_mm",
            minimum=0.001,
        ),
        ray_max_lateral_distance_mm=_finite_float(
            _required(
                localization_raw,
                "ray_max_lateral_distance_mm",
                "localization",
            ),
            "localization.ray_max_lateral_distance_mm",
            minimum=0.0,
        ),
        ray_angle_tolerance_deg=_finite_float(
            _required(
                localization_raw,
                "ray_angle_tolerance_deg",
                "localization",
            ),
            "localization.ray_angle_tolerance_deg",
            minimum=0.001,
        ),
        min_anchor_confidence=min_anchor_confidence,
        max_prior_heading_innovation_deg=_finite_float(
            _required(
                localization_raw,
                "max_prior_heading_innovation_deg",
                "localization",
            ),
            "localization.max_prior_heading_innovation_deg",
            minimum=0.001,
        ),
        position_uncertainty_floor_mm=_finite_float(
            _required(
                localization_raw,
                "position_uncertainty_floor_mm",
                "localization",
            ),
            "localization.position_uncertainty_floor_mm",
            minimum=0.001,
        ),
        heading_uncertainty_floor_deg=_finite_float(
            _required(
                localization_raw,
                "heading_uncertainty_floor_deg",
                "localization",
            ),
            "localization.heading_uncertainty_floor_deg",
            minimum=0.001,
        ),
    )
    static_landmarks_raw = _mapping(
        localization_raw["static_landmarks"],
        "localization.static_landmarks",
    )
    static_landmark_names = {
        "max_track_age_ms",
        "max_prior_position_uncertainty_mm",
        "max_prior_heading_uncertainty_deg",
        "cross_base_radius_mm",
        "safe_zone_base_margin_mm",
        "confirmation_hits",
        "max_confirmation_age_ms",
        "confirmation_ground_tolerance_mm",
        "confirmation_axis_tolerance_deg",
    }
    _reject_unknown(
        static_landmarks_raw,
        static_landmark_names,
        "localization.static_landmarks",
    )
    static_landmarks = StaticLandmarkTrackingConfig(
        max_track_age_ms=_finite_float(
            static_landmarks_raw["max_track_age_ms"],
            "localization.static_landmarks.max_track_age_ms",
            minimum=0.001,
        ),
        max_prior_position_uncertainty_mm=_finite_float(
            static_landmarks_raw["max_prior_position_uncertainty_mm"],
            "localization.static_landmarks.max_prior_position_uncertainty_mm",
            minimum=0.001,
        ),
        max_prior_heading_uncertainty_deg=_finite_float(
            static_landmarks_raw["max_prior_heading_uncertainty_deg"],
            "localization.static_landmarks.max_prior_heading_uncertainty_deg",
            minimum=0.001,
        ),
        cross_base_radius_mm=_finite_float(
            static_landmarks_raw["cross_base_radius_mm"],
            "localization.static_landmarks.cross_base_radius_mm",
            minimum=0.001,
        ),
        safe_zone_base_margin_mm=_finite_float(
            static_landmarks_raw["safe_zone_base_margin_mm"],
            "localization.static_landmarks.safe_zone_base_margin_mm",
            minimum=0.001,
        ),
        confirmation_hits=_positive_int(
            static_landmarks_raw["confirmation_hits"],
            "localization.static_landmarks.confirmation_hits",
        ),
        max_confirmation_age_ms=_finite_float(
            static_landmarks_raw["max_confirmation_age_ms"],
            "localization.static_landmarks.max_confirmation_age_ms",
            minimum=0.001,
        ),
        confirmation_ground_tolerance_mm=_finite_float(
            static_landmarks_raw["confirmation_ground_tolerance_mm"],
            "localization.static_landmarks.confirmation_ground_tolerance_mm",
            minimum=0.001,
        ),
        confirmation_axis_tolerance_deg=_finite_float(
            static_landmarks_raw["confirmation_axis_tolerance_deg"],
            "localization.static_landmarks.confirmation_axis_tolerance_deg",
            minimum=0.001,
        ),
    )
    safe_zone_corners_raw = _mapping(
        localization_raw["safe_zone_corners"],
        "localization.safe_zone_corners",
    )
    safe_zone_corner_names = {
        "max_observation_age_ms",
        "min_baseline_mm",
        "max_k0_corner_distance_error_mm",
        "max_fit_residual_mm",
        "position_uncertainty_floor_mm",
        "heading_uncertainty_floor_deg",
    }
    _reject_unknown(
        safe_zone_corners_raw,
        safe_zone_corner_names,
        "localization.safe_zone_corners",
    )
    safe_zone_corners = SafeZoneCornerLocalizerConfig(
        max_observation_age_ms=_finite_float(
            safe_zone_corners_raw["max_observation_age_ms"],
            "localization.safe_zone_corners.max_observation_age_ms",
            minimum=0.001,
        ),
        min_baseline_mm=_finite_float(
            safe_zone_corners_raw["min_baseline_mm"],
            "localization.safe_zone_corners.min_baseline_mm",
            minimum=0.001,
        ),
        max_k0_corner_distance_error_mm=_finite_float(
            safe_zone_corners_raw["max_k0_corner_distance_error_mm"],
            "localization.safe_zone_corners.max_k0_corner_distance_error_mm",
            minimum=0.001,
        ),
        max_fit_residual_mm=_finite_float(
            safe_zone_corners_raw["max_fit_residual_mm"],
            "localization.safe_zone_corners.max_fit_residual_mm",
            minimum=0.001,
        ),
        position_uncertainty_floor_mm=_finite_float(
            safe_zone_corners_raw["position_uncertainty_floor_mm"],
            "localization.safe_zone_corners.position_uncertainty_floor_mm",
            minimum=0.001,
        ),
        heading_uncertainty_floor_deg=_finite_float(
            safe_zone_corners_raw["heading_uncertainty_floor_deg"],
            "localization.safe_zone_corners.heading_uncertainty_floor_deg",
            minimum=0.001,
        ),
    )
    fusion_raw = _mapping(localization_raw["fusion"], "localization.fusion")
    _reject_unknown(
        fusion_raw,
        {
            "enabled",
            "initial_pose",
            "imu_calibration",
            "encoder_distance_noise_fraction",
            "encoder_heading_noise_std_deg",
            "gyro_noise_std_rad_s",
            "gyro_bias_random_walk_std_rad_s_per_sqrt_s",
            "stationary_gyro_noise_std_rad_s",
            "stationary_encoder_delta_count",
            "allow_wheel_only",
            "wheel_only_covariance_scale",
            "dropped_sample_covariance_scale",
            "max_interpolated_overrun_samples",
            "max_sample_interval_ms",
            "max_telemetry_age_ms",
            "max_encoder_speed_mm_s",
            "max_visual_alignment_error_ms",
            "visual_innovation_gate",
            "history_duration_ms",
            "max_tilt_deg",
            "impact_accel_threshold_mm_s2",
        },
        "localization.fusion",
    )
    fusion_enabled = _required(fusion_raw, "enabled", "localization.fusion")
    allow_wheel_only = _required(
        fusion_raw, "allow_wheel_only", "localization.fusion"
    )
    if not isinstance(fusion_enabled, bool) or not isinstance(allow_wheel_only, bool):
        raise ValueError(
            "localization.fusion enabled flags must be booleans."
        )
    initial_pose_raw = _mapping(
        _required(fusion_raw, "initial_pose", "localization.fusion"),
        "localization.fusion.initial_pose",
    )
    _reject_unknown(
        initial_pose_raw,
        {
            "x_mm",
            "y_mm",
            "heading_deg",
            "position_uncertainty_mm",
            "heading_uncertainty_deg",
            "confidence",
        },
        "localization.fusion.initial_pose",
    )
    initial_confidence = _threshold(
        _required(
            initial_pose_raw,
            "confidence",
            "localization.fusion.initial_pose",
        ),
        "localization.fusion.initial_pose.confidence",
    )
    if initial_confidence > 1.0:
        raise ValueError(
            "localization.fusion.initial_pose.confidence must be <= 1."
        )
    imu_raw = _mapping(
        _required(fusion_raw, "imu_calibration", "localization.fusion"),
        "localization.fusion.imu_calibration",
    )
    _reject_unknown(
        imu_raw,
        {
            "reference_temperature_c",
            "gyro_bias_rad_s",
            "gyro_bias_temperature_coefficient_rad_s_per_c",
            "gyro_cross_axis_scale",
            "accel_bias_mm_s2",
            "accel_bias_temperature_coefficient_mm_s2_per_c",
            "accel_cross_axis_scale",
            "sensor_to_robot_rotation",
        },
        "localization.fusion.imu_calibration",
    )
    imu_location = "localization.fusion.imu_calibration"
    try:
        imu_frame_calibration = ImuFrameCalibration(
            reference_temperature_c=_finite_float(
                _required(imu_raw, "reference_temperature_c", imu_location),
                f"{imu_location}.reference_temperature_c",
                minimum=-float("inf"),
            ),
            gyro_bias_rad_s=_float_vector3(
                _required(imu_raw, "gyro_bias_rad_s", imu_location),
                f"{imu_location}.gyro_bias_rad_s",
            ),
            gyro_bias_temperature_coefficient_rad_s_per_c=_float_vector3(
                _required(
                    imu_raw,
                    "gyro_bias_temperature_coefficient_rad_s_per_c",
                    imu_location,
                ),
                f"{imu_location}.gyro_bias_temperature_coefficient_rad_s_per_c",
            ),
            gyro_cross_axis_scale=_float_matrix3(
                _required(imu_raw, "gyro_cross_axis_scale", imu_location),
                f"{imu_location}.gyro_cross_axis_scale",
            ),
            accel_bias_mm_s2=_float_vector3(
                _required(imu_raw, "accel_bias_mm_s2", imu_location),
                f"{imu_location}.accel_bias_mm_s2",
            ),
            accel_bias_temperature_coefficient_mm_s2_per_c=_float_vector3(
                _required(
                    imu_raw,
                    "accel_bias_temperature_coefficient_mm_s2_per_c",
                    imu_location,
                ),
                f"{imu_location}.accel_bias_temperature_coefficient_mm_s2_per_c",
            ),
            accel_cross_axis_scale=_float_matrix3(
                _required(imu_raw, "accel_cross_axis_scale", imu_location),
                f"{imu_location}.accel_cross_axis_scale",
            ),
            sensor_to_robot_rotation=_float_matrix3(
                _required(imu_raw, "sensor_to_robot_rotation", imu_location),
                f"{imu_location}.sensor_to_robot_rotation",
            ),
        )
    except ValueError as exc:
        raise ValueError(
            f"{imu_location} is invalid: {exc}"
        ) from exc
    fusion = FusionConfig(
        enabled=fusion_enabled,
        initial_pose=FieldPose2D(
            FieldPoint(
                _finite_float(
                    _required(initial_pose_raw, "x_mm", "localization.fusion.initial_pose"),
                    "localization.fusion.initial_pose.x_mm",
                    minimum=-float("inf"),
                ),
                _finite_float(
                    _required(initial_pose_raw, "y_mm", "localization.fusion.initial_pose"),
                    "localization.fusion.initial_pose.y_mm",
                    minimum=-float("inf"),
                ),
            ),
            math.radians(
                _finite_float(
                    _required(
                        initial_pose_raw,
                        "heading_deg",
                        "localization.fusion.initial_pose",
                    ),
                    "localization.fusion.initial_pose.heading_deg",
                    minimum=-float("inf"),
                )
            ),
        ),
        initial_position_uncertainty_mm=_finite_float(
            _required(
                initial_pose_raw,
                "position_uncertainty_mm",
                "localization.fusion.initial_pose",
            ),
            "localization.fusion.initial_pose.position_uncertainty_mm",
            minimum=0.001,
        ),
        initial_heading_uncertainty_rad=math.radians(
            _finite_float(
                _required(
                    initial_pose_raw,
                    "heading_uncertainty_deg",
                    "localization.fusion.initial_pose",
                ),
                "localization.fusion.initial_pose.heading_uncertainty_deg",
                minimum=0.001,
            )
        ),
        initial_confidence=initial_confidence,
        imu_frame_calibration=imu_frame_calibration,
        encoder_distance_noise_fraction=_finite_float(
            fusion_raw["encoder_distance_noise_fraction"],
            "localization.fusion.encoder_distance_noise_fraction",
            minimum=0.0,
        ),
        encoder_heading_noise_std_rad=math.radians(
            _finite_float(
                fusion_raw["encoder_heading_noise_std_deg"],
                "localization.fusion.encoder_heading_noise_std_deg",
                minimum=0.001,
            )
        ),
        gyro_noise_std_rad_s=_finite_float(
            fusion_raw["gyro_noise_std_rad_s"],
            "localization.fusion.gyro_noise_std_rad_s",
            minimum=0.001,
        ),
        gyro_bias_random_walk_std_rad_s_per_sqrt_s=_finite_float(
            fusion_raw["gyro_bias_random_walk_std_rad_s_per_sqrt_s"],
            "localization.fusion.gyro_bias_random_walk_std_rad_s_per_sqrt_s",
            minimum=0.000001,
        ),
        stationary_gyro_noise_std_rad_s=_finite_float(
            fusion_raw["stationary_gyro_noise_std_rad_s"],
            "localization.fusion.stationary_gyro_noise_std_rad_s",
            minimum=0.000001,
        ),
        stationary_encoder_delta_count=_nonnegative_int(
            fusion_raw["stationary_encoder_delta_count"],
            "localization.fusion.stationary_encoder_delta_count",
        ),
        allow_wheel_only=allow_wheel_only,
        wheel_only_covariance_scale=_finite_float(
            fusion_raw["wheel_only_covariance_scale"],
            "localization.fusion.wheel_only_covariance_scale",
            minimum=1.0,
        ),
        dropped_sample_covariance_scale=_finite_float(
            fusion_raw["dropped_sample_covariance_scale"],
            "localization.fusion.dropped_sample_covariance_scale",
            minimum=1.0,
        ),
        max_interpolated_overrun_samples=_nonnegative_int(
            fusion_raw["max_interpolated_overrun_samples"],
            "localization.fusion.max_interpolated_overrun_samples",
        ),
        max_sample_interval_ms=_finite_float(
            fusion_raw["max_sample_interval_ms"],
            "localization.fusion.max_sample_interval_ms",
            minimum=0.001,
        ),
        max_telemetry_age_ms=_finite_float(
            fusion_raw["max_telemetry_age_ms"],
            "localization.fusion.max_telemetry_age_ms",
            minimum=0.001,
        ),
        max_encoder_speed_mm_s=_finite_float(
            fusion_raw["max_encoder_speed_mm_s"],
            "localization.fusion.max_encoder_speed_mm_s",
            minimum=0.001,
        ),
        max_visual_alignment_error_ms=_finite_float(
            fusion_raw["max_visual_alignment_error_ms"],
            "localization.fusion.max_visual_alignment_error_ms",
            minimum=0.001,
        ),
        visual_innovation_gate=_finite_float(
            fusion_raw["visual_innovation_gate"],
            "localization.fusion.visual_innovation_gate",
            minimum=0.001,
        ),
        history_duration_ms=_finite_float(
            fusion_raw["history_duration_ms"],
            "localization.fusion.history_duration_ms",
            minimum=0.001,
        ),
        max_tilt_deg=_finite_float(
            fusion_raw["max_tilt_deg"],
            "localization.fusion.max_tilt_deg",
            minimum=0.001,
        ),
        impact_accel_threshold_mm_s2=_finite_float(
            fusion_raw["impact_accel_threshold_mm_s2"],
            "localization.fusion.impact_accel_threshold_mm_s2",
            minimum=0.001,
        ),
    )
    if fusion.enabled and (
        not motion.enabled
        or motion.wheel_track_m is None
        or not motion.odometry.enabled
        or not uart.enabled
    ):
        raise ValueError(
            "Enabled localization.fusion requires motion with wheel_track_m, "
            "UART and motion.odometry; visual field features and ground mapping "
            "are only required by the visual localization path."
        )
    localization = LocalizationRuntimeConfig(
        center_cross_localization,
        static_landmarks,
        safe_zone_corners,
        fusion,
    )

    hailo_raw = _mapping(root.get("hailo", {}), "hailo")
    _reject_unknown(
        hailo_raw,
        {
            "enabled",
            "hef_path",
            "postprocess_onnx_path",
            "output_mapping_path",
            "raw_classes",
            "backend_score_threshold",
            "max_detections",
        },
        "hailo",
    )
    hailo_enabled = hailo_raw.get("enabled", False)
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
    raw_classes_value = hailo_raw.get("raw_classes", [])
    if not isinstance(raw_classes_value, list):
        raise ValueError("hailo.raw_classes must be a list.")
    raw_classes = tuple(
        _string(value, f"hailo.raw_classes[{index}]")
        for index, value in enumerate(raw_classes_value)
    )
    if len(set(raw_classes)) != len(raw_classes):
        raise ValueError("hailo.raw_classes must not contain duplicates.")
    expected_pose_classes = tuple(item.value for item in POSE_MODEL_CLASSES)
    if raw_classes and raw_classes != expected_pose_classes:
        raise ValueError(
            "hailo.raw_classes must exactly match YOLO Pose v3 order "
            f"{list(expected_pose_classes)!r}, got {list(raw_classes)!r}."
        )

    backend_score_threshold = _threshold(
        hailo_raw.get("backend_score_threshold", 0.01),
        "hailo.backend_score_threshold",
    )
    if backend_score_threshold > 1.0:
        raise ValueError("hailo.backend_score_threshold must be <= 1.0.")
    if backend_score_threshold > detection_threshold:
        raise ValueError(
            "hailo.backend_score_threshold must be <= "
            "perception.detection_threshold, got "
            f"{backend_score_threshold} > "
            f"{detection_threshold}."
        )
    max_detections = _positive_int(
        hailo_raw.get("max_detections", 100),
        "hailo.max_detections",
    )

    required_assets = (
        hef_path,
        postprocess_onnx_path,
        output_mapping_path,
    )
    if hailo_enabled and (
        any(value is None for value in required_assets) or not raw_classes
    ):
        raise ValueError(
            "Enabled hailo requires all asset paths and raw_classes."
        )
    hailo = HailoConfig(
        enabled=hailo_enabled,
        hef_path=hef_path,
        postprocess_onnx_path=postprocess_onnx_path,
        output_mapping_path=output_mapping_path,
        raw_classes=raw_classes,
        backend_score_threshold=backend_score_threshold,
        max_detections=max_detections,
    )
    if cluster_breakup.enabled and (
        not hailo.enabled or not geometry.ground_mapping_enabled
    ):
        raise ValueError(
            "Enabled motion.cluster_breakup requires Hailo and ground mapping."
        )
    if match.enabled:
        if not motion.enabled or not motion.odometry.enabled:
            raise ValueError(
                "Enabled match requires motion and "
                "motion.odometry."
            )
        if not motion.gripper.enabled:
            raise ValueError(
                "Enabled match requires motion.gripper.enabled=true."
            )
        if not hailo.enabled or not geometry.ground_mapping_enabled:
            raise ValueError(
                "Enabled match requires Hailo and ground mapping."
            )
        initial = localization.fusion.initial_pose
        if not (
            math.isclose(initial.position.x, 1350.0, abs_tol=1e-6)
            and math.isclose(initial.position.y, 1350.0, abs_tol=1e-6)
            and math.isclose(initial.heading_rad, -math.pi / 2.0, abs_tol=1e-6)
        ):
            raise ValueError(
                "Enabled match requires initial pose "
                "[1350 mm, 1350 mm, -90 deg]."
            )
        if remote.enabled and (
            remote.role is not RemoteRole.SERVER
            or remote.access_mode is not RemoteAccessMode.OBSERVE_ONLY
        ):
            raise ValueError(
                "match remote access must be a server in "
                "observe_only mode."
            )

    near_raw = _mapping(root.get("near_field_grasp", {}), "near_field_grasp")
    _reject_unknown(near_raw, {item.name for item in fields(NearFieldGraspConfig)}, "near_field_grasp")
    near_field_grasp = NearFieldGraspConfig(**near_raw)

    return AppConfig(
        camera,
        geometry,
        recording,
        processing,
        uart,
        remote,
        motion,
        match,
        tracking,
        world,
        mission,
        perception,
        localization,
        hailo,
        green_grab,
        near_field_grasp,
        match_cc,
    )
