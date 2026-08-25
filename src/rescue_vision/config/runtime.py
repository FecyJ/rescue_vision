"""运行配置及几何对象装配。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rescue_vision.communication.remote import RemoteAccessMode, RemoteRole
from rescue_vision.communication.remote_messages import RemoteTopic
from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossLocalizerConfig,
    FieldPose2D,
    FusionConfig,
    OdometryCalibration,
)
from rescue_vision.mission import MissionConfig
from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    HsvColorClassifierConfig,
    HsvRange,
    TargetClass,
)
from rescue_vision.perception.field_feature_types import FieldFeatureConfig
from rescue_vision.perception.field_boundary import FieldBoundaryConfig
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
    """返回与示例配置一致且默认关闭几何/场地检测的感知参数。"""

    return {
        "detection_threshold": 0.25,
        "k0_threshold": 0.50,
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
        "field_features": {
            "enabled": False,
            "colors": {
                "safe_red": [
                    {"lower": [0, 80, 80], "upper": [12, 255, 255]},
                    {"lower": [170, 80, 80], "upper": [179, 255, 255]},
                ],
                "safe_blue": [
                    {"lower": [90, 60, 70], "upper": [110, 255, 255]},
                ],
                "start_magenta": [
                    {"lower": [140, 80, 80], "upper": [165, 255, 255]},
                ],
                "entrance_purple": [
                    {"lower": [130, 60, 50], "upper": [160, 255, 255]},
                ],
                "dark_marking": [
                    {"lower": [0, 0, 0], "upper": [179, 255, 80]},
                ],
            },
            "morphology": {
                "kernel_size": 5,
                "open_iterations": 1,
                "close_iterations": 2,
            },
            "region_filter": {
                "min_area_fraction": 0.002,
                "min_rectangularity": 0.55,
                "dimension_tolerance_fraction": 0.40,
            },
            "safe_zone": {
                "entrance_color_fraction": 0.10,
                "divider_dark_fraction": 0.10,
            },
            "center_cross": {
                "min_axis_span_fraction": 0.20,
                "max_gap_fraction": 0.06,
                "min_gap_count": 2,
                "perpendicular_tolerance_deg": 15.0,
                "local_window_fraction": 0.015,
                "local_contrast_threshold": 10,
                "max_saturation": 80,
                "min_floor_value": 120,
                "min_line_support_fraction": 0.10,
                "min_white_surround_fraction": 0.60,
                "min_axis_balance_fraction": 0.08,
                "min_intersection_margin_fraction": 0.02,
            },
            "boundary": {
                "canny_low_threshold": 50,
                "canny_high_threshold": 150,
                "min_line_length_fraction": 0.20,
                "corner_tolerance_deg": 20.0,
                "max_features": 8,
                "min_vertical_support_count": 3,
                "side_contrast_threshold": 15,
            },
        },
        "field_boundary": {
            "enabled": False,
            "hard_mask_enabled": False,
            "min_candidate_confidence": 0.25,
            "min_confirmations": 2,
            "max_missed_frames": 1,
            "line_angle_tolerance_deg": 12.0,
            "line_distance_tolerance_mm": 120.0,
            "ransac_inlier_distance_mm": 60.0,
            "rectangle_tolerance_fraction": 0.20,
            "boundary_band_mm": 120.0,
            "segment_extension_mm": 150.0,
            "min_filter_confidence": 0.55,
            "max_mask_age_ms": 150.0,
            "mask_blur_radius_px": 5,
            "neutral_fill_bgr": [114, 114, 114],
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
            "encoder_distance_noise_fraction": 0.02,
            "encoder_heading_noise_std_deg": 1.0,
            "gyro_noise_std_rad_s": 0.03,
            "gyro_bias_random_walk_std_rad_s_per_sqrt_s": 0.001,
            "stationary_gyro_noise_std_rad_s": 0.01,
            "stationary_encoder_delta_count": 0,
            "allow_wheel_only": True,
            "wheel_only_covariance_scale": 4.0,
            "dropped_sample_covariance_scale": 3.0,
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
        )


@dataclass(frozen=True, slots=True)
class OdometryRuntimeConfig:
    enabled: bool
    encoder_counts_per_revolution: int | None
    left_wheel_radius_mm: float | None
    right_wheel_radius_mm: float | None
    gyro_z_bias_rad_s: float | None

    def build_calibration(self) -> OdometryCalibration | None:
        if not self.enabled:
            return None
        assert self.encoder_counts_per_revolution is not None
        assert self.left_wheel_radius_mm is not None
        assert self.right_wheel_radius_mm is not None
        assert self.gyro_z_bias_rad_s is not None
        return OdometryCalibration(
            self.encoder_counts_per_revolution,
            self.left_wheel_radius_mm,
            self.right_wheel_radius_mm,
            self.gyro_z_bias_rad_s,
        )


@dataclass(frozen=True, slots=True)
class MotionRuntimeConfig:
    enabled: bool
    wheel_track_m: float | None
    max_linear_velocity_m_s: float
    max_angular_velocity_rad_s: float
    max_wheel_velocity_m_s: float
    max_wheel_acceleration_m_s2: float
    max_remote_command_valid_for_ms: int
    gripper: GripperRuntimeConfig
    odometry: OdometryRuntimeConfig

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
                max_wheel_acceleration_m_s2=(
                    self.max_wheel_acceleration_m_s2
                ),
                max_remote_command_valid_for_ms=(
                    self.max_remote_command_valid_for_ms
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
    field_features: FieldFeatureConfig
    field_boundary: FieldBoundaryConfig

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

    def build_field_feature_detector(
        self,
        *,
        static_map: StaticFieldMap,
        max_observation_age_ms: float,
        ground_projector: GroundProjector | None = None,
    ):
        """按配置创建传统视觉场地特征检测器；禁用时返回 ``None``。"""

        if not self.field_features.enabled:
            return None
        from rescue_vision.perception.field_feature_detector import (
            FieldFeatureDetector,
        )

        return FieldFeatureDetector(
            self.field_features,
            static_map=static_map,
            max_observation_age_ms=max_observation_age_ms,
            ground_projector=ground_projector,
        )

    def build_field_boundary_estimator(
        self,
        *,
        static_map: StaticFieldMap,
        ground_projector: GroundProjector | None,
    ):
        """按配置创建局部场界时序估计器；禁用时返回 ``None``。"""

        if not self.field_boundary.enabled:
            return None
        if not self.field_features.enabled:
            raise RuntimeError(
                "Enabled field_boundary requires perception.field_features."
            )
        if ground_projector is None or ground_projector.bev_config is None:
            raise RuntimeError("Enabled field_boundary requires BEV ground mapping.")
        from rescue_vision.perception import FieldBoundaryEstimator

        return FieldBoundaryEstimator(
            self.field_boundary,
            static_map=static_map,
            ground_projector=ground_projector,
        )


@dataclass(frozen=True, slots=True)
class HailoConfig:
    enabled: bool
    hef_path: Path | None
    postprocess_onnx_path: Path | None
    output_mapping_path: Path | None
    raw_classes: tuple[str, ...]
    class_mapping: tuple[TargetClass, ...]
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

    def model_class_mapping(self) -> dict[int, TargetClass]:
        return dict(enumerate(self.class_mapping))


@dataclass(frozen=True, slots=True)
class RuntimeGeometry:
    camera_model: CameraModel
    ground_projector: GroundProjector | None


@dataclass(frozen=True, slots=True)
class LocalizationRuntimeConfig:
    center_cross: CenterCrossLocalizerConfig
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
    tracking: TrackingConfig
    world: WorldRuntimeConfig
    mission: MissionConfig
    perception: PerceptionConfig
    localization: LocalizationRuntimeConfig
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
            class_mapping=self.hailo.model_class_mapping(),
            detection_threshold=self.perception.detection_threshold,
            k0_threshold=self.perception.k0_threshold,
            color_classifier=self.perception.color_classifier,
            max_observation_age_ms=self.processing.max_observation_age_ms,
            ground_projector=ground_projector,
        )

    def build_center_cross_localizer(
        self,
        *,
        ground_projector: GroundProjector | None,
    ) -> CenterCrossLocalizer | None:
        """Build center-cross localization when all geometric inputs exist."""

        if (
            not self.localization.center_cross.enabled
            or not self.perception.field_features.enabled
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
            "tracking",
            "world",
            "mission",
            "perception",
            "localization",
            "hailo",
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
            "max_wheel_acceleration_m_s2",
            "max_remote_command_valid_for_ms",
            "gripper",
            "odometry",
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
            "Enabled motion.gripper requires four endpoint angles and "
            "full_travel_time_s and angle_sum_deg."
        )
    gripper = GripperRuntimeConfig(
        enabled=gripper_enabled,
        open_left_angle_deg=gripper_values["open_left_angle_deg"],
        open_right_angle_deg=gripper_values["open_right_angle_deg"],
        closed_left_angle_deg=gripper_values["closed_left_angle_deg"],
        closed_right_angle_deg=gripper_values["closed_right_angle_deg"],
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
            "gyro_z_bias_rad_s",
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
        ("gyro_z_bias_rad_s", -float("inf")),
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
        gyro_z_bias_rad_s=odometry_floats["gyro_z_bias_rad_s"],
    )
    if odometry.enabled:
        odometry.build_calibration()
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
        max_wheel_velocity_m_s=_finite_float(
            motion_raw.get("max_wheel_velocity_m_s", 0.30),
            "motion.max_wheel_velocity_m_s",
            minimum=0.001,
        ),
        max_wheel_acceleration_m_s2=_finite_float(
            motion_raw.get("max_wheel_acceleration_m_s2", 0.50),
            "motion.max_wheel_acceleration_m_s2",
            minimum=0.001,
        ),
        max_remote_command_valid_for_ms=max_remote_validity,
        gripper=gripper,
        odometry=odometry,
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
            "unknown_suspect_threshold",
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
        unknown_suspect_threshold=_threshold(
            world_raw.get("unknown_suspect_threshold", 0.50),
            "world.unknown_suspect_threshold",
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
        {"center_cross", "regions"},
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
        StaticFieldMap(center_cross_map, tuple(physical_regions)),
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
            "danger_avoid_distance_mm",
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
        danger_avoid_distance_mm=_finite_float(
            mission_raw.get("danger_avoid_distance_mm", 500.0),
            "mission.danger_avoid_distance_mm",
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
            "color_classifier",
            "target_ground_geometry",
            "field_features",
            "field_boundary",
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
    field_raw = _mapping(
        _required(perception_raw, "field_features", "perception"),
        "perception.field_features",
    )
    _reject_unknown(
        field_raw,
        {
            "enabled",
            "colors",
            "morphology",
            "region_filter",
            "safe_zone",
            "center_cross",
            "boundary",
        },
        "perception.field_features",
    )
    field_enabled = _required(
        field_raw,
        "enabled",
        "perception.field_features",
    )
    if not isinstance(field_enabled, bool):
        raise ValueError("perception.field_features.enabled must be a boolean.")

    field_colors_raw = _mapping(
        _required(field_raw, "colors", "perception.field_features"),
        "perception.field_features.colors",
    )
    field_color_names = {
        "safe_red",
        "safe_blue",
        "start_magenta",
        "entrance_purple",
        "dark_marking",
    }
    if set(field_colors_raw) != field_color_names:
        raise ValueError(
            "perception.field_features.colors keys must exactly be "
            f"{sorted(field_color_names)!r}."
        )
    field_ranges = {
        name: _hsv_ranges(
            field_colors_raw[name],
            f"perception.field_features.colors.{name}",
        )
        for name in sorted(field_color_names)
    }

    morphology_raw = _mapping(
        _required(field_raw, "morphology", "perception.field_features"),
        "perception.field_features.morphology",
    )
    _reject_unknown(
        morphology_raw,
        {"kernel_size", "open_iterations", "close_iterations"},
        "perception.field_features.morphology",
    )
    region_filter_raw = _mapping(
        _required(field_raw, "region_filter", "perception.field_features"),
        "perception.field_features.region_filter",
    )
    _reject_unknown(
        region_filter_raw,
        {
            "min_area_fraction",
            "min_rectangularity",
            "dimension_tolerance_fraction",
        },
        "perception.field_features.region_filter",
    )
    safe_zone_raw = _mapping(
        _required(field_raw, "safe_zone", "perception.field_features"),
        "perception.field_features.safe_zone",
    )
    _reject_unknown(
        safe_zone_raw,
        {
            "entrance_color_fraction",
            "divider_dark_fraction",
        },
        "perception.field_features.safe_zone",
    )
    center_raw = _mapping(
        _required(field_raw, "center_cross", "perception.field_features"),
        "perception.field_features.center_cross",
    )
    _reject_unknown(
        center_raw,
        {
            "min_axis_span_fraction",
            "max_gap_fraction",
            "min_gap_count",
            "perpendicular_tolerance_deg",
            "local_window_fraction",
            "local_contrast_threshold",
            "max_saturation",
            "min_floor_value",
            "min_line_support_fraction",
            "min_white_surround_fraction",
            "min_axis_balance_fraction",
            "min_intersection_margin_fraction",
        },
        "perception.field_features.center_cross",
    )
    boundary_raw = _mapping(
        _required(field_raw, "boundary", "perception.field_features"),
        "perception.field_features.boundary",
    )
    _reject_unknown(
        boundary_raw,
        {
            "canny_low_threshold",
            "canny_high_threshold",
            "min_line_length_fraction",
            "corner_tolerance_deg",
            "max_features",
            "min_vertical_support_count",
            "side_contrast_threshold",
        },
        "perception.field_features.boundary",
    )
    field_features = FieldFeatureConfig(
        enabled=field_enabled,
        safe_red=field_ranges["safe_red"],
        safe_blue=field_ranges["safe_blue"],
        start_magenta=field_ranges["start_magenta"],
        entrance_purple=field_ranges["entrance_purple"],
        dark_marking=field_ranges["dark_marking"],
        morphology_kernel_size=_positive_int(
            _required(
                morphology_raw,
                "kernel_size",
                "perception.field_features.morphology",
            ),
            "perception.field_features.morphology.kernel_size",
        ),
        open_iterations=_nonnegative_int(
            _required(
                morphology_raw,
                "open_iterations",
                "perception.field_features.morphology",
            ),
            "perception.field_features.morphology.open_iterations",
        ),
        close_iterations=_nonnegative_int(
            _required(
                morphology_raw,
                "close_iterations",
                "perception.field_features.morphology",
            ),
            "perception.field_features.morphology.close_iterations",
        ),
        min_region_area_fraction=_threshold(
            _required(
                region_filter_raw,
                "min_area_fraction",
                "perception.field_features.region_filter",
            ),
            "perception.field_features.region_filter.min_area_fraction",
        ),
        min_rectangularity=_threshold(
            _required(
                region_filter_raw,
                "min_rectangularity",
                "perception.field_features.region_filter",
            ),
            "perception.field_features.region_filter.min_rectangularity",
        ),
        dimension_tolerance_fraction=_threshold(
            _required(
                region_filter_raw,
                "dimension_tolerance_fraction",
                "perception.field_features.region_filter",
            ),
            "perception.field_features.region_filter.dimension_tolerance_fraction",
        ),
        entrance_color_fraction=_threshold(
            _required(
                safe_zone_raw,
                "entrance_color_fraction",
                "perception.field_features.safe_zone",
            ),
            "perception.field_features.safe_zone.entrance_color_fraction",
        ),
        divider_dark_fraction=_threshold(
            _required(
                safe_zone_raw,
                "divider_dark_fraction",
                "perception.field_features.safe_zone",
            ),
            "perception.field_features.safe_zone.divider_dark_fraction",
        ),
        center_min_axis_span_fraction=_threshold(
            _required(
                center_raw,
                "min_axis_span_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.min_axis_span_fraction",
        ),
        center_max_gap_fraction=_threshold(
            _required(
                center_raw,
                "max_gap_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.max_gap_fraction",
        ),
        center_min_gap_count=_positive_int(
            _required(
                center_raw,
                "min_gap_count",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.min_gap_count",
        ),
        center_perpendicular_tolerance_deg=_finite_float(
            _required(
                center_raw,
                "perpendicular_tolerance_deg",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.perpendicular_tolerance_deg",
            minimum=0.001,
        ),
        center_local_window_fraction=_threshold(
            _required(
                center_raw,
                "local_window_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.local_window_fraction",
        ),
        center_local_contrast_threshold=_positive_int(
            _required(
                center_raw,
                "local_contrast_threshold",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.local_contrast_threshold",
        ),
        center_max_saturation=_nonnegative_int(
            _required(
                center_raw,
                "max_saturation",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.max_saturation",
        ),
        center_min_floor_value=_nonnegative_int(
            _required(
                center_raw,
                "min_floor_value",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.min_floor_value",
        ),
        center_min_line_support_fraction=_threshold(
            _required(
                center_raw,
                "min_line_support_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.min_line_support_fraction",
        ),
        center_min_white_surround_fraction=_threshold(
            _required(
                center_raw,
                "min_white_surround_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.min_white_surround_fraction",
        ),
        center_min_axis_balance_fraction=_threshold(
            _required(
                center_raw,
                "min_axis_balance_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross.min_axis_balance_fraction",
        ),
        center_min_intersection_margin_fraction=_threshold(
            _required(
                center_raw,
                "min_intersection_margin_fraction",
                "perception.field_features.center_cross",
            ),
            "perception.field_features.center_cross."
            "min_intersection_margin_fraction",
        ),
        boundary_canny_low_threshold=_nonnegative_int(
            _required(
                boundary_raw,
                "canny_low_threshold",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.canny_low_threshold",
        ),
        boundary_canny_high_threshold=_positive_int(
            _required(
                boundary_raw,
                "canny_high_threshold",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.canny_high_threshold",
        ),
        boundary_min_line_length_fraction=_threshold(
            _required(
                boundary_raw,
                "min_line_length_fraction",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.min_line_length_fraction",
        ),
        boundary_corner_tolerance_deg=_finite_float(
            _required(
                boundary_raw,
                "corner_tolerance_deg",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.corner_tolerance_deg",
            minimum=0.001,
        ),
        boundary_max_features=_positive_int(
            _required(
                boundary_raw,
                "max_features",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.max_features",
        ),
        boundary_min_vertical_support_count=_positive_int(
            _required(
                boundary_raw,
                "min_vertical_support_count",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.min_vertical_support_count",
        ),
        boundary_side_contrast_threshold=_positive_int(
            _required(
                boundary_raw,
                "side_contrast_threshold",
                "perception.field_features.boundary",
            ),
            "perception.field_features.boundary.side_contrast_threshold",
        ),
    )
    field_boundary_raw = _mapping(
        _required(perception_raw, "field_boundary", "perception"),
        "perception.field_boundary",
    )
    field_boundary_names = {
        "enabled",
        "hard_mask_enabled",
        "min_candidate_confidence",
        "min_confirmations",
        "max_missed_frames",
        "line_angle_tolerance_deg",
        "line_distance_tolerance_mm",
        "ransac_inlier_distance_mm",
        "rectangle_tolerance_fraction",
        "boundary_band_mm",
        "segment_extension_mm",
        "min_filter_confidence",
        "max_mask_age_ms",
        "mask_blur_radius_px",
        "neutral_fill_bgr",
    }
    _reject_unknown(
        field_boundary_raw,
        field_boundary_names,
        "perception.field_boundary",
    )
    for flag in ("enabled", "hard_mask_enabled"):
        if not isinstance(field_boundary_raw[flag], bool):
            raise ValueError(f"perception.field_boundary.{flag} must be a boolean.")
    neutral_fill_raw = field_boundary_raw["neutral_fill_bgr"]
    if not isinstance(neutral_fill_raw, list) or len(neutral_fill_raw) != 3:
        raise ValueError(
            "perception.field_boundary.neutral_fill_bgr must be [b, g, r]."
        )
    field_boundary = FieldBoundaryConfig(
        enabled=field_boundary_raw["enabled"],
        hard_mask_enabled=field_boundary_raw["hard_mask_enabled"],
        min_candidate_confidence=_threshold(
            field_boundary_raw["min_candidate_confidence"],
            "perception.field_boundary.min_candidate_confidence",
        ),
        min_confirmations=_positive_int(
            field_boundary_raw["min_confirmations"],
            "perception.field_boundary.min_confirmations",
        ),
        max_missed_frames=_nonnegative_int(
            field_boundary_raw["max_missed_frames"],
            "perception.field_boundary.max_missed_frames",
        ),
        line_angle_tolerance_deg=_finite_float(
            field_boundary_raw["line_angle_tolerance_deg"],
            "perception.field_boundary.line_angle_tolerance_deg",
            minimum=0.001,
        ),
        line_distance_tolerance_mm=_finite_float(
            field_boundary_raw["line_distance_tolerance_mm"],
            "perception.field_boundary.line_distance_tolerance_mm",
            minimum=0.001,
        ),
        ransac_inlier_distance_mm=_finite_float(
            field_boundary_raw["ransac_inlier_distance_mm"],
            "perception.field_boundary.ransac_inlier_distance_mm",
            minimum=0.001,
        ),
        rectangle_tolerance_fraction=_threshold(
            field_boundary_raw["rectangle_tolerance_fraction"],
            "perception.field_boundary.rectangle_tolerance_fraction",
        ),
        boundary_band_mm=_finite_float(
            field_boundary_raw["boundary_band_mm"],
            "perception.field_boundary.boundary_band_mm",
            minimum=0.001,
        ),
        segment_extension_mm=_finite_float(
            field_boundary_raw["segment_extension_mm"],
            "perception.field_boundary.segment_extension_mm",
            minimum=0.001,
        ),
        min_filter_confidence=_threshold(
            field_boundary_raw["min_filter_confidence"],
            "perception.field_boundary.min_filter_confidence",
        ),
        max_mask_age_ms=_finite_float(
            field_boundary_raw["max_mask_age_ms"],
            "perception.field_boundary.max_mask_age_ms",
            minimum=0.001,
        ),
        mask_blur_radius_px=_nonnegative_int(
            field_boundary_raw["mask_blur_radius_px"],
            "perception.field_boundary.mask_blur_radius_px",
        ),
        neutral_fill_bgr=tuple(neutral_fill_raw),
    )
    perception = PerceptionConfig(
        detection_threshold=detection_threshold,
        k0_threshold=k0_threshold,
        color_classifier=color_classifier,
        target_ground_geometry=target_ground_geometry,
        field_features=field_features,
        field_boundary=field_boundary,
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
    fusion_raw = _mapping(localization_raw["fusion"], "localization.fusion")
    _reject_unknown(
        fusion_raw,
        {
            "enabled",
            "initial_pose",
            "encoder_distance_noise_fraction",
            "encoder_heading_noise_std_deg",
            "gyro_noise_std_rad_s",
            "gyro_bias_random_walk_std_rad_s_per_sqrt_s",
            "stationary_gyro_noise_std_rad_s",
            "stationary_encoder_delta_count",
            "allow_wheel_only",
            "wheel_only_covariance_scale",
            "dropped_sample_covariance_scale",
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
        not localization_enabled
        or not motion.odometry.enabled
        or not uart.enabled
        or not perception.field_features.enabled
        or not geometry.ground_mapping_enabled
    ):
        raise ValueError(
            "Enabled localization.fusion requires localization, UART, "
            "motion.odometry, field features and ground mapping."
        )
    localization = LocalizationRuntimeConfig(center_cross_localization, fusion)

    hailo_raw = _mapping(root.get("hailo", {}), "hailo")
    _reject_unknown(
        hailo_raw,
        {
            "enabled",
            "hef_path",
            "postprocess_onnx_path",
            "output_mapping_path",
            "raw_classes",
            "class_mapping",
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
        class_mapping=class_mapping,
        backend_score_threshold=backend_score_threshold,
        max_detections=max_detections,
    )

    return AppConfig(
        camera,
        geometry,
        recording,
        processing,
        uart,
        remote,
        motion,
        tracking,
        world,
        mission,
        perception,
        localization,
        hailo,
    )
