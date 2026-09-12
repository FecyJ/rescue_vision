from __future__ import annotations

import math
import sys
from dataclasses import replace
from enum import Enum

import numpy as np
import pytest

from rescue_vision.app.breakup_planner import BreakupPlan

from rescue_vision.app import (
    GripperPosture,
    MatchSequence,
    MatchState,
    MatchPreflight,
    MatchStartArea,
    configure_match_start_area,
)
from rescue_vision.config import MatchRuntimeConfig, load_runtime_config
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.localization import (
    FieldPose2D,
    SafeZoneCornerLocalizer,
    normalize_angle,
)
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    FieldFeatureDetectionResult,
    FieldPoseKeypoint,
    PerceptionSnapshot,
    RoiColorSegmentation,
    SafeZoneColor,
    SafeZoneObservation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import MultiTargetTracker, TrackingConfig
from rescue_vision.world import (
    PhysicalRegionKind,
    PhysicalStaticRegion,
    StaticFieldMap,
    StaticSafeZoneLandmarks,
    TeamColor,
    default_static_field_map,
)


def observation(
    frame_sequence: int,
    timestamp_ns: int,
    ground: GroundPoint,
    *,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
    box_x: float = 10.0,
) -> TargetObservation:
    box = UndistortedBoundingBox(box_x, 10.0, box_x + 10.0, 20.0)
    segmentation = RoiColorSegmentation(
        candidate_class=target_class,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=box,
        mask=np.full((10, 10), 255, dtype=np.uint8),
        color_fraction=1.0,
        dominance=1.0,
    )
    return TargetObservation(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        image_size=(100, 100),
        model_target_class=target_class,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(target_class, 1.0),
        detection_confidence=0.95,
        box=box,
        color_segmentation=segmentation,
        k0=UndistortedPixel(box_x + 5.0, 15.0),
        k0_confidence=0.95,
        ground_point=ground,
        quality=frozenset(),
    )


def snapshot(
    frame_sequence: int,
    timestamp_ns: int,
    *observations_: TargetObservation,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        observations=tuple(observations_),
        field_features=None,
    )


def safe_zone_snapshot(
    frame_sequence: int,
    timestamp_ns: int,
    k0: GroundPoint | None,
    k1: GroundPoint | None,
    k2: GroundPoint | None,
    *,
    box: UndistortedBoundingBox | None = None,
    image_size: tuple[int, int] = (100, 100),
) -> PerceptionSnapshot:
    box = box or UndistortedBoundingBox(10.0, 10.0, 90.0, 90.0)

    def keypoint(ground: GroundPoint | None) -> FieldPoseKeypoint:
        if ground is None:
            return FieldPoseKeypoint(None, None, 0.0)
        return FieldPoseKeypoint(UndistortedPixel(50.0, 50.0), ground, 0.9)

    zone = SafeZoneObservation(
        box,
        keypoint(k0),
        keypoint(k1),
        keypoint(k2),
        SafeZoneColor.RED,
        0.9,
        frozenset(),
    )
    features = FieldFeatureDetectionResult(
        frame_sequence,
        timestamp_ns,
        timestamp_ns,
        image_size,
        (zone,),
        None,
    )
    return PerceptionSnapshot(
        frame_sequence,
        timestamp_ns,
        timestamp_ns,
        (),
        features,
    )


def field_to_ground(pose: FieldPose2D, point: FieldPoint) -> GroundPoint:
    delta_x = point.x - pose.position.x
    delta_y = point.y - pose.position.y
    cosine = math.cos(pose.heading_rad)
    sine = math.sin(pose.heading_rad)
    return GroundPoint(
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
    )


def safe_zone_snapshot_for_pose(
    frame_sequence: int,
    timestamp_ns: int,
    pose: FieldPose2D,
) -> PerceptionSnapshot:
    return safe_zone_snapshot(
        frame_sequence,
        timestamp_ns,
        field_to_ground(pose, FieldPoint(0.0, 1137.0)),
        field_to_ground(pose, FieldPoint(-330.0, 1137.0)),
        field_to_ground(pose, FieldPoint(330.0, 1137.0)),
    )


def make_static_map() -> StaticFieldMap:
    return StaticFieldMap(
        default_static_field_map().center_cross,
        (
            PhysicalStaticRegion(
                "field",
                PhysicalRegionKind.FIELD,
                (
                    FieldPoint(-1500.0, -1500.0),
                    FieldPoint(1500.0, -1500.0),
                    FieldPoint(1500.0, 1500.0),
                    FieldPoint(-1500.0, 1500.0),
                ),
            ),
            PhysicalStaticRegion(
                "red-material",
                PhysicalRegionKind.RED_MATERIAL,
                (
                    FieldPoint(-300.0, 1200.0),
                    FieldPoint(0.0, 1200.0),
                    FieldPoint(0.0, 1500.0),
                    FieldPoint(-300.0, 1500.0),
                ),
            ),
            PhysicalStaticRegion(
                "red-injured",
                PhysicalRegionKind.RED_INJURED,
                (
                    FieldPoint(0.0, 1200.0),
                    FieldPoint(300.0, 1200.0),
                    FieldPoint(300.0, 1500.0),
                    FieldPoint(0.0, 1500.0),
                ),
            ),
            PhysicalStaticRegion(
                "blue-injured",
                PhysicalRegionKind.BLUE_INJURED,
                (
                    FieldPoint(-300.0, -1500.0),
                    FieldPoint(0.0, -1500.0),
                    FieldPoint(0.0, -1200.0),
                    FieldPoint(-300.0, -1200.0),
                ),
            ),
            PhysicalStaticRegion(
                "blue-material",
                PhysicalRegionKind.BLUE_MATERIAL,
                (
                    FieldPoint(0.0, -1500.0),
                    FieldPoint(300.0, -1500.0),
                    FieldPoint(300.0, -1200.0),
                    FieldPoint(0.0, -1200.0),
                ),
            ),
        ),
        (
            StaticSafeZoneLandmarks(
                TeamColor.RED,
                FieldPoint(0.0, 1137.0),
                FieldPoint(-330.0, 1137.0),
                FieldPoint(330.0, 1137.0),
                True,
                True,
            ),
            StaticSafeZoneLandmarks(
                TeamColor.BLUE,
                FieldPoint(0.0, -1137.0),
                FieldPoint(330.0, -1137.0),
                FieldPoint(-330.0, -1137.0),
                True,
                True,
            ),
        ),
    )


def runtime_config(**overrides: object) -> MatchRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "opportunistic_single_green_enabled": True,
        "green_grab_offset_mm": 150.0,
        "safe_zone_fallback_target_field": FieldPoint(-165.0, 1137.0),
        # Most state-transition tests focus on the following action. The
        # dataclass default still gates on three frames, so pin the production
        # single-frame value explicitly here.
        "green_alignment_stable_frames": 1,
    }
    values.update(overrides)
    return MatchRuntimeConfig(**values)  # type: ignore[arg-type]


def make_sequence(
    *, config: MatchRuntimeConfig | None = None,
    initial_field_position: FieldPoint | None = FieldPoint(0.0, 0.0),
    team_color: TeamColor = TeamColor.RED,
    near_field_grasp_config: NearFieldGraspConfig | None = None,
) -> MatchSequence:
    runtime = config or runtime_config()
    return MatchSequence(
        runtime,
        tracker=MultiTargetTracker(
            TrackingConfig(
                confirmation_hits=2,
                max_association_ground_mm=250.0,
                min_association_iou=0.1,
                max_coast_ms=600.0,
                confidence_decay_per_second=0.8,
                min_confidence=0.15,
            )
        ),
        gripper_full_travel_time_s=1.0,
        team_color=team_color,
        safe_zone_corner_localizer=SafeZoneCornerLocalizer(make_static_map()),
        static_map=make_static_map(),
        near_field_grasp_config=near_field_grasp_config,
        breakup_target_geometry=load_runtime_config("configs/runtime.match.yaml").perception.target_ground_geometry,
        initial_field_position=initial_field_position,
    )



def inject_breakup_plan(sequence, *, heading=0.0, approach=100.0):
    sequence._breakup_plan = BreakupPlan((1, 2), (1, 2), 1, GroundPoint(450,0), heading,
        approach, sequence.config.breakup_forward_distance_m*1000,
        sequence.config.breakup_backward_distance_m*1000, 80., 0, 1,
        (FieldPoint(450,0), FieldPoint(480,0)), FieldPoint(450,0), 160.)
    sequence._breakup_forward_base_distance_m = sequence._breakup_backward_base_distance_m = 0.
    sequence._cluster_approach_base_distance_m = 0.
    sequence._breakup_retreat_mm = sequence.config.breakup_backward_distance_m*1000

def start_sequence(sequence: MatchSequence) -> None:
    ready = sequence.preflight(
        0,
        MatchPreflight(True, True, True, True, True, True),
    )
    assert ready.state is MatchState.PREFLIGHT
    sequence.start(1)
    sequence._started = True
    sequence.state = MatchState.SEARCH_CLUSTER


def test_start_area_3_mirrors_match_pose_route_and_team_color() -> None:
    area_2 = load_runtime_config("configs/runtime.match.yaml")
    area_3 = configure_match_start_area(area_2, MatchStartArea.AREA_3)

    assert area_3.world.team_color is TeamColor.BLUE
    assert area_3.localization.fusion.initial_pose.position == FieldPoint(
        -1350.0, -1350.0
    )
    assert area_3.localization.fusion.initial_pose.heading_rad == pytest.approx(
        math.pi / 2.0
    )
    assert area_3.match.safe_zone_fallback_target_field == FieldPoint(
        130.0, -1115.0
    )
    assert area_3.match.safe_zone_injured_target_field == FieldPoint(
        -130.0, -1115.0
    )
    assert area_3.match.safe_zone_d2_braking_overrun_x_mm == pytest.approx(-20.0)
    assert area_3.world.static_map is area_2.world.static_map

    sequence = MatchSequence.from_app_config(area_2, start_area=3)
    assert sequence._team_color is TeamColor.BLUE
    assert sequence.estimated_field_position == FieldPoint(-1350.0, -1350.0)
    assert sequence._safe_zone_transport_endpoint() == FieldPoint(130.0, -1115.0)


def test_blue_start_area_uses_negative_y_safe_zone_route() -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_fallback_target_field=FieldPoint(135.0, -1115.0),
            safe_zone_injured_target_field=FieldPoint(-135.0, -1115.0),
            safe_zone_calibration_start_offset_mm=700.0,
            safe_zone_open_offset_mm=300.0,
            safe_zone_d2_to_final_braking_overrun_mm=25.0,
            safe_zone_d2_braking_overrun_x_mm=-20.0,
        ),
        initial_field_position=FieldPoint(0.0, -700.0),
        team_color=TeamColor.BLUE,
    )

    assert sequence._safe_zone_d1_target() == FieldPoint(155.0, -415.0)
    assert sequence._safe_zone_d2_target() == FieldPoint(155.0, -815.0)
    assert sequence._safe_zone_final_target_y_mm() == pytest.approx(-1090.0)

    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "stopping_before_calibration"
    sequence.step(
        10,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    rotating = sequence.step(
        300_000_010,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert rotating.reason == "safe_zone_no_bbox_rotate_toward_minus_90"
    assert rotating.angular_velocity_rad_s < 0.0

    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_y_at_d2"
    turn_to_blue_zone = sequence.step(
        300_000_011,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert turn_to_blue_zone.reason == "safe_zone_d2_turn_to_minus_90"
    assert turn_to_blue_zone.angular_velocity_rad_s < 0.0


@pytest.mark.parametrize(
    ("team_color", "endpoint", "position"),
    [
        (TeamColor.RED, FieldPoint(-165.0, 1137.0), FieldPoint(80.0, 700.0)),
        (TeamColor.BLUE, FieldPoint(165.0, -1137.0), FieldPoint(-80.0, -700.0)),
    ],
)
def test_transport_already_beyond_d1_calibrates_in_place(
    team_color: TeamColor,
    endpoint: FieldPoint,
    position: FieldPoint,
) -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_fallback_target_field=endpoint,
            safe_zone_calibration_start_offset_mm=500.0,
        ),
        initial_field_position=position,
        team_color=team_color,
    )
    start_sequence(sequence)

    decision = sequence._start_safe_zone_transport(
        10,
        transport_opened=False,
        posture=GripperPosture.CLOSED,
        reason="would_start_d1_line",
    )

    assert decision.reason == "gripper_closed_already_beyond_d1_start_calibration"
    assert decision.state is MatchState.TRANSPORT_RELEASE
    assert decision.linear_velocity_m_s == 0.0
    assert sequence.safe_zone_route_phase == "stopping_before_calibration"


def test_cluster_search_uses_fast_speed_until_collectible_information_appears() -> None:
    sequence = make_sequence(
        config=runtime_config(
            cluster_search_angular_velocity_rad_s=-0.2,
            cluster_search_empty_angular_velocity_rad_s=-0.6,
            cluster_min_detections=2,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence._breakup_only = True

    empty = sequence.step(
        10,
        perception=snapshot(1, 10),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    blue_only = sequence.step(
        20,
        perception=snapshot(
            2,
            20,
            observation(
                2,
                20,
                GroundPoint(500.0, 0.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        heading_rad=-0.01,
        cumulative_distance_m=0.0,
    )
    orange_seen = sequence.step(
        30,
        perception=snapshot(
            3,
            30,
            observation(
                3,
                30,
                GroundPoint(500.0, 0.0),
                target_class=TargetClass.ORANGE_INJURED,
            ),
        ),
        heading_rad=-0.02,
        cumulative_distance_m=0.0,
    )

    assert empty.angular_velocity_rad_s == pytest.approx(-0.6)
    assert blue_only.angular_velocity_rad_s == pytest.approx(-0.6)
    assert orange_seen.angular_velocity_rad_s == pytest.approx(-0.2)
    assert (
        empty.reason
        == blue_only.reason
        == orange_seen.reason
        == "search_cluster_right"
    )


def test_cluster_search_uses_fast_speed_when_targets_are_inside_safe_zone_bbox() -> None:
    sequence = make_sequence(
        config=runtime_config(
            cluster_search_angular_velocity_rad_s=-0.2,
            cluster_search_empty_angular_velocity_rad_s=-0.6,
            cluster_min_detections=2,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence._breakup_only = True

    safe_zone = safe_zone_snapshot(
        1,
        10,
        None,
        None,
        None,
        box=UndistortedBoundingBox(10.0, 10.0, 90.0, 90.0),
    )
    perception = replace(
        safe_zone,
        observations=(
            observation(
                1,
                10,
                GroundPoint(500.0, 0.0),
                target_class=TargetClass.ORANGE_INJURED,
                box_x=40.0,
            ),
        ),
    )

    decision = sequence.step(
        10,
        perception=perception,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert decision.angular_velocity_rad_s == pytest.approx(-0.6)
    assert decision.reason == "search_cluster_right"


def test_match_cli_passes_selected_start_area(monkeypatch) -> None:
    from rescue_vision.app import match as match_module
    import rescue_vision.app.match_runtime as match_runtime

    received: dict[str, object] = {}
    monkeypatch.setattr(
        match_runtime,
        "_run_hardware",
        lambda *args, **kwargs: received.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-match",
            "--config",
            "configs/runtime.match.yaml",
            "--start-area",
            "3",
        ],
    )

    match_module.main()

    assert received["start_area"] is MatchStartArea.AREA_3


def test_start_area_parser_accepts_equivalent_enum_instance() -> None:
    class OtherModuleStartArea(Enum):
        AREA_3 = "3"

    assert MatchStartArea.parse(OtherModuleStartArea.AREA_3) is MatchStartArea.AREA_3


def test_config_uses_configured_gripper_and_transport_values() -> None:
    config = load_runtime_config("configs/runtime.match.yaml")

    assert config.motion.gripper.transport_left_angle_deg is not None
    assert config.motion.gripper.transport_right_angle_deg is not None
    assert (
        config.motion.gripper.transport_left_angle_deg
        + config.motion.gripper.transport_right_angle_deg
        == 180.0
    )
    assert config.near_field_grasp.max_range_mm == pytest.approx(450.0)
    assert config.near_field_grasp.max_targets == 3
    assert config.match.opportunistic_single_green_enabled
    assert (
        config.match.opportunistic_single_green_clearance_mm
        == pytest.approx(120.0)
    )
    assert config.match.safe_zone_grab_to_d1_speed_m_s == pytest.approx(
        0.3
    )
    assert config.match.safe_zone_d1_to_d2_speed_m_s == pytest.approx(
        0.35
    )
    assert config.match.safe_zone_d2_to_final_speed_m_s == pytest.approx(
        0.45
    )
    assert config.match.safe_zone_orange_d2_to_final_speed_m_s == pytest.approx(
        0.45
    )
    assert (
        config.match.safe_zone_d2_to_final_max_wheel_acceleration_m_s2
        == pytest.approx(4.0)
    )
    assert (
        config.match.safe_zone_d2_to_final_braking_overrun_mm
        == pytest.approx(70.0)
    )
    assert (
        config.match.safe_zone_orange_d2_to_final_braking_overrun_mm
        == pytest.approx(80.0)
    )
    assert config.match.breakup_max_wheel_acceleration_m_s2 == pytest.approx(4.0)
    assert config.match.breakup_confirmation_frames == 1
    assert config.match.startup_turn_settle_time_s == pytest.approx(
        0.1
    )
    assert config.match.startup_forward_settle_time_s == pytest.approx(
        0.1
    )
    assert config.match.breakup_settle_time_s == pytest.approx(0.1)
    assert config.match.green_alignment_tolerance_mm == pytest.approx(40.0)
    assert config.match.green_alignment_hysteresis_mm == pytest.approx(20.0)
    assert config.match.green_alignment_timeout_ms == pytest.approx(6000.0)
    assert config.match.green_alignment_stable_frames == 1
    assert config.match.green_alignment_min_wheel_velocity_m_s == pytest.approx(
        0.01
    )
    assert config.near_field_grasp.center_tolerance_mm == pytest.approx(5.0)
    assert config.near_field_grasp.alignment_hysteresis_mm == pytest.approx(10.0)
    assert config.near_field_grasp.alignment_timeout_ms == pytest.approx(1800.0)
    # 无方案等待必须短于停稳提交预算：车停稳后重复观测不改变门禁结果，
    # 空等只会推迟换候选/解团；提交预算要覆盖实测 1.0~2.0 s 的确认链路。
    assert config.near_field_grasp.no_plan_wait_ms == pytest.approx(800.0)
    assert (
        config.near_field_grasp.no_plan_wait_ms
        < config.near_field_grasp.alignment_timeout_ms
    )
    assert config.near_field_grasp.grasp_commit_max_observation_age_ms == pytest.approx(
        150.0
    )
    assert config.near_field_grasp.confirmation_frames == 1
    assert config.near_field_grasp.fine_alignment_zone_rad == pytest.approx(0.08)
    assert config.near_field_grasp.fine_alignment_min_wheel_velocity_m_s == pytest.approx(
        0.0
    )
    assert config.near_field_grasp.orange_isolation_radius_mm == pytest.approx(
        60.0
    )
    assert config.match.safe_zone_fallback_target_field == FieldPoint(
        -130.0, 1115.0
    )
    assert config.match.safe_zone_injured_target_field == FieldPoint(
        130.0, 1115.0
    )
    assert config.match.safe_zone_calibration_start_offset_mm == 700.0
    assert config.match.safe_zone_open_offset_mm == 300.0
    assert (
        config.match.safe_zone_d2_braking_overrun_x_mm
        == pytest.approx(20.0)
    )
    assert (
        config.match.safe_zone_d2_braking_overrun_y_mm
        == pytest.approx(0.0)
    )
    assert (
        config.match.safe_zone_calibration_stop_speed_threshold_m_s
        == pytest.approx(0.02)
    )
    assert (
        config.match.safe_zone_calibration_stop_confirm_time_s
        == pytest.approx(0.3)
    )
    assert config.match.action_settle_time_s == pytest.approx(0.1)
    assert config.match.breakup_field_half_extent_mm == pytest.approx(
        1500.0
    )
    assert config.match.breakup_gripper_offset_mm == pytest.approx(
        200.0
    )
    assert config.match.safe_zone_exit_distance_m == pytest.approx(0.5)
    assert config.match.cluster_search_empty_angular_velocity_rad_s == pytest.approx(-0.8)
    assert config.match.safe_zone_key_search_angular_velocity_rad_s == pytest.approx(0.8)
    assert config.match.safe_zone_bbox_turn_kp_rad_s == pytest.approx(0.8)
    assert config.match.safe_zone_bbox_turn_max_angular_velocity_rad_s == pytest.approx(0.25)
    assert config.match.safe_zone_bbox_turn_deadband_ratio == pytest.approx(0.02)
    assert config.match.safe_zone_bbox_turn_resume_ratio == pytest.approx(0.03)
    assert config.match.safe_zone_keypoint_reobserve_timeout_s == pytest.approx(0.8)
    assert config.match.safe_zone_keypoint_reverse_speed_m_s == pytest.approx(0.08)
    assert config.match.safe_zone_keypoint_reverse_max_distance_m == pytest.approx(0.20)
    assert config.match.safe_zone_bbox_edge_margin_px == pytest.approx(12.0)
    assert isinstance(
        MatchSequence.from_app_config(config),
        MatchSequence,
    )


@pytest.mark.parametrize(
    "phase",
    [
        "stopping_before_d2_opening",
        "opening_at_d2",
        "align_y_at_d2",
        "stopping_after_d2_heading",
        "closing_before_final_forward",
        "forward_final_closed",
        "stopping_before_exit_opening",
        "opening_after_transport",
    ],
)
def test_d2_acceleration_limit_is_active_only_for_d2_route(
    phase: str,
) -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_d2_to_final_max_wheel_acceleration_m_s2=0.12,
        )
    )
    start_sequence(sequence)
    sequence._safe_zone_phase = phase
    sequence.state = (
        MatchState.TRANSPORT_FORWARD
        if phase == "forward_final_closed"
        else MatchState.TRANSPORT_RELEASE
    )

    assert sequence.safe_zone_motion_acceleration_limit_m_s2 == pytest.approx(0.12)

    sequence._safe_zone_phase = "forward_d2_line"
    assert sequence.safe_zone_motion_acceleration_limit_m_s2 is None


@pytest.mark.parametrize(
    "state",
    [
        MatchState.ALIGN_CLUSTER_ONCE,
        MatchState.APPROACH_CLUSTER,
        MatchState.BREAKUP_SETTLE,
        MatchState.BREAKUP_FORWARD,
        MatchState.OPEN_GRIPPER_SETTLE,
        MatchState.BREAKUP_BACKWARD,
        MatchState.CLOSE_GRIPPER_SETTLE,
        MatchState.CLOSE_GRIPPER_SPIN,
        MatchState.CHECK_ISOLATED_GREEN,
        MatchState.RELOCATE_FORWARD,
    ],
)
def test_breakup_acceleration_limit_is_active_for_breakup_states(
    state: MatchState,
) -> None:
    sequence = make_sequence(
        config=runtime_config(breakup_max_wheel_acceleration_m_s2=0.12)
    )
    sequence.state = state

    assert sequence.breakup_motion_acceleration_limit_m_s2 == pytest.approx(0.12)
    assert sequence.motion_acceleration_limit_m_s2 == pytest.approx(0.12)

    sequence.state = MatchState.SEARCH_CLUSTER
    assert sequence.breakup_motion_acceleration_limit_m_s2 is None
    assert sequence.motion_acceleration_limit_m_s2 is None


def test_uses_configured_transport_endpoint_for_d1_and_d2() -> None:
    endpoint = FieldPoint(-210.0, 1200.0)
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_fallback_target_field=endpoint,
            safe_zone_calibration_start_offset_mm=250.0,
            safe_zone_open_offset_mm=100.0,
            safe_zone_d2_braking_overrun_x_mm=15.0,
            safe_zone_d2_braking_overrun_y_mm=-12.0,
        ),
        initial_field_position=FieldPoint(-100.0, 700.0),
    )

    assert sequence._safe_zone_transport_endpoint() == endpoint
    assert sequence._safe_zone_d1_target() == FieldPoint(-225.0, 962.0)
    assert sequence._safe_zone_d2_target() == FieldPoint(-225.0, 1112.0)
    assert sequence._safe_zone_final_target_y_mm() == pytest.approx(1200.0)

    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d1_line"
    turn_to_d1 = sequence.step(
        10,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert turn_to_d1.reason == "safe_zone_turn_to_d1_line"

    d1_forward_start = sequence.step(
        20,
        perception=None,
        heading_rad=math.atan2(262.0, -125.0),
        cumulative_distance_m=0.0,
    )
    assert d1_forward_start.state is MatchState.TRANSPORT_FORWARD
    assert sequence._transport_forward_distance_m == pytest.approx(
        math.hypot(125.0, 262.0) / 1000.0
    )


def test_d2_target_applies_signed_xy_braking_overrun_compensation() -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_d2_braking_overrun_x_mm=35.0,
            safe_zone_d2_braking_overrun_y_mm=-12.0,
        )
    )

    # Nominal d2 is (-165, 1000); the effective stop target subtracts the
    # signed expected post-brake displacement in FieldPoint coordinates.
    assert sequence._safe_zone_d2_target() == FieldPoint(-200.0, 1012.0)


def test_d2_line_distance_uses_compensated_target() -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_d2_braking_overrun_x_mm=15.0,
            safe_zone_d2_braking_overrun_y_mm=20.0,
            transport_align_tolerance_mm=1.0,
        ),
        initial_field_position=FieldPoint(-120.0, 700.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d2_line"

    target = FieldPoint(-180.0, 980.0)
    heading = math.atan2(target.y - 700.0, target.x + 120.0)
    decision = sequence.step(
        10,
        perception=None,
        heading_rad=heading,
        cumulative_distance_m=0.0,
    )

    assert decision.state is MatchState.TRANSPORT_FORWARD
    assert sequence._d2_line_start_position == FieldPoint(-120.0, 700.0)
    assert sequence._transport_forward_distance_m == pytest.approx(
        math.hypot(target.x + 120.0, target.y - 700.0) / 1000.0
    )


def test_d1_bbox_search_rotates_until_keypoints_are_visible() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "stopping_before_calibration"
    bbox = UndistortedBoundingBox(0.0, 10.0, 40.0, 90.0)

    bbox_only = safe_zone_snapshot(1, 10, None, None, None, box=bbox)
    waiting_for_stop = sequence.step(
        10,
        perception=bbox_only,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting_for_stop.reason == "safe_zone_waiting_for_vehicle_stop_before_calibration"

    search_started = sequence.step(
        300_000_010,
        perception=bbox_only,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert search_started.reason == "safe_zone_center_full_bbox_before_keypoints"
    assert search_started.angular_velocity_rad_s > 0.0

    rotating = sequence.step(
        300_000_011,
        perception=bbox_only,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert rotating.reason == "safe_zone_center_full_bbox_before_keypoints"
    assert rotating.linear_velocity_m_s == 0.0
    assert rotating.angular_velocity_rad_s > 0.0

    keypoints_visible = sequence.step(
        300_000_012,
        perception=safe_zone_snapshot(
            2,
            300_000_012,
            GroundPoint(400.0, 0.0),
            GroundPoint(350.0, 50.0),
            GroundPoint(350.0, -50.0),
            box=UndistortedBoundingBox(10.0, 10.0, 90.0, 90.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert keypoints_visible.reason == "safe_zone_keypoints_seen_stop_before_calibration"

    stopped_waiting = sequence.step(
        300_000_013,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert stopped_waiting.reason == "safe_zone_waiting_for_vehicle_stop_after_keypoints"

    stopped = sequence.step(
        600_000_013,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert stopped.reason == "safe_zone_keypoints_stopped_start_calibration"


def test_centered_missing_keypoints_reverses_until_they_appear() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "searching_safe_zone_keypoints"

    centered_partial = safe_zone_snapshot(
        1,
        10,
        GroundPoint(400.0, 0.0),
        GroundPoint(350.0, 50.0),
        None,
        box=UndistortedBoundingBox(40.0, 20.0, 60.0, 80.0),
    )
    started = sequence.step(
        10,
        perception=centered_partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert started.reason == "safe_zone_keypoints_missing_start_reobserve"
    assert started.linear_velocity_m_s == 0.0
    assert started.angular_velocity_rad_s == 0.0
    assert sequence._safe_zone_phase == "reobserving_safe_zone_keypoints"

    reversing = sequence.step(
        20,
        perception=centered_partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert reversing.reason == "safe_zone_keypoints_reobserve_waiting_for_new_frame"
    assert reversing.linear_velocity_m_s == 0.0
    assert reversing.angular_velocity_rad_s == 0.0

    complete = sequence.step(
        30,
        perception=safe_zone_snapshot(
            2,
            30,
            GroundPoint(450.0, 0.0),
            GroundPoint(400.0, 50.0),
            GroundPoint(400.0, -50.0),
            box=UndistortedBoundingBox(40.0, 20.0, 60.0, 80.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=-0.05,
        left_speed_feedback_m_s=-0.05,
        right_speed_feedback_m_s=-0.05,
    )
    assert complete.reason == "safe_zone_keypoints_reobserved_stop_before_calibration"
    assert complete.linear_velocity_m_s == 0.0
    assert complete.angular_velocity_rad_s == 0.0
    assert sequence._safe_zone_phase == "stopping_after_bbox_keypoints"


def test_complete_keypoints_are_centered_before_calibration_stop() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "searching_safe_zone_keypoints"

    off_center = sequence.step(
        10,
        perception=safe_zone_snapshot(
            1,
            10,
            GroundPoint(400.0, 0.0),
            GroundPoint(350.0, 50.0),
            GroundPoint(350.0, -50.0),
            box=UndistortedBoundingBox(10.0, 10.0, 50.0, 90.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert off_center.reason == "safe_zone_center_full_bbox_before_keypoints"
    assert off_center.angular_velocity_rad_s > 0.0
    assert sequence._safe_zone_phase == "searching_safe_zone_keypoints"

    centered = sequence.step(
        20,
        perception=safe_zone_snapshot(
            2,
            20,
            GroundPoint(400.0, 0.0),
            GroundPoint(350.0, 50.0),
            GroundPoint(350.0, -50.0),
            box=UndistortedBoundingBox(40.0, 10.0, 60.0, 90.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert centered.reason == "safe_zone_keypoints_seen_stop_before_calibration"
    assert centered.linear_velocity_m_s == 0.0
    assert centered.angular_velocity_rad_s == 0.0
    assert sequence._safe_zone_phase == "stopping_after_bbox_keypoints"


def test_collecting_partial_keypoints_restarts_bbox_search() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "collecting_safe_zone_keys_closed"
    bbox = UndistortedBoundingBox(0.0, 10.0, 40.0, 90.0)

    decision = sequence.step(
        10,
        perception=safe_zone_snapshot(
            1,
            10,
            GroundPoint(400.0, 0.0),
            GroundPoint(350.0, 50.0),
            None,
            box=bbox,
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )

    assert sequence._safe_zone_phase == "reobserving_safe_zone_keypoints"
    assert decision.reason == "safe_zone_keypoints_missing_start_reobserve"
    assert decision.linear_velocity_m_s == 0.0
    assert decision.angular_velocity_rad_s == 0.0

    waiting = sequence.step(
        100,
        perception=sequence._latest_perception,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting.reason == "safe_zone_keypoints_reobserve_waiting_for_new_frame"
    assert waiting.angular_velocity_rad_s == 0.0

    expired = sequence.step(
        800_000_011,
        perception=sequence._latest_perception,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert sequence._safe_zone_phase == "searching_safe_zone_keypoints"
    assert expired.reason == "safe_zone_center_full_bbox_before_keypoints"
    assert expired.angular_velocity_rad_s > 0.0


def test_safe_zone_bbox_turn_uses_proportional_speed_and_deadband() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "searching_safe_zone_keypoints"
    sequence._safe_zone_keypoint_scan_attempted = True
    sequence._safe_zone_keypoint_reverse_attempted = True

    centered = sequence.step(
        10,
        perception=safe_zone_snapshot(
            1,
            10,
            None,
            None,
            None,
            box=UndistortedBoundingBox(40.0, 20.0, 60.0, 80.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert centered.angular_velocity_rad_s == 0.0

    small_error = sequence.step(
        20,
        perception=safe_zone_snapshot(
            2,
            20,
            None,
            None,
            None,
            box=UndistortedBoundingBox(41.0, 20.0, 61.0, 80.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert small_error.angular_velocity_rad_s == 0.0

    large_error = sequence.step(
        30,
        perception=safe_zone_snapshot(
            3,
            30,
            None,
            None,
            None,
            box=UndistortedBoundingBox(60.0, 20.0, 80.0, 80.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert large_error.angular_velocity_rad_s == pytest.approx(-0.25)

    near_error = sequence.step(
        40,
        perception=safe_zone_snapshot(
            4,
            40,
            None,
            None,
            None,
            box=UndistortedBoundingBox(47.5, 20.0, 67.5, 80.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert near_error.angular_velocity_rad_s == pytest.approx(-0.12)


def test_safe_zone_bbox_reversal_waits_for_stop_and_new_frame() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "searching_safe_zone_keypoints"

    left = safe_zone_snapshot(
        1,
        10,
        None,
        None,
        None,
        box=UndistortedBoundingBox(20.0, 20.0, 40.0, 80.0),
    )
    right = safe_zone_snapshot(
        2,
        20,
        None,
        None,
        None,
        box=UndistortedBoundingBox(60.0, 20.0, 80.0, 80.0),
    )
    right_new = safe_zone_snapshot(
        3,
        500_000_000,
        None,
        None,
        None,
        box=UndistortedBoundingBox(60.0, 20.0, 80.0, 80.0),
    )
    first = sequence.step(
        10,
        perception=left,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert first.angular_velocity_rad_s > 0.0

    requested = sequence.step(
        20,
        perception=right,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.1,
        right_speed_feedback_m_s=-0.1,
    )
    assert requested.angular_velocity_rad_s == 0.0

    waiting_stop = sequence.step(
        100,
        perception=right,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting_stop.reason == "safe_zone_waiting_for_vehicle_stop_before_reversal"

    waiting_frame = sequence.step(
        300_000_100,
        perception=right,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting_frame.reason == "safe_zone_waiting_for_new_frame_before_reversal"

    resumed = sequence.step(
        500_000_000,
        perception=right_new,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert resumed.angular_velocity_rad_s < 0.0
    assert resumed.reason == "safe_zone_center_full_bbox_before_keypoints"


def test_safe_zone_keypoint_reobserve_accepts_only_a_new_complete_frame() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "collecting_safe_zone_keys_closed"
    partial = safe_zone_snapshot(
        1,
        10,
        GroundPoint(400.0, 0.0),
        GroundPoint(350.0, 50.0),
        None,
    )
    started = sequence.step(
        10,
        perception=partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert started.reason == "safe_zone_keypoints_missing_start_reobserve"

    duplicate = sequence.step(
        100,
        perception=partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert duplicate.reason == "safe_zone_keypoints_reobserve_waiting_for_new_frame"

    complete = sequence.step(
        200,
        perception=safe_zone_snapshot(
            2,
            200,
            GroundPoint(400.0, 0.0),
            GroundPoint(350.0, 50.0),
            GroundPoint(350.0, -50.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert complete.reason == "safe_zone_keypoints_reobserved_stop_before_calibration"
    assert sequence._safe_zone_phase == "stopping_after_bbox_keypoints"


def test_safe_zone_centered_missing_keypoints_get_one_bounded_scan() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "collecting_safe_zone_keys_closed"
    centered_partial = safe_zone_snapshot(
        1,
        10,
        GroundPoint(400.0, 0.0),
        GroundPoint(350.0, 50.0),
        None,
        box=UndistortedBoundingBox(40.0, 20.0, 60.0, 80.0),
    )
    sequence.step(
        10,
        perception=centered_partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    scanning = sequence.step(
        800_000_011,
        perception=centered_partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert sequence._safe_zone_phase == "scanning_safe_zone_keypoints"
    assert scanning.reason == "safe_zone_keypoint_reobserve_scan"
    assert scanning.angular_velocity_rad_s == pytest.approx(0.25)

    scan_done = sequence.step(
        1_600_000_011,
        perception=centered_partial,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert sequence._safe_zone_phase == "reobserving_safe_zone_keypoints"
    assert scan_done.reason == "safe_zone_keypoint_scan_complete_reobserve"


def test_rejected_safe_zone_calibration_restarts_turning_search() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "collecting_safe_zone_keys_closed"
    triple = (
        GroundPoint(400.0, 0.0),
        GroundPoint(350.0, 50.0),
        GroundPoint(350.0, -50.0),
    )
    sequence._safe_zone_key_samples = [triple] * 4
    perception = safe_zone_snapshot(
        1,
        10,
        *triple,
        box=UndistortedBoundingBox(12.0, 10.0, 52.0, 90.0),
    )

    decision = sequence.step(
        10,
        perception=perception,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert sequence._safe_zone_phase == "searching_safe_zone_keypoints"
    assert decision.reason == "safe_zone_visual_calibration_rejected_rotate_for_keypoints"
    assert decision.linear_velocity_m_s == 0.0
    assert decision.angular_velocity_rad_s > 0.0
    assert sequence._safe_zone_key_samples == []


def test_d1_without_bbox_rotates_toward_plus_90_until_keypoints_appear() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_RELEASE
    sequence._safe_zone_phase = "stopping_before_calibration"

    waiting_for_stop = sequence.step(
        10,
        perception=None,
        heading_rad=math.pi,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting_for_stop.reason == "safe_zone_waiting_for_vehicle_stop_before_calibration"

    rotating = sequence.step(
        300_000_010,
        perception=None,
        heading_rad=math.pi,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert sequence._safe_zone_phase == "searching_safe_zone_keypoints"
    assert rotating.reason == "safe_zone_no_bbox_rotate_toward_plus_90"
    assert rotating.linear_velocity_m_s == 0.0
    assert rotating.angular_velocity_rad_s < 0.0

    still_rotating = sequence.step(
        300_000_011,
        perception=None,
        heading_rad=math.radians(100.0),
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert still_rotating.reason == "safe_zone_no_bbox_rotate_toward_plus_90"
    assert still_rotating.angular_velocity_rad_s < 0.0

    keypoints_visible = sequence.step(
        300_000_012,
        perception=safe_zone_snapshot(
            2,
            300_000_012,
            GroundPoint(400.0, 0.0),
            GroundPoint(350.0, 50.0),
            GroundPoint(350.0, -50.0),
        ),
        heading_rad=math.radians(100.0),
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert keypoints_visible.reason == "safe_zone_keypoints_seen_stop_before_calibration"
    assert keypoints_visible.linear_velocity_m_s == 0.0
    assert keypoints_visible.angular_velocity_rad_s == 0.0

    counter_clockwise_sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(counter_clockwise_sequence)
    counter_clockwise_sequence.state = MatchState.TRANSPORT_RELEASE
    counter_clockwise_sequence._safe_zone_phase = "stopping_before_calibration"
    counter_clockwise_sequence.step(
        10,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    counter_clockwise = counter_clockwise_sequence.step(
        300_000_010,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert counter_clockwise.reason == "safe_zone_no_bbox_rotate_toward_plus_90"
    assert counter_clockwise.angular_velocity_rad_s > 0.0


def test_directly_aligns_without_confirm_reverse_scan() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=150.0,
        )
    )
    start_sequence(sequence)

    first = sequence.step(
        10,
        perception=snapshot(1, 10, observation(1, 10, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert first.state is MatchState.SEARCH_CLUSTER
    sequence.state = MatchState.SEARCH_CLUSTER
    found = sequence.step(
        20,
        perception=snapshot(2, 20, observation(2, 20, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert found.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert "transport_confirm_green" not in {state.value for state in MatchState}
    assert found.gripper_posture is GripperPosture.TRANSPORT


def test_search_preempts_breakup_for_confirmed_single_green() -> None:
    sequence = make_sequence(
        config=runtime_config(
            opportunistic_single_green_enabled=True,
            opportunistic_single_green_clearance_mm=120.0,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    first = sequence.step(
        10,
        perception=snapshot(
            1,
            10,
            observation(1, 10, GroundPoint(500.0, 0.0)),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert first.state is MatchState.SEARCH_CLUSTER

    found = sequence.step(
        20,
        perception=snapshot(
            2,
            20,
            observation(2, 20, GroundPoint(500.0, 0.0)),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert found.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert found.reason == "green_path_clear_opportunistic_single:1"
    assert found.linear_velocity_m_s == 0.0
    assert found.angular_velocity_rad_s == 0.0
    assert found.gripper_posture is GripperPosture.TRANSPORT


def test_first_search_enters_common_cluster_search() -> None:
    sequence = make_sequence(
        config=runtime_config(
            opportunistic_single_green_enabled=True,
        )
    )
    ready = sequence.preflight(
        0,
        MatchPreflight(True, True, True, True, True, True),
    )
    assert ready.state is MatchState.PREFLIGHT
    sequence.start(1)
    sequence.state = MatchState.STARTUP_FORWARD_SETTLE
    sequence._settle_until_ns = 1

    started_search = sequence.step(
        2,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert started_search.state is MatchState.SEARCH_CLUSTER
    assert started_search.reason == "startup_forward_settled_search_cluster"
    assert started_search.angular_velocity_rad_s == pytest.approx(-0.30)


def test_first_search_does_not_break_black_only_group() -> None:
    sequence = make_sequence(
        config=runtime_config(opportunistic_single_green_enabled=True)
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER
    first = snapshot(
        1,
        10,
        observation(
            1,
            10,
            GroundPoint(600.0, -60.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=10.0,
        ),
        observation(
            1,
            10,
            GroundPoint(600.0, 60.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=30.0,
        ),
    )
    second = snapshot(
        2,
        20,
        observation(
            2,
            20,
            GroundPoint(600.0, -60.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=10.0,
        ),
        observation(
            2,
            20,
            GroundPoint(600.0, 60.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=30.0,
        ),
    )

    sequence.step(
        10,
        perception=first,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    found = sequence.step(
        20,
        perception=second,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert found.state is MatchState.SEARCH_CLUSTER
    assert found.reason == "search_cluster_right"


def test_green_cluster_preference_only_applies_to_required_first_delivery() -> None:
    sequence = make_sequence()
    green_point = GroundPoint(500.0, 0.0)
    sequence._tracker.update(10, [observation(1, 10, green_point)])
    sequence._tracker.update(20, [observation(2, 20, green_point)])

    assert sequence._preferred_cluster_ground_points(20) == frozenset(
        {green_point}
    )
    sequence._transport_count = 1
    assert sequence._preferred_cluster_ground_points(20) == frozenset()


def test_green_alignment_carries_fine_wheel_velocity_floor() -> None:
    sequence = make_sequence(
        config=runtime_config(opportunistic_single_green_enabled=False)
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    sequence._selected_track_id = 1

    decision = None
    for frame in range(1, 13):
        timestamp_ns = frame * 10
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(300.0, 100.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )

    assert decision is not None
    assert decision.reason == "align_green_relative_y_to_zero"
    assert decision.min_wheel_velocity_m_s == pytest.approx(0.01)


def _seed_green_alignment(sequence: MatchSequence) -> None:
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    sequence._selected_track_id = 1
    sequence._green_reference = GroundPoint(300.0, 0.0)
    sequence._green_reference_heading_rad = 0.0
    sequence._green_reference_distance_m = 0.15


def test_green_alignment_requires_stable_frames_before_forward() -> None:
    sequence = make_sequence(
        config=runtime_config(green_alignment_stable_frames=3)
    )
    start_sequence(sequence)
    _seed_green_alignment(sequence)

    decisions = []
    for frame in range(1, 4):
        timestamp_ns = frame + 1
        decisions.append(
            sequence.step(
                timestamp_ns,
                perception=snapshot(
                    frame,
                    timestamp_ns,
                    observation(frame, timestamp_ns, GroundPoint(300.0, 0.0)),
                ),
                heading_rad=0.0,
                cumulative_distance_m=0.0,
            )
        )

    assert [item.state for item in decisions[:2]] == [
        MatchState.TRANSPORT_ALIGN_GREEN,
        MatchState.TRANSPORT_ALIGN_GREEN,
    ]
    assert decisions[0].reason == "green_alignment_stabilizing"
    assert decisions[1].reason == "green_alignment_stabilizing"
    assert decisions[2].state is MatchState.TRANSPORT_APPROACH_GREEN


def test_green_alignment_timeout_restarts_search() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_alignment_stable_frames=3,
            green_alignment_timeout_ms=0.1,
        )
    )
    start_sequence(sequence)
    _seed_green_alignment(sequence)

    first = sequence.step(
        2,
        perception=snapshot(
            1,
            2,
            observation(1, 2, GroundPoint(300.0, 0.0)),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    timed_out = sequence.step(
        100_003,
        perception=snapshot(
            2,
            100_003,
            observation(2, 100_003, GroundPoint(300.0, 0.0)),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert first.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert timed_out.state is MatchState.SEARCH_CLUSTER
    assert timed_out.reason == "green_alignment_timeout_restart_search"
    assert timed_out.soft_brake is False


def test_far_opportunistic_green_realigns_at_near_standoff() -> None:
    sequence = make_sequence(
        config=runtime_config(
            opportunistic_single_green_enabled=True,
            opportunistic_single_green_clearance_mm=120.0,
            opportunistic_single_green_realign_standoff_mm=350.0,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(500.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN

    for frame in range(3, 13):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(500.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == pytest.approx(0.15)

    moving = sequence.step(
        130,
        perception=snapshot(13, 130, observation(13, 130, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert moving.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert moving.linear_velocity_m_s > 0.0
    assert moving.reason == "approach_green_to_near_standoff_before_realign"

    near_standoff = sequence.step(
        140,
        perception=snapshot(14, 140, observation(14, 140, GroundPoint(350.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert near_standoff.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert near_standoff.reason == "green_near_standoff_reached_start_realign"
    assert near_standoff.linear_velocity_m_s == 0.0
    assert sequence._green_realign_pending is False
    assert sequence._green_realign_done is True

    for frame in range(15, 25):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                # The second reference can still be slightly farther than the
                # configured standoff; it must not trigger a third alignment.
                observation(frame, frame * 10, GroundPoint(360.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.15,
        )

    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == pytest.approx(0.21)


def test_far_direct_green_realigns_at_near_standoff() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=150.0,
            opportunistic_single_green_enabled=True,
            opportunistic_single_green_realign_standoff_mm=350.0,
        )
    )
    start_sequence(sequence)

    first = sequence.step(
        10,
        perception=snapshot(1, 10, observation(1, 10, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert first.state is MatchState.SEARCH_CLUSTER
    sequence.state = MatchState.SEARCH_CLUSTER

    sequence._update_tracker(20, snapshot(2, 20, observation(2, 20, GroundPoint(500.0, 0.0))))
    # Exercise the direct handoff API; common search now uses grasp-priority entry.
    found = sequence._begin_green_transport(20, sequence._tracker.tracks[0])
    assert found.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert found.reason == "green_path_clear_align_direct:1"
    assert sequence._opportunistic_single_green is False

    for frame in range(3, 13):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(500.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == pytest.approx(0.15)

    moving = sequence.step(
        130,
        perception=snapshot(13, 130, observation(13, 130, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert moving.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert moving.reason == "approach_green_to_near_standoff_before_realign"

    near_standoff = sequence.step(
        140,
        perception=snapshot(14, 140, observation(14, 140, GroundPoint(350.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert near_standoff.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert near_standoff.reason == "green_near_standoff_reached_start_realign"
    assert sequence._green_realign_done is True

    for frame in range(15, 25):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(360.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.15,
        )

    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == pytest.approx(0.21)


def test_green_alignment_waits_before_forward_when_settle_enabled() -> None:
    sequence = make_sequence(
        config=runtime_config(
            action_settle_time_s=0.5,
            green_grab_offset_mm=150.0,
            opportunistic_single_green_enabled=True,
        )
    )
    start_sequence(sequence)

    sequence.step(
        10,
        perception=snapshot(1, 10, observation(1, 10, GroundPoint(300.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence.step(
        20,
        perception=snapshot(2, 20, observation(2, 20, GroundPoint(300.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    for frame in range(3, 13):
        timestamp_ns = 500_000_030 + (frame - 3) * 10
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(300.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert decision.linear_velocity_m_s == 0.0
    assert decision.reason == "green_aligned_y_zero_start_approach"

    waiting = sequence.step(
        600_000_120,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert waiting.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert waiting.linear_velocity_m_s == 0.0
    assert waiting.reason == "green_waiting_after_alignment"

    moving = sequence.step(
        1_100_000_120,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert moving.linear_velocity_m_s > 0.0
    assert moving.reason == "approach_green_x_minus_150_no_realign"


def test_safe_zone_waits_before_d1_line_forward_when_settle_enabled() -> None:
    sequence = make_sequence(
        config=runtime_config(action_settle_time_s=0.5),
        initial_field_position=FieldPoint(100.0, 500.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d1_line"
    d1_heading = math.atan2(137.0, -265.0)

    turn_to_d1 = sequence.step(
        10,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert turn_to_d1.reason == "safe_zone_turn_to_d1_line"

    d1_forward_start = sequence.step(
        20,
        perception=None,
        heading_rad=d1_heading,
        cumulative_distance_m=0.0,
    )
    assert d1_forward_start.state is MatchState.TRANSPORT_FORWARD
    assert d1_forward_start.reason == (
        "safe_zone_d1_line_heading_reached_start_forward"
    )

    waiting = sequence.step(
        100_000_020,
        perception=None,
        heading_rad=d1_heading,
        cumulative_distance_m=0.0,
    )
    assert waiting.state is MatchState.TRANSPORT_FORWARD
    assert waiting.linear_velocity_m_s == 0.0
    assert waiting.reason == "safe_zone_waiting_after_d1_line_alignment"

    moving = sequence.step(
        600_000_020,
        perception=None,
        heading_rad=d1_heading,
        cumulative_distance_m=0.0,
    )
    assert moving.linear_velocity_m_s > 0.0
    assert moving.reason == "safe_zone_forward_along_gripper_to_d1_line"


def test_green_approach_holds_aligned_heading() -> None:
    sequence = make_sequence(config=runtime_config(green_grab_offset_mm=150.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_approach_distance_m = 1.0
    sequence._green_reference = GroundPoint(500.0, 0.0)
    sequence._green_reference_heading_rad = 0.0

    decision = sequence.step(
        10,
        perception=None,
        heading_rad=0.1,
        cumulative_distance_m=0.0,
    )

    assert decision.linear_velocity_m_s == pytest.approx(0.08)
    assert decision.angular_velocity_rad_s < 0.0


@pytest.mark.parametrize(
    ("phase", "desired_heading", "expected_speed"),
    [
        ("forward_d1_line", math.pi / 2.0, 0.071),
        ("forward_d2_line", 0.3, 0.083),
        ("forward_final_closed", math.pi / 2.0, 0.097),
    ],
)
def test_safe_zone_forward_uses_segment_speeds(
    phase, desired_heading, expected_speed
) -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_fallback_heading_tolerance_rad=0.05,
            safe_zone_grab_to_d1_speed_m_s=0.071,
            safe_zone_d1_to_d2_speed_m_s=0.083,
            safe_zone_d2_to_final_speed_m_s=0.097,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_FORWARD
    sequence._safe_zone_phase = phase
    sequence._transport_forward_base_distance_m = 0.0
    sequence._transport_forward_distance_m = 1.0
    sequence._d1_line_heading_rad = (
        desired_heading if phase == "forward_d1_line" else None
    )
    sequence._d2_line_heading_rad = (
        desired_heading if phase == "forward_d2_line" else None
    )

    decision = sequence.step(
        10,
        perception=None,
        heading_rad=desired_heading + 0.1,
        cumulative_distance_m=0.0,
    )

    assert decision.linear_velocity_m_s == pytest.approx(expected_speed)
    assert decision.angular_velocity_rad_s < 0.0


def test_final_forward_applies_braking_overrun() -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_d2_to_final_braking_overrun_mm=25.0,
        ),
        initial_field_position=FieldPoint(-185.0, 990.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "stopping_after_d2_heading"

    waiting = sequence.step(
        10,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting.reason == "safe_zone_waiting_for_vehicle_stop_after_d2_heading"

    started = sequence.step(
        310_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert started.state is MatchState.TRANSPORT_RELEASE
    assert started.reason == "safe_zone_d2_heading_90_stopped_start_closing_gripper"

    started = sequence.step(
        1_310_000_011,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert started.state is MatchState.TRANSPORT_FORWARD
    assert sequence._safe_zone_final_target_y_mm() == pytest.approx(1112.0)
    assert sequence._transport_forward_distance_m == pytest.approx(0.122)


def test_orange_final_forward_uses_dedicated_speed_and_braking_overrun() -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_d2_to_final_speed_m_s=0.097,
            safe_zone_orange_d2_to_final_speed_m_s=0.041,
            safe_zone_d2_to_final_braking_overrun_mm=25.0,
            safe_zone_orange_d2_to_final_braking_overrun_mm=7.0,
            safe_zone_injured_target_field=FieldPoint(165.0, 1200.0),
            safe_zone_fallback_heading_tolerance_rad=0.05,
        ),
        initial_field_position=FieldPoint(165.0, 1100.0),
    )
    start_sequence(sequence)
    sequence._transport_target_classes = (TargetClass.ORANGE_INJURED,)
    sequence.state = MatchState.TRANSPORT_FORWARD
    sequence._safe_zone_phase = "forward_final_closed"
    sequence._transport_forward_base_distance_m = 0.0
    sequence._transport_forward_distance_m = 1.0

    assert sequence._safe_zone_d2_to_final_speed_m_s() == pytest.approx(0.041)
    assert sequence._safe_zone_final_target_y_mm() == pytest.approx(1193.0)
    assert sequence._safe_zone_d2_to_final_braking_overrun_mm() == pytest.approx(
        7.0
    )

    decision = sequence.step(
        10,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
    )

    assert decision.linear_velocity_m_s == pytest.approx(0.041)


def test_d2_opens_before_heading_when_settle_enabled() -> None:
    sequence = make_sequence(
        config=runtime_config(action_settle_time_s=0.5),
        initial_field_position=FieldPoint(-165.0, 700.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d2_line"

    line_start = sequence.step(
        10,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
    )
    assert line_start.state is MatchState.TRANSPORT_FORWARD

    reached = sequence.step(
        500_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.3,
    )
    assert reached.state is MatchState.TRANSPORT_RELEASE
    assert reached.reason == "safe_zone_d2_reached_wait_before_opening"
    assert reached.gripper_posture is GripperPosture.CLOSED

    waiting = sequence.step(
        600_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.3,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting.reason == "safe_zone_d2_waiting_for_vehicle_stop_before_opening"
    assert waiting.gripper_posture is GripperPosture.CLOSED

    stop_confirming = sequence.step(
        900_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.3,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert stop_confirming.reason == "safe_zone_d2_waiting_before_opening"

    opening = sequence.step(
        1_000_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.3,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opening.state is MatchState.TRANSPORT_RELEASE
    assert opening.reason == "safe_zone_d2_reached_start_opening"
    assert opening.gripper_posture is GripperPosture.OPEN

    opening_wait = sequence.step(
        1_500_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.3,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opening_wait.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert opening_wait.reason == "gripper_opened_at_d2_start_turn_to_90"
    assert opening_wait.gripper_posture is GripperPosture.OPEN

    heading_start = sequence.step(
        2_000_000_010,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.3,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert heading_start.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert heading_start.reason == "safe_zone_d2_heading_90_reached_stop_before_forward"
    assert heading_start.gripper_posture is GripperPosture.OPEN


def test_search_keeps_breakup_only_behavior_when_opportunity_disabled() -> None:
    sequence = make_sequence(
        config=runtime_config(opportunistic_single_green_enabled=False)
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(500.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "search_cluster_right"


def test_search_does_not_direct_grab_non_green_target() -> None:
    sequence = make_sequence(
        config=runtime_config(opportunistic_single_green_enabled=True)
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(
                    frame,
                    timestamp_ns,
                    GroundPoint(500.0, 0.0),
                    target_class=TargetClass.BLACK_CORE,
                ),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "search_cluster_right"


def test_search_prefers_safe_single_over_unnecessary_breakup() -> None:
    sequence = make_sequence(
        config=runtime_config(
            opportunistic_single_green_enabled=True,
            opportunistic_single_green_clearance_mm=120.0,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(
                    frame,
                    timestamp_ns,
                    GroundPoint(500.0, 0.0),
                ),
                observation(
                    frame,
                    timestamp_ns,
                    GroundPoint(500.0, 100.0),
                    target_class=TargetClass.BLACK_CORE,
                    box_x=30.0,
                ),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.selected_track_id == 1


def test_opportunistic_grab_rechecks_singleton_after_reference_collection() -> None:
    sequence = make_sequence(
        config=runtime_config(
            opportunistic_single_green_enabled=True,
            opportunistic_single_green_clearance_mm=120.0,
        )
    )
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(500.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN

    for frame in range(3, 12):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(500.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
        assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN

    recheck = sequence.step(
        120,
        perception=snapshot(
            12,
            120,
            observation(12, 120, GroundPoint(500.0, 0.0)),
            observation(
                12,
                120,
                GroundPoint(500.0, 100.0),
                target_class=TargetClass.BLACK_CORE,
                box_x=30.0,
            ),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert recheck.state is MatchState.SEARCH_CLUSTER
    assert recheck.reason == "green_reference_not_singleton_restart_breakup_search"


def test_lost_green_target_restarts_search_after_bounded_wait() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=150.0,
            green_align_hold_ms=500.0,
        )
    )
    start_sequence(sequence)
    sequence._selected_track_id = 1
    sequence._selected_green_ground = GroundPoint(500.0, 0.0)
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN

    waiting = sequence.step(
        10,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert waiting.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert waiting.linear_velocity_m_s == 0.0

    restarted = sequence.step(
        510_000_011,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert restarted.state is MatchState.SEARCH_CLUSTER
    assert restarted.reason == "green_target_timeout_restart_breakup_search"


def test_cluster_alignment_aims_at_actual_member() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER

    def cluster_snapshot(frame: int, timestamp_ns: int) -> PerceptionSnapshot:
        return snapshot(
            frame,
            timestamp_ns,
            observation(frame, timestamp_ns, GroundPoint(400.0, 100.0)),
            observation(
                frame,
                timestamp_ns,
                GroundPoint(500.0, -50.0),
                target_class=TargetClass.BLACK_CORE,
                box_x=30.0,
            ),
            observation(
                frame,
                timestamp_ns,
                GroundPoint(450.0, 0.0),
                target_class=TargetClass.ORANGE_INJURED,
                box_x=50.0,
            ),
        )

    sequence.step(
        10,
        perception=cluster_snapshot(1, 10),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence.step(
        20,
        perception=cluster_snapshot(2, 20),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    measurement = sequence._cluster_ground_measurement(20)
    assert measurement is not None
    assert measurement.center in (GroundPoint(400,100), GroundPoint(500,-50), GroundPoint(450,0))
    assert measurement.nearest_forward_x_mm == pytest.approx(math.hypot(measurement.center.x, measurement.center.y))
    assert sequence._breakup_proposal.aim_id in sequence._breakup_proposal.contact_ids


def test_all_blue_cluster_is_ignored_for_breakup() -> None:
    sequence = make_sequence()
    sequence._latest_heading_rad = 0.0
    observations = [
        observation(
            1,
            10,
            GroundPoint(400.0, -20.0),
            target_class=TargetClass.BLUE_DANGER,
            box_x=10.0,
        ),
        observation(
            1,
            10,
            GroundPoint(400.0, 20.0),
            target_class=TargetClass.BLUE_DANGER,
            box_x=30.0,
        ),
    ]
    sequence._tracker.update(10, observations)
    sequence._tracker.update(
        20,
        [
            replace(
                item,
                frame_sequence=2,
                capture_timestamp_ns=20,
                result_timestamp_ns=20,
            )
            for item in observations
        ],
    )

    assert sequence._cluster_ground_measurement(20) is None
    assert sequence._last_cluster_rejection_reason == "breakup_no_contact_plan_or_retry_exhausted"


def test_mixed_cluster_with_blue_is_still_available_for_breakup() -> None:
    sequence = make_sequence()
    sequence._latest_heading_rad = 0.0
    observations = [
        observation(1, 10, GroundPoint(400.0, -20.0)),
        observation(
            1,
            10,
            GroundPoint(400.0, 20.0),
            target_class=TargetClass.BLUE_DANGER,
            box_x=30.0,
        ),
    ]
    sequence._tracker.update(10, observations)
    sequence._tracker.update(
        20,
        [
            replace(
                item,
                frame_sequence=2,
                capture_timestamp_ns=20,
                result_timestamp_ns=20,
            )
            for item in observations
        ],
    )

    sequence._latest_perception = snapshot(2, 20, *(replace(item, frame_sequence=2,
        capture_timestamp_ns=20, result_timestamp_ns=20) for item in observations))
    assert sequence._cluster_ground_measurement(20) is not None


def test_cluster_reference_locks_real_contact_after_three_stopped_frames() -> None:
    sequence = make_sequence(config=runtime_config(opportunistic_single_green_enabled=False,
        cluster_align_hold_ms=3000, safe_zone_calibration_stop_confirm_time_s=0.01,
        breakup_backward_distance_m=0.4, breakup_confirmation_frames=3))
    start_sequence(sequence)
    sequence._breakup_only = True
    for frame in range(1, 10):
        now = frame * 20_000_000
        result = sequence.step(now, perception=snapshot(frame, now,
            observation(frame, now, GroundPoint(400, 0)),
            observation(frame, now, GroundPoint(450, 0), target_class=TargetClass.BLUE_DANGER, box_x=30)),
            heading_rad=0., cumulative_distance_m=0., left_speed_feedback_m_s=0., right_speed_feedback_m_s=0.)
        if result.reason == "breakup_plan_frozen":
            break
    assert result.state is MatchState.BREAKUP_FORWARD
    assert result.reason == "breakup_plan_frozen"
    assert len(sequence._breakup_reference_frames) == sequence.config.breakup_confirmation_frames
    locked = sequence._breakup_plan
    assert locked is not None and locked.aim in (GroundPoint(400,0), GroundPoint(450,0))
    assert sequence._breakup_plan is locked


def test_ignores_nearby_object_outside_forward_corridor() -> None:
    sequence = make_sequence(config=runtime_config(green_grab_offset_mm=150.0))
    start_sequence(sequence)
    green = observation(1, 10, GroundPoint(500.0, 0.0))
    nearby_but_behind_target = observation(
        1,
        10,
        GroundPoint(510.0, 0.0),
        target_class=TargetClass.BLACK_CORE,
        box_x=30.0,
    )
    sequence.step(
        10,
        perception=snapshot(1, 10, green, nearby_but_behind_target),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    found = sequence.step(
        20,
        perception=snapshot(
            2,
            20,
            observation(2, 20, GroundPoint(500.0, 0.0)),
            observation(
                2,
                20,
                GroundPoint(510.0, 0.0),
                target_class=TargetClass.BLACK_CORE,
                box_x=30.0,
            ),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert found.state is MatchState.TRANSPORT_ALIGN_GREEN


def test_relative_forward_corridor_rejects_blue_blocker_and_logs_details() -> None:
    sequence = make_sequence(config=runtime_config(
        green_path_half_width_mm=50.0,
        breakup_backward_distance_m=0.4,
    ))
    start_sequence(sequence)
    green_point = GroundPoint(500.0, 200.0)
    blue_point = GroundPoint(300.0, 120.0)
    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, green_point),
                observation(
                    frame,
                    timestamp_ns,
                    blue_point,
                    target_class=TargetClass.BLUE_DANGER,
                    box_x=30.0,
                ),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
        if frame == 1:
            sequence.state = MatchState.SEARCH_CLUSTER

    assert decision.state is MatchState.BREAKUP_SETTLE
    diagnostic = sequence.green_isolation_diagnostic(20)
    assert "class=blue_danger" in diagnostic
    assert "aligned_forward=" in diagnostic
    assert "aligned_lateral=" in diagnostic
    assert "blocked=true" in diagnostic
    assert "path=path_track_2" in diagnostic


def _corridor_decision(green_point: GroundPoint, blue_point: GroundPoint, **sequence_kwargs):
    """在搜索态用同一帧绿/蓝布局推进一步，返回该帧决策。"""

    sequence = make_sequence(
        config=runtime_config(
            breakup_backward_distance_m=0.4,
        ),
        **sequence_kwargs,
    )
    start_sequence(sequence)
    for frame, timestamp_ns in ((1, 10), (2, 20)):
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, green_point),
                observation(
                    frame,
                    timestamp_ns,
                    blue_point,
                    target_class=TargetClass.BLUE_DANGER,
                    box_x=30.0,
                ),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
        if frame == 1:
            sequence.state = MatchState.SEARCH_CLUSTER
    return decision


def test_path_blocking_uses_physical_contact_threshold_not_fixed_half_width() -> None:
    # 参考绿块 (500,200)，航向 0。蓝块横向偏置 40 mm 时两块实体仍可能接触，
    # 偏置 46 mm 时已经不在夹取路径上。
    green_point = GroundPoint(500.0, 200.0)
    on_path_blue = GroundPoint(263.6, 148.5)
    off_path_blue = GroundPoint(261.4, 154.1)
    near_config = NearFieldGraspConfig(
        clearance_mm=4.0,
        corridor_lateral_margin_mm=1.0,
    )
    # 内切半径 20+20 加实际夹爪余量 3，正是图中 43 mm 的门限；固定 50 mm
    # 会把 46 mm 的蓝块也算成阻挡并提前进入解团。
    assert 43.0 < 46.0 < 50.0

    blocked = _corridor_decision(
        green_point,
        on_path_blue,
        near_field_grasp_config=near_config,
    )
    assert blocked.state is MatchState.BREAKUP_SETTLE

    clear = _corridor_decision(
        green_point,
        off_path_blue,
        near_field_grasp_config=near_config,
    )
    assert clear.state is MatchState.TRANSPORT_ALIGN_GREEN


def test_path_blocking_falls_back_to_configured_half_width_without_geometry() -> None:
    sequence = make_sequence(
        config=runtime_config(green_path_half_width_mm=40.0),
        near_field_grasp_config=NearFieldGraspConfig(),
    )
    sequence._breakup_target_geometry = None
    threshold = sequence._path_block_half_width_mm(
        TargetClass.GREEN_SUPPLY,
        TargetClass.BLUE_DANGER,
    )
    assert threshold == pytest.approx(40.0)

    sequence._breakup_target_geometry = load_runtime_config(
        "configs/runtime.match.yaml"
    ).perception.target_ground_geometry
    # 有物理尺寸时按两块实体内切半径加近场夹爪余量计算，与固定半宽无关。
    assert sequence._path_block_half_width_mm(
        TargetClass.GREEN_SUPPLY,
        TargetClass.BLUE_DANGER,
    ) == pytest.approx(
        20.0 + 20.0 + 4.0 / 2.0 + 10.0
    )


def test_path_blocker_rejects_target_but_non_green_does_not_recheck() -> None:
    sequence = make_sequence(config=runtime_config(
        green_grab_offset_mm=150.0,
        breakup_backward_distance_m=0.4,
    ))
    start_sequence(sequence)
    for frame, timestamp_ns in ((1, 10), (2, 20)):
        blocker = observation(
            frame,
            timestamp_ns,
            GroundPoint(300.0, 20.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=30.0,
        )
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(500.0, 0.0)),
                blocker,
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
        if frame == 1:
            sequence.state = MatchState.SEARCH_CLUSTER
    assert decision.state is MatchState.BREAKUP_SETTLE

    # 另起一轮验证近目标的 x-150 定距，以及非绿色近目标不触发复核。
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=150.0,
            green_preclose_recheck_hold_ms=50.0,
        )
    )
    start_sequence(sequence)
    sequence.step(
        10,
        perception=snapshot(1, 10, observation(1, 10, GroundPoint(300.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    found = sequence.step(
        20,
        perception=snapshot(2, 20, observation(2, 20, GroundPoint(300.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert found.state is MatchState.TRANSPORT_ALIGN_GREEN
    aligned = None
    for frame in range(3, 13):
        aligned = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(300.0, 0.0)),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
    assert aligned is not None
    assert aligned.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == 0.15

    moving = sequence.step(
        135,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert moving.linear_velocity_m_s > 0.0
    assert moving.angular_velocity_rad_s == 0.0

    # 途中出现非绿色物块不会触发新的绿色预闭爪复核；到位后先进入
    # 停车复核窗口，窗口结束后才合爪。
    recheck = sequence.step(
        140,
        perception=snapshot(
            14,
            140,
            observation(14, 140, GroundPoint(300.0, 0.0)),
            observation(
                14,
                140,
                GroundPoint(100.0, 0.0),
                target_class=TargetClass.BLACK_CORE,
                box_x=30.0,
            ),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert recheck.state is MatchState.TRANSPORT_PRE_CLOSE_RECHECK
    assert recheck.gripper_posture is GripperPosture.TRANSPORT

    waiting = sequence.step(
        100_000_140,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert waiting.state is MatchState.TRANSPORT_PRE_CLOSE_RECHECK
    assert waiting.reason == "green_preclose_rechecking_neighbor"
    assert waiting.linear_velocity_m_s == 0.0
    assert waiting.angular_velocity_rad_s == 0.0

    close = sequence.step(
        200_000_140,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert close.state is MatchState.TRANSPORT_CLOSE_GRIPPER
    assert close.reason == "green_preclose_recheck_complete_close_gripper"
    assert close.gripper_posture is GripperPosture.CLOSED


def test_preclose_rechecks_next_green_and_then_closes() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=150.0,
            green_preclose_recheck_range_mm=200.0,
            green_preclose_recheck_hold_ms=50.0,
            green_preclose_max_carried_blocks=2,
            opportunistic_single_green_enabled=False,
        )
    )
    start_sequence(sequence)

    # Seed the already selected block and a second green block that will be
    # inside the 200 mm vehicle-origin range after the first approach.
    sequence._tracker.update(
        10,
        [
            observation(1, 10, GroundPoint(300.0, 0.0), box_x=10.0),
            observation(1, 10, GroundPoint(210.0, 0.0), box_x=30.0),
        ],
    )
    sequence._selected_track_id = 1
    sequence._selected_green_ground = GroundPoint(300.0, 0.0)
    sequence._green_preclose_carried_count = 1
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_approach_base_distance_m = 0.0
    sequence._green_approach_distance_m = 0.15
    sequence._green_reference = GroundPoint(300.0, 0.0)
    sequence._green_reference_heading_rad = 0.0

    started_recheck = sequence.step(
        20,
        perception=snapshot(
            2,
            20,
            observation(2, 20, GroundPoint(150.0, 0.0), box_x=10.0),
            observation(2, 20, GroundPoint(190.0, 0.0), box_x=30.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert started_recheck.state is MatchState.TRANSPORT_PRE_CLOSE_RECHECK
    assert started_recheck.reason == (
        "green_grab_offset_reached_start_preclose_recheck"
    )

    recheck = sequence.step(
        30,
        perception=snapshot(
            3,
            30,
            observation(3, 30, GroundPoint(150.0, 0.0), box_x=10.0),
            observation(3, 30, GroundPoint(190.0, 0.0), box_x=30.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert recheck.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert recheck.reason == "green_preclose_green_found_realign:2"
    assert recheck.selected_track_id == 2
    assert sequence._green_preclose_carried_count == 2
    assert sequence._green_preclose_consumed_track_ids == {1}

    # The second block is re-sampled and aligned before the final approach.
    for frame in range(3, 13):
        timestamp_ns = frame * 10
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame,
                timestamp_ns,
                observation(frame, timestamp_ns, GroundPoint(190.0, 0.0), box_x=30.0),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.15,
        )
    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == pytest.approx(0.04)

    moving = sequence.step(
        130,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.15,
    )
    assert moving.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert moving.linear_velocity_m_s > 0.0

    closed = sequence.step(
        140,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.19,
    )
    assert closed.state is MatchState.TRANSPORT_CLOSE_GRIPPER
    assert closed.gripper_posture is GripperPosture.CLOSED


def test_preclose_recheck_waits_for_new_neighbor_confirmation() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=175.0,
            green_preclose_recheck_range_mm=240.0,
            green_preclose_recheck_hold_ms=500.0,
            green_preclose_max_carried_blocks=2,
        )
    )
    start_sequence(sequence)

    # The first block is already confirmed.  The second block is only exposed
    # after the vehicle reaches the grab offset, so its first post-stop frame
    # must not be enough to trigger a realign.
    sequence._tracker.update(
        10,
        [observation(1, 10, GroundPoint(300.0, 0.0), box_x=10.0)],
    )
    sequence._tracker.update(
        20,
        [observation(2, 20, GroundPoint(300.0, 0.0), box_x=10.0)],
    )
    sequence._selected_track_id = 1
    sequence._selected_green_ground = GroundPoint(300.0, 0.0)
    sequence._green_preclose_carried_count = 1
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_approach_base_distance_m = 0.0
    sequence._green_approach_distance_m = 0.125
    sequence._green_reference = GroundPoint(300.0, 0.0)
    sequence._green_reference_heading_rad = 0.0

    started = sequence.step(
        30,
        perception=snapshot(
            3,
            30,
            observation(3, 30, GroundPoint(175.0, 0.0), box_x=10.0),
            # This is outside the old forward-only interpretation, but inside
            # the vehicle-origin recheck radius and intentionally lies to the
            # side of the vehicle.
            observation(3, 30, GroundPoint(0.0, 220.0), box_x=30.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.125,
    )
    assert started.state is MatchState.TRANSPORT_PRE_CLOSE_RECHECK
    assert started.gripper_posture is GripperPosture.TRANSPORT
    assert started.linear_velocity_m_s == 0.0
    assert started.angular_velocity_rad_s == 0.0

    confirmed = sequence.step(
        40,
        perception=snapshot(
            4,
            40,
            observation(4, 40, GroundPoint(175.0, 0.0), box_x=10.0),
            observation(4, 40, GroundPoint(0.0, 220.0), box_x=30.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.125,
    )
    assert confirmed.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert confirmed.reason == "green_preclose_green_found_realign:2"
    assert confirmed.selected_track_id == 2
    assert sequence._green_preclose_carried_count == 2

    for frame in range(5, 15):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(0.0, 220.0), box_x=30.0),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.125,
        )
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN

    turned = sequence.step(
        150,
        perception=snapshot(
            15,
            150,
            observation(15, 150, GroundPoint(0.0, 220.0), box_x=30.0),
        ),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.125,
    )
    assert turned.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert sequence._green_approach_distance_m == pytest.approx(0.045)


def test_preclose_recheck_respects_max_carried_blocks() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_preclose_recheck_range_mm=200.0,
            green_preclose_max_carried_blocks=2,
        )
    )
    start_sequence(sequence)
    for timestamp_ns in (10, 20):
        sequence.step(
            timestamp_ns,
            perception=snapshot(
                timestamp_ns // 10,
                timestamp_ns,
                observation(timestamp_ns // 10, timestamp_ns, GroundPoint(150.0, 0.0)),
                observation(
                    timestamp_ns // 10,
                    timestamp_ns,
                    GroundPoint(100.0, 0.0),
                    box_x=30.0,
                ),
            ),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )
    sequence._selected_track_id = 1
    sequence._green_preclose_carried_count = 2
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_approach_base_distance_m = 0.0
    sequence._green_approach_distance_m = 0.0

    decision = sequence.step(
        30,
        perception=snapshot(
            3,
            30,
            observation(3, 30, GroundPoint(150.0, 0.0)),
            observation(3, 30, GroundPoint(100.0, 0.0), box_x=30.0),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )

    assert decision.state is MatchState.TRANSPORT_CLOSE_GRIPPER
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert sequence.selected_track_id == 1


def test_d1_visual_calibration_uses_compensated_straight_line() -> None:
    sequence = make_sequence(
        config=runtime_config(
            green_grab_offset_mm=150.0,
            safe_zone_calibration_start_offset_mm=500.0,
            safe_zone_open_offset_mm=137.0,
            safe_zone_d2_braking_overrun_x_mm=20.0,
            safe_zone_d2_braking_overrun_y_mm=10.0,
            safe_zone_grab_to_d1_speed_m_s=0.15,
            safe_zone_d1_to_d2_speed_m_s=0.15,
            safe_zone_d2_to_final_speed_m_s=0.15,
            transport_align_tolerance_mm=1.0,
            safe_zone_fallback_heading_tolerance_rad=0.05,
        ),
        initial_field_position=FieldPoint(100.0, 500.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_CLOSE_GRIPPER
    sequence._gripper_phase_started_ns = 1

    started_route = sequence.step(
        1_000_000_002,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert started_route.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert started_route.reason == "gripper_closed_start_safe_zone_d1_line"

    d1_target = FieldPoint(-185.0, 627.0)
    d1_delta_x = d1_target.x - 100.0
    d1_delta_y = d1_target.y - 500.0
    d1_line_heading = math.atan2(d1_delta_y, d1_delta_x)
    d1_line_distance_m = math.hypot(d1_delta_x, d1_delta_y) / 1000.0

    turn_to_d1_line = sequence.step(
        1_000_000_003,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert turn_to_d1_line.reason == "safe_zone_turn_to_d1_line"
    assert turn_to_d1_line.angular_velocity_rad_s > 0.0

    d1_line_start = sequence.step(
        1_000_000_004,
        perception=None,
        heading_rad=d1_line_heading,
        cumulative_distance_m=0.0,
    )
    assert d1_line_start.state is MatchState.TRANSPORT_FORWARD
    assert d1_line_start.reason == (
        "safe_zone_d1_line_heading_reached_start_forward"
    )
    assert sequence._transport_forward_distance_m == pytest.approx(
        d1_line_distance_m
    )

    d1_line_mid = sequence.step(
        1_000_000_005,
        perception=None,
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m / 2.0,
        left_speed_feedback_m_s=0.15,
        right_speed_feedback_m_s=0.15,
    )
    assert d1_line_mid.reason == "safe_zone_forward_along_gripper_to_d1_line"
    assert d1_line_mid.linear_velocity_m_s == pytest.approx(0.15)
    assert d1_line_mid.angular_velocity_rad_s == 0.0

    calibration_start = sequence.step(
        1_000_000_006,
        perception=None,
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert calibration_start.state is MatchState.TRANSPORT_RELEASE
    assert calibration_start.reason == (
        "safe_zone_d1_coordinate_threshold_reached_stop_before_calibration"
    )
    assert calibration_start.gripper_posture is GripperPosture.CLOSED

    still_moving = sequence.step(
        2_000_000_000,
        perception=None,
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.15,
        right_speed_feedback_m_s=0.15,
    )
    assert still_moving.reason == "safe_zone_waiting_for_vehicle_stop_before_calibration"

    stopped_waiting = sequence.step(
        2_000_000_001,
        perception=None,
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert stopped_waiting.reason == "safe_zone_waiting_for_vehicle_stop_before_calibration"

    calibrated_pose = FieldPose2D(FieldPoint(-120.0, 700.0), math.radians(88.0))
    keypoints_seen = sequence.step(
        2_300_000_002,
        perception=safe_zone_snapshot_for_pose(
            1,
            2_300_000_002,
            calibrated_pose,
        ),
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert keypoints_seen.reason == "safe_zone_keypoints_seen_stop_before_calibration"

    keypoints_stop_waiting = sequence.step(
        2_300_000_003,
        perception=safe_zone_snapshot_for_pose(
            2,
            2_300_000_003,
            calibrated_pose,
        ),
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert keypoints_stop_waiting.reason == (
        "safe_zone_waiting_for_vehicle_stop_after_keypoints"
    )

    stopped = sequence.step(
        2_600_000_004,
        perception=safe_zone_snapshot_for_pose(
            3,
            2_600_000_004,
            calibrated_pose,
        ),
        heading_rad=d1_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert stopped.reason == "safe_zone_keypoints_stopped_start_calibration"

    calibration_result = None
    for frame in range(4, 9):
        calibration_result = sequence.step(
            2_600_000_004 + frame,
            perception=safe_zone_snapshot_for_pose(
                frame,
                2_600_000_004 + frame,
                calibrated_pose,
            ),
            heading_rad=d1_line_heading,
            cumulative_distance_m=d1_line_distance_m,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
    assert calibration_result is not None
    assert calibration_result.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert calibration_result.reason == "safe_zone_visual_calibrated_start_d2_line"
    assert calibration_result.gripper_posture is GripperPosture.CLOSED
    assert sequence.safe_zone_calibration_pose is not None
    assert sequence.safe_zone_calibration_pose.position.x == pytest.approx(-120.0)
    assert sequence.safe_zone_calibration_pose.position.y == pytest.approx(700.0)
    assert sequence.safe_zone_calibration_pose.heading_rad == pytest.approx(
        math.radians(88.0)
    )
    assert sequence._fallback_field_position is not None
    assert sequence._fallback_field_position.x == pytest.approx(-120.0)
    assert sequence._fallback_field_position.y == pytest.approx(700.0)

    d2_target = FieldPoint(-185.0, 990.0)
    d2_delta_x = d2_target.x - calibrated_pose.position.x
    d2_delta_y = d2_target.y - calibrated_pose.position.y
    d2_line_heading = math.atan2(d2_delta_y, d2_delta_x)
    d2_line_distance_m = math.hypot(d2_delta_x, d2_delta_y) / 1000.0
    raw_d2_line_heading = d2_line_heading - sequence._heading_offset_rad
    raw_heading_90 = math.pi / 2.0 - sequence._heading_offset_rad

    turn_to_d2_line = sequence.step(
        2_600_000_020,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert turn_to_d2_line.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert turn_to_d2_line.reason == "safe_zone_turn_to_d2_line"
    assert turn_to_d2_line.angular_velocity_rad_s > 0.0
    assert turn_to_d2_line.gripper_posture is GripperPosture.CLOSED

    d2_line_start = sequence.step(
        2_600_000_021,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert d2_line_start.state is MatchState.TRANSPORT_FORWARD
    assert d2_line_start.reason == "safe_zone_d2_line_heading_reached_start_forward"
    assert sequence._transport_forward_distance_m == pytest.approx(d2_line_distance_m)
    assert d2_line_start.gripper_posture is GripperPosture.CLOSED

    d2_line_mid = sequence.step(
        2_600_000_022,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m / 2.0,
        left_speed_feedback_m_s=0.15,
        right_speed_feedback_m_s=0.15,
    )
    assert d2_line_mid.reason == "safe_zone_forward_along_d1_d2_line"
    assert d2_line_mid.linear_velocity_m_s == pytest.approx(0.15)
    assert d2_line_mid.angular_velocity_rad_s == 0.0

    d2_reached = sequence.step(
        2_600_000_023,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert d2_reached.state is MatchState.TRANSPORT_RELEASE
    assert d2_reached.reason == "safe_zone_d2_coordinate_threshold_reached_wait_before_opening"
    assert d2_reached.gripper_posture is GripperPosture.CLOSED
    assert sequence._fallback_field_position is not None
    assert sequence._fallback_field_position.x == pytest.approx(-185.0, abs=0.2)
    assert sequence._fallback_field_position.y == pytest.approx(990.0, abs=0.2)

    d2_stop_wait = sequence.step(
        3_700_000_023,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert d2_stop_wait.state is MatchState.TRANSPORT_RELEASE
    assert d2_stop_wait.reason == "safe_zone_d2_waiting_for_vehicle_stop_before_opening"
    assert d2_stop_wait.gripper_posture is GripperPosture.CLOSED

    d2_opening = sequence.step(
        4_000_000_023,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert d2_opening.state is MatchState.TRANSPORT_RELEASE
    assert d2_opening.reason == "safe_zone_d2_reached_start_opening"
    assert d2_opening.gripper_posture is GripperPosture.OPEN

    d2_opening_wait = sequence.step(
        4_500_000_023,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert d2_opening_wait.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert d2_opening_wait.reason == "gripper_opened_at_d2_start_turn_to_90"
    assert d2_opening_wait.gripper_posture is GripperPosture.OPEN

    opening_complete = sequence.step(
        5_000_000_023,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opening_complete.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert opening_complete.reason == "safe_zone_d2_turn_to_90"
    assert opening_complete.gripper_posture is GripperPosture.OPEN

    turn_to_90 = sequence.step(
        5_000_000_024,
        perception=None,
        heading_rad=raw_d2_line_heading,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert turn_to_90.reason == "safe_zone_d2_turn_to_90"
    assert turn_to_90.angular_velocity_rad_s < 0.0
    assert turn_to_90.gripper_posture is GripperPosture.OPEN

    heading_90_reached = sequence.step(
        5_000_000_025,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert heading_90_reached.reason == "safe_zone_d2_heading_90_reached_stop_before_forward"
    assert heading_90_reached.gripper_posture is GripperPosture.OPEN

    heading_stop_wait = sequence.step(
        5_000_000_026,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert heading_stop_wait.reason == "safe_zone_waiting_for_vehicle_stop_after_d2_heading"
    assert heading_stop_wait.gripper_posture is GripperPosture.OPEN

    heading_stopped = sequence.step(
        5_300_000_026,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert heading_stopped.state is MatchState.TRANSPORT_RELEASE
    assert heading_stopped.reason == "safe_zone_d2_heading_90_stopped_start_closing_gripper"
    assert sequence._transport_forward_distance_m == pytest.approx(0.147)
    assert heading_stopped.gripper_posture is GripperPosture.CLOSED

    closed = sequence.step(
        6_300_000_026,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert closed.state is MatchState.TRANSPORT_FORWARD
    assert closed.reason == "gripper_closed_after_d2_heading_start_forward_settle"
    assert closed.gripper_posture is GripperPosture.CLOSED

    finished = sequence.step(
        6_300_000_027,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m + 0.147,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert finished.state is MatchState.TRANSPORT_RELEASE
    assert finished.reason == "safe_zone_reached_transport_endpoint_wait_before_opening"
    assert finished.gripper_posture is GripperPosture.CLOSED

    stopped_before_opening = sequence.step(
        6_300_000_028,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m + 0.147,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert stopped_before_opening.reason == "safe_zone_waiting_for_vehicle_stop_before_opening"
    assert stopped_before_opening.gripper_posture is GripperPosture.CLOSED

    opening = sequence.step(
        6_600_000_028,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m + 0.147,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opening.state is MatchState.TRANSPORT_RELEASE
    assert opening.reason == "safe_zone_transport_stopped_start_opening"
    assert opening.gripper_posture is GripperPosture.OPEN

    opened = sequence.step(
        7_600_000_029,
        perception=None,
        heading_rad=raw_heading_90,
        cumulative_distance_m=d1_line_distance_m + d2_line_distance_m + 0.147,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opened.state is MatchState.RETURN_BACKUP
    assert opened.reason == "gripper_opened_after_safe_zone_push_start_exit"
    assert opened.gripper_posture is GripperPosture.OPEN
    assert sequence._fallback_field_position is not None
    assert sequence._fallback_field_position.x == pytest.approx(-185.0, abs=0.2)
    assert sequence._fallback_field_position.y == pytest.approx(1137.0, abs=0.2)
    assert sequence.estimated_field_heading_rad == pytest.approx(math.radians(90.0))


def test_d2_line_requires_both_coordinate_thresholds() -> None:
    sequence = make_sequence(
        config=runtime_config(transport_align_tolerance_mm=30.0),
        initial_field_position=FieldPoint(-120.0, 700.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d2_line"

    target = FieldPoint(-165.0, 1000.0)
    heading = math.atan2(target.y - 700.0, target.x + 120.0)
    line_start = sequence.step(
        10,
        perception=None,
        heading_rad=heading,
        cumulative_distance_m=0.0,
    )
    assert line_start.state is MatchState.TRANSPORT_FORWARD

    # x 先进入 ±30 mm 阈值，但 y 仍明显未到 d2；不能提前结束 d2 直线段。
    travel_to_x_threshold_m = (15.1 / abs(math.cos(heading))) / 1000.0
    still_forward = sequence.step(
        20,
        perception=None,
        heading_rad=heading,
        cumulative_distance_m=travel_to_x_threshold_m,
    )
    assert still_forward.state is MatchState.TRANSPORT_FORWARD
    assert still_forward.reason == "safe_zone_forward_along_d1_d2_line"
    assert sequence._fallback_field_position is not None
    assert abs(sequence._fallback_field_position.x - target.x) <= 30.0
    assert abs(sequence._fallback_field_position.y - target.y) > 30.0

    d2_reached = sequence.step(
        30,
        perception=None,
        heading_rad=heading,
        cumulative_distance_m=sequence._transport_forward_distance_m,
    )
    assert d2_reached.state is MatchState.TRANSPORT_RELEASE
    assert d2_reached.reason == "safe_zone_d2_coordinate_threshold_reached_wait_before_opening"
    assert sequence._fallback_field_position is not None
    assert sequence._fallback_field_position.x == pytest.approx(target.x, abs=0.2)
    assert sequence._fallback_field_position.y == pytest.approx(target.y, abs=0.2)


def test_d1_line_stops_when_target_is_crossed() -> None:
    sequence = make_sequence(
        config=runtime_config(
            safe_zone_fallback_target_field=FieldPoint(-165.0, 800.0),
            safe_zone_calibration_start_offset_mm=500.0,
            safe_zone_d2_braking_overrun_x_mm=0.0,
            safe_zone_d2_braking_overrun_y_mm=0.0,
            transport_align_tolerance_mm=30.0,
        ),
        initial_field_position=FieldPoint(-165.0, 0.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d1_line"

    line_start = sequence.step(
        10,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
    )
    assert line_start.state is MatchState.TRANSPORT_FORWARD

    crossed = sequence.step(
        20,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.36,
    )
    assert crossed.state is MatchState.TRANSPORT_RELEASE
    assert crossed.reason == (
        "safe_zone_d1_coordinate_threshold_reached_stop_before_calibration"
    )
    assert crossed.linear_velocity_m_s == 0.0


def test_d2_line_stops_when_target_is_crossed() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(-165.0, 700.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    sequence._safe_zone_phase = "align_d2_line"

    line_start = sequence.step(
        10,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
    )
    assert line_start.state is MatchState.TRANSPORT_FORWARD

    crossed = sequence.step(
        20,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.36,
    )
    assert crossed.state is MatchState.TRANSPORT_RELEASE
    assert crossed.reason == "safe_zone_d2_coordinate_threshold_reached_wait_before_opening"
    assert crossed.linear_velocity_m_s == 0.0


def test_exits_safe_zone_before_rearming_cluster_search() -> None:
    sequence = make_sequence(
        config=runtime_config(
            required_transports=4,
            opportunistic_single_green_enabled=True,
            safe_zone_exit_distance_m=0.30,
            safe_zone_calibration_stop_confirm_time_s=0.30,
        ),
        initial_field_position=FieldPoint(-165.0, 1100.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_FORWARD
    sequence._safe_zone_phase = "forward_final_closed"
    sequence._transport_forward_base_distance_m = 0.0
    sequence._transport_forward_distance_m = 0.1

    def safe_zone_targets(frame: int, timestamp_ns: int) -> PerceptionSnapshot:
        return snapshot(
            frame,
            timestamp_ns,
            observation(frame, timestamp_ns, GroundPoint(235.0, 35.0)),
            observation(frame, timestamp_ns, GroundPoint(236.0, 36.0)),
        )

    finished_route = sequence.step(
        10,
        perception=safe_zone_targets(1, 10),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert finished_route.state is MatchState.TRANSPORT_RELEASE
    assert finished_route.reason == "safe_zone_reached_transport_endpoint_wait_before_opening"
    assert finished_route.gripper_posture is GripperPosture.CLOSED

    waiting_before_exit = sequence.step(
        20,
        perception=safe_zone_targets(2, 20),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert waiting_before_exit.reason == "safe_zone_waiting_for_vehicle_stop_before_opening"
    assert waiting_before_exit.gripper_posture is GripperPosture.CLOSED

    opening_started = sequence.step(
        320_000_020,
        perception=safe_zone_targets(3, 320_000_020),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opening_started.state is MatchState.TRANSPORT_RELEASE
    assert opening_started.reason == "safe_zone_transport_stopped_start_opening"
    assert opening_started.gripper_posture is GripperPosture.OPEN

    opening_wait = sequence.step(
        320_000_021,
        perception=safe_zone_targets(4, 320_000_021),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opening_wait.state is MatchState.RETURN_BACKUP
    assert opening_wait.reason == "gripper_opened_after_safe_zone_push_start_exit"

    opened = sequence.step(
        1_320_000_021,
        perception=safe_zone_targets(5, 1_320_000_021),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert opened.state is MatchState.RETURN_BACKUP
    assert opened.reason == "safe_zone_vehicle_stopped_start_exit"
    assert opened.gripper_posture is GripperPosture.OPEN

    exit_started = sequence.step(
        1_620_000_021,
        perception=safe_zone_targets(6, 1_620_000_021),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert exit_started.reason == "safe_zone_exit_reverse_to_field"

    reversing = sequence.step(
        1_620_000_022,
        perception=safe_zone_targets(7, 1_620_000_022),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.09,
        left_speed_feedback_m_s=-0.08,
        right_speed_feedback_m_s=-0.08,
    )
    assert reversing.state is MatchState.RETURN_BACKUP
    assert reversing.reason == "safe_zone_exit_reverse_to_field"

    exit_distance_reached = sequence.step(
        1_620_000_023,
        perception=safe_zone_targets(8, 1_620_000_023),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=-0.08,
        right_speed_feedback_m_s=-0.08,
    )
    assert exit_distance_reached.reason == "safe_zone_exit_distance_reached_wait_for_stop"

    exit_stop_waiting = sequence.step(
        1_620_000_024,
        perception=safe_zone_targets(9, 1_620_000_024),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert exit_stop_waiting.reason == "safe_zone_waiting_for_vehicle_stop_after_exit"

    exit_stopped = sequence.step(
        1_920_000_025,
        perception=safe_zone_targets(10, 1_920_000_025),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert exit_stopped.reason == "safe_zone_exit_stopped_start_visual_calibration"
    assert exit_stopped.state is MatchState.TRANSPORT_RELEASE
    assert exit_stopped.gripper_posture is GripperPosture.OPEN
    assert sequence._tracker.tracks != ()

    corrected_pose = FieldPose2D(FieldPoint(-165.0, 800.0), math.pi / 2.0)
    keypoints_seen = sequence.step(
        1_920_000_026,
        perception=safe_zone_snapshot_for_pose(11, 1_920_000_026, corrected_pose),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert keypoints_seen.reason == "safe_zone_keypoints_seen_stop_before_calibration"
    assert keypoints_seen.gripper_posture is GripperPosture.OPEN

    calibration_stop_waiting = sequence.step(
        1_920_000_027,
        perception=safe_zone_snapshot_for_pose(12, 1_920_000_027, corrected_pose),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert calibration_stop_waiting.reason == "safe_zone_waiting_for_vehicle_stop_after_keypoints"

    calibration_started = sequence.step(
        2_220_000_028,
        perception=safe_zone_snapshot_for_pose(13, 2_220_000_028, corrected_pose),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert calibration_started.reason == "safe_zone_keypoints_stopped_start_calibration"

    calibrated = None
    for frame in range(14, 19):
        calibrated = sequence.step(
            2_220_000_015 + frame,
            perception=safe_zone_snapshot_for_pose(
                frame,
                2_220_000_015 + frame,
                corrected_pose,
            ),
            heading_rad=math.pi / 2.0,
            cumulative_distance_m=-0.20,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
    assert calibrated is not None
    assert calibrated.reason == "safe_zone_exit_visual_calibrated_waiting_for_new_perception"
    assert calibrated.state is MatchState.RETURN_BACKUP
    assert calibrated.gripper_posture is GripperPosture.OPEN
    assert sequence._tracker.tracks == ()

    same_frame = sequence.step(
        2_220_000_034,
        perception=safe_zone_snapshot_for_pose(18, 2_220_000_033, corrected_pose),
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert same_frame.state is MatchState.RETURN_BACKUP
    assert same_frame.reason == "safe_zone_exit_waiting_for_new_perception"

    fresh_frame = snapshot(19, 2_220_000_035)
    search_started = sequence.step(
        2_220_000_035,
        perception=fresh_frame,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert search_started.state is MatchState.SEARCH_CLUSTER
    assert search_started.reason == "safe_zone_exit_complete_start_search"

    search = sequence.step(
        2_220_000_036,
        perception=fresh_frame,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=-0.20,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert search.state is MatchState.SEARCH_CLUSTER
    assert search.reason == "search_cluster_right"
    assert search.angular_velocity_rad_s == pytest.approx(
        sequence.config.cluster_search_empty_angular_velocity_rad_s
    )
    assert sequence._tracker.tracks == ()


@pytest.mark.parametrize("team,heading", [(TeamColor.RED, math.pi / 2), (TeamColor.BLUE, -math.pi / 2)])
@pytest.mark.parametrize("phase", [MatchState.APPROACH_CLUSTER,
                                   MatchState.BREAKUP_FORWARD,
                                   MatchState.BREAKUP_BACKWARD,
                                   MatchState.RELOCATE_FORWARD])
def test_breakup_blocks_entire_route_to_own_safe_zone(team, heading, phase) -> None:
    # 合成航位只验证路径门禁，不代表实车定位精度。
    sequence = make_sequence(config=runtime_config(
        breakup_forward_distance_m=3.0, breakup_backward_distance_m=3.0,
        cluster_relocate_distance_m=3.0,
    ), initial_field_position=FieldPoint(0.0, 0.0))
    sequence._breakup_static_map = load_runtime_config("configs/runtime.match.yaml").world.static_map
    sequence._team_color = team
    start_sequence(sequence)
    sequence.state = phase
    sequence._cluster_reference_distance_mm = 1000.0
    if phase is MatchState.BREAKUP_BACKWARD:
        heading = normalize_angle(heading + math.pi)
    inject_breakup_plan(sequence, heading=heading)
    decision = sequence.step(10, perception=None, heading_rad=heading, cumulative_distance_m=0.0)
    # 路线终点已在安全区外，仍必须识别中途穿越；后退按负位移检查。
    assert decision.reason == "target_path_intersects_safe_zone_stop_and_reselect"
    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.linear_velocity_m_s == decision.angular_velocity_rad_s == 0.0
    stopped = sequence.step(20, perception=None, heading_rad=0.0, cumulative_distance_m=0.0)
    assert stopped.reason == "safe_zone_reselect_waiting_for_stop"
    assert stopped.linear_velocity_m_s == 0.0


@pytest.mark.parametrize("missing", ["map", "position", "heading"])
def test_breakup_missing_guard_inputs_holds(missing) -> None:
    sequence = make_sequence(initial_field_position=None if missing == "position" else FieldPoint(0.0, 0.0))
    sequence._breakup_static_map = None if missing == "map" else load_runtime_config("configs/runtime.match.yaml").world.static_map
    start_sequence(sequence)
    sequence.state = MatchState.BREAKUP_FORWARD
    inject_breakup_plan(sequence)
    decision = sequence.step(10, perception=None, heading_rad=None if missing == "heading" else 0.0,
                             cumulative_distance_m=0.0)
    assert decision.reason == "breakup_safe_zone_guard_missing_pose_or_map"
    assert decision.linear_velocity_m_s == 0.0


def test_breakup_opponent_zone_also_stops_and_reselects() -> None:
    sequence = make_sequence(config=runtime_config(breakup_forward_distance_m=3.0),
                             initial_field_position=FieldPoint(0.0, 0.0))
    sequence._breakup_static_map = load_runtime_config("configs/runtime.match.yaml").world.static_map
    start_sequence(sequence)
    sequence.state = MatchState.BREAKUP_FORWARD
    inject_breakup_plan(sequence)
    decision = sequence.step(10, perception=None, heading_rad=-math.pi / 2, cumulative_distance_m=0.0)
    assert decision.linear_velocity_m_s == 0.0
    assert decision.state is MatchState.SEARCH_CLUSTER


def test_breakup_clearance_and_tangent_are_blocked() -> None:
    static_map = load_runtime_config("configs/runtime.match.yaml").world.static_map
    polygon = static_map.safe_zone_polygon_field(TeamColor.RED)
    assert polygon is not None
    sequence = make_sequence(config=runtime_config(breakup_forward_distance_m=3.0),
                             initial_field_position=FieldPoint(max(p.x for p in polygon) + 240.0, 0.0))
    sequence._breakup_static_map = static_map
    start_sequence(sequence)
    sequence.state = MatchState.BREAKUP_FORWARD
    inject_breakup_plan(sequence)
    decision = sequence.step(10, perception=None, heading_rad=math.pi / 2, cumulative_distance_m=0.0)
    assert decision.reason == "target_path_intersects_safe_zone_stop_and_reselect"


def test_breakup_guard_does_not_block_authorized_delivery() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 1100.0))
    sequence._breakup_static_map = load_runtime_config("configs/runtime.match.yaml").world.static_map
    sequence._latest_heading_rad = math.pi / 2
    sequence.state = MatchState.TRANSPORT_FORWARD
    decision = sequence._decision(10, 0.15, 0.0, "delivery")
    assert sequence._guard_breakup_motion(decision, 0.0) is decision


def test_factory_wires_configured_guard_geometry() -> None:
    config = load_runtime_config("configs/runtime.match.yaml")
    sequence = MatchSequence.from_app_config(config)
    assert sequence._near_field_pickup is not None
    assert sequence._near_field_pickup.commit_age_ns == 150_000_000
    assert config.near_field_grasp.confirmation_frames == 1
    assert sequence._near_field_pickup.fine_alignment_zone_rad == pytest.approx(
        0.08
    )
    assert sequence._near_field_pickup.fine_alignment_min_wheel_velocity_m_s == 0.0
    assert sequence._breakup_static_map is config.world.static_map
    assert sequence._breakup_clearance_mm == pytest.approx(
        config.match.robot_footprint_radius_mm
        + config.match.safety_margin_mm
    )
    assert config.world.static_map.safe_zone_polygon_field(config.world.team_color) is not None


def test_approach_checks_following_push_before_moving() -> None:
    sequence = make_sequence(config=runtime_config(breakup_forward_distance_m=0.7),
                             initial_field_position=FieldPoint(0.0, 500.0))
    sequence._breakup_static_map = load_runtime_config(
        "configs/runtime.match.yaml"
    ).world.static_map
    start_sequence(sequence)
    sequence.state = MatchState.APPROACH_CLUSTER
    # 接近本身仅 100 mm 安全，但随后前推将越界。
    sequence._cluster_reference_distance_mm = sequence.config.cluster_breakup_standoff_mm + 100.0
    sequence._cluster_reference_field_point = FieldPoint(0.0, 500.0)
    inject_breakup_plan(sequence)
    decision = sequence.step(10, perception=None, heading_rad=math.pi / 2, cumulative_distance_m=0.0)
    assert decision.reason == "target_path_intersects_safe_zone_stop_and_reselect"
    assert decision.linear_velocity_m_s == 0.0


def test_breakup_safe_parallel_route_continues() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    sequence._breakup_static_map = load_runtime_config(
        "configs/runtime.match.yaml"
    ).world.static_map
    start_sequence(sequence)
    sequence.state = MatchState.BREAKUP_FORWARD
    inject_breakup_plan(sequence)
    decision = sequence.step(10, perception=None, heading_rad=0.0, cumulative_distance_m=0.0)
    assert decision.state is MatchState.BREAKUP_FORWARD
    assert decision.linear_velocity_m_s == sequence.config.breakup_forward_speed_m_s


def test_reselect_waits_for_stop_and_new_frame_then_grabs_next_green() -> None:
    sequence = make_sequence(config=runtime_config(
        opportunistic_single_green_enabled=True,
        safe_zone_calibration_stop_confirm_time_s=0.01,
    ), initial_field_position=FieldPoint(0.0, 500.0))
    start_sequence(sequence)
    sequence.state = MatchState.BREAKUP_FORWARD
    inject_breakup_plan(sequence)
    blocked = sequence.step(10, perception=None, heading_rad=math.pi / 2,
                            cumulative_distance_m=0.0)
    assert blocked.state is MatchState.SEARCH_CLUSTER
    assert blocked.linear_velocity_m_s == 0.0

    def tick(time: int, frame=None, speed=0.0):
        return sequence.step(time, perception=frame, heading_rad=0.0,
                             cumulative_distance_m=0.0,
                             left_speed_feedback_m_s=speed, right_speed_feedback_m_s=speed)

    rolling = tick(20, snapshot(1, 20), speed=0.1)
    assert rolling.reason == "safe_zone_reselect_waiting_for_stop"
    assert rolling.angular_velocity_rad_s == 0.0
    tick(30)
    old_frame = snapshot(2, 40)
    stopped = tick(10_000_030, old_frame)
    assert stopped.reason == "safe_zone_reselect_waiting_for_new_frame"
    assert tick(10_000_040, old_frame).linear_velocity_m_s == 0.0
    assert sequence._tracker.tracks == ()
    for frame, time in ((3, 10_000_050), (4, 10_000_060)):
        decision = tick(time, snapshot(frame, time, observation(frame, time, GroundPoint(500.0, 0.0))))
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.selected_track_id is not None
    assert not sequence._path_recovery


def test_selection_skips_unsafe_largest_group_for_next_cluster() -> None:
    sequence = make_sequence()
    sequence._latest_heading_rad = math.pi / 2
    unsafe = [GroundPoint(1400.0, y) for y in (-20.0, 0.0, 20.0, 40.0)]
    safe = [GroundPoint(100.0, y) for y in (-1200.0, -1220.0, -1240.0)]
    assert sequence._largest_ground_group(unsafe + safe) == safe
    # 持续看到同一危险团不会再次选中，不依赖 tracker ID 黑名单。
    assert sequence._largest_ground_group(unsafe) is None
    assert sequence._largest_ground_group(unsafe) is None


def test_selection_prefers_green_group_before_larger_non_green_group() -> None:
    sequence = make_sequence()
    sequence._latest_heading_rad = 0.0

    observations = [
        observation(1, 10, GroundPoint(600.0, -20.0), box_x=10.0),
        observation(1, 10, GroundPoint(600.0, 20.0), box_x=30.0),
        observation(
            1,
            10,
            GroundPoint(300.0, -30.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=50.0,
        ),
        observation(
            1,
            10,
            GroundPoint(300.0, 0.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=70.0,
        ),
        observation(
            1,
            10,
            GroundPoint(300.0, 30.0),
            target_class=TargetClass.BLACK_CORE,
            box_x=90.0,
        ),
    ]
    sequence._tracker.update(10, observations)
    sequence._tracker.update(
        20,
        [
            replace(
                item,
                frame_sequence=2,
                capture_timestamp_ns=20,
                result_timestamp_ns=20,
            )
            for item in observations
        ],
    )

    sequence._latest_perception = snapshot(2, 20, *(replace(item, frame_sequence=2,
        capture_timestamp_ns=20, result_timestamp_ns=20) for item in observations))
    measurement = sequence._cluster_ground_measurement(20)

    assert measurement is not None
    assert measurement.center in (GroundPoint(600.0, -20.0), GroundPoint(600.0, 20.0))


def test_breakup_center_uses_field_boundary_minus_gripper_offset() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._latest_heading_rad = 0.0

    outside = [
        GroundPoint(1400.0, -20.0),
        GroundPoint(1400.0, 20.0),
    ]
    inside = [
        GroundPoint(600.0, -20.0),
        GroundPoint(600.0, 20.0),
    ]

    assert sequence._largest_ground_group(outside) is None
    assert "breakup_center_out_of_field_boundary" in (
        sequence._last_cluster_rejection_reason or ""
    )
    assert sequence._largest_ground_group(inside) == inside
    assert sequence._breakup_center_within_field_boundary(
        FieldPoint(1300.0, -1300.0)
    )
    assert not sequence._breakup_center_within_field_boundary(
        FieldPoint(1301.0, 0.0)
    )

    # 中点仍在 ±1300 mm 内，但完成接近和配置的前推距离会越界，也必须放弃。
    endpoint_outside = [
        GroundPoint(1200.0, -20.0),
        GroundPoint(1200.0, 20.0),
    ]
    assert sequence._largest_ground_group(endpoint_outside) is None
    assert "breakup_end_out_of_field_boundary" in (
        sequence._last_cluster_rejection_reason or ""
    )


@pytest.mark.parametrize("heading", [math.pi / 2, -math.pi / 2])
def test_green_selection_skips_both_safe_zones(heading) -> None:
    sequence = make_sequence(config=runtime_config(opportunistic_single_green_enabled=True))
    start_sequence(sequence)
    sequence.state = MatchState.SEARCH_CLUSTER
    for frame, time in ((1, 10), (2, 20)):
        decision = sequence.step(time, perception=snapshot(
            frame, time, observation(frame, time, GroundPoint(1400.0, 0.0)),
            observation(frame, time, GroundPoint(300.0, -900.0), box_x=30.0),
        ), heading_rad=heading, cumulative_distance_m=0.0)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert sequence._selected_green_ground == GroundPoint(300.0, -900.0)


def test_grasp_exclusions_mark_delivered_supplies_but_not_danger() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._latest_heading_rad = 0.0

    # 初始位姿在场地原点、航向 0，机器人系坐标与场地坐标一致。
    delivered = observation(1, 10, GroundPoint(0.0, 1400.0))
    outside = observation(1, 10, GroundPoint(300.0, -900.0))
    danger_inside = observation(
        1, 10, GroundPoint(100.0, 1300.0), target_class=TargetClass.BLUE_DANGER,
    )

    filtered = sequence.grasp_excluded_observation_indices(
        snapshot(1, 10, delivered, outside, danger_inside)
    )

    assert filtered == frozenset({0})


def test_grasp_exclusions_mark_carried_members_during_refill_scan() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._latest_heading_rad = 0.0
    sequence._near_field_grasp_config = NearFieldGraspConfig()
    stowed_limit = sequence._greedy_new_target_min_x_mm()
    assert stowed_limit == pytest.approx(160.0)

    # 爪内成员：地面投影落在扫描本身要求新物资越过的界限之内。
    carried = observation(1, 10, GroundPoint(stowed_limit - 60.0, 10.0))
    new_supply = observation(1, 10, GroundPoint(stowed_limit + 140.0, -40.0))
    carried_danger = observation(
        1, 10, GroundPoint(stowed_limit - 60.0, -20.0),
        target_class=TargetClass.BLUE_DANGER,
    )
    frame = snapshot(1, 10, carried, new_supply, carried_danger)
    sequence._greedy_active = True

    during_scan = sequence.grasp_excluded_observation_indices(frame)
    assert during_scan == frozenset({0})

    # 没有补夹扫描时不能启用该界限，否则会把正在接近的目标一并滤掉。
    sequence._greedy_active = False
    untouched = sequence.grasp_excluded_observation_indices(frame)
    assert untouched == frozenset()


def test_near_field_plan_with_delivered_member_is_not_committed() -> None:
    from test_near_field_grasp import selector as near_selector, target as near_target

    near_selector_ = near_selector()
    plan = near_selector_.select((near_target(1, x=300.0, y=0.0),)).plan
    assert plan is not None

    # 航向 0 时场地坐标 = 起点 + 机器人系坐标，用起点把同一个计划移到安全区内。
    outside = make_sequence(initial_field_position=FieldPoint(0.0, 0.0))
    start_sequence(outside)
    outside._latest_heading_rad = 0.0
    assert outside._near_field_plan_has_delivered_member(plan) is False
    assert outside._near_field_plan_has_delivered_member(None) is False

    inside = make_sequence(initial_field_position=FieldPoint(-100.0, 1400.0))
    start_sequence(inside)
    inside._latest_heading_rad = 0.0
    assert inside._near_field_plan_has_delivered_member(plan) is True


def test_green_approach_rechecks_route_and_clears_selection() -> None:
    sequence = make_sequence(initial_field_position=FieldPoint(0.0, 500.0))
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._selected_track_id = 42
    sequence._green_approach_distance_m = 0.8
    sequence._green_reference_heading_rad = math.pi / 2.0
    blocked = sequence.step(10, perception=None, heading_rad=math.pi / 2,
                            cumulative_distance_m=0.0)
    assert blocked.state is MatchState.SEARCH_CLUSTER
    assert blocked.selected_track_id is None
    assert blocked.linear_velocity_m_s == 0.0
    assert blocked.gripper_posture is GripperPosture.TRANSPORT


def test_green_reference_safe_zone_route_reselects_after_stop() -> None:
    sequence = make_sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
        initial_field_position=FieldPoint(0.0, 0.0),
    )
    start_sequence(sequence)
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    sequence._selected_track_id = 1
    for frame in range(1, 12):
        decision = sequence.step(
            frame * 10,
            perception=snapshot(
                frame,
                frame * 10,
                observation(frame, frame * 10, GroundPoint(1400.0, 0.0)),
            ),
            heading_rad=math.pi / 2.0,
            cumulative_distance_m=0.0,
        )
    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "target_path_intersects_safe_zone_stop_and_reselect"
    assert decision.linear_velocity_m_s == decision.angular_velocity_rad_s == 0.0


def test_all_blue_largest_cluster_does_not_hide_smaller_mixed_cluster():
    sequence = make_sequence()
    sequence._latest_heading_rad = 0.0
    sequence._transport_count = 1
    objects = [
        observation(1, 10, GroundPoint(400, -300 + index * 30),
                    target_class=TargetClass.BLUE_DANGER, box_x=5 + index * 15)
        for index in range(3)
    ] + [
        observation(1, 10, GroundPoint(400, 300), box_x=55),
        observation(1, 10, GroundPoint(400, 330), target_class=TargetClass.BLACK_CORE, box_x=75),
    ]
    sequence._tracker.update(10, objects)
    sequence._tracker.update(20, [replace(item, frame_sequence=2, capture_timestamp_ns=20,
                                        result_timestamp_ns=20) for item in objects])
    sequence._latest_perception = snapshot(2, 20, *(replace(item, frame_sequence=2,
        capture_timestamp_ns=20, result_timestamp_ns=20) for item in objects))
    measurement = sequence._cluster_ground_measurement(20)
    assert measurement is not None
    assert measurement.center in (GroundPoint(400,300), GroundPoint(400,330))
    assert len(sequence._cluster_selected_track_ids) == 2
