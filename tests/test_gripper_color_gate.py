from __future__ import annotations

from dataclasses import replace
import math

import cv2
import numpy as np
import pytest

from rescue_vision.app.match import MatchState, GripperPosture
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    FakeInferenceBackend,
    TargetClass,
    TargetPoseDetector,
    UndistortedBoundingBox,
)
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.perception.gripper_color import GripperColorConfig, GripperColorObservation, observe_gripper_colors
from rescue_vision.motion.protocol import SensorFlags
from test_perception import color_config
from test_match import observation, snapshot, make_sequence, start_sequence, safe_zone_snapshot_for_pose
from test_match_near_field import _sequence, _complete_supply_pickup
from test_near_field_grasp import target
from test_gripper_width_sequence import motion_sample
from rescue_vision.localization import FieldPose2D

G, K, O, B = TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE, TargetClass.ORANGE_INJURED, TargetClass.BLUE_DANGER


def color_snapshot(frame, capture, result, colors):
    evidence = GripperColorObservation(
        (UndistortedPixel(40, 70), UndistortedPixel(60, 70), UndistortedPixel(60, 99)),
        tuple((c, 0.1) for c in colors), frozenset(colors),
    )
    return replace(snapshot(frame, capture), result_timestamp_ns=result, timing=None, gripper_color=evidence)


def orange_detection(frame, capture, box, *, k0=None):
    detected = observation(frame, capture, GroundPoint(200, 0), target_class=O)
    roi_box = UndistortedBoundingBox(
        math.floor(box.x_min), math.floor(box.y_min),
        math.ceil(box.x_max), math.ceil(box.y_max),
    )
    return replace(
        detected,
        box=box,
        k0=(
            UndistortedPixel((box.x_min + box.x_max) / 2, box.y_max)
            if k0 is None
            else k0
        ),
        color_segmentation=replace(
            detected.color_segmentation,
            roi_box=roi_box,
            mask=np.full(
                (
                    int(roi_box.y_max - roi_box.y_min),
                    int(roi_box.x_max - roi_box.x_min),
                ),
                255,
                dtype=np.uint8,
            ),
        ),
    )



def blue_detection(frame, capture, box):
    detected = orange_detection(frame, capture, box)
    return replace(detected, model_target_class=B, target_class=G,
                   k0=None, k0_confidence=0.0, ground_point=None)


def test_independent_roi_detects_conflicting_patch_without_model_detection():
    image = np.full((400, 600, 3), 220, np.uint8)
    image[320:380, 275:305] = (0, 200, 0)
    image[350:375, 315:335] = (255, 180, 0)
    image[0:280] = (0, 130, 255)  # 大面积爪外橙色不参与门禁
    image[350:352, 300:302] = 0  # 小黑点不当色块
    detector = TargetPoseDetector(FakeInferenceBackend(()), detection_threshold=0.1,
        k0_threshold=0.5, color_classifier=color_config(), max_observation_age_ms=1000)
    with detector:
        result = detector.detect_realtime(CameraFrame(1, 0, image), image, result_timestamp_ns=100)
    assert result.observations == ()
    assert result.gripper_color.present_classes == {G, B}
    assert dict(result.gripper_color.component_fractions)[B] >= 0.03


@pytest.mark.parametrize('override', [
    {'polygon_normalized': ((-0.1, 0), (1, 0), (1, 1))},
    {'polygon_normalized': ((0, 0), (1, 1), (0, 1), (1, 0))},
    {'polygon_normalized': ((0, 0), (0, 0), (0, 0))},
    {'min_component_fraction': float('nan')}, {'min_component_fraction': 0}, {'enabled': 1},
    {'orange_bbox_min_color_fraction': 0},
    {'orange_bbox_min_color_fraction': 1.1},
    {'orange_bbox_min_color_fraction': float('nan')},
    {'orange_distinct_max_bbox_iou': -0.1},
    {'orange_distinct_max_bbox_iou': 1.1},
    {'orange_distinct_min_k0_distance_px': 0},
    {'orange_distinct_min_k0_distance_px': float('inf')},
])
def test_invalid_roi_config_is_rejected(override):
    with pytest.raises(ValueError):
        GripperColorConfig(**override)


@pytest.mark.parametrize('transports,cargo,seen,conflict', [
    (0, (G,), {G}, False), (0, (G,), {G, K}, True),
    (0, (G,), {O}, True), (0, (G,), {B}, True),
    (1, (G, K), {G, K}, False), (1, (K,), {G}, False),
    (1, (G, K), {O}, True), (1, (G, K), {B}, True),
    (1, (O,), {O}, False), (1, (O,), {G}, True),
    (1, (O,), {K}, True), (1, (O,), {B}, True),
])
def test_gate_uses_current_trip_legality_without_reclassifying_targets(transports, cargo, seen, conflict):
    seq = _sequence(transports=transports)
    seq._transport_target_classes = cargo
    seq._cargo_capture_floor_ns = 100
    seq._latest_perception = color_snapshot(1, 100, 200, seen)
    assert bool(seq._gripper_color_conflict(200)) is conflict
    assert seq._transport_target_classes == cargo


def test_old_preclose_future_and_stale_colors_do_not_release_current_cargo():
    seq = _sequence(transports=1)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 1_000_000_000
    for capture, result, now in [(900_000_000, 1_300_000_000, 1_300_000_000),
                                 (1_100_000_000, 1_500_000_000, 1_000_000_000),
                                 (1_100_000_000, 1_500_000_000, 10_000_000_000)]:
        seq._latest_perception = color_snapshot(1, capture, result, {B})
        assert seq._gripper_color_conflict(now) is None


def test_two_orange_model_boxes_overlapping_gripper_roi_trigger_release():
    seq = _sequence(transports=1)
    seq._started = True
    seq._transport_target_classes = (O,)
    seq._cargo_capture_floor_ns = 100
    latest = color_snapshot(1, 100, 200, {O})
    overlapping = (
        orange_detection(1, 100, UndistortedBoundingBox(40, 70, 46, 78)),
        # HSV may disagree, but the four-class model remains category authority.
        replace(
            orange_detection(1, 100, UndistortedBoundingBox(54, 90, 60, 98)),
            target_class=G,
        ),
    )
    seq._latest_perception = replace(latest, observations=overlapping)

    assert seq._gripper_color_conflict(200) == (
        "multiple_distinct_orange_detections_in_gripper_roi:"
        "candidate_count=2,bbox_iou=0.0000,k0_distance_px=24.4"
    )
    decision = seq.step(200, perception=seq._latest_perception, heading_rad=0.0,
                        cumulative_distance_m=0.0)
    assert decision.state is MatchState.MISGRASP_OPEN
    assert decision.gripper_posture is GripperPosture.OPEN


@pytest.mark.parametrize(
    ("second_color_fraction", "expected_conflict"),
    [
        (0.1499, False),
        (0.15, True),
    ],
)
def test_two_orange_boxes_require_minimum_orange_fraction_in_each_bbox(
    second_color_fraction: float,
    expected_conflict: bool,
) -> None:
    seq = _sequence(transports=1)
    seq._transport_target_classes = (O,)
    seq._cargo_capture_floor_ns = 100
    latest = color_snapshot(1, 100, 200, {O})
    second = orange_detection(
        1,
        100,
        UndistortedBoundingBox(54, 90, 60, 98),
    )
    second = replace(
        second,
        color_segmentation=replace(
            second.color_segmentation,
            color_fraction=second_color_fraction,
        ),
    )
    seq._latest_perception = replace(
        latest,
        observations=(
            orange_detection(1, 100, UndistortedBoundingBox(40, 70, 46, 78)),
            second,
        ),
    )

    conflict = seq._gripper_color_conflict(200)
    assert bool(conflict) is expected_conflict
    diagnostic = seq.gripper_color_diagnostic(200)
    assert "orange_bbox_roi_overlap_count=2" in diagnostic
    expected_qualified = 2 if expected_conflict else 1
    assert f"orange_bbox_color_qualified_count={expected_qualified}" in diagnostic


@pytest.mark.parametrize(
    ("state", "safe_zone_phase"),
    [
        (MatchState.TRANSPORT_RELEASE, "closing_before_final_forward"),
        (MatchState.TRANSPORT_FORWARD, "forward_final_closed"),
        # The grab-transport bench variant uses this open final-push phase.
        (MatchState.TRANSPORT_FORWARD, "forward_final_open"),
        (MatchState.TRANSPORT_RELEASE, "stopping_before_exit_opening"),
        (MatchState.TRANSPORT_RELEASE, "opening_after_transport"),
        (MatchState.RETURN_BACKUP, "idle"),
    ],
)
def test_misgrasp_gate_is_disabled_during_final_push_and_safe_zone_exit(
    state: MatchState,
    safe_zone_phase: str,
):
    seq = _sequence(transports=1)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 100
    seq.state = state
    seq._safe_zone_phase = safe_zone_phase
    seq._return_phase = "exit_reverse" if state is MatchState.RETURN_BACKUP else "idle"
    seq._latest_perception = color_snapshot(1, 100, 200, {B})

    assert seq._gripper_color_conflict(200) is None


def test_misgrasp_gate_remains_active_before_d2_final_push():
    seq = _sequence(transports=1)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 100
    seq.state = MatchState.TRANSPORT_FORWARD
    seq._safe_zone_phase = "forward_d2_line"
    seq._latest_perception = color_snapshot(1, 100, 200, {B})

    assert seq._gripper_color_conflict(200) == "conflicting_gripper_colors:blue_danger"


@pytest.mark.parametrize("second_box", [
    UndistortedBoundingBox(10, 10, 20, 20),
    # Merely touching the left ROI boundary has no positive-area overlap.
    UndistortedBoundingBox(30, 60, 40, 70),
])
def test_fewer_than_two_overlapping_orange_model_boxes_do_not_release(second_box):
    seq = _sequence(transports=1)
    seq._transport_target_classes = (O,)
    seq._cargo_capture_floor_ns = 100
    latest = color_snapshot(1, 100, 200, {O})
    detections = (
        orange_detection(1, 100, UndistortedBoundingBox(42, 74, 48, 82)),
        orange_detection(1, 100, second_box),
    )
    seq._latest_perception = replace(latest, observations=detections)

    assert seq._gripper_color_conflict(200) is None


@pytest.mark.parametrize("first_box,first_k0,second_box,second_k0", [
    # Near-identical boxes and K0 values are duplicate detections.
    (
        UndistortedBoundingBox(42, 74, 52, 86), UndistortedPixel(47, 84),
        UndistortedBoundingBox(43, 75, 53, 87), UndistortedPixel(48, 85),
    ),
    # Distinct boxes alone are insufficient when K0 remains too close.
    (
        UndistortedBoundingBox(40, 70, 46, 78), UndistortedPixel(49, 84),
        UndistortedBoundingBox(54, 90, 60, 98), UndistortedPixel(51, 86),
    ),
    # Distinct K0 values alone are insufficient when boxes still overlap heavily.
    (
        UndistortedBoundingBox(40, 70, 60, 99), UndistortedPixel(42, 73),
        UndistortedBoundingBox(41, 71, 60, 99), UndistortedPixel(58, 97),
    ),
])
def test_duplicate_orange_detections_require_distinct_bbox_and_k0(
    first_box, first_k0, second_box, second_k0,
):
    seq = _sequence(transports=1)
    seq._transport_target_classes = (O,)
    seq._cargo_capture_floor_ns = 100
    latest = color_snapshot(1, 100, 200, {O})
    seq._latest_perception = replace(latest, observations=(
        orange_detection(1, 100, first_box, k0=first_k0),
        orange_detection(1, 100, second_box, k0=second_k0),
    ))

    assert seq._gripper_color_conflict(200) is None


def test_missing_k0_never_proves_two_distinct_orange_targets():
    seq = _sequence(transports=1)
    seq._transport_target_classes = (O,)
    seq._cargo_capture_floor_ns = 100
    latest = color_snapshot(1, 100, 200, {O})
    missing_k0 = replace(
        orange_detection(1, 100, UndistortedBoundingBox(54, 90, 60, 98)),
        k0=None,
        k0_confidence=0.0,
        ground_point=None,
    )
    seq._latest_perception = replace(latest, observations=(
        orange_detection(1, 100, UndistortedBoundingBox(40, 70, 46, 78)),
        missing_k0,
    ))

    assert seq._gripper_color_conflict(200) is None


def test_two_orange_boxes_from_preclose_frame_do_not_release_current_cargo():
    seq = _sequence(transports=1)
    seq._transport_target_classes = (O,)
    seq._cargo_capture_floor_ns = 200
    latest = color_snapshot(1, 100, 200, {O})
    seq._latest_perception = replace(latest, observations=(
        orange_detection(1, 100, UndistortedBoundingBox(40, 70, 46, 78)),
        orange_detection(1, 100, UndistortedBoundingBox(54, 90, 60, 98)),
    ))

    assert seq._gripper_color_conflict(200) is None


@pytest.mark.parametrize('count', [2, 3])
def test_two_leaves_one_slot_and_three_skips_supplement(count):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    members = tuple(target(i=i+1, x=300, y=(i-(count-1)/2)*30) for i in range(count))
    decision = _complete_supply_pickup(seq, members)
    assert seq.carried_target_count == count
    if count == 3:
        assert decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
        assert not seq._can_greedy_pickup()
    else:
        assert decision.state is MatchState.TRANSPORT_GREEDY_SCAN
        assert seq.near_field_policy.max_targets == 1
        seq._begin_near_field_grasp(400_000_000)
        decision = _complete_supply_pickup(seq, (target(i=9, x=300, timestamp=400_000_000),), now=400_000_000)
        assert seq.carried_target_count == 3
        assert decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE


@pytest.mark.parametrize('blue_bbox_only', [False, True])
@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5, 300, 250), (10, 600, 400)])
def test_delayed_color_conflict_opens_reverses_250mm_then_tries_locked_breakup(poll_ms, delay_ms, period_ms, blue_bbox_only):
    seq = _sequence(transports=1)
    seq._started = True
    seq.config = replace(seq.config, green_max_age_ms=1200)
    seq._transport_target_classes = (G, K)
    seq._cargo_capture_floor_ns = 0
    seq.state = MatchState.TRANSPORT_GREEDY_SCAN
    seq._greedy_started_ns = 0
    latest = None
    distance = 0.0
    velocity = 0.0
    opened = reversed_at = None
    for ms in range(poll_ms, 6000, poll_ms):
        now = ms * 1_000_000
        distance += velocity * poll_ms / 1000
        if ms % 10 == 0:
            seq.observe_grasp_motion(replace(motion_sample(now), left_encoder_count=round(distance * 10000), right_encoder_count=round(distance * 10000)))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            frame = (ms-delay_ms)//period_ms + 1
            latest = color_snapshot(frame, now-delay_ms*1_000_000, now, {G, B})
            if blue_bbox_only:
                capture = now - delay_ms * 1_000_000
                blue = replace(orange_detection(frame, capture, UndistortedBoundingBox(30, 60, 40, 70)),
                               model_target_class=B)
                latest = replace(color_snapshot(frame, capture, now, {G}),
                                 observations=(blue,))
        decision = seq.step(now, perception=latest, heading_rad=0.0, cumulative_distance_m=distance,
                            left_speed_feedback_m_s=velocity, right_speed_feedback_m_s=velocity)
        velocity = decision.linear_velocity_m_s
        if decision.state is MatchState.MISGRASP_OPEN:
            opened = opened or ms
            assert decision.gripper_posture is GripperPosture.OPEN
        if velocity < 0:
            reversed_at = reversed_at or ms
            assert decision.gripper_posture is GripperPosture.OPEN
        if decision.reason == 'misgrasp_released_backed_250mm_breakup_locked_group':
            assert -distance == pytest.approx(0.250, abs=0.002)
            assert seq.carried_target_count == 0
            assert decision.angular_velocity_rad_s == 0
            assert decision.state is MatchState.BREAKUP_SETTLE
            assert seq._misgrasp_release_pose.position == FieldPoint(0, 0)
            break
    else:
        pytest.fail(f'No recovery progress: {decision}')
    assert opened <= delay_ms + period_ms
    assert reversed_at - opened < 400
    # 旧冲突帧仍在，也不能重新启动开爪/后退；同区域须重新检测选组。
    decision = seq.step(now+poll_ms*1_000_000, perception=latest, heading_rad=0.0,
                        cumulative_distance_m=distance, left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
    assert decision.state is MatchState.BREAKUP_SETTLE
    assert decision.angular_velocity_rad_s == 0
    # 没有物块几何时，短窗口内退出；旧颜色帧不能重新启动同一次误夹。
    exit_started_ms = ms
    for ms in range(ms + 2 * poll_ms, ms + 2000, poll_ms):
        now = ms * 1_000_000
        seq.observe_grasp_motion(replace(motion_sample(now), left_encoder_count=round(distance * 10000), right_encoder_count=round(distance * 10000)))
        decision = seq.step(now, perception=latest, heading_rad=0.0,
                            cumulative_distance_m=distance, left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        if decision.state is MatchState.SEARCH_CLUSTER and decision.angular_velocity_rad_s != 0:
            break
    else:
        pytest.fail(f'No bounded no-plan exit: {decision}')
    assert ms - exit_started_ms <= (seq.config.breakup_no_plan_reobserve_ms
                                   + seq.config.safe_zone_calibration_stop_confirm_time_s * 1000
                                   + 2 * poll_ms)
    assert not seq._misgrasp_breakup_active


@pytest.mark.parametrize('mode', ['motion', 'invalid', 'stale', 'duplicate'])
def test_recovery_does_not_reverse_without_real_stationary_telemetry(mode):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    seq._begin_misgrasp_recovery(0, 'blue')
    seq.observe_grasp_motion(motion_sample(10_000_000))
    seq.observe_grasp_motion(motion_sample(20_000_000))
    now = 200_000_000
    if mode == 'motion':
        seq.observe_grasp_motion(motion_sample(now, count=1))
    elif mode == 'invalid':
        seq.observe_grasp_motion(motion_sample(now, flags=SensorFlags(0)))
    elif mode == 'duplicate':
        seq.observe_grasp_motion(replace(motion_sample(now), sample_timestamp_us=20_000))
    else:
        now = 5_000_000_000
    decision = seq._step_misgrasp_recovery(now, 0.0)
    assert decision.linear_velocity_m_s == 0
    assert decision.state is MatchState.MISGRASP_OPEN


@pytest.mark.parametrize('side,missing', [(-1, 1), (1, 2)])
def test_safe_zone_calibrates_with_only_opposite_pair_and_full_bbox(side, missing):
    seq = make_sequence(initial_field_position=FieldPoint(side*165, 700))
    seq.config = replace(seq.config, safe_zone_fallback_target_field=FieldPoint(side*165, 1137))
    start_sequence(seq)
    seq._latest_heading_rad = seq._raw_heading_rad = math.pi/2
    pose = FieldPose2D(FieldPoint(side*165, 700), math.pi/2)
    from rescue_vision.perception import UndistortedBoundingBox
    seq._safe_zone_stop_since_ns = 1
    for frame in range(1, 3):
        snap = safe_zone_snapshot_for_pose(frame, frame*100, pose)
        zone = snap.field_features.safe_zones[0]
        name = 'image_left_landmark' if missing == 1 else 'image_right_landmark'
        zone = replace(zone, box=UndistortedBoundingBox(10 if side < 0 else 30, 10, 70 if side < 0 else 90, 90),
                       **{name: replace(getattr(zone, name), ground=None, undistorted=None, confidence=0.0)})
        seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=(zone,)))
        assert seq._safe_zone_has_required_ground_keypoints(zone)
        assert seq._safe_zone_required_points_visible(zone)
        assert seq._collect_safe_zone_key_sample(frame*100)
        assert not seq._collect_safe_zone_key_sample(frame*100)
    assert seq._lock_safe_zone_calibration_plan(500)
    assert seq.safe_zone_calibration_pose.position.x == pytest.approx(pose.position.x)
    assert seq.safe_zone_calibration_pose.heading_rad == pytest.approx(pose.heading_rad)
    assert len(seq._safe_zone_calibration_pose.used_roles) == 2


def test_runtime_config_builds_enabled_gripper_color_observation():
    config = load_runtime_config('configs/runtime.match.yaml')
    assert config.perception.gripper_color.enabled
    assert config.perception.gripper_color.polygon_normalized == (
        (0.47, 0.85), (0.53, 0.85), (0.57, 0.97), (0.43, 0.97),
    )
    assert config.perception.gripper_color.min_component_fraction == 0.03
    assert config.perception.gripper_color.black_min_thickness_fraction == 0.12
    assert config.perception.gripper_color.orange_bbox_min_color_fraction == 0.15
    assert config.perception.gripper_color.orange_distinct_max_bbox_iou == 0.20
    assert config.perception.gripper_color.orange_distinct_min_k0_distance_px == 20.0


def test_default_gripper_roi_excludes_patch_ahead_of_jaw_tips() -> None:
    image = np.full((400, 600, 3), 220, np.uint8)
    image[310:335, 285:315] = (255, 180, 0)

    observation = observe_gripper_colors(
        image,
        GripperColorConfig(),
        color_config(),
    )

    assert observation is not None
    assert B not in observation.present_classes


def test_default_gripper_roi_keeps_patch_inside_jaws() -> None:
    image = np.full((400, 600, 3), 220, np.uint8)
    image[350:380, 285:315] = (255, 180, 0)

    observation = observe_gripper_colors(
        image,
        GripperColorConfig(),
        color_config(),
    )

    assert observation is not None
    assert B in observation.present_classes


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('shape', ['horizontal', 'vertical', 'diagonal', 'dashed', 'cross'])
def test_black_floor_lines_inside_gripper_do_not_trigger_release(scale, shape):
    image = np.full((400 * scale, 600 * scale, 3), 220, np.uint8)
    def line(a, b):
        cv2.line(image, tuple(v * scale for v in a), tuple(v * scale for v in b), (0, 0, 0), 4 * scale)
    if shape in {'horizontal', 'cross'}:
        line((220, 360), (380, 360))
    if shape in {'vertical', 'cross'}:
        line((300, 290), (300, 399))
    if shape == 'diagonal':
        line((255, 399), (335, 290))
    if shape == 'dashed':
        for y in range(290, 399, 18):
            line((300, y), (300, y + 12))
    # 无线宽过滤时，至少某些长线超过原面积门限；修复不提高面积阈值。
    obs = observe_gripper_colors(image, GripperColorConfig(), color_config())
    assert K not in obs.present_classes
    seq = _sequence(transports=0)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 0
    seq._latest_perception = replace(snapshot(1, 100), gripper_color=obs)
    assert seq._gripper_color_conflict(100) is None


@pytest.mark.parametrize('attached_line', [False, True])
def test_real_black_triangle_remains_conflicting_even_when_touching_floor_line(attached_line):
    image = np.full((400, 600, 3), 220, np.uint8)
    triangle = np.array([[300, 326], [273, 376], [327, 376]], np.int32)
    cv2.fillConvexPoly(image, triangle, (0, 0, 0))
    if attached_line:
        cv2.line(image, (300, 285), (300, 399), (0, 0, 0), 4)
    obs = observe_gripper_colors(image, GripperColorConfig(), color_config())
    assert K in obs.present_classes
    seq = _sequence(transports=0)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 0
    seq._latest_perception = replace(snapshot(1, 100), gripper_color=obs)
    assert seq._gripper_color_conflict(100) == 'conflicting_gripper_colors:black_core'


@pytest.mark.parametrize('color', [B, O, G])
def test_black_line_filter_does_not_filter_other_colors(color):
    hsv_colors = {B: (100, 230, 220), O: (10, 230, 220), G: (60, 230, 220)}
    hsv = np.full((400, 600, 3), (0, 0, 220), np.uint8)
    cv2.line(hsv, (220, 360), (380, 360), hsv_colors[color], 4)
    obs = observe_gripper_colors(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), GripperColorConfig(), color_config())
    assert color in obs.present_classes


@pytest.mark.parametrize('value', [0, -0.1, 1.1, float('nan'), float('inf'), True])
def test_black_line_thickness_config_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='black_min_thickness_fraction'):
        GripperColorConfig(black_min_thickness_fraction=value)


@pytest.mark.parametrize('hue,color', [(60, G), (10, O), (100, B)])
def test_dark_colored_faces_are_not_misreported_as_black(hue, color):
    hsv = np.full((400, 600, 3), (0, 0, 220), np.uint8)
    hsv[330:380, 280:320] = (hue, 230, 45)
    obs = observe_gripper_colors(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), GripperColorConfig(), color_config())
    assert obs.present_classes == {color}


def test_first_green_with_dark_face_does_not_release_but_actual_black_beside_it_does():
    hsv = np.full((400, 600, 3), (0, 0, 220), np.uint8)
    hsv[325:345, 280:318] = (60, 230, 190)
    hsv[345:378, 280:318] = (60, 230, 45)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    seq = _sequence(transports=0)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 0
    obs = observe_gripper_colors(image, GripperColorConfig(), color_config())
    seq._latest_perception = replace(snapshot(1, 100), gripper_color=obs)
    assert seq._gripper_color_conflict(100) is None
    image[353:381, 318:334] = (20, 20, 20)
    obs = observe_gripper_colors(image, GripperColorConfig(), color_config())
    seq._latest_perception = replace(snapshot(2, 200), gripper_color=obs)
    assert seq._gripper_color_conflict(200) == 'conflicting_gripper_colors:black_core'


@pytest.mark.parametrize("box,expected", [
    ((30, 60, 40, 70), True),  # Single corner contact.
    ((60, 75, 65, 85), True),  # Edge contact.
    ((48, 78, 52, 82), True),  # Crosses the sloped ROI edge.
    ((54, 75, 56, 80), True),  # Fully inside.
    ((30, 60, 70, 100), True),  # Contains the ROI.
    ((30, 60, 39.999, 70), False),
    ((40, 90, 45, 95), False),  # In ROI bounding box, outside polygon.
])
def test_blue_model_bbox_contact_triggers_misgrasp_without_hsv_or_k0(box, expected):
    seq = _sequence(transports=1)
    seq._started = True
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 100
    blue = replace(orange_detection(1, 100, UndistortedBoundingBox(*box), k0=UndistortedPixel(50, 80)),
                   model_target_class=B, target_class=G,
                   k0=None, k0_confidence=0.0, ground_point=None)
    seq._latest_perception = replace(color_snapshot(1, 100, 200, set()),
                                     observations=(blue,))
    assert bool(seq._gripper_color_conflict(200)) is expected
    if expected:
        decision = seq.step(200, perception=seq._latest_perception,
                            heading_rad=0.0, cumulative_distance_m=0.0)
        assert decision.state is MatchState.MISGRASP_OPEN
        assert decision.gripper_posture is GripperPosture.OPEN


@pytest.mark.parametrize("capture,result,now", [
    (900_000_000, 1_300_000_000, 1_300_000_000),
    (1_100_000_000, 1_500_000_000, 1_000_000_000),
    (1_100_000_000, 1_500_000_000, 10_000_000_000),
])
def test_blue_bbox_contact_respects_capture_and_result_age(capture, result, now):
    seq = _sequence(transports=1)
    seq._transport_target_classes = (G,)
    seq._cargo_capture_floor_ns = 1_000_000_000
    blue = replace(orange_detection(1, capture, UndistortedBoundingBox(30, 60, 40, 70)),
                   model_target_class=B)
    seq._latest_perception = replace(color_snapshot(1, capture, result, set()),
                                     observations=(blue,))
    assert seq._gripper_color_conflict(now) is None
