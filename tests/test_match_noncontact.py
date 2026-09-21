from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.match import MatchState, GripperPosture
from rescue_vision.app.match import MatchPreflight
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D
from rescue_vision.perception import TargetClass
from test_match import make_sequence, runtime_config, start_sequence, snapshot, observation, safe_zone_snapshot_for_pose
from noncontact_support import MotionPlant


def test_match_startup_overshoot_corrects_without_terminal_stop():
    seq = make_sequence(config=runtime_config(
        startup_turn_angle_rad=0.4,
        startup_turn_angular_velocity_rad_s=-1.0,
        startup_forward_distance_m=0.3,
        startup_forward_speed_m_s=0.5,
    ), initial_field_position=FieldPoint(0, 0))
    seq.preflight(0, MatchPreflight(True, True, True, True, True, True))
    seq.start(1)
    plant = MotionPlant(seq, heading=-math.pi/2, dt=0.01)
    plant.until(lambda d: d.state is MatchState.SEARCH_CLUSTER, seconds=8)
    assert all(d.state is not MatchState.TERMINAL_STOP for d in plant.records)
    assert not any("waiting_for_stop" in d.reason for d in plant.records)


@pytest.mark.parametrize("dt,latency,interval", [(0.005, 0.6, 0.25), (0.01, 0.3, 0.4)])
def test_transport_to_d1_then_d2_and_release_sequence(dt, latency, interval):
    seq = make_sequence(config=runtime_config(safe_zone_grab_to_d1_speed_m_s=0.7,
        safe_zone_d1_to_d2_speed_m_s=0.5, safe_zone_exit_distance_m=0.3,
        safe_zone_open_offset_mm=300, safe_zone_keypoint_reobserve_timeout_s=1.2,
        safe_zone_fallback_max_angular_velocity_rad_s=1.0),
        initial_field_position=FieldPoint(100, 200))
    start_sequence(seq)
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase = "align_d1_line"
    plant = MotionPlant(seq, heading=math.pi/2, dt=dt, latency_s=latency,
        frame_interval_s=interval, perception=safe_zone_snapshot_for_pose)
    plant.until(lambda d: d.reason == "safe_zone_d1_point_reached_stopped")
    d1 = seq._safe_zone_d1_target()
    assert math.hypot(seq.estimated_field_position.x-d1.x, seq.estimated_field_position.y-d1.y) <= 30
    assert abs(plant.linear) <= seq._motion_profile.stop_wheel_speed_m_s
    plant.until(lambda d: d.reason == "safe_zone_visual_calibrated_start_d2_line", seconds=5)
    assert seq.safe_zone_calibration_pose is not None
    corrected = seq.estimated_field_position
    plant.until(lambda d: d.reason == "safe_zone_d2_point_reached_stopped")
    d2 = seq._safe_zone_d2_target()
    assert math.hypot(seq.estimated_field_position.x-d2.x, seq.estimated_field_position.y-d2.y) <= 20
    assert corrected != seq.estimated_field_position
    plant.until(lambda d: d.reason == "safe_zone_exit_complete_start_search", seconds=20)
    reasons = [d.reason for d in plant.records]
    expected = ["safe_zone_d2_point_reached_stopped", "safe_zone_d2_reached_start_opening",
                "gripper_opened_at_d2_start_turn_to_90", "safe_zone_d2_heading_reached_stopped",
                "safe_zone_d2_heading_90_stopped_start_closing_gripper",
                "gripper_closed_after_d2_heading_start_forward_settle",
                "safe_zone_reached_transport_endpoint_wait_before_opening",
                "safe_zone_transport_stopped_start_opening", "safe_zone_exit_complete_start_search"]
    indices = [reasons.index(reason) for reason in expected]
    assert indices == sorted(indices)
    assert seq.carried_target_count == 0
    assert plant.decision.angular_velocity_rad_s != 0
    for d in plant.records:
        if "noncontact=d1," in d.reason or "noncontact=d2," in d.reason:
            assert d.gripper_posture is GripperPosture.CLOSED


@pytest.mark.parametrize("target_class", [
    TargetClass.GREEN_SUPPLY,
    TargetClass.BLACK_CORE,
    TargetClass.ORANGE_INJURED,
])
def test_transport_ignores_ordinary_targets_beside_pickup(
    target_class: TargetClass,
):
    seq = make_sequence(config=runtime_config(safe_zone_grab_to_d1_speed_m_s=0.5),
                        initial_field_position=FieldPoint(-165, 0))
    start_sequence(seq)
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase = "align_d1_line"

    def nearby_target(frame, now, pose):
        del pose
        return snapshot(frame, now, observation(frame, now, GroundPoint(183, 40),
                        target_class=target_class))

    plant = MotionPlant(seq, heading=math.pi/2, perception=nearby_target, latency_s=0.3)
    plant.until(lambda d: d.reason == "safe_zone_d1_point_reached_stopped", seconds=8)
    assert all(d.state is not MatchState.TERMINAL_STOP for d in plant.records)
    assert not any("path_blocked" in d.reason for d in plant.records)
    assert any(d.linear_velocity_m_s > 0 for d in plant.records)


def test_danger_does_not_block_transport_and_cargo_is_preserved():
    seq = make_sequence(config=runtime_config(safe_zone_grab_to_d1_speed_m_s=0.5),
                        initial_field_position=FieldPoint(-165, 0))
    start_sequence(seq)
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase = "align_d1_line"
    def danger(frame, now, pose):
        del pose
        return snapshot(frame, now, observation(frame, now, GroundPoint(300, 0),
                        target_class=TargetClass.BLUE_DANGER))
    plant = MotionPlant(seq, heading=math.pi/2, perception=danger, latency_s=0.3)
    plant.until(lambda d: d.reason == "safe_zone_d1_point_reached_stopped", seconds=8)
    assert all(d.state is not MatchState.TERMINAL_STOP for d in plant.records)
    assert not any("path_reobserve" in d.reason or "path_blocked" in d.reason
                   for d in plant.records)
    assert any(d.linear_velocity_m_s > 0 for d in plant.records)
    assert seq.carried_target_count == 1


def test_safe_zone_exit_reverse_ignores_persistent_delivered_targets():
    seq = make_sequence(config=runtime_config(
        safe_zone_exit_distance_m=0.8,
        return_backup_speed_m_s=1.5,
    ), initial_field_position=FieldPoint(-128, 1128))
    start_sequence(seq)
    seq.state = MatchState.RETURN_BACKUP
    seq._return_phase = "exit_reverse"

    def delivered_targets(frame, now, pose):
        del pose
        return snapshot(
            frame,
            now,
            observation(frame, now, GroundPoint(114, -20)),
            observation(frame, now, GroundPoint(192, 16)),
        )

    plant = MotionPlant(
        seq,
        heading=math.pi / 2,
        perception=delivered_targets,
        latency_s=0.3,
        frame_interval_s=0.3,
    )
    # Keep the synthetic protocol sequence non-negative while its encoder
    # position decreases through the 0.8 m reverse action.
    plant.distance = 1.0
    plant.until(lambda d: d.linear_velocity_m_s < 0, seconds=1)
    plant.until(lambda d: d.reason == "safe_zone_exit_distance_reached_wait_for_stop",
                seconds=4)

    exit_records = [d for d in plant.records
                    if d.state is MatchState.RETURN_BACKUP]
    # The 0.8 m action is too short to reach the configured 1.5 m/s before
    # its braking point, but it must immediately use the fastest feasible
    # reverse profile instead of being replaced by a zero-speed path hold.
    assert min(d.linear_velocity_m_s for d in exit_records) < -1.2
    assert not any("path_reobserve" in d.reason or "path_blocked" in d.reason
                   for d in exit_records)


def test_keypoint_reverse_is_not_blocked_by_targets_or_zones():
    seq = make_sequence(initial_field_position=FieldPoint(-128, 1128))
    start_sequence(seq)
    seq.state = MatchState.TRANSPORT_RELEASE
    seq._safe_zone_phase = "reversing_for_safe_zone_keypoints"

    def nearby_targets(frame, now, pose):
        del pose
        return snapshot(
            frame,
            now,
            observation(frame, now, GroundPoint(80, 0)),
            observation(frame, now, GroundPoint(130, 25)),
        )

    plant = MotionPlant(seq, heading=math.pi / 2, perception=nearby_targets,
                        latency_s=0.3, frame_interval_s=0.3)
    plant.distance = 1.0
    plant.until(lambda d: d.linear_velocity_m_s < 0, seconds=1)
    plant.until(lambda d: d.reason == "safe_zone_keypoint_reverse_limit_reached",
                seconds=3)

    assert not any("path_reobserve" in d.reason or "path_blocked" in d.reason
                   or "reverse_path_unavailable" in d.reason
                   for d in plant.records)


def test_arc_integration_matches_quarter_circle():
    seq = make_sequence()
    seq._update_fallback_field_position(0, 0)
    seq._update_fallback_field_position(math.pi/2, math.pi/2)
    assert seq.estimated_field_position.x == pytest.approx(1000)
    assert seq.estimated_field_position.y == pytest.approx(1000)
