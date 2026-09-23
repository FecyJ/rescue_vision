from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.match import MatchState, GripperPosture
from rescue_vision.app.match import MatchPreflight
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D, normalize_angle
from rescue_vision.motion.relative_action import (
    RelativeActionCommand,
    RelativeActionPhase,
)
from rescue_vision.perception import TargetClass
from test_gripper_width_sequence import motion_sample
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
    d1_forward = plant.until(
        lambda d: d.reason.startswith("noncontact=safe_zone_d1_forward")
    )
    assert d1_forward.linear_velocity_m_s > 0.0
    assert abs(d1_forward.angular_velocity_rad_s) <= 1.0

    seq.state = MatchState.TRANSPORT_FORWARD
    seq._safe_zone_phase = "forward_d2_line"
    seq._d2_line_start_position = seq.estimated_field_position
    seq._d2_line_heading_rad = plant.heading
    seq._transport_forward_base_distance_m = plant.distance
    seq._transport_forward_distance_m = 0.5
    seq._action_settle_phase = None
    d2_forward = seq.step(
        plant.time_ns + 1,
        perception=plant.latest,
        heading_rad=plant.heading,
        cumulative_distance_m=plant.distance,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert d2_forward.reason.startswith("noncontact=safe_zone_d2_forward")
    assert d2_forward.linear_velocity_m_s > 0.0
    assert d2_forward.angular_velocity_rad_s == pytest.approx(0.0)


def test_completed_final_push_opens_and_starts_reverse_in_same_cycle(monkeypatch):
    """末段闭环完成的决策必须同时张爪倒车，不能再停一轮。"""

    seq = make_sequence(
        config=runtime_config(
            safe_zone_exit_distance_m=0.8,
            return_backup_speed_m_s=1.5,
        ),
        initial_field_position=FieldPoint(-130.0, 1115.0),
    )
    start_sequence(seq)
    seq._transport_target_classes = (
        TargetClass.GREEN_SUPPLY,
        TargetClass.BLACK_CORE,
    )
    seq.state = MatchState.TRANSPORT_FORWARD
    seq._safe_zone_phase = "forward_final_closed"
    seq._transport_forward_distance_m = 0.3
    complete = RelativeActionCommand(
        linear_velocity_m_s=0.0,
        angular_velocity_rad_s=0.0,
        phase=RelativeActionPhase.COMPLETE,
        complete=True,
        timed_out=False,
        reason="complete",
        position_error=0.0,
        heading_error_rad=0.0,
        braking_distance=0.0,
        telemetry_delay_s=0.01,
        use_zero_min_wheel_velocity=True,
    )
    monkeypatch.setattr(
        seq,
        "_safe_zone_forward_command",
        lambda *args, **kwargs: complete,
    )

    released = seq.step(
        1_000_000_000,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=2.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )

    assert released.state is MatchState.RETURN_BACKUP
    assert released.reason == "safe_zone_exit_reverse_open_loop"
    assert released.linear_velocity_m_s == pytest.approx(-1.5)
    assert released.gripper_posture is GripperPosture.OPEN
    assert seq._safe_zone_phase == "idle"
    assert seq._return_phase == "exit_reverse"
    assert seq._transport_count == 1


def test_safe_zone_alignment_does_not_start_forward_while_spinning_through_heading():
    """Regression for the 20260922 18:27 match route failure.

    The vehicle crossed the 0.02 rad D1 tolerance at high angular speed.  The
    old state machine treated that one sample as aligned and immediately
    requested about 1 m/s forward, which pushed the carried green block into
    the opposite safe zone.  Alignment must own braking and real stationarity
    before the straight segment can begin.
    """

    seq = make_sequence(
        config=runtime_config(
            safe_zone_fallback_heading_tolerance_rad=0.02,
            safe_zone_fallback_max_angular_velocity_rad_s=3.0,
            safe_zone_grab_to_d1_speed_m_s=1.0,
            action_settle_time_s=0.0,
        ),
        initial_field_position=FieldPoint(32.0, -404.0),
    )
    start_sequence(seq)
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase = "align_d1_line"
    target = seq._safe_zone_d1_target()
    target_heading = math.atan2(
        target.y - seq.estimated_field_position.y,
        target.x - seq.estimated_field_position.x,
    )
    plant = MotionPlant(seq, heading=target_heading - 1.2, dt=0.01)
    plant.angular = 2.5

    dangerous_crossing_seen = False
    started = None
    for _ in range(600):
        decision = plant.tick()
        heading_error = abs(normalize_angle(target_heading - plant.heading))
        if heading_error < 0.02 and abs(plant.angular) > 0.06:
            dangerous_crossing_seen = True
            assert decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
            assert decision.linear_velocity_m_s == 0.0
            assert "safe_zone_turn_to_d1_line" in decision.reason
        if decision.state is MatchState.TRANSPORT_FORWARD:
            started = decision
            break

    assert dangerous_crossing_seen
    assert started is not None
    assert started.reason == "safe_zone_d1_line_heading_reached_start_forward"
    assert abs(plant.angular) <= 0.06
    assert abs(normalize_angle(target_heading - plant.heading)) <= 0.02
    assert not any(
        decision.linear_velocity_m_s > 0.0
        for decision in plant.records
        if decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    )


def test_safe_zone_final_push_heading_waits_for_profiled_turn_stop():
    seq = make_sequence(
        config=runtime_config(
            safe_zone_fallback_heading_tolerance_rad=0.02,
            safe_zone_fallback_max_angular_velocity_rad_s=3.0,
        ),
        initial_field_position=FieldPoint(-165.0, 837.0),
    )
    start_sequence(seq)
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase = "align_y_at_d2"
    target_heading = seq._safe_zone_forward_heading_rad()
    plant = MotionPlant(seq, heading=target_heading - 1.2, dt=0.01)
    plant.angular = 2.5

    dangerous_crossing_seen = False
    released = None
    for _ in range(600):
        decision = plant.tick()
        heading_error = abs(normalize_angle(target_heading - plant.heading))
        if heading_error < 0.02 and abs(plant.angular) > 0.06:
            dangerous_crossing_seen = True
            assert decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
        if decision.state is MatchState.TRANSPORT_RELEASE:
            released = decision
            break

    assert dangerous_crossing_seen
    assert released is not None
    assert released.reason == "safe_zone_d2_heading_90_stopped_start_closing_gripper"
    assert abs(plant.angular) <= 0.06
    assert abs(normalize_angle(target_heading - plant.heading)) <= 0.02


def test_safe_zone_d2_does_not_change_phase_until_feedback_confirms_stop():
    seq = make_sequence(
        config=runtime_config(safe_zone_d1_to_d2_speed_m_s=0.5),
        initial_field_position=FieldPoint(-165.0, 700.0),
    )
    start_sequence(seq)
    seq.state = MatchState.TRANSPORT_FORWARD
    seq._safe_zone_phase = "forward_d2_line"
    seq._transport_forward_base_distance_m = 0.0
    seq._transport_forward_distance_m = 0.1
    seq._d2_line_heading_rad = math.pi / 2.0

    seq.observe_grasp_motion(motion_sample(1, count=0))
    moving = seq.step(
        1,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert moving.state is MatchState.TRANSPORT_FORWARD

    seq.observe_grasp_motion(motion_sample(100_000_000, count=1000))
    crossed_while_moving = seq.step(
        100_000_000,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.2,
        right_speed_feedback_m_s=0.2,
    )
    assert crossed_while_moving.state is MatchState.TRANSPORT_FORWARD
    assert "phase=settle" in crossed_while_moving.reason

    for timestamp_ns in (200_000_000, 300_000_000):
        seq.observe_grasp_motion(motion_sample(timestamp_ns, count=1000))
        waiting = seq.step(
            timestamp_ns,
            perception=None,
            heading_rad=math.pi / 2.0,
            cumulative_distance_m=0.1,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
        assert waiting.state is MatchState.TRANSPORT_FORWARD

    seq.observe_grasp_motion(motion_sample(400_000_000, count=1000))
    complete = seq.step(
        400_000_000,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.1,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert complete.state is MatchState.TRANSPORT_RELEASE
    assert complete.reason == "safe_zone_d2_closed_loop_complete_wait_before_opening"


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
    plant.until(
        lambda d: d.reason == "safe_zone_d1_closed_loop_complete_stop_before_calibration",
        seconds=8,
    )
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
    plant.until(
        lambda d: d.reason == "safe_zone_d1_closed_loop_complete_stop_before_calibration",
        seconds=8,
    )
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
    # Keep the synthetic protocol sequence non-negative. The exit action is
    # open-loop, so its completion must not depend on encoder movement.
    plant.distance = 1.0
    plant.until(lambda d: d.linear_velocity_m_s < 0, seconds=1)
    plant.until(lambda d: d.reason == "safe_zone_exit_distance_reached_wait_for_stop",
                seconds=4)

    exit_records = [d for d in plant.records
                    if d.state is MatchState.RETURN_BACKUP]
    # It must immediately use the configured high reverse speed instead of
    # waiting for encoder progress or a perception-based path hold.
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
