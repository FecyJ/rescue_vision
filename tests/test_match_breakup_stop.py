"""Regressions for match_20260914_1650: brake flooding and endpoint re-entry."""
from __future__ import annotations

from dataclasses import replace
import struct

import pytest

from rescue_vision.app.match import GripperPosture, MatchDecision, MatchState
from rescue_vision.app.match_runtime import _apply_match_motion
from rescue_vision.motion import MessageType, MotionController, SensorFlags
from test_gripper_width_sequence import motion_sample
from test_match_breakup import frozen_plan, sequence
from test_motion import FakeCarChannel, FakeClock, command_reply_frame, limits


@pytest.mark.parametrize('poll_ms', [5, 10])
def test_changing_stop_diagnostics_do_not_flood_brake_or_starve_zero_heartbeats(poll_ms):
    clock = FakeClock()
    channel = FakeCarChannel([command_reply_frame(
        0, 0, command_sequence=0, command_type=MessageType.SOFT_BRAKE)])
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.synchronize(timeout_s=.01)
    channel.sent.clear()
    pending = []
    sent_count = 0
    braking = False
    for ms in range(0, 401, poll_ms):
        clock.timestamp_ns = ms*1_000_000
        controller.update(now_ns=clock.timestamp_ns)
        while pending and pending[0][0] <= ms:
            _, payload = pending.pop(0)
            channel.received.append(command_reply_frame(
                ms, ms*1000, command_sequence=struct.unpack_from('<H', payload, 1)[0],
                command_type=MessageType(payload[0])))
        controller.drain_messages()
        decision = MatchDecision(clock.timestamp_ns,
            MatchState.BREAKUP_FORWARD if ms < 200 else MatchState.BREAKUP_BACKWARD,
            0, 0, GripperPosture.CLOSED,
            f'breakup_segment_stop:motion_age_ms={ms%31}.2 gyro={ms/1000}', soft_brake=True)
        braking = _apply_match_motion(controller, decision, braking=braking)
        for payload in channel.sent[sent_count:]:
            pending.append((ms+30, payload))
        sent_count = len(channel.sent)
    brakes = [payload for payload in channel.sent if payload[0] == MessageType.SOFT_BRAKE]
    wheels = [payload for payload in channel.sent if payload[0] == MessageType.SET_WHEEL_SPEED]
    assert len(brakes) == 1
    assert len(wheels) >= 7
    assert all(struct.unpack_from('<hh', payload, 3) == (0, 0) for payload in wheels)
    assert controller.motion_synchronized
    # A real new drive ends the braking episode; the following brake is sent.
    moving = replace(decision, linear_velocity_m_s=.1, soft_brake=False)
    braking = _apply_match_motion(controller, moving, braking=braking)
    assert not braking
    assert _apply_match_motion(controller, decision, braking=braking)
    assert sum(payload[0] == MessageType.SOFT_BRAKE for payload in channel.sent) == 2


@pytest.mark.parametrize('forward', [True, False])
@pytest.mark.parametrize('poll_ms', [5, 10])
def test_endpoint_stays_latched_after_rollback_then_advances_when_stopped(forward, poll_ms):
    flow = sequence()
    frozen_plan(flow, forward=500, backward=300)
    flow.state = MatchState.BREAKUP_FORWARD if forward else MatchState.BREAKUP_BACKWARD
    sign = 1 if forward else -1
    endpoint = .5 if forward else .3
    stopped_at = None
    for ms in range(0, 401, poll_ms):
        now = ms*1_000_000
        # Cross the endpoint, then roll back 50 mm (like 818 -> 451 mm in the log).
        distance = sign*(endpoint+.01 if ms == 0 else endpoint-.05)
        flow.observe_grasp_motion(motion_sample(now, count=10000+round(distance*10000)))
        decision = flow.step(now, perception=None, heading_rad=0,
            cumulative_distance_m=distance,
            left_speed_feedback_m_s=.1 if ms < 100 else 0,
            right_speed_feedback_m_s=.1 if ms < 100 else 0)
        if flow.state is not (MatchState.BREAKUP_FORWARD if forward else MatchState.BREAKUP_BACKWARD):
            break
        assert decision.linear_velocity_m_s == 0
        assert decision.angular_velocity_rad_s == 0
        assert decision.gripper_posture is GripperPosture.CLOSED
        assert decision.soft_brake
        if stopped_at is None:
            stopped_at = flow._breakup_segment_stop_started_ns
        assert flow._breakup_segment_stop_started_ns == stopped_at == 0
    assert ms < 400
    assert flow.state is (MatchState.BREAKUP_BACKWARD if forward else MatchState.SEARCH_CLUSTER)
    if forward:
        assert flow._breakup_backward_base_distance_m == distance
        assert flow._breakup_retreat_mm == 300
        assert flow._breakup_segment_stop_started_ns is None


@pytest.mark.parametrize('failure', ['continued_motion', 'telemetry_gap', 'sensor_invalid'])
def test_braking_without_control_confirmation_has_bounded_latched_failure(failure):
    flow = sequence()
    frozen_plan(flow, forward=500, backward=300)
    flow.state = MatchState.BREAKUP_FORWARD
    deadline = None
    for ms in range(0, 3001, 5):
        now = ms*1_000_000
        if ms == 0 or failure != 'telemetry_gap':
            sample = motion_sample(now, count=5000+ms)
            if failure == 'sensor_invalid':
                sample = replace(sample, sensor_flags=SensorFlags.LEFT_ENCODER_VALID)
            flow.observe_grasp_motion(sample)
        decision = flow.step(now, perception=None, heading_rad=0,
            cumulative_distance_m=.51+ms*.0001,
            left_speed_feedback_m_s=.1, right_speed_feedback_m_s=.1)
        assert decision.linear_velocity_m_s == decision.angular_velocity_rad_s == 0
        assert decision.soft_brake
        if deadline is None:
            deadline = flow._breakup_segment_stop_deadline_ns
        assert flow._breakup_segment_stop_deadline_ns == deadline
        if flow.state is MatchState.TERMINAL_STOP:
            break
    assert now <= deadline+5_000_000
    assert flow.state is MatchState.TERMINAL_STOP
    assert decision.reason.startswith('breakup_stop_unconfirmed:')
    assert 'deadline_ns=' in decision.reason
    again = flow.step(now+5_000_000, perception=None, heading_rad=0,
                      cumulative_distance_m=.1)
    assert again.state is MatchState.TERMINAL_STOP
    assert again.linear_velocity_m_s == 0
