"""1932 recording: committed pickup ownership and evidence/scan timing."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.gripper_width_sequence import GraspPreparation
from rescue_vision.app.match import MatchState
from rescue_vision.app.near_field_grasp import GraspSelection, NearFieldHandoffPrior
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from test_gripper_width_sequence import motion_sample
from test_match import snapshot
from test_match_breakup import frozen_plan
from test_match_near_field import _sequence, _preparation
from test_near_field_grasp import selector, target
from test_match_safe_zone_latency import setup_sequence, delayed_pair


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250), (10,600,400)])
@pytest.mark.parametrize('late_kind', ['recovery', 'exit', 'original_plan'])
def test_committed_refill_finishes_despite_late_planner_and_stowed_target(
    poll_ms, delay_ms, period_ms, late_kind,
):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    seq._greedy_active = True
    seq._transport_target_classes = (TargetClass.BLACK_CORE,)
    capture_ns = 100_000_000
    published_ns = capture_ns + delay_ms * 1_000_000
    plan = selector().select((target(cls=TargetClass.BLACK_CORE, timestamp=capture_ns),)).plan
    assert plan is not None
    prep = replace(_preparation(plan, timestamp_ns=capture_ns), prepared_timestamp_ns=published_ns)
    recovery = replace(prep, selection=GraspSelection(None, ('blocked_target:3:blue_danger',)),
                       recovery_plan=frozen_plan(seq), recovery_checked=True)
    empty = replace(prep, selection=GraspSelection(None, ()), recovery_checked=True)
    late = {'recovery': recovery, 'exit': empty, 'original_plan': prep}[late_kind]
    distance = 0.0
    advanced = False
    crossed_stowed_limit = False
    scene = None
    for ms in range(0, delay_ms + 1600, poll_ms):
        now = ms * 1_000_000
        if ms % 10 == 0:
            seq.observe_grasp_motion(motion_sample(now, count=round(distance * 10000)))
        seq._last_timestamp_ns = now
        seq._latest_cumulative_distance_m = distance
        seq._record_pose_history(now, 0.0, distance)
        if now < published_ns:
            continue
        # Camera continues independently; peripheral IDs/detections flicker.
        if (ms-delay_ms-100) % period_ms == 0:
            items = [target(cls=TargetClass.BLACK_CORE, x=300-distance*1000,
                            timestamp=now-delay_ms*1_000_000).observation]
            if ms % (2*period_ms):
                items.append(target(99, x=500, y=350, timestamp=items[0].capture_timestamp_ns).observation)
            scene = replace(snapshot(ms+1, items[0].capture_timestamp_ns, *items), result_timestamp_ns=now, timing=None)
        seq._latest_perception = scene
        prepared = prep if now == published_ns else late
        decision = seq._step_near_field_grasp(now, distance, prepared, True)
        assert 'carried_or_delivered' not in decision.reason
        assert 'recovery_waiting' not in decision.reason
        if decision.linear_velocity_m_s > 0:
            advanced = True
            distance = min(plan.forward_distance_mm/1000, distance + poll_ms/1000 * .35)
        if 300-distance*1000 <= seq._greedy_new_target_min_x_mm():
            crossed_stowed_limit = True
        if decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE:
            break
    else:
        pytest.fail(decision.reason)
    assert advanced and crossed_stowed_limit
    assert seq.carried_target_count == 2
    assert decision.reason == 'greedy_return:supplementary_pickup_complete'
    # No new scene is required to finish the already issued close command.
    assert seq._counted_pickup_session == 1


def test_evidence_timeout_preserves_target_and_deadline_without_reusing_failed_frame():
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq._begin_greedy_scan(0)
    capture = 100_000_000
    item = target(x=300, timestamp=capture, cls=TargetClass.BLACK_CORE)
    seq._last_timestamp_ns = capture
    seq._record_pose_history(capture, 0.0, 0.0)
    seq._latest_perception = snapshot(1, capture, item.observation)
    tracked, = seq._tracker.update(capture, (item.observation,))
    seq._begin_green_transport(capture, tracked, group_preview=True)
    task = seq._grasp_task
    assert task is not None
    deadline = task.deadline_ns
    seq._begin_near_field_grasp(capture, handoff_prior=NearFieldHandoffPrior(
        TargetClass.BLACK_CORE, GroundPoint(300,0), tracked.track_id))
    decision = seq._finish_greedy_pickup(900_000_000, 'scene_evidence_unavailable')
    assert decision.state is MatchState.TRANSPORT_GREEDY_SCAN
    assert decision.angular_velocity_rad_s != 0
    assert not seq._near_field_failures
    assert task.deadline_ns == deadline
    assert 'physical_failure_recorded=false' in seq.near_field_last_failure_diagnostic
    # Next physical observation is immediately eligible, even with a new ID.
    now = 950_000_000
    seq._last_timestamp_ns = now
    seq._record_pose_history(now, 0.0, 0.0)
    item = target(x=300, timestamp=now, cls=TargetClass.BLACK_CORE)
    seq._latest_perception = snapshot(2, now, item.observation)
    tracked, = seq._tracker.update(now, (item.observation,))
    assert not seq._target_attempt_blocked(replace(tracked, track_id=999), now)
    decision = seq._step_greedy_scan(now, 0.0)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert seq._grasp_task is task
    assert task.deadline_ns == deadline


def test_greedy_timeout_counts_scan_not_candidate_processing(monkeypatch):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq._begin_greedy_scan(0)
    monkeypatch.setattr(seq, '_find_approach_seed', lambda *args, **kwargs: None)
    seq._step_greedy_scan(0, 0.0)
    seq._step_greedy_scan(100_000_000, 0.0)
    seq._begin_near_field_grasp(100_000_000, handoff_prior=NearFieldHandoffPrior(
        TargetClass.BLACK_CORE, GroundPoint(300,0), 7))
    decision = seq._finish_greedy_pickup(10_000_000_000, 'scene_evidence_unavailable')
    assert decision.state is MatchState.TRANSPORT_GREEDY_SCAN
    assert seq._greedy_scan_elapsed_ns == 100_000_000
    assert seq._greedy_started_ns == 0
    decision = seq._step_greedy_scan(100_000_000_000, 0.0)
    assert decision.reason == 'greedy_return:scan_timeout'


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250), (10,600,400)])
def test_d1_visual_corrects_large_odometry_drift_before_d2(poll_ms, delay_ms, period_ms):
    seq = setup_sequence()
    seq._fallback_field_position = FieldPoint(350, 700)
    latest = None
    for ms in range(0, 2500, poll_ms):
        now = ms*1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            latest = delayed_pair(ms//period_ms+1, (ms-delay_ms)*1_000_000, delay_ms)
        if latest is None:
            continue
        decision = seq.step(now, perception=latest, heading_rad=math.radians(60),
                            cumulative_distance_m=0, left_speed_feedback_m_s=0,
                            right_speed_feedback_m_s=0)
        if seq.safe_zone_calibration_pose is not None:
            break
    assert decision.reason == 'safe_zone_visual_calibrated_start_d2_line'
    assert seq.estimated_field_position.x == pytest.approx(0)
    assert seq.estimated_field_position.y == pytest.approx(700)
    assert seq.estimated_field_heading_rad == pytest.approx(math.pi/2)
    assert seq._safe_zone_d2_target().x < 0


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250), (10,600,400)])
def test_delayed_ready_plan_cannot_ignore_current_danger(poll_ms, delay_ms, period_ms):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    capture = 100_000_000
    arrival = capture + delay_ms*1_000_000
    member = target(timestamp=capture)
    plan = selector().select((member,)).plan
    assert plan is not None
    prepared = replace(_preparation(plan, timestamp_ns=capture), prepared_timestamp_ns=arrival)
    for ms in range(0, delay_ms+100+period_ms, poll_ms):
        now = ms*1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if now < arrival:
            continue
        # The async result is ready, but a newer camera frame shows intrusion.
        fresh_capture = capture + 50_000_000
        danger = target(2, x=250, cls=TargetClass.BLUE_DANGER, timestamp=fresh_capture)
        visible = target(timestamp=fresh_capture)
        seq._latest_perception = replace(snapshot(2, fresh_capture,
            visible.observation, danger.observation), result_timestamp_ns=now, timing=None)
        seq._step_near_field_grasp(now, 0, prepared, True)
        assert seq.near_field_active_plan is None


def test_repeated_evidence_failure_keeps_original_attempt_deadline():
    seq = _sequence(transports=1)
    seq._started = True
    seq._latest_heading_rad = 0.0
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    seq._begin_greedy_scan(0)
    seq._adopt_grasp_task(1, track_id=7, target_class=TargetClass.BLACK_CORE,
                         point=GroundPoint(300,0), field=FieldPoint(300,0))
    task = seq.grasp_task
    assert task is not None
    deadline = task.deadline_ns
    for now in (100_000_000, 1_000_000_000):
        seq._begin_near_field_grasp(now, handoff_prior=NearFieldHandoffPrior(
            TargetClass.BLACK_CORE, GroundPoint(300,0), 7))
        seq._finish_greedy_pickup(now+100_000_000, 'scene_evidence_unavailable')
        assert seq.grasp_task is task
        assert task.deadline_ns == deadline
        assert not seq._near_field_failures
    seq._begin_near_field_grasp(deadline-1, handoff_prior=NearFieldHandoffPrior(
        TargetClass.BLACK_CORE, GroundPoint(300,0), 7))
    decision = seq.step(deadline, perception=None, heading_rad=0, cumulative_distance_m=0)
    assert decision.state is not MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert seq.grasp_task is None
    assert decision.reason == 'greedy_return:pickup_attempt_timeout'
    assert not seq._near_field_failures


@pytest.mark.parametrize('rejection', ['blocked_target:4:blue_danger',
                                     'blocked_target:5:green_supply',
                                     'incidental_capacity_exceeded'])
def test_loaded_physical_rejection_is_not_reported_as_pending_evidence(rejection):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.0
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)*2
    seq._begin_greedy_scan(0)
    seq._begin_near_field_grasp(1, handoff_prior=NearFieldHandoffPrior(
        TargetClass.BLACK_CORE, GroundPoint(300,0), 7))
    prepared = GraspPreparation(10, GraspSelection(None, (rejection,)), (),
        session_id=seq.near_field_session_id, prepared_timestamp_ns=20)
    decision = seq._step_near_field_grasp(20, 0, prepared, True)
    assert decision.state is MatchState.TRANSPORT_GREEDY_SCAN
    assert decision.angular_velocity_rad_s != 0
    assert any(f.reason == 'no_safe_supplementary_grasp' for f in seq._near_field_failures)
    assert seq.carried_target_count == 2


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250),(10,600,400)])
def test_alternating_greedy_candidates_cannot_renew_attempt_forever(poll_ms, delay_ms, period_ms):
    seq = _sequence(transports=1)
    seq._started = True
    seq._latest_heading_rad = 0.0
    seq._transport_target_classes = (TargetClass.GREEN_SUPPLY,)*2
    seq._begin_greedy_scan(0)
    deadline = round(seq.config.grasp_task_timeout_ms*1e6)
    for ms in range(0, int(deadline/1e6)+poll_ms, poll_ms):
        now = ms*1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if ms % period_ms == 0:
            index = ms//period_ms
            cls = TargetClass.BLACK_CORE if index % 2 else TargetClass.GREEN_SUPPLY
            point = GroundPoint(210 if index % 2 else 240, 20)
            seq._grasp_task = None
            seq._adopt_grasp_task(now, track_id=index+1, target_class=cls,
                                 point=point, field=FieldPoint(point.x,point.y))
            assert seq.grasp_task.deadline_ns == deadline
            seq._begin_near_field_grasp(now, handoff_prior=NearFieldHandoffPrior(cls,point,index+1))
        decision = seq.step(now, perception=None, heading_rad=0, cumulative_distance_m=0)
        if not seq._greedy_active:
            break
    assert ms <= deadline/1e6
    assert not seq._greedy_active
    assert decision.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert seq.carried_target_count == 2
