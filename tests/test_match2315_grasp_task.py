"""Production preparation contracts: no hand-made ready results or lock handshake."""
from __future__ import annotations

from dataclasses import replace

import pytest

from rescue_vision.app.gripper_width_sequence import GraspPreparationSession, GripperWidthPickupState
from rescue_vision.app.near_field_grasp import GraspTargetTracker, NearFieldHandoffPrior
from rescue_vision.motion.stationary import StationaryMotionEvidence
from rescue_vision.tracking import TrackingConfig
from test_gripper_width_sequence import motion_sample, sequence, snapshot_at
from test_near_field_grasp import BLUE, GREEN, projector, selector, target


@pytest.mark.parametrize('poll_ms', [5, 10])
@pytest.mark.parametrize('delay_ms', [300, 600, 800, 1100, 1500])
def test_one_stable_scene_opens_without_consumer_lock_or_new_frame(poll_ms, delay_ms):
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(),
                           projector(), planner.config), planner,
    )
    pickup = sequence(max_observation_age_ms=800)
    capture_ns = 100_000_000
    published_ns = capture_ns + delay_ms * 1_000_000
    scene = snapshot_at(1, capture_ns, (target(),))
    prepared = None
    for now in range(0, published_ns + poll_ms * 1_000_000, poll_ms * 1_000_000):
        pickup.observe_motion(motion_sample(now))
        if now == published_ns:
            prepared = session.update(scene, locked_ids=None,
                                      handoff_prior=NearFieldHandoffPrior(GREEN, target().observation.ground_point, 52))
            prepared = replace(prepared, prepared_timestamp_ns=now)
            assert prepared.ready
            assert prepared.confirmation_progress == (1, 1)
            assert session.update(scene, locked_ids=None) is not None
        decision = pickup.step(now, prepared, cumulative_distance_m=0)
        if decision.gripper_angles_deg is not None:
            break
    assert decision.state is GripperWidthPickupState.OPENING
    assert decision.timestamp_ns == published_ns
    forward = pickup.step(published_ns + pickup.travel_ns, None, cumulative_distance_m=0)
    assert forward.linear_velocity_m_s > 0


def test_illegal_enlargement_does_not_hold_legal_singleton():
    seed = target(x=300, y=0)
    peer = replace(target(2, x=300, y=70), confirmed=False)
    danger = target(3, x=320, y=120, cls=BLUE)
    result = selector().select((seed, peer, danger))
    assert result.plan is not None
    assert result.plan.member_ids == (1,)
    confirmed = selector().select((seed, replace(peer, confirmed=True), danger))
    assert confirmed.plan is not None and confirmed.plan.member_ids == (1,)


@pytest.mark.parametrize('kind,reason', [('wheel', 'encoder_motion'), ('gyro', 'rotation'),
                                       ('duplicate', 'duplicate_or_reversed_device_sample')])
def test_stationary_diagnostics_explain_actual_invalidation(kind, reason):
    evidence = StationaryMotionEvidence(max_gap_ns=150_000_000, max_gyro_rad_s=.03)
    evidence.observe(motion_sample(0))
    evidence.observe(motion_sample(10_000_000))
    assert evidence.stationary_since(10_000_000) == 10_000_000
    sample = motion_sample(20_000_000, count=1 if kind == 'wheel' else 0,
                           gyro=100_000 if kind == 'gyro' else 0)
    if kind == 'duplicate':
        sample = motion_sample(10_000_000)
    evidence.observe(sample)
    assert evidence.stationary_since(20_000_000) is None
    assert f'stationary_reason={reason}' in evidence.diagnostic(20_000_000)


def test_stable_scene_returns_grasp_obstruction_and_recovery_together():
    from rescue_vision.app.breakup_planner import BreakupSceneContext
    from rescue_vision.app.gripper_width_sequence import GraspSceneAction
    from rescue_vision.config import load_runtime_config
    from rescue_vision.geometry.types import FieldPoint
    config = load_runtime_config('configs/runtime.match.yaml')
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(),
                           projector(), planner.config), planner,
    )
    context = BreakupSceneContext(replace(config.match, breakup_confirmation_frames=1),
                                 FieldPoint(0, 0), 0, config.world.static_map,
                                 (-1500, 1500, -1500, 1500), 110.15)
    scene = snapshot_at(1, 100_000_000, (target(x=350), target(2, x=320, cls=BLUE)))
    prepared = session.update(scene, locked_ids=None, recovery_context=context,
                              handoff_prior=NearFieldHandoffPrior(GREEN, target(x=350).observation.ground_point, 52))
    assert prepared.selection.plan is None
    assert any(reason.startswith('blocked_target:') for reason in prepared.selection.rejections)
    assert prepared.action is GraspSceneAction.RECOVERY
    assert prepared.ready
    assert prepared.recovery_plan.capture_timestamp_ns == scene.capture_timestamp_ns
    assert prepared.confirmation_progress == (1, 1)
    assert session.update(scene, locked_ids=None, recovery_context=context) is prepared


@pytest.mark.parametrize('poll_ms,frame_ms,delay_ms', [(5,250,300),(10,400,600)])
def test_search_recovery_and_grasp_keep_one_task_until_transport(poll_ms, frame_ms, delay_ms):
    from rescue_vision.app.match import GraspRoute, MatchState
    from test_match_near_field import _sequence
    from rescue_vision.config import load_runtime_config
    flow = _sequence()
    flow.config = replace(flow.config,
                          green_max_age_ms=load_runtime_config('configs/runtime.match.yaml').match.green_max_age_ms,
                          opportunistic_single_green_enabled=True,
                          breakup_confirmation_frames=1,
                          safe_zone_calibration_stop_confirm_time_s=.01,
                          grasp_task_timeout_ms=20_000)
    flow._started = True
    flow.state = MatchState.SEARCH_CLUSTER
    flow._near_field_route = GraspRoute.DECIDING
    planner = selector(confirmation_frames=1)
    session = None
    session_id = -1
    latest = prepared = None
    distance = 0.0
    peak_distance = 0.0
    task_identity = None
    deadline = None
    recovering = recovered = opened = False
    retreat_distance = None
    queued = []
    states = set()
    last = None
    for ms in range(0, 18_000, poll_ms):
        now = ms * 1_000_000
        if last is not None:
            distance += last.linear_velocity_m_s * poll_ms / 1000
        peak_distance = max(peak_distance, distance)
        flow.observe_grasp_motion(motion_sample(now, count=round(distance*10_000)))
        if ms % frame_ms == 0:
            # Contact moves the green block forward; retreat exposes that same
            # block at its measured new position, before the handoff completes.
            shift_mm = max(0.0, peak_distance*1000-220)
            members = (target(1, x=350+shift_mm-distance*1000),
                       target(2, x=320+shift_mm-distance*1000,
                              y=350 if shift_mm > 0 else 0, cls=BLUE))
            queued.append((ms+delay_ms, snapshot_at(ms//frame_ms+1, now, members)))
        while queued and queued[0][0] <= ms:
            _, latest = queued.pop(0)
            latest = replace(latest, result_timestamp_ns=now, timing=None)
        if flow.near_field_session_id != session_id:
            session_id = flow.near_field_session_id
            session = GraspPreparationSession(
                GraspTargetTracker(TrackingConfig(1,80,.1,500,1,.1).build_tracker(),
                                   projector(), planner.config), planner)
            prepared = None
        if (flow.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP and latest is not None
                and flow.near_field_observation_window_open(now)
                and flow.grasp_scene_capture_valid(latest, now)
                and flow.near_field_active_plan is None):
            result = session.update(latest, locked_ids=flow.near_field_locked_ids,
                                    session_id=session_id, policy=flow.near_field_policy,
                                    handoff_prior=flow.near_field_handoff_prior,
                                    require_handoff=flow.near_field_handoff_required,
                                    recovery_context=flow.grasp_recovery_context(latest))
            if prepared is None or result.capture_timestamp_ns != prepared.capture_timestamp_ns:
                prepared = replace(result, prepared_timestamp_ns=now)
        clear = None
        if prepared is not None and prepared.selection.plan is not None:
            clear = flow.near_field_plan_path_clear(prepared.selection.plan, 0)
        last = flow.step(now, perception=latest, heading_rad=0, cumulative_distance_m=distance,
                         left_speed_feedback_m_s=0 if last is None else last.linear_velocity_m_s,
                         right_speed_feedback_m_s=0 if last is None else last.linear_velocity_m_s,
                         near_field_preparation=prepared, near_field_path_clear=clear)
        states.add(flow.state)
        if flow.state in {MatchState.BREAKUP_SETTLE, MatchState.BREAKUP_FORWARD,
                          MatchState.BREAKUP_BACKWARD, MatchState.CHECK_ISOLATED_GREEN}:
            assert last.gripper_posture.value == 'closed'
        if flow.state is MatchState.BREAKUP_BACKWARD and retreat_distance is None:
            retreat_distance = distance
            assert flow._breakup_actual_forward_mm == pytest.approx(500, abs=1)
        if flow.grasp_task is not None:
            if task_identity is None:
                task_identity, deadline = flow.grasp_task.task_id, flow.grasp_task.deadline_ns
            assert flow.grasp_task.task_id == task_identity, last.reason
            assert flow.grasp_task.deadline_ns == deadline
        recovering |= flow.state is MatchState.BREAKUP_FORWARD
        if recovering and last.reason == 'breakup_complete_replan_grasp_task':
            recovered = True
            assert retreat_distance-distance == pytest.approx(.3, abs=.001)
            assert flow.near_field_handoff_required
            assert flow.grasp_task.marked_targets
        if recovered and flow.near_field_active_plan is not None:
            opened = True
        if opened and flow.grasp_task is None and flow._transport_target_classes:
            break
    assert recovering, (last, states)
    assert recovered, (last, states)
    assert opened, (last, states)
    assert flow.grasp_task is None, (last, states)
    assert flow._transport_target_classes == (GREEN,)


@pytest.mark.parametrize('risk', ['danger', 'missing_core', 'danger_without_k0'])
def test_late_grasp_result_cannot_override_newer_risk(risk):
    from test_match_near_field import _sequence, _preparation
    from rescue_vision.app.match import MatchState
    flow = _sequence()
    flow._started = True
    capture, now = 100_000_000, 1_100_000_000
    plan = selector().select((target(timestamp=capture, frame=1),)).plan
    for stamp in range(0, now+1, 10_000_000):
        flow.observe_grasp_motion(motion_sample(stamp))
    members = [] if risk == 'missing_core' else [target()]
    if risk != 'missing_core':
        danger = target(2, x=300, cls=BLUE)
        if risk == 'danger_without_k0':
            danger = replace(danger, observation=replace(danger.observation, ground_point=None))
        members.append(danger)
    newer = snapshot_at(2, 400_000_000, members)
    preparation = replace(_preparation(plan, timestamp_ns=capture), prepared_timestamp_ns=now)
    decision = flow.step(now, perception=newer, heading_rad=0, cumulative_distance_m=0,
                         near_field_preparation=preparation, near_field_path_clear=True)
    assert flow.near_field_active_plan is None
    assert decision.gripper_angles_deg is None
    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP


def test_failure_records_entry_and_core_once_even_if_preview_and_local_ids_differ():
    from test_match_near_field import _sequence, _preparation
    from rescue_vision.geometry.types import FieldPoint, GroundPoint
    flow = _sequence()
    flow._latest_heading_rad = 0
    flow._adopt_grasp_task(0, track_id=52, target_class=GREEN,
                           point=GroundPoint(300,0), field=FieldPoint(300,0))
    task = flow.grasp_task
    deadline = task.deadline_ns
    plan = selector().select((target(7, x=400, y=0),)).plan
    task.core = plan.members
    flow._remember_near_field_failure(100, 'route_blocked', _preparation(plan))
    flow._remember_near_field_failure(101, 'confirmation_timeout', _preparation(plan))
    assert len(flow._near_field_failures) == 1
    failure = flow._near_field_failures[0]
    assert failure.track_id == 52
    assert failure.field_point == FieldPoint(300,0)
    assert FieldPoint(300,0) in failure.region_field_points
    flow._tracker.update(102, [target(99,x=300,frame=2,timestamp=102).observation])
    changed_id = flow._tracker.tracks[0]
    assert flow._target_attempt_blocked(changed_id, 102)
    flow._adopt_grasp_task(103, track_id=99, target_class=GREEN,
                           point=GroundPoint(300,0), field=FieldPoint(300,0))
    assert flow.grasp_task is task
    assert task.deadline_ns == deadline


def test_loaded_task_has_no_recovery_capability():
    from test_match_near_field import _sequence
    flow = _sequence(transports=1)
    flow._transport_target_classes = (GREEN,)
    flow._greedy_active = True
    scene = snapshot_at(1, 10, (target(),))
    assert flow.grasp_recovery_context(scene) is None


def test_no_plan_and_computing_result_use_separate_bounded_windows():
    from test_match_near_field import _sequence
    flow = _sequence()
    flow._near_field_confirmation_started_ns = 0
    no_plan_ns = round(flow._near_field_no_plan_wait_ms()*1e6)
    flow.grasp_planning_pending = True
    assert flow._near_field_route_decision(no_plan_ns, None) is None
    flow.grasp_planning_pending = False
    assert flow._near_field_route_decision(no_plan_ns+1, None) is not None


def test_async_scene_cancellation_does_not_wait_and_old_result_cannot_publish():
    from threading import Event
    import time
    from rescue_vision.app.gripper_width_sequence import GraspPreparationWorker
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(TrackingConfig(1,80,.1,500,1,.1).build_tracker(),
                           projector(), planner.config), planner)
    entered, release, processed = Event(), Event(), Event()
    update = session.update

    def slow_update(scene, **kwargs):
        if kwargs['session_id'] == 1:
            entered.set()
            assert release.wait(2)
        result = update(scene, **kwargs)
        if kwargs['session_id'] == 2:
            processed.set()
        return result

    session.update = slow_update
    with GraspPreparationWorker(session, planner) as worker:
        worker.begin(1)
        try:
            worker.submit(snapshot_at(1, 100, (target(),)), session_id=1,
                          policy=planner.default_policy, locked_ids=None)
            assert entered.wait(1)
            assert worker.pending(1)
            # This completes while the previous computation is still blocked.
            worker.begin(2)
            assert not worker.pending(1)
            assert worker.latest(1) is None
            worker.submit(snapshot_at(2, 200, (target(x=330),)), session_id=2,
                          policy=planner.default_policy, locked_ids=None)
            release.set()
            assert processed.wait(1)
            deadline = time.monotonic()+1
            while worker.latest(2) is None and time.monotonic() < deadline:
                time.sleep(.005)
            result = worker.latest(2)
            assert result is not None and result.ready
            assert result.selection.plan.capture_timestamp_ns == 200
            assert worker.latest(1) is None
        finally:
            release.set()


def test_observation_budget_uses_measured_delivery_and_does_not_grow_on_poll():
    from test_match_near_field import _sequence
    from rescue_vision.app.match import GraspRoute
    flow = _sequence()
    flow._near_field_route = GraspRoute.DECIDING
    flow._perception_interval_ns = 400_000_000
    flow._latest_perception = replace(snapshot_at(1, 100_000_000, (target(),)),
                                      result_timestamp_ns=700_000_000, timing=None)
    for stamp in range(0, 700_000_001, 10_000_000):
        flow.observe_grasp_motion(motion_sample(stamp))
    assert flow.near_field_observation_window_open(700_000_000)
    budget = flow._near_field_no_plan_wait_ms()
    assert budget == 1400
    assert budget < flow._near_field_handoff_timeout_ms()
    flow._perception_interval_ns = 800_000_000
    flow.near_field_observation_window_open(710_000_000)
    assert flow._near_field_no_plan_wait_ms() == budget


def test_approach_displacement_is_not_read_as_target_replacement():
    """接近位移不能把同一物理目标误判成可执行核心替换。

    2315 场景：任务在远场按机器人系 (700,0) 采纳，接近 400 mm 后停稳，
    同一目标机器人系变为 (300,0) 而场地坐标不变。旧实现按机器人系
    100 mm 判距，必然记录 entry_replaced_by_executable_core 并改写入口。
    """

    import math

    from test_match_near_field import _preparation, _sequence
    from rescue_vision.geometry.types import FieldPoint, GroundPoint
    flow = _sequence()
    flow._started = True
    flow._latest_heading_rad = 0.0
    flow._record_pose_history(0, 0.0, 0.0)
    flow._adopt_grasp_task(0, track_id=52, target_class=GREEN,
                           point=GroundPoint(700.0, 0.0), field=FieldPoint(700.0, 0.0))
    task = flow.grasp_task
    deadline = task.deadline_ns
    # 接近后停稳：机器人场地位置前进 400 mm，目标场地坐标保持 (700,0)。
    flow._fallback_field_position = FieldPoint(400.0, 0.0)
    flow._record_pose_history(900_000_000, 0.0, 0.4)
    capture, now = 900_000_000, 1_000_000_000
    member = target(timestamp=capture, frame=1, x=300.0, y=0.0)
    plan = selector().select((member,)).plan
    assert plan is not None
    for stamp in range(0, now + 1, 10_000_000):
        flow.observe_grasp_motion(motion_sample(stamp))
    preparation = _preparation(plan, timestamp_ns=capture)
    flow.step(now, perception=snapshot_at(1, capture, (member,)),
              heading_rad=0.0, cumulative_distance_m=0.4,
              near_field_preparation=preparation, near_field_path_clear=True)
    assert all(record.reason != 'entry_replaced_by_executable_core'
               for record in flow._near_field_failures)
    assert flow.grasp_task is task
    assert task.deadline_ns == deadline
    assert task.core and task.core[0].track_id == member.track_id
    # 入口锚点跟随当前几何：机器人系刷新为停稳后的观测，场地坐标不变。
    assert task.entry_ground == GroundPoint(300.0, 0.0)
    assert task.entry_field is not None
    assert math.hypot(task.entry_field.x - 700.0, task.entry_field.y) <= 1e-6


def test_breakup_push_keeps_entry_through_tracker_continuity():
    """解团推移换局部 ID 后按场地坐标和主轨迹关联同一物理目标。

    推移 160 mm 超过入口场地锚点的身份容差；主 tracker 仍保持同一轨迹
    时，成员与入口轨迹的当前场地位置一致，不得记为换目标。入口锚点
    跟随推移后的几何，使后续位移按新位置累计。
    """

    import math

    from rescue_vision.tracking import TrackingConfig
    from test_match_near_field import _preparation, _sequence
    from rescue_vision.geometry.types import FieldPoint, GroundPoint
    flow = _sequence()
    flow._started = True
    flow._latest_heading_rad = 0.0
    flow._record_pose_history(0, 0.0, 0.0)
    # 生产主 tracker 的关联门限为 250 mm（runtime.match.yaml 不覆盖 tracking
    # 默认值）；推移 160 mm 时主轨迹保持同 ID。
    flow._tracker = TrackingConfig(1, 250.0, 0.1, 500.0, 1.0, 0.1).build_tracker()
    first = target(1, x=250.0, y=0.0, timestamp=0, frame=1)
    flow._tracker.update(0, (first.observation,))
    track = flow._tracker.tracks[0]
    flow._adopt_grasp_task(0, track_id=track.track_id, target_class=GREEN,
                           point=GroundPoint(250.0, 0.0), field=FieldPoint(250.0, 0.0))
    task = flow.grasp_task
    deadline = task.deadline_ns
    # 解团把入口目标推移 160 mm（超过 100 mm 身份容差），主轨迹保持同 ID。
    pushed = target(1, x=410.0, y=0.0, timestamp=500_000_000, frame=2)
    flow._tracker.update(500_000_000, (pushed.observation,))
    assert [item.track_id for item in flow._tracker.tracks] == [track.track_id]
    member = target(7, x=410.0, y=0.0, timestamp=500_000_000, frame=3)
    plan = selector().select((member,)).plan
    assert plan is not None
    capture, now = 500_000_000, 600_000_000
    for stamp in range(0, now + 1, 10_000_000):
        flow.observe_grasp_motion(motion_sample(stamp))
    preparation = _preparation(plan, timestamp_ns=capture)
    flow.step(now, perception=snapshot_at(3, capture, (member,)),
              heading_rad=0.0, cumulative_distance_m=0.0,
              near_field_preparation=preparation, near_field_path_clear=True)
    assert all(record.reason != 'entry_replaced_by_executable_core'
               for record in flow._near_field_failures)
    assert flow.grasp_task is task
    assert task.deadline_ns == deadline
    assert task.core and task.core[0].track_id == member.track_id
    assert task.entry_field is not None
    assert math.hypot(task.entry_field.x - 410.0, task.entry_field.y) <= 1e-6
    assert task.entry_track_id == track.track_id


def test_selected_handoff_prior_compensates_approach_motion():
    """交接先验按采集位姿补偿，停车后不因接近位移失配。"""

    from test_match_near_field import _sequence
    from rescue_vision.geometry.types import FieldPoint
    flow = _sequence()
    flow._started = True
    flow._latest_heading_rad = 0.0
    flow._record_pose_history(0, 0.0, 0.0)
    far = target(1, x=700.0, y=0.0, timestamp=0, frame=1)
    flow._tracker.update(0, (far.observation,))
    flow._selected_track_id = flow._tracker.tracks[0].track_id
    # 机器人前进 300 mm；先验必须换算到当前机器人系 (400,0)。
    flow._fallback_field_position = FieldPoint(300.0, 0.0)
    flow._record_pose_history(400_000_000, 0.0, 0.3)
    prior = flow._selected_handoff_prior(400_000_000)
    assert prior is not None
    assert prior.ground_point.x == pytest.approx(400.0)
    assert prior.ground_point.y == pytest.approx(0.0)
    assert prior.source_track_id == flow._tracker.tracks[0].track_id
