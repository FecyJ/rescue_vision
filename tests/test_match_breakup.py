"""Dynamic breakup regressions with frame/encoder/stop evidence; no hardware."""
from dataclasses import replace
import math

import pytest

from rescue_vision.app.breakup_planner import BreakupPlan
from rescue_vision.app.match import MatchState, GripperPosture
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from test_match import make_sequence, runtime_config, snapshot, observation


def sequence(*, initial_field_position=FieldPoint(0.0, 0.0)):
    cfg = load_runtime_config('configs/runtime.match.yaml').match
    # 这些用例验证的是「不同采集帧累积确认」这套机制本身，因此固定 3 帧
    # 前置条件，不跟随生产配置（生产用 1 帧，见 runtime.match.yaml）。
    seq = make_sequence(config=replace(cfg, cluster_align_hold_ms=3000,
        breakup_confirmation_frames=3,
        safe_zone_calibration_stop_confirm_time_s=0.01, opportunistic_single_green_enabled=False),
        initial_field_position=initial_field_position)
    seq._started = True
    seq.state = MatchState.SEARCH_CLUSTER
    seq._breakup_only = True
    return seq


def tick(seq, frame, ms, *, distance=0., speed=0., observations=None):
    ns = int(ms*1e6)
    if observations is None:
        observations = (observation(frame, ns, GroundPoint(450-distance*1000, 0)),
                        observation(frame, ns, GroundPoint(480-distance*1000, 0), target_class=TargetClass.BLUE_DANGER, box_x=40))
    return seq.step(ns, perception=snapshot(frame, ns, *observations), heading_rad=0,
                    cumulative_distance_m=distance, left_speed_feedback_m_s=speed, right_speed_feedback_m_s=speed)


def frozen_plan(seq, *, forward=400., approach=100., backward=400., heading=0.):
    """Explicit plan injection tests execution guards independently of planner rejection."""
    p = BreakupPlan((1,2), (1,2), 1, GroundPoint(450,0), heading, approach, forward,
                    backward, 80, 0, 1, (FieldPoint(450,0),FieldPoint(480,0)), FieldPoint(450,0),160)
    seq._breakup_plan = p
    seq._breakup_forward_base_distance_m = seq._breakup_backward_base_distance_m = 0.
    seq._cluster_approach_base_distance_m = 0.
    seq._breakup_retreat_mm = backward
    return p


def test_current_contact_core_freezes_after_three_distinct_frames():
    seq = sequence()
    tick(seq, 1, 10)
    detected = tick(seq, 2, 20)
    assert detected.state is MatchState.BREAKUP_SETTLE
    assert detected.reason == "cluster_seen_stop_collect_reference"
    for frame, ms in ((3, 30), (4, 50), (5, 60)):
        decision = tick(seq, frame, ms)
    assert decision.state is MatchState.BREAKUP_FORWARD
    assert decision.reason == "breakup_plan_frozen"
    assert len(seq._breakup_reference_frames) == seq.config.breakup_confirmation_frames
    assert seq._breakup_plan is not None
    assert seq._breakup_plan.approach_distance_mm == 0.0
    assert seq._breakup_plan.capture_timestamp_ns == 60_000_000


def test_cluster_inside_the_safe_zone_is_not_a_breakup_target():
    """已放入安全区的物资不能再被选中去接近或解团。"""

    def observe_and_plan(y_mm):
        seq = sequence(initial_field_position=FieldPoint(0.0, 980.0))
        seq._latest_heading_rad = 0.0
        for frame, ms in ((1, 10), (2, 20), (3, 30)):
            ns = int(ms * 1e6)
            seq.step(
                ns,
                perception=snapshot(
                    frame, ns,
                    observation(frame, ns, GroundPoint(250.0, y_mm)),
                    observation(frame, ns, GroundPoint(300.0, y_mm), box_x=40.0),
                ),
                heading_rad=0.0,
                cumulative_distance_m=0.0,
            )
        return seq

    # 车在场地 (0,980) 朝 +x：机器人系 y=320 对应场地 y=1300，落在红色安全区。
    delivered = observe_and_plan(320.0)
    assert delivered._choose_breakup_plan(int(30e6)) is None
    assert delivered._last_cluster_rejection_reason == (
        "breakup_no_contact_plan_or_retry_exhausted"
    )
    # 安全区内成员在成组前就被排除，不会留下"距离/净空"类拒绝。
    assert delivered._breakup_plan_rejections == ()

    # 同一对物资在安全区外（场地 y=880）仍可解团。
    reachable = observe_and_plan(-100.0)
    assert reachable._choose_breakup_plan(int(30e6)) is not None


def test_stationary_interval_relatch_without_displacement_keeps_confirmation():
    """设备样本抖动只重新落定静止区间；机器人没动就不能清空确认进度。"""

    seq = sequence()
    for f in range(1, 4):
        tick(seq, f, f * 20)
    confirmed = set(seq._breakup_reference_frames)
    assert confirmed
    assert seq._breakup_confirm_pose == (0.0, 0.0)

    # 遥测层抖动：位姿不变，静止区间重新落定。
    tick(seq, 4, 100, speed=.1)
    assert seq._breakup_reference_frames == confirmed
    assert seq._breakup_confirm_pose == (0.0, 0.0)

    resumed = tick(seq, 5, 140)
    assert resumed.state is MatchState.BREAKUP_SETTLE
    assert confirmed <= seq._breakup_reference_frames


def test_attempt_without_contact_plan_leaves_after_the_reobserve_window():
    """没有合法接触计划时不能占满按确认帧数计的完整确认预算。"""

    seq = sequence()
    seq._start_breakup_attempt(0, None)
    assert seq.state is MatchState.BREAKUP_SETTLE
    # 单个物资达不到 cluster_min_detections，本次尝试不可能选出接触计划。
    window_ns = round(seq.config.breakup_no_plan_reobserve_ms * 1e6)
    full_budget_ns = (seq._cluster_align_hold_ns
                      + seq.config.breakup_confirmation_frames
                      * round(seq.config.green_max_age_ms * 1e6))
    assert full_budget_ns > 2 * window_ns

    def lone(frame, ms):
        return (observation(frame, int(ms * 1e6), GroundPoint(450.0, 0.0)),)

    tick(seq, 0, 0, observations=lone(0, 0))
    tick(seq, 1, 20, observations=lone(1, 20))
    inside = tick(seq, 2, window_ns / 1e6 / 2, observations=lone(2, 500))
    assert inside.state is MatchState.BREAKUP_SETTLE
    assert seq._breakup_proposal is None
    assert seq._breakup_plan_rejections
    assert "group_too_small_or_no_allowed_class" in seq._breakup_plan_rejections[0]

    expired = tick(seq, 3, window_ns / 1e6 + 200, observations=lone(3, 1200))
    assert expired.state is MatchState.SEARCH_CLUSTER
    assert expired.reason == "breakup_reference_timeout_reselect"
    assert "aim_id=none" in (seq.breakup_last_failure_diagnostic or "")
    assert "group_too_small_or_no_allowed_class" in (
        seq.breakup_last_failure_diagnostic or ""
    )


def test_attempt_with_contact_plan_keeps_the_full_confirmation_budget():
    """有计划时仍按确认帧数计预算，不被无计划重观测窗口提前截断。"""

    seq = sequence()
    plan = BreakupPlan((1, 2), (1, 2), 1, GroundPoint(450, 0), 0.0, 100.0, 400.0,
                       400.0, 80.0, 0, 1, (FieldPoint(450, 0), FieldPoint(480, 0)),
                       FieldPoint(450, 0), 160.0)
    seq._start_breakup_attempt(0, plan)
    assert seq._breakup_proposal is plan

    window_ms = seq.config.breakup_no_plan_reobserve_ms
    beyond = tick(seq, 1, window_ms + 200)
    assert beyond.state is MatchState.BREAKUP_SETTLE
    assert beyond.reason.startswith("breakup_reference:")


def test_motion_resets_confirmation_and_timeout_returns_search():
    seq = sequence()
    for f in range(1, 4):
        tick(seq, f, f * 20)
    assert seq._breakup_reference_frames
    # 真实位移才作废既有确认；机器人在首帧确认后移动过，几何不再可信。
    tick(seq, 4, 100, distance=.01, speed=.1)
    assert not seq._breakup_reference_frames
    # 预算按待确认帧数逐帧计，测试跟着配置算而不是写死毫秒数。
    budget_ms = (
        seq.config.cluster_align_hold_ms
        + seq.config.breakup_confirmation_frames * seq.config.green_max_age_ms
    )
    decision = tick(seq, 8, budget_ms + 200, speed=.1)
    assert decision.state is MatchState.SEARCH_CLUSTER
    assert 'timeout' in decision.reason


def test_unrelated_group_cannot_replace_confirming_core_or_extend_deadline():
    seq = sequence()
    for f in range(1, 4):
        tick(seq, f, f * 20)
    deadline_start = seq._breakup_phase_started_ns
    progress = len(seq._breakup_reference_frames)
    seq._tracker.reset()
    ns = 160_000_000
    tick(seq, 8, 160, observations=(observation(8, ns, GroundPoint(600, 50)),
                                    observation(8, ns, GroundPoint(650, 50),
                                               target_class=TargetClass.BLUE_DANGER, box_x=40)))
    assert len(seq._breakup_reference_frames) == progress
    assert not seq._breakup_reference_current
    assert seq._breakup_plan is None
    assert seq._breakup_phase_started_ns == deadline_start


def test_forward_stops_before_open_and_retreat_uses_actual_travel():
    seq=sequence(); frozen_plan(seq,forward=200)
    seq.state=MatchState.BREAKUP_FORWARD
    moving=tick(seq,1,10,distance=.05,speed=.1,observations=())
    slow=tick(seq,2,20,distance=.175,speed=.1,observations=())
    assert 0 < slow.linear_velocity_m_s < moving.linear_velocity_m_s
    braking=tick(seq,3,30,distance=.1995,speed=.1,observations=())
    assert braking.gripper_posture is GripperPosture.CLOSED
    assert braking.soft_brake
    tick(seq,4,40,distance=.200,observations=())
    opened=tick(seq,5,60,distance=.200,observations=())
    assert opened.state is MatchState.OPEN_GRIPPER_SETTLE
    assert opened.gripper_posture is GripperPosture.OPEN
    # 后退距离 = 实际前进行程(200 mm) + 退出净空 + 刹车余量，跟着配置算。
    assert seq._breakup_retreat_mm == pytest.approx(
        200.0 - (seq._breakup_plan.forward_distance_mm - seq._breakup_plan.penetration_mm)
        + seq.config.breakup_retreat_clearance_mm
        + seq.config.breakup_braking_margin_mm
    )
    assert len(seq._breakup_attempts)==1
    assert tick(seq,6,70,distance=.200,observations=()).state is MatchState.OPEN_GRIPPER_SETTLE
    assert tick(seq,7,1100,distance=.200,observations=()).state is MatchState.BREAKUP_BACKWARD


def test_retry_budget_is_only_consumed_by_completed_pushes():
    seq = sequence()
    for frame, ms in ((1, 10), (2, 20), (3, 30), (4, 40), (5, 50)):
        tick(seq, frame, ms)
    plan = seq._breakup_plan
    assert plan is not None
    # 预算按配置算，不写死次数。
    seq._breakup_attempts = [plan] * seq.config.breakup_max_attempts
    assert seq._choose_breakup_plan(50_000_000) is None
    seq._breakup_attempts = []
    assert seq._choose_breakup_plan(50_000_000) is not None


def test_guard_uses_dynamic_plan_not_configured_limit():
    seq=sequence(); seq._fallback_field_position=FieldPoint(0,500)
    seq.state=MatchState.BREAKUP_FORWARD
    frozen_plan(seq,forward=50,heading=math.pi/2)
    decision=seq.step(10,perception=None,heading_rad=math.pi/2,cumulative_distance_m=0.,
                      left_speed_feedback_m_s=0.,right_speed_feedback_m_s=0.)
    assert decision.linear_velocity_m_s > 0
    seq._breakup_plan=replace(seq._breakup_plan,forward_distance_mm=1200)
    blocked=seq.step(20,perception=None,heading_rad=math.pi/2,cumulative_distance_m=0.,
                     left_speed_feedback_m_s=0.,right_speed_feedback_m_s=0.)
    assert blocked.state is MatchState.SEARCH_CLUSTER
    assert blocked.linear_velocity_m_s==0


def test_breakup_plan_is_computed_at_most_once_per_input_frame():
    """重规划对同一输入帧最多执行一次，5 ms 控制周期里不重复跑边界搜索。"""

    seq = sequence()
    seq._latest_heading_rad = 0.0

    def feed(frame, ms):
        ns = int(ms * 1e6)
        snapshot_ = (observation(frame, ns, GroundPoint(450.0, 0.0)),
                     observation(frame, ns, GroundPoint(480.0, 0.0),
                                 target_class=TargetClass.BLUE_DANGER, box_x=40))
        seq._tracker.update(ns, snapshot_)
        seq._latest_perception = snapshot(frame, ns, *snapshot_)
        return ns

    feed(1, 10)
    confirmed_ns = feed(2, 20)

    assert seq._breakup_plan_for_this_frame(confirmed_ns) is not None
    # 同一帧再问一次不再重算：本帧已经有结论，重复边界搜索不改变结果。
    assert seq._breakup_plan_for_this_frame(confirmed_ns) is None

    next_ns = feed(3, 320)
    assert seq._breakup_plan_for_this_frame(next_ns) is not None
