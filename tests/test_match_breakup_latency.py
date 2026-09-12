"""Dynamic breakup regressions with asynchronous frames and stop evidence."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.breakup_planner import BreakupPlan
from rescue_vision.app.gripper_width_sequence import GraspPreparation, GraspSelection
from rescue_vision.app.match import GraspRoute, MatchState
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception.field_feature_types import (
    FieldFeatureDetectionResult,
    FieldPoseKeypoint,
    SafeZoneColor,
    SafeZonePoseObservation,
)
from rescue_vision.perception.types import UndistortedBoundingBox
from rescue_vision.perception import TargetClass
from test_gripper_width_sequence import motion_sample
from test_match import observation, snapshot
from test_match_breakup import sequence
from test_match_near_field import _sequence as near_field_sequence


def delayed_snapshot(frame: int, capture_ns: int, *, peripheral: bool = False):
    members = [
        observation(frame, capture_ns, GroundPoint(480, 0)),
        observation(
            frame,
            capture_ns,
            GroundPoint(450, 0),
            target_class=TargetClass.BLUE_DANGER,
            box_x=40,
        ),
    ]
    if peripheral:
        members.append(
            observation(
                frame,
                capture_ns,
                GroundPoint(490, 75),
                target_class=TargetClass.BLACK_CORE,
                box_x=60,
            )
        )
    return snapshot(frame, capture_ns, *members)


@pytest.mark.parametrize(
    "delay_ms,period_ms",
    [(300, 250), (500, 250), (500, 350), (600, 400), (800, 400)],
)
def test_latency_and_peripheral_flicker_eventually_freezes_direct_plan(
    delay_ms, period_ms
):
    seq = sequence()
    seq.config = replace(
        seq.config,
        safe_zone_calibration_stop_confirm_time_s=0.3,
        breakup_confirmation_frames=3,
    )
    latest = None
    decision = None
    for ms in range(0, 4500, 5):
        now = ms * 1_000_000
        if ms % 10 == 0:
            seq.observe_grasp_motion(motion_sample(now))
        if ms >= delay_ms and (ms - delay_ms) % period_ms == 0:
            frame = (ms - delay_ms) // period_ms + 1
            latest = delayed_snapshot(
                frame,
                now - delay_ms * 1_000_000,
                peripheral=frame % 2 == 0,
            )
        decision = seq.step(
            now,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
        if seq.state is MatchState.BREAKUP_FORWARD:
            break
    assert decision is not None
    assert seq._breakup_plan is not None, decision.reason
    assert seq._breakup_plan.aim.y == 0
    assert len(seq._breakup_reference_frames) == seq.config.breakup_confirmation_frames
    assert decision.reason == "breakup_plan_frozen"


def test_timeout_scans_instead_of_reselecting_same_region():
    seq = sequence()
    for ms in (10, 20):
        latest = delayed_snapshot(ms, ms * 1_000_000)
        seq.step(
            ms * 1_000_000,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
    assert seq.state is MatchState.BREAKUP_SETTLE
    decision = seq.step(
        5_000_000_000,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert decision.state is MatchState.SEARCH_CLUSTER
    for frame in range(30, 35):
        now = 5_000_000_000 + frame * 10_000_000
        decision = seq.step(
            now,
            perception=delayed_snapshot(frame, now - 300_000_000),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
        assert decision.state is MatchState.SEARCH_CLUSTER
        assert decision.angular_velocity_rad_s != 0


def ready_breakup_sequence():
    seq = sequence()
    seq._started = True
    seq.state = MatchState.BREAKUP_SETTLE
    seq.config = replace(
        seq.config,
        breakup_confirmation_frames=3,
        safe_zone_calibration_stop_confirm_time_s=0.01,
    )
    seq._start_breakup_observation(0)
    seq._breakup_only = True
    latest = None
    decision = None
    for ms in range(0, 351, 10):
        now = ms * 1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if ms >= 50 and ms % 100 == 50:
            frame = ms // 100
            capture = now - 20_000_000
            latest = snapshot(
                frame,
                capture,
                observation(frame, capture, GroundPoint(280, 0)),
                observation(
                    frame,
                    capture,
                    GroundPoint(250, 0),
                    target_class=TargetClass.BLUE_DANGER,
                    box_x=40,
                ),
            )
        decision = seq.step(
            now,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
        if seq.state is MatchState.BREAKUP_FORWARD:
            break
    assert decision is not None
    assert seq._breakup_plan is not None, decision.reason
    return seq, latest, decision


def test_freeze_does_not_require_near_field_preparation_or_second_reference():
    seq, latest, decision = ready_breakup_sequence()
    assert latest is not None
    assert decision.state is MatchState.BREAKUP_FORWARD
    assert decision.reason == "breakup_plan_frozen"
    assert seq.near_field_observation_window_open(decision.timestamp_ns) is False
    assert len(seq._breakup_reference_frames) == 3


def test_motion_invalidates_confirmation_before_freeze():
    seq = sequence()
    seq._started = True
    seq.state = MatchState.BREAKUP_SETTLE
    seq.config = replace(
        seq.config,
        breakup_confirmation_frames=3,
        safe_zone_calibration_stop_confirm_time_s=0.01,
    )
    seq._start_breakup_observation(0)
    seq._breakup_only = True
    latest = None
    for ms in (0, 10, 20):
        now = ms * 1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        frame = max(1, ms // 10)
        latest = snapshot(
            frame,
            now,
            observation(frame, now, GroundPoint(280, 0)),
            observation(
                frame,
                now,
                GroundPoint(250, 0),
                target_class=TargetClass.BLUE_DANGER,
                box_x=40,
            ),
        )
        seq.step(
            now,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
    now = 30_000_000
    seq.observe_grasp_motion(motion_sample(now, count=1))
    decision = seq.step(
        now,
        perception=latest,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    assert decision.state is MatchState.BREAKUP_SETTLE
    assert decision.linear_velocity_m_s == 0.0



def blocked_corridor_snapshot(frame: int, capture_ns: int):
    """最近绿的前向走廊里有黑核，两者相距超过成团距离，没有可解团的团。"""

    return snapshot(
        frame,
        capture_ns,
        observation(frame, capture_ns, GroundPoint(450, 0)),
        observation(
            frame,
            capture_ns,
            GroundPoint(150, 0),
            target_class=TargetClass.BLACK_CORE,
            box_x=40,
        ),
    )


def blocked_near_field_sequence():
    """近场路由待决、场上只有互不成团的单件，解团规划器必然选不出计划。"""

    seq = near_field_sequence(transports=0)
    seq._started = True
    seq.config = replace(
        seq.config,
        safe_zone_calibration_stop_confirm_time_s=0.01,
        opportunistic_single_green_enabled=False,
    )
    seq.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    seq._near_field_route = GraspRoute.DECIDING
    seq._near_field_confirmation_started_ns = 0
    # 近场路由总是带着一个已选目标；失败记忆要落在它的物理区域上。
    seq._selected_track_id = 1
    seq._selected_green_ground = GroundPoint(450.0, 0.0)
    return seq


def blocked_preparation(timestamp_ns: int = 0) -> GraspPreparation:
    return GraspPreparation(
        timestamp_ns,
        GraspSelection(None, ("blocked_target:2:black_core",)),
        (),
        session_id=1,
    )


def test_blocked_corridor_without_contact_plan_never_stops_the_robot():
    """近场判定阻挡但当前区域没有接触计划：不得停车等待，必须有界换区域。

    现场日志里的循环是近场路由反复宣告阻挡、解团规划器又算不出接触计划，
    于是停车、等重观测窗口、放弃、转向下一个目标再停车。这里按 5 ms 控制
    周期和 300 ms 帧间隔断言：既不进入解团停车，也不在控制周期里空转。
    """

    seq = blocked_near_field_sequence()
    preparation = blocked_preparation()
    latest = blocked_corridor_snapshot(1, 0)
    stopped, idle = [], []
    decision = None
    for ms in range(0, 1805, 5):
        now = ms * 1_000_000
        if ms and ms % 300 == 0:
            latest = blocked_corridor_snapshot(ms // 300 + 1, now - 300_000_000)
        decision = seq.step(
            now,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
            near_field_preparation=preparation,
        )
        if decision.state is MatchState.BREAKUP_SETTLE:
            stopped.append(ms)
        elif decision.angular_velocity_rad_s == 0.0:
            idle.append((ms, decision.reason))

    assert decision is not None
    assert not stopped, stopped[:3]
    assert not idle, idle[:3]
    assert seq.state is MatchState.SEARCH_CLUSTER
    # 失败记忆落在物理区域上：同一几何不再重下同一条处方。
    assert any(
        failure.reason == "breakup_no_contact_plan"
        for failure in seq._near_field_failures
    )
    assert seq._selected_track_id is None
    green = next(
        track for track in seq._tracker.tracks
        if track.target_class is TargetClass.GREEN_SUPPLY
    )
    assert seq._target_attempt_blocked(green, now) is True


def test_blocked_corridor_region_is_reopened_after_real_displacement():
    """区域封锁只针对未变的物理几何；车真的换位后必须重新允许尝试。"""

    seq = blocked_near_field_sequence()
    preparation = blocked_preparation()
    latest = blocked_corridor_snapshot(1, 0)
    for ms in range(0, 605, 5):
        now = ms * 1_000_000
        if ms and ms % 300 == 0:
            latest = blocked_corridor_snapshot(ms // 300 + 1, now - 300_000_000)
        seq.step(
            now,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
            near_field_preparation=preparation,
        )
    assert seq._near_field_failures
    green = next(
        track for track in seq._tracker.tracks
        if track.target_class is TargetClass.GREEN_SUPPLY
    )
    assert seq._target_attempt_blocked(green, now) is True

    for ms in range(600, 1205, 5):
        now = ms * 1_000_000
        if ms % 300 == 0:
            latest = blocked_corridor_snapshot(ms // 300 + 1, now - 300_000_000)
        seq.step(
            now,
            perception=latest,
            heading_rad=0.0,
            cumulative_distance_m=0.2,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
            near_field_preparation=preparation,
        )
    assert seq._target_attempt_blocked(green, now) is False


# --- 一圈内必须落地（A/C）与失败记忆收紧（B）、安全区 bbox（D） ---------------

GROUP_SPECS = (
    (450.0, 0.0, TargetClass.GREEN_SUPPLY, 10.0),
    (450.0, 60.0, TargetClass.BLACK_CORE, 40.0),
)
NEIGHBOUR_SPECS = (
    (850.0, 0.0, TargetClass.GREEN_SUPPLY, 10.0),
    (850.0, 60.0, TargetClass.BLACK_CORE, 40.0),
)


def scene(frame: int, capture_ns: int, specs, heading_rad: float = 0.0):
    """按当前航向把固定场地坐标的目标换算成机器人地面系观测。

    车只在原地转时物块不会移动，观测必须随航向一起转；否则用机器人系点
    反算出的场地坐标会跟着航向漂移，失败记忆也就对不上同一个物理点。
    """

    cosine = math.cos(heading_rad)
    sine = math.sin(heading_rad)
    return tuple(
        observation(
            frame,
            capture_ns,
            GroundPoint(cosine * x + sine * y, -sine * x + cosine * y),
            target_class=cls,
            box_x=box_x,
        )
        for x, y, cls, box_x in specs
    )


def feed(seq, frame: int, ms: int, specs, heading_rad: float = 0.0):
    """送一帧观测；返回 (capture_ns, 观测元组)。"""

    capture_ns = int(ms * 1e6)
    members = scene(frame, capture_ns, specs, heading_rad)
    seq._tracker.update(capture_ns, members)
    return capture_ns, members


def confirm_scene(seq, specs, *, frames=((1, 10), (2, 20))):
    """连续两帧确认成员，返回最后一帧的 (frame, capture_ns, 观测)。"""

    latest = None
    for frame, ms in frames:
        capture_ns, members = feed(seq, frame, ms, specs)
        latest = (frame, capture_ns, members)
        seq._latest_perception = snapshot(frame, capture_ns, *members)
    return latest


def breakup_plan(aim_x: float, aim_y: float = 0.0):
    """只用于预置失败记忆的最小计划；aim_field 是唯一被比对的字段。"""

    return BreakupPlan(
        (1, 2), (1,), 1, GroundPoint(aim_x, aim_y), 0.0, 0.0, 200.0, 100.0,
        40.0, 0, 1,
        (FieldPoint(aim_x, aim_y), FieldPoint(aim_x, aim_y + 60.0)),
        FieldPoint(aim_x, aim_y), 160.0,
    )


def test_one_full_rotation_must_commit_an_action():
    """一圈用完不能绕过失败记忆，必须执行一次有效换位。

    现场 20260912_0053 从 63.3 s 转到 85.4 s（约两圈）都不行动，因为每次
    近场受阻回搜索都会清零旋转预算，失败记忆只会累积。这里断言一圈之内
    必然出现实际的有界换位；同一未变化失败目标仍不能被重新提交。
    """

    seq = sequence()
    seq._latest_heading_rad = 0.0
    # 唯一可解团的瞄准点被记忆占住：不提交就只能一直转。
    seq._breakup_failed_aims = [FieldPoint(450.0, 0.0)]
    seq._breakup_attempts = [
        breakup_plan(450.0)
        for _ in range(int(seq.config.breakup_max_attempts))
    ]
    lap_s = (
        seq.config.cluster_search_sweep_angle_rad
        / abs(seq.config.cluster_search_angular_velocity_rad_s)
    )

    heading = 0.0
    distance = 0.0
    latest = None
    relocation_ms = None
    moved = False
    unchanged_target_plan = None
    for ms in range(0, int((lap_s + 2.0 + seq.config.cluster_relocate_distance_m / seq.config.cluster_relocate_speed_m_s) * 1000) + 1, 5):
        now = ms * 1_000_000
        if ms % 300 == 0:
            frame = ms // 300 + 1
            position = seq.estimated_field_position
            moved_specs = tuple((x-position.x, y-position.y, cls, box)
                                for x,y,cls,box in GROUP_SPECS)
            capture_ns, members = feed(seq, frame, (frame - 1) * 300, moved_specs, heading)
            latest = snapshot(frame, capture_ns, *members)
        seq.observe_grasp_motion(motion_sample(now))
        decision = seq.step(
            now,
            perception=latest,
            heading_rad=heading,
            cumulative_distance_m=distance,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
        if decision.reason == "rotation_budget_relocate_start":
            relocation_ms = ms
            # Before the encoder-backed translation, the remembered physical
            # aim is still unchanged and must remain rejected.
            unchanged_target_plan = seq._choose_breakup_plan(now)
        if (
            decision.state is MatchState.RELOCATE_FORWARD
            and decision.linear_velocity_m_s > 0.0
        ):
            moved = True
            distance += decision.linear_velocity_m_s * 0.005
        heading += decision.angular_velocity_rad_s * 0.005
        if moved and decision.state is MatchState.SEARCH_CLUSTER:
            break

    assert relocation_ms is not None, "一圈内没有选择有界换位"
    assert relocation_ms / 1000.0 <= lap_s + 2.0
    assert moved, "没有发生实际非零换位动作"
    assert distance >= seq.config.cluster_relocate_distance_m - 1e-6, (decision.reason, seq.estimated_field_position, heading, ms)
    assert seq.state is MatchState.SEARCH_CLUSTER
    assert unchanged_target_plan is None
    assert any(
        "reason=group_in_failed_region" in line
        for line in seq._breakup_plan_rejections
    )


def test_effective_relocation_reopens_the_same_physical_aim():
    """只有真实换位后，同一物理瞄准点才可重新评估。"""

    seq = sequence()
    seq._latest_heading_rad = 0.0
    feed(seq, 1, 10, GROUP_SPECS)
    capture_ns, members = feed(seq, 2, 20, GROUP_SPECS)
    seq._latest_perception = snapshot(2, capture_ns, *members)
    plan = seq._choose_breakup_plan(capture_ns)
    assert plan is not None

    seq._breakup_plan = plan
    seq._resume_dynamic_search(capture_ns, "breakup_reference_timeout_reselect")
    assert seq._choose_breakup_plan(capture_ns) is None

    # Keep the target at the same field location while the robot translates
    # 300 mm; a mere heading/session change is not used as the unlock.
    seq._fallback_field_position = FieldPoint(300.0, 0.0)
    moved_specs = tuple(
        (x - 300.0, y, target_class, box_x)
        for x, y, target_class, box_x in GROUP_SPECS
    )
    feed(seq, 2, 320, moved_specs)
    capture_ns, members = feed(seq, 3, 330, moved_specs)
    seq._latest_perception = snapshot(3, capture_ns, *members)

    reopened = seq._choose_breakup_plan(capture_ns)
    assert reopened is not None
    assert reopened.aim_field.x == pytest.approx(plan.aim_field.x)
    assert reopened.aim_field.y == pytest.approx(plan.aim_field.y)


def test_rotation_budget_relocates_when_nothing_but_blue_remains():
    """只剩蓝块也必须在避开蓝块后换位，不能反复整圈扫描。"""

    seq = sequence()
    seq._latest_heading_rad = 0.0
    lap_s = (
        seq.config.cluster_search_sweep_angle_rad
        / abs(seq.config.cluster_search_empty_angular_velocity_rad_s)
    )
    blue_only = ((450.0, 0.0, TargetClass.BLUE_DANGER, 40.0),)

    heading = 0.0
    latest = None
    seen_reasons = set()
    for ms in range(0, int((lap_s + 3.0) * 1000) + 1, 5):
        now = ms * 1_000_000
        if ms % 300 == 0:
            frame = ms // 300 + 1
            capture_ns, members = feed(seq, frame, (frame - 1) * 300, blue_only, heading)
            latest = snapshot(frame, capture_ns, *members)
        seq.observe_grasp_motion(motion_sample(now))
        decision = seq.step(
            now,
            perception=latest,
            heading_rad=heading,
            cumulative_distance_m=0.0,
            left_speed_feedback_m_s=0.0,
            right_speed_feedback_m_s=0.0,
        )
        seen_reasons.add(decision.reason)
        heading += decision.angular_velocity_rad_s * 0.005

    assert "rotation_budget_commit" not in seen_reasons
    assert "rotation_budget_relocate_start" in seen_reasons
    assert "relocate_forward" in seen_reasons
    assert seq.state is MatchState.RELOCATE_FORWARD


def test_failed_attempts_do_not_bleed_into_a_neighbouring_group():
    """失败历史只认被瞄准的那个物理点，不再用 460 mm 邻域连坐。"""

    seq = sequence()
    seq._latest_heading_rad = 0.0
    _, capture_ns, _ = confirm_scene(seq, GROUP_SPECS + NEIGHBOUR_SPECS)
    seq._breakup_attempts = [
        breakup_plan(450.0)
        for _ in range(int(seq.config.breakup_max_attempts))
    ]

    plan = seq._choose_breakup_plan(capture_ns)

    assert plan is not None
    assert plan.aim_field.x > 600.0
    assert any(
        "reason=group_attempts_exhausted" in line
        for line in seq._breakup_plan_rejections
    )


def test_failed_attempt_records_the_aim_not_the_whole_group():
    """一次确认超时只封锁它真正瞄准的物理点，不再把整片区域判死。

    现场 20260912_0053 一次确认超时后，同一条 `group_in_failed_region` 让
    100 mm 内所有团连续 20 秒无法解团。
    """

    seq = sequence()
    seq._latest_heading_rad = 0.0
    _, capture_ns, _ = confirm_scene(seq, GROUP_SPECS + NEIGHBOUR_SPECS)
    plan = seq._choose_breakup_plan(capture_ns)
    assert plan is not None

    seq._breakup_plan = plan
    seq._resume_dynamic_search(capture_ns, "breakup_reference_timeout_reselect")

    # 记忆里只应有被瞄准的那一个物理点，不是整组成员足迹。
    assert seq._breakup_failed_aims == [plan.aim_field]
    # 该瞄准点被封锁，相距 400 mm 的另一团仍可选。
    alternative = seq._choose_breakup_plan(capture_ns)
    assert alternative is not None
    assert alternative.aim_field.x > 600.0
    assert any(
        "reason=group_in_failed_region" in line
        for line in seq._breakup_plan_rejections
    )


def safe_zone_features(frame: int, capture_ns: int, zone_box: UndistortedBoundingBox):
    """只用于 bbox 重叠判定的安全区场地特征。"""

    keypoint = FieldPoseKeypoint(None, None, 0.0)
    return FieldFeatureDetectionResult(
        frame_sequence=frame,
        capture_timestamp_ns=capture_ns,
        result_timestamp_ns=capture_ns,
        image_size=(100, 100),
        safe_zones=(
            SafeZonePoseObservation(
                box=zone_box,
                ground_anchor=keypoint,
                image_left_landmark=keypoint,
                image_right_landmark=keypoint,
                physical_color=SafeZoneColor.RED,
                confidence=0.9,
                quality=frozenset(),
            ),
        ),
        center_cross=None,
    )


def with_safe_zone(frame: int, capture_ns: int, zone_box, members):
    base = snapshot(frame, capture_ns, *members)
    return replace(
        base,
        field_features=safe_zone_features(frame, capture_ns, zone_box),
    )


def test_delivered_member_is_excluded_by_safe_zone_bbox_overlap():
    """图像上压在安全区上的物块不再是候选，不依赖场地位姿推算。

    目标框 x∈[10,20]、y∈[10,20]，中心 (15,15)；安全区框取 (0,0,12,12) 只
    与目标框重叠、不含其中心，因此只有重叠判定能拦住它。
    """

    seq = sequence()
    seq._latest_heading_rad = 0.0
    confirm_scene(seq, GROUP_SPECS)
    capture_ns, members = feed(seq, 3, 320, GROUP_SPECS)
    target = seq._tracker.tracks[0]

    seq._latest_perception = with_safe_zone(
        3, capture_ns, UndistortedBoundingBox(0.0, 0.0, 12.0, 12.0), members
    )
    assert seq._target_overlaps_safe_zone_bbox(target.box) is True
    assert seq._graspable_target_is_usable(target) is False
    # 已交付成员不再参与成团，原先的两成员团因此不成立。
    assert seq._choose_breakup_plan(capture_ns) is None

    capture_ns, members = feed(seq, 4, 620, GROUP_SPECS)
    seq._latest_perception = with_safe_zone(
        4, capture_ns, UndistortedBoundingBox(60.0, 60.0, 90.0, 90.0), members
    )
    assert seq._target_overlaps_safe_zone_bbox(target.box) is False
    assert seq._graspable_target_is_usable(target) is True
    assert seq._choose_breakup_plan(capture_ns) is not None


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5, 300, 250), (10, 600, 400)])
def test_failed_entry_proposal_has_independent_no_plan_deadline(poll_ms, delay_ms, period_ms):
    """Repeated waiting_distinct_capture must not erase a failed planning result."""
    from test_match_breakup import frozen_plan
    seq = sequence()
    proposal = frozen_plan(seq, heading=0.)
    seq._start_breakup_attempt(0, proposal)
    latest = None
    exited = None
    for ms in range(0, 2500, poll_ms):
        now = ms * 1_000_000
        if ms % 10 == 0:
            seq.observe_grasp_motion(motion_sample(now))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            capture = (ms-delay_ms)*1_000_000
            frame = (ms-delay_ms)//period_ms+1
            latest = snapshot(frame, capture, observation(frame, capture, GroundPoint(450, 0)))
        decision = seq.step(now, perception=latest, heading_rad=0., cumulative_distance_m=0.,
                            left_speed_feedback_m_s=0., right_speed_feedback_m_s=0.)
        if decision.state is MatchState.SEARCH_CLUSTER:
            exited = ms
            break
    assert exited is not None
    assert exited <= seq.config.breakup_no_plan_reobserve_ms + 100
    assert 'timeout' in decision.reason
    assert seq._breakup_plan is not proposal or decision.linear_velocity_m_s == 0.


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5, 300, 250), (10, 600, 400)])
def test_valid_core_uses_confirmation_budget_then_really_moves(poll_ms, delay_ms, period_ms):
    seq = sequence()
    seq._start_breakup_attempt(0)
    latest = None
    moved = None
    for ms in range(0, 3000, poll_ms):
        now = ms*1_000_000
        if ms % 10 == 0:
            seq.observe_grasp_motion(motion_sample(now))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            frame = (ms-delay_ms)//period_ms+1
            latest = delayed_snapshot(frame, (ms-delay_ms)*1_000_000, peripheral=frame%2 == 0)
        decision = seq.step(now, perception=latest, heading_rad=0., cumulative_distance_m=0.,
                            left_speed_feedback_m_s=0., right_speed_feedback_m_s=0.)
        if decision.state is MatchState.BREAKUP_FORWARD and decision.linear_velocity_m_s > 0:
            moved = ms
            break
    assert moved is not None, decision.reason
    assert moved <= delay_ms + 4*period_ms + 100
    assert len(seq._breakup_reference_frames) == 3


def test_old_breakup_frame_cannot_consume_another_confirmation_slot():
    """旧帧即使仍在年龄窗口内，也不能伪造停稳后的新几何。"""

    seq = sequence()
    seq._start_breakup_attempt(0)
    for stamp in range(0, 91_000_000, 10_000_000):
        seq.observe_grasp_motion(motion_sample(stamp))

    first_capture = 90_000_000
    seq.step(
        first_capture,
        perception=snapshot(
            9,
            first_capture,
            *scene(9, first_capture, ((450., 0., TargetClass.GREEN_SUPPLY, 10.), (480., 0., TargetClass.BLUE_DANGER, 40.))),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    seq.observe_grasp_motion(motion_sample(100_000_000))

    fresh_capture = 100_000_000
    fresh = snapshot(
        10,
        fresh_capture,
        *scene(10, fresh_capture, ((450., 0., TargetClass.GREEN_SUPPLY, 10.), (480., 0., TargetClass.BLUE_DANGER, 40.))),
    )
    seq.step(
        fresh_capture,
        perception=fresh,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    confirmed = set(seq._breakup_reference_frames)
    assert confirmed

    old_capture = 50_000_000
    old = snapshot(
        9,
        old_capture,
        *scene(9, old_capture, ((450., 0., TargetClass.GREEN_SUPPLY, 10.), (480., 0., TargetClass.BLUE_DANGER, 40.))),
    )
    decision = seq.step(
        110_000_000,
        perception=old,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )

    assert seq._breakup_reference_frames == confirmed
    assert seq._breakup_last_capture_ns == fresh_capture
    assert decision.state is MatchState.BREAKUP_SETTLE
    assert decision.linear_velocity_m_s == 0.0


def test_telemetry_loss_keeps_breakup_stationary_gate_closed():
    """遥测断档/无效样本不能让确认直接进入前推。"""

    seq = sequence()
    seq._start_breakup_attempt(0)
    for stamp in range(0, 101_000_000, 10_000_000):
        seq.observe_grasp_motion(motion_sample(stamp))
    capture = 100_000_000
    first = snapshot(1, capture, *scene(1, capture, GROUP_SPECS))
    seq.step(
        capture,
        perception=first,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )
    progress = len(seq._breakup_reference_frames)

    # A changed encoder sample invalidates the stationary interval.
    seq.observe_grasp_motion(motion_sample(110_000_000, count=1))
    second = snapshot(2, 110_000_000, *scene(2, 110_000_000, GROUP_SPECS))
    decision = seq.step(
        110_000_000,
        perception=second,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.0,
        right_speed_feedback_m_s=0.0,
    )

    assert len(seq._breakup_reference_frames) == progress
    assert seq.state is MatchState.BREAKUP_SETTLE
    assert decision.linear_velocity_m_s == 0.0
    assert "waiting_stationarity" in decision.reason


@pytest.mark.parametrize('delay_ms,period_ms,poll_ms', [(300,250,5), (600,400,10)])
def test_empty_scan_near_safe_zone_moves_with_delayed_frames(delay_ms, period_ms, poll_ms):
    seq = sequence(initial_field_position=FieldPoint(-171,543))
    seq._breakup_only = False
    heading = -2.6
    latest = None
    moved = False
    for ms in range(0, 16000, poll_ms):
        now = ms*1_000_000
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            capture = now-delay_ms*1_000_000
            latest = snapshot((ms-delay_ms)//period_ms+1, capture)
        seq.observe_grasp_motion(motion_sample(now))
        decision = seq.step(now, perception=latest, heading_rad=heading,
            cumulative_distance_m=0., left_speed_feedback_m_s=0., right_speed_feedback_m_s=0.)
        heading += decision.angular_velocity_rad_s*poll_ms/1000.
        if decision.linear_velocity_m_s > 0:
            moved = True
            assert decision.state is MatchState.RELOCATE_FORWARD
            break
    assert moved, decision.reason


@pytest.mark.parametrize('point', [GroundPoint(450,0), GroundPoint(500,180), None])
def test_relocation_rejects_danger_in_full_gripper_sweep_or_missing_geometry(point):
    seq = sequence()
    seq._latest_heading_rad = 0.
    seq._record_pose_history(0, 0., 0.)
    seq._latest_perception = snapshot(1,0,observation(1,0,point,target_class=TargetClass.BLUE_DANGER))
    assert seq._relocation_path_clear(0,0.,0.3) is not True


def test_relocation_rechecks_new_danger_before_next_forward_command():
    seq = sequence()
    seq.state = MatchState.RELOCATE_FORWARD
    seq._relocate_forward_base_distance_m = 0.
    decision = seq.step(0, perception=snapshot(1,0,observation(1,0,GroundPoint(400,0),
        target_class=TargetClass.BLUE_DANGER)), heading_rad=0., cumulative_distance_m=0.)
    assert decision.linear_velocity_m_s == 0.
    assert decision.state is MatchState.SEARCH_CLUSTER


@pytest.mark.parametrize('cls', list(TargetClass))
def test_safe_zone_bbox_objects_are_removed_from_planning_input(cls):
    seq = near_field_sequence(transports=1)
    inside = observation(1,0,GroundPoint(300,0),target_class=cls,box_x=10)
    outside = observation(1,0,GroundPoint(700,100),box_x=60)
    raw = with_safe_zone(1,0,UndistortedBoundingBox(0,0,40,40),(inside,outside))
    filtered = seq.planning_perception(raw)
    assert filtered.observations == (outside,)
    assert raw.observations == (inside,outside)
    assert filtered.field_features is raw.field_features
    assert seq.planning_perception(filtered) is filtered


@pytest.mark.parametrize('delay_ms,period_ms', [(300,250),(600,400)])
def test_breakup_search_still_approaches_far_collectible(delay_ms,period_ms):
    seq = near_field_sequence(transports=1)
    seq._started = True
    seq.config = replace(seq.config, green_max_age_ms=1000,
                         opportunistic_single_green_enabled=True)
    seq.state = MatchState.SEARCH_CLUSTER
    seq._breakup_only = True
    latest = None
    progressed = False
    for ms in range(0, 2500, 5):
        now = ms*1_000_000
        if ms >= delay_ms and (ms-delay_ms)%period_ms == 0:
            frame = (ms-delay_ms)//period_ms+1
            capture = now-delay_ms*1_000_000
            latest = snapshot(frame,capture,observation(frame,capture,GroundPoint(700,0)))
        decision = seq.step(now,perception=latest,heading_rad=0.,cumulative_distance_m=0.,
            left_speed_feedback_m_s=0.,right_speed_feedback_m_s=0.)
        if decision.linear_velocity_m_s > 0:
            progressed = True
            assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
            break
    assert progressed, decision.reason
