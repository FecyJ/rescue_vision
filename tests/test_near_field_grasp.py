from __future__ import annotations

from test_target_ground_geometry import config as physical_geometry_config

from dataclasses import replace
import math

import numpy as np
import pytest

from rescue_vision.app.near_field_grasp import (
    GraspTarget,
    GraspTargetTracker,
    NearFieldGraspPolicy,
    NearFieldGraspSelector,
    NearFieldHandoffPrior,
    polygon_distance,
)
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.motion.gripper_kinematics import GripperKinematics
from rescue_vision.perception.gripper_width import TargetGroundEnvelope, measure_target_envelope
from rescue_vision.perception.types import (
    ClassProbabilities, ColorSegmentationStatus, ObservationQuality, RoiColorSegmentation,
    TargetClass, TargetObservation, UndistortedBoundingBox,
)
from rescue_vision.tracking import TrackingConfig

GREEN = TargetClass.GREEN_SUPPLY
BLACK = TargetClass.BLACK_CORE
BLUE = TargetClass.BLUE_DANGER
ORANGE = TargetClass.ORANGE_INJURED


def projector():
    return GroundProjector(np.asarray([[0, -1, 575], [-1, 0, 500], [0, 0, 1]], dtype=float))


def target(i=1, x=300., y=0., cls=GREEN, width=40., depth=40., timestamp=0, frame=0):
    u, v = 500 - y, 800 - x
    box = UndistortedBoundingBox(math.floor(u-width/2), math.floor(v-depth/2), math.ceil(u+width/2), math.ceil(v+depth/2))
    accepted = True
    segmentation = RoiColorSegmentation(cls, ColorSegmentationStatus.ACCEPTED if accepted else ColorSegmentationStatus.INSUFFICIENT,
        box, np.full((int(box.y_max-box.y_min), int(box.x_max-box.x_min)), 255 if accepted else 0, np.uint8), .9 if accepted else 0, 1. if accepted else 0)
    obs = TargetObservation(frame, timestamp, timestamp, (1000, 1000), cls, cls,
        ClassProbabilities.from_top_class(cls, 1), .95, box, segmentation, UndistortedPixel(u,v), .95, GroundPoint(x,y), frozenset())
    corners = (GroundPoint(x-depth/2,y-width/2), GroundPoint(x+depth/2,y-width/2), GroundPoint(x+depth/2,y+width/2), GroundPoint(x-depth/2,y+width/2))
    env = TargetGroundEnvelope(frame,timestamp,cls, GroundPoint(x,y),corners)
    return GraspTarget(i,obs,env,True,cls in (GREEN,BLACK,ORANGE))


def selector(**changes):
    return NearFieldGraspSelector(replace(NearFieldGraspConfig(), **changes), projector(), GripperKinematics(),
        open_servo_angles_deg=(0,180),closed_servo_angles_deg=(90,90), target_geometry=physical_geometry_config())


def test_envelope_measurement_does_not_require_center_alignment():
    t = target(y=50)
    result = measure_target_envelope(t.observation, projector())
    assert result is not None
    assert result.center.y == 50
    assert min(p.y for p in result.corners) == pytest.approx(30.5)
    assert max(p.y for p in result.corners) == pytest.approx(69.5)


def test_candidate_geometry_log_contains_horizontal_gate_values():
    diagnostic = selector().candidate_geometry(target())
    line = diagnostic.as_log_line()

    assert diagnostic.x0_mm == pytest.approx(280.0)
    assert diagnostic.x1_mm == pytest.approx(320.0)
    assert diagnostic.depth_mm == pytest.approx(40.0)
    assert diagnostic.k0_x_mm == pytest.approx(300.0)
    assert diagnostic.opening_width_mm == pytest.approx(44.0)
    assert diagnostic.target_final_x_mm == pytest.approx(110.0)
    assert diagnostic.corridor_start_x_mm == pytest.approx(60.0)
    # 诊断前端与实际扫掠走廊共用公式：夹爪末端前伸 110.15 mm 加行程 190 mm。
    assert diagnostic.corridor_end_x_mm == pytest.approx(360.15, abs=0.01)
    assert diagnostic.corridor_half_width_mm == pytest.approx(32.0)
    assert diagnostic.forward_distance_mm == pytest.approx(190.0)
    assert "x0_mm=280.00" in line
    assert "x1_mm=320.00" in line
    assert "depth_mm=40.00" in line
    assert "k0_x_mm=300.00" in line
    assert "opening_mm=44.00" in line
    assert "target_final_x_mm=110.00" in line
    assert "corridor_start_x_mm=60.00" in line
    assert "corridor_end_x_mm=360.15" in line
    assert "corridor_half_width_mm=32.00" in line
    assert "forward_distance_mm=190.00" in line


@pytest.mark.parametrize('classes', [(GREEN,), (BLACK,), (GREEN,GREEN), (BLACK,BLACK), (GREEN,BLACK), (GREEN,BLACK,GREEN)])
def test_selects_supply_combinations_and_span_includes_gaps(classes):
    ts = tuple(target(i+1,y=(i-(len(classes)-1)/2)*60,cls=cls) for i,cls in enumerate(classes))
    plan = selector().select(ts).plan
    assert plan is not None
    assert plan.member_ids == tuple(range(1,len(classes)+1))
    assert plan.width_mm == 40+(len(classes)-1)*60
    assert plan.opening_width_mm == plan.width_mm+4
    assert plan.forward_distance_mm < max(p.x for t in ts for p in t.envelope.corners)
    assert plan.forward_distance_mm == pytest.approx(300-110.0)


@pytest.mark.parametrize('cls',[BLUE])
def test_forbidden_between_members_is_never_collected(cls):
    result = selector().select((target(1,y=-60),target(2,y=60,cls=BLACK),target(3,x=200,cls=cls)))
    assert result.plan is None or 3 not in result.plan.member_ids
    assert any('blocked_target:3' in reason for reason in result.rejections)


def test_orange_target_is_allowed_as_a_single_member_with_its_distance_formula():
    orange = target(cls=ORANGE, x=300.0, depth=40.0)
    grasp_selector = selector()
    plan = grasp_selector.select((orange,)).plan

    assert plan is not None
    assert plan.member_ids == (1,)
    # K0 bottom centre determines travel; top-mask depth cannot extend it.
    assert plan.forward_distance_mm == pytest.approx(188.0)
    diagnostic = grasp_selector.candidate_geometry(orange)
    assert diagnostic.target_final_x_mm == pytest.approx(112.0)
    assert plan.forward_distance_mm == pytest.approx(300.0 - 112.0)


@pytest.mark.parametrize(
    "cls,depth,width",
    [(GREEN, 40.0, 40.0), (ORANGE, 60.0, 30.0), (ORANGE, 90.0, 80.0), (BLACK, 120.0, 20.0)],
)
def test_corridor_end_diagnostic_matches_the_executed_sweep(cls, depth, width):
    """诊断走廊前端必须等于实际扫掠矩形前端，不能另算一套更短的行程。"""

    grasp_selector = selector()
    item = target(cls=cls, x=300.0, depth=depth, width=width)
    plan = grasp_selector.select((item,)).plan
    assert plan is not None
    # 对称目标不需要转向，走廊未旋转，可直接比较前向坐标。
    assert plan.alignment_angle_rad == pytest.approx(0.0)
    diagnostic = grasp_selector.candidate_geometry(item)
    sweep_front = max(point.x for point in plan.regions[0])
    assert sweep_front == pytest.approx(diagnostic.corridor_end_x_mm, abs=1e-9)
    # 前端必须包含夹爪末端前伸，不能退化成 corridor_start_x_mm 加行程。
    assert diagnostic.corridor_end_x_mm > 60.0 + diagnostic.forward_distance_mm


@pytest.mark.parametrize("depth", [30.0, 60.0, 120.0])
def test_orange_travel_follows_k0_not_the_top_projection_depth(depth):
    """橙色前进终点由可靠底面 K0 决定，上表面投影拉长不增加行程。"""

    grasp_selector = selector()
    orange = target(cls=ORANGE, x=300.0, depth=depth)
    plan = grasp_selector.select((orange,)).plan
    assert plan is not None
    diagnostic = grasp_selector.candidate_geometry(orange)

    assert diagnostic.depth_mm == pytest.approx(depth)
    assert plan.forward_distance_mm == pytest.approx(188.0)
    assert diagnostic.forward_distance_mm == pytest.approx(188.0)
    # 扫掠前端随开口几何变化，但不随投影深度增长。
    assert diagnostic.corridor_end_x_mm == pytest.approx(
        max(point.x for point in plan.regions[0])
    )


def test_orange_target_cannot_be_combined_with_another_member():
    result = selector().select(
        (target(1, cls=ORANGE), target(2, x=309.0, cls=GREEN))
    )

    assert any("orange_not_isolated_track:2" in reason for reason in result.rejections)


def test_orange_target_rejects_neighbor_at_inclusive_ten_mm_radius():
    orange = target(1, x=300.0, cls=ORANGE)
    neighbor = target(2, x=310.0, cls=GREEN)

    result = selector().select((orange, neighbor))
    orange_diagnostic = next(
        item
        for item in selector().candidate_geometries((orange, neighbor))
        if item.track_id == orange.track_id
    )

    assert any("orange_not_isolated_track:2" in reason for reason in result.rejections)
    assert orange_diagnostic.eligible is False


def test_orange_target_ignores_neighbor_without_current_ground_position():
    orange = target(1, cls=ORANGE)
    unknown_neighbor = target(2, x=500.0, cls=GREEN)
    unknown_neighbor = replace(
        unknown_neighbor,
        observation=replace(unknown_neighbor.observation, ground_point=None),
        envelope=None,
    )

    result = selector().select((orange, unknown_neighbor))

    assert result.plan is not None


def test_orange_recheck_keeps_the_same_isolation_hard_gate():
    grasp_selector = selector()
    orange = target(1, cls=ORANGE)
    plan = grasp_selector.select((orange,)).plan
    assert plan is not None

    rejection = grasp_selector.recheck(
        plan,
        (orange, target(2, x=310.0, cls=GREEN)),
        progress_mm=0,
    )

    assert rejection is not None
    assert rejection.startswith("orange_not_isolated_track:2")


def test_clean_orange_target_is_selectable_in_near_field_tracker():
    grasp_selector = selector()
    tracker = GraspTargetTracker(
        TrackingConfig(1, 80, 0.1, 500, 1, 0.1).build_tracker(),
        projector(),
        grasp_selector.config,
    )

    tracked = tracker.update(0, (target(cls=ORANGE).observation,))

    assert len(tracked) == 1
    assert tracked[0].selectable


def test_blue_outside_forward_corridor_does_not_block():
    # 新走廊只覆盖夹爪开口在前进方向扫过的矩形，不再覆盖历史臂/车体全包络。
    result = selector().select((target(),target(2,x=140,y=170,cls=BLUE,width=10,depth=10)))
    assert result.plan is not None


def test_blue_adjacent_but_outside_physical_sweep_allows_single_green():
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    safe = selector().select((target(), target(2, x=300, y=70, cls=BLUE)), policy=policy)
    assert safe.plan is not None and safe.plan.member_ids == (1,)
    blocked = selector().select((target(), target(2, x=300, y=20, cls=BLUE)), policy=policy)
    assert blocked.plan is None
    assert 'blocked_target:2:blue_danger' in blocked.rejections


def test_blue_sweep_checks_k0_only():
    """蓝块扫掠门禁只判断 K0，不用实体半径扩大走廊。"""

    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    allowed = selector().select(
        (target(), target(2, x=300, y=56, cls=BLUE)),
        policy=policy,
    )
    assert allowed.plan is not None
    assert allowed.plan.member_ids == (1,)

    outside_by_k0 = selector().select(
        (target(), target(2, x=300, y=50, cls=BLUE)),
        policy=policy,
    )
    assert outside_by_k0.plan is not None

    inside_by_k0 = selector().select(
        (target(), target(2, x=300, y=30, cls=BLUE)),
        policy=policy,
    )
    assert inside_by_k0.plan is None
    assert 'blocked_target:2:blue_danger' in inside_by_k0.rejections


def test_side_adjacent_orange_does_not_block_single_green():
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    result = selector().select(
        (
            target(1, x=300.0, y=0.0),
            target(2, x=300.0, y=70.0, cls=ORANGE),
        ),
        policy=policy,
    )

    assert result.plan is not None
    assert result.plan.member_ids == (1,)
    assert not any(
        reason.startswith("side_adjacent_incompatible")
        for reason in result.rejections
    )


def test_side_neighbor_with_front_back_separation_is_allowed():
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    result = selector().select(
        (
            target(1, x=300.0, y=0.0),
            target(2, x=390.0, y=70.0, cls=BLUE),
        ),
        policy=policy,
    )

    assert result.plan is not None
    assert result.plan.member_ids == (1,)


def test_dense_pure_supply_group_is_not_rejected_by_side_neighbor_gate():
    result = selector().select(
        (
            target(1, x=300.0, y=-40.0),
            target(2, x=300.0, y=0.0),
            target(3, x=300.0, y=40.0, cls=BLACK),
        )
    )

    assert result.plan is not None
    assert len(result.plan.members) == 3
    assert all(
        member.observation.target_class in (GREEN, BLACK)
        for member in result.plan.members
    )


def test_blue_keypoint_inside_forward_corridor_blocks():
    result = selector().select((target(), target(2, x=200, y=30, cls=BLUE, width=10, depth=10)))
    assert result.plan is None
    assert any('blocked_target:2' in r for r in result.rejections)


def test_first_single_green_uses_another_safe_green_when_corridor_is_blocked():
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    nearest = target(1, x=300.0, y=0.0)
    farther = target(2, x=350.0, y=200.0)
    blocker = target(3, x=200.0, y=0.0, cls=BLUE, width=10.0, depth=10.0)

    result = selector().select(
        (nearest, farther, blocker),
        policy=policy,
    )

    # docs/正式流程设计.md: 首轮候选的前进走廊被蓝色/橙色/黑色/未知目标阻挡时
    # 先尝试其它安全绿色候选，只有没有合法抓取方案时才进入局部解团。因此被
    # 阻挡的最近绿色淘汰、更远的合法绿色接管，同时仍只规划一个绿色。
    assert result.plan is not None
    assert result.plan.member_ids == (2,)
    assert any("blocked_target:3:blue_danger" in reason for reason in result.rejections)


def test_first_single_green_continues_the_aligned_handoff_target():
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    closer = target(1, x=180.0, y=-120.0)
    aligned = replace(
        target(2, x=320.0, y=0.0),
        handoff_matched=True,
    )

    result = selector().select((closer, aligned), policy=policy)

    assert result.plan is not None
    assert result.plan.member_ids == (2,)


def test_first_single_green_falls_back_when_handoff_target_is_outside_near_field():
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    closer = target(1, x=180.0, y=-120.0)
    aligned = replace(
        target(2, x=520.0, y=0.0),
        handoff_matched=True,
    )

    result = selector(max_range_mm=450.0).select((closer, aligned), policy=policy)

    # 交接目标越过近场半径时只是排序偏好失效，不能让近场对空计划空转。
    assert result.plan is not None
    assert result.plan.member_ids == (1,)
    assert "outside_near_field" in result.rejections


def test_first_single_green_falls_back_when_handoff_target_has_no_envelope():
    """复现真机 20260911_1408 日志：交接目标退化为 unknown/无地面包络。"""

    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    # 远场交接的绿色已失去颜色包络（现场为 class=unknown, xy=unknown），
    # 单帧无法重新生成计划；近场必须改选同帧合法绿色而不是等满预算。
    aligned = replace(
        target(2, x=300.0, y=0.0),
        envelope=None,
        handoff_matched=True,
    )
    other = target(5, x=260.0, y=110.0)

    result = selector().select((aligned, other), policy=policy)

    assert result.plan is not None
    assert result.plan.member_ids == (5,)


def test_larger_supply_group_beats_single_handoff_target():
    aligned = replace(
        target(1, x=300.0, y=-230.0, cls=BLACK),
        handoff_matched=True,
    )
    higher_scoring_other_group = (
        target(2, x=300.0, y=190.0, cls=GREEN),
        target(3, x=300.0, y=235.0, cls=GREEN),
    )

    result = selector(max_range_mm=600.0).select(
        (aligned, *higher_scoring_other_group)
    )

    assert result.plan is not None
    assert result.plan.member_ids == (2, 3)


def test_selector_does_not_wait_for_unconfirmed_adjacent_supply():
    green = replace(target(x=334, y=5), confirmed=False, handoff_matched=True)
    black = replace(target(2, x=332, y=-39, cls=BLACK), confirmed=False)
    planner = selector()
    waiting = planner.select((green, black))
    assert waiting.plan is not None
    assert waiting.plan.member_ids == (1,)
    confirmed = planner.select((green, replace(black, confirmed=True)))
    assert confirmed.plan is not None
    assert confirmed.plan.member_ids == (1, 2)
    first_policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    first = planner.select((green, black), policy=first_policy)
    assert first.plan is not None and first.plan.member_ids == (1,)
    distant = planner.select((green, replace(target(2, y=-200, cls=BLACK), confirmed=False)))
    assert distant.plan is not None


def test_handoff_prior_matches_only_the_nearest_local_target_and_keeps_its_id():
    grasp_config = NearFieldGraspConfig()
    tracker = GraspTargetTracker(
        TrackingConfig(1, 250, .1, 500, 1, .1).build_tracker(),
        projector(),
        grasp_config,
    )
    tracker.set_handoff_prior(
        NearFieldHandoffPrior(GREEN, GroundPoint(300.0, 0.0), 74)
    )

    current = tracker.update(
        0,
        (
            target(1, x=310.0, y=0.0).observation,
            target(2, x=390.0, y=0.0).observation,
        ),
    )
    assert [item.track_id for item in current if item.handoff_matched] == [1]

    ambiguous = replace(
        target(1, x=312.0, y=0.0, cls=GREEN, timestamp=1, frame=1).observation,
        model_target_class=GREEN,
    )
    current = tracker.update(
        1,
        (ambiguous, target(2, x=392.0, y=0.0, timestamp=1, frame=1).observation),
    )
    assert [item.track_id for item in current if item.handoff_matched] == [1]


def test_off_axis_group_aligns_before_fresh_corridor_validation():
    s = selector()
    unaligned = s.select((
        target(1, x=330, y=-150),
        target(2, x=350, y=-90),
        target(3, x=180, y=-100, cls=BLUE, width=10, depth=10),
    ))
    # 仅按 K0 扫掠时，单个第二目标的 K0 可避开蓝块；被蓝块阻挡的
    # 更大候选组仍应留下明确拒绝诊断，并由可执行替代组接管。
    assert unaligned.plan is not None
    assert unaligned.plan.member_ids == (2,)
    assert 'blocked_target:3:blue_danger' in unaligned.rejections

    aligned = s.select((
        target(1, x=330, y=-30, frame=1, timestamp=1),
        target(2, x=350, y=30, frame=1, timestamp=1),
        target(3, x=180, y=0, cls=BLUE, width=10, depth=10, frame=1, timestamp=1),
    ), locked_ids=(1, 2))
    assert aligned.plan is None
    assert aligned.preview_plan is not None
    assert aligned.preview_plan.member_ids == (1, 2)
    assert any('blocked_target:3:blue_danger' in reason for reason in aligned.rejections)


def test_alignment_tolerance_includes_its_boundary():
    plan = selector(center_tolerance_mm=20.0).select(
        (target(y=20.0),)
    ).plan

    assert plan is not None
    assert plan.alignment_angle_rad == 0.0


def test_danger_long_mask_outside_corridor_does_not_replace_its_k0():
    # 颜色掩码的纵向投影可以很长，但走廊门禁只看危险目标的 K0。
    result = selector().select((target(), target(2, x=430, cls=BLUE, width=10, depth=400)))
    assert result.plan is not None
    assert result.plan.member_ids == (1,)


def test_plan_region_is_the_opening_corridor_only():
    plan = selector().select((target(),)).plan
    assert plan is not None
    assert len(plan.regions) == 1
    x0, x1, y0, y1 = (
        min(point.x for point in plan.regions[0]),
        max(point.x for point in plan.regions[0]),
        min(point.y for point in plan.regions[0]),
        max(point.y for point in plan.regions[0]),
    )
    assert x0 == 60.0
    left_servo, right_servo = plan.opening_servo_angles_deg
    fingertips_x = max(GripperKinematics().left_tip_position(90-left_servo).x,
                       GripperKinematics().right_tip_position(right_servo-90).x)
    assert x1 == pytest.approx(fingertips_x + plan.forward_distance_mm)
    assert (y0, y1) == pytest.approx((-32.0, 32.0))


def test_supply_keypoint_inside_corridor_is_added_to_the_group():
    result = selector().select((target(1), target(2, x=200, y=20)))
    assert result.plan is not None
    assert result.plan.member_ids == (1, 2)


def test_maximum_opening_exact_boundary_and_excess():
    s = selector()
    t = target(width=s.maximum_opening_mm-4)
    assert s.select((t,)).plan is not None
    # 单个物块也不能通过收窄开口去夹包络的一部分；真实最大行程是硬门禁。
    over = s.select((target(width=s.maximum_opening_mm-3.9),)).plan
    assert over is None
    assert "maximum_opening_exceeded" in s.select(
        (target(width=s.maximum_opening_mm-3.9),)
    ).rejections
    # 多成员组同样必须淘汰——一组的边界不能靠收窄开口代表。
    group = s._group_geometry((
        target(1, x=300.0, y=-30.0, width=s.maximum_opening_mm-3.9),
        target(2, x=300.0, y=30.0, width=s.maximum_opening_mm-3.9),
    ))
    assert "maximum_opening_exceeded" in group.reasons


def test_residual_center_error_uses_independent_edge_opening():
    grasp_selector = selector()
    target_object = target(y=4)
    plan = grasp_selector.select((target_object,)).plan
    assert plan is not None and plan.alignment_angle_rad == 0
    assert plan.opening_width_mm == 44
    left_angle, right_angle = plan.opening_servo_angles_deg
    assert grasp_selector.kinematics.left_tip_position(90 - left_angle).y == pytest.approx(26.0)
    assert grasp_selector.kinematics.right_tip_position(right_angle - 90).y == pytest.approx(-18.0)


def test_depth_spread_and_capture_depth_are_not_target_hard_constraints():
    s = selector()
    ts = (target(1,x=300,y=-30),target(2,x=321,y=30))
    result = s.select(ts,locked_ids=(1,2))
    assert result.plan is not None
    assert 'depth_spread_exceeded' not in result.rejections
    result = s.select((target(depth=120),))
    assert result.plan is not None
    assert 'outside_capture_depth' not in result.rejections


def test_forward_distance_is_set_by_farthest_k0_and_has_a_limit():
    assert selector().select((target(x=320, depth=120),)).plan.forward_distance_mm == pytest.approx(210.0)
    result = selector(max_range_mm=700).select((target(x=600),))
    assert result.plan is None
    assert 'forward_distance_exceeded' in result.rejections


def test_current_danger_without_ground_point_blocks_unproven_near_field_corridor():
    blocked = target(2,x=200,cls=BLUE)
    blocked = replace(blocked, envelope=None, observation=replace(blocked.observation,k0=None,ground_point=None))
    result = selector().select((target(),blocked))
    assert result.plan is not None


def test_missing_geometry_far_image_region_cannot_be_assumed_clear():
    blocked = target(2,x=50,y=450,cls=BLUE)
    blocked = replace(blocked,envelope=None,observation=replace(blocked.observation,k0=None,ground_point=None))
    result = selector().select((target(),blocked))
    assert result.plan is not None


def test_danger_conflict_and_low_confidence_are_retained_as_obstacles():
    config = TrackingConfig(1,80,.1,500,1,.5)
    tracker = GraspTargetTracker(config.build_tracker(),projector(),NearFieldGraspConfig())
    conflict = replace(target().observation, model_target_class=BLUE)
    low = replace(target(2,y=60,cls=BLUE).observation,detection_confidence=.1)
    targets = tracker.update(0,(conflict,low))
    assert len(targets)==2
    assert all(not t.selectable for t in targets)
    clear = replace(target(timestamp=1,frame=1).observation)
    assert not tracker.update(1,(clear,))[0].selectable


def test_low_confidence_near_field_target_keeps_its_spatial_id():
    tracker = GraspTargetTracker(
        TrackingConfig(1, 80, .1, 500, 1, .5).build_tracker(),
        projector(),
        NearFieldGraspConfig(),
    )
    first = replace(
        target(timestamp=0, frame=0).observation,
        detection_confidence=.1,
    )
    second = replace(
        target(timestamp=1, frame=1).observation,
        detection_confidence=.1,
    )
    first_target = tracker.update(0, (first,))[0]
    second_target = tracker.update(1, (second,))[0]
    assert first_target.track_id == second_target.track_id
    assert not first_target.selectable
    assert not second_target.selectable


def test_handoff_prior_allows_matching_tentative_target_without_changing_tracker_status():
    grasp_config = NearFieldGraspConfig()
    grasp_selector = NearFieldGraspSelector(
        grasp_config,
        projector(),
        GripperKinematics(),
        target_geometry=physical_geometry_config(),
        open_servo_angles_deg=(0, 180),
        closed_servo_angles_deg=(90, 90),
    )
    tracker = GraspTargetTracker(
        TrackingConfig(2, 80, .1, 500, 1, .1).build_tracker(),
        projector(),
        grasp_config,
    )
    tracker.set_handoff_prior(
        NearFieldHandoffPrior(GREEN, GroundPoint(425.0, -29.0), 74)
    )

    current = tracker.update(
        0,
        (target(x=430.0, y=-25.0, timestamp=0, frame=0).observation,),
    )
    assert len(current) == 1
    near_target = current[0]
    assert not near_target.confirmed
    assert near_target.handoff_matched
    assert near_target.selectable
    selected = grasp_selector.select(current)
    assert selected.plan is not None
    assert "handoff_prior_match=true" in grasp_selector.candidate_geometry(
        near_target
    ).as_log_line()


def test_handoff_prior_does_not_bypass_class_or_distance_gate():
    grasp_config = NearFieldGraspConfig()
    grasp_selector = NearFieldGraspSelector(
        grasp_config,
        projector(),
        GripperKinematics(),
        target_geometry=physical_geometry_config(),
        open_servo_angles_deg=(0, 180),
        closed_servo_angles_deg=(90, 90),
    )
    prior = NearFieldHandoffPrior(GREEN, GroundPoint(425.0, -29.0), 74)

    for observed in (
        target(cls=ORANGE, x=430, y=-25, timestamp=0, frame=0),
        target(x=700, y=300, timestamp=0, frame=0),
    ):
        tracker = GraspTargetTracker(
            TrackingConfig(2, 80, .1, 500, 1, .1).build_tracker(),
            projector(),
            grasp_config,
        )
        tracker.set_handoff_prior(prior)
        current = tracker.update(0, (observed.observation,))
        assert not current[0].confirmed
        assert not current[0].handoff_matched
        assert grasp_selector.select(current).plan is None


def test_nonclass_quality_does_not_veto_model_supply():
    tracker = GraspTargetTracker(
        TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(),
        projector(), NearFieldGraspConfig(),
    )
    obs = replace(
        target().observation,
        quality=frozenset((ObservationQuality.K0_UNAVAILABLE,)),
    )
    tracked = tracker.update(0, (obs,))
    assert tracked[0].selectable
    assert selector().select(tracked).plan is not None


def test_explicit_danger_conflict_does_not_recover_as_supply():
    tracker = GraspTargetTracker(
        TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(),
        projector(),
        NearFieldGraspConfig(),
    )
    conflict = replace(
        target(timestamp=0, frame=0).observation,
        model_target_class=BLUE,
        quality=frozenset(),
    )
    assert not tracker.update(0, (conflict,))[0].selectable
    for timestamp in (1, 2, 3, 4):
        clean = target(timestamp=timestamp, frame=timestamp).observation
        assert not tracker.update(timestamp, (clean,))[0].selectable


def test_new_blue_and_new_supply_abort_active_corridor():
    s=selector()
    plan=s.select((target(),)).plan
    assert plan is not None
    for cls in (BLUE,ORANGE,GREEN):
        assert 'new_sweep_obstacle' in s.recheck(plan,(target(x=290),target(2,x=200,y=30,cls=cls)),progress_mm=10)


def test_member_class_conflict_or_width_growth_invalidates_plan():
    s=selector(); plan=s.select((target(),)).plan
    assert s.recheck(plan,(replace(target(),selectable=False),),progress_mm=0)=='member_class_risk:1'
    assert s.recheck(plan,(target(width=80),),progress_mm=0)=='member_outside_opening:1'


def test_rule_score_is_primary_and_weights_rank_equal_score_plans_deterministically():
    # 两侧独立组合的动作不互相覆盖。
    ts=(target(1,x=300,y=-250),target(2,x=300,y=250))
    s=selector(max_range_mm=600)
    a=s.select(ts).plan; b=s.select(tuple(reversed(ts))).plan
    assert a is not None and b is not None and a.member_ids==b.member_ids
    # 规则总分高的方案不会被更短的低分方案覆盖。
    ts=(target(1,x=180,y=-230),target(2,x=344,y=205,cls=BLACK),target(3,x=316,y=247,cls=BLACK))
    plan=selector(max_range_mm=600).select(ts).plan
    assert plan is not None
    assert plan.member_ids == (2, 3)
    assert plan.score.rule_points == 20


@pytest.mark.parametrize(
    "supplies",
    (
        (GREEN, GREEN, GREEN),
        (GREEN, BLACK),
    ),
)
def test_more_members_win_equal_fifteen_point_supply_plan(supplies):
    orange = target(1, x=300, y=-150, cls=ORANGE, width=40)
    supply_targets = tuple(
        target(index + 2, x=300, y=50 + index * 45, cls=cls)
        for index, cls in enumerate(supplies)
    )
    result = selector(max_range_mm=600).select((orange, *supply_targets))

    assert result.plan is not None
    assert result.plan.member_ids == tuple(range(2, len(supplies)+2))
    assert result.plan.score.rule_points == 15
    assert result.plan.score.orange_priority == 0


def test_handoff_supply_does_not_override_single_orange_priority():
    orange = target(1, x=300, y=-150, cls=ORANGE)
    handoff_supply = replace(
        target(2, x=300, y=150, cls=GREEN),
        handoff_matched=True,
    )

    result = selector(max_range_mm=600).select((orange, handoff_supply))

    assert result.plan is not None
    assert result.plan.member_ids == (1,)
    assert result.plan.members[0].observation.target_class is ORANGE


def test_locked_plan_uses_range_hysteresis_but_unlocked_plan_does_not():
    far = target(1, x=480.0, y=0.0)
    grasp_selector = selector(max_range_mm=450.0, range_hysteresis_mm=50.0)

    assert grasp_selector.select((far,)).plan is None
    locked = grasp_selector.select((far,), locked_ids=(1,))

    assert locked.plan is not None
    assert locked.plan.member_ids == (1,)


def test_near_field_tracker_does_not_swap_orange_and_supply_ids():
    grasp_config = NearFieldGraspConfig()
    tracker = GraspTargetTracker(
        TrackingConfig(1, 250, .1, 500, 1, .1).build_tracker(),
        projector(),
        grasp_config,
    )
    first = tracker.update(
        0,
        (target(1, x=300.0, cls=ORANGE).observation,),
    )
    second = tracker.update(
        1,
        (target(1, x=300.0, cls=GREEN, timestamp=1, frame=1).observation,),
    )

    assert first[0].track_id != second[-1].track_id


def test_twenty_point_supply_plan_still_beats_single_orange():
    result = selector(max_range_mm=600).select(
        (
            target(1, x=300, y=-150, cls=ORANGE, width=40),
            target(2, x=300, y=50, cls=BLACK),
            target(3, x=300, y=95, cls=BLACK),
        )
    )
    assert result.plan is not None
    assert result.plan.member_ids == (2, 3)
    assert result.plan.score.rule_points == 20


def test_each_farthest_x_anchor_gets_an_independent_corridor_endpoint():
    # 远目标的走廊被危险物挡住时，危险物之后的目标不应否决较短锚点。
    short = target(1, x=180, y=0)
    far = target(2, x=350, y=0)
    danger = target(3, x=300, y=0, cls=BLUE, width=10, depth=10)
    plan = selector(max_range_mm=500).select((short, far, danger)).plan
    assert plan is not None
    assert plan.member_ids == (1,)
    assert plan.forward_distance_mm == pytest.approx(70.0)


def test_selection_uses_the_latest_same_identity_geometry():
    s = selector()
    narrow = s.select((target(width=30, frame=1, timestamp=1),)).plan
    wide = s.select((target(width=60, frame=2, timestamp=2),)).plan
    assert narrow is not None and wide is not None
    assert narrow.opening_width_mm == pytest.approx(34.0)
    assert wide.opening_width_mm == pytest.approx(64.0)


@pytest.mark.parametrize('changes',[{'max_targets':4},{'count_weight':-1},{'count_weight':float('nan')},{'confirmation_frames':0},{'alignment_timeout_ms':0},{'orange_isolation_radius_mm':0},{'grasp_commit_max_observation_age_ms':0},{'fine_alignment_zone_rad':0},{'fine_alignment_min_wheel_velocity_m_s':-1},{'corridor_lateral_margin_mm':True},{'orange_priority_weight':0,'count_weight':0,'clearance_weight':0,'distance_weight':0,'alignment_weight':0},{'orange_priority_weight':0.4,'count_weight':0.2,'clearance_weight':0.1,'distance_weight':0.1,'alignment_weight':0.1}])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        replace(NearFieldGraspConfig(),**changes)


def test_corridor_reference_cannot_be_beyond_target_final_position():
    with pytest.raises(ValueError, match="corridor_start_x_mm"):
        NearFieldGraspConfig(target_final_x_mm=50.0, corridor_start_x_mm=60.0)
    with pytest.raises(ValueError, match="greedy_target_final_x_mm"):
        NearFieldGraspConfig(greedy_target_final_x_mm=50.0, corridor_start_x_mm=60.0)


def test_greedy_policy_uses_its_own_supply_endpoint():
    grasp_selector = selector(target_final_x_mm=142.0, greedy_target_final_x_mm=95.0)
    item = target(x=300.0)
    normal_policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    greedy_policy = NearFieldGraspPolicy(
        frozenset((GREEN,)),
        1,
        target_final_x_mm=95.0,
    )

    normal = grasp_selector.select((item,), policy=normal_policy).plan
    greedy = grasp_selector.select((item,), policy=greedy_policy).plan
    assert normal is not None and greedy is not None
    assert greedy.forward_distance_mm > normal.forward_distance_mm
    assert grasp_selector.candidate_geometry(
        item,
        policy=greedy_policy,
    ).target_final_x_mm == pytest.approx(95.0)


def test_polygon_collision_includes_touching_and_containment():
    a=target().envelope.corners
    assert polygon_distance(a,target(x=340).envelope.corners)==0
    assert polygon_distance(a,target(width=10,depth=10).envelope.corners)==0
    assert polygon_distance(a,target(x=350).envelope.corners)==pytest.approx(10)


def test_coasting_target_does_not_block_near_field_corridor():
    tracker=GraspTargetTracker(TrackingConfig(1,80,.1,500,1,.1).build_tracker(),projector(),NearFieldGraspConfig())
    tracker.update(0,(target().observation,target(2,x=200,y=25,cls=BLUE).observation))
    obs=target(timestamp=10_000_000,frame=1).observation
    current=tracker.update(10_000_000,(obs,))
    assert len(current)==2
    assert any(not t.observed and t.observation.target_class is BLUE for t in current)
    assert selector().select(current).plan is not None


def test_near_field_tracker_does_not_restore_deduplicated_detection_as_obstacle():
    tracker = GraspTargetTracker(
        TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(),
        projector(),
        NearFieldGraspConfig(),
    )
    first = target(1, timestamp=1, frame=1).observation
    duplicate = target(2, x=304, y=2, timestamp=1, frame=1).observation
    current = tracker.update(1, (first, duplicate))
    assert len(current) == 1
    assert current[0].selectable


def test_1930_orange_side_rear_supply_outside_full_sweep_is_not_mixed_transport():
    s = selector(orange_isolation_radius_mm=100, corridor_lateral_margin_mm=1)
    orange = target(12,x=312.6,y=-144.5,cls=ORANGE,depth=80)
    green = target(13,x=378.4,y=-94.2)
    result = s.select((orange,green))
    assert result.plan is not None
    assert result.plan.member_ids == (12,)


@pytest.mark.parametrize('cls', [BLUE])
def test_orange_side_rear_exception_never_relaxes_danger_or_unknown(cls):
    s = selector(orange_isolation_radius_mm=100, corridor_lateral_margin_mm=1)
    orange = target(12,x=312.6,y=-144.5,cls=ORANGE,depth=80)
    neighbor = target(13,x=378.4,y=-94.2,cls=cls)
    assert s.select((orange,neighbor)).plan is None


def test_orange_side_rear_supply_with_k0_outside_sweep_is_allowed():
    s = selector(orange_isolation_radius_mm=100, corridor_lateral_margin_mm=1)
    orange = target(12,x=312.6,y=-144.5,cls=ORANGE,depth=80)
    neighbor = target(13,x=378.4,y=-94.2,width=140)
    assert s._orange_isolation_rejection(orange,(orange,neighbor)) is None


@pytest.mark.parametrize("classes", [(GREEN,GREEN),(GREEN,BLACK),(BLACK,BLACK)])
def test_group_crossing_near_field_entry_radius_is_grabbed_together(classes):
    members = (target(1,x=416.7,y=12.4,cls=classes[0]),
               target(2,x=459.7,y=57.5,cls=classes[1]))
    result = selector().select(members)
    assert result.plan is not None, result.rejections
    assert result.plan.member_ids == (1,2)
    assert result.plan.forward_distance_mm <= selector().config.max_forward_distance_mm


def test_group_entry_exception_does_not_allow_excessive_forward_travel():
    result = selector().select((target(1,x=400,y=0),target(2,x=650,y=50)))
    assert result.plan is None or result.plan.member_ids != (1,2)
    assert any("forward_distance_exceeded" in reason for reason in result.rejections)


def test_three_greens_beat_two_blacks_when_both_groups_are_reachable():
    members = (target(1,x=300,y=-170,cls=BLACK),target(2,x=300,y=-125,cls=BLACK),
               target(3,x=300,y=40),target(4,x=300,y=85),target(5,x=300,y=130))
    result = selector(max_range_mm=600).select(members)
    assert result.plan is not None, result.rejections
    assert len(result.plan.members) == 3


def test_outside_group_still_requires_a_near_field_entry_member():
    result = selector().select((target(1,x=500,y=0),target(2,x=500,y=45)))
    assert result.plan is None
    assert "outside_near_field" in result.rejections


@pytest.mark.parametrize("cls,radius", [(GREEN, math.hypot(40, 40) / 2), (BLACK, 40 / math.sqrt(3))])
def test_supply_grasp_travel_places_physical_base_behind_closed_tips(cls, radius):
    grasp_selector = selector(target_final_x_mm=142.0)
    plan = grasp_selector.select((target(cls=cls, x=300.0),)).plan
    assert plan is not None
    closed_front = grasp_selector.kinematics.left_tip_position(0).x
    assert 300.0 - plan.forward_distance_mm + radius <= closed_front + 1e-9
    assert plan.forward_distance_mm == pytest.approx(300.0 - (closed_front - radius))
    diagnostic = grasp_selector.candidate_geometry(target(cls=cls, x=300.0))
    assert diagnostic.target_final_x_mm == pytest.approx(closed_front - radius)
    assert diagnostic.corridor_end_x_mm == pytest.approx(max(p.x for p in plan.regions[0]))


def test_supply_endpoint_extension_checks_newly_swept_danger():
    grasp_selector = selector(target_final_x_mm=142.0)
    supply = target(x=300.0)
    # The old centre-only travel ended at about 328 mm. The physical-base
    # endpoint must also check the newly traversed space beyond that point.
    danger = target(2, x=335.0, cls=BLUE)
    selection = grasp_selector.select((supply, danger))
    assert selection.plan is None
    assert 'blocked_target:2:blue_danger' in selection.rejections


@pytest.mark.parametrize("size", [40.0, 60.0])
def test_supply_depth_uses_configured_base_size_and_preserves_distance_limit(size):
    grasp_selector = selector(target_final_x_mm=142.0)
    grasp_selector.target_geometry = physical_geometry_config(green_size_mm=size)
    item = target(x=300.0, depth=150.0)
    plan = grasp_selector.select((item,)).plan
    assert plan is not None
    expected = 300.0 + math.hypot(size, size) / 2 - 157.5
    assert plan.forward_distance_mm == pytest.approx(expected)
    grasp_selector.config = replace(grasp_selector.config, max_forward_distance_mm=expected - 1)
    rejected = grasp_selector.select((item,))
    assert rejected.plan is None
    assert "forward_distance_exceeded" in rejected.rejections


def test_supply_group_depth_checks_each_member_not_only_farthest_k0():
    grasp_selector = selector(target_final_x_mm=142.0)
    # The slightly nearer box extends farther than the tetrahedron.
    plan = grasp_selector.select((target(1, x=300, y=-30),
                                  target(2, x=302, y=30, cls=BLACK))).plan
    assert plan is not None
    assert plan.member_ids == (1, 2)
    assert plan.forward_distance_mm == pytest.approx(300 + math.hypot(40, 40) / 2 - 157.5)
