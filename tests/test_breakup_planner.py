from dataclasses import replace
import math

import pytest

from rescue_vision.app.breakup_planner import (
    BreakupTarget,
    physical_radii,
    plan_breakup,
    same_local_group,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from rescue_vision.perception.target_ground_geometry import (
    RegularTetrahedronTargetGeometry,
)


def target(i, x, y, cls=TargetClass.GREEN_SUPPLY):
    return BreakupTarget(i, 100, cls, GroundPoint(x, y), 20, 30)


def plans(targets, **kwargs):
    runtime = load_runtime_config('configs/runtime.match.yaml')
    args = dict(config=runtime.match, origin=FieldPoint(0, 0), heading_rad=0,
                static_map=runtime.world.static_map, field_bounds=(-1500, 1500, -1500, 1500),
                front_mm=157.5, allowed_classes=frozenset({TargetClass.GREEN_SUPPLY}))
    args.update(kwargs)
    return plan_breakup(tuple(targets), **args)


def test_aim_is_real_member_even_when_mean_is_empty():
    items = [target(1, 450, -110), target(2, 450, -80), target(3, 450, 80), target(4, 450, 110)]
    result = plans(items)
    assert result
    assert all(p.aim in [t.center for t in items] for p in result)
    assert all(abs(p.aim.y) >= 80 for p in result)
    assert all(p.aim_id in p.contact_ids for p in result)


def test_rejections_explain_why_each_candidate_was_skipped():
    """选不出计划时必须给出逐候选原因，而不是只有一句笼统结论。"""

    rejections: list[str] = []
    assert not plans([target(1, 450, 0)], rejections=rejections)
    assert rejections == [
        "aim=none reason=group_too_small_or_no_allowed_class "
        "size=1 ids=(1,) min_detections=2"
    ]

    # 行程上限连接触跨度都覆盖不了时仍逐候选淘汰，原因里给出相对门槛。
    runtime = load_runtime_config('configs/runtime.match.yaml')
    short_travel = replace(runtime.match, breakup_forward_distance_m=0.115)
    rejections = []
    assert not plans(
        [target(1, 450, 0), target(2, 450, 80)],
        config=short_travel,
        rejections=rejections,
    )
    assert rejections
    assert all("reason=penetration_below_minimum" in line for line in rejections)


def tetrahedron_target(i, x, y):
    """按生产几何构造 40 mm 正四面体成员，不写死内切/外接半径。"""

    geometry = RegularTetrahedronTargetGeometry(edge_mm=40.0)
    contact, safety = physical_radii(geometry)
    return BreakupTarget(i, 100, TargetClass.BLACK_CORE, GroundPoint(x, y),
                         contact, safety)


def test_shallow_contact_span_is_plannable_within_its_own_span():
    """比固定门槛浅的团只要整段推穿就成立；黑核原先永远选不出瞄准点。"""

    runtime = load_runtime_config('configs/runtime.match.yaml')
    floor_mm = (runtime.match.breakup_min_penetration_mm
                + runtime.match.breakup_braking_margin_mm)
    # 两个成员相距 80 mm 成团，但只有瞄准成员横穿爪中线，跨度只有自身接触盘直径。
    items = [tetrahedron_target(1, 450, 0), tetrahedron_target(2, 450, 80)]
    result = plans(items, allowed_classes=frozenset({TargetClass.BLACK_CORE}))

    assert result
    plan = result[0]
    assert plan.contact_ids == (1,)
    assert 2 * items[0].contact_radius_mm < floor_mm
    assert plan.penetration_mm >= 2 * items[0].contact_radius_mm
    assert plan.forward_distance_mm == 500
    # 整段推穿：前推行程覆盖接触跨度，不是被门槛截断的部分推进。
    assert plan.forward_distance_mm >= plan.penetration_mm


def test_shallow_span_still_requires_pushing_through_the_whole_contact_set():
    """浅团的放宽不是降低推进要求：行程不够整段推穿时仍按相对门槛淘汰。"""

    runtime = load_runtime_config('configs/runtime.match.yaml')
    span_mm = 2 * tetrahedron_target(1, 450, 0).contact_radius_mm
    short_travel = replace(runtime.match, breakup_forward_distance_m=0.115)
    rejections: list[str] = []
    assert not plans(
        [tetrahedron_target(1, 450, 0), tetrahedron_target(2, 450, 80)],
        config=short_travel,
        allowed_classes=frozenset({TargetClass.BLACK_CORE}),
        rejections=rejections,
    )
    assert rejections
    assert all("reason=penetration_below_minimum" in line for line in rejections)
    assert all(f"minimum_mm={span_mm:.1f}" in line for line in rejections)


def test_non_contact_ids_are_excluded_from_grouping_and_aims():
    """已交付成员不能成组或当瞄准点，原先的两成员团因此不再成立。"""

    items = [target(1, 450, 0), target(2, 480, 0)]
    assert plans(items)

    rejections: list[str] = []
    assert not plans(items, non_contact_ids=frozenset({2}), rejections=rejections)
    assert rejections == [
        "aim=none reason=group_too_small_or_no_allowed_class "
        "size=1 ids=(1,) min_detections=2"
    ]


def test_excluding_a_member_keeps_the_remaining_group_plannable():
    """排除已交付成员只影响选组，不因排除本身把剩余合法成员一起卡住。"""

    items = [target(1, 450, 0), target(2, 480, 0), target(3, 450, 100)]
    baseline = plans(items)
    excluded = plans(items, non_contact_ids=frozenset({3}))

    assert baseline
    assert excluded
    assert all(3 not in plan.member_ids for plan in excluded)


def test_rejections_stay_empty_when_a_plan_exists():
    rejections: list[str] = []
    result = plans(
        [target(1, 450, 0), target(2, 480, 0, TargetClass.BLUE_DANGER)],
        rejections=rejections,
    )

    assert result
    assert rejections == []


def test_dense_core_and_blue_mixed_group_are_candidates():
    result = plans([target(1, 450, 0), target(2, 480, 0, TargetClass.BLUE_DANGER), target(3, 510, 0)])
    assert result[0].contact_ids == (1, 2, 3)
    runtime = load_runtime_config("configs/runtime.match.yaml")
    assert runtime.match.breakup_min_penetration_mm <= result[0].penetration_mm
    assert result[0].penetration_mm <= result[0].forward_distance_mm == 500
    assert not plans([target(1, 450, 0, TargetClass.BLUE_DANGER), target(2, 480, 0, TargetClass.BLUE_DANGER)])


def test_priority_group_that_blocks_grasp_beats_larger_unrelated_group():
    result = plans(
        [
            target(1, 450, 0),
            target(2, 480, 0, TargetClass.BLUE_DANGER),
            target(3, 450, 500),
            target(4, 480, 500),
            target(5, 510, 500),
        ],
        priority_ids=frozenset({1, 2}),
    )
    assert result
    assert set(result[0].member_ids) == {1, 2}


def test_distance_is_rotated_and_bounded():
    items = [target(1, 350, 350), target(2, 375, 375)]
    cfg = load_runtime_config('configs/runtime.match.yaml').match
    p = plans(items)[0]
    assert p.approach_distance_mm == pytest.approx(math.hypot(350, 350)-20-260)
    assert p.forward_distance_mm == cfg.breakup_forward_distance_m * 1000
    assert p.backward_distance_mm == cfg.breakup_backward_distance_m * 1000
    q = plans([target(1, 250, 0), target(2, 280, 0)], approach=False)[0]
    assert q.approach_distance_mm == 0
    assert q.forward_distance_mm == p.forward_distance_mm
    assert q.penetration_mm != p.penetration_mm


def test_boundaries_and_both_safe_zones_reject_push():
    items = [target(1, 450, 0), target(2, 480, 0, TargetClass.BLUE_DANGER)]
    assert not plans(items, origin=FieldPoint(950, 0))
    for sign in (-1, 1):
        assert not plans(items, origin=FieldPoint(0, sign*650), heading_rad=sign*math.pi/2)


def test_retreat_cap_rejects_unreleasable_plan():
    cfg = replace(load_runtime_config('configs/runtime.match.yaml').match, breakup_backward_distance_m=0.05)
    assert not plans([target(1, 450, 0), target(2, 480, 0)], config=cfg)


def test_retry_prefers_new_contact_and_requires_more_depth_if_same():
    items = [target(1, 450, 0), target(2, 480, 0), target(3, 510, 0)]
    p = plans(items)[0]
    result = plans(items, attempt=2, previous_aim=p.aim_field, previous_penetration_mm=p.penetration_mm)
    assert result and result[0].aim_id != p.aim_id


def test_spatial_identity_survives_id_reset_but_not_distant_group():
    a = (FieldPoint(400, 0), FieldPoint(450, 0))
    assert same_local_group(a, (FieldPoint(430, 10), FieldPoint(480, 10)), 100)
    assert not same_local_group(a, (FieldPoint(900, 0), FieldPoint(950, 0)), 100)


@pytest.mark.parametrize('name,value', [('breakup_forward_distance_m', float('nan')), ('breakup_min_penetration_mm', True), ('breakup_max_attempts', 5), ('breakup_backward_distance_m', -1)])
def test_invalid_configuration(name, value):
    with pytest.raises(ValueError):
        replace(load_runtime_config('configs/runtime.match.yaml').match, **{name: value})


def test_reverse_distance_is_independent_of_contact_approach_gap():
    cfg = replace(load_runtime_config('configs/runtime.match.yaml').match,
                  breakup_backward_distance_m=0.12)
    candidates = plans([target(1, 500, 0), target(2, 500, 60)], config=cfg, approach=False)
    assert candidates
    plan = candidates[0]
    assert plan.forward_distance_mm > 300
    assert plan.backward_distance_mm == 120
