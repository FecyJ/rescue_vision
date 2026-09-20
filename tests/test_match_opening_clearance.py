"""Opening single green must clear neighboring bodies, not just K0 centers."""
from __future__ import annotations

from dataclasses import replace

import pytest

from rescue_vision.app.near_field_grasp import NearFieldGraspPolicy
from test_near_field_grasp import BLUE, BLACK, ORANGE, GREEN, selector, target
from test_match_near_field import _sequence


@pytest.mark.parametrize('cls', [BLUE, ORANGE, BLACK, GREEN])
def test_opening_rejects_body_intrusion_with_center_outside_sweep(cls):
    planner = selector()
    seed = target(x=350)
    neighbor = target(2, x=320, y=45, cls=cls)
    policy = NearFieldGraspPolicy(frozenset((GREEN,)), 1)
    ordinary = planner.select((seed, neighbor), policy=policy, locked_ids=(1,))
    assert ordinary.plan is not None
    strict = replace(policy, obstacle_extent_required=True)
    opening = planner.select((seed, neighbor), policy=strict, locked_ids=(1,))
    assert opening.plan is None
    assert f'blocked_target:2:{cls.value}' in opening.rejections
    assert planner.recheck(ordinary.plan, (seed, neighbor), progress_mm=0, policy=strict) == (
        f'new_sweep_obstacle:2:{cls.value}'
    )
    # The exact same body guard applies to a newly entering obstacle after lock.
    assert planner.recheck(ordinary.plan, (seed, neighbor), progress_mm=0, policy=policy) is None


def test_opening_keeps_isolated_green_and_does_not_change_later_policy():
    flow = _sequence()
    assert flow.near_field_policy.obstacle_extent_required
    assert selector().select((target(), target(2, y=300, cls=BLUE)),
                             policy=flow.near_field_policy).plan is not None
    flow._transport_count = 1
    assert not flow.near_field_policy.obstacle_extent_required
    flow._greedy_active = True
    assert not flow.near_field_policy.obstacle_extent_required


@pytest.mark.parametrize('poll_ms,frame_ms,delay_ms', [(5, 250, 300), (10, 400, 600)])
def test_opening_side_body_recovery_advances_to_transport(poll_ms, frame_ms, delay_ms, monkeypatch):
    # Exercise the production session/task path with delayed frames, keeping the
    # blue K0 outside the narrow grasp corridor while its body intersects it.
    import test_match2315_grasp_task as regression
    make_target = regression.target

    def side_target(*args, **kwargs):
        if kwargs.get('cls') is BLUE and kwargs.get('y') == 0:
            kwargs['y'] = 45
        return make_target(*args, **kwargs)

    monkeypatch.setattr(regression, 'target', side_target)
    regression.test_search_recovery_and_grasp_keep_one_task_until_transport(
        poll_ms, frame_ms, delay_ms,
    )
