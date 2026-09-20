from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.match import MatchState
from rescue_vision.app.near_field_grasp import NearFieldGraspPolicy
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from test_match_liveness import _greedy_sequence_for_test
from test_match_near_field import _sequence
from test_near_field_grasp import GREEN, BLACK, selector, target


@pytest.mark.parametrize('delay_ms', [300, 450, 600])
@pytest.mark.parametrize('poll_ms', [5, 10])
@pytest.mark.parametrize('state', [MatchState.SEARCH_CLUSTER, MatchState.RELOCATE_FORWARD])
def test_delayed_far_orange_uses_current_pose_and_leaves_search(delay_ms, poll_ms, state):
    sequence = _sequence(transports=1, max_observation_age_ms=800)
    sequence.config = replace(sequence.config, green_max_age_ms=800)
    sequence._fallback_field_position = FieldPoint(900, 0)
    sequence._latest_heading_rad = math.pi / 6
    sequence.state = state
    sequence._started = True
    for ms in range(0, delay_ms + 1, poll_ms):
        heading = math.pi / 2 - (math.pi / 3) * ms / delay_ms
        sequence._record_pose_history(ms * 1_000_000, heading, 0.0)
    now = delay_ms * 1_000_000
    sequence._last_timestamp_ns = now
    obs = replace(target(cls=TargetClass.ORANGE_INJURED).observation, ground_point=GroundPoint(1100, 0))
    track = sequence._tracker.update(0, (obs,))[0]
    assert sequence._candidate_path_blocked(track.ground_point, breakup=False)
    current = sequence._current_ground_point_for_track(track, now)
    assert current is not None
    assert not sequence._candidate_path_blocked(current, breakup=False)
    if state is MatchState.RELOCATE_FORWARD:
        decision = sequence._step_relocate_forward(now, 0.0)
    else:
        decision = sequence._step_dynamic_cluster_search(now, sequence._latest_heading_rad)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert sequence.selected_track_id == track.track_id
    assert sequence._tracker.tracks[0].last_seen_timestamp_ns == 0
    sequence._record_pose_history(now + poll_ms * 1_000_000, sequence._latest_heading_rad, 0.0)
    action = sequence._step_formal_green_align(now + poll_ms * 1_000_000)
    assert action.angular_velocity_rad_s != 0


@pytest.mark.parametrize('local_handoff', [False, True])
@pytest.mark.parametrize('reason', ['candidate_replan', 'confirmation_timeout', 'no_direct_plan'])
def test_failed_greedy_candidate_resumes_original_scan_and_selects_far_alternative(reason, local_handoff):
    sequence = _greedy_sequence_for_test()
    first_ns, now = 100_000_000, 200_000_000
    for stamp in (first_ns, now):
        sequence._record_pose_history(stamp, 0.0, 0.0)
    first, other = sequence._tracker.update(first_ns, (
        target(x=350 if local_handoff else 700, timestamp=first_ns).observation,
        target(2, x=700, y=400, timestamp=first_ns).observation,
    ))
    sequence._last_timestamp_ns = now
    sequence._selected_track_id = first.track_id
    sequence._greedy_progress_rad = 1.0
    if local_handoff:
        sequence._begin_near_field_grasp(now, handoff_prior=sequence._selected_handoff_prior(now))
        assert sequence._selected_track_id is None
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    decision = sequence._finish_greedy_pickup(now, reason)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert sequence.selected_track_id == other.track_id
    assert sequence._greedy_started_ns == 0
    assert sequence._greedy_progress_rad == 1.0
    assert sequence._target_attempt_blocked(first, now)
    assert sequence._target_attempt_blocked(replace(first, track_id=first.track_id + 100), now)
    assert sequence._transport_target_classes == (GREEN,)
    expired = sequence._step_greedy_scan(100_000_000_000, 0.0)
    assert expired.reason == 'greedy_return:scan_timeout'


def test_reachable_parallel_supply_does_not_wait_for_tracker_confirmation():
    planner = selector(side_neighbor_lateral_margin_mm=60)
    first = replace(target(x=300, y=-35), handoff_matched=True)
    companion = replace(target(2, x=300, y=35, cls=BLACK), confirmed=False)
    waiting = planner.select((first, companion))
    assert waiting.rejections == ()
    assert waiting.plan is not None and waiting.plan.member_ids == (1,)
    confirmed = planner.select((first, replace(companion, confirmed=True)))
    assert confirmed.plan is not None
    assert confirmed.plan.member_ids == (1, 2)
    assert confirmed.plan.opening_width_mm <= confirmed.plan.maximum_opening_mm
    single = planner.select((first, companion), policy=NearFieldGraspPolicy(frozenset((GREEN, BLACK)), 1))
    assert single.plan is not None and single.plan.member_ids == (1,)


@pytest.mark.parametrize('failure', ['expired', 'missing_pose', 'danger'])
def test_far_scan_still_rejects_unusable_geometry_and_danger(failure):
    sequence = _greedy_sequence_for_test()
    capture, now = 100_000_000, 400_000_000
    if failure != 'missing_pose':
        sequence._record_pose_history(capture, 0.0, 0.0)
        sequence._record_pose_history(now, 0.0, 0.0)
    observations = [target(x=700, timestamp=capture).observation]
    if failure == 'danger':
        observations.append(target(2, x=400, cls=TargetClass.BLUE_DANGER, timestamp=capture).observation)
    sequence._tracker.update(capture, tuple(observations))
    if failure == 'expired':
        now = capture + round((sequence.config.green_max_age_ms + 1) * 1e6)
        sequence._record_pose_history(now, 0.0, 0.0)
    assert sequence._find_approach_seed(now, prefer_nearest=True) is None
