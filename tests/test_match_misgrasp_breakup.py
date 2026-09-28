"""Released cargo owns one short breakup, across slow perception and ID changes."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.match import MatchState, GripperPosture
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D
from rescue_vision.perception import TargetClass
from rescue_vision.motion.protocol import SensorFlags
from test_gripper_width_sequence import motion_sample
from test_match import observation, snapshot
from test_match_breakup import frozen_plan, sequence
from test_match_near_field import _sequence as near_field_sequence


def released_sequence():
    seq = sequence(initial_field_position=FieldPoint(-250, 0))
    seq.config = replace(seq.config, green_max_age_ms=1200)
    seq._misgrasp_release_pose = FieldPose2D(FieldPoint(0, 0), 0.0)
    seq._misgrasp_heading_rad = 0.0
    seq._misgrasp_breakup_active = True
    seq._start_breakup_attempt(0)
    seq._settle_until_ns = 100_000_000
    return seq


def released_snapshot(frame, capture_ns, result_ns, *, y=0.0, peripheral=False,
                      outside=False):
    # Release was at (0,0), jaws x=52.5..157.5. Robot has backed up 250 mm.
    members = [observation(frame, capture_ns, GroundPoint(370, y))]
    if outside:
        members = []
    # Better ranked nearby cargo must not replace the released cargo.
    members.append(observation(frame, capture_ns, GroundPoint(450, 230), box_x=60))
    if peripheral:
        members.append(observation(frame, capture_ns, GroundPoint(365, 70),
                                  target_class=TargetClass.BLACK_CORE, box_x=40))
    return replace(snapshot(frame, capture_ns, *members), result_timestamp_ns=result_ns,
                   timing=None)


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5, 300, 250), (10, 600, 400)])
def test_released_group_freezes_and_moves_without_search_with_flicker_and_new_ids(
    poll_ms, delay_ms, period_ms,
):
    seq = released_sequence()
    latest = None
    for ms in range(0, 3500, poll_ms):
        now = ms * 1_000_000
        if ms % 10 == 0:
            seq.observe_grasp_motion(motion_sample(now))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            frame = (ms-delay_ms)//period_ms + 1
            # Renumber established tracks without erasing their independent
            # detector confirmation; release ownership must not depend on IDs.
            seq._tracker._tracks = {t.track_id + 100: replace(t, track_id=t.track_id + 100)
                                    for t in seq._tracker.tracks}
            seq._tracker._next_track_id += 100
            latest = released_snapshot(frame, now-delay_ms*1_000_000, now,
                                       peripheral=frame % 2 == 0)
        decision = seq.step(now, perception=latest, heading_rad=0, cumulative_distance_m=0,
                            left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        assert decision.angular_velocity_rad_s == 0
        assert decision.state in {MatchState.BREAKUP_SETTLE, MatchState.BREAKUP_FORWARD}
        if decision.linear_velocity_m_s > 0:
            break
    else:
        pytest.fail(seq.cluster_diagnostic(now))
    plan = seq._breakup_plan
    assert plan.aim_field == FieldPoint(120, 0)
    assert plan.forward_distance_mm == pytest.approx(seq.config.misgrasp_breakup_forward_distance_m * 1000)
    assert plan.backward_distance_mm == 80
    assert plan.approach_distance_mm == 0
    assert seq._misgrasp_breakup_active
    assert len(seq._breakup_reference_frames) == 3
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert ms < 2500


@pytest.mark.parametrize('y,allowed', [(35, True), (100, True)])
def test_release_footprint_allows_safe_contact_heading(y, allowed):
    seq = released_sequence()
    for ms in (0, 10, 20):
        seq.observe_grasp_motion(motion_sample(ms*1_000_000))
        scene = released_snapshot(ms+1, ms*1_000_000, ms*1_000_000, y=y)
        decision = seq.step(ms*1_000_000, perception=scene, heading_rad=0,
                            cumulative_distance_m=0, left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
    plan = seq._choose_breakup_plan(20_000_000)
    assert (plan is not None) is allowed
    if allowed:
        assert plan.heading_rad == pytest.approx(math.atan2(y, 370))
        seq.observe_grasp_motion(motion_sample(30_000_000))
        decision = seq.step(30_000_000, perception=scene, heading_rad=0,
                            cumulative_distance_m=0, left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        assert decision.angular_velocity_rad_s > 0
        # Corrections are still bounded against the original release heading.
        assert not seq._misgrasp_breakup_heading_allowed(replace(plan, heading_rad=math.radians(35)))


def test_configured_release_heading_limit_reports_rejection():
    seq = released_sequence()
    seq.config = replace(seq.config,
                         misgrasp_breakup_max_heading_change_rad=math.radians(10))
    for ms in (0, 10, 20):
        seq.observe_grasp_motion(motion_sample(ms * 1_000_000))
        seq.step(ms * 1_000_000,
                 perception=released_snapshot(ms+1, ms*1_000_000,
                                              ms*1_000_000, y=100),
                 heading_rad=0, cumulative_distance_m=0,
                 left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
    assert seq._choose_breakup_plan(20_000_000) is None
    assert any('release_heading_exceeds_limit' in reason
               for reason in seq._breakup_plan_rejections)


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5, 300, 250), (10, 600, 400)])
def test_non_single_green_release_stays_in_breakup_with_near_field_enabled(
    poll_ms, delay_ms, period_ms,
):
    config = load_runtime_config('configs/runtime.match.yaml').match
    seq = near_field_sequence(transports=1, config=replace(config,
        green_max_age_ms=1200, breakup_confirmation_frames=1))
    seq._started = True
    seq._fallback_field_position = FieldPoint(-250, 0)
    seq._misgrasp_release_pose = FieldPose2D(FieldPoint(0, 0), 0.0)
    seq._misgrasp_heading_rad = 0.0
    seq._misgrasp_breakup_active = True
    seq._start_breakup_attempt(0)
    seq._settle_until_ns = 100_000_000
    for ms in range(0, 2200, poll_ms):
        now = ms * 1_000_000
        if ms % 10 == 0:
            seq.observe_grasp_motion(motion_sample(now))
        scene = (released_snapshot((ms-delay_ms) // period_ms + 1,
                                   (ms-delay_ms) // period_ms * period_ms * 1_000_000,
                                   now)
                 if ms >= delay_ms and (ms-delay_ms) % period_ms == 0 else None)
        decision = seq.step(now, perception=scene, heading_rad=0,
                            cumulative_distance_m=0, left_speed_feedback_m_s=0,
                            right_speed_feedback_m_s=0)
        assert decision.state in {MatchState.BREAKUP_SETTLE, MatchState.BREAKUP_FORWARD}
        assert seq._grasp_task is None
        if decision.linear_velocity_m_s > 0:
            assert decision.gripper_posture is GripperPosture.CLOSED
            assert ms < 1600
            break
    else:
        pytest.fail(seq.cluster_diagnostic(now))


def test_failed_release_does_not_handoff_same_physical_target_to_near_field():
    config = load_runtime_config('configs/runtime.match.yaml').match
    seq = near_field_sequence(transports=1, config=replace(config,
        green_max_age_ms=1200, breakup_confirmation_frames=1))
    seq._started = True
    seq._fallback_field_position = FieldPoint(-250, 0)
    seq._misgrasp_release_pose = FieldPose2D(FieldPoint(0, 0), 0.0)
    seq._misgrasp_heading_rad = 0.0
    seq._misgrasp_breakup_active = True
    seq._start_breakup_attempt(0)
    seq._settle_until_ns = 100_000_000
    for ms in range(0, 1800, 10):
        now = ms * 1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        scene = (released_snapshot(ms // 250 + 1, ms // 250 * 250_000_000,
                                   now, outside=True)
                 if ms % 250 == 0 else None)
        decision = seq.step(now, perception=scene, heading_rad=0,
                            cumulative_distance_m=0, left_speed_feedback_m_s=0,
                            right_speed_feedback_m_s=0)
        assert decision.state is not MatchState.TRANSPORT_NEAR_FIELD_GRASP
        if decision.state is MatchState.SEARCH_CLUSTER:
            break
    else:
        pytest.fail(seq.cluster_diagnostic(now))
    assert seq._near_field_failures[-1].field_point == FieldPoint(105, 0)
    for offset_ms in (10, 20, 30):
        now += offset_ms * 1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        scene = released_snapshot(100 + offset_ms, now, now)
        scene = replace(scene, observations=scene.observations[:1])
        decision = seq.step(now, perception=scene, heading_rad=0,
                            cumulative_distance_m=0, left_speed_feedback_m_s=0,
                            right_speed_feedback_m_s=0)
        assert decision.state is MatchState.SEARCH_CLUSTER
        assert any(seq._target_attempt_blocked(target, now)
                   for target in seq._tracker.tracks)


@pytest.mark.parametrize('mode', ['missing', 'outside', 'old', 'motion', 'invalid', 'danger'])
def test_unusable_released_geometry_cannot_commit_and_exits_when_observable(mode):
    seq = released_sequence()
    latest = None
    for ms in range(0, 1800, 10):
        now = ms*1_000_000
        sample = replace(motion_sample(now), left_encoder_count=ms if mode == 'motion' else 0,
                         right_encoder_count=ms if mode == 'motion' else 0)
        if mode == 'invalid':
            sample = motion_sample(now, flags=SensorFlags(0))
        seq.observe_grasp_motion(sample)
        if ms % 250 == 0:
            latest = released_snapshot(ms//250+1, now, now, outside=mode in {'missing', 'outside'})
            if mode == 'missing':
                latest = snapshot(ms//250+1, now)
            elif mode == 'old':
                latest = released_snapshot(1, 0, now)
            elif mode == 'danger':
                # An unselected blue obstacle farther forward is still checked:
                # near the field edge, its predicted push would leave the field.
                seq._fallback_field_position = FieldPoint(1000, 0)
                seq._misgrasp_release_pose = FieldPose2D(FieldPoint(1250, 0), 0)
                danger = observation(ms//250+1, now, GroundPoint(430, 0),
                                     target_class=TargetClass.BLUE_DANGER, box_x=40)
                latest = replace(latest, observations=latest.observations+(danger,))
        decision = seq.step(now, perception=latest, heading_rad=0, cumulative_distance_m=0,
                            left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        assert decision.state is not MatchState.BREAKUP_FORWARD
        if decision.state is MatchState.SEARCH_CLUSTER:
            assert not seq._misgrasp_breakup_active
            break
    if mode in {'missing', 'outside', 'old', 'danger'}:
        assert decision.state is MatchState.SEARCH_CLUSTER
        # Missing/old captures get one delivery window, never the multi-frame
        # confirmation budget; real danger still cannot commit.
        assert ms <= seq._breakup_no_plan_budget_ms + 50
        assert ms < 1500
        # Same area can reappear; only a new misgrasp can request another short attempt.
        now += 10_000_000
        seq.observe_grasp_motion(motion_sample(now))
        decision = seq.step(now, perception=latest, heading_rad=0, cumulative_distance_m=0,
                            left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        assert not seq._misgrasp_breakup_active


@pytest.mark.parametrize('key', ['misgrasp_breakup_forward_distance_m',
                               'misgrasp_breakup_backward_distance_m',
                               'misgrasp_breakup_max_heading_change_rad'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_short_breakup_config_rejects_invalid_values(key, value):
    cfg = load_runtime_config('configs/runtime.match.yaml').match
    with pytest.raises(ValueError, match=key):
        replace(cfg, **{key: value})


def test_short_breakup_config_rejects_large_heading_adjustment():
    cfg = load_runtime_config('configs/runtime.match.yaml').match
    with pytest.raises(ValueError, match='misgrasp_breakup_max_heading_change_rad'):
        replace(cfg, misgrasp_breakup_max_heading_change_rad=math.pi)


def test_match_yaml_has_independent_misgrasp_distances_and_speed_limits():
    cfg = load_runtime_config('configs/runtime.match.yaml').match
    assert (cfg.misgrasp_backup_distance_m, cfg.misgrasp_backup_speed_m_s) == (0.25, 1.5)
    assert (cfg.misgrasp_breakup_forward_distance_m,
            cfg.misgrasp_breakup_forward_speed_m_s) == (0.35, 1.5)
    assert (cfg.misgrasp_breakup_backward_distance_m,
            cfg.misgrasp_breakup_backward_speed_m_s) == (0.08, 1.5)


@pytest.mark.parametrize('key', [
    'misgrasp_backup_distance_m', 'misgrasp_backup_speed_m_s',
    'misgrasp_breakup_forward_speed_m_s', 'misgrasp_breakup_backward_speed_m_s',
])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_new_misgrasp_config_rejects_invalid_values(key, value):
    cfg = load_runtime_config('configs/runtime.match.yaml').match
    with pytest.raises(ValueError, match=key):
        replace(cfg, **{key: value})


def test_misgrasp_backup_uses_own_distance_and_speed_limit():
    seq = sequence()
    seq.config = replace(seq.config, misgrasp_backup_distance_m=0.12,
                         misgrasp_backup_speed_m_s=0.05, return_backup_speed_m_s=1.5)
    seq._latest_heading_rad = 0.0
    seq._begin_misgrasp_recovery(0, 'conflicting_gripper_colors:blue_danger')
    for ms in range(0, 1210, 10):
        seq.observe_grasp_motion(motion_sample(ms * 1_000_000))
    decision = seq._step_misgrasp_recovery(1_200_000_000, 0.0)
    assert decision.state is MatchState.MISGRASP_BACKUP
    assert decision.linear_velocity_m_s == pytest.approx(-0.05)
    assert seq._step_misgrasp_recovery(1_210_000_000, -0.11).state is MatchState.MISGRASP_BACKUP
    decision = seq._step_misgrasp_recovery(1_220_000_000, -0.12)
    assert decision.state is MatchState.MISGRASP_SETTLE
    assert decision.reason == 'misgrasp_backup_distance_reached_stop'


def test_release_breakup_uses_own_speed_limits_without_changing_normal_breakup():
    seq = released_sequence()
    seq.config = replace(seq.config, misgrasp_breakup_forward_speed_m_s=0.04,
                         misgrasp_breakup_backward_speed_m_s=0.03,
                         breakup_forward_speed_m_s=1.5, breakup_backward_speed_m_s=1.5)
    frozen_plan(seq, forward=300, approach=0, backward=80)
    seq._latest_heading_rad = 0.0
    seq.state = MatchState.BREAKUP_FORWARD
    decision = seq._step_dynamic_breakup(10_000_000, 0.0)
    assert decision is not None
    assert decision.linear_velocity_m_s == pytest.approx(0.04)
    seq.state = MatchState.BREAKUP_BACKWARD
    seq._breakup_backward_base_distance_m = 0.3
    decision = seq._step_dynamic_breakup(20_000_000, 0.3)
    assert decision is not None
    assert decision.linear_velocity_m_s == pytest.approx(-0.03)

    seq._misgrasp_breakup_active = False
    seq.state = MatchState.BREAKUP_FORWARD
    decision = seq._step_dynamic_breakup(30_000_000, 0.0)
    assert decision is not None
    assert decision.linear_velocity_m_s > 0.04
