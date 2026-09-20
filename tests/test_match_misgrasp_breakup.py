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
from test_match_breakup import sequence


def released_sequence():
    seq = sequence(initial_field_position=FieldPoint(-250, 0))
    seq.config = replace(seq.config, green_max_age_ms=1200, cluster_min_detections=1)
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
    assert plan.forward_distance_mm == 300
    assert plan.backward_distance_mm == 80
    assert plan.approach_distance_mm == 0
    assert seq._misgrasp_breakup_active
    assert len(seq._breakup_reference_frames) == 3
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert ms < 2500


@pytest.mark.parametrize('y,allowed', [(35, True), (100, False)])
def test_only_small_correction_relative_to_release_heading_is_allowed(y, allowed):
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
        # A series of small turns must never drift beyond the release limit.
        assert not seq._misgrasp_breakup_heading_allowed(replace(plan, heading_rad=math.radians(15)))


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
        assert ms <= seq.config.breakup_no_plan_reobserve_ms + 50
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
