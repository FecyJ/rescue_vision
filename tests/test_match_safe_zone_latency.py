"""Fast control, delayed two-point visual calibration and continuous stop evidence."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app import MatchState
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import FieldPose2D, SafeZoneCornerLocalizer
from rescue_vision.motion.protocol import SensorFlags
from rescue_vision.perception import UndistortedBoundingBox
from test_gripper_width_sequence import motion_sample
from test_match import make_sequence, make_static_map, safe_zone_snapshot_for_pose, start_sequence


def setup_sequence():
    seq = make_sequence(initial_field_position=FieldPoint(0, 700))
    start_sequence(seq)
    seq.state = MatchState.TRANSPORT_RELEASE
    seq._safe_zone_phase = 'stopping_before_calibration'
    seq._safe_zone_corner_localizer = SafeZoneCornerLocalizer(make_static_map(), replace(
        seq._safe_zone_corner_localizer.config, max_observation_age_ms=800))
    return seq


def delayed_pair(frame, capture_ns, delay_ms):
    snap = safe_zone_snapshot_for_pose(frame, capture_ns,
        FieldPose2D(FieldPoint(0, 700), math.pi/2))
    zone = snap.field_features.safe_zones[0]
    # Left bbox chooses K0/K2; the unused corner flickers every frame.
    if frame % 2:
        zone = replace(zone, image_left_landmark=replace(zone.image_left_landmark,
            ground=None, undistorted=None, confidence=0))
    zone = replace(zone, box=UndistortedBoundingBox(10, 10, 70, 90))
    result_ns = capture_ns + delay_ms * 1_000_000
    return replace(snap, result_timestamp_ns=result_ns, timing=None, field_features=replace(
        snap.field_features, result_timestamp_ns=result_ns, safe_zones=(zone,)))


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250),(10,500,350),(5,600,400)])
def test_two_delayed_frames_calibrate_once_without_turning(poll_ms, delay_ms, period_ms):
    seq = setup_sequence()
    latest = None
    frame = 0
    for ms in range(0, 2501, poll_ms):
        now = ms * 1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            frame += 1
            latest = delayed_pair(frame, (ms-delay_ms)*1_000_000, delay_ms)
        if ms < delay_ms:
            continue
        decision = seq.step(now, perception=latest, heading_rad=math.pi/2,
            cumulative_distance_m=0, left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        assert decision.angular_velocity_rad_s == 0
        if seq.safe_zone_calibration_pose is not None:
            break
    assert seq.safe_zone_calibration_pose is not None
    assert ms <= delay_ms + 2*period_ms + 350
    assert seq._safe_zone_phase == 'align_d2_line'
    assert len(seq._safe_zone_key_samples) == 2
    assert len(seq._safe_zone_calibration_pose.used_roles) == 2
    assert seq.safe_zone_calibration_pose.position.y == pytest.approx(700)


@pytest.mark.parametrize('invalid', ['motion', 'gyro', 'invalid_flags', 'duplicate_device', 'gap', 'old_capture'])
def test_calibration_does_not_accept_invalid_stationary_evidence(invalid):
    seq = setup_sequence()
    for ms in range(0, 701, 10):
        seq.observe_grasp_motion(motion_sample(ms*1_000_000))
    now = 710_000_000
    if invalid == 'motion':
        seq.observe_grasp_motion(motion_sample(now, count=1))
    elif invalid == 'gyro':
        seq.observe_grasp_motion(motion_sample(now, gyro=500_000))
    elif invalid == 'invalid_flags':
        seq.observe_grasp_motion(motion_sample(now, flags=SensorFlags(0)))
    elif invalid == 'duplicate_device':
        seq.observe_grasp_motion(replace(motion_sample(700_000_000), received_timestamp_ns=now))
    elif invalid == 'gap':
        now = 2_000_000_000
    capture = 0 if invalid == 'old_capture' else 400_000_000
    seq._latest_perception = delayed_pair(1, capture, 300)
    assert not seq._collect_safe_zone_key_sample(now)
    assert seq.safe_zone_calibration_pose is None
    assert not seq._safe_zone_key_samples


def test_same_frame_cannot_complete_confirmation_or_reset_timeout():
    seq = setup_sequence()
    latest = delayed_pair(1, 400_000_000, 300)
    for ms in range(0, 2001, 5):
        now = ms*1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if ms < 700:
            continue
        decision = seq.step(now, perception=latest,
            heading_rad=math.pi/2, cumulative_distance_m=0,
            left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        if seq._safe_zone_phase == 'align_d2_line':
            break
    assert seq.safe_zone_calibration_pose is None
    assert seq._safe_zone_phase == 'align_d2_line'
    assert 'continue_odometry' in decision.reason


@pytest.mark.parametrize('name', ['match', 'match_nb', 'cc', 'strategy'])
def test_production_safe_zone_age_budget_matches_perception(name):
    from rescue_vision.config import load_runtime_config
    config = load_runtime_config(f'configs/runtime.{name}.yaml')
    assert config.localization.safe_zone_corners.max_observation_age_ms == config.processing.max_observation_age_ms
    assert config.match.safe_zone_bbox_turn_max_angular_velocity_rad_s == pytest.approx(.25)


@pytest.mark.parametrize('flag', ['external_stop_requested', 'safety_accident', 'lost_control'])
def test_direct_safety_still_preempts_calibration(flag):
    from rescue_vision.mission import SafetySignals
    seq = setup_sequence()
    for ms in range(0, 701, 10):
        seq.observe_grasp_motion(motion_sample(ms*1_000_000))
    now = 700_000_000
    decision = seq.step(now, perception=delayed_pair(1, 400_000_000, 300),
        heading_rad=math.pi/2, cumulative_distance_m=0,
        left_speed_feedback_m_s=0, right_speed_feedback_m_s=0,
        safety=replace(SafetySignals.nominal(now), **{flag: True}))
    assert decision.state is MatchState.TERMINAL_STOP
    assert decision.linear_velocity_m_s == decision.angular_velocity_rad_s == 0
    assert seq.safe_zone_calibration_pose is None


def test_two_opposite_errors_cannot_hide_unstable_points_in_average():
    from rescue_vision.geometry.types import GroundPoint
    seq = setup_sequence()
    seq._safe_zone_stop_since_ns = 1
    seq._latest_heading_rad = seq._raw_heading_rad = math.pi/2
    for frame, offset in ((1, 100), (2, -100)):
        snap = delayed_pair(frame, frame*100, 0)
        zone = snap.field_features.safe_zones[0]
        zone = replace(zone,
            ground_anchor=replace(zone.ground_anchor, ground=GroundPoint(
                zone.ground_anchor.ground.x + offset, zone.ground_anchor.ground.y)),
            image_right_landmark=replace(zone.image_right_landmark, ground=GroundPoint(
                zone.image_right_landmark.ground.x + offset, zone.image_right_landmark.ground.y)))
        seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=(zone,)))
        assert seq._collect_safe_zone_key_sample(frame*100)
    assert not seq._lock_safe_zone_calibration_plan(200)
    assert seq._safe_zone_calibration_last_failure == 'unstable_required_pair'
    assert seq.safe_zone_calibration_pose is None


def test_expired_reobserve_cannot_continue_with_invalid_control_telemetry():
    seq = setup_sequence()
    for ms in range(0, 701, 10):
        seq.observe_grasp_motion(motion_sample(ms*1_000_000))
    seq._latest_perception = delayed_pair(1, 400_000_000, 300)
    seq._begin_safe_zone_keypoint_reobserve(700_000_000)
    decision = seq._step_safe_zone_keypoint_reobserve(2_000_000_000)
    assert seq._safe_zone_phase == 'reobserving_safe_zone_keypoints'
    assert 'control_state_unavailable' in decision.reason
    assert decision.linear_velocity_m_s == decision.angular_velocity_rad_s == 0


def test_disappearing_bbox_during_reobserve_keeps_original_deadline():
    seq = setup_sequence()
    seq._latest_perception = delayed_pair(1, 400_000_000, 300)
    seq._begin_safe_zone_keypoint_reobserve(700_000_000)
    deadline = seq._safe_zone_observation_deadline_ns
    snap = delayed_pair(2, 500_000_000, 300)
    seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=()))
    decision = seq._step_safe_zone_keypoint_reobserve(800_000_000)
    assert decision.angular_velocity_rad_s == 0
    assert seq._safe_zone_observation_deadline_ns == deadline


def test_formal_match_waits_extra_two_seconds_before_odometry_fallback():
    from rescue_vision.config import load_runtime_config

    config = load_runtime_config('configs/runtime.match.yaml')
    assert config.match.safe_zone_keypoint_reobserve_timeout_s == pytest.approx(2.8)

    seq = setup_sequence()
    seq.config = replace(
        seq.config,
        safe_zone_keypoint_reobserve_timeout_s=(
            config.match.safe_zone_keypoint_reobserve_timeout_s
        ),
    )
    feed_stationary(seq, 0, 700)
    seq._begin_safe_zone_keypoint_reobserve(700_000_000)

    feed_stationary(seq, 705, 2_700)
    waiting = seq._step_safe_zone_keypoint_reobserve(2_700_000_000)
    assert seq._safe_zone_phase == 'reobserving_safe_zone_keypoints'
    assert waiting.reason.startswith('safe_zone_wait:required_pair_missing')

    feed_stationary(seq, 2_705, 3_500)
    fallback = seq._step_safe_zone_keypoint_reobserve(3_500_000_000)
    assert seq._safe_zone_phase == 'align_d2_line'
    assert fallback.reason == (
        'safe_zone_calibration_unavailable_continue_odometry:'
        'required_pair_unavailable'
    )


@pytest.mark.parametrize('box', [
    (0, 10, 70, 90), (30, 10, 100, 90), (10, 0, 70, 90),
    (10, 10, 70, 100), (0, 10, 100, 90), (1, 10, 70, 90),
])
def test_reliable_pair_in_clipped_bbox_never_contributes_to_calibration(box):
    seq = setup_sequence()
    seq._safe_zone_stop_since_ns = 1
    full = delayed_pair(1, 100, 0)
    seq._latest_perception = full
    assert seq._collect_safe_zone_key_sample(100)
    snap = delayed_pair(2, 200, 0)
    zone = replace(snap.field_features.safe_zones[0], box=UndistortedBoundingBox(*box))
    seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=(zone,)))
    assert seq._safe_zone_has_required_ground_keypoints(zone)
    assert not seq._safe_zone_required_points_visible(zone)
    assert not seq._collect_safe_zone_key_sample(200)
    assert not seq._safe_zone_key_samples  # Full and clipped samples cannot be averaged together.
    assert not seq._lock_safe_zone_calibration_plan(200)
    assert seq.safe_zone_calibration_pose is None


def test_current_clipped_bbox_prevents_cached_pair_commit():
    seq = setup_sequence()
    seq._safe_zone_stop_since_ns = 1
    for frame in (1, 2):
        seq._latest_perception = delayed_pair(frame, frame*100, 0)
        assert seq._collect_safe_zone_key_sample(frame*100)
    snap = delayed_pair(3, 300, 0)
    zone = replace(snap.field_features.safe_zones[0], box=UndistortedBoundingBox(0, 10, 70, 90))
    seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=(zone,)))
    assert not seq._lock_safe_zone_calibration_plan(300)
    assert seq.safe_zone_calibration_pose is None


@pytest.mark.parametrize('delay_ms,period_ms', [(300,250), (600,400)])
def test_clipped_pair_scans_until_full_then_calibrates_from_stopped_frames(delay_ms, period_ms):
    seq = setup_sequence()
    seq._safe_zone_phase = 'searching_safe_zone_keypoints'
    latest = None
    frame = 0
    turn_commands = []
    last_angular_velocity = 0.0
    stop_command_ms = None
    for ms in range(0, 4001, 5):
        now = ms * 1_000_000
        moving = abs(last_angular_velocity) > 0
        heading = math.pi/2 + max(ms-delay_ms, 0) / 1000 * .25
        seq.observe_grasp_motion(motion_sample(now, gyro=250_000 if moving else 0))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            frame += 1
            capture_ms = ms-delay_ms
            latest = delayed_pair(frame, capture_ms*1_000_000, delay_ms)
            if capture_ms < 1000:
                zone = replace(latest.field_features.safe_zones[0], box=UndistortedBoundingBox(0,10,70,90))
                latest = replace(latest, field_features=replace(latest.field_features, safe_zones=(zone,)))
        if latest is None:
            continue
        decision = seq.step(now, perception=latest, heading_rad=heading,
            cumulative_distance_m=0, left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
        if decision.angular_velocity_rad_s:
            turn_commands.append(decision.angular_velocity_rad_s)
        elif last_angular_velocity and stop_command_ms is None:
            stop_command_ms = ms
        last_angular_velocity = decision.angular_velocity_rad_s
        if seq.safe_zone_calibration_pose is not None:
            break
    assert turn_commands and all(speed > 0 for speed in turn_commands)
    assert stop_command_ms is not None
    assert seq.safe_zone_calibration_pose is not None
    assert seq._safe_zone_phase == 'align_d2_line'
    assert seq._safe_zone_calibration_snapshot.capture_timestamp_ns >= stop_command_ms * 1_000_000
    assert ms <= 3200


@pytest.mark.parametrize('box,expected_phase', [
    ((0,10,70,90), 'scanning_safe_zone_keypoints'),
    ((30,10,100,90), 'scanning_safe_zone_keypoints'),
    ((0,10,100,90), 'reversing_for_safe_zone_keypoints'),
    ((10,0,70,90), 'reversing_for_safe_zone_keypoints'),
])
def test_clipped_bbox_with_all_three_points_still_adjusts(box, expected_phase):
    seq = setup_sequence()
    seq._latest_cumulative_distance_m = 0
    seq._latest_heading_rad = math.pi/2
    snap = delayed_pair(2, 100, 0)
    zone = replace(snap.field_features.safe_zones[0], box=UndistortedBoundingBox(*box))
    seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=(zone,)))
    assert seq._safe_zone_keypoint_count(zone) == 3
    seq._step_safe_zone_bbox_key_search(100)
    assert seq._safe_zone_phase == expected_phase
    assert seq.safe_zone_calibration_pose is None


def test_complete_bbox_margin_boundary_is_inclusive_and_does_not_require_centering():
    seq = setup_sequence()
    seq.config = replace(seq.config, safe_zone_bbox_edge_margin_px=12)
    snap = delayed_pair(1, 100, 0)
    zone = replace(snap.field_features.safe_zones[0], box=UndistortedBoundingBox(12,12,70,88))
    seq._latest_perception = replace(snap, field_features=replace(snap.field_features, safe_zones=(zone,)))
    assert seq._safe_zone_bbox_fully_visible(zone)
    decision = seq._step_safe_zone_bbox_key_search(100)
    assert decision.angular_velocity_rad_s == 0
    assert seq._safe_zone_phase == 'stopping_after_bbox_keypoints'


def calibration_poll(seq, now_ns, snap):
    seq._latest_perception = snap
    seq._latest_heading_rad = seq._raw_heading_rad = math.pi/2
    seq._latest_speed_feedback = (0., 0.)
    return seq._step_transport_release(now_ns)


def feed_stationary(seq, start_ms, end_ms, *, count=0):
    for ms in range(start_ms, end_ms+1, 5):
        seq.observe_grasp_motion(motion_sample(ms*1_000_000, count=count))


@pytest.mark.parametrize('arrival_ms', [700, 705])
def test_first_valid_frame_at_observation_deadline_gets_one_confirmation_window(arrival_ms):
    seq = setup_sequence()
    seq._safe_zone_phase = 'collecting_safe_zone_keys_closed'
    seq._safe_zone_observation_deadline_ns = 700_000_000
    feed_stationary(seq, 0, arrival_ms)
    first = delayed_pair(1, (arrival_ms-300)*1_000_000, 300)
    calibration_poll(seq, arrival_ms*1_000_000, first)
    assert len(seq._safe_zone_key_samples) == 1
    assert seq._safe_zone_confirmation_deadline_ns == (arrival_ms+800)*1_000_000
    feed_stationary(seq, arrival_ms+5, arrival_ms+400)
    second = delayed_pair(2, (arrival_ms+100)*1_000_000, 300)
    decision = calibration_poll(seq, (arrival_ms+400)*1_000_000, second)
    assert decision.reason == 'safe_zone_visual_calibrated_start_d2_line'
    assert seq.safe_zone_calibration_pose is not None


def test_second_valid_frame_at_confirmation_deadline_is_consumed_before_timeout():
    seq = setup_sequence()
    seq._safe_zone_phase = 'collecting_safe_zone_keys_closed'
    seq._safe_zone_observation_deadline_ns = 700_000_000
    feed_stationary(seq, 0, 700)
    calibration_poll(seq, 700_000_000, delayed_pair(1,400_000_000,300))
    deadline = seq._safe_zone_confirmation_deadline_ns
    feed_stationary(seq, 705, 1500)
    decision = calibration_poll(seq, deadline, delayed_pair(2,900_000_000,600))
    assert decision.reason == 'safe_zone_visual_calibrated_start_d2_line'
    assert seq.safe_zone_calibration_pose is not None


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250), (10,600,400)])
def test_one_missing_frame_preserves_first_sample_without_renewing_deadline(poll_ms, delay_ms, period_ms):
    seq = setup_sequence()
    seq._safe_zone_phase = 'collecting_safe_zone_keys_closed'
    seq._safe_zone_observation_deadline_ns = 800_000_000
    first_ms = 700
    feed_stationary(seq, 0, first_ms)
    first = delayed_pair(1,(first_ms-delay_ms)*1_000_000,delay_ms)
    calibration_poll(seq, first_ms*1_000_000, first)
    deadline = seq._safe_zone_confirmation_deadline_ns
    for ms in range(first_ms+poll_ms, first_ms+2*period_ms+1, poll_ms):
        seq.observe_grasp_motion(motion_sample(ms*1_000_000))
        if ms < first_ms+period_ms:
            snap = first
        elif ms < first_ms+2*period_ms:
            snap = delayed_pair(2,(first_ms+period_ms-delay_ms)*1_000_000,delay_ms)
            zone = snap.field_features.safe_zones[0]
            zone = replace(zone, ground_anchor=replace(zone.ground_anchor, ground=None, undistorted=None, confidence=0))
            snap = replace(snap, field_features=replace(snap.field_features,safe_zones=(zone,)))
        else:
            snap = delayed_pair(3,(ms-delay_ms)*1_000_000,delay_ms)
        decision = calibration_poll(seq,ms*1_000_000,snap)
        assert seq._safe_zone_confirmation_deadline_ns == deadline
        if ms < first_ms+2*period_ms:
            assert len(seq._safe_zone_key_samples) == 1
    assert decision.reason == 'safe_zone_visual_calibrated_start_d2_line'


def test_motion_between_two_good_frames_invalidates_buffered_first_frame():
    seq = setup_sequence()
    seq._safe_zone_phase = 'collecting_safe_zone_keys_closed'
    seq._safe_zone_observation_deadline_ns = 700_000_000
    feed_stationary(seq,0,700)
    calibration_poll(seq,700_000_000,delayed_pair(1,400_000_000,300))
    deadline = seq._safe_zone_confirmation_deadline_ns
    feed_stationary(seq,705,1100,count=1)
    decision = calibration_poll(seq,1_100_000_000,delayed_pair(2,800_000_000,300))
    assert seq.safe_zone_calibration_pose is None
    assert len(seq._safe_zone_key_samples) == 1
    assert seq._safe_zone_calibration_snapshot.capture_timestamp_ns == 800_000_000
    assert seq._safe_zone_confirmation_deadline_ns == deadline
    assert decision.angular_velocity_rad_s == 0


def test_reobserved_first_frame_is_consumed_at_deadline_without_extra_stop_session():
    seq = setup_sequence()
    seq._latest_perception = delayed_pair(1,100_000_000,300)
    seq._begin_safe_zone_keypoint_reobserve(400_000_000)
    feed_stationary(seq,0,1200)
    seq._latest_heading_rad = seq._raw_heading_rad = math.pi/2
    seq._latest_perception = delayed_pair(2,900_000_000,300)
    decision = seq._step_safe_zone_keypoint_reobserve(1_200_000_000)
    assert seq._safe_zone_phase == 'collecting_safe_zone_keys_closed'
    assert len(seq._safe_zone_key_samples) == 1
    assert seq._safe_zone_confirmation_deadline_ns == 2_000_000_000
    assert decision.angular_velocity_rad_s == 0


def test_failed_fit_retry_cannot_renew_confirmation_deadline():
    from rescue_vision.geometry.types import GroundPoint
    seq = setup_sequence()
    seq._safe_zone_phase = 'collecting_safe_zone_keys_closed'
    seq._safe_zone_observation_deadline_ns = 700_000_000
    for frame, now_ms in ((1,700),(2,1000),(3,1300)):
        feed_stationary(seq,0 if frame == 1 else now_ms-295,now_ms)
        snap = delayed_pair(frame,(now_ms-300)*1_000_000,300)
        zone = snap.field_features.safe_zones[0]
        zone = replace(zone,image_right_landmark=replace(zone.image_right_landmark,ground=GroundPoint(437,-900)))
        snap=replace(snap,field_features=replace(snap.field_features,safe_zones=(zone,)))
        seq._latest_perception=snap
        if seq._safe_zone_phase == 'reobserving_safe_zone_keypoints':
            seq._step_safe_zone_keypoint_reobserve(now_ms*1_000_000)
        else:
            calibration_poll(seq,now_ms*1_000_000,snap)
        assert seq._safe_zone_confirmation_deadline_ns == 1_500_000_000
    assert seq.safe_zone_calibration_pose is None
