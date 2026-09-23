from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.gate_clearance import (
    GateObject, gate_obstructions, make_clearance_session, maybe_begin_clearance,
    step_clearance, _sweep_risk,
)
from rescue_vision.app.match import MatchState, GripperPosture
from rescue_vision.config import load_runtime_config
from rescue_vision.config.gate_clearance import GateClearanceConfig
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from rescue_vision.world import TeamColor
from noncontact_support import MotionPlant
from test_match import snapshot, observation, field_to_ground, runtime_config
from test_match_near_field import _sequence

GREEN, BLACK, ORANGE, BLUE = (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE,
                             TargetClass.ORANGE_INJURED, TargetClass.BLUE_DANGER)
MAP = load_runtime_config('configs/runtime.match.yaml').world.static_map


@pytest.mark.parametrize('cls,x,expected', [
    (ORANGE,-150,True), (GREEN,150,True), (BLACK,150,True),
    (BLUE,-150,True), (BLUE,150,True), (GREEN,-150,False),
    (BLACK,-150,False), (ORANGE,150,False),
])
@pytest.mark.parametrize('team,sign', [(TeamColor.RED,1), (TeamColor.BLUE,-1)])
def test_gate_class_region_and_team_mapping(cls,x,expected,team,sign):
    obj = GateObject(cls,FieldPoint(sign*x,sign*1100),30)
    assert bool(gate_obstructions((obj,),MAP,team,GateClearanceConfig())) is expected


@pytest.mark.parametrize('x,y,expected', [(-150,1000,True),(-150,999,False),
    (-150,1200,False),(-301,1100,False),(-150,1250,False)])
def test_trigger_uses_front_strip_not_inside_zone(x,y,expected):
    obj=GateObject(ORANGE,FieldPoint(x,y),30)
    assert bool(gate_obstructions(
        (obj,),MAP,TeamColor.RED,GateClearanceConfig()
    )) is expected


def test_trigger_strip_thresholds_are_configurable():
    orange = lambda x,y: (GateObject(ORANGE,FieldPoint(x,y),30),)
    shifted = GateClearanceConfig(front_edge_inset_mm=63,front_depth_mm=40)
    assert gate_obstructions(orange(-150,1100),MAP,TeamColor.RED,shifted)
    assert not gate_obstructions(orange(-150,1090),MAP,TeamColor.RED,shifted)
    assert not gate_obstructions(orange(-150,1150),MAP,TeamColor.RED,shifted)
    narrowed = GateClearanceConfig(lateral_inset_mm=50)
    assert gate_obstructions(orange(-150,1100),MAP,TeamColor.RED,narrowed)
    assert not gate_obstructions(orange(-25,1100),MAP,TeamColor.RED,narrowed)


@pytest.mark.parametrize('start_x', [-600,600])
@pytest.mark.parametrize('team,y_sign', [(TeamColor.RED,1),(TeamColor.BLUE,-1)])
def test_nearest_s_outward_heading_and_center_disposal(start_x,team,y_sign):
    session=make_clearance_session(
        GateClearanceConfig(),FieldPoint(start_x,y_sign*800),team,0,
    )
    x_sign=1 if start_x>0 else -1
    assert session.actions[0].name=='align_stash_heading'
    assert session.actions[1].point==FieldPoint(x_sign*620,y_sign*885)
    assert session.actions[1].kind=='straight'
    assert session.actions[1].target_heading_rad==pytest.approx(session.actions[0].value)
    assert math.cos(session.actions[2].value)==pytest.approx(x_sign)
    assert session.actions[4].value==-0.12
    assert session.actions[5].name=='sweep_close'
    assert session.actions[5].kind=='gripper' and not session.actions[5].opened
    assert math.cos(session.actions[6].value)==pytest.approx(-x_sign)
    assert session.actions[7].point==FieldPoint(-x_sign*500,y_sign*885)
    assert session.actions[7].kind=='straight'
    assert session.actions[7].value==pytest.approx(1.0)
    assert session.actions[7].target_heading_rad is not None
    assert math.cos(session.actions[7].target_heading_rad)==pytest.approx(-x_sign)
    assert not session.actions[7].opened
    assert math.hypot(session.actions[9].point.x,session.actions[9].point.y)==pytest.approx(195)
    assert session.actions[9].kind=='straight'
    assert session.actions[9].target_heading_rad==pytest.approx(session.actions[8].value)
    assert math.cos(session.actions[8].value)*x_sign>0
    assert math.sin(session.actions[8].value)*(-y_sign)>0


@pytest.mark.parametrize('field,value', [('enabled','yes'),('sweep_speed_m_s',float('nan')),
    ('release_reverse_m',0),('center_stop_radius_mm',1100),
    ('front_edge_inset_mm',-1),('lateral_inset_mm',-1)])
def test_bad_experiment_config_is_rejected(field,value):
    with pytest.raises(ValueError,match='gate_clearance'):
        replace(GateClearanceConfig(),**{field:value})


def test_runtime_disables_experimental_clearance_explicitly():
    config=load_runtime_config('configs/runtime.match.yaml').match.gate_clearance
    assert not config.enabled
    assert config.sweep_heading_tolerance_rad==pytest.approx(math.radians(2),abs=0.0001)


def gate_sequence(*,enabled=True,position=FieldPoint(-500,800)):
    seq=_sequence(transports=1,config=runtime_config(
        gate_clearance=GateClearanceConfig(enabled=enabled),
        safe_zone_fallback_max_angular_velocity_rad_s=1.0,
        green_max_age_ms=1800, action_settle_time_s=0))
    seq._started=True
    seq._fallback_field_position=position
    seq._transport_target_classes=(GREEN,)
    seq.state=MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase='align_d2_line'
    seq._safe_zone_calibration_pose=object()
    return seq


def test_black_carry_hold_uses_unmodified_closed_calibration():
    seq=gate_sequence(enabled=False)
    seq._near_field_pickup.closed_angles=(80,100)
    seq._transport_target_classes=(BLACK,GREEN)
    d=seq._decision(1,0.2,0,'carrying')
    assert d.gripper_angles_deg is None
    assert seq._decision(2,0,0,'release',posture=GripperPosture.OPEN).gripper_angles_deg is None
    seq._transport_target_classes=(GREEN,)
    assert seq._decision(3,0.2,0,'carrying').gripper_angles_deg is None


def test_retrigger_disabled_and_stale_capture_cannot_start_clearance():
    seq=gate_sequence(enabled=False)
    assert maybe_begin_clearance(seq,1) is None
    seq.config=replace(seq.config,gate_clearance=GateClearanceConfig(enabled=True))
    seq._gate_clearance_attempted=True
    assert maybe_begin_clearance(seq,1) is None
    seq._gate_clearance_attempted=False
    seq._latest_perception=snapshot(1,1,observation(1,1,GroundPoint(300,0),target_class=ORANGE))
    assert maybe_begin_clearance(seq,3_000_000_000) is None


def test_clearance_is_blocked_until_d1_visual_calibration(monkeypatch):
    seq=gate_sequence(position=FieldPoint(-100,700))
    seq._latest_perception=snapshot(1,1)
    monkeypatch.setattr(
        'rescue_vision.app.gate_clearance._objects',
        lambda _sequence,_now: (GateObject(ORANGE,FieldPoint(-150,1100),30),),
    )
    seq._safe_zone_calibration_pose=None
    seq._safe_zone_phase='align_d1_line'
    assert maybe_begin_clearance(seq,1) is None
    seq._safe_zone_calibration_pose=object()
    assert maybe_begin_clearance(seq,1) is None
    seq._safe_zone_phase='align_d2_line'
    decision=maybe_begin_clearance(seq,1)
    assert decision is not None
    assert decision.reason.startswith('gate_clearance_triggered:')
    assert 'orange_injured@(-150.0,1100.0)' in decision.reason


def test_missing_motion_feedback_holds_clearance_action():
    seq=gate_sequence(position=FieldPoint(-500,800))
    seq._gate_clearance=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        1,attempt_started_ns=1,
    )
    seq.state=MatchState.GATE_CLEARANCE
    seq._latest_heading_rad=None
    seq._latest_cumulative_distance_m=None
    decision=step_clearance(seq,2)
    assert decision.linear_velocity_m_s==0
    assert decision.angular_velocity_rad_s==0
    assert 'initial_stop_waiting' in decision.reason


def test_clearance_waits_for_real_stop_before_freezing_stash_route():
    seq=gate_sequence(position=FieldPoint(-100,700))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        1,attempt_started_ns=1,
    )
    seq._gate_clearance=session
    seq.state=MatchState.GATE_CLEARANCE
    waiting=step_clearance(seq,2)
    assert waiting.linear_velocity_m_s==0
    assert waiting.angular_velocity_rad_s==0
    assert 'initial_stop_waiting' in waiting.reason
    assert not session.initial_stop_confirmed


def test_moving_trigger_brakes_before_selecting_stash_side():
    seq=gate_sequence(position=FieldPoint(-100,700))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        1_000_000_000,attempt_started_ns=1_000_000_000,
    )
    seq._gate_clearance=session
    seq.state=MatchState.GATE_CLEARANCE
    plant=MotionPlant(seq,heading=math.pi/2)
    plant.linear=0.4
    decision=plant.tick()
    assert decision.linear_velocity_m_s==0
    assert decision.angular_velocity_rad_s==0
    assert 'initial_stop_waiting' in decision.reason
    assert not session.initial_stop_confirmed
    plant.until(lambda d:'initial_stop_complete' in d.reason,seconds=2)
    assert session.initial_stop_confirmed
    assert session.action.name=='align_stash_heading'


def test_old_zone_edge_danger_is_outside_inward_sweep(monkeypatch):
    seq=gate_sequence(position=FieldPoint(-500,885))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        1,attempt_started_ns=1,
    )
    session.index=7
    seq._gate_clearance=session
    monkeypatch.setattr(
        'rescue_vision.app.gate_clearance._objects',
        lambda _sequence,_now: (GateObject(BLUE,FieldPoint(0,1190),30),),
    )
    assert _sweep_risk(seq,2) is None


def test_new_danger_intrusion_that_would_leave_field_stops_sweep(monkeypatch):
    seq=gate_sequence(position=FieldPoint(-500,885))
    field_right=seq._physical_field_bounds()[1]
    config=replace(
        seq.config.gate_clearance,
        side_x_mm=field_right-100,
    )
    session=make_clearance_session(
        config,FieldPoint(-config.side_x_mm,885),TeamColor.RED,
        1,attempt_started_ns=1,
    )
    session.index=7
    seq._gate_clearance=session
    monkeypatch.setattr(
        'rescue_vision.app.gate_clearance._objects',
        lambda _sequence,_now: (GateObject(BLUE,FieldPoint(0,885),30),),
    )
    assert _sweep_risk(seq,2)=='danger_sweep_out_of_field'


def test_sweep_heading_drift_brakes_before_more_linear_motion():
    seq=gate_sequence(position=FieldPoint(500,885))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        1,attempt_started_ns=1,
    )
    session.initial_stop_confirmed=True
    session.index=7
    session.released=True
    seq._gate_clearance=session
    seq.state=MatchState.GATE_CLEARANCE
    seq._latest_heading_rad=math.pi-0.05

    decision=step_clearance(seq,2)

    assert decision.linear_velocity_m_s==0
    assert decision.angular_velocity_rad_s==0
    assert decision.soft_brake
    assert decision.reason.startswith('gate_clearance:sweep_heading_drift_brake,')
    assert session.sweep_realign_pending
    assert session.sweep_realign_count==1


def test_sweep_cross_track_error_exits_instead_of_driving_into_zone():
    seq=gate_sequence(position=FieldPoint(500,906))
    session=make_clearance_session(
        seq.config.gate_clearance,FieldPoint(500,885),TeamColor.RED,
        1,attempt_started_ns=1,
    )
    session.initial_stop_confirmed=True
    session.index=7
    session.released=True
    seq._gate_clearance=session
    seq.state=MatchState.GATE_CLEARANCE
    seq._latest_heading_rad=math.pi

    decision=step_clearance(seq,2)

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert 'sweep_cross_track_outside:error_mm=21.0,limit_mm=20.0' in decision.reason
    assert decision.linear_velocity_m_s==0


@pytest.mark.parametrize('dt,latency,interval',[(.005,.6,.25),(.01,.3,.4)])
def test_delayed_perception_clearance_completes_without_point_follower_or_delivery(dt,latency,interval):
    seq=gate_sequence()
    original_count=seq._transport_count
    def scene(frame,stamp,pose):
        points=[]
        session=seq._gate_clearance
        if session is None or session.index<8:
            points.append((ORANGE,FieldPoint(-150,1100)))
        # 外围目标闪烁，不影响已提交的暂存或扫掠动作。
        if frame%2:
            points.append((BLACK,FieldPoint(900,500)))
        return snapshot(frame,stamp,*(observation(frame,stamp,field_to_ground(pose,p),
                        target_class=cls) for cls,p in points))
    plant=MotionPlant(seq,heading=math.pi/2,dt=dt,latency_s=latency,
                      frame_interval_s=interval,perception=scene)
    plant.until(lambda d:d.reason.startswith('gate_clearance_triggered:'),seconds=3)
    plant.until(lambda d:d.reason=='gate_clearance_complete_search',seconds=40)
    reasons=[d.reason for d in plant.records]
    assert any(reason.startswith('gate_clearance:initial_stop_complete,') for reason in reasons)
    for stage in ('align_stash_heading','to_stash_s','stash_reverse_120mm',
                  'align_sweep_heading','sweep_via_midpoint','align_field_center',
                  'center_forward_200mm_radius'):
        assert any(reason.startswith(f'gate_clearance:{stage}:complete,') for reason in reasons)
    assert any(reason.startswith('gate_clearance:sweep_close,') for reason in reasons)
    stash=[d for d in plant.records if 'noncontact=gate_to_stash_s,' in d.reason]
    assert stash
    assert all('point_tracking:' not in d.reason and 'point_correction:' not in d.reason for d in stash)
    assert max(abs(d.angular_velocity_rad_s) for d in stash)<0.15
    sweep=[d for d in plant.records if 'noncontact=gate_sweep_via_midpoint,' in d.reason]
    assert any(d.linear_velocity_m_s>0.3 for d in sweep)
    assert all('point_tracking:' not in d.reason for d in sweep)
    assert max(abs(d.angular_velocity_rad_s) for d in sweep)<0.15
    assert all(d.gripper_posture is GripperPosture.CLOSED for d in sweep)
    center=[d for d in plant.records if 'noncontact=gate_center_forward_200mm_radius,' in d.reason]
    assert center
    assert all('point_tracking:' not in d.reason and 'point_correction:' not in d.reason for d in center)
    assert max(abs(d.angular_velocity_rad_s) for d in center)<0.15
    opened=next(d for d in plant.records if d.reason.startswith('gate_clearance:center_open'))
    field_text=opened.reason.split('field_position=(',1)[1].split(')',1)[0]
    field_x,field_y=(float(value) for value in field_text.split(','))
    assert math.hypot(field_x,field_y)<=200
    assert seq.carried_target_count==0 and seq._transport_count==original_count
    assert seq.state is MatchState.SEARCH_CLUSTER
    assert not seq._gate_clearance_attempted


def test_failure_after_stash_exits_to_search():
    seq=gate_sequence(position=FieldPoint(-500,885))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        0,attempt_started_ns=0,
    )
    session.initial_stop_confirmed=True
    session.index=11
    session.released=True
    seq._gate_clearance=session
    seq._gate_clearance_attempted=True
    seq._transport_target_classes=()
    seq.state=MatchState.GATE_CLEARANCE
    plant=MotionPlant(seq,heading=math.pi)
    decision=step_clearance(seq,31_000_000_000)
    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason.startswith('gate_clearance_failed_search:')
    assert seq._gate_clearance is None
