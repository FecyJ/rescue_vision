from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.gate_clearance import (
    GateObject, gate_obstructions, make_clearance_session, maybe_begin_clearance,
    step_clearance, _segment_risk,
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


@pytest.mark.parametrize('team,sign', [(TeamColor.RED,1),(TeamColor.BLUE,-1)])
@pytest.mark.parametrize('classes,red_point,red_heading', [
    ((GREEN,),FieldPoint(-450,1022.5),0.0),
    ((BLACK,GREEN),FieldPoint(-450,1022.5),0.0),
    ((ORANGE,),FieldPoint(450,1022.5),math.pi),
])
def test_clearance_route_uses_cargo_lane_and_team_mirror(team,sign,classes,red_point,red_heading):
    session=make_clearance_session(
        GateClearanceConfig(),FieldPoint(0,sign*800),team,0,target_classes=classes,
    )
    lateral_heading=(red_heading if sign>0 else math.remainder(red_heading+math.pi,2*math.pi))
    assert session.lane_point==FieldPoint(sign*red_point.x,sign*red_point.y)
    assert session.actions[0].name=='to_lane'
    assert session.actions[0].point==session.lane_point
    assert session.actions[1].name=='align_lateral_heading'
    assert math.cos(session.actions[1].value)==pytest.approx(math.cos(lateral_heading))
    assert math.sin(session.actions[1].value)==pytest.approx(math.sin(lateral_heading))
    assert session.actions[2].value==pytest.approx(0.9)
    assert session.actions[3].value==pytest.approx(-0.6)
    assert len(session.actions)==4
    assert session.d2_push_end_point.y==pytest.approx(session.lane_point.y+sign*200)


@pytest.mark.parametrize('field,value', [('enabled','yes'),('transit_speed_m_s',float('nan')),
    ('d2_push_distance_m',0),('lane_x_abs_mm',0),
    ('front_edge_inset_mm',-1),('lateral_inset_mm',-1)])
def test_bad_experiment_config_is_rejected(field,value):
    with pytest.raises(ValueError,match='gate_clearance'):
        replace(GateClearanceConfig(),**{field:value})


def test_runtime_enables_experimental_clearance_explicitly():
    config=load_runtime_config('configs/runtime.match.yaml').match.gate_clearance
    assert config.enabled
    assert config.lane_x_abs_mm==pytest.approx(450)
    assert config.lane_y_abs_mm==pytest.approx(1022.5)
    assert config.d2_push_distance_m==pytest.approx(0.2)


def gate_sequence(*,enabled=True,position=FieldPoint(-500,800)):
    seq=_sequence(transports=1,config=runtime_config(
        gate_clearance=GateClearanceConfig(enabled=enabled),
        required_transports=2,
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


def test_clearance_waits_for_real_stop_before_freezing_route():
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


def test_moving_trigger_brakes_before_selecting_lane():
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
    assert session.action.name=='to_lane'


def test_danger_outside_clearance_lane_does_not_block_route(monkeypatch):
    seq=gate_sequence(position=FieldPoint(-500,885))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        1,attempt_started_ns=1,
    )
    seq._gate_clearance=session
    monkeypatch.setattr(
        'rescue_vision.app.gate_clearance._objects',
        lambda _sequence,_now: (GateObject(BLUE,FieldPoint(0,500),30),),
    )
    assert _segment_risk(seq,2,session.lane_point,session.lateral_end_point) is None


def test_danger_push_toward_field_edge_aborts_lateral_route(monkeypatch):
    seq=gate_sequence(position=FieldPoint(-1300,900))
    config=replace(
        seq.config.gate_clearance,
        lane_x_abs_mm=1300,
        lateral_forward_distance_m=0.2,
    )
    seq.config=replace(seq.config,gate_clearance=config)
    session=make_clearance_session(
        config,FieldPoint(-1300,900),TeamColor.RED,
        1,attempt_started_ns=1,
    )
    seq._gate_clearance=session
    monkeypatch.setattr(
        'rescue_vision.app.gate_clearance._objects',
        lambda _sequence,_now: (GateObject(BLUE,FieldPoint(-1300,1022.5),30),),
    )
    assert (
        _segment_risk(seq,2,session.lane_point,session.lateral_end_point)
        == 'danger_route_out_of_field'
    )


@pytest.mark.parametrize('dt,latency,interval',[(.005,.6,.25),(.01,.3,.4)])
def test_delayed_perception_clearance_delivers_and_uses_normal_exit(dt,latency,interval):
    seq=gate_sequence()
    original_count=seq._transport_count
    def scene(frame,stamp,pose):
        points=[]
        session=seq._gate_clearance
        if session is None or session.index<len(session.actions):
            points.append((ORANGE,FieldPoint(-150,1100)))
        # 外围目标闪烁，不影响已提交的门前动作。
        if frame%2:
            points.append((BLACK,FieldPoint(900,500)))
        return snapshot(frame,stamp,*(observation(frame,stamp,field_to_ground(pose,p),
                        target_class=cls) for cls,p in points))
    plant=MotionPlant(seq,heading=math.pi/2,dt=dt,latency_s=latency,
                      frame_interval_s=interval,perception=scene)
    plant.until(lambda d:d.reason.startswith('gate_clearance_triggered:'),seconds=3)
    plant.until(lambda d:d.state is MatchState.FINISH_STOP,seconds=40)
    reasons=[d.reason for d in plant.records]
    assert any(reason.startswith('gate_clearance:initial_stop_complete,') for reason in reasons)
    for stage in ('to_lane','align_lateral_heading','lateral_forward_900mm'):
        assert any(reason.startswith(f'gate_clearance:{stage}:complete,') for reason in reasons)
    assert any(reason.startswith(
        'gate_clearance:lateral_reverse_600mm:complete_handoff_to_normal_d2,'
    ) for reason in reasons)
    lateral_forward=[d for d in plant.records if 'noncontact=gate_lateral_forward_900mm,' in d.reason]
    lateral_reverse=[d for d in plant.records if 'noncontact=gate_lateral_reverse_600mm,' in d.reason]
    d2_push=[d for d in plant.records if 'noncontact=safe_zone_forward_final_closed,' in d.reason]
    assert lateral_forward and any(d.linear_velocity_m_s>0 for d in lateral_forward)
    assert lateral_reverse and any(d.linear_velocity_m_s<0 for d in lateral_reverse)
    assert d2_push and any(d.linear_velocity_m_s>0 for d in d2_push)
    assert all(d.gripper_posture is GripperPosture.CLOSED for d in d2_push)
    exit_reverse=[d for d in plant.records if d.reason.startswith('safe_zone_exit_reverse_open_loop')]
    assert exit_reverse and any(d.linear_velocity_m_s<0 for d in exit_reverse)
    assert seq.carried_target_count==0 and seq._transport_count==original_count+1
    assert seq.state is MatchState.FINISH_STOP
    assert not seq._gate_clearance_attempted


def test_failure_before_delivery_resumes_normal_transport():
    seq=gate_sequence(position=FieldPoint(-500,885))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        0,attempt_started_ns=0,
    )
    session.initial_stop_confirmed=True
    session.index=2
    seq._gate_clearance=session
    seq._gate_clearance_attempted=True
    seq.state=MatchState.GATE_CLEARANCE
    decision=step_clearance(seq,31_000_000_000)
    assert decision.state in {
        MatchState.TRANSPORT_ALIGN_RED_ZONE,
        MatchState.TRANSPORT_FORWARD,
        MatchState.TRANSPORT_RELEASE,
    }
    assert seq._transport_target_classes == (GREEN,)
    assert seq._gate_clearance is None
