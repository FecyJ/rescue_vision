from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.gate_clearance import (
    GateObject, gate_obstructions, make_clearance_session, maybe_begin_clearance,
    reacquire_plan_matches_cargo, step_clearance, _sweep_risk,
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
    assert bool(gate_obstructions((obj,),MAP,team,200)) is expected


@pytest.mark.parametrize('x,y,expected', [(-150,1000,True),(-150,999,False),
    (-150,1200,False),(-301,1100,False),(-150,1250,False)])
def test_trigger_uses_front_strip_not_inside_zone(x,y,expected):
    obj=GateObject(ORANGE,FieldPoint(x,y),30)
    assert bool(gate_obstructions((obj,),MAP,TeamColor.RED,200)) is expected


@pytest.mark.parametrize('start_x', [-600,600])
def test_nearest_s_outward_heading_and_center_disposal(start_x):
    session=make_clearance_session(GateClearanceConfig(),FieldPoint(start_x,800),TeamColor.RED,(GREEN,),0,110)
    sign=1 if start_x>0 else -1
    assert session.actions[0].point==FieldPoint(sign*470,1115)
    assert math.cos(session.actions[1].value)==pytest.approx(sign)
    assert session.actions[3].value==-0.12
    assert math.cos(session.actions[4].value)==pytest.approx(-sign)
    assert session.actions[5].point==FieldPoint(-sign*350,1115)
    assert session.actions[5].opened
    assert session.actions[7].point==FieldPoint(-sign*350,650)
    assert session.stash==FieldPoint(sign*580,1115)


def test_reacquire_requires_the_original_class_multiset():
    session=make_clearance_session(
        GateClearanceConfig(),FieldPoint(-600,800),TeamColor.RED,
        (GREEN,BLACK),0,110,
    )
    from test_near_field_grasp import selector, target
    complete=selector().select((target(1,y=-30),target(2,y=30,cls=BLACK))).plan
    partial=selector().select((target(1),)).plan
    wrong=selector().select((target(1),target(2,y=50))).plan
    assert reacquire_plan_matches_cargo(session,complete)
    assert not reacquire_plan_matches_cargo(session,partial)
    assert not reacquire_plan_matches_cargo(session,wrong)


@pytest.mark.parametrize('field,value', [('enabled','yes'),('sweep_speed_m_s',float('nan')),
    ('release_reverse_m',0),('observation_timeout_ms',True),('center_release_y_mm',1100)])
def test_bad_experiment_config_is_rejected(field,value):
    with pytest.raises(ValueError,match='gate_clearance'):
        replace(GateClearanceConfig(),**{field:value})


def test_runtime_enables_experimental_clearance_explicitly():
    assert load_runtime_config(
        'configs/runtime.match.yaml'
    ).match.gate_clearance.enabled


def gate_sequence(*,enabled=True,position=FieldPoint(-350,800)):
    seq=_sequence(transports=1,config=runtime_config(
        gate_clearance=GateClearanceConfig(enabled=enabled),
        safe_zone_fallback_max_angular_velocity_rad_s=1.0,
        green_max_age_ms=1800, action_settle_time_s=0))
    seq._started=True
    seq._fallback_field_position=position
    seq._transport_target_classes=(GREEN,)
    seq.state=MatchState.TRANSPORT_ALIGN_RED_ZONE
    seq._safe_zone_phase='align_d1_line'
    return seq


def test_black_closing_offsets_and_carry_hold_do_not_change_open_commands():
    seq=gate_sequence(enabled=False)
    seq._near_field_pickup.closed_angles=(80,100)
    seq._transport_target_classes=(BLACK,GREEN)
    d=seq._decision(1,0.2,0,'carrying')
    assert d.gripper_angles_deg==(85,105)
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


def test_missing_motion_feedback_holds_clearance_action():
    seq=gate_sequence(position=FieldPoint(-350,800))
    seq._gate_clearance=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        (GREEN,),1,110,attempt_started_ns=1,
    )
    seq.state=MatchState.GATE_CLEARANCE
    seq._latest_heading_rad=None
    seq._latest_cumulative_distance_m=None
    decision=step_clearance(seq,2)
    assert decision.linear_velocity_m_s==0
    assert decision.angular_velocity_rad_s==0
    assert 'critical_motion_pose_unavailable' in decision.reason


def test_new_danger_intrusion_stops_sweep(monkeypatch):
    seq=gate_sequence(position=FieldPoint(-350,1115))
    session=make_clearance_session(
        seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,
        (GREEN,),1,110,attempt_started_ns=1,
    )
    session.index=5
    seq._gate_clearance=session
    monkeypatch.setattr(
        'rescue_vision.app.gate_clearance._objects',
        lambda _sequence,_now: (GateObject(BLUE,FieldPoint(0,1190),30),),
    )
    assert _sweep_risk(seq,2)=='danger_sweep_intersects_safe_zone'


@pytest.mark.parametrize('dt,latency,interval',[(.005,.6,.25),(.01,.3,.4)])
def test_delayed_perception_clearance_reaches_stash_again_without_counting_delivery(dt,latency,interval):
    seq=gate_sequence()
    original_count=seq._transport_count
    def scene(frame,stamp,pose):
        session=seq._gate_clearance
        points=[]
        if session is None or session.index<6:
            points.append((ORANGE,FieldPoint(-150,1100)))
        if session is not None and session.released:
            points.append((GREEN,session.stash))
        # 外围目标闪烁，不影响已提交的暂存或扫掠动作。
        if frame%2:
            points.append((BLACK,FieldPoint(900,500)))
        return snapshot(frame,stamp,*(observation(frame,stamp,field_to_ground(pose,p),
                        target_class=cls) for cls,p in points))
    plant=MotionPlant(seq,heading=math.pi/2,dt=dt,latency_s=latency,
                      frame_interval_s=interval,perception=scene)
    plant.until(lambda d:d.reason=='gate_clearance_triggered',seconds=3)
    session=seq._gate_clearance
    plant.until(lambda d:seq.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP,seconds=65)
    reasons=[d.reason for d in plant.records]
    for stage in ('stash_reverse_120mm','turn_180','sweep_via_midpoint','to_field_center',
                  'center_reverse_120mm','return_to_stash_s','face_stash'):
        assert f'gate_clearance:{stage}:complete' in reasons
    sweep=[d for d in plant.records if 'noncontact=gate_sweep_via_midpoint,' in d.reason]
    assert any(d.linear_velocity_m_s>0.3 for d in sweep)
    assert all(d.gripper_posture is GripperPosture.OPEN for d in sweep)
    assert seq.carried_target_count==0 and seq._transport_count==original_count
    assert session.reacquiring and seq._gate_clearance_attempted
    assert abs(seq.estimated_field_position.x+470)<30
    assert seq.near_field_handoff_prior.target_class is GREEN


def test_missing_stash_times_out_and_cannot_restart_same_sweep():
    seq=gate_sequence(position=FieldPoint(-350,1115))
    session=make_clearance_session(seq.config.gate_clearance,seq.estimated_field_position,TeamColor.RED,(GREEN,),0,110)
    session.index=13
    session.released=True
    session.recovery_attempted=True
    seq._gate_clearance=session
    seq._gate_clearance_attempted=True
    seq._transport_target_classes=()
    seq.state=MatchState.GATE_CLEARANCE
    plant=MotionPlant(seq,heading=math.pi)
    plant.until(lambda d:d.state is MatchState.SEARCH_CLUSTER,seconds=1.6)
    assert plant.decision.angular_velocity_rad_s!=0
    assert seq._gate_clearance is None and seq._gate_clearance_attempted
    seq._transport_target_classes=(GREEN,)
    assert maybe_begin_clearance(seq,plant.time_ns+1) is None
