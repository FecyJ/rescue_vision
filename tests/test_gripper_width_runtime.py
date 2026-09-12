"""独立入口装配回归：虚拟时间、帧源和UART，不打开设备或窗口。"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from rescue_vision.app import gripper_width as app
from rescue_vision.app.gripper_width_sequence import GraspPreparationSession
from rescue_vision.config import load_runtime_config
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.perception import PerceptionSnapshot
from rescue_vision.tracking import TrackingConfig
from test_near_field_grasp import target, projector


@pytest.mark.parametrize('once',[True,False])
@pytest.mark.parametrize('latency_ms', [0, 350])
def test_runtime_finishes_holds_and_releases_resources(monkeypatch,tmp_path,once,latency_ms):
    clock=NS(now=1_000_000_000)
    hardware=NS(distance=0.,speed=0.,closed=False,stopped=False,frame=0,commands=[],logs=[],completed=False)

    from test_gripper_width_sequence import motion_sample

    def odom():
        return motion_sample(clock.now, count=round(hardware.distance * 1000))
    class Status:
        emergency_stop_latched=False
        protocol_ready=True
        reply_queue_full=False
        tx_degraded=False
        rx_degraded=True  # 诊断状态不能独立阻止本次动作。
        gripper_output_available=True

    class Channel:
        def __enter__(self): return self
        def __exit__(self,*_): hardware.closed=True

    class Controller:
        def update(self,**_): pass
        def drain_messages(self): return (odom(),Status())
        def synchronize(self,*,on_message,**_):
            on_message(odom()); on_message(Status())
        def query_state(self): pass
        def soft_brake(self): hardware.speed=0
        def drive_wheel_limited(self,speed,angular,**_kwargs):
            assert angular==0
            hardware.speed=speed
        def set_gripper_angles(self,*angles): hardware.commands.append(angles)

    class Encoder:
        def __init__(self,*_,**__): pass
        def submit(self,_): pass
        @property
        def distance_m(self): return hardware.distance

    class Pump:
        def __init__(self,*_): pass
        def start_in_background(self): return object()
        def wait_until_started(self,_,*,on_wait): on_wait()
        def check_health(self): pass
        def stop(self): hardware.stopped=True

    class Renderer:
        def __init__(self,*_,**__): pass
        def latest_snapshot(self):
            hardware.frame+=1
            obs=target(x=300-hardware.distance*1000,timestamp=clock.now-latency_ms*1_000_000-1,frame=hardware.frame).observation
            return PerceptionSnapshot(hardware.frame,obs.capture_timestamp_ns,clock.now,(obs,),None)
        def latest_fresh_snapshot(self, now_ns, max_age_ms):
            return self.latest_snapshot()

    class Worker:
        def __init__(self,session,selector,renderer,local_preview):
            assert isinstance(session,GraspPreparationSession)
            self.session=session; self.value=None
            self.exit_requested=False; self.error=None
        def __enter__(self): return self
        def __exit__(self,*_): hardware.completed=True
        def submit(self,snapshot,ids,*,excluded_observation_indices=frozenset()):
            self.value=self.session.update(
                snapshot,
                locked_ids=ids,
                excluded_observation_indices=excluded_observation_indices,
            )
        def latest(self): return self.value
        def log(self,text,**_):
            hardware.logs.append(text)
            if 'pickup_result=' in text and not once:
                self.exit_requested=True

    calibration=NS(open_left_angle_deg=0,open_right_angle_deg=180,closed_left_angle_deg=90,closed_right_angle_deg=90,full_travel_time_s=.1)
    motion=NS(enabled=True,gripper=NS(enabled=True,build_calibration=lambda:calibration),
        odometry=NS(build_calibration=lambda:object(),max_consecutive_overrun_samples=1),wheel_track_m=.2,
        max_wheel_velocity_m_s=.2,build_controller=lambda channel:Controller(),synchronization_timeout_s=1)
    config=NS(near_field_grasp=NearFieldGraspConfig(),hailo=NS(enabled=True),motion=motion,
        uart=NS(enabled=True,build_channel=Channel),tracking=TrackingConfig(1,80,.1,500,1,.1),
        perception=load_runtime_config("configs/runtime.match.yaml").perception,
        processing=NS(max_observation_age_ms=500),match=NS(robot_footprint_radius_mm=160,
        green_approach_speed_m_s=.1,green_alignment_kp_rad_s=1,
        green_alignment_max_angular_velocity_rad_s=.35,
        green_alignment_min_wheel_velocity_m_s=.01))
    monkeypatch.setattr(app,'load_runtime_config',lambda _:config)
    monkeypatch.setattr('rescue_vision.app.manual_capture.build_camera_pipeline',lambda _:NS(ground_projector=projector(),source=object(),prepare=lambda x:x))
    for name,value in [('CarSystemStatus',Status),('EncoderTravelTracker',Encoder),('CameraPerceptionPump',Pump),('PerceptionFrameRenderer',Renderer),('_PreparationWorker',Worker)]:
        monkeypatch.setattr(app,name,value)
    monkeypatch.setattr(app.time,'monotonic_ns',lambda:clock.now)
    def sleep(_):
        clock.now+=10_000_000
        hardware.distance+=hardware.speed*.01
        assert clock.now<12_000_000_000,'runtime failed to finish'
    monkeypatch.setattr(app.time,'sleep',sleep)
    app._run(tmp_path/'unused.yaml',supervised_stop_ready=True,local_preview=False,once=once)
    assert hardware.closed and hardware.stopped and hardware.completed
    assert hardware.speed==0 and len(hardware.commands)==2
    assert hardware.commands[-1]==(90,90)
    assert any('capture_confirmed=False' in text for text in hardware.logs)


def test_entrypoint_can_tee_logs_to_configured_directory(monkeypatch, tmp_path, capsys):
    def fake_session(*_args, **_kwargs):
        print("grasp_candidate track_id=7 x0_mm=1.00 x1_mm=2.00")

    monkeypatch.setattr(app, "_run_session", fake_session)
    log_dir = tmp_path / "field-logs"

    app._run(
        tmp_path / "unused.yaml",
        supervised_stop_ready=True,
        local_preview=False,
        once=True,
        log_dir=log_dir,
    )

    captured = capsys.readouterr()
    files = tuple(log_dir.glob("gripper_width_*.log"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "grasp_candidate track_id=7" in text
    assert "grasp_candidate track_id=7" in captured.out
