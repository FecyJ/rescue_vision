from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rescue_vision.app.manual_capture import (
    CameraPipeline,
    CaptureSession,
    _validate_mode,
    build_session_status,
    run_manual_capture_session,
)
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.communication import (
    CaptureAction,
    CaptureStopReason,
    DebugCaptureCommand,
    DebugMotionCommand,
    ImageCoordinateSystem,
    MotionControlMode,
    ReceivedRemoteMessage,
    ReceivedUartLine,
    RemoteAccessMode,
    RemoteDisconnectedError,
    RemoteRole,
    RemoteStream,
    RemoteTopic,
    VehicleSafetyMode,
    VehicleStateObservation,
)
from rescue_vision.data.check_recording import inspect_recording
from rescue_vision.motion import MotionController, MotionLimits, RemoteMotionExecutor


class FakeSource:
    def __init__(self, frame: CameraFrame) -> None:
        self.frame = frame
        self.read_count = 0

    def read(self, timeout: float | None = None) -> CameraFrame:
        del timeout
        result = CameraFrame(
            sequence=self.frame.sequence + self.read_count,
            timestamp_ns=time.monotonic_ns(),
            image_bgr=self.frame.image_bgr,
            metadata=self.frame.metadata,
        )
        self.read_count += 1
        return result

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FailingSource(FakeSource):
    def read(self, timeout: float | None = None) -> CameraFrame:
        del timeout
        raise OSError("camera failed")


class FakeCarChannel:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send_line(self, payload: bytes) -> None:
        self.sent.append(payload)

    def receive_line(self, timeout: float | None = None):
        del timeout
        raise TimeoutError


class TelemetryAfterMotionChannel(FakeCarChannel):
    def __init__(self) -> None:
        super().__init__()
        self.telemetry_sent = False

    def receive_line(
        self,
        timeout: float | None = None,
    ) -> ReceivedUartLine:
        del timeout
        if (
            not self.telemetry_sent
            and any(payload.startswith(b"m") for payload in self.sent)
        ):
            self.telemetry_sent = True
            return ReceivedUartLine(
                sequence=0,
                received_timestamp_ns=time.monotonic_ns(),
                payload=b"t1,0.1,0.1,0.1,0.1,90,90",
            )
        raise TimeoutError


class FakeConnection:
    def __init__(self, controls: list[ReceivedRemoteMessage]) -> None:
        self.controls = list(controls)
        self.observations: list[tuple[str, bytes]] = []

    def receive_control(self, timeout: float | None = None):
        del timeout
        if self.controls:
            return self.controls.pop(0)
        raise RemoteDisconnectedError("test disconnect")

    def send_reliable_observation(
        self,
        topic: str,
        payload: bytes,
        **_kwargs: object,
    ) -> None:
        self.observations.append((topic, payload))

    def send_observation(
        self,
        topic: str,
        payload: bytes,
        **_kwargs: object,
    ) -> None:
        self.observations.append((topic, payload))


def _received(topic: RemoteTopic, payload: bytes, sequence: int):
    return ReceivedRemoteMessage(
        stream=RemoteStream.CONTROL,
        topic=topic.value,
        content_type="application/json",
        sequence=sequence,
        sender_timestamp_ns=1,
        received_timestamp_ns=time.monotonic_ns(),
        attributes={},
        payload=payload,
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        schema_version=8,
        camera=SimpleNamespace(image_size=(4, 3), fps=20),
        recording=SimpleNamespace(queue_capacity=8, image_format="png"),
        remote=SimpleNamespace(access_mode=RemoteAccessMode.DEBUG_CONTROL),
        motion=SimpleNamespace(
            max_linear_velocity_m_s=0.25,
            max_angular_velocity_rad_s=1.0,
            max_remote_command_valid_for_ms=500,
        ),
    )


def test_manual_session_routes_capture_and_motion_then_stops_on_disconnect(
    tmp_path,
) -> None:
    config = _config()
    frame = CameraFrame(
        sequence=4,
        timestamp_ns=123,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    pipeline = CameraPipeline(
        FakeSource(frame),
        None,
        ImageCoordinateSystem.RAW_PIXEL,
        None,
    )
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={"schema_version": 8},
        pipeline=pipeline,
    )
    start = DebugCaptureCommand(
        request_id="start-1",
        issued_timestamp_ns=1,
        action=CaptureAction.START,
    )
    motion = DebugMotionCommand(
        command_id="drive-1",
        issued_timestamp_ns=1,
        valid_for_ms=500,
        deadman_enabled=True,
        control_mode=MotionControlMode.TWIST,
        linear_velocity_m_s=0.1,
        angular_velocity_rad_s=0.0,
    )
    connection = FakeConnection(
        [
            _received(RemoteTopic.DEBUG_CAPTURE, start.to_payload(), 0),
            _received(RemoteTopic.DEBUG_MOTION, motion.to_payload(), 1),
        ]
    )
    car = TelemetryAfterMotionChannel()
    executor = RemoteMotionExecutor(
        MotionController(
            car,
            MotionLimits(0.2, 0.25, 1.0, 0.3, 500),
        )
    )
    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=10.0,
    )

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            executor,
            capture,
            pipeline,
            status,
            video_fps=10.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )

    assert car.sent[-2:] == [b"m0.1,0.1", b"b0,0"]
    recordings = list((tmp_path / "recordings").iterdir())
    assert len(recordings) == 1
    assert capture.recorder is None
    assert capture.stop_reason is not None
    recording_report = inspect_recording(recordings[0])
    motion_report = recording_report["auxiliary_stream_reports"][
        "manual_motion"
    ]
    assert motion_report["event_counts"]["motion_command"] == 1
    assert motion_report["event_counts"]["wheel_telemetry"] == 1
    assert motion_report["event_counts"]["safety_stop"] == 1
    assert motion_report["covers_frame_time_range"] is True
    session_path = recordings[0] / "session.json"
    session_document = json.loads(session_path.read_text(encoding="utf-8"))
    session_document["auxiliary_streams"] = {}
    session_path.write_text(
        json.dumps(session_document),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires exactly"):
        inspect_recording(recordings[0])
    assert {
        topic for topic, _payload in connection.observations
    } >= {
        RemoteTopic.SESSION_STATUS.value,
        RemoteTopic.VEHICLE_STATE.value,
        RemoteTopic.CAPTURE_STATUS.value,
        RemoteTopic.VIDEO_FRAME.value,
    }
    vehicle_payload = next(
        payload
        for topic, payload in connection.observations
        if topic == RemoteTopic.VEHICLE_STATE.value
    )
    vehicle_state = VehicleStateObservation.from_payload(vehicle_payload)
    assert vehicle_state.control_ready
    assert (
        vehicle_state.safety_mode
        is VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP
    )
    assert not vehicle_state.watchdog_armed

    second_capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={"schema_version": 8},
        pipeline=pipeline,
    )
    second_connection = FakeConnection([])
    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            second_connection,
            executor,
            second_capture,
            pipeline,
            build_session_status(
                config,
                server_instance_id="test-server",
                video_fps=10.0,
            ),
            video_fps=10.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )
    second_vehicle_payload = next(
        payload
        for topic, payload in second_connection.observations
        if topic == RemoteTopic.VEHICLE_STATE.value
    )
    second_vehicle = VehicleStateObservation.from_payload(
        second_vehicle_payload
    )
    assert second_vehicle.last_received_motion_command_id is None
    assert second_vehicle.last_applied_motion_command_id is None
    assert car.sent[-1] == b"b0,0"


def test_recording_queue_overflow_faults_capture_and_requires_stop(
    tmp_path,
    monkeypatch,
) -> None:
    config = _config()
    pipeline = CameraPipeline(
        FakeSource(
            CameraFrame(
                sequence=0,
                timestamp_ns=0,
                image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            )
        ),
        None,
        ImageCoordinateSystem.RAW_PIXEL,
        None,
    )
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={"schema_version": 8},
        pipeline=pipeline,
    )
    capture.execute(
        DebugCaptureCommand(
            request_id="start",
            issued_timestamp_ns=1,
            action=CaptureAction.START,
        ),
        None,
    )
    assert capture.recorder is not None
    monkeypatch.setattr(capture.recorder, "record", lambda _frame: False)
    car = FakeCarChannel()
    executor = RemoteMotionExecutor(
        MotionController(
            car,
            MotionLimits(0.2, 0.25, 1.0, 0.3, 500),
        )
    )

    with pytest.raises(RuntimeError, match="queue overflowed"):
        run_manual_capture_session(
            FakeConnection([]),
            executor,
            capture,
            pipeline,
            build_session_status(
                config,
                server_instance_id="test-server",
                video_fps=10.0,
            ),
            video_fps=10.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )

    assert capture.faulted
    assert capture.recorder is None
    assert car.sent == [b"b0,0"]


def test_manual_capture_rejects_competition_observe_only_mode() -> None:
    config = _config()
    config.remote = SimpleNamespace(
        enabled=True,
        role=RemoteRole.SERVER,
        access_mode=RemoteAccessMode.OBSERVE_ONLY,
    )
    config.uart = SimpleNamespace(enabled=True)
    config.motion.enabled = True

    with pytest.raises(RuntimeError, match="server mode|debug_control"):
        _validate_mode(
            config,
            video_fps=10.0,
            supervised_physical_stop_ready=True,
        )


def test_manual_capture_requires_explicit_physical_stop_acknowledgement() -> None:
    config = _config()
    config.remote = SimpleNamespace(
        enabled=True,
        role=RemoteRole.SERVER,
        access_mode=RemoteAccessMode.DEBUG_CONTROL,
    )
    config.uart = SimpleNamespace(enabled=True)
    config.motion.enabled = True

    with pytest.raises(RuntimeError, match="physical emergency stop"):
        _validate_mode(
            config,
            video_fps=10.0,
            supervised_physical_stop_ready=False,
        )


def test_camera_failure_faults_capture_and_stops_motion(tmp_path) -> None:
    config = _config()
    frame = CameraFrame(
        sequence=0,
        timestamp_ns=0,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    pipeline = CameraPipeline(
        FailingSource(frame),
        None,
        ImageCoordinateSystem.RAW_PIXEL,
        None,
    )
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={"schema_version": 8},
        pipeline=pipeline,
    )
    car = FakeCarChannel()
    executor = RemoteMotionExecutor(
        MotionController(
            car,
            MotionLimits(0.2, 0.25, 1.0, 0.3, 500),
        )
    )

    with pytest.raises(OSError, match="camera failed"):
        run_manual_capture_session(
            FakeConnection([]),
            executor,
            capture,
            pipeline,
            build_session_status(
                config,
                server_instance_id="test-server",
                video_fps=10.0,
            ),
            video_fps=10.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )

    assert capture.faulted
    assert capture.stop_reason is CaptureStopReason.CAMERA_ERROR
    assert car.sent == [b"b0,0"]
