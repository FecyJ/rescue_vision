from __future__ import annotations

import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import rescue_vision.app.manual_capture as manual_capture_module
from rescue_vision.app.manual_capture import (
    BevFrameRenderer,
    CameraPipeline,
    CaptureSession,
    VehicleState,
    _accept_with_shutdown,
    _validate_mode,
    build_session_status,
    run_manual_capture_session,
)
from rescue_vision.app.field_map import FieldMapSnapshotRenderer
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.communication import (
    CaptureAction,
    CaptureStopReason,
    DebugCaptureCommand,
    DebugGripperCommand,
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
    VideoFrameMode,
    VideoModeCommand,
)
from rescue_vision.data.check_recording import inspect_recording
from rescue_vision.motion import (
    ExecutedRemoteGripper,
    ExecutedRemoteMotion,
    GripperCalibration,
    MotionController,
    MotionLimits,
    RemoteGripperExecutor,
    RemoteGripperResult,
    RemoteMotionExecutor,
    RemoteMotionResult,
)
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.world import (
    PhysicalRegionKind,
    PhysicalStaticRegion,
    TeamColor,
    default_static_field_map,
)
from rescue_vision.world.static_map import StaticFieldMap


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
    def __init__(
        self,
        controls: list[ReceivedRemoteMessage],
        *,
        empty_polls_before_disconnect: int = 0,
    ) -> None:
        self.controls = list(controls)
        self.empty_polls_before_disconnect = empty_polls_before_disconnect
        self.observations: list[tuple[str, bytes]] = []
        self.observation_attributes: list[tuple[str, dict[str, object]]] = []

    def receive_control(self, timeout: float | None = None):
        del timeout
        if self.controls:
            return self.controls.pop(0)
        if self.empty_polls_before_disconnect > 0:
            self.empty_polls_before_disconnect -= 1
            raise TimeoutError
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
        **kwargs: object,
    ) -> None:
        self.observations.append((topic, payload))
        attributes = kwargs.get("attributes")
        self.observation_attributes.append(
            (
                topic,
                dict(attributes) if isinstance(attributes, dict) else {},
            )
        )


class FakePerceptionRenderer:
    def __init__(self) -> None:
        self.latest_frame: CameraFrame | None = None

    def submit(self, frame: CameraFrame) -> None:
        image = frame.image_bgr.copy()
        image[0, 0] = (255, 0, 255)
        self.latest_frame = CameraFrame(
            sequence=frame.sequence,
            timestamp_ns=frame.timestamp_ns,
            image_bgr=image,
        )

    def latest(self) -> CameraFrame | None:
        return self.latest_frame

    def clear_latest(self) -> None:
        self.latest_frame = None


class FakeBevRenderer(FakePerceptionRenderer):
    pass


class LaggingFakePerceptionRenderer(FakePerceptionRenderer):
    """模拟推理结果比当前相机帧落后一帧。"""

    def __init__(self) -> None:
        super().__init__()
        self.pending_frame: CameraFrame | None = None

    def submit(self, frame: CameraFrame) -> None:
        if self.pending_frame is not None:
            self.latest_frame = self.pending_frame
        image = frame.image_bgr.copy()
        image[0, 0] = (255, 0, 255)
        self.pending_frame = CameraFrame(
            sequence=frame.sequence,
            timestamp_ns=frame.timestamp_ns,
            image_bgr=image,
        )

    def clear_latest(self) -> None:
        self.latest_frame = None
        self.pending_frame = None


class PollingServer:
    def __init__(self, connection: object, *, timeouts_before_connection: int) -> None:
        self.connection = connection
        self.timeouts_before_connection = timeouts_before_connection
        self.accept_timeouts: list[float] = []

    def accept(self, timeout: float | None = None) -> object:
        assert timeout is not None
        self.accept_timeouts.append(timeout)
        if len(self.accept_timeouts) <= self.timeouts_before_connection:
            raise TimeoutError
        return self.connection


class CountingDrainController:
    def __init__(self) -> None:
        self.drain_count = 0

    def drain_messages(self) -> tuple[object, ...]:
        self.drain_count += 1
        return ()


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

        camera=SimpleNamespace(image_size=(4, 3), fps=20),
        recording=SimpleNamespace(queue_capacity=8, image_format="png"),
        remote=SimpleNamespace(access_mode=RemoteAccessMode.DEBUG_CONTROL),
        motion=SimpleNamespace(
            max_linear_velocity_m_s=0.25,
            max_angular_velocity_rad_s=1.0,
            max_remote_command_valid_for_ms=500,
            gripper=SimpleNamespace(enabled=True),
        ),
    )


def _gripper_calibration() -> GripperCalibration:
    return GripperCalibration(
        open_left_angle_deg=20.0,
        open_right_angle_deg=174.0,
        closed_left_angle_deg=80.0,
        closed_right_angle_deg=114.0,
        full_travel_time_s=1.0,
    )


def test_accept_wait_drains_uart_between_tcp_polls() -> None:
    connection = object()
    server = PollingServer(connection, timeouts_before_connection=2)
    controller = CountingDrainController()

    accepted = _accept_with_shutdown(
        server,  # type: ignore[arg-type]
        controller,  # type: ignore[arg-type]
        timeout_s=1.0,
        stop_requested=lambda: False,
    )

    assert accepted is connection
    assert controller.drain_count == 3
    assert server.accept_timeouts == [0.1, 0.1, 0.1]


def test_session_status_only_advertises_configured_gripper() -> None:
    config = _config()
    config.motion.gripper.enabled = False

    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=10.0,
    )

    assert not status.gripper_control_available


def test_camera_only_status_disables_actuators_but_keeps_vehicle_heartbeat() -> None:
    status = build_session_status(
        _config(),
        server_instance_id="test-server",
        video_fps=10.0,
        video_modes=(VideoFrameMode.RAW, VideoFrameMode.BEV),
        camera_only=True,
    )

    assert not status.motion_control_available
    assert not status.gripper_control_available
    assert status.vehicle_state_available
    assert status.vehicle_state_period_ms == 100
    assert status.max_linear_velocity_m_s is None
    assert status.max_angular_velocity_rad_s is None
    assert status.capture_control_available
    assert status.video_modes == (VideoFrameMode.RAW, VideoFrameMode.BEV)


def test_camera_only_session_publishes_static_map_with_unlocalized_robot(
    tmp_path,
) -> None:
    config = _config()
    pipeline = CameraPipeline(
        FakeSource(CameraFrame(0, 0, np.zeros((3, 4, 3), np.uint8))),
        None,
        ImageCoordinateSystem.RAW_PIXEL,
        None,
    )
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={},
        pipeline=pipeline,
    )
    landmarks = default_static_field_map()
    static_map = StaticFieldMap(
        landmarks.center_cross,
        (
            PhysicalStaticRegion(
                "field",
                PhysicalRegionKind.FIELD,
                (
                    FieldPoint(-100.0, -100.0),
                    FieldPoint(100.0, -100.0),
                    FieldPoint(100.0, 100.0),
                    FieldPoint(-100.0, 100.0),
                ),
            ),
        ),
    )
    renderer = FieldMapSnapshotRenderer(static_map, TeamColor.UNKNOWN)
    connection = FakeConnection([])
    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=10.0,
        camera_only=True,
        map_snapshot_available=True,
    )

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            None,
            None,
            capture,
            pipeline,
            status,
            video_fps=10.0,
            jpeg_quality=80,
            map_renderer=renderer,
        )

    map_payload = next(
        payload
        for topic, payload in connection.observations
        if topic == RemoteTopic.MAP_SNAPSHOT.value
    )
    map_attributes = next(
        attributes
        for topic, attributes in connection.observation_attributes
        if topic == RemoteTopic.MAP_SNAPSHOT.value
    )
    assert (
        cv2.imdecode(np.frombuffer(map_payload, np.uint8), cv2.IMREAD_COLOR)
        is not None
    )
    assert map_attributes["coordinate_system"] == "field_mm"
    assert map_attributes["robot_localized"] is False
    assert map_attributes["robot_x_mm"] is None


def test_accept_timeout_is_a_clean_stop(monkeypatch) -> None:
    moments = iter((10.0, 11.0))
    monkeypatch.setattr(
        manual_capture_module.time,
        "monotonic",
        lambda: next(moments),
    )
    server = PollingServer(object(), timeouts_before_connection=0)
    controller = CountingDrainController()

    accepted = _accept_with_shutdown(
        server,  # type: ignore[arg-type]
        controller,  # type: ignore[arg-type]
        timeout_s=0.5,
        stop_requested=lambda: False,
    )

    assert accepted is None
    assert controller.drain_count == 1
    assert server.accept_timeouts == []


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
        config_snapshot={},
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
    gripper = DebugGripperCommand(
        command_id="grip-1",
        issued_timestamp_ns=1,
        valid_for_ms=500,
        open_pressed=False,
        close_pressed=True,
    )
    connection = FakeConnection(
        [
            _received(RemoteTopic.DEBUG_CAPTURE, start.to_payload(), 0),
            _received(RemoteTopic.DEBUG_MOTION, motion.to_payload(), 1),
            _received(RemoteTopic.DEBUG_GRIPPER, gripper.to_payload(), 2),
        ]
    )
    car = TelemetryAfterMotionChannel()
    executor = RemoteMotionExecutor(
        MotionController(
            car,
            MotionLimits(0.2, 0.25, 1.0, 0.3, 0.5, 500),
        )
    )
    gripper_executor = RemoteGripperExecutor(
        executor.controller,
        _gripper_calibration(),
    )
    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=10.0,
    )
    assert status.gripper_control_available
    assert status.max_control_command_valid_for_ms == 500

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            executor,
            gripper_executor,
            capture,
            pipeline,
            status,
            video_fps=10.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )

    assert len(car.sent) >= 3
    motion_payload = next(
        payload for payload in car.sent if payload.startswith(b"m")
    )
    assert motion_payload.startswith(b"m")
    limited_left, limited_right = (
        float(value) for value in motion_payload[1:].split(b",")
    )
    assert 0.0 < limited_left < 0.1
    assert limited_right == pytest.approx(limited_left)
    assert any(payload.startswith(b"g") for payload in car.sent)
    assert car.sent[-1] == b"b0,0"


    recordings = list((tmp_path / "recordings").iterdir())
    assert len(recordings) == 1
    assert capture.recorder is None
    assert capture.stop_reason is not None
    recording_report = inspect_recording(recordings[0])
    motion_report = recording_report["auxiliary_stream_reports"][
        "manual_motion"
    ]
    assert motion_report["event_counts"]["motion_command"] == 1
    assert motion_report["event_counts"]["gripper_command"] == 1
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
        config_snapshot={},
        pipeline=pipeline,
    )
    second_connection = FakeConnection([])
    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            second_connection,
            executor,
            gripper_executor,
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


def test_camera_only_session_keeps_video_and_offline_vehicle_heartbeat(tmp_path) -> None:
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
        config_snapshot={},
        pipeline=pipeline,
        motion_logging_enabled=False,
    )
    connection = FakeConnection([], empty_polls_before_disconnect=5)
    status = build_session_status(
        config,
        server_instance_id="camera-only-server",
        video_fps=1_000.0,
        camera_only=True,
    )

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            None,
            None,
            capture,
            pipeline,
            status,
            video_fps=1_000.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.UNAVAILABLE,
        )

    topics = [topic for topic, _payload in connection.observations]
    assert RemoteTopic.SESSION_STATUS.value in topics
    assert RemoteTopic.VIDEO_FRAME.value in topics
    assert RemoteTopic.CAPTURE_STATUS.value in topics
    assert RemoteTopic.VEHICLE_STATE.value in topics
    vehicle_payload = next(
        payload
        for topic, payload in connection.observations
        if topic == RemoteTopic.VEHICLE_STATE.value
    )
    vehicle = VehicleStateObservation.from_payload(vehicle_payload)
    assert not vehicle.control_ready
    assert not vehicle.uart_connected
    assert vehicle.safety_mode is VehicleSafetyMode.UNAVAILABLE
    assert vehicle.stop_reason.value == "uart_fault"


def test_camera_only_session_ignores_valid_legacy_actuator_heartbeats(tmp_path) -> None:
    config = _config()
    frame = CameraFrame(
        sequence=0,
        timestamp_ns=0,
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
        config_snapshot={},
        pipeline=pipeline,
        motion_logging_enabled=False,
    )
    motion = DebugMotionCommand(
        command_id="legacy-drive",
        issued_timestamp_ns=1,
        valid_for_ms=500,
        deadman_enabled=True,
        control_mode=MotionControlMode.TWIST,
        linear_velocity_m_s=0.1,
        angular_velocity_rad_s=0.0,
    )
    gripper = DebugGripperCommand(
        command_id="legacy-grip",
        issued_timestamp_ns=2,
        valid_for_ms=500,
        open_pressed=True,
        close_pressed=False,
    )
    connection = FakeConnection(
        [
            _received(RemoteTopic.DEBUG_MOTION, motion.to_payload(), 0),
            _received(RemoteTopic.DEBUG_GRIPPER, gripper.to_payload(), 1),
        ]
    )
    status = build_session_status(
        config,
        server_instance_id="camera-only-server",
        video_fps=10.0,
        camera_only=True,
    )

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            None,
            None,
            capture,
            pipeline,
            status,
            video_fps=10.0,
            jpeg_quality=80,
            safety_mode=VehicleSafetyMode.UNAVAILABLE,
        )

    vehicle_payload = next(
        payload
        for topic, payload in connection.observations
        if topic == RemoteTopic.VEHICLE_STATE.value
    )
    vehicle = VehicleStateObservation.from_payload(vehicle_payload)
    assert vehicle.last_received_motion_command_id is None
    assert vehicle.last_received_gripper_command_id is None


def test_camera_only_recording_uses_camera_schema_without_motion_log(tmp_path) -> None:
    config = _config()
    frame = CameraFrame(
        sequence=0,
        timestamp_ns=time.monotonic_ns(),
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
        config_snapshot={},
        pipeline=pipeline,
        motion_logging_enabled=False,
    )
    started = capture.execute(
        DebugCaptureCommand(
            request_id="camera-start",
            issued_timestamp_ns=1,
            action=CaptureAction.START,
        ),
        frame,
    )
    assert started.artifact_id is not None
    capture.record(frame)
    capture.execute(
        DebugCaptureCommand(
            request_id="camera-stop",
            issued_timestamp_ns=2,
            action=CaptureAction.STOP,
        ),
        frame,
    )

    recording = tmp_path / "recordings" / started.artifact_id
    report = inspect_recording(recording)
    assert report["recording_kind"] == "camera"
    assert report["auxiliary_stream_reports"] == {}
    assert not (recording / "motion.jsonl").exists()


def test_manual_session_switches_between_raw_and_perception_video(tmp_path) -> None:
    config = _config()
    source_frame = CameraFrame(
        sequence=4,
        timestamp_ns=1,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    pipeline = CameraPipeline(
        FakeSource(source_frame),
        None,
        ImageCoordinateSystem.RAW_PIXEL,
        None,
    )
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={},
        pipeline=pipeline,
    )
    mode_command = VideoModeCommand(
        request_id="video-1",
        issued_timestamp_ns=1,
        mode=VideoFrameMode.PERCEPTION,
    )
    connection = FakeConnection(
        [_received(RemoteTopic.VIDEO_MODE, mode_command.to_payload(), 0)],
        empty_polls_before_disconnect=100,
    )
    car = FakeCarChannel()
    executor = RemoteMotionExecutor(
        MotionController(
            car,
            MotionLimits(0.2, 0.25, 1.0, 0.3, 0.5, 500),
        )
    )
    renderer = LaggingFakePerceptionRenderer()
    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=1_000.0,
        video_modes=(VideoFrameMode.RAW, VideoFrameMode.PERCEPTION),
    )

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            executor,
            None,
            capture,
            pipeline,
            status,
            video_fps=1_000.0,
            jpeg_quality=80,
            perception_renderer=renderer,  # type: ignore[arg-type]
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )

    modes = [
        attributes["mode"]
        for topic, attributes in connection.observation_attributes
        if topic == RemoteTopic.VIDEO_FRAME.value
    ]
    assert "raw" in modes
    assert "perception" in modes


def test_bev_renderer_keeps_latest_result_off_motion_loop() -> None:
    projector = GroundProjector(
        np.eye(3),
        BevConfig(0.0, 30.0, 0.0, 40.0, 10.0),
    )
    renderer = BevFrameRenderer(projector)
    source = CameraFrame(
        sequence=7,
        timestamp_ns=11,
        image_bgr=np.zeros((30, 40, 3), dtype=np.uint8),
    )
    renderer.start()
    try:
        renderer.submit(source)
        deadline = time.monotonic() + 1.0
        rendered = None
        while time.monotonic() < deadline:
            rendered = renderer.latest()
            if rendered is not None:
                break
            time.sleep(0.001)
        assert rendered is not None
        assert rendered.sequence == source.sequence
        assert rendered.timestamp_ns == source.timestamp_ns
        assert rendered.image_bgr.shape == (3, 4, 3)
    finally:
        renderer.stop()


def test_manual_session_publishes_bev_with_robot_ground_mapping(tmp_path) -> None:
    config = _config()
    source_frame = CameraFrame(
        sequence=4,
        timestamp_ns=1,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    projector = GroundProjector(
        np.eye(3),
        BevConfig(0.0, 30.0, 0.0, 40.0, 10.0),
    )
    pipeline = CameraPipeline(
        FakeSource(source_frame),
        None,
        ImageCoordinateSystem.UNDISTORTED_PIXEL,
        "test-calibration",
        projector,
    )
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={},
        pipeline=pipeline,
    )
    command = VideoModeCommand(
        request_id="video-bev",
        issued_timestamp_ns=1,
        mode=VideoFrameMode.BEV,
    )
    connection = FakeConnection(
        [_received(RemoteTopic.VIDEO_MODE, command.to_payload(), 0)],
        empty_polls_before_disconnect=100,
    )
    executor = RemoteMotionExecutor(
        MotionController(
            FakeCarChannel(),
            MotionLimits(0.2, 0.25, 1.0, 0.3, 0.5, 500),
        )
    )
    renderer = FakeBevRenderer()
    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=1_000.0,
        video_modes=(VideoFrameMode.RAW, VideoFrameMode.BEV),
    )

    with pytest.raises(RemoteDisconnectedError):
        run_manual_capture_session(
            connection,
            executor,
            None,
            capture,
            pipeline,
            status,
            video_fps=1_000.0,
            jpeg_quality=80,
            bev_renderer=renderer,  # type: ignore[arg-type]
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        )

    bev_attributes = next(
        attributes
        for topic, attributes in connection.observation_attributes
        if topic == RemoteTopic.VIDEO_FRAME.value
        and attributes.get("mode") == "bev"
    )
    assert bev_attributes["coordinate_system"] == "bev_pixel"
    assert bev_attributes["width"] == 4
    assert bev_attributes["height"] == 3
    assert bev_attributes["bev_x_max_mm"] == pytest.approx(30.0)
    assert bev_attributes["bev_y_max_mm"] == pytest.approx(40.0)
    assert bev_attributes["bev_mm_per_pixel"] == pytest.approx(10.0)


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
        config_snapshot={},
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
            MotionLimits(0.2, 0.25, 1.0, 0.3, 0.5, 500),
        )
    )
    gripper_executor = RemoteGripperExecutor(
        executor.controller,
        _gripper_calibration(),
    )

    with pytest.raises(RuntimeError, match="queue overflowed"):
        run_manual_capture_session(
            FakeConnection([]),
            executor,
            gripper_executor,
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


def test_vehicle_state_cannot_claim_unverified_firmware_watchdog() -> None:
    with pytest.raises(ValueError, match="fresh CarSafetyStatus"):
        VehicleState(safety_mode=VehicleSafetyMode.FIRMWARE_WATCHDOG)


def test_vehicle_state_reports_zero_twist_as_braking() -> None:
    vehicle = VehicleState(
        safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP
    )
    vehicle.on_motion(
        ExecutedRemoteMotion(
            command_id="centered-1",
            result=RemoteMotionResult.APPLIED,
            received_timestamp_ns=10,
            deadline_timestamp_ns=210,
            deadman_enabled=True,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
        )
    )

    observation = vehicle.observation()
    assert observation.motion_state.value == "braking"
    assert observation.stop_reason.value == "none"
    assert observation.last_applied_motion_command_id == "centered-1"


def test_vehicle_state_tracks_gripper_command_and_firmware_angles() -> None:
    vehicle = VehicleState(
        safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP
    )
    vehicle.on_gripper(
        ExecutedRemoteGripper(
            command_id="grip-1",
            result=RemoteGripperResult.APPLIED,
            received_timestamp_ns=10,
            deadline_timestamp_ns=210,
            open_pressed=False,
            close_pressed=True,
        )
    )
    vehicle.on_car_message(
        manual_capture_module.CarTelemetry(
            uart_sequence=1,
            received_timestamp_ns=20,
            controller_timestamp_ms=5,
            actual_left_m_s=0.0,
            actual_right_m_s=0.0,
            target_left_m_s=0.0,
            target_right_m_s=0.0,
            servo_left_deg=27.0,
            servo_right_deg=167.0,
        )
    )

    observation = vehicle.observation()
    assert observation.gripper_left_angle_deg == pytest.approx(27.0)
    assert observation.gripper_right_angle_deg == pytest.approx(167.0)
    assert observation.last_received_gripper_command_id == "grip-1"
    assert observation.last_applied_gripper_command_id == "grip-1"


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


def test_camera_only_mode_does_not_require_uart_or_physical_stop() -> None:
    config = _config()
    config.remote = SimpleNamespace(
        enabled=True,
        role=RemoteRole.SERVER,
        access_mode=RemoteAccessMode.DEBUG_CONTROL,
    )
    config.uart = SimpleNamespace(enabled=False)
    config.motion.enabled = False

    _validate_mode(
        config,
        video_fps=10.0,
        supervised_physical_stop_ready=False,
        camera_only=True,
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
        config_snapshot={},
        pipeline=pipeline,
    )
    car = FakeCarChannel()
    executor = RemoteMotionExecutor(
        MotionController(
            car,
            MotionLimits(0.2, 0.25, 1.0, 0.3, 0.5, 500),
        )
    )
    gripper_executor = RemoteGripperExecutor(
        executor.controller,
        _gripper_calibration(),
    )

    with pytest.raises(OSError, match="camera failed"):
        run_manual_capture_session(
            FakeConnection([]),
            executor,
            gripper_executor,
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
