from __future__ import annotations

import json
import struct
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
    ManualCaptureRuntime,
    VehicleState,
    _accept_with_shutdown,
    _validate_mode,
    build_session_status,
    run_manual_capture_session,
)
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.communication import (
    CaptureAction,
    CaptureStopReason,
    DebugCaptureCommand,
    DebugGripperCommand,
    DebugMotionCommand,
    ImageCoordinateSystem,
    MapStateObservation,
    MotionControlMode,
    ReceivedRemoteMessage,
    ReceivedUartFrame,
    RemoteAccessMode,
    RemoteDisconnectedError,
    RemoteRole,
    RemoteStream,
    RemoteTopic,
    TeamColor as RemoteTeamColor,
    VehicleSafetyMode,
    VehicleStateObservation,
    VideoFrameMode,
    VideoModeCommand,
)
from rescue_vision.data.check_recording import inspect_recording
from rescue_vision.motion import (
    CarStopReason,
    CarSystemStatus,
    ExecutedRemoteGripper,
    ExecutedRemoteMotion,
    GripperCalibration,
    MotionController,
    MotionLimits,
    MessageType,
    OdometryImu,
    RemoteGripperExecutor,
    RemoteGripperResult,
    RemoteMotionExecutor,
    RemoteMotionResult,
    SensorFlags,
    SystemFlags,
    encode_soft_brake_command,
    pack_protocol_frame,
)
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.world import (
    TeamColor,
)


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

    def send_frame(self, payload: bytes) -> None:
        self.sent.append(payload)

    def receive_frame(self, timeout: float | None = None):
        del timeout
        raise TimeoutError


class TelemetryAfterMotionChannel(FakeCarChannel):
    def __init__(self) -> None:
        super().__init__()
        self.telemetry_sent = False

    def receive_frame(
        self,
        timeout: float | None = None,
    ) -> ReceivedUartFrame:
        del timeout
        if not self.telemetry_sent:
            self.telemetry_sent = True
            return ReceivedUartFrame(
                sequence=0,
                received_timestamp_ns=time.monotonic_ns(),
                payload=pack_protocol_frame(
                    MessageType.SYSTEM_STATUS,
                    struct.pack(
                        "<HQHIHBHH",
                        1,
                        1_000,
                        300,
                        10,
                        int(SystemFlags.WATCHDOG_ARMED),
                        int(CarStopReason.RUNNING),
                        9000,
                        9000,
                    ),
                ),
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
    def __init__(self, messages: tuple[object, ...] = ()) -> None:
        self.drain_count = 0
        self.messages = messages

    def drain_messages(self) -> tuple[object, ...]:
        self.drain_count += 1
        messages = self.messages
        self.messages = ()
        return messages


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
        angle_sum_deg=194.0,
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


def test_accept_wait_distributes_uart_messages_to_localization() -> None:
    connection = object()
    server = PollingServer(connection, timeouts_before_connection=1)
    telemetry = object()
    controller = CountingDrainController((telemetry,))
    received: list[object] = []

    accepted = _accept_with_shutdown(
        server,  # type: ignore[arg-type]
        controller,  # type: ignore[arg-type]
        timeout_s=1.0,
        stop_requested=lambda: False,
        on_car_message=received.append,  # type: ignore[arg-type]
    )

    assert accepted is connection
    assert received == [telemetry]


def test_session_uart_callback_fans_out_odometry_once() -> None:
    telemetry = OdometryImu(
        uart_sequence=1,
        received_timestamp_ns=2_000_000,
        telemetry_sequence=1,
        sample_timestamp_us=1000,
        left_encoder_count=10,
        right_encoder_count=11,
        gyro_x_urad_s=0,
        gyro_y_urad_s=0,
        gyro_z_urad_s=0,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=9807,
        imu_temperature_cdeg=2500,
        sensor_flags=(
            SensorFlags.IMU_VALID
            | SensorFlags.IMU_CALIBRATED
            | SensorFlags.LEFT_ENCODER_VALID
            | SensorFlags.RIGHT_ENCODER_VALID
        ),
    )
    runtime = object.__new__(ManualCaptureRuntime)
    vehicle_messages: list[object] = []
    recorded_messages: list[object] = []
    localization_messages: list[object] = []
    runtime.vehicle = SimpleNamespace(on_car_message=vehicle_messages.append)
    runtime.capture = SimpleNamespace(record_car_message=recorded_messages.append)
    runtime.map_localization = SimpleNamespace(
        submit_odometry=localization_messages.append
    )

    runtime._on_car_message(telemetry)

    assert vehicle_messages == [telemetry]
    assert recorded_messages == [telemetry]
    assert localization_messages == [telemetry]


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
    assert status.vehicle_state_period_ms == 500
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
    connection = FakeConnection([])
    status = build_session_status(
        config,
        server_instance_id="test-server",
        video_fps=10.0,
        camera_only=True,
        map_state_available=True,
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
            map_state_available=True,
            map_team_color=RemoteTeamColor.UNKNOWN,
        )

    map_payload = next(
        payload
        for topic, payload in connection.observations
        if topic == RemoteTopic.MAP_STATE.value
    )
    map_state = MapStateObservation.from_payload(map_payload)
    assert map_state.team_color is RemoteTeamColor.UNKNOWN
    assert not map_state.robot_localized
    assert map_state.robot_x_mm is None
    assert map_state.targets == ()


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

    assert car.sent[-1][0] == MessageType.SOFT_BRAKE


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
    assert car.sent[-1][0] == MessageType.SOFT_BRAKE


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


def test_remote_bev_prefers_same_frame_localization_overlay() -> None:
    plain = CameraFrame(7, 11, np.zeros((30, 40, 3), np.uint8))
    annotated_image = np.zeros((30, 40, 3), np.uint8)
    annotated_image[:, :] = (0, 0, 255)
    annotated = CameraFrame(7, 11, annotated_image)
    projector = GroundProjector(
        np.eye(3),
        BevConfig(0.0, 300.0, 0.0, 400.0, 10.0),
    )
    runtime = object.__new__(ManualCaptureRuntime)
    runtime.latest_frame = plain
    runtime.video_mode = VideoFrameMode.BEV
    runtime.map_localization = SimpleNamespace(
        latest_bev_frame=lambda: annotated,
    )
    runtime.bev_renderer = SimpleNamespace(latest=lambda: plain)
    runtime.minimum_rendered_sequence = None
    runtime.last_sent_video_sequence = None
    runtime.connection = FakeConnection([])
    runtime.pipeline = CameraPipeline(
        FakeSource(plain),
        None,
        ImageCoordinateSystem.UNDISTORTED_PIXEL,
        "test-calibration",
        projector,
    )
    runtime.jpeg_quality = 100

    runtime._send_current_video()

    payload = next(
        payload
        for topic, payload in runtime.connection.observations
        if topic == RemoteTopic.VIDEO_FRAME.value
    )
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert float(np.mean(decoded[:, :, 2])) > 240.0
    assert float(np.mean(decoded[:, :, :2])) < 10.0


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
    assert car.sent == [encode_soft_brake_command(0)]


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
    with pytest.raises(ValueError, match="fresh CarSystemStatus"):
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


def test_manual_cycle_services_motion_between_slow_image_operations(
    tmp_path,
    monkeypatch,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self.timestamp_ns = 0

        def __call__(self) -> int:
            return self.timestamp_ns

        def advance(self, seconds: float) -> None:
            self.timestamp_ns += round(seconds * 1e9)

    class SlowPipeline:
        def __init__(self, source, clock) -> None:
            self.source = source
            self.clock = clock
            self.coordinate_system = ImageCoordinateSystem.RAW_PIXEL
            self.calibration_id = None
            self.ground_projector = None

        def prepare(self, frame):
            self.clock.advance(0.06)
            return frame

    clock = Clock()
    monkeypatch.setattr(manual_capture_module.time, "monotonic_ns", clock)
    frame = CameraFrame(
        sequence=0,
        timestamp_ns=0,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    config = _config()
    pipeline = SlowPipeline(FakeSource(frame), clock)
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={},
        pipeline=pipeline,
    )
    car = FakeCarChannel()
    controller = MotionController(
        car,
        MotionLimits(0.2, 0.25, 1.0, 0.3, 0.5, 500),
        monotonic_ns=clock,
    )
    controller.forward(0.2)
    runtime = ManualCaptureRuntime(
        connection=FakeConnection([]),
        executor=RemoteMotionExecutor(controller, monotonic_ns=clock),
        gripper_executor=None,
        capture=capture,
        pipeline=pipeline,
        session_status=build_session_status(
            config,
            server_instance_id="timing-test",
            video_fps=10.0,
        ),
        video_fps=10.0,
        jpeg_quality=80,
        perception_renderer=None,
        bev_renderer=None,
        map_state_available=False,
        map_team_color=None,
        map_localization=None,
        stop_requested=lambda: False,
        safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
    )
    runtime._send_current_video = lambda: clock.advance(0.06)

    runtime._cycle()

    wheel_commands = [
        struct.unpack("<Hhh", payload[1:-2])[1:]
        for payload in car.sent
        if payload[0] == MessageType.SET_WHEEL_SPEED
    ]
    assert wheel_commands == [(30, 30), (60, 60)]


def test_idle_manual_cycle_only_prepares_frames_at_video_rate(
    tmp_path,
    monkeypatch,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self.timestamp_ns = 0

        def __call__(self) -> int:
            return self.timestamp_ns

        def advance(self, seconds: float) -> None:
            self.timestamp_ns += round(seconds * 1e9)

    class CountingPipeline:
        def __init__(self, source) -> None:
            self.source = source
            self.coordinate_system = ImageCoordinateSystem.RAW_PIXEL
            self.calibration_id = None
            self.ground_projector = None
            self.prepare_count = 0

        def prepare(self, frame):
            self.prepare_count += 1
            return frame

    clock = Clock()
    monkeypatch.setattr(manual_capture_module.time, "monotonic_ns", clock)
    frame = CameraFrame(
        sequence=0,
        timestamp_ns=0,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    config = _config()
    pipeline = CountingPipeline(FakeSource(frame))
    capture = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={},
        pipeline=pipeline,
    )
    runtime = ManualCaptureRuntime(
        connection=FakeConnection([]),
        executor=None,
        gripper_executor=None,
        capture=capture,
        pipeline=pipeline,
        session_status=build_session_status(
            config,
            server_instance_id="video-rate-test",
            video_fps=2.0,
            camera_only=True,
        ),
        video_fps=2.0,
        jpeg_quality=80,
        perception_renderer=None,
        bev_renderer=None,
        map_state_available=False,
        map_team_color=None,
        map_localization=None,
        stop_requested=lambda: False,
        safety_mode=VehicleSafetyMode.UNAVAILABLE,
    )

    runtime._cycle()
    for _ in range(10):
        runtime._cycle()
    assert pipeline.prepare_count == 1
    assert pipeline.source.read_count == 1

    clock.advance(0.5)
    runtime._cycle()
    assert pipeline.prepare_count == 2
    assert pipeline.source.read_count == 2


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
        CarSystemStatus(
            uart_sequence=1,
            received_timestamp_ns=20,
            status_sequence=1,
            controller_timestamp_us=5_000,
            watchdog_timeout_ms=300,
            last_motion_command_age_ms=10,
            system_flags=SystemFlags.WATCHDOG_ARMED,
            stop_reason=CarStopReason.RUNNING,
            servo_left_target_cdeg=2700,
            servo_right_target_cdeg=16700,
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


def test_manual_capture_requires_latest_queue_slot_for_each_image_topic() -> None:
    config = _config()
    config.remote = SimpleNamespace(
        enabled=True,
        role=RemoteRole.SERVER,
        access_mode=RemoteAccessMode.DEBUG_CONTROL,
        observation_queue_capacity=2,
    )
    config.uart = SimpleNamespace(enabled=False)
    config.motion.enabled = False
    config.world = SimpleNamespace(
        static_map=SimpleNamespace(regions=(object(),)),
    )

    with pytest.raises(RuntimeError, match="at least 3"):
        _validate_mode(
            config,
            video_fps=10.0,
            supervised_physical_stop_ready=False,
            camera_only=True,
        )

    config.remote.observation_queue_capacity = 3
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
    assert car.sent == [encode_soft_brake_command(0)]
