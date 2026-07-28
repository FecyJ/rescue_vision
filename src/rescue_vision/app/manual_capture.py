"""受监督手动驾驶与车载相机采集应用。"""

from __future__ import annotations

import argparse
import json
import signal
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import yaml

from rescue_vision.camera.frame import CameraFrame, FrameSource
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.record_cli import undistort_camera_frame
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.communication import (
    CaptureAction,
    CaptureRecordingState,
    CaptureRequestResult,
    CaptureStatusObservation,
    CaptureStopReason,
    DebugCaptureCommand,
    ImageCoordinateSystem,
    ReceivedRemoteMessage,
    RemoteAccessMode,
    RemoteDisconnectedError,
    RemoteMessageConnection,
    RemoteRole,
    RemoteSessionStatus,
    RemoteStream,
    RemoteTcpServer,
    RemoteTopic,
    UartError,
    VehicleMotionState,
    VehicleSafetyMode,
    VehicleStateObservation,
    VehicleStopReason,
    VideoFrameAttributes,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.config.runtime import AppConfig
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.geometry.camera_model import (
    IMAGE_BORDER_FILL_VALUE,
    CameraModel,
)
from rescue_vision.motion import (
    CarTelemetry,
    ExecutedRemoteMotion,
    MotionController,
    MANUAL_MOTION_LOG_FILENAME,
    MANUAL_MOTION_STREAM_NAME,
    ManualMotionLogWriter,
    ParsedCarMessage,
    RemoteMotionExecutor,
    RemoteMotionResult,
    run_remote_motion,
)
from rescue_vision.versioning import git_version


SESSION_STATUS_PERIOD_MS = 1_000
VEHICLE_STATUS_PERIOD_MS = 100
CAPTURE_STATUS_PERIOD_MS = 500


@dataclass(frozen=True, slots=True)
class CameraPipeline:
    """配置驱动的帧源和图像坐标身份。"""

    source: FrameSource
    camera_model: CameraModel | None
    coordinate_system: ImageCoordinateSystem
    intrinsics_fingerprint_sha256: str | None

    def prepare(self, frame: CameraFrame) -> CameraFrame:
        if self.camera_model is None:
            return frame
        return undistort_camera_frame(frame, camera_model=self.camera_model)


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    action: CaptureAction
    result: CaptureRequestResult
    artifact_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class CaptureSession:
    """单 TCP 连接内的录制、抓拍和事件标记状态机。"""

    def __init__(
        self,
        *,
        output_root: Path,
        config: AppConfig,
        config_snapshot: dict[str, object],
        pipeline: CameraPipeline,
    ) -> None:
        self.output_root = output_root
        self.config = config
        self.config_snapshot = config_snapshot
        self.pipeline = pipeline
        self.recorder: FrameRecorder | None = None
        self.motion_log: ManualMotionLogWriter | None = None
        self.recording_id: str | None = None
        self.recording_directory: Path | None = None
        self.status_sequence = 0
        self.last_outcome: tuple[str, CaptureOutcome] | None = None
        self.outcomes: dict[str, CaptureOutcome] = {}
        self.stop_reason: CaptureStopReason | None = None
        self.faulted = False
        self.accepted_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0

    def status(self) -> CaptureStatusObservation:
        outcome = None if self.last_outcome is None else self.last_outcome[1]
        recorder = self.recorder
        status = CaptureStatusObservation(
            status_sequence=self.status_sequence,
            timestamp_ns=time.monotonic_ns(),
            recording_state=(
                CaptureRecordingState.FAULT
                if self.faulted
                else (
                    CaptureRecordingState.RECORDING
                    if recorder is not None
                    else CaptureRecordingState.IDLE
                )
            ),
            recording_id=self.recording_id if recorder is not None else None,
            accepted_frames=(
                self.accepted_frames
                if recorder is None
                else recorder.accepted_frames
            ),
            written_frames=(
                self.written_frames
                if recorder is None
                else recorder.written_frames
            ),
            dropped_frames=(
                self.dropped_frames
                if recorder is None
                else recorder.dropped_frames
            ),
            available_disk_bytes=shutil.disk_usage(self.output_root).free,
            last_request_id=(
                None if self.last_outcome is None else self.last_outcome[0]
            ),
            last_request_action=None if outcome is None else outcome.action,
            last_request_result=(
                CaptureRequestResult.NONE
                if outcome is None
                else outcome.result
            ),
            last_request_artifact_id=(
                None if outcome is None else outcome.artifact_id
            ),
            stop_reason=self.stop_reason,
            error_code=None if outcome is None else outcome.error_code,
            error_message=None if outcome is None else outcome.error_message,
        )
        self.status_sequence += 1
        return status

    def execute(
        self,
        command: DebugCaptureCommand,
        latest_frame: CameraFrame | None,
    ) -> CaptureOutcome:
        previous = self.outcomes.get(command.request_id)
        if previous is not None:
            self.last_outcome = (command.request_id, previous)
            return previous
        try:
            if command.action is CaptureAction.START:
                outcome = self._start(command)
            elif command.action is CaptureAction.STOP:
                outcome = self._stop(command)
            elif latest_frame is None:
                outcome = self._rejected(command, "no camera frame is available")
            elif command.action is CaptureAction.SNAPSHOT:
                outcome = self._snapshot(command, latest_frame)
            else:
                outcome = self._mark_event(command, latest_frame)
        except Exception:
            self.fail(CaptureStopReason.WRITE_ERROR)
            outcome = CaptureOutcome(
                action=command.action,
                result=CaptureRequestResult.FAILED,
                error_code="write_error",
                error_message="capture action failed; inspect vehicle log",
            )
        self.outcomes[command.request_id] = outcome
        self.last_outcome = (command.request_id, outcome)
        return outcome

    def accepted(self, command: DebugCaptureCommand) -> None:
        self.last_outcome = (
            command.request_id,
            CaptureOutcome(
                action=command.action,
                result=CaptureRequestResult.ACCEPTED,
            ),
        )

    def record(self, frame: CameraFrame) -> None:
        if self.recorder is not None and not self.recorder.record(frame):
            self.fail(CaptureStopReason.WRITE_ERROR)
            raise RuntimeError(
                "Recording queue overflowed; manual driving was stopped."
            )

    def fail(self, reason: CaptureStopReason) -> None:
        self.faulted = True
        self.stop_reason = reason

    def close(self) -> None:
        recorder = self.recorder
        if recorder is None:
            return
        self.recorder = None
        self.recording_id = None
        self.recording_directory = None
        if not self.faulted:
            self.stop_reason = CaptureStopReason.APPLICATION_SHUTDOWN
        self._close_recording_resources(recorder)

    def _start(self, command: DebugCaptureCommand) -> CaptureOutcome:
        if self.recorder is not None:
            return self._rejected(command, "recording is already active")
        if self.faulted:
            return self._rejected(command, "capture session is faulted")
        recording_id = _artifact_id("recording")
        directory = self.output_root / "recordings" / recording_id
        tags = {name: "unknown" for name in REQUIRED_TAGS}
        tags.update(command.session_tags)
        camera_model = self.pipeline.camera_model
        recorder = FrameRecorder(
            directory,
            image_size=self.config.camera.image_size,
            config_snapshot=self.config_snapshot,
            versions={
                "code": git_version(),
                "config_schema": str(self.config.schema_version),
                "opencv": cv2.__version__,
            },
            session_tags=tags,
            queue_capacity=self.config.recording.queue_capacity,
            image_format=self.config.recording.image_format,
            image_coordinate_system=self.pipeline.coordinate_system.value,
            intrinsics_fingerprint_sha256=(
                self.pipeline.intrinsics_fingerprint_sha256
            ),
            valid_pixel_ratio=(
                cv2.countNonZero(camera_model.valid_mask)
                / camera_model.valid_mask.size
                if camera_model is not None
                else None
            ),
            undistort_fill_value=(
                IMAGE_BORDER_FILL_VALUE if camera_model is not None else None
            ),
            auxiliary_streams={
                MANUAL_MOTION_STREAM_NAME: MANUAL_MOTION_LOG_FILENAME
            },
            recording_kind="supervised_manual_motion",
        )
        recorder.start()
        motion_log = ManualMotionLogWriter(
            directory / MANUAL_MOTION_LOG_FILENAME
        )
        try:
            motion_log.start(timestamp_ns=time.monotonic_ns())
        except BaseException as exc:
            try:
                recorder.stop()
            except BaseException as cleanup_error:
                exc.add_note(
                    "Frame recorder cleanup after motion log start failure "
                    f"also failed: {cleanup_error!r}"
                )
            raise
        self.recorder = recorder
        self.motion_log = motion_log
        self.recording_id = recording_id
        self.recording_directory = directory
        self.stop_reason = None
        self.accepted_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0
        return CaptureOutcome(
            command.action,
            CaptureRequestResult.COMPLETED,
            recording_id,
        )

    def _stop(self, command: DebugCaptureCommand) -> CaptureOutcome:
        recorder = self.recorder
        if recorder is None:
            return self._rejected(command, "no recording is active")
        artifact_id = self.recording_id
        self._close_recording_resources(recorder)
        self.stop_reason = CaptureStopReason.REQUESTED
        return CaptureOutcome(
            command.action,
            CaptureRequestResult.COMPLETED,
            artifact_id,
        )

    def _snapshot(
        self,
        command: DebugCaptureCommand,
        frame: CameraFrame,
    ) -> CaptureOutcome:
        artifact_id = _artifact_id("snapshot")
        directory = self.output_root / "snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{artifact_id}.jpg"
        ok, encoded = cv2.imencode(
            ".jpg",
            frame.image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode snapshot.")
        image_path.write_bytes(encoded.tobytes())
        metadata = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "request_id": command.request_id,
            "label": command.label,
            "frame_sequence": frame.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "image_path": image_path.name,
            "coordinate_system": self.pipeline.coordinate_system.value,
            "intrinsics_fingerprint_sha256": (
                self.pipeline.intrinsics_fingerprint_sha256
            ),
        }
        (directory / f"{artifact_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return CaptureOutcome(
            command.action,
            CaptureRequestResult.COMPLETED,
            artifact_id,
        )

    def _mark_event(
        self,
        command: DebugCaptureCommand,
        frame: CameraFrame,
    ) -> CaptureOutcome:
        if self.recording_directory is None:
            return self._rejected(command, "event marking requires recording")
        artifact_id = _artifact_id("event")
        with (self.recording_directory / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as events:
            events.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_id": artifact_id,
                        "request_id": command.request_id,
                        "label": command.label,
                        "frame_sequence": frame.sequence,
                        "timestamp_ns": frame.timestamp_ns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return CaptureOutcome(
            command.action,
            CaptureRequestResult.COMPLETED,
            artifact_id,
        )

    @staticmethod
    def _rejected(
        command: DebugCaptureCommand,
        message: str,
    ) -> CaptureOutcome:
        return CaptureOutcome(
            command.action,
            CaptureRequestResult.REJECTED,
            error_code="state_conflict",
            error_message=message,
        )

    def _remember_counts(self, recorder: FrameRecorder) -> None:
        self.accepted_frames = recorder.accepted_frames
        self.written_frames = recorder.written_frames
        self.dropped_frames = recorder.dropped_frames

    def record_motion(self, outcome: ExecutedRemoteMotion) -> None:
        if self.motion_log is not None:
            self.motion_log.record_motion(outcome)

    def record_motion_timeout(
        self,
        *,
        command_id: str,
        timestamp_ns: int,
    ) -> None:
        if self.motion_log is not None:
            self.motion_log.record_timeout(
                command_id=command_id,
                timestamp_ns=timestamp_ns,
            )

    def record_car_message(self, message: ParsedCarMessage) -> None:
        if self.motion_log is not None:
            self.motion_log.record_car_message(message)

    def record_safety_stop(
        self,
        *,
        reason: VehicleStopReason,
        timestamp_ns: int,
    ) -> None:
        if self.motion_log is not None:
            self.motion_log.record_safety_stop(
                reason=reason.value,
                timestamp_ns=timestamp_ns,
            )

    def _close_recording_resources(self, recorder: FrameRecorder) -> None:
        motion_log = self.motion_log
        self.motion_log = None
        self.recorder = None
        self.recording_id = None
        self.recording_directory = None
        primary_error: BaseException | None = None
        if motion_log is not None:
            try:
                motion_log.stop(timestamp_ns=time.monotonic_ns())
            except BaseException as exc:
                primary_error = exc
        try:
            recorder.stop()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                primary_error.add_note(
                    f"Frame recorder cleanup also failed: {exc!r}"
                )
        finally:
            self._remember_counts(recorder)
        if primary_error is not None:
            raise primary_error


class VehicleState:
    """把运动执行结果和最新轮速遥测汇总为协议观察。"""

    def __init__(self, *, safety_mode: VehicleSafetyMode) -> None:
        if safety_mode is not VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP:
            raise ValueError(
                "Manual capture currently requires "
                "supervised_physical_stop; firmware_watchdog needs fresh "
                "CarSafetyStatus gating before it can be selected."
            )
        self.safety_mode = safety_mode
        self.sequence = 0
        self.telemetry: CarTelemetry | None = None
        self.motion_state = VehicleMotionState.STOPPED
        self.stop_reason = VehicleStopReason.DEADMAN_RELEASE
        self.last_received_command_id: str | None = None
        self.last_applied_command_id: str | None = None

    def on_car_message(self, message: ParsedCarMessage) -> None:
        if isinstance(message, CarTelemetry):
            self.telemetry = message

    def on_motion(self, outcome: ExecutedRemoteMotion) -> None:
        self.last_received_command_id = outcome.command_id
        if outcome.result is RemoteMotionResult.APPLIED:
            self.last_applied_command_id = outcome.command_id
            if (
                outcome.linear_velocity_m_s == 0.0
                and outcome.angular_velocity_rad_s == 0.0
            ):
                self.motion_state = VehicleMotionState.BRAKING
            else:
                self.motion_state = VehicleMotionState.MOVING
            self.stop_reason = VehicleStopReason.NONE
        elif outcome.result is RemoteMotionResult.STOPPED_DEADMAN:
            self.motion_state = VehicleMotionState.BRAKING
            self.stop_reason = VehicleStopReason.DEADMAN_RELEASE
        else:
            self.motion_state = VehicleMotionState.BRAKING
            self.stop_reason = VehicleStopReason.COMMAND_EXPIRED

    def on_motion_timeout(self) -> None:
        self.motion_state = VehicleMotionState.BRAKING
        self.stop_reason = VehicleStopReason.COMMAND_EXPIRED

    def observation(self) -> VehicleStateObservation:
        telemetry = self.telemetry
        observation = VehicleStateObservation(
            state_sequence=self.sequence,
            timestamp_ns=time.monotonic_ns(),
            control_ready=True,
            safety_mode=self.safety_mode,
            uart_connected=True,
            watchdog_armed=(
                self.safety_mode is VehicleSafetyMode.FIRMWARE_WATCHDOG
            ),
            emergency_stop_latched=False,
            motion_state=self.motion_state,
            stop_reason=self.stop_reason,
            controller_uptime_ms=(
                None if telemetry is None else telemetry.controller_timestamp_ms
            ),
            target_left_velocity_m_s=(
                None if telemetry is None else telemetry.target_left_m_s
            ),
            target_right_velocity_m_s=(
                None if telemetry is None else telemetry.target_right_m_s
            ),
            measured_left_velocity_m_s=(
                None if telemetry is None else telemetry.actual_left_m_s
            ),
            measured_right_velocity_m_s=(
                None if telemetry is None else telemetry.actual_right_m_s
            ),
            heading_rad=None,
            heading_reference=None,
            last_received_motion_command_id=self.last_received_command_id,
            last_applied_motion_command_id=self.last_applied_command_id,
        )
        self.sequence += 1
        return observation


class ManualCaptureRuntime:
    """把运动安全循环、相机、采集和观察流装配为一个会话。"""

    def __init__(
        self,
        *,
        connection: RemoteMessageConnection,
        executor: RemoteMotionExecutor,
        capture: CaptureSession,
        pipeline: CameraPipeline,
        session_status: RemoteSessionStatus,
        video_fps: float,
        jpeg_quality: int,
        stop_requested: Callable[[], bool],
        safety_mode: VehicleSafetyMode,
    ) -> None:
        self.connection = connection
        self.executor = executor
        self.capture = capture
        self.pipeline = pipeline
        self.session_status = session_status
        self.video_period_ns = int(1_000_000_000 / video_fps)
        self.jpeg_quality = jpeg_quality
        self.stop_requested = stop_requested
        self.vehicle = VehicleState(safety_mode=safety_mode)
        self.latest_frame: CameraFrame | None = None
        self.last_camera_frame_ns = time.monotonic_ns()
        self.next_video_ns = 0
        now_ns = time.monotonic_ns()
        self.next_session_status_ns = now_ns + SESSION_STATUS_PERIOD_MS * 1_000_000
        self.next_vehicle_status_ns = now_ns + VEHICLE_STATUS_PERIOD_MS * 1_000_000
        self.next_capture_status_ns = now_ns + CAPTURE_STATUS_PERIOD_MS * 1_000_000

    def run(self) -> None:
        self._send_initial_status()
        run_remote_motion(
            self.connection,
            self.executor,
            stop_requested=self.stop_requested,
            on_car_message=self._on_car_message,
            on_motion_executed=self._on_motion,
            on_motion_timeout=self._on_motion_timeout,
            on_other_control=self._handle_other_control,
            on_cycle=self._cycle,
            poll_interval_s=0.02,
        )

    def _on_car_message(self, message: ParsedCarMessage) -> None:
        self.vehicle.on_car_message(message)
        self.capture.record_car_message(message)

    def _on_motion(self, outcome: ExecutedRemoteMotion) -> None:
        self.vehicle.on_motion(outcome)
        self.capture.record_motion(outcome)

    def _on_motion_timeout(self) -> None:
        command_id = self.vehicle.last_received_command_id
        self.vehicle.on_motion_timeout()
        if command_id is not None:
            self.capture.record_motion_timeout(
                command_id=command_id,
                timestamp_ns=time.monotonic_ns(),
            )

    def _cycle(self) -> None:
        try:
            frame = self.pipeline.source.read(timeout=0.01)
        except TimeoutError:
            frame = None
            if time.monotonic_ns() - self.last_camera_frame_ns >= 1_000_000_000:
                self.capture.fail(CaptureStopReason.CAMERA_ERROR)
                raise RuntimeError(
                    "Camera produced no frame for 1.0 seconds; "
                    "manual driving was stopped."
                )
        except BaseException:
            self.capture.fail(CaptureStopReason.CAMERA_ERROR)
            raise
        if frame is not None:
            try:
                self.latest_frame = self.pipeline.prepare(frame)
            except BaseException:
                self.capture.fail(CaptureStopReason.CAMERA_ERROR)
                raise
            self.last_camera_frame_ns = time.monotonic_ns()
            self.capture.record(self.latest_frame)
        now_ns = time.monotonic_ns()
        if self.latest_frame is not None and now_ns >= self.next_video_ns:
            _send_video_frame(
                self.connection,
                self.latest_frame,
                self.pipeline,
                jpeg_quality=self.jpeg_quality,
            )
            self.next_video_ns = now_ns + self.video_period_ns
        if now_ns >= self.next_session_status_ns:
            self.session_status = replace(
                self.session_status,
                timestamp_ns=now_ns,
            )
            self.connection.send_reliable_observation(
                RemoteTopic.SESSION_STATUS.value,
                self.session_status.to_payload(),
                content_type="application/json",
            )
            self.next_session_status_ns = (
                now_ns + SESSION_STATUS_PERIOD_MS * 1_000_000
            )
        if now_ns >= self.next_vehicle_status_ns:
            self.connection.send_observation(
                RemoteTopic.VEHICLE_STATE.value,
                self.vehicle.observation().to_payload(),
                content_type="application/json",
            )
            self.next_vehicle_status_ns = (
                now_ns + VEHICLE_STATUS_PERIOD_MS * 1_000_000
            )
        if now_ns >= self.next_capture_status_ns:
            self._send_capture_status()
            self.next_capture_status_ns = (
                now_ns + CAPTURE_STATUS_PERIOD_MS * 1_000_000
            )

    def _send_initial_status(self) -> None:
        self.connection.send_reliable_observation(
            RemoteTopic.SESSION_STATUS.value,
            self.session_status.to_payload(),
            content_type="application/json",
        )
        self.connection.send_observation(
            RemoteTopic.VEHICLE_STATE.value,
            self.vehicle.observation().to_payload(),
            content_type="application/json",
        )
        self._send_capture_status()

    def _handle_other_control(self, message: ReceivedRemoteMessage) -> None:
        if (
            message.stream is not RemoteStream.CONTROL
            or message.topic != RemoteTopic.DEBUG_CAPTURE.value
            or message.content_type != "application/json"
            or message.attributes
        ):
            raise ValueError(
                f"Unsupported remote control message {message.topic!r}."
            )
        command = DebugCaptureCommand.from_payload(message.payload)
        if command.request_id not in self.capture.outcomes:
            self.capture.accepted(command)
            self._send_capture_status()
        outcome = self.capture.execute(command, self.latest_frame)
        self._send_capture_status()
        if outcome.result is CaptureRequestResult.FAILED:
            raise RuntimeError(
                f"Capture request {command.request_id!r} failed."
            )

    def _send_capture_status(self) -> None:
        self.connection.send_reliable_observation(
            RemoteTopic.CAPTURE_STATUS.value,
            self.capture.status().to_payload(),
            content_type="application/json",
        )


def build_camera_pipeline(config: AppConfig) -> CameraPipeline:
    camera_model = config.build_camera_model()
    source_class = (
        Picamera2Source
        if config.camera.backend == "picamera2"
        else RpicamSource
    )
    source = source_class(
        image_size=config.camera.image_size,
        fps=config.camera.fps,
        lens_position=config.camera.lens_position,
    )
    return CameraPipeline(
        source,
        camera_model,
        (
            ImageCoordinateSystem.UNDISTORTED_PIXEL
            if camera_model is not None
            else ImageCoordinateSystem.RAW_PIXEL
        ),
        (
            None
            if camera_model is None
            else camera_model.calibration.fingerprint()
        ),
    )


def build_session_status(
    config: AppConfig,
    *,
    server_instance_id: str,
    video_fps: float,
) -> RemoteSessionStatus:
    return RemoteSessionStatus(
        session_id=f"session-{uuid.uuid4()}",
        server_instance_id=server_instance_id,
        timestamp_ns=time.monotonic_ns(),
        access_mode=config.remote.access_mode,
        motion_control_available=True,
        capture_control_available=True,
        video_stream_available=True,
        map_snapshot_available=False,
        vehicle_state_available=True,
        capture_status_available=True,
        target_heading_control_available=False,
        session_status_period_ms=SESSION_STATUS_PERIOD_MS,
        vehicle_state_period_ms=VEHICLE_STATUS_PERIOD_MS,
        map_snapshot_period_ms=None,
        capture_status_period_ms=CAPTURE_STATUS_PERIOD_MS,
        video_nominal_fps=video_fps,
        max_linear_velocity_m_s=config.motion.max_linear_velocity_m_s,
        max_angular_velocity_rad_s=config.motion.max_angular_velocity_rad_s,
        max_motion_command_valid_for_ms=(
            config.motion.max_remote_command_valid_for_ms
        ),
    )


def run_manual_capture_session(
    connection: RemoteMessageConnection,
    executor: RemoteMotionExecutor,
    capture: CaptureSession,
    pipeline: CameraPipeline,
    session_status: RemoteSessionStatus,
    *,
    video_fps: float,
    jpeg_quality: int,
    stop_requested: Callable[[], bool] = lambda: False,
    safety_mode: VehicleSafetyMode = VehicleSafetyMode.UNAVAILABLE,
) -> None:
    runtime = ManualCaptureRuntime(
        connection=connection,
        executor=executor,
        capture=capture,
        pipeline=pipeline,
        session_status=session_status,
        video_fps=video_fps,
        jpeg_quality=jpeg_quality,
        stop_requested=stop_requested,
        safety_mode=safety_mode,
    )
    try:
        runtime.run()
    except BaseException as exc:
        if isinstance(exc, RemoteDisconnectedError):
            stop_reason = VehicleStopReason.REMOTE_DISCONNECTED
        elif isinstance(exc, UartError):
            stop_reason = VehicleStopReason.UART_FAULT
        elif capture.stop_reason is CaptureStopReason.CAMERA_ERROR:
            stop_reason = VehicleStopReason.CAMERA_FAULT
        else:
            stop_reason = VehicleStopReason.UNKNOWN
        try:
            capture.record_safety_stop(
                reason=stop_reason,
                timestamp_ns=time.monotonic_ns(),
            )
        except BaseException as log_error:
            exc.add_note(f"Safety stop logging also failed: {log_error!r}")
        try:
            capture.close()
        except BaseException as cleanup_error:
            exc.add_note(
                f"Capture session cleanup also failed: {cleanup_error!r}"
            )
        raise
    else:
        try:
            capture.record_safety_stop(
                reason=VehicleStopReason.APPLICATION_SHUTDOWN,
                timestamp_ns=time.monotonic_ns(),
            )
        finally:
            capture.close()


def _send_video_frame(
    connection: RemoteMessageConnection,
    frame: CameraFrame,
    pipeline: CameraPipeline,
    *,
    jpeg_quality: int,
) -> None:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame.image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode the remote JPEG frame.")
    height, width = frame.image_bgr.shape[:2]
    attributes = VideoFrameAttributes(
        frame_sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        width=width,
        height=height,
        coordinate_system=pipeline.coordinate_system,
        intrinsics_fingerprint_sha256=(
            pipeline.intrinsics_fingerprint_sha256
        ),
    )
    connection.send_observation(
        RemoteTopic.VIDEO_FRAME.value,
        encoded.tobytes(),
        content_type="image/jpeg",
        attributes=attributes.to_attributes(),
        sender_timestamp_ns=frame.timestamp_ns,
    )


def _artifact_id(prefix: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _accept_with_shutdown(
    server: RemoteTcpServer,
    *,
    timeout_s: float,
    stop_requested: Callable[[], bool],
) -> RemoteMessageConnection | None:
    deadline = time.monotonic() + timeout_s
    while not stop_requested():
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError("Timed out waiting for remote TCP client.")
        try:
            return server.accept(timeout=min(0.5, remaining_s))
        except TimeoutError:
            continue
    return None


def _validate_mode(
    config: AppConfig,
    *,
    video_fps: float,
    supervised_physical_stop_ready: bool,
) -> None:
    if not config.remote.enabled or config.remote.role is not RemoteRole.SERVER:
        raise RuntimeError("Manual capture requires remote server mode.")
    if config.remote.access_mode is not RemoteAccessMode.DEBUG_CONTROL:
        raise RuntimeError("Manual capture requires remote debug_control mode.")
    if not config.uart.enabled or not config.motion.enabled:
        raise RuntimeError("Manual capture requires enabled UART and motion.")
    if not supervised_physical_stop_ready:
        raise RuntimeError(
            "Current firmware watchdog state is unavailable; pass "
            "--supervised-physical-stop-ready only after a physical emergency "
            "stop is ready and an operator will supervise the full session."
        )
    if video_fps > config.camera.fps:
        raise RuntimeError("--video-fps must not exceed camera.fps.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run supervised manual driving and onboard capture."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--accept-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help=(
            "Acknowledge that a physical emergency stop and continuous human "
            "supervision are ready while the firmware watchdog is unavailable."
        ),
    )
    args = parser.parse_args()
    if args.accept_timeout_seconds <= 0:
        parser.error("--accept-timeout-seconds must be positive")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config = load_runtime_config(config_path)
    _validate_mode(
        config,
        video_fps=args.video_fps,
        supervised_physical_stop_ready=args.supervised_physical_stop_ready,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config_snapshot = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    server = config.remote.build_server()
    channel = config.uart.build_channel()
    controller: MotionController | None = config.motion.build_controller(channel)
    executor = config.motion.build_remote_executor(controller)
    assert server is not None
    assert channel is not None
    assert executor is not None
    pipeline = build_camera_pipeline(config)
    server_instance_id = f"manual-capture-{uuid.uuid4()}"
    shutdown_requested = threading.Event()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: shutdown_requested.set(),
    )

    try:
        with server, channel:
            pipeline.source.start()
            try:
                while not shutdown_requested.is_set():
                    connection = _accept_with_shutdown(
                        server,
                        timeout_s=args.accept_timeout_seconds,
                        stop_requested=shutdown_requested.is_set,
                    )
                    if connection is None:
                        break
                    capture = CaptureSession(
                        output_root=output_root,
                        config=config,
                        config_snapshot=config_snapshot,
                        pipeline=pipeline,
                    )
                    status = build_session_status(
                        config,
                        server_instance_id=server_instance_id,
                        video_fps=args.video_fps,
                    )
                    try:
                        with connection:
                            run_manual_capture_session(
                                connection,
                                executor,
                                capture,
                                pipeline,
                                status,
                                video_fps=args.video_fps,
                                jpeg_quality=args.jpeg_quality,
                                stop_requested=shutdown_requested.is_set,
                                safety_mode=(
                                    VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP
                                ),
                            )
                    except RemoteDisconnectedError:
                        # 新连接会创建全新的状态和命令期限，不继承死手使能。
                        continue
                    except KeyboardInterrupt:
                        break
                    except BaseException:
                        if not capture.faulted:
                            capture.fail(CaptureStopReason.UNKNOWN)
                        raise
            finally:
                # run_remote_motion 已先停车；这里重复停车覆盖连接前/装配期异常。
                try:
                    executor.stop()
                finally:
                    pipeline.source.stop()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
