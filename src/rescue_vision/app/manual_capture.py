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
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import yaml

from rescue_vision.camera.frame import CameraFrame, FrameSource
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.record_cli import undistort_camera_frame
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.exception_notes import add_exception_note
from rescue_vision.communication import (
    CaptureAction,
    CaptureRecordingState,
    CaptureRequestResult,
    CaptureStatusObservation,
    CaptureStopReason,
    DebugCaptureCommand,
    DebugGripperCommand,
    DebugMotionCommand,
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
    VideoFrameMode,
    VideoModeCommand,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.config.runtime import AppConfig
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.geometry.camera_model import (
    IMAGE_BORDER_FILL_VALUE,
    CameraModel,
)
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.motion import (
    CarTelemetry,
    ExecutedRemoteGripper,
    ExecutedRemoteMotion,
    MotionController,
    MANUAL_MOTION_LOG_FILENAME,
    MANUAL_MOTION_STREAM_NAME,
    ManualMotionLogWriter,
    ParsedCarMessage,
    RemoteGripperExecutor,
    RemoteGripperResult,
    RemoteMotionExecutor,
    RemoteMotionResult,
    run_remote_motion,
)
from rescue_vision.perception import PerceptionFrameRenderer
from rescue_vision.app.field_map import (
    FieldMapSnapshotRenderer,
    LatestCenterCrossLocalization,
)
SESSION_STATUS_PERIOD_MS = 1_000
VEHICLE_STATUS_PERIOD_MS = 100
CAMERA_ONLY_VEHICLE_STATUS_PERIOD_MS = 500
CAPTURE_STATUS_PERIOD_MS = 500
MAP_SNAPSHOT_PERIOD_MS = 500
CAMERA_ONLY_CONTROL_BATCH_LIMIT = 32


@dataclass(frozen=True, slots=True)
class CameraPipeline:
    """配置驱动的帧源和图像坐标身份。"""

    source: FrameSource
    camera_model: CameraModel | None
    coordinate_system: ImageCoordinateSystem
    calibration_id: str | None
    ground_projector: GroundProjector | None = None

    def prepare(self, frame: CameraFrame) -> CameraFrame:
        if self.camera_model is None:
            return frame
        return undistort_camera_frame(frame, camera_model=self.camera_model)


class BevFrameRenderer:
    """在有界最新帧后台旁路中生成 BEV，不阻塞运动安全循环。"""

    def __init__(self, ground_projector: GroundProjector) -> None:
        if ground_projector.bev_config is None:
            raise ValueError("BEV renderer requires a projector with BEV config.")
        self._ground_projector = ground_projector
        self._condition = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pending_frame: CameraFrame | None = None
        self._latest_frame: CameraFrame | None = None
        self._worker_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("BevFrameRenderer is already started.")
        self._stop_event.clear()
        self._condition.clear()
        with self._lock:
            self._pending_frame = None
            self._latest_frame = None
            self._worker_error = None
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="rescue-bev-video",
            daemon=True,
        )
        self._started = True
        try:
            self._thread.start()
        except BaseException:
            self._started = False
            self._thread = None
            raise

    def submit(self, frame: CameraFrame) -> None:
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            self._pending_frame = frame
        self._condition.set()

    def latest(self) -> CameraFrame | None:
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            return self._latest_frame

    def clear_latest(self) -> None:
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            self._pending_frame = None
            self._latest_frame = None

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._condition.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("BevFrameRenderer worker did not stop.")
        self._thread = None
        self._started = False
        self._condition.clear()
        self._raise_worker_error()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("BevFrameRenderer is not started.")

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("BEV video renderer failed.") from self._worker_error

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._condition.wait()
                if self._stop_event.is_set():
                    break
                with self._lock:
                    frame = self._pending_frame
                    self._pending_frame = None
                    self._condition.clear()
                if frame is None:
                    continue
                image_bgr = self._ground_projector.make_bev_image(frame.image_bgr)
                rendered = CameraFrame(
                    sequence=frame.sequence,
                    timestamp_ns=frame.timestamp_ns,
                    image_bgr=image_bgr,
                    metadata=frame.metadata,
                )
                with self._lock:
                    self._latest_frame = rendered
        except BaseException as exc:
            self._worker_error = exc
            self._stop_event.set()
            self._condition.set()


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
        motion_logging_enabled: bool = True,
    ) -> None:
        self.output_root = output_root
        self.config = config
        self.config_snapshot = config_snapshot
        self.pipeline = pipeline
        if not isinstance(motion_logging_enabled, bool):
            raise TypeError("motion_logging_enabled must be a boolean.")
        self.motion_logging_enabled = motion_logging_enabled
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
            session_tags=tags,
            queue_capacity=self.config.recording.queue_capacity,
            image_format=self.config.recording.image_format,
            image_coordinate_system=self.pipeline.coordinate_system.value,
            calibration_id=self.pipeline.calibration_id,
            valid_pixel_ratio=(
                cv2.countNonZero(camera_model.valid_mask)
                / camera_model.valid_mask.size
                if camera_model is not None
                else None
            ),
            undistort_fill_value=(
                IMAGE_BORDER_FILL_VALUE if camera_model is not None else None
            ),
            auxiliary_streams=(
                {MANUAL_MOTION_STREAM_NAME: MANUAL_MOTION_LOG_FILENAME}
                if self.motion_logging_enabled
                else {}
            ),
            recording_kind=(
                "supervised_manual_motion"
                if self.motion_logging_enabled
                else "camera"
            ),
        )
        recorder.start()
        motion_log: ManualMotionLogWriter | None = None
        if self.motion_logging_enabled:
            motion_log = ManualMotionLogWriter(
                directory / MANUAL_MOTION_LOG_FILENAME
            )
            try:
                motion_log.start(timestamp_ns=time.monotonic_ns())
            except BaseException as exc:
                try:
                    recorder.stop()
                except BaseException as cleanup_error:
                    add_exception_note(
                        exc,
                        "Frame recorder cleanup after motion log start failure "
                        f"also failed: {cleanup_error!r}",
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
            "artifact_id": artifact_id,
            "request_id": command.request_id,
            "label": command.label,
            "frame_sequence": frame.sequence,
            "timestamp_ns": frame.timestamp_ns,
            "image_path": image_path.name,
            "coordinate_system": self.pipeline.coordinate_system.value,
            "calibration_id": self.pipeline.calibration_id,
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

    def record_gripper(self, outcome: ExecutedRemoteGripper) -> None:
        if self.motion_log is not None:
            self.motion_log.record_gripper(outcome)

    def record_gripper_timeout(
        self,
        *,
        command_id: str,
        timestamp_ns: int,
    ) -> None:
        if self.motion_log is not None:
            self.motion_log.record_gripper_timeout(
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
                add_exception_note(
                    primary_error,
                    f"Frame recorder cleanup also failed: {exc!r}",
                )
        finally:
            self._remember_counts(recorder)
        if primary_error is not None:
            raise primary_error


class VehicleState:
    """把运动/夹爪执行结果和最新 UART 遥测汇总为协议观察。"""

    def __init__(self, *, safety_mode: VehicleSafetyMode) -> None:
        if safety_mode is VehicleSafetyMode.FIRMWARE_WATCHDOG:
            raise ValueError(
                "Manual capture firmware_watchdog needs fresh "
                "CarSafetyStatus gating before it can be selected."
            )
        if safety_mode not in {
            VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
            VehicleSafetyMode.UNAVAILABLE,
        }:
            raise ValueError(f"Unsupported safety_mode {safety_mode!r}.")
        self.safety_mode = safety_mode
        self.sequence = 0
        self.telemetry: CarTelemetry | None = None
        self.motion_state = VehicleMotionState.STOPPED
        self.stop_reason = (
            VehicleStopReason.UART_FAULT
            if safety_mode is VehicleSafetyMode.UNAVAILABLE
            else VehicleStopReason.DEADMAN_RELEASE
        )
        self.last_received_command_id: str | None = None
        self.last_applied_command_id: str | None = None
        self.last_received_gripper_command_id: str | None = None
        self.last_applied_gripper_command_id: str | None = None

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

    def on_gripper(self, outcome: ExecutedRemoteGripper) -> None:
        self.last_received_gripper_command_id = outcome.command_id
        if outcome.result in {
            RemoteGripperResult.APPLIED,
            RemoteGripperResult.STOPPED,
        }:
            self.last_applied_gripper_command_id = outcome.command_id

    def observation(self) -> VehicleStateObservation:
        telemetry = self.telemetry
        uart_connected = self.safety_mode is not VehicleSafetyMode.UNAVAILABLE
        observation = VehicleStateObservation(
            state_sequence=self.sequence,
            timestamp_ns=time.monotonic_ns(),
            control_ready=uart_connected,
            safety_mode=self.safety_mode,
            uart_connected=uart_connected,
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
            gripper_left_angle_deg=(
                None if telemetry is None else telemetry.servo_left_deg
            ),
            gripper_right_angle_deg=(
                None if telemetry is None else telemetry.servo_right_deg
            ),
            heading_rad=None,
            heading_reference=None,
            last_received_motion_command_id=self.last_received_command_id,
            last_applied_motion_command_id=self.last_applied_command_id,
            last_received_gripper_command_id=(
                self.last_received_gripper_command_id
            ),
            last_applied_gripper_command_id=(
                self.last_applied_gripper_command_id
            ),
        )
        self.sequence += 1
        return observation


class ManualCaptureRuntime:
    """把运动安全循环、相机、采集和观察流装配为一个会话。"""

    def __init__(
        self,
        *,
        connection: RemoteMessageConnection,
        executor: RemoteMotionExecutor | None,
        gripper_executor: RemoteGripperExecutor | None,
        capture: CaptureSession,
        pipeline: CameraPipeline,
        session_status: RemoteSessionStatus,
        video_fps: float,
        jpeg_quality: int,
        perception_renderer: PerceptionFrameRenderer | None,
        bev_renderer: BevFrameRenderer | None,
        map_renderer: FieldMapSnapshotRenderer | None,
        map_localization: LatestCenterCrossLocalization | None,
        stop_requested: Callable[[], bool],
        safety_mode: VehicleSafetyMode,
    ) -> None:
        self.connection = connection
        self.executor = executor
        self.gripper_executor = gripper_executor
        self.capture = capture
        self.pipeline = pipeline
        self.session_status = session_status
        self.video_period_ns = int(1_000_000_000 / video_fps)
        self.jpeg_quality = jpeg_quality
        self.perception_renderer = perception_renderer
        self.bev_renderer = bev_renderer
        self.map_renderer = map_renderer
        self.map_localization = map_localization
        self.stop_requested = stop_requested
        if (executor is None) != (not session_status.motion_control_available):
            raise ValueError(
                "executor presence must match motion_control_available."
            )
        if gripper_executor is not None and executor is None:
            raise ValueError("gripper_executor requires a motion executor.")
        if (map_renderer is None) != (not session_status.map_snapshot_available):
            raise ValueError(
                "map_renderer presence must match map_snapshot_available."
            )
        self.vehicle = VehicleState(safety_mode=safety_mode)
        self.video_mode = VideoFrameMode.RAW
        self.last_sent_video_sequence: int | None = None
        self.minimum_rendered_sequence: int | None = None
        self.latest_frame: CameraFrame | None = None
        self.last_camera_frame_ns = time.monotonic_ns()
        self.next_video_ns = 0
        now_ns = time.monotonic_ns()
        self.next_session_status_ns = now_ns + SESSION_STATUS_PERIOD_MS * 1_000_000
        self.next_vehicle_status_ns = now_ns + VEHICLE_STATUS_PERIOD_MS * 1_000_000
        self.next_capture_status_ns = now_ns + CAPTURE_STATUS_PERIOD_MS * 1_000_000
        self.next_map_snapshot_ns = now_ns

    def run(self) -> None:
        self._send_initial_status()
        try:
            if self.executor is None:
                self._run_camera_only()
            else:
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
        finally:
            if self.gripper_executor is not None:
                self.gripper_executor.stop()

    def _run_camera_only(self) -> None:
        while not self.stop_requested():
            self._cycle()
            try:
                message = self.connection.receive_control(timeout=0.02)
            except TimeoutError:
                continue
            self._handle_other_control(message)
            for _ in range(CAMERA_ONLY_CONTROL_BATCH_LIMIT - 1):
                try:
                    message = self.connection.receive_control(timeout=0)
                except TimeoutError:
                    break
                self._handle_other_control(message)

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
        if self.gripper_executor is not None:
            timed_out_command_id = self.gripper_executor.check_timeout()
            if timed_out_command_id is None:
                self.gripper_executor.update()
            else:
                self.capture.record_gripper_timeout(
                    command_id=timed_out_command_id,
                    timestamp_ns=time.monotonic_ns(),
                )
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
            if self.map_localization is not None:
                self.map_localization.submit(self.latest_frame)
            if self.video_mode is VideoFrameMode.PERCEPTION:
                if self.perception_renderer is None:
                    raise RuntimeError(
                        "Perception video mode is not available in this session."
                    )
                self.perception_renderer.submit(self.latest_frame)
            elif self.video_mode is VideoFrameMode.BEV:
                if self.bev_renderer is None:
                    raise RuntimeError("BEV video mode is not available in this session.")
                self.bev_renderer.submit(self.latest_frame)
        now_ns = time.monotonic_ns()
        if self.latest_frame is not None and now_ns >= self.next_video_ns:
            self._send_current_video()
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
            if self.session_status.vehicle_state_available:
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
        if now_ns >= self.next_map_snapshot_ns:
            self._send_map_snapshot(now_ns)
            self.next_map_snapshot_ns = now_ns + MAP_SNAPSHOT_PERIOD_MS * 1_000_000

    def _send_initial_status(self) -> None:
        self.connection.send_reliable_observation(
            RemoteTopic.SESSION_STATUS.value,
            self.session_status.to_payload(),
            content_type="application/json",
        )
        if self.session_status.vehicle_state_available:
            self.connection.send_observation(
                RemoteTopic.VEHICLE_STATE.value,
                self.vehicle.observation().to_payload(),
                content_type="application/json",
            )
        self._send_capture_status()
        now_ns = time.monotonic_ns()
        self._send_map_snapshot(now_ns)
        self.next_map_snapshot_ns = now_ns + MAP_SNAPSHOT_PERIOD_MS * 1_000_000

    def _handle_other_control(self, message: ReceivedRemoteMessage) -> None:
        if (
            message.stream is not RemoteStream.CONTROL
            or message.content_type != "application/json"
            or message.attributes
        ):
            raise ValueError(
                f"Unsupported remote control message {message.topic!r}."
            )
        if message.topic == RemoteTopic.VIDEO_MODE.value:
            command = VideoModeCommand.from_payload(message.payload)
            if command.mode not in self.session_status.video_modes:
                raise ValueError(
                    f"Video mode {command.mode.value!r} is not available."
                )
            if (
                command.mode is VideoFrameMode.PERCEPTION
                and self.perception_renderer is None
            ):
                raise ValueError("Perception video mode is not configured.")
            if command.mode is VideoFrameMode.BEV and self.bev_renderer is None:
                raise ValueError("BEV video mode is not configured.")
            self.video_mode = command.mode
            self.last_sent_video_sequence = None
            if command.mode in (VideoFrameMode.PERCEPTION, VideoFrameMode.BEV):
                renderer = (
                    self.perception_renderer
                    if command.mode is VideoFrameMode.PERCEPTION
                    else self.bev_renderer
                )
                assert renderer is not None
                renderer.clear_latest()
                self.minimum_rendered_sequence = (
                    None if self.latest_frame is None else self.latest_frame.sequence
                )
            if command.mode is VideoFrameMode.PERCEPTION:
                assert self.perception_renderer is not None
                if self.latest_frame is not None:
                    self.perception_renderer.submit(self.latest_frame)
            elif command.mode is VideoFrameMode.BEV:
                assert self.bev_renderer is not None
                if self.latest_frame is not None:
                    self.bev_renderer.submit(self.latest_frame)
            else:
                self.minimum_rendered_sequence = None
            return
        if message.topic == RemoteTopic.DEBUG_MOTION.value:
            if self.executor is None:
                # 部分旧客户端即使 capability=false 仍会周期发送运动心跳。
                # 严格解析后安全丢弃，避免仅相机调试会话被无动作消息断开。
                DebugMotionCommand.from_payload(message.payload)
                return
            raise ValueError("Unexpected motion command outside the motion loop.")
        if message.topic == RemoteTopic.DEBUG_GRIPPER.value:
            if self.gripper_executor is None:
                if self.executor is None:
                    # 与运动心跳相同：仅相机模式没有 UART，合法夹爪状态不会
                    # 产生任何物理动作，可以兼容性丢弃；畸形 payload 仍会报错。
                    DebugGripperCommand.from_payload(message.payload)
                    return
                raise ValueError(
                    "Remote gripper control is disabled by runtime config."
                )
            outcome = self.gripper_executor.execute(message)
            self.vehicle.on_gripper(outcome)
            self.capture.record_gripper(outcome)
            return
        if message.topic != RemoteTopic.DEBUG_CAPTURE.value:
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

    def _send_current_video(self) -> None:
        frame = self.latest_frame
        if self.video_mode is VideoFrameMode.PERCEPTION:
            if self.perception_renderer is None:
                raise RuntimeError("Perception video mode is not configured.")
            frame = self.perception_renderer.latest()
            if frame is None:
                return
            if self.minimum_rendered_sequence is None:
                self.minimum_rendered_sequence = frame.sequence
            if frame.sequence < self.minimum_rendered_sequence:
                return
        elif self.video_mode is VideoFrameMode.BEV:
            if self.bev_renderer is None:
                raise RuntimeError("BEV video mode is not configured.")
            frame = self.bev_renderer.latest()
            if frame is None:
                return
            if self.minimum_rendered_sequence is None:
                self.minimum_rendered_sequence = frame.sequence
            if frame.sequence < self.minimum_rendered_sequence:
                return
        if frame is None or frame.sequence == self.last_sent_video_sequence:
            return
        _send_video_frame(
            self.connection,
            frame,
            self.pipeline,
            jpeg_quality=self.jpeg_quality,
            mode=self.video_mode,
        )
        self.last_sent_video_sequence = frame.sequence

    def _send_capture_status(self) -> None:
        self.connection.send_reliable_observation(
            RemoteTopic.CAPTURE_STATUS.value,
            self.capture.status().to_payload(),
            content_type="application/json",
        )

    def _send_map_snapshot(self, timestamp_ns: int) -> None:
        if self.map_renderer is None:
            return
        robot = (
            None
            if self.map_localization is None
            else self.map_localization.latest_robot_pose(timestamp_ns)
        )
        snapshot = self.map_renderer.render(
            timestamp_ns=timestamp_ns,
            robot=robot,
        )
        self.connection.send_observation(
            RemoteTopic.MAP_SNAPSHOT.value,
            snapshot.png_bytes,
            content_type="image/png",
            attributes=snapshot.attributes.to_attributes(),
            sender_timestamp_ns=timestamp_ns,
        )


def build_camera_pipeline(config: AppConfig) -> CameraPipeline:
    geometry = config.build_geometry()
    camera_model = None if geometry is None else geometry.camera_model
    ground_projector = None if geometry is None else geometry.ground_projector
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
            else camera_model.calibration.calibration_id
        ),
        ground_projector,
    )


def build_session_status(
    config: AppConfig,
    *,
    server_instance_id: str,
    video_fps: float,
    video_modes: tuple[VideoFrameMode, ...] = (VideoFrameMode.RAW,),
    camera_only: bool = False,
    map_snapshot_available: bool = False,
) -> RemoteSessionStatus:
    if not isinstance(camera_only, bool):
        raise TypeError("camera_only must be a boolean.")
    motion_available = not camera_only
    return RemoteSessionStatus(
        session_id=f"session-{uuid.uuid4()}",
        server_instance_id=server_instance_id,
        timestamp_ns=time.monotonic_ns(),
        access_mode=config.remote.access_mode,
        motion_control_available=motion_available,
        gripper_control_available=(
            motion_available and config.motion.gripper.enabled
        ),
        capture_control_available=True,
        video_stream_available=True,
        video_modes=video_modes,
        map_snapshot_available=map_snapshot_available,
        vehicle_state_available=True,
        capture_status_available=True,
        target_heading_control_available=False,
        session_status_period_ms=SESSION_STATUS_PERIOD_MS,
        vehicle_state_period_ms=(
            CAMERA_ONLY_VEHICLE_STATUS_PERIOD_MS
            if camera_only
            else VEHICLE_STATUS_PERIOD_MS
        ),
        map_snapshot_period_ms=(
            MAP_SNAPSHOT_PERIOD_MS if map_snapshot_available else None
        ),
        capture_status_period_ms=CAPTURE_STATUS_PERIOD_MS,
        video_nominal_fps=video_fps,
        max_linear_velocity_m_s=(
            config.motion.max_linear_velocity_m_s if motion_available else None
        ),
        max_angular_velocity_rad_s=(
            config.motion.max_angular_velocity_rad_s if motion_available else None
        ),
        max_control_command_valid_for_ms=(
            config.motion.max_remote_command_valid_for_ms
        ),
    )


def run_manual_capture_session(
    connection: RemoteMessageConnection,
    executor: RemoteMotionExecutor | None,
    gripper_executor: RemoteGripperExecutor | None,
    capture: CaptureSession,
    pipeline: CameraPipeline,
    session_status: RemoteSessionStatus,
    *,
    video_fps: float,
    jpeg_quality: int,
    perception_renderer: PerceptionFrameRenderer | None = None,
    bev_renderer: BevFrameRenderer | None = None,
    map_renderer: FieldMapSnapshotRenderer | None = None,
    map_localization: LatestCenterCrossLocalization | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
    safety_mode: VehicleSafetyMode = VehicleSafetyMode.UNAVAILABLE,
) -> None:
    runtime = ManualCaptureRuntime(
        connection=connection,
        executor=executor,
        gripper_executor=gripper_executor,
        capture=capture,
        pipeline=pipeline,
        session_status=session_status,
        video_fps=video_fps,
        jpeg_quality=jpeg_quality,
        perception_renderer=perception_renderer,
        bev_renderer=bev_renderer,
        map_renderer=map_renderer,
        map_localization=map_localization,
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
            add_exception_note(
                exc,
                f"Safety stop logging also failed: {log_error!r}",
            )
        try:
            capture.close()
        except BaseException as cleanup_error:
            add_exception_note(
                exc,
                f"Capture session cleanup also failed: {cleanup_error!r}",
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
    mode: VideoFrameMode = VideoFrameMode.RAW,
) -> None:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame.image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode the remote JPEG frame.")
    height, width = frame.image_bgr.shape[:2]
    bev_config: BevConfig | None = None
    coordinate_system = pipeline.coordinate_system
    if mode is VideoFrameMode.BEV:
        if pipeline.ground_projector is None:
            raise RuntimeError("BEV video requires a configured ground projector.")
        bev_config = pipeline.ground_projector.bev_config
        if bev_config is None:
            raise RuntimeError("BEV video requires a configured BEV mapping.")
        coordinate_system = ImageCoordinateSystem.BEV_PIXEL
    attributes = VideoFrameAttributes(
        frame_sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        width=width,
        height=height,
        coordinate_system=coordinate_system,
        calibration_id=pipeline.calibration_id,
        mode=mode,
        bev_x_min_mm=None if bev_config is None else bev_config.x_min,
        bev_x_max_mm=None if bev_config is None else bev_config.x_max,
        bev_y_min_mm=None if bev_config is None else bev_config.y_min,
        bev_y_max_mm=None if bev_config is None else bev_config.y_max,
        bev_mm_per_pixel=(
            None if bev_config is None else bev_config.mm_per_pixel
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
    controller: MotionController | None,
    *,
    timeout_s: float,
    stop_requested: Callable[[], bool],
) -> RemoteMessageConnection | None:
    deadline = time.monotonic() + timeout_s
    while not stop_requested():
        # STM32 continuously publishes 10 Hz telemetry, including while no
        # remote client is connected.  Keep the UART's bounded queue healthy
        # instead of leaving it unconsumed for the whole accept timeout.
        if controller is not None:
            controller.drain_messages()
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return None
        try:
            return server.accept(timeout=min(0.1, remaining_s))
        except TimeoutError:
            continue
    return None


def _validate_mode(
    config: AppConfig,
    *,
    video_fps: float,
    supervised_physical_stop_ready: bool,
    camera_only: bool = False,
) -> None:
    if not isinstance(camera_only, bool):
        raise TypeError("camera_only must be a boolean.")
    if not config.remote.enabled or config.remote.role is not RemoteRole.SERVER:
        raise RuntimeError("Manual capture requires remote server mode.")
    if config.remote.access_mode is not RemoteAccessMode.DEBUG_CONTROL:
        raise RuntimeError("Manual capture requires remote debug_control mode.")
    if not camera_only and (not config.uart.enabled or not config.motion.enabled):
        raise RuntimeError("Manual capture requires enabled UART and motion.")
    if not camera_only and not supervised_physical_stop_ready:
        raise RuntimeError(
            "Current firmware watchdog state is unavailable; pass "
            "--supervised-physical-stop-ready only after a physical emergency "
            "stop is ready and an operator will supervise the full session."
        )
    if video_fps > config.camera.fps:
        raise RuntimeError("--video-fps must not exceed camera.fps.")
    observation_capacity = getattr(
        config.remote,
        "observation_queue_capacity",
        None,
    )
    static_map = getattr(getattr(config, "world", None), "static_map", None)
    map_available = bool(getattr(static_map, "regions", ()))
    required_observation_capacity = 3 if map_available else 2
    if (
        observation_capacity is not None
        and observation_capacity < required_observation_capacity
    ):
        raise RuntimeError(
            "remote.observation_queue_capacity must be at least "
            f"{required_observation_capacity} for video, vehicle state"
            f"{' and map' if map_available else ''}; got "
            f"{observation_capacity}."
        )


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
        "--camera-only",
        action="store_true",
        help=(
            "Run video, BEV and capture without opening UART or advertising "
            "motion/gripper control capabilities."
        ),
    )
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
        camera_only=args.camera_only,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config_snapshot = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    server = config.remote.build_server()
    channel = None if args.camera_only else config.uart.build_channel()
    controller: MotionController | None = (
        None if args.camera_only else config.motion.build_controller(channel)
    )
    executor = (
        None if args.camera_only else config.motion.build_remote_executor(controller)
    )
    gripper_executor = (
        None
        if args.camera_only
        else config.motion.build_remote_gripper_executor(controller)
    )
    assert server is not None
    if not args.camera_only:
        assert channel is not None
        assert controller is not None
        assert executor is not None
    pipeline = build_camera_pipeline(config)
    perception_renderer = (
        PerceptionFrameRenderer(config.build_target_pose_detector)
        if config.hailo.enabled
        else None
    )
    bev_renderer = (
        BevFrameRenderer(pipeline.ground_projector)
        if pipeline.ground_projector is not None
        and pipeline.ground_projector.bev_config is not None
        else None
    )
    map_renderer = (
        FieldMapSnapshotRenderer(config.world.static_map, config.world.team_color)
        if config.world.static_map.regions
        else None
    )
    field_detector = (
        config.perception.build_field_feature_detector(
            static_map=config.world.static_map,
            max_observation_age_ms=config.processing.max_observation_age_ms,
            ground_projector=pipeline.ground_projector,
        )
        if pipeline.ground_projector is not None
        else None
    )
    center_cross_localizer = config.build_center_cross_localizer(
        ground_projector=pipeline.ground_projector,
    )
    map_localization = (
        LatestCenterCrossLocalization(
            field_detector,
            center_cross_localizer,
            valid_mask=pipeline.camera_model.valid_mask,
            max_pose_age_ms=config.processing.max_observation_age_ms,
        )
        if field_detector is not None
        and center_cross_localizer is not None
        and pipeline.camera_model is not None
        else None
    )
    video_modes = (VideoFrameMode.RAW,) + (
        (VideoFrameMode.PERCEPTION,) if perception_renderer is not None else ()
    ) + ((VideoFrameMode.BEV,) if bev_renderer is not None else ())
    server_instance_id = f"manual-capture-{uuid.uuid4()}"
    shutdown_requested = threading.Event()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: shutdown_requested.set(),
    )

    try:
        with ExitStack() as resources:
            resources.enter_context(server)
            if channel is not None:
                resources.enter_context(channel)
            try:
                if perception_renderer is not None:
                    perception_renderer.start()
                if bev_renderer is not None:
                    bev_renderer.start()
                if map_localization is not None:
                    map_localization.start()
                pipeline.source.start()
                while not shutdown_requested.is_set():
                    connection = _accept_with_shutdown(
                        server,
                        controller,
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
                        motion_logging_enabled=not args.camera_only,
                    )
                    status = build_session_status(
                        config,
                        server_instance_id=server_instance_id,
                        video_fps=args.video_fps,
                        video_modes=video_modes,
                        camera_only=args.camera_only,
                        map_snapshot_available=map_renderer is not None,
                    )
                    try:
                        with connection:
                            run_manual_capture_session(
                                connection,
                                executor,
                                gripper_executor,
                                capture,
                                pipeline,
                                status,
                                video_fps=args.video_fps,
                                jpeg_quality=args.jpeg_quality,
                                perception_renderer=perception_renderer,
                                bev_renderer=bev_renderer,
                                map_renderer=map_renderer,
                                map_localization=map_localization,
                                stop_requested=shutdown_requested.is_set,
                                safety_mode=(
                                    VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP
                                    if not args.camera_only
                                    else VehicleSafetyMode.UNAVAILABLE
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
                    if executor is not None:
                        executor.stop()
                finally:
                    try:
                        pipeline.source.stop()
                    finally:
                        try:
                            if perception_renderer is not None:
                                perception_renderer.stop()
                        finally:
                            try:
                                if bev_renderer is not None:
                                    bev_renderer.stop()
                            finally:
                                if map_localization is not None:
                                    map_localization.stop()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
