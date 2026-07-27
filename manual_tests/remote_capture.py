"""接收电脑端采集指令，并用真实相机执行录制、抓拍和事件标记。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

if __package__:
    from manual_tests.remote_camera_support import (
        CAPTURE_STATUS_PERIOD_MS,
        SESSION_STATUS_PERIOD_MS,
        RemoteCameraPipeline,
        build_camera_session_status,
        build_remote_camera_pipeline,
        refresh_session_status,
        send_video_frame,
    )
else:
    from remote_camera_support import (
        CAPTURE_STATUS_PERIOD_MS,
        SESSION_STATUS_PERIOD_MS,
        RemoteCameraPipeline,
        build_camera_session_status,
        build_remote_camera_pipeline,
        refresh_session_status,
        send_video_frame,
    )
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.communication import (
    CaptureAction,
    CaptureRecordingState,
    CaptureRequestResult,
    CaptureStatusObservation,
    CaptureStopReason,
    DebugCaptureCommand,
    RemoteAccessMode,
    RemoteDisconnectedError,
    RemoteMessageConnection,
    RemoteRole,
    RemoteTopic,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.config.runtime import AppConfig
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE
from rescue_vision.versioning import git_version


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    action: CaptureAction
    result: CaptureRequestResult
    artifact_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class CaptureTestSession:
    """单连接内的最小采集状态机；所有路径均由车端生成。"""

    def __init__(
        self,
        *,
        output_root: Path,
        config_path: Path,
        config: AppConfig,
        pipeline: RemoteCameraPipeline,
    ) -> None:
        self.output_root = output_root
        self.config_path = config_path
        self.config = config
        self.pipeline = pipeline
        self.recorder: FrameRecorder | None = None
        self.recording_id: str | None = None
        self.recording_directory: Path | None = None
        self.status_sequence = 0
        self.last_outcome: tuple[str, RequestOutcome] | None = None
        self.outcomes: dict[str, RequestOutcome] = {}
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

    def accepted(self, command: DebugCaptureCommand) -> None:
        self.last_outcome = (
            command.request_id,
            RequestOutcome(
                action=command.action,
                result=CaptureRequestResult.ACCEPTED,
            ),
        )

    def execute(
        self,
        command: DebugCaptureCommand,
        latest_frame: CameraFrame,
    ) -> RequestOutcome:
        previous = self.outcomes.get(command.request_id)
        if previous is not None:
            self.last_outcome = (command.request_id, previous)
            return previous

        try:
            if command.action is CaptureAction.START:
                outcome = self._start(command)
            elif command.action is CaptureAction.STOP:
                outcome = self._stop(command)
            elif command.action is CaptureAction.SNAPSHOT:
                outcome = self._snapshot(command, latest_frame)
            else:
                outcome = self._mark_event(command, latest_frame)
        except Exception as error:
            print(
                f"request_failed={command.request_id} error={error!r}",
                flush=True,
            )
            outcome = RequestOutcome(
                action=command.action,
                result=CaptureRequestResult.FAILED,
                error_code="internal_error",
                error_message="capture action failed; inspect vehicle log",
            )
        self.outcomes[command.request_id] = outcome
        self.last_outcome = (command.request_id, outcome)
        return outcome

    def record(self, frame: CameraFrame) -> None:
        if self.recorder is not None:
            self.recorder.record(frame)

    def close(self) -> None:
        if self.recorder is None:
            return
        recorder = self.recorder
        self.recorder = None
        self.recording_id = None
        self.recording_directory = None
        self.stop_reason = CaptureStopReason.APPLICATION_SHUTDOWN
        try:
            recorder.stop()
        finally:
            self._remember_counts(recorder)

    def _start(self, command: DebugCaptureCommand) -> RequestOutcome:
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
            config_snapshot=yaml.safe_load(
                self.config_path.read_text(encoding="utf-8")
            ),
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
                IMAGE_BORDER_FILL_VALUE
                if camera_model is not None
                else None
            ),
        )
        recorder.start()
        self.recorder = recorder
        self.recording_id = recording_id
        self.recording_directory = directory
        self.stop_reason = None
        self.accepted_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0
        return RequestOutcome(
            action=command.action,
            result=CaptureRequestResult.COMPLETED,
            artifact_id=recording_id,
        )

    def _stop(self, command: DebugCaptureCommand) -> RequestOutcome:
        if self.recorder is None:
            return self._rejected(command, "no recording is active")
        recorder = self.recorder
        artifact_id = self.recording_id
        try:
            recorder.stop()
        except Exception:
            self.faulted = True
            self.stop_reason = CaptureStopReason.WRITE_ERROR
            raise
        finally:
            self._remember_counts(recorder)
            self.recorder = None
            self.recording_id = None
            self.recording_directory = None
        self.stop_reason = CaptureStopReason.REQUESTED
        return RequestOutcome(
            action=command.action,
            result=CaptureRequestResult.COMPLETED,
            artifact_id=artifact_id,
        )

    def _snapshot(
        self,
        command: DebugCaptureCommand,
        frame: CameraFrame,
    ) -> RequestOutcome:
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
        return RequestOutcome(
            action=command.action,
            result=CaptureRequestResult.COMPLETED,
            artifact_id=artifact_id,
        )

    def _mark_event(
        self,
        command: DebugCaptureCommand,
        frame: CameraFrame,
    ) -> RequestOutcome:
        if self.recording_directory is None:
            return self._rejected(command, "event marking requires recording")
        artifact_id = _artifact_id("event")
        with (self.recording_directory / "events.jsonl").open(
            "a",
            encoding="utf-8",
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
        return RequestOutcome(
            action=command.action,
            result=CaptureRequestResult.COMPLETED,
            artifact_id=artifact_id,
        )

    @staticmethod
    def _rejected(
        command: DebugCaptureCommand,
        message: str,
    ) -> RequestOutcome:
        return RequestOutcome(
            action=command.action,
            result=CaptureRequestResult.REJECTED,
            error_code="state_conflict",
            error_message=message,
        )

    def _remember_counts(self, recorder: FrameRecorder) -> None:
        self.accepted_frames = recorder.accepted_frames
        self.written_frames = recorder.written_frames
        self.dropped_frames = recorder.dropped_frames


def _artifact_id(prefix: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _send_capture_status(
    connection: RemoteMessageConnection,
    session: CaptureTestSession,
) -> None:
    connection.send_reliable_observation(
        RemoteTopic.CAPTURE_STATUS.value,
        session.status().to_payload(),
        content_type="application/json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Accept capture controls and execute recordings/snapshots locally."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config = load_runtime_config(config_path)
    if not config.remote.enabled:
        raise RuntimeError("remote.enabled must be true")
    if config.remote.role is not RemoteRole.SERVER:
        raise RuntimeError(
            "remote_capture.py accepts desktop clients; set remote.role: server"
        )
    if config.remote.access_mode is not RemoteAccessMode.DEBUG_CONTROL:
        raise RuntimeError(
            "remote_capture.py executes controls; set access_mode: debug_control"
        )
    if args.video_fps > config.camera.fps:
        raise RuntimeError("--video-fps must not exceed camera.fps")
    output_root.mkdir(parents=True, exist_ok=True)

    server = config.remote.build_server()
    assert server is not None
    pipeline = build_remote_camera_pipeline(config)
    session_status = build_camera_session_status(
        config,
        server_instance_id=f"remote-capture-{uuid.uuid4()}",
        video_nominal_fps=args.video_fps,
        capture_control_available=True,
    )
    capture_session = CaptureTestSession(
        output_root=output_root,
        config_path=config_path,
        config=config,
        pipeline=pipeline,
    )
    video_period_ns = int(1_000_000_000 / args.video_fps)
    status_period_ns = SESSION_STATUS_PERIOD_MS * 1_000_000
    capture_period_ns = CAPTURE_STATUS_PERIOD_MS * 1_000_000

    with server:
        print(
            f"listening={config.remote.host}:{server.bound_port} "
            f"output_root={output_root}",
            flush=True,
        )
        pipeline.source.start()
        try:
            connection = server.accept(timeout=args.timeout_seconds)
            with connection:
                connection.send_reliable_observation(
                    RemoteTopic.SESSION_STATUS.value,
                    session_status.to_payload(),
                    content_type="application/json",
                )
                _send_capture_status(connection, capture_session)
                print("client_connected=true", flush=True)
                next_video_ns = 0
                now_ns = time.monotonic_ns()
                next_status_ns = now_ns + status_period_ns
                next_capture_status_ns = now_ns + capture_period_ns
                try:
                    while True:
                        frame = pipeline.prepare(
                            pipeline.source.read(timeout=1.0)
                        )
                        capture_session.record(frame)
                        now_ns = time.monotonic_ns()

                        if now_ns >= next_status_ns:
                            session_status = refresh_session_status(
                                connection,
                                session_status,
                                timestamp_ns=now_ns,
                            )
                            next_status_ns = now_ns + status_period_ns
                        if now_ns >= next_capture_status_ns:
                            _send_capture_status(connection, capture_session)
                            next_capture_status_ns = now_ns + capture_period_ns
                        if now_ns >= next_video_ns:
                            send_video_frame(
                                connection,
                                frame,
                                pipeline,
                                jpeg_quality=args.jpeg_quality,
                            )
                            next_video_ns = now_ns + video_period_ns

                        while True:
                            try:
                                message = connection.receive_control(timeout=0)
                            except TimeoutError:
                                break
                            if (
                                message.topic
                                != RemoteTopic.DEBUG_CAPTURE.value
                                or message.content_type != "application/json"
                                or message.attributes
                            ):
                                raise ValueError(
                                    "Only control/debug/capture JSON with empty "
                                    "attributes is accepted."
                                )
                            command = DebugCaptureCommand.from_payload(
                                message.payload
                            )
                            if command.request_id not in capture_session.outcomes:
                                capture_session.accepted(command)
                                _send_capture_status(
                                    connection,
                                    capture_session,
                                )
                            outcome = capture_session.execute(command, frame)
                            _send_capture_status(connection, capture_session)
                            print(
                                f"request_id={command.request_id} "
                                f"action={command.action.value} "
                                f"result={outcome.result.value} "
                                f"artifact_id={outcome.artifact_id}",
                                flush=True,
                            )
                except RemoteDisconnectedError:
                    print("client_connected=false", flush=True)
                except KeyboardInterrupt:
                    print("stopped_by_user=true", flush=True)
        finally:
            try:
                capture_session.close()
            finally:
                pipeline.source.stop()


if __name__ == "__main__":
    main()
