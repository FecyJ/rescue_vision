"""固定出发姿态的目标团解团真车试验入口。"""

from __future__ import annotations

import argparse
import math
from queue import Empty, Full, Queue
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Protocol
from uuid import uuid4

from rescue_vision.camera.frame import CameraFrame, FrameSource
from rescue_vision.config import (
    AppConfig,
    ClusterBreakupRuntimeConfig,
    load_runtime_config,
)
from rescue_vision.communication import (
    ImageCoordinateSystem,
    MapStateObservation,
    ReceivedRemoteMessage,
    RemoteAccessMode,
    RemoteMessageConnection,
    RemoteRole,
    RemoteSessionStatus,
    RemoteStream,
    TeamColor as RemoteTeamColor,
    RemoteTcpServer,
    RemoteTopic,
    VideoFrameMode,
    VideoModeCommand,
)
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.localization import (
    FusedPoseEstimate,
    OdometryCalibration,
    OdometryImuFusion,
    VisualAnchorHealth,
)
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    MotionController,
    OdometryImu,
    SensorFlags,
)
from rescue_vision.perception import (
    PerceptionFrameRenderer,
    PerceptionSnapshot,
    TargetClass,
    TargetObservation,
)

_REMOTE_CONTROL_BATCH_LIMIT = 4

# 旁路 worker 的耗时诊断打印间隔；只用于定位感知链路瓶颈。
_TIMING_REPORT_INTERVAL_NS = 2_000_000_000


class BreakupState(str, Enum):
    WAIT_ODOMETRY = "wait_odometry"
    LEAVE_START = "leave_start"
    SEARCH_CLUSTER = "search_cluster"
    CENTER_CLUSTER = "center_cluster"
    APPROACH_CLUSTER = "approach_cluster"
    BREAKUP_PUSH = "breakup_push"
    BREAKUP_RELEASE = "breakup_release"
    BREAKUP_OPEN_RETREAT = "breakup_open_retreat"
    BREAKUP_CLOSE = "breakup_close"
    RETREAT = "retreat"
    SCAN_GREEN = "scan_green"
    GREEN_FOUND = "green_found"
    FAULT = "fault"


class GripperPosture(str, Enum):
    CLOSED = "closed"
    TRANSPORT = "transport"
    OPEN = "open"


class _PerceptionSubmitter(Protocol):
    def start(self) -> None: ...

    def submit(self, frame: CameraFrame) -> None: ...

    def check_health(self) -> None: ...

    def stop(self) -> None: ...


class CameraPerceptionPump:
    """在运动线程之外取帧、去畸变并提交最新帧推理。"""

    def __init__(
        self,
        source: FrameSource,
        prepare: Callable[[CameraFrame], CameraFrame],
        perception: _PerceptionSubmitter,
    ) -> None:
        if not isinstance(source, FrameSource):
            raise TypeError("source must implement FrameSource.")
        if not callable(prepare):
            raise TypeError("prepare must be callable.")
        self._source = source
        self._prepare = prepare
        self._perception = perception
        self._stop_event = Event()
        self._error_lock = Lock()
        self._worker_error: BaseException | None = None
        self._startup_error: BaseException | None = None
        self._thread: Thread | None = None
        self._startup_thread: Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("CameraPerceptionPump is already started.")
        self._stop_event.clear()
        with self._error_lock:
            self._worker_error = None
        perception_started = False
        source_started = False
        try:
            self._perception.start()
            perception_started = True
            self._source.start()
            source_started = True
            self._thread = Thread(
                target=self._worker_loop,
                name="rescue-cluster-breakup-camera",
                daemon=True,
            )
            self._thread.start()
            self._started = True
        except BaseException:
            if source_started:
                self._source.stop()
            if perception_started:
                self._perception.stop()
            self._thread = None
            raise

    def start_in_background(self) -> Thread:
        """并行启动相机/Hailo；调用方可同时消费 UART 并执行同步。"""

        if self._started:
            raise RuntimeError("CameraPerceptionPump is already started.")
        if self._startup_thread is not None and self._startup_thread.is_alive():
            raise RuntimeError("CameraPerceptionPump startup is already running.")
        with self._error_lock:
            self._startup_error = None

        def bootstrap() -> None:
            try:
                self.start()
            except BaseException as exc:
                with self._error_lock:
                    self._startup_error = exc

        thread = Thread(
            target=bootstrap,
            name="rescue-cluster-breakup-camera-start",
            daemon=True,
        )
        self._startup_thread = thread
        thread.start()
        return thread

    def wait_until_started(
        self,
        thread: Thread | None = None,
        *,
        timeout_s: float = 15.0,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        """等待异步启动完成，同时允许调用方继续服务 UART。"""

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive.")
        if on_wait is not None and not callable(on_wait):
            raise TypeError("on_wait must be callable or None.")
        startup_thread = thread or self._startup_thread
        if startup_thread is None:
            raise RuntimeError("CameraPerceptionPump has no background startup.")
        deadline = time.monotonic() + float(timeout_s)
        while startup_thread.is_alive():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError("CameraPerceptionPump startup timed out.")
            startup_thread.join(timeout=min(0.01, remaining_s))
            if on_wait is not None:
                on_wait()
        if startup_thread.is_alive():
            raise RuntimeError("CameraPerceptionPump startup timed out.")
        self._startup_thread = None
        with self._error_lock:
            error = self._startup_error
            self._startup_error = None
        if error is not None:
            raise RuntimeError("Camera/perception input pump failed to start.") from error
        if not self._started:
            raise RuntimeError("Camera/perception input pump did not start.")

    def check_health(self) -> None:
        if not self._started:
            raise RuntimeError("CameraPerceptionPump is not started.")
        self._raise_worker_error()
        self._perception.check_health()

    def stop(self) -> None:
        startup_thread = self._startup_thread
        if startup_thread is not None and startup_thread.is_alive():
            startup_thread.join(timeout=15.0)
        self._startup_thread = None
        if not self._started:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            # 正常 read 最多阻塞 50 ms；此处只用于异常帧源的退出兜底。
            self._source.stop()
            thread.join(timeout=2.0)
        thread_alive = thread is not None and thread.is_alive()
        self._thread = None
        self._started = False
        try:
            self._source.stop()
        finally:
            self._perception.stop()
        if thread_alive:
            raise RuntimeError("CameraPerceptionPump worker did not stop.")
        self._raise_worker_error()

    def _worker_loop(self) -> None:
        frame_count = 0
        read_ms_total = 0.0
        read_ms_max = 0.0
        prepare_ms_total = 0.0
        prepare_ms_max = 0.0
        next_report_ns = time.monotonic_ns() + _TIMING_REPORT_INTERVAL_NS
        try:
            while not self._stop_event.is_set():
                read_start_ns = time.monotonic_ns()
                try:
                    frame = self._source.read(timeout=0.05)
                except TimeoutError:
                    continue
                read_ms = (time.monotonic_ns() - read_start_ns) / 1_000_000.0
                prepare_start_ns = time.monotonic_ns()
                self._perception.submit(self._prepare(frame))
                prepare_ms = (time.monotonic_ns() - prepare_start_ns) / 1_000_000.0
                frame_count += 1
                read_ms_total += read_ms
                read_ms_max = max(read_ms_max, read_ms)
                prepare_ms_total += prepare_ms
                prepare_ms_max = max(prepare_ms_max, prepare_ms)
                now_ns = time.monotonic_ns()
                if now_ns >= next_report_ns:
                    if frame_count:
                        print(
                            "camera_pump_timing=(frames="
                            f"{frame_count},"
                            f"read_ms_avg={read_ms_total / frame_count:.1f},"
                            f"read_ms_max={read_ms_max:.1f},"
                            f"prepare_ms_avg={prepare_ms_total / frame_count:.1f},"
                            f"prepare_ms_max={prepare_ms_max:.1f})",
                            flush=True,
                        )
                    frame_count = 0
                    read_ms_total = 0.0
                    read_ms_max = 0.0
                    prepare_ms_total = 0.0
                    prepare_ms_max = 0.0
                    next_report_ns = now_ns + _TIMING_REPORT_INTERVAL_NS
        except BaseException as exc:
            with self._error_lock:
                self._worker_error = exc

    def _raise_worker_error(self) -> None:
        with self._error_lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Camera/perception input pump failed.") from error


class _VideoPipeline(Protocol):
    coordinate_system: ImageCoordinateSystem
    calibration_id: str | None
    ground_projector: object | None


class _ObservationConnection(Protocol):
    def send_reliable_observation(
        self,
        topic: str,
        payload: bytes,
        **kwargs: object,
    ) -> None: ...

    def send_observation(
        self,
        topic: str,
        payload: bytes,
        **kwargs: object,
    ) -> None: ...

    def check_health(self) -> None: ...


class RemotePerceptionPublisher:
    """在独立线程中低频发送最新 perception 可视化帧。"""

    def __init__(
        self,
        connection: _ObservationConnection,
        pipeline: _VideoPipeline,
        session_status: RemoteSessionStatus,
        *,
        jpeg_quality: int = 80,
        min_publish_interval_s: float = 1.0,
    ) -> None:
        for method_name in (
            "send_reliable_observation",
            "send_observation",
            "check_health",
        ):
            if not callable(getattr(connection, method_name, None)):
                raise TypeError(
                    "connection must provide "
                    f"{method_name}() for remote observation."
                )
        if not isinstance(session_status, RemoteSessionStatus):
            raise TypeError("session_status must be a RemoteSessionStatus.")
        if session_status.access_mode is not RemoteAccessMode.OBSERVE_ONLY:
            raise ValueError("Perception publisher requires observe_only status.")
        if session_status.video_modes != (VideoFrameMode.PERCEPTION,):
            raise ValueError(
                "Perception publisher status must advertise only perception video."
            )
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
            raise ValueError("jpeg_quality must be an integer in [1, 100].")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [1, 100].")
        interval = float(min_publish_interval_s)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("min_publish_interval_s must be finite and positive.")
        self._connection = connection
        self._pipeline = pipeline
        self._session_status = session_status
        self._jpeg_quality = jpeg_quality
        self._min_publish_interval_ns = round(interval * 1_000_000_000)
        self._condition = Event()
        self._stop_event = Event()
        self._lock = Lock()
        self._pending_frame: CameraFrame | None = None
        self._last_submitted_sequence: int | None = None
        self._worker_error: BaseException | None = None
        self._thread: Thread | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RemotePerceptionPublisher is already started.")
        self._stop_event.clear()
        self._condition.clear()
        with self._lock:
            self._pending_frame = None
            self._last_submitted_sequence = None
            self._worker_error = None
        # The session status must be queued before any JPEG observation. The
        # connection itself is already started by RemotePerceptionTransport.
        self._connection.send_reliable_observation(
            RemoteTopic.SESSION_STATUS.value,
            self._session_status.to_payload(),
            content_type="application/json",
        )
        self._thread = Thread(
            target=self._worker_loop,
            name="rescue-cluster-breakup-remote-observation",
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
        if not isinstance(frame, CameraFrame):
            raise TypeError("frame must be a CameraFrame.")
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            if (
                self._last_submitted_sequence is not None
                and frame.sequence < self._last_submitted_sequence
            ):
                raise ValueError("Perception video frame sequence moved backwards.")
            if frame.sequence == self._last_submitted_sequence:
                return
            self._last_submitted_sequence = frame.sequence
            self._pending_frame = frame
        self._condition.set()

    def check_health(self) -> None:
        self._require_started()
        self._raise_worker_error()
        self._connection.check_health()
        self._drain_mode_requests()
        self._raise_worker_error()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._condition.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("RemotePerceptionPublisher worker did not stop.")
        self._thread = None
        self._started = False
        self._condition.clear()
        self._raise_worker_error()

    def _worker_loop(self) -> None:
        next_status_ns = time.monotonic_ns() + (
            self._session_status.session_status_period_ms * 1_000_000
        )
        next_frame_ns = 0
        try:
            while not self._stop_event.is_set():
                now_ns = time.monotonic_ns()
                if now_ns >= next_status_ns:
                    self._connection.send_reliable_observation(
                        RemoteTopic.SESSION_STATUS.value,
                        replace(
                            self._session_status,
                            timestamp_ns=now_ns,
                        ).to_payload(),
                        content_type="application/json",
                        sender_timestamp_ns=now_ns,
                    )
                    next_status_ns = now_ns + (
                        self._session_status.session_status_period_ms * 1_000_000
                    )

                frame: CameraFrame | None = None
                if now_ns >= next_frame_ns:
                    with self._lock:
                        frame = self._pending_frame
                        self._pending_frame = None
                    if frame is not None:
                        # Importing the camera/GUI assembly stays on this worker;
                        # the motion loop never performs JPEG encoding.
                        from rescue_vision.app.manual_capture import send_video_frame

                        send_video_frame(
                            self._connection,
                            frame,
                            self._pipeline,
                            jpeg_quality=self._jpeg_quality,
                            mode=VideoFrameMode.PERCEPTION,
                        )
                        next_frame_ns = time.monotonic_ns() + (
                            self._min_publish_interval_ns
                        )

                wait_ns = 50_000_000
                if next_status_ns > now_ns:
                    wait_ns = min(wait_ns, next_status_ns - now_ns)
                if next_frame_ns > now_ns:
                    wait_ns = min(wait_ns, next_frame_ns - now_ns)
                self._condition.wait(timeout=max(wait_ns, 1) / 1_000_000_000.0)
                self._condition.clear()
        except BaseException as error:
            with self._lock:
                self._worker_error = error

    def _drain_mode_requests(self) -> None:
        receive_control = getattr(self._connection, "receive_control", None)
        if not callable(receive_control):
            return
        for _ in range(_REMOTE_CONTROL_BATCH_LIMIT):
            try:
                message = receive_control(timeout=0)
            except TimeoutError:
                return
            if not isinstance(message, ReceivedRemoteMessage):
                raise RuntimeError("Remote mode request has an invalid message type.")
            if (
                message.stream is not RemoteStream.CONTROL
                or message.content_type != "application/json"
                or message.attributes
                or message.topic != RemoteTopic.VIDEO_MODE.value
            ):
                raise RuntimeError(
                    "Cluster breakup observe_only accepts only video mode requests."
                )
            command = VideoModeCommand.from_payload(message.payload)
            if command.mode is not VideoFrameMode.PERCEPTION:
                raise RuntimeError(
                    "Cluster breakup remote observation only supports perception video."
                )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("RemotePerceptionPublisher is not started.")

    def _raise_worker_error(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Remote perception publisher failed.") from error


class RemoteLocalizationPublisher:
    """在独立线程中发布最新编码器+IMU位姿 JSON。"""

    def __init__(
        self,
        connection: _ObservationConnection,
        team_color: RemoteTeamColor,
        *,
        publish_interval_s: float = 0.2,
    ) -> None:
        if not callable(getattr(connection, "send_observation", None)):
            raise TypeError("connection must provide send_observation().")
        if not callable(getattr(connection, "check_health", None)):
            raise TypeError("connection must provide check_health().")
        if not isinstance(team_color, RemoteTeamColor):
            raise TypeError("team_color must be a RemoteTeamColor.")
        interval = float(publish_interval_s)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("publish_interval_s must be finite and positive.")
        self._connection = connection
        self._team_color = team_color
        self._publish_interval_ns = round(interval * 1_000_000_000)
        self._condition = Event()
        self._stop_event = Event()
        self._lock = Lock()
        self._pending: tuple[FusedPoseEstimate | None, int] = (None, 0)
        self._state_sequence = 0
        self._worker_error: BaseException | None = None
        self._thread: Thread | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RemoteLocalizationPublisher is already started.")
        self._stop_event.clear()
        self._condition.clear()
        with self._lock:
            self._pending = (None, 0)
            self._state_sequence = 0
            self._worker_error = None
        self._thread = Thread(
            target=self._worker_loop,
            name="rescue-cluster-breakup-remote-localization",
            daemon=True,
        )
        self._started = True
        try:
            self._thread.start()
        except BaseException:
            self._started = False
            self._thread = None
            raise

    def submit(
        self,
        estimate: FusedPoseEstimate | None,
        timestamp_ns: int,
    ) -> None:
        if estimate is not None and not isinstance(estimate, FusedPoseEstimate):
            raise TypeError("estimate must be FusedPoseEstimate or None.")
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            self._pending = (estimate, timestamp_ns)
        self._condition.set()

    def check_health(self) -> None:
        self._require_started()
        self._raise_worker_error()
        self._connection.check_health()
        self._raise_worker_error()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._condition.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("RemoteLocalizationPublisher worker did not stop.")
        self._thread = None
        self._started = False
        self._condition.clear()
        self._raise_worker_error()

    def _worker_loop(self) -> None:
        next_publish_ns = 0
        try:
            while not self._stop_event.is_set():
                now_ns = time.monotonic_ns()
                if now_ns >= next_publish_ns:
                    with self._lock:
                        estimate, estimate_timestamp_ns = self._pending
                        state_sequence = self._state_sequence
                        self._state_sequence += 1
                    state = self._make_state(
                        state_sequence=state_sequence,
                        timestamp_ns=now_ns,
                        estimate=estimate,
                        estimate_timestamp_ns=estimate_timestamp_ns,
                    )
                    self._connection.send_observation(
                        RemoteTopic.MAP_STATE.value,
                        state.to_payload(),
                        content_type="application/json",
                        sender_timestamp_ns=now_ns,
                    )
                    next_publish_ns = now_ns + self._publish_interval_ns
                wait_ns = max(next_publish_ns - now_ns, 1)
                self._condition.wait(
                    timeout=min(wait_ns, 50_000_000) / 1_000_000_000.0
                )
                self._condition.clear()
        except BaseException as error:
            with self._lock:
                self._worker_error = error

    def _make_state(
        self,
        *,
        state_sequence: int,
        timestamp_ns: int,
        estimate: FusedPoseEstimate | None,
        estimate_timestamp_ns: int,
    ) -> MapStateObservation:
        if (
            estimate is None
            or estimate.pose is None
            or estimate.estimate_timestamp_ns is None
            or estimate.position_uncertainty_mm is None
            or estimate.heading_uncertainty_rad is None
        ):
            return MapStateObservation(
                state_sequence=state_sequence,
                timestamp_ns=timestamp_ns,
                team_color=self._team_color,
                robot_localized=False,
                robot_x_mm=None,
                robot_y_mm=None,
                robot_heading_rad=None,
                localization_timestamp_ns=None,
                localization_confidence=None,
                localization_position_uncertainty_mm=None,
                localization_heading_uncertainty_rad=None,
                localization_source=None,
                targets=(),
            )
        pose = estimate.pose
        return MapStateObservation(
            state_sequence=state_sequence,
            timestamp_ns=timestamp_ns,
            team_color=self._team_color,
            robot_localized=True,
            robot_x_mm=pose.position.x,
            robot_y_mm=pose.position.y,
            robot_heading_rad=pose.heading_rad,
            localization_timestamp_ns=min(
                estimate.estimate_timestamp_ns,
                estimate_timestamp_ns,
            ),
            localization_confidence=estimate.confidence,
            localization_position_uncertainty_mm=estimate.position_uncertainty_mm,
            localization_heading_uncertainty_rad=estimate.heading_uncertainty_rad,
            # 视觉闭环定位接通后锚点来源会变成 center_cross/safe_zone 等，直接
            # 透传真实来源，观察端才能看到“已由视觉纠偏”而非永远显示 odometry_imu。
            localization_source=estimate.anchor_source or "odometry_imu",
            targets=(),
        )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("RemoteLocalizationPublisher is not started.")

    def _raise_worker_error(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Remote localization publisher failed.") from error


class RemotePerceptionTransport:
    """异步接受一个观察客户端，不让正常流程等待 TCP 连接。"""

    def __init__(
        self,
        server: RemoteTcpServer,
        pipeline: _VideoPipeline,
        config: AppConfig,
        *,
        jpeg_quality: int = 80,
        min_publish_interval_s: float = 1.0,
        map_team_color: RemoteTeamColor | None = None,
    ) -> None:
        if not isinstance(server, RemoteTcpServer):
            raise TypeError("server must be a RemoteTcpServer.")
        if config.remote.access_mode is not RemoteAccessMode.OBSERVE_ONLY:
            raise ValueError("RemotePerceptionTransport requires observe_only mode.")
        if config.remote.role is not RemoteRole.SERVER:
            raise ValueError("RemotePerceptionTransport requires server role.")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
            raise ValueError("jpeg_quality must be an integer in [1, 100].")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [1, 100].")
        interval = float(min_publish_interval_s)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("min_publish_interval_s must be finite and positive.")
        if map_team_color is not None and not isinstance(
            map_team_color,
            RemoteTeamColor,
        ):
            raise TypeError("map_team_color must be a RemoteTeamColor or None.")
        observation_capacity = getattr(
            config.remote,
            "observation_queue_capacity",
            None,
        )
        required_capacity = 2 if map_team_color is not None else 1
        if (
            observation_capacity is not None
            and observation_capacity < required_capacity
        ):
            raise ValueError(
                "remote.observation_queue_capacity must be at least "
                f"{required_capacity} for cluster breakup observation topics."
            )
        self._server = server
        self._pipeline = pipeline
        self._config = config
        self._jpeg_quality = jpeg_quality
        self._min_publish_interval_s = min_publish_interval_s
        self._map_team_color = map_team_color
        self._stop_event = Event()
        self._lock = Lock()
        self._worker_error: BaseException | None = None
        self._connection: RemoteMessageConnection | None = None
        self._publisher: RemotePerceptionPublisher | None = None
        self._localization_publisher: RemoteLocalizationPublisher | None = None
        self._thread: Thread | None = None
        self._started = False

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._publisher is not None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RemotePerceptionTransport is already started.")
        self._stop_event.clear()
        with self._lock:
            self._worker_error = None
            self._connection = None
            self._publisher = None
            self._localization_publisher = None
        self._server.start()
        self._thread = Thread(
            target=self._accept_loop,
            name="rescue-cluster-breakup-remote-accept",
            daemon=True,
        )
        self._started = True
        try:
            self._thread.start()
        except BaseException:
            self._started = False
            self._thread = None
            self._server.stop()
            raise

    def submit(self, frame: CameraFrame) -> None:
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            publisher = self._publisher
        if publisher is not None:
            publisher.submit(frame)

    def submit_localization(
        self,
        estimate: FusedPoseEstimate | None,
        timestamp_ns: int,
    ) -> None:
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            publisher = self._localization_publisher
        if publisher is not None:
            publisher.submit(estimate, timestamp_ns)

    def check_health(self) -> None:
        self._require_started()
        self._raise_worker_error()
        with self._lock:
            publisher = self._publisher
            localization_publisher = self._localization_publisher
        if publisher is not None:
            publisher.check_health()
        if localization_publisher is not None:
            localization_publisher.check_health()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._server.stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        thread_alive = thread is not None and thread.is_alive()
        with self._lock:
            publisher = self._publisher
            localization_publisher = self._localization_publisher
            connection = self._connection
        cleanup_error: BaseException | None = None
        if localization_publisher is not None:
            try:
                localization_publisher.stop()
            except BaseException as error:
                cleanup_error = error
        if publisher is not None:
            try:
                publisher.stop()
            except BaseException as error:
                cleanup_error = error
        if connection is not None:
            try:
                connection.stop()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        self._thread = None
        self._started = False
        if thread_alive:
            raise RuntimeError("Remote observation accept worker did not stop.")
        self._raise_worker_error()
        if cleanup_error is not None:
            raise RuntimeError("Remote observation cleanup failed.") from cleanup_error

    def _accept_loop(self) -> None:
        connection: RemoteMessageConnection | None = None
        publisher: RemotePerceptionPublisher | None = None
        localization_publisher: RemoteLocalizationPublisher | None = None
        try:
            while not self._stop_event.is_set():
                try:
                    connection = self._server.accept(timeout=0.1)
                except TimeoutError:
                    continue
                if self._stop_event.is_set():
                    connection.stop()
                    return
                connection.start()
                status = build_cluster_observation_status(
                    self._config,
                    server_instance_id=f"cluster-breakup-{uuid4()}",
                    video_nominal_fps=1.0 / self._min_publish_interval_s,
                    map_state_available=self._map_team_color is not None,
                )
                publisher = RemotePerceptionPublisher(
                    connection,
                    self._pipeline,
                    status,
                    jpeg_quality=self._jpeg_quality,
                    min_publish_interval_s=self._min_publish_interval_s,
                )
                publisher.start()
                if self._map_team_color is not None:
                    localization_publisher = RemoteLocalizationPublisher(
                        connection,
                        self._map_team_color,
                    )
                    localization_publisher.start()
                with self._lock:
                    self._connection = connection
                    self._publisher = publisher
                    self._localization_publisher = localization_publisher
                return
        except BaseException as error:
            if localization_publisher is not None:
                try:
                    localization_publisher.stop()
                except BaseException:
                    pass
            if publisher is not None:
                try:
                    publisher.stop()
                except BaseException:
                    pass
            if connection is not None:
                try:
                    connection.stop()
                except BaseException:
                    pass
            if not self._stop_event.is_set():
                with self._lock:
                    self._worker_error = error

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("RemotePerceptionTransport is not started.")

    def _raise_worker_error(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Remote observation transport failed.") from error


def build_cluster_observation_status(
    config: AppConfig,
    *,
    server_instance_id: str,
    video_nominal_fps: float = 1.0,
    map_state_available: bool = False,
) -> RemoteSessionStatus:
    """构造无控制能力、可选 perception/map 观察能力的会话状态。"""

    if not config.remote.enabled:
        raise ValueError("observe_only perception transport requires remote.enabled.")
    if config.remote.role is not RemoteRole.SERVER:
        raise ValueError("observe_only perception transport requires server role.")
    if config.remote.access_mode is not RemoteAccessMode.OBSERVE_ONLY:
        raise ValueError("observe_only perception transport requires observe_only mode.")
    if not isinstance(map_state_available, bool):
        raise TypeError("map_state_available must be a boolean.")
    return RemoteSessionStatus(
        session_id=f"cluster-breakup-session-{uuid4()}",
        server_instance_id=server_instance_id,
        timestamp_ns=time.monotonic_ns(),
        access_mode=RemoteAccessMode.OBSERVE_ONLY,
        motion_control_available=False,
        gripper_control_available=False,
        capture_control_available=False,
        video_stream_available=True,
        video_modes=(VideoFrameMode.PERCEPTION,),
        map_state_available=map_state_available,
        vehicle_state_available=False,
        capture_status_available=False,
        target_heading_control_available=False,
        session_status_period_ms=1_000,
        vehicle_state_period_ms=None,
        map_state_period_ms=200 if map_state_available else None,
        capture_status_period_ms=None,
        video_nominal_fps=video_nominal_fps,
        max_linear_velocity_m_s=None,
        max_angular_velocity_rad_s=None,
        max_control_command_valid_for_ms=(
            config.motion.max_remote_command_valid_for_ms
        ),
    )


@dataclass(frozen=True, slots=True)
class BreakupDecision:
    timestamp_ns: int
    state: BreakupState
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    gripper_posture: GripperPosture
    reason: str


@dataclass(frozen=True, slots=True)
class TargetedClusterMeasurement:
    """一次带身份目标团的实时机器人地面测量。

    该类型只用于重复解团：上层以 ``FieldPoint`` 锁定目标团，再把本帧重新
    识别到的同一批成员转换为机器人地面测量。没有测量时，定向解团实例只
    搜索等待，绝不回退到画面中的其它目标团。
    """

    frame_sequence: int
    member_track_ids: tuple[int, ...]
    green_track_ids: tuple[int, ...]
    center_ground: GroundPoint
    nearest_forward_distance_mm: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer.")
        if len(self.member_track_ids) < 2 or len(set(self.member_track_ids)) != len(
            self.member_track_ids
        ):
            raise ValueError(
                "member_track_ids must contain at least two unique track IDs."
            )
        if any(
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id <= 0
            for track_id in self.member_track_ids
        ):
            raise ValueError("member_track_ids must contain positive integers.")
        if not self.green_track_ids or not set(self.green_track_ids).issubset(
            self.member_track_ids
        ):
            raise ValueError(
                "green_track_ids must be a non-empty subset of member_track_ids."
            )
        if not isinstance(self.center_ground, GroundPoint):
            raise ValueError("center_ground must be a GroundPoint.")
        if not (
            math.isfinite(self.center_ground.x)
            and math.isfinite(self.center_ground.y)
        ):
            raise ValueError("center_ground must be finite.")
        nearest = float(self.nearest_forward_distance_mm)
        if not math.isfinite(nearest) or nearest <= 0.0:
            raise ValueError(
                "nearest_forward_distance_mm must be finite and positive."
            )


@dataclass(frozen=True, slots=True)
class _ClusterView:
    horizontal_error_ratio: float
    raw_horizontal_error_ratio: float
    nearest_forward_distance_mm: float | None
    center_tolerance_ratio: float
    frame_sequence: int


class EncoderTravelTracker:
    """把连续双轮编码器计数转换为机器人中心累计有符号路程。"""

    def __init__(
        self,
        calibration: OdometryCalibration,
        *,
        max_wheel_velocity_m_s: float,
        max_consecutive_overrun_samples: int | None = 1,
    ) -> None:
        if not isinstance(calibration, OdometryCalibration):
            raise TypeError("calibration must be an OdometryCalibration.")
        maximum = float(max_wheel_velocity_m_s)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_wheel_velocity_m_s must be finite and positive.")
        if (
            max_consecutive_overrun_samples is not None
            and (
                isinstance(max_consecutive_overrun_samples, bool)
                or not isinstance(max_consecutive_overrun_samples, int)
                or max_consecutive_overrun_samples < 0
            )
        ):
            raise ValueError(
                "max_consecutive_overrun_samples must be a non-negative integer "
                f"or None, got {max_consecutive_overrun_samples!r}."
            )
        self._calibration = calibration
        self._maximum = maximum
        self._max_consecutive_overrun_samples = max_consecutive_overrun_samples
        self._previous: OdometryImu | None = None
        self._distance_m = 0.0
        self._left_distance_m = 0.0
        self._right_distance_m = 0.0
        self._consecutive_overrun_samples = 0

    @property
    def distance_m(self) -> float | None:
        return None if self._previous is None else self._distance_m

    @property
    def left_distance_m(self) -> float:
        return self._left_distance_m

    @property
    def right_distance_m(self) -> float:
        return self._right_distance_m

    @property
    def latest_encoder_counts(self) -> tuple[int, int] | None:
        if self._previous is None:
            return None
        return (
            self._previous.left_encoder_count,
            self._previous.right_encoder_count,
        )

    @property
    def consecutive_overrun_samples(self) -> int:
        return self._consecutive_overrun_samples

    def forward_sign_mismatch(self, *, minimum_wheel_travel_m: float = 0.02) -> bool:
        minimum = float(minimum_wheel_travel_m)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_wheel_travel_m must be finite and positive.")
        return (
            abs(self._left_distance_m) >= minimum
            and abs(self._right_distance_m) >= minimum
            and self._left_distance_m * self._right_distance_m < 0.0
        )

    def diagnostic(self) -> str:
        counts = self.latest_encoder_counts
        counts_text = "none" if counts is None else f"({counts[0]},{counts[1]})"
        return (
            f"center_distance_m={self._distance_m:.4f} "
            f"left_distance_m={self._left_distance_m:.4f} "
            f"right_distance_m={self._right_distance_m:.4f} "
            f"encoder_counts={counts_text} "
            f"consecutive_overrun_samples={self._consecutive_overrun_samples}"
        )

    def submit(self, message: OdometryImu) -> float:
        if not isinstance(message, OdometryImu):
            raise TypeError("message must be an OdometryImu.")
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
        if message.sensor_flags & required != required:
            raise RuntimeError("Both wheel encoders must be valid during breakup.")
        is_overrun = bool(message.sensor_flags & SensorFlags.SAMPLE_OVERRUN)
        if (
            is_overrun
            and self._max_consecutive_overrun_samples is not None
            and self._consecutive_overrun_samples
            >= self._max_consecutive_overrun_samples
        ):
            raise RuntimeError(
                "Consecutive odometry sample overruns exceed the configured "
                f"limit ({self._max_consecutive_overrun_samples}); "
                f"telemetry_sequence={message.telemetry_sequence}."
            )
        previous = self._previous
        if previous is None:
            self._previous = message
            self._consecutive_overrun_samples = 1 if is_overrun else 0
            return self._distance_m
        if message.sample_timestamp_us <= previous.sample_timestamp_us:
            raise RuntimeError("Odometry sample timestamps must increase.")
        left_delta = message.left_encoder_count - previous.left_encoder_count
        right_delta = message.right_encoder_count - previous.right_encoder_count
        scale = 2.0 * math.pi / self._calibration.encoder_counts_per_revolution
        left_m = left_delta * self._calibration.left_wheel_radius_mm * scale / 1000.0
        right_m = (
            right_delta * self._calibration.right_wheel_radius_mm * scale / 1000.0
        )
        dt_s = (
            message.sample_timestamp_us - previous.sample_timestamp_us
        ) / 1_000_000.0
        # 采样时间不受 UART 批量接收影响；保留原有量化余量和计数跳变检查。
        if max(abs(left_m), abs(right_m)) > self._maximum * dt_s * 2.0 + 0.005:
            raise RuntimeError("Encoder delta exceeds the configured wheel speed bound.")
        self._previous = message
        self._consecutive_overrun_samples = (
            self._consecutive_overrun_samples + 1 if is_overrun else 0
        )
        self._distance_m += 0.5 * (left_m + right_m)
        self._left_distance_m += left_m
        self._right_distance_m += right_m
        return self._distance_m


class _OdometryFusionSink(Protocol):
    def submit_odometry(self, message: OdometryImu) -> object: ...

    def latest_estimate(self, timestamp_ns: int) -> FusedPoseEstimate: ...


class OdometryFusionPump:
    """在有界旁路中顺序处理编码器+IMU融合，避免阻塞运动循环。"""

    def __init__(
        self,
        fusion: _OdometryFusionSink,
        *,
        queue_capacity: int = 256,
    ) -> None:
        if not callable(getattr(fusion, "submit_odometry", None)):
            raise TypeError("fusion must provide submit_odometry().")
        if not callable(getattr(fusion, "latest_estimate", None)):
            raise TypeError("fusion must provide latest_estimate().")
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer.")
        self._fusion = fusion
        self._queue_capacity = queue_capacity
        self._queue: Queue[OdometryImu] = Queue(maxsize=queue_capacity)
        self._stop_event = Event()
        self._error_lock = Lock()
        self._worker_error: BaseException | None = None
        self._thread: Thread | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError("OdometryFusionPump is already started.")
        self._stop_event.clear()
        self._queue = Queue(maxsize=self._queue_capacity)
        with self._error_lock:
            self._worker_error = None
        self._thread = Thread(
            target=self._worker_loop,
            name="rescue-cluster-breakup-odometry-fusion",
            daemon=True,
        )
        self._started = True
        try:
            self._thread.start()
        except BaseException:
            self._started = False
            self._thread = None
            raise

    def submit_odometry(self, message: OdometryImu) -> None:
        if not isinstance(message, OdometryImu):
            raise TypeError("message must be an OdometryImu.")
        self._require_started()
        self._raise_worker_error()
        try:
            self._queue.put_nowait(message)
        except Full as exc:
            error = RuntimeError(
                "Odometry fusion queue is full; localization side path is too slow."
            )
            with self._error_lock:
                if self._worker_error is None:
                    self._worker_error = error
            self._stop_event.set()
            raise error from exc

    def latest_estimate(self, timestamp_ns: int) -> FusedPoseEstimate:
        self._require_started()
        self._raise_worker_error()
        return self._fusion.latest_estimate(timestamp_ns)

    def pose_at(self, timestamp_ns: int) -> FusedPoseEstimate:
        """返回历史时间点的融合位姿，供相机观测做时间对齐。"""

        self._require_started()
        self._raise_worker_error()
        pose_at = getattr(self._fusion, "pose_at", None)
        if not callable(pose_at):
            raise RuntimeError("fusion does not provide pose_at().")
        estimate = pose_at(timestamp_ns)
        if not isinstance(estimate, FusedPoseEstimate):
            raise RuntimeError("fusion pose_at() returned an invalid estimate.")
        return estimate

    def inject_disturbance(
        self,
        position_uncertainty_mm: float,
        heading_uncertainty_rad: float,
    ) -> None:
        """转发一次扰动协方差注入；下层融合器不提供该方法是装配错误。"""

        self._require_started()
        self._raise_worker_error()
        inject = getattr(self._fusion, "inject_disturbance", None)
        if not callable(inject):
            raise RuntimeError("fusion does not provide inject_disturbance().")
        inject(
            position_uncertainty_mm=position_uncertainty_mm,
            heading_uncertainty_rad=heading_uncertainty_rad,
        )

    def visual_anchor_health(self) -> VisualAnchorHealth | None:
        """转发视觉锚健康快照；测试替身不提供该诊断时返回 None。"""

        self._require_started()
        self._raise_worker_error()
        read = getattr(self._fusion, "visual_anchor_health", None)
        if not callable(read):
            return None
        value = read()
        if not isinstance(value, VisualAnchorHealth):
            raise RuntimeError(
                "fusion visual_anchor_health() returned an invalid value."
            )
        return value

    @property
    def continuity_loss_reason(self) -> str | None:
        """返回融合器最近一次连续性清除原因；测试替身可以不提供该诊断。"""

        value = getattr(self._fusion, "continuity_loss_reason", None)
        return value if isinstance(value, str) and value else None

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 0.5,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        """等待首个有效融合状态，并允许调用方继续消费 UART。"""

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not 0.0 < float(timeout_s) < float("inf")
        ):
            raise ValueError("timeout_s must be finite and positive.")
        if on_wait is not None and not callable(on_wait):
            raise TypeError("on_wait must be callable or None.")
        self._require_started()
        deadline = time.monotonic() + float(timeout_s)
        while True:
            self._raise_worker_error()
            estimate = self._fusion.latest_estimate(time.monotonic_ns())
            if estimate.pose is not None:
                return
            if on_wait is not None:
                on_wait()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    "Odometry fusion did not produce an initial pose during "
                    "the startup synchronization window."
                )
            time.sleep(min(0.005, remaining))

    def check_health(self) -> None:
        self._require_started()
        self._raise_worker_error()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("OdometryFusionPump worker did not stop.")
        self._thread = None
        self._started = False
        self._raise_worker_error()

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    message = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                self._fusion.submit_odometry(message)
        except BaseException as error:
            with self._error_lock:
                self._worker_error = error
            self._stop_event.set()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("OdometryFusionPump is not started.")

    def _raise_worker_error(self) -> None:
        with self._error_lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Odometry fusion pump failed.") from error


def _submit_breakup_odometry(
    tracker: EncoderTravelTracker,
    fusion: _OdometryFusionSink | None,
    message: OdometryImu,
) -> object | None:
    """Distribute one UART odometry sample to distance and pose consumers."""

    tracker.submit(message)
    if fusion is None:
        return None
    return fusion.submit_odometry(message)


def _rotation_direction_name(angular_velocity_rad_s: float) -> str:
    """按带符号角速度返回旋转方向名；左转为正。"""

    return "left" if angular_velocity_rad_s > 0.0 else "right"


class ClusterBreakupSequence:
    """从出发区到解团并找到第一枚绿色目标的确定性流程。

    默认模式从当前 ``PerceptionSnapshot`` 按像素框寻找首次目标团。定向模式
    只接受调用方传入的 ``TargetedClusterMeasurement``，供上层按身份锁定的
    FieldPoint 目标团重新测量；测量缺失时不会回退到其它图像目标。
    """

    def __init__(
        self,
        config: ClusterBreakupRuntimeConfig,
        *,
        gripper_full_travel_time_s: float,
        skip_departure: bool = False,
        breakup_distance_m: float | None = None,
        targeted_cluster_mode: bool = False,
        targeted_center_tolerance_rad: float | None = None,
    ) -> None:
        if not isinstance(config, ClusterBreakupRuntimeConfig):
            raise TypeError("config must be a ClusterBreakupRuntimeConfig.")
        if not config.enabled:
            raise ValueError("ClusterBreakupSequence requires enabled config.")
        if (
            isinstance(gripper_full_travel_time_s, bool)
            or not isinstance(gripper_full_travel_time_s, (int, float))
            or not math.isfinite(float(gripper_full_travel_time_s))
            or float(gripper_full_travel_time_s) <= 0.0
        ):
            raise ValueError(
                "gripper_full_travel_time_s must be finite and positive."
            )
        if not isinstance(skip_departure, bool):
            raise TypeError("skip_departure must be a boolean.")
        if not isinstance(targeted_cluster_mode, bool):
            raise TypeError("targeted_cluster_mode must be a boolean.")
        if targeted_cluster_mode:
            if (
                targeted_center_tolerance_rad is None
                or isinstance(targeted_center_tolerance_rad, bool)
                or not isinstance(targeted_center_tolerance_rad, (int, float))
                or not math.isfinite(float(targeted_center_tolerance_rad))
                or not 0.0 < float(targeted_center_tolerance_rad) < math.pi
            ):
                raise ValueError(
                    "targeted_center_tolerance_rad must be finite and in (0, pi) "
                    "for targeted cluster mode."
                )
        elif targeted_center_tolerance_rad is not None:
            raise ValueError(
                "targeted_center_tolerance_rad is only valid in targeted cluster mode."
            )
        if breakup_distance_m is not None and (
            isinstance(breakup_distance_m, bool)
            or not isinstance(breakup_distance_m, (int, float))
            or not math.isfinite(float(breakup_distance_m))
            or float(breakup_distance_m) <= 0.0
        ):
            raise ValueError(
                "breakup_distance_m must be finite and positive when provided."
            )
        self.config = config
        self._gripper_full_travel_time_s = float(gripper_full_travel_time_s)
        self._skip_departure = skip_departure
        self._targeted_cluster_mode = targeted_cluster_mode
        self._targeted_center_tolerance_rad = targeted_center_tolerance_rad
        self._breakup_distance_m = (
            config.breakup_distance_m
            if breakup_distance_m is None
            else float(breakup_distance_m)
        )
        self.state = BreakupState.WAIT_ODOMETRY
        self._state_started_ns: int | None = None
        self._state_distance_m: float | None = None
        self._last_timestamp_ns: int | None = None
        self._last_cluster_seen_ns: int | None = None
        self._last_frame_sequence: int | None = None
        self._centered_frames = 0
        self._filtered_center_error_ratio: float | None = None
        self._last_center_filter_frame_sequence: int | None = None
        self._center_turn_sign: int | None = None
        self._center_reverse_candidate_sign: int | None = None
        self._center_reverse_candidate_frames = 0
        self._approach_travel_distance_m: float | None = None
        self._green_frames = 0

    @property
    def breakup_distance_m(self) -> float:
        """当前实例 BREAKUP_PUSH 使用的编码器定距，单位 m。"""

        return self._breakup_distance_m

    @property
    def approach_locked(self) -> bool:
        """目标团已完成初始对准并锁定接近定距。"""

        return (
            self.state is BreakupState.APPROACH_CLUSTER
            and self._approach_travel_distance_m is not None
        )

    @property
    def _search_direction_name(self) -> str:
        return _rotation_direction_name(
            self.config.search_angular_velocity_rad_s
        )

    def step(
        self,
        *,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        perception: PerceptionSnapshot | None,
        targeted_cluster: TargetedClusterMeasurement | None = None,
    ) -> BreakupDecision:
        self._validate_inputs(timestamp_ns, cumulative_distance_m, perception)
        if targeted_cluster is not None and not isinstance(
            targeted_cluster, TargetedClusterMeasurement
        ):
            raise TypeError(
                "targeted_cluster must be a TargetedClusterMeasurement or None."
            )
        if not self._targeted_cluster_mode and targeted_cluster is not None:
            raise ValueError(
                "targeted_cluster is only accepted by a targeted cluster sequence."
            )
        if self._state_started_ns is None:
            self._state_started_ns = timestamp_ns
        self._last_timestamp_ns = timestamp_ns

        if self.state is BreakupState.WAIT_ODOMETRY:
            if cumulative_distance_m is None:
                if self._elapsed_s(timestamp_ns) >= self.config.motion_phase_timeout_s:
                    return self._fault(timestamp_ns, "odometry_start_timeout")
                return self._decision(timestamp_ns, 0.0, 0.0, "waiting_odometry")
            if self._skip_departure:
                self._transition(
                    BreakupState.SEARCH_CLUSTER,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    self.config.search_angular_velocity_rad_s,
                    f"initial_position_complete_search_{self._search_direction_name}",
                )
            self._transition(BreakupState.LEAVE_START, timestamp_ns, cumulative_distance_m)
            return self._decision(
                timestamp_ns,
                self.config.departure_speed_m_s,
                0.0,
                "leave_start_fixed_distance",
            )

        if self.state is BreakupState.LEAVE_START:
            if self._motion_timed_out(timestamp_ns):
                return self._fault(timestamp_ns, "departure_timeout")
            if self._reached_distance(
                cumulative_distance_m,
                self.config.departure_distance_m,
            ):
                self._transition(
                    BreakupState.SEARCH_CLUSTER,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    self.config.search_angular_velocity_rad_s,
                    f"departure_complete_search_{self._search_direction_name}",
                )
            return self._decision(
                timestamp_ns,
                self.config.departure_speed_m_s,
                0.0,
                "leave_start_fixed_distance",
            )

        if self.state is BreakupState.SEARCH_CLUSTER:
            if self._elapsed_s(timestamp_ns) >= self.config.search_timeout_s:
                return self._fault(timestamp_ns, "cluster_search_timeout")
            cluster = self._active_cluster_view(
                perception,
                targeted_cluster,
                minimum_count=self.config.cluster_min_detections,
            )
            if cluster is not None:
                self._last_cluster_seen_ns = timestamp_ns
                self._transition(
                    BreakupState.CENTER_CLUSTER,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._centering_decision(timestamp_ns, cluster)
            return self._decision(
                timestamp_ns,
                0.0,
                self.config.search_angular_velocity_rad_s,
                f"search_cluster_{self._search_direction_name}",
            )

        if self.state is BreakupState.CENTER_CLUSTER:
            cluster = self._active_cluster_view(
                perception,
                targeted_cluster,
                minimum_count=1,
            )
            if cluster is None:
                if self._last_cluster_seen_ns is None:
                    self._last_cluster_seen_ns = timestamp_ns
                lost_ms = (timestamp_ns - self._last_cluster_seen_ns) / 1_000_000.0
                if lost_ms >= self.config.target_loss_timeout_ms:
                    return self._fault(timestamp_ns, "cluster_lost")
                return self._decision(timestamp_ns, 0.0, 0.0, "cluster_temporarily_lost")
            self._last_cluster_seen_ns = timestamp_ns
            if self._is_new_cluster_frame(cluster.frame_sequence):
                if (
                    abs(cluster.horizontal_error_ratio)
                    <= cluster.center_tolerance_ratio
                ):
                    self._centered_frames += 1
                else:
                    self._centered_frames = 0
            if self._centered_frames < self.config.center_confirm_frames:
                return self._centering_decision(timestamp_ns, cluster)
            self._transition(
                BreakupState.APPROACH_CLUSTER,
                timestamp_ns,
                cumulative_distance_m,
            )

        if self.state is BreakupState.APPROACH_CLUSTER:
            return self._approach_decision(
                timestamp_ns,
                cumulative_distance_m,
                perception,
                targeted_cluster,
            )

        if self.state is BreakupState.BREAKUP_PUSH:
            if self._motion_timed_out(timestamp_ns):
                return self._fault(timestamp_ns, "breakup_push_timeout")
            if self._reached_distance(
                cumulative_distance_m,
                self._breakup_distance_m,
            ):
                self._transition(
                    BreakupState.BREAKUP_RELEASE,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "breakup_complete_stop_and_open_gripper",
                )
            return self._decision(
                timestamp_ns,
                self.config.breakup_speed_m_s,
                0.0,
                "breakup_push_fixed_distance",
            )

        if self.state is BreakupState.BREAKUP_RELEASE:
            if self._motion_timed_out(timestamp_ns):
                return self._fault(timestamp_ns, "gripper_release_timeout")
            if self._elapsed_s(timestamp_ns) < self._gripper_full_travel_time_s:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "hold_open_gripper_before_open_retreat",
                )
            if not self.config.post_breakup_retreat_enabled:
                self._transition(
                    BreakupState.BREAKUP_CLOSE,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "open_gripper_in_place_close",
                )
            self._transition(
                BreakupState.BREAKUP_OPEN_RETREAT,
                timestamp_ns,
                cumulative_distance_m,
            )
            return self._decision(
                timestamp_ns,
                -self.config.retreat_speed_m_s,
                0.0,
                "gripper_open_and_retreat",
            )

        if self.state is BreakupState.BREAKUP_OPEN_RETREAT:
            if self._motion_timed_out(timestamp_ns):
                return self._fault(timestamp_ns, "open_gripper_retreat_timeout")
            if self._reached_distance(
                cumulative_distance_m,
                self.config.gripper_open_retreat_distance_m,
            ):
                self._transition(
                    BreakupState.BREAKUP_CLOSE,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "open_retreat_complete_stop_and_close_gripper",
                )
            return self._decision(
                timestamp_ns,
                -self.config.retreat_speed_m_s,
                0.0,
                "open_gripper_retreat_fixed_distance",
            )

        if self.state is BreakupState.BREAKUP_CLOSE:
            if self._motion_timed_out(timestamp_ns):
                return self._fault(timestamp_ns, "gripper_close_timeout")
            if self._elapsed_s(timestamp_ns) < self._gripper_full_travel_time_s:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "hold_closed_gripper_before_retreat",
                )
            if not self.config.post_breakup_retreat_enabled:
                self._transition(
                    BreakupState.SCAN_GREEN,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    self.config.scan_green_angular_velocity_rad_s,
                    "close_gripper_in_place_scan_green",
                )
            self._transition(
                BreakupState.RETREAT,
                timestamp_ns,
                cumulative_distance_m,
            )
            return self._decision(
                timestamp_ns,
                -self.config.retreat_speed_m_s,
                0.0,
                "closed_gripper_retreat",
            )

        if self.state is BreakupState.RETREAT:
            if self._motion_timed_out(timestamp_ns):
                return self._fault(timestamp_ns, "retreat_timeout")
            if self._reached_distance(
                cumulative_distance_m,
                self.config.retreat_distance_m,
            ):
                self._transition(BreakupState.SCAN_GREEN, timestamp_ns, cumulative_distance_m)
                return self._decision(
                    timestamp_ns,
                    0.0,
                    self.config.scan_green_angular_velocity_rad_s,
                    "retreat_complete_close_gripper_scan_green",
                )
            return self._decision(
                timestamp_ns,
                -self.config.retreat_speed_m_s,
                0.0,
                "retreat_with_gripper_closed",
            )

        if self.state is BreakupState.SCAN_GREEN:
            green_seen = self._green_seen(perception)
            if self._is_new_frame(perception):
                self._green_frames = self._green_frames + 1 if green_seen else 0
            if self._green_frames >= self.config.green_confirm_frames:
                self._transition(
                    BreakupState.GREEN_FOUND,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(timestamp_ns, 0.0, 0.0, "green_target_found")
            return self._decision(
                timestamp_ns,
                0.0,
                self.config.scan_green_angular_velocity_rad_s,
                f"scan_green_{_rotation_direction_name(self.config.scan_green_angular_velocity_rad_s)}",
            )

        return self._decision(timestamp_ns, 0.0, 0.0, self.state.value)

    def _approach_decision(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        perception: PerceptionSnapshot | None,
        targeted_cluster: TargetedClusterMeasurement | None,
    ) -> BreakupDecision:
        if self._motion_timed_out(timestamp_ns):
            return self._fault(timestamp_ns, "cluster_approach_timeout")

        # 初始对准只在 CENTER_CLUSTER 完成。进入接近后锁定当时的 K0 距离，
        # 用编码器完成剩余路程，避免再次用目标框横向误差修正或等待每一帧视觉。
        # 这样视觉处理出现短时延迟时仍保持直行，但目标距离必须先取得一次有效值。
        if self._approach_travel_distance_m is None:
            cluster = self._active_cluster_view(
                perception,
                targeted_cluster,
                minimum_count=1,
            )
            if cluster is None:
                if self._last_cluster_seen_ns is None:
                    self._last_cluster_seen_ns = timestamp_ns
                lost_ms = (timestamp_ns - self._last_cluster_seen_ns) / 1_000_000.0
                if lost_ms >= self.config.target_loss_timeout_ms:
                    return self._fault(timestamp_ns, "cluster_lost")
                return self._decision(timestamp_ns, 0.0, 0.0, "cluster_temporarily_lost")
            self._last_cluster_seen_ns = timestamp_ns
            if cluster.nearest_forward_distance_mm is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "cluster_distance_missing")
            approach_distance_mm = (
                cluster.nearest_forward_distance_mm
                - self.config.gripper_open_distance_mm
            )
            if approach_distance_mm <= 0.0:
                self._transition(
                    BreakupState.BREAKUP_PUSH,
                    timestamp_ns,
                    cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    self.config.breakup_speed_m_s,
                    0.0,
                    "closed_gripper_and_breakup_push",
                )
            self._approach_travel_distance_m = approach_distance_mm / 1000.0

        if self._state_distance_m is None:
            if cumulative_distance_m is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "approach_waiting_for_odometry",
                )
            self._state_distance_m = cumulative_distance_m
        if self._reached_distance(
            cumulative_distance_m,
            self._approach_travel_distance_m,
        ):
            self._transition(
                BreakupState.BREAKUP_PUSH,
                timestamp_ns,
                cumulative_distance_m,
            )
            return self._decision(
                timestamp_ns,
                self.config.breakup_speed_m_s,
                0.0,
                "approach_locked_distance_reached_breakup_push",
            )
        return self._decision(
            timestamp_ns,
            self.config.approach_speed_m_s,
            0.0,
            "approach_cluster_locked_heading",
        )

    def _decision(
        self,
        timestamp_ns: int,
        linear: float,
        angular: float,
        reason: str,
    ) -> BreakupDecision:
        posture = (
            GripperPosture.OPEN
            if self.state in {
                BreakupState.BREAKUP_RELEASE,
                BreakupState.BREAKUP_OPEN_RETREAT,
            }
            else GripperPosture.CLOSED
        )
        return BreakupDecision(timestamp_ns, self.state, linear, angular, posture, reason)

    def _centering_decision(
        self,
        timestamp_ns: int,
        cluster: _ClusterView,
    ) -> BreakupDecision:
        return self._decision(
            timestamp_ns,
            0.0,
            self._centering_angular(
                cluster.horizontal_error_ratio,
                cluster.raw_horizontal_error_ratio,
                tolerance_ratio=cluster.center_tolerance_ratio,
            ),
            "center_cluster",
        )

    def _centering_angular(
        self,
        horizontal_error_ratio: float,
        raw_horizontal_error_ratio: float | None = None,
        *,
        tolerance_ratio: float | None = None,
    ) -> float:
        # 图像右侧为正误差；机器人需右转，项目角速度右转为负。
        raw_error = (
            horizontal_error_ratio
            if raw_horizontal_error_ratio is None
            else raw_horizontal_error_ratio
        )
        tolerance = (
            self.config.center_tolerance_ratio
            if tolerance_ratio is None
            else float(tolerance_ratio)
        )
        reversal_limit = tolerance + self.config.center_reverse_deadband_ratio
        if abs(horizontal_error_ratio) <= tolerance and abs(raw_error) <= tolerance:
            self._center_reverse_candidate_sign = None
            self._center_reverse_candidate_frames = 0
            return 0.0
        direction_error = (
            raw_error if abs(raw_error) > reversal_limit else horizontal_error_ratio
        )
        if abs(direction_error) <= tolerance:
            self._center_reverse_candidate_sign = None
            self._center_reverse_candidate_frames = 0
            return 0.0
        requested_sign = -1 if direction_error > 0.0 else 1
        if self._center_turn_sign is None:
            self._center_turn_sign = requested_sign
        elif requested_sign != self._center_turn_sign:
            if abs(raw_error) <= reversal_limit:
                self._center_reverse_candidate_sign = None
                self._center_reverse_candidate_frames = 0
                return 0.0
            if self._center_reverse_candidate_sign == requested_sign:
                self._center_reverse_candidate_frames += 1
            else:
                self._center_reverse_candidate_sign = requested_sign
                self._center_reverse_candidate_frames = 1
            if (
                self._center_reverse_candidate_frames
                < self.config.center_reverse_confirm_frames
            ):
                return 0.0
            self._center_turn_sign = requested_sign
            self._center_reverse_candidate_sign = None
            self._center_reverse_candidate_frames = 0
        else:
            self._center_reverse_candidate_sign = None
            self._center_reverse_candidate_frames = 0
        control_error = horizontal_error_ratio
        if abs(control_error) <= tolerance and abs(raw_error) > tolerance:
            control_error = raw_error
        requested = -self.config.center_kp_rad_s * control_error
        maximum = self.config.center_max_angular_velocity_rad_s
        return min(max(requested, -maximum), maximum)

    def _cluster_view(
        self,
        perception: PerceptionSnapshot | None,
        *,
        minimum_count: int,
    ) -> _ClusterView | None:
        if perception is None or perception.dropped_stale_age_ms is not None:
            return None
        observations = perception.observations
        if len(observations) < minimum_count:
            return None
        width = observations[0].image_size[0]
        if any(item.image_size[0] != width for item in observations):
            raise ValueError("Perception observations must share one image size.")
        cluster = self._largest_observation_group(observations, width)
        if len(cluster) < minimum_count:
            return None
        x_min = min(item.box.x_min for item in cluster)
        x_max = max(item.box.x_max for item in cluster)
        center_u = 0.5 * (x_min + x_max)
        horizontal_error = (center_u - width * 0.5) / (width * 0.5)
        if self._last_center_filter_frame_sequence != perception.frame_sequence:
            alpha = self.config.center_error_filter_alpha
            if self._filtered_center_error_ratio is None:
                self._filtered_center_error_ratio = horizontal_error
            else:
                self._filtered_center_error_ratio += alpha * (
                    horizontal_error - self._filtered_center_error_ratio
                )
            self._last_center_filter_frame_sequence = perception.frame_sequence
        assert self._filtered_center_error_ratio is not None
        forward_distances = [
            item.ground_point.x
            for item in cluster
            if item.ground_point is not None and item.ground_point.x > 0.0
        ]
        return _ClusterView(
            horizontal_error_ratio=self._filtered_center_error_ratio,
            raw_horizontal_error_ratio=horizontal_error,
            nearest_forward_distance_mm=(
                min(forward_distances) if forward_distances else None
            ),
            center_tolerance_ratio=self.config.center_tolerance_ratio,
            frame_sequence=perception.frame_sequence,
        )

    def _active_cluster_view(
        self,
        perception: PerceptionSnapshot | None,
        targeted_cluster: TargetedClusterMeasurement | None,
        *,
        minimum_count: int,
    ) -> _ClusterView | None:
        if not self._targeted_cluster_mode:
            return self._cluster_view(perception, minimum_count=minimum_count)
        if targeted_cluster is None:
            return None
        assert self._targeted_center_tolerance_rad is not None
        # Targeted mode uses a robot-ground bearing. Positive GroundPoint.y is
        # left, while the existing centering controller expects positive error
        # on the right, so the sign is inverted here.
        raw_error = -math.atan2(
            targeted_cluster.center_ground.y,
            targeted_cluster.center_ground.x,
        )
        if self._last_center_filter_frame_sequence != targeted_cluster.frame_sequence:
            alpha = self.config.center_error_filter_alpha
            if self._filtered_center_error_ratio is None:
                self._filtered_center_error_ratio = raw_error
            else:
                self._filtered_center_error_ratio += alpha * (
                    raw_error - self._filtered_center_error_ratio
                )
            self._last_center_filter_frame_sequence = targeted_cluster.frame_sequence
        assert self._filtered_center_error_ratio is not None
        return _ClusterView(
            horizontal_error_ratio=self._filtered_center_error_ratio,
            raw_horizontal_error_ratio=raw_error,
            nearest_forward_distance_mm=(
                targeted_cluster.nearest_forward_distance_mm
            ),
            center_tolerance_ratio=self._targeted_center_tolerance_rad,
            frame_sequence=targeted_cluster.frame_sequence,
        )

    def _largest_observation_group(
        self,
        observations: Sequence[TargetObservation],
        width: int,
    ) -> tuple[TargetObservation, ...]:
        """按水平间隙把观测框分成目标团，返回成员最多的团。

        相邻框的水平间隙超过 ``cluster_group_gap_ratio * width`` 即分属不同团；
        成员数相同时取水平跨度更大的团。散落在目标团外的单个目标不会被计入
        联合框中心或最近前向距离。
        """

        gap_limit = self.config.cluster_group_gap_ratio * width
        groups: list[list[TargetObservation]] = []
        current: list[TargetObservation] = []
        current_max_x = 0.0
        for item in sorted(observations, key=lambda entry: entry.box.x_min):
            if current and item.box.x_min - current_max_x > gap_limit:
                groups.append(current)
                current = []
                current_max_x = 0.0
            current.append(item)
            current_max_x = max(current_max_x, item.box.x_max)
        if current:
            groups.append(current)
        return tuple(
            max(
                groups,
                key=lambda group: (
                    len(group),
                    max(entry.box.x_max for entry in group)
                    - min(entry.box.x_min for entry in group),
                ),
            )
        )

    @staticmethod
    def _green_seen(perception: PerceptionSnapshot | None) -> bool:
        return bool(
            perception is not None
            and perception.dropped_stale_age_ms is None
            and any(
                item.target_class is TargetClass.GREEN_SUPPLY
                for item in perception.observations
            )
        )

    def _is_new_frame(self, perception: PerceptionSnapshot | None) -> bool:
        if perception is None or perception.frame_sequence == self._last_frame_sequence:
            return False
        self._last_frame_sequence = perception.frame_sequence
        return True

    def _is_new_cluster_frame(self, frame_sequence: int) -> bool:
        if frame_sequence == self._last_frame_sequence:
            return False
        self._last_frame_sequence = frame_sequence
        return True

    def _transition(
        self,
        state: BreakupState,
        timestamp_ns: int,
        distance_m: float | None,
    ) -> None:
        self.state = state
        self._state_started_ns = timestamp_ns
        self._state_distance_m = distance_m
        if state is BreakupState.CENTER_CLUSTER:
            self._centered_frames = 0
            self._approach_travel_distance_m = None
        if state is BreakupState.SEARCH_CLUSTER:
            self._filtered_center_error_ratio = None
            self._last_center_filter_frame_sequence = None
            self._center_turn_sign = None
            self._center_reverse_candidate_sign = None
            self._center_reverse_candidate_frames = 0
            self._approach_travel_distance_m = None
        if state is BreakupState.BREAKUP_PUSH:
            self._approach_travel_distance_m = None
        if state is BreakupState.SCAN_GREEN:
            self._green_frames = 0

    def _fault(self, timestamp_ns: int, reason: str) -> BreakupDecision:
        self._transition(BreakupState.FAULT, timestamp_ns, self._state_distance_m)
        return self._decision(timestamp_ns, 0.0, 0.0, reason)

    def _elapsed_s(self, timestamp_ns: int) -> float:
        assert self._state_started_ns is not None
        return (timestamp_ns - self._state_started_ns) / 1_000_000_000.0

    def _motion_timed_out(self, timestamp_ns: int) -> bool:
        return self._elapsed_s(timestamp_ns) >= self.config.motion_phase_timeout_s

    def _distance_since(self, distance_m: float | None) -> float:
        if distance_m is None or self._state_distance_m is None:
            return 0.0
        return abs(distance_m - self._state_distance_m)

    def _reached_distance(
        self,
        distance_m: float | None,
        target_m: float,
    ) -> bool:
        return self._distance_since(distance_m) >= target_m - 1e-9

    def _validate_inputs(
        self,
        timestamp_ns: int,
        distance_m: float | None,
        perception: PerceptionSnapshot | None,
    ) -> None:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise ValueError("timestamp_ns must not move backwards.")
        if distance_m is not None and not math.isfinite(float(distance_m)):
            raise ValueError("cumulative_distance_m must be finite when present.")
        if perception is not None and not isinstance(perception, PerceptionSnapshot):
            raise TypeError("perception must be a PerceptionSnapshot or None.")


def _run_hardware(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    jpeg_quality: int = 80,
    observer_image_interval_s: float = 1.0,
) -> None:
    # Imports stay local so test collection never imports camera/Hailo hardware APIs.
    from rescue_vision.app.manual_capture import build_camera_pipeline

    config = load_runtime_config(config_path)
    breakup = config.motion.cluster_breakup
    if not breakup.enabled:
        raise RuntimeError("motion.cluster_breakup.enabled must be true.")
    if not supervised_stop_ready:
        raise RuntimeError(
            "A physical emergency stop and continuous supervision are required "
            "until the firmware watchdog has been verified."
        )
    if config.remote.enabled and (
        config.remote.role is not RemoteRole.SERVER
        or config.remote.access_mode is not RemoteAccessMode.OBSERVE_ONLY
    ):
        raise RuntimeError(
            "Enabled remote observation for cluster breakup requires a server "
            "configured with remote.access_mode=observe_only."
        )
    channel = config.uart.build_channel()
    assert channel is not None
    controller = config.motion.build_controller(channel)
    assert controller is not None
    calibration = config.motion.odometry.build_calibration()
    gripper = config.motion.gripper.build_calibration()
    assert calibration is not None and gripper is not None
    tracker = EncoderTravelTracker(
        calibration,
        max_wheel_velocity_m_s=config.motion.max_wheel_velocity_m_s,
        max_consecutive_overrun_samples=(
            config.motion.odometry.max_consecutive_overrun_samples
        ),
    )
    odometry_fusion = config.build_odometry_imu_fusion()
    fusion_pump = (
        None if odometry_fusion is None else OdometryFusionPump(odometry_fusion)
    )
    remote_team_color = (
        None
        if odometry_fusion is None
        else RemoteTeamColor(config.world.team_color.value)
    )
    sequence = ClusterBreakupSequence(
        breakup,
        gripper_full_travel_time_s=gripper.full_travel_time_s,
    )
    pipeline = build_camera_pipeline(config)
    visual_localization = config.build_visual_localization_pipeline(
        ground_projector=pipeline.ground_projector,
        fusion=odometry_fusion,
    )
    renderer = PerceptionFrameRenderer(
        lambda: config.build_target_pose_detector(
            ground_projector=pipeline.ground_projector
        )
    )
    camera_perception = CameraPerceptionPump(
        pipeline.source,
        pipeline.prepare,
        renderer,
    )
    remote_transport: RemotePerceptionTransport | None = None
    if config.remote.enabled:
        remote_server = config.remote.build_server()
        assert remote_server is not None
        remote_transport = RemotePerceptionTransport(
            remote_server,
            pipeline,
            config,
            jpeg_quality=jpeg_quality,
            min_publish_interval_s=observer_image_interval_s,
            map_team_color=remote_team_color,
        )
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_stop)
    last_state: BreakupState | None = None
    last_posture: GripperPosture | None = None
    latest_snapshot: PerceptionSnapshot | None = None
    camera_perception_started = False
    camera_start_thread: Thread | None = None
    next_progress_ns = 0
    latest_status: CarSystemStatus | None = None
    active_motion_since_ns: int | None = None

    def consume_odometry(message: OdometryImu) -> None:
        # This is the only current consumer of localization input in the
        # breakup flow. Visual corrections are deliberately not submitted.
        _submit_breakup_odometry(
            tracker,
            fusion_pump,
            message,
        )

    def handle_sync_message(message: object) -> None:
        nonlocal latest_status
        if isinstance(message, OdometryImu):
            consume_odometry(message)
        elif isinstance(message, CarSystemStatus):
            latest_status = message

    def service_uart_during_camera_startup() -> None:
        """在相机/Hailo 预热期间继续排空车端遥测。"""

        nonlocal latest_status
        controller.update(now_ns=time.monotonic_ns())
        for message in controller.drain_messages():
            if isinstance(message, OdometryImu):
                consume_odometry(message)
            elif isinstance(message, CarSystemStatus):
                latest_status = message
                if message.emergency_stop_latched:
                    raise RuntimeError(
                        "STM32 emergency stop is latched during camera startup."
                    )

    def synchronize_controller() -> None:
        controller.synchronize(
            timeout_s=config.motion.synchronization_timeout_s,
            on_message=handle_sync_message,
        )
        if fusion_pump is not None:
            fusion_pump.wait_until_ready(on_wait=service_uart_during_camera_startup)

    try:
        fusion_pump_started = False
        if fusion_pump is not None:
            fusion_pump.start()
            fusion_pump_started = True
        if remote_transport is not None:
            # 绑定监听和接受客户端都在旁路线程中，主流程不等待观察端。
            remote_transport.start()
        with channel:
            try:
                # Hailo/相机预热与运动通道同步并行；等待预热期间主线程仍排空
                # UART，避免 100 Hz 遥测在启动门禁处挤满接收队列。
                camera_start_thread = camera_perception.start_in_background()
                synchronize_controller()
                camera_perception.wait_until_started(
                    camera_start_thread,
                    on_wait=service_uart_during_camera_startup,
                )
                camera_perception_started = True
                controller.query_state()
                while not stop_requested:
                    if controller.needs_synchronization:
                        synchronize_controller()
                    now_ns = time.monotonic_ns()
                    controller.update(now_ns=now_ns)
                    for message in controller.drain_messages():
                        if isinstance(message, OdometryImu):
                            consume_odometry(message)
                        elif isinstance(message, CarCommandReply):
                            if message.result is CommandResult.SEQUENCE_OLD:
                                # The protocol adapter has already issued the
                                # SOFT_BRAKE resynchronization.  Wait for its
                                # accepted reply before asking for motion again.
                                continue
                            if message.result is not CommandResult.ACCEPTED:
                                raise RuntimeError(
                                    "STM32 rejected command "
                                    f"{message.command_type.name.lower()}: "
                                    f"{message.result.name.lower()}."
                                )
                        elif isinstance(message, CarSystemStatus):
                            latest_status = message
                            if message.emergency_stop_latched:
                                raise RuntimeError(
                                    "STM32 emergency stop is latched; power-cycle or "
                                    "use the firmware-defined physical reset procedure."
                                )
                            if (
                                not message.protocol_ready
                                or message.reply_queue_full
                                or message.tx_degraded
                            ):
                                raise RuntimeError(
                                    "STM32 UART health is degraded; "
                                    f"protocol_ready={message.protocol_ready}, "
                                    f"reply_queue_full={message.reply_queue_full}, "
                                    f"tx_degraded={message.tx_degraded}, "
                                    f"rx_degraded={message.rx_degraded}."
                                )
                    if remote_transport is not None:
                        remote_transport.check_health()
                        rendered = renderer.latest()
                        if rendered is not None:
                            remote_transport.submit(rendered)
                    camera_perception.check_health()
                    localization_estimate: FusedPoseEstimate | None = None
                    if fusion_pump is not None:
                        fusion_pump.check_health()
                        localization_estimate = fusion_pump.latest_estimate(now_ns)
                        if (
                            tracker.distance_m is not None
                            and localization_estimate.pose is None
                        ):
                            quality = ",".join(
                                sorted(
                                    item.value
                                    for item in localization_estimate.quality
                                )
                            )
                            raise RuntimeError(
                                "Encoder/IMU localization is unavailable after "
                                f"odometry became valid; quality={quality or 'unknown'}."
                            )
                    if remote_transport is not None:
                        remote_transport.submit_localization(
                            localization_estimate,
                            now_ns,
                        )
                    candidate = renderer.latest_snapshot()
                    if candidate is not None:
                        latest_snapshot = candidate
                        if (
                            visual_localization is not None
                            and candidate.field_features is not None
                        ):
                            visual_localization.submit(candidate.field_features)
                    if latest_snapshot is not None:
                        age_ms = (
                            now_ns - latest_snapshot.capture_timestamp_ns
                        ) / 1_000_000.0
                        if age_ms > config.processing.max_observation_age_ms:
                            latest_snapshot = None
                    decision = sequence.step(
                        timestamp_ns=now_ns,
                        cumulative_distance_m=tracker.distance_m,
                        perception=latest_snapshot,
                    )
                    motion_requested = (
                        decision.linear_velocity_m_s != 0.0
                        or decision.angular_velocity_rad_s != 0.0
                    )
                    if motion_requested:
                        if active_motion_since_ns is None:
                            active_motion_since_ns = now_ns
                    else:
                        active_motion_since_ns = None
                    if (
                        active_motion_since_ns is not None
                        and now_ns - active_motion_since_ns >= 750_000_000
                        and latest_status is not None
                        and now_ns - latest_status.received_timestamp_ns <= 500_000_000
                        and not latest_status.motor_output_enabled
                    ):
                        raise RuntimeError(
                            "STM32 motor output remained disabled after a motion "
                            "request; "
                            f"stop_reason={latest_status.stop_reason.name.lower()} "
                            f"watchdog_armed={latest_status.watchdog_armed} "
                            "last_motion_command_age_ms="
                            f"{latest_status.last_motion_command_age_ms}."
                        )
                    if (
                        decision.state is BreakupState.LEAVE_START
                        and tracker.forward_sign_mismatch()
                    ):
                        raise RuntimeError(
                            "Forward encoder signs disagree with the protocol; "
                            f"{tracker.diagnostic()}"
                        )
                    if decision.state is not last_state:
                        print(f"{decision.state.value}: {decision.reason}", flush=True)
                        last_state = decision.state
                    if now_ns >= next_progress_ns:
                        localization_text = "localization=disabled"
                        if localization_estimate is not None:
                            if localization_estimate.pose is None:
                                localization_text = "localization=unavailable"
                            else:
                                pose = localization_estimate.pose
                                quality = ",".join(
                                    sorted(
                                        item.value
                                        for item in localization_estimate.quality
                                    )
                                )
                                localization_text = (
                                    "localization=("
                                    f"x_mm={pose.position.x:.1f},"
                                    f"y_mm={pose.position.y:.1f},"
                                    f"heading_rad={pose.heading_rad:.3f},"
                                    f"quality={quality or 'fused'})"
                                )
                        status_text = (
                            "status=none"
                            if latest_status is None
                            else (
                                "status=("
                                f"motor_output={latest_status.motor_output_enabled},"
                                f"watchdog={latest_status.watchdog_armed},"
                                f"estop={latest_status.emergency_stop_latched},"
                                f"stop_reason={latest_status.stop_reason.name.lower()},"
                                "motion_age_ms="
                                f"{latest_status.last_motion_command_age_ms})"
                            )
                        )
                        print(
                            "progress "
                            f"state={decision.state.value} "
                            f"{tracker.diagnostic()} "
                            "target_wheel_m_s="
                            f"{controller.target_wheel_speeds_m_s} "
                            "commanded_wheel_m_s="
                            f"{controller.commanded_wheel_speeds_m_s} "
                            f"{localization_text} "
                            f"{status_text}",
                            flush=True,
                        )
                        next_progress_ns = now_ns + 1_000_000_000
                    if decision.gripper_posture is not last_posture:
                        if decision.gripper_posture is GripperPosture.OPEN:
                            angles = (
                                gripper.open_left_angle_deg,
                                gripper.open_right_angle_deg,
                            )
                        elif decision.gripper_posture is GripperPosture.TRANSPORT:
                            transport_angles = gripper.transport_angles_deg
                            if transport_angles is None:
                                raise RuntimeError(
                                    "Transport gripper posture is not configured."
                                )
                            angles = transport_angles
                        else:
                            angles = (
                                gripper.closed_left_angle_deg,
                                gripper.closed_right_angle_deg,
                            )
                        controller.set_gripper_angles(*angles)
                        last_posture = decision.gripper_posture
                    controller.drive_wheel_limited(
                        decision.linear_velocity_m_s,
                        decision.angular_velocity_rad_s,
                    )
                    time.sleep(0.005)
                    if decision.state in {BreakupState.GREEN_FOUND, BreakupState.FAULT}:
                        if decision.state is BreakupState.FAULT:
                            raise RuntimeError(
                                f"Breakup sequence fault: {decision.reason}; "
                                f"{tracker.diagnostic()}"
                            )
                        break
            finally:
                try:
                    controller.soft_brake()
                finally:
                    if camera_start_thread is not None or camera_perception_started:
                        camera_perception.stop()
    finally:
        try:
            if remote_transport is not None:
                remote_transport.stop()
        finally:
            try:
                if fusion_pump_started:
                    fusion_pump.stop()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-start cluster breakup and stop after SCAN_GREEN finds green."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm a physical emergency stop and continuous test supervision.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for the optional observe_only perception image.",
    )
    parser.add_argument(
        "--observer-image-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum interval between optional perception images.",
    )
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if not math.isfinite(args.observer_image_interval_seconds) or (
        args.observer_image_interval_seconds <= 0.0
    ):
        parser.error(
            "--observer-image-interval-seconds must be finite and positive"
        )
    _run_hardware(
        args.config.expanduser().resolve(),
        supervised_stop_ready=args.supervised_physical_stop_ready,
        jpeg_quality=args.jpeg_quality,
        observer_image_interval_s=args.observer_image_interval_seconds,
    )


if __name__ == "__main__":
    main()
