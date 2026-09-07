"""最新帧本地预览及远程观察发布；不参与正式流程运动决策。"""
from __future__ import annotations
import math
from threading import Event, Lock, Thread
from typing import Protocol
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.viewer import OpenCvFrameViewer
from rescue_vision.localization import FusedPoseEstimate

class _MatchRemoteTransport(Protocol):
    def check_health(self) -> None: ...

    def submit_localization(
        self,
        estimate: FusedPoseEstimate | None,
        timestamp_ns: int,
    ) -> None: ...

    def submit(self, frame: object) -> None: ...


def _overlay_local_preview_status(
    frame: CameraFrame | None,
    *,
    state_text: str | None = None,
    reason_text: str | None = None,
    localization_lines: tuple[str, ...] | None = None,
    process_timestamp_ms: float | None = None,
) -> CameraFrame | None:
    """在本地预览帧上叠加状态和定位信息；只修改显示副本。"""

    if frame is None or (
        state_text is None
        and reason_text is None
        and not localization_lines
        and process_timestamp_ms is None
    ):
        return frame
    if process_timestamp_ms is not None:
        if (
            isinstance(process_timestamp_ms, bool)
            or not isinstance(process_timestamp_ms, (int, float))
            or not math.isfinite(float(process_timestamp_ms))
            or float(process_timestamp_ms) < 0.0
        ):
            raise ValueError(
                "process_timestamp_ms must be finite and non-negative when present."
            )
        process_timestamp_ms = float(process_timestamp_ms)
    import cv2

    image = frame.image_bgr.copy()
    image_width = int(image.shape[1])
    image_height = int(image.shape[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    margin = 12
    padding = 8
    half_panel_width = max(1, (image_width - 3 * margin) // 2)
    max_text_width = max(1, half_panel_width - 2 * padding)

    def fit_line(line: str) -> str:
        if cv2.getTextSize(line, font, scale, thickness)[0][0] <= max_text_width:
            return line
        suffix = "..."
        clipped = line
        while clipped and cv2.getTextSize(
            clipped + suffix, font, scale, thickness
        )[0][0] > max_text_width:
            clipped = clipped[:-1]
        return clipped + suffix if clipped else suffix


    def draw_panel(
        raw_lines: tuple[str, ...],
        *,
        right_aligned: bool,
        first_line_color: tuple[int, int, int],
    ) -> None:
        if not raw_lines:
            return
        lines = tuple(fit_line(line) for line in raw_lines)
        line_sizes = tuple(
            cv2.getTextSize(line, font, scale, thickness) for line in lines
        )
        line_height = max(size[0][1] for size in line_sizes)
        line_baseline = max(size[1] for size in line_sizes)
        line_step = line_height + line_baseline + 8
        panel_width = min(
            half_panel_width,
            max(size[0][0] for size in line_sizes) + 2 * padding,
        )
        panel_bottom = min(
            image_height,
            padding + len(lines) * line_step + padding,
        )
        panel_left = (
            image_width - margin - panel_width
            if right_aligned
            else margin
        )
        cv2.rectangle(
            image,
            (panel_left, 0),
            (panel_left + panel_width, max(0, panel_bottom)),
            (20, 20, 20),
            cv2.FILLED,
        )
        text_y = padding + line_height
        for index, (line, size) in enumerate(zip(lines, line_sizes, strict=True)):
            text_width = size[0][0]
            text_x = (
                panel_left + panel_width - padding - text_width
                if right_aligned
                else panel_left + padding
            )
            cv2.putText(
                image,
                line,
                (text_x, text_y),
                font,
                scale,
                first_line_color if index == 0 else (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_step

    draw_panel(
        tuple(
            line
            for line in (
                (
                    None
                    if process_timestamp_ms is None
                    else f"process_t={process_timestamp_ms:.1f}ms"
                ),
                None if state_text is None else f"state={state_text}",
                None if reason_text is None else f"reason={reason_text}",
            )
            if line is not None
        ),
        right_aligned=False,
        first_line_color=(0, 230, 255),
    )
    draw_panel(
        tuple(localization_lines or ()),
        right_aligned=True,
        first_line_color=(0, 230, 255),
    )
    return CameraFrame(
        sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        image_bgr=image,
        metadata=frame.metadata,
    )


class _LocalPreview:
    """在独立线程显示最新 perception 帧，不占用运动控制循环。"""

    def __init__(self, *, title: str = "Match perception") -> None:
        self._viewer = OpenCvFrameViewer(title)
        self._condition = Event()
        self._stop_event = Event()
        self._lock = Lock()
        self._pending_frame: tuple[
            CameraFrame,
            str | None,
            str | None,
            tuple[str, ...] | None,
            float | None,
        ] | None = None
        self._worker_error: BaseException | None = None
        self._thread: Thread | None = None
        self._user_requested_stop = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Local preview is already started.")
        self._stop_event.clear()
        self._condition.clear()
        with self._lock:
            self._pending_frame = None
            self._worker_error = None
            self._user_requested_stop = False
        self._thread = Thread(
            target=self._worker_loop,
            name="rescue-20-point-local-preview",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            raise

    def submit(
        self,
        frame: CameraFrame | None,
        *,
        state_text: str | None = None,
        reason_text: str | None = None,
        localization_lines: tuple[str, ...] | None = None,
        process_timestamp_ms: float | None = None,
    ) -> None:
        if frame is None:
            return
        if not isinstance(frame, CameraFrame):
            raise TypeError("local preview frame must be a CameraFrame or None.")
        if process_timestamp_ms is not None:
            if (
                isinstance(process_timestamp_ms, bool)
                or not isinstance(process_timestamp_ms, (int, float))
                or not math.isfinite(float(process_timestamp_ms))
                or float(process_timestamp_ms) < 0.0
            ):
                raise ValueError(
                    "process_timestamp_ms must be finite and non-negative when present."
                )
            process_timestamp_ms = float(process_timestamp_ms)
        if self._thread is None:
            raise RuntimeError("Local preview is not started.")
        self._raise_worker_error()
        with self._lock:
            self._pending_frame = (
                frame,
                state_text,
                reason_text,
                localization_lines,
                process_timestamp_ms,
            )
        self._condition.set()

    def check_health(self) -> None:
        if self._thread is None:
            raise RuntimeError("Local preview is not started.")
        self._raise_worker_error()

    @property
    def user_requested_stop(self) -> bool:
        with self._lock:
            return self._user_requested_stop

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        self._condition.set()
        thread.join(timeout=2.0)
        self._thread = None
        self._condition.clear()
        self._viewer.close()
        if thread.is_alive():
            raise RuntimeError("Local preview worker did not stop.")
        self._raise_worker_error()

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._condition.wait(timeout=0.1)
                self._condition.clear()
                with self._lock:
                    pending = self._pending_frame
                    self._pending_frame = None
                if pending is None:
                    continue
                (
                    frame,
                    state_text,
                    reason_text,
                    localization_lines,
                    process_timestamp_ms,
                ) = pending
                display_frame = _overlay_local_preview_status(
                    frame,
                    state_text=state_text,
                    reason_text=reason_text,
                    localization_lines=localization_lines,
                    process_timestamp_ms=process_timestamp_ms,
                )
                assert display_frame is not None
                if not self._viewer.show(display_frame.image_bgr):
                    with self._lock:
                        self._user_requested_stop = True
                    break
        except BaseException as exc:
            with self._lock:
                self._worker_error = exc
            self._stop_event.set()
        finally:
            self._viewer.close()

    def _raise_worker_error(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Local preview failed.") from error


def _publish_remote_match_state(
    remote_transport: _MatchRemoteTransport | None,
    *,
    pose: FusedPoseEstimate | None,
    timestamp_ns: int,
    rendered: object | None,
) -> None:
    """提交本周期最新定位和可选 perception 帧，不复制定位源。

    定位旁路暂不可用时提交 ``None``，观察端按 ``robot_localized=false``
    发布，不沿用过期坐标。
    """

    if remote_transport is None:
        return
    remote_transport.check_health()
    remote_transport.submit_localization(pose, timestamp_ns)
    if rendered is not None:
        remote_transport.submit(rendered)
