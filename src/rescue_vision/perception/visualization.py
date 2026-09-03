"""任务目标 perception 结果的叠加渲染和最新帧后台旁路。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.perception.detector import TargetPoseDetector
from rescue_vision.perception.field_feature_types import FieldFeatureDetectionResult, SafeZoneColor
from rescue_vision.perception.types import TargetClass, TargetObservation


_TARGET_COLORS: dict[TargetClass, tuple[int, int, int]] = {
    TargetClass.GREEN_SUPPLY: (0, 200, 0),
    TargetClass.BLACK_CORE: (80, 80, 80),
    TargetClass.ORANGE_INJURED: (0, 128, 255),
    TargetClass.BLUE_DANGER: (255, 200, 0),
    TargetClass.UNKNOWN: (255, 0, 255),
}
_K0_COLOR = (0, 0, 255)
_STALE_COLOR = (0, 0, 255)

# 推理旁路 worker 的耗时诊断打印间隔；只用于定位感知链路瓶颈。
_TIMING_REPORT_INTERVAL_NS = 2_000_000_000


@dataclass(frozen=True, slots=True)
class PerceptionSnapshot:
    """后台目标推理的最新结构化结果。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    observations: tuple[TargetObservation, ...]
    field_features: FieldFeatureDetectionResult | None
    dropped_stale_age_ms: float | None = None


def _target_color(target_class: TargetClass) -> tuple[int, int, int]:
    return _TARGET_COLORS.get(target_class, _TARGET_COLORS[TargetClass.UNKNOWN])


def _draw_text(
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        image_bgr,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_observation_mask(
    image_bgr: np.ndarray,
    observation: TargetObservation,
) -> None:
    roi_box = observation.color_segmentation.roi_box
    x_min = max(0, int(roi_box.x_min))
    y_min = max(0, int(roi_box.y_min))
    x_max = min(image_bgr.shape[1], int(roi_box.x_max))
    y_max = min(image_bgr.shape[0], int(roi_box.y_max))
    if x_max <= x_min or y_max <= y_min:
        return

    roi = image_bgr[y_min:y_max, x_min:x_max]
    mask = observation.color_segmentation.mask
    height = min(roi.shape[0], mask.shape[0])
    width = min(roi.shape[1], mask.shape[1])
    if height <= 0 or width <= 0:
        return
    selected = mask[:height, :width] != 0
    if not np.any(selected):
        return

    color = np.asarray(
        _target_color(observation.color_segmentation.candidate_class),
        dtype=np.float32,
    )
    selected_roi = roi[:height, :width]
    selected_roi[selected] = (
        selected_roi[selected].astype(np.float32) * 0.55 + color * 0.45
    ).astype(np.uint8)


def render_target_observations(
    image_bgr: np.ndarray,
    observations: Sequence[TargetObservation],
    *,
    dropped_stale_age_ms: float | None = None,
    field_features: FieldFeatureDetectionResult | None = None,
) -> np.ndarray:
    """在图像副本上叠加检测框、颜色掩码、K0 和质量信息。

    输入图像必须是 BGR、``uint8`` 且形状为 ``(height, width, 3)``。输出始终
    是独立的可写副本；输入图像和观测中的只读掩码不会被修改。观测使用
    去畸变图像坐标，输出保持同一像素坐标系。
    """

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise ValueError(
            "image_bgr must be a uint8 BGR array with shape "
            f"(height, width, 3), got dtype={getattr(image_bgr, 'dtype', None)}, "
            f"shape={getattr(image_bgr, 'shape', None)}."
        )
    if dropped_stale_age_ms is not None:
        if not np.isfinite(float(dropped_stale_age_ms)) or dropped_stale_age_ms < 0:
            raise ValueError(
                "dropped_stale_age_ms must be finite and non-negative when present."
            )

    preview = np.ascontiguousarray(image_bgr).copy()
    image_size = (int(preview.shape[1]), int(preview.shape[0]))
    for observation in observations:
        if not isinstance(observation, TargetObservation):
            raise TypeError(
                "observations must contain TargetObservation values, got "
                f"{type(observation).__name__}."
            )
        if observation.image_size != image_size:
            raise ValueError(
                f"Observation image_size {observation.image_size} does not match "
                f"image {image_size}."
            )
        _draw_observation_mask(preview, observation)

        box = observation.box
        color = _target_color(observation.target_class)
        start = (round(box.x_min), round(box.y_min))
        end = (round(box.x_max), round(box.y_max))
        cv2.rectangle(preview, start, end, color, 2)
        if observation.k0 is not None:
            cv2.circle(
                preview,
                (round(observation.k0.u), round(observation.k0.v)),
                5,
                _K0_COLOR,
                -1,
            )
        quality = ",".join(
            item.value for item in sorted(observation.quality, key=lambda item: item.value)
        )
        label = (
            f"{observation.target_class.value} "
            f"conf={observation.detection_confidence:.2f}"
        )
        if quality:
            label += f" quality={quality}"
        _draw_text(
            preview,
            label,
            (max(0, start[0]), max(20, start[1] - 8)),
            color,
        )

    if field_features is not None:
        if field_features.image_size != image_size:
            raise ValueError("field-feature image_size does not match image.")
        cross = field_features.center_cross
        if cross is not None:
            box = cross.box
            cv2.rectangle(
                preview,
                (round(box.x_min), round(box.y_min)),
                (round(box.x_max), round(box.y_max)),
                (0, 255, 255),
                2,
            )
            if cross.intersection.undistorted is not None:
                point = cross.intersection.undistorted
                cv2.drawMarker(
                    preview,
                    (round(point.u), round(point.v)),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    14,
                    2,
                )
            for axis in cross.axes:
                cv2.line(
                    preview,
                    (round(axis.start_undistorted.u), round(axis.start_undistorted.v)),
                    (round(axis.end_undistorted.u), round(axis.end_undistorted.v)),
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
        for zone in field_features.safe_zones:
            color = (
                (0, 0, 255)
                if zone.physical_color is SafeZoneColor.RED
                else (255, 0, 0)
                if zone.physical_color is SafeZoneColor.BLUE
                else (255, 0, 255)
            )
            box = zone.box
            cv2.rectangle(
                preview,
                (round(box.x_min), round(box.y_min)),
                (round(box.x_max), round(box.y_max)),
                color,
                2,
            )
            for index, keypoint in enumerate((zone.ground_anchor, zone.image_left_landmark, zone.image_right_landmark)):
                if keypoint.undistorted is None:
                    continue
                point = keypoint.undistorted
                cv2.circle(preview, (round(point.u), round(point.v)), 5, color, -1)
                _draw_text(preview, f"K{index}", (round(point.u) + 5, round(point.v) - 5), color)

    if dropped_stale_age_ms is not None:
        cv2.rectangle(
            preview,
            (0, 0),
            (preview.shape[1] - 1, preview.shape[0] - 1),
            _STALE_COLOR,
            1,
        )
        _draw_text(
            preview,
            f"perception: STALE dropped age={dropped_stale_age_ms:.1f} ms",
            (8, max(12, min(24, preview.shape[0] - 2))),
            _STALE_COLOR,
        )
    return preview


class PerceptionFrameRenderer:
    """使用最新帧旁路运行 perception，并保留最新可视化结果。

    构造函数只保存 detector factory。调用方必须显式 ``start()``；``start()``
    会在进入后台帧处理前完成 detector/Hailo 资源初始化，因此模型加载失败会
    在资源装配阶段暴露，不会在车辆开始运动后突然占用控制进程。``submit()``
    只保留最新待处理帧，不会让相机或运动安全循环等待推理。``stop()`` 会在
    后台线程结束后关闭 detector。
    """

    def __init__(
        self,
        detector_factory: Callable[[], TargetPoseDetector | None],
    ) -> None:
        if not callable(detector_factory):
            raise TypeError("detector_factory must be callable.")
        self._detector_factory = detector_factory
        self._condition = Event()
        self._render_condition = Event()
        self._lock = Lock()
        self._pending_frame: CameraFrame | None = None
        self._pending_render: tuple[CameraFrame, PerceptionSnapshot] | None = None
        self._latest_frame: CameraFrame | None = None
        self._latest_snapshot: PerceptionSnapshot | None = None
        self._worker_error: BaseException | None = None
        self._detector: TargetPoseDetector | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._render_thread: Thread | None = None
        self._started = False
        self._render_ms_total = 0.0
        self._render_ms_max = 0.0
        self._render_frame_count = 0

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError("PerceptionFrameRenderer is already started.")
        self._stop_event.clear()
        self._condition.clear()
        self._render_condition.clear()
        with self._lock:
            self._pending_frame = None
            self._pending_render = None
            self._latest_frame = None
            self._latest_snapshot = None
            self._worker_error = None
            self._detector = None
            self._render_ms_total = 0.0
            self._render_ms_max = 0.0
            self._render_frame_count = 0
        detector: TargetPoseDetector | None = None
        try:
            detector = self._detector_factory()
            if detector is None:
                raise RuntimeError(
                    "Perception video mode requires an enabled Hailo detector."
                )
            with self._lock:
                self._detector = detector
            self._thread = Thread(
                target=self._worker_loop,
                name="rescue-perception-video",
                daemon=True,
            )
            self._render_thread = Thread(
                target=self._render_worker_loop,
                name="rescue-perception-render",
                daemon=True,
            )
            self._started = True
            self._thread.start()
            self._render_thread.start()
        except BaseException as start_error:
            self._started = False
            self._thread = None
            if detector is not None:
                try:
                    detector.close()
                except BaseException as close_error:
                    start_error.add_note(
                        f"perception detector cleanup also failed: {close_error!r}"
                    )
            with self._lock:
                self._detector = None
            raise

    def submit(self, frame: CameraFrame) -> None:
        if not isinstance(frame, CameraFrame):
            raise TypeError("frame must be CameraFrame.")
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

    def latest_snapshot(self) -> PerceptionSnapshot | None:
        """返回最新完成推理的结构化观测；不等待可视化渲染。"""

        self._require_started()
        self._raise_worker_error()
        with self._lock:
            return self._latest_snapshot

    def clear_latest(self) -> None:
        """丢弃模式切换前的结果和待处理帧，不影响 detector 生命周期。"""

        self._require_started()
        self._raise_worker_error()
        with self._lock:
            self._latest_frame = None
            self._latest_snapshot = None
            self._pending_frame = None
            self._pending_render = None

    def check_health(self) -> None:
        self._require_started()
        self._raise_worker_error()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._condition.set()
        self._render_condition.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("PerceptionFrameRenderer worker did not stop.")
        render_thread = self._render_thread
        if render_thread is not None:
            render_thread.join(timeout=5.0)
        if render_thread is not None and render_thread.is_alive():
            raise RuntimeError("PerceptionFrameRenderer render worker did not stop.")
        self._thread = None
        self._render_thread = None
        self._started = False
        self._condition.clear()
        self._render_condition.clear()
        self._raise_worker_error()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("PerceptionFrameRenderer is not started.")

    def _raise_worker_error(self) -> None:
        error = self._worker_error
        if error is not None:
            raise RuntimeError("Perception video renderer failed.") from error

    def _worker_loop(self) -> None:
        with self._lock:
            detector = self._detector
        if detector is None:
            with self._lock:
                self._worker_error = RuntimeError(
                    "Perception video mode detector was not initialized."
                )
            return
        frame_count = 0
        detect_ms_total = 0.0
        detect_ms_max = 0.0
        age_ms_total = 0.0
        age_ms_max = 0.0
        stale_dropped_count = 0
        next_report_ns = monotonic_ns() + _TIMING_REPORT_INTERVAL_NS
        try:
            while not self._stop_event.is_set():
                self._condition.wait(timeout=0.05)
                self._condition.clear()
                while not self._stop_event.is_set():
                    with self._lock:
                        frame = self._pending_frame
                        self._pending_frame = None
                    if frame is None:
                        break
                    detect_start_ns = monotonic_ns()
                    result = detector.detect_realtime(
                        frame,
                        frame.image_bgr,
                        result_timestamp_ns=max(
                            monotonic_ns(),
                            frame.timestamp_ns,
                        ),
                    )
                    detect_ms = (monotonic_ns() - detect_start_ns) / 1_000_000.0
                    completed_ns = max(
                        (item.result_timestamp_ns for item in result.observations),
                        default=(
                            result.field_features.result_timestamp_ns
                            if result.field_features is not None
                            else monotonic_ns()
                        ),
                    )
                    snapshot = PerceptionSnapshot(
                        frame_sequence=frame.sequence,
                        capture_timestamp_ns=frame.timestamp_ns,
                        result_timestamp_ns=completed_ns,
                        observations=result.observations,
                        field_features=result.field_features,
                        dropped_stale_age_ms=result.dropped_stale_age_ms,
                    )
                    with self._lock:
                        self._latest_snapshot = snapshot
                        self._pending_render = (frame, snapshot)
                    self._render_condition.set()
                    frame_count += 1
                    detect_ms_total += detect_ms
                    detect_ms_max = max(detect_ms_max, detect_ms)
                    # Measure freshness at the point the control snapshot is
                    # published; rendering is an independent observer path.
                    age_ms = (monotonic_ns() - frame.timestamp_ns) / 1_000_000.0
                    age_ms_total += age_ms
                    age_ms_max = max(age_ms_max, age_ms)
                    if result.stale_dropped:
                        stale_dropped_count += 1
                    now_ns = monotonic_ns()
                    if now_ns >= next_report_ns:
                        with self._lock:
                            render_frame_count = self._render_frame_count
                            render_ms_total = self._render_ms_total
                            render_ms_max = self._render_ms_max
                            self._render_frame_count = 0
                            self._render_ms_total = 0.0
                            self._render_ms_max = 0.0
                        if frame_count:
                            print(
                                "perception_timing=(frames="
                                f"{frame_count},"
                                f"detect_ms_avg={detect_ms_total / frame_count:.1f},"
                                f"detect_ms_max={detect_ms_max:.1f},"
                                f"render_ms_avg={render_ms_total / max(1, render_frame_count):.1f},"
                                f"render_ms_max={render_ms_max:.1f},"
                                f"age_ms_avg={age_ms_total / frame_count:.1f},"
                                f"age_ms_max={age_ms_max:.1f},"
                                f"stale_dropped={stale_dropped_count})",
                                flush=True,
                            )
                        frame_count = 0
                        detect_ms_total = 0.0
                        detect_ms_max = 0.0
                        age_ms_total = 0.0
                        age_ms_max = 0.0
                        stale_dropped_count = 0
                        next_report_ns = now_ns + _TIMING_REPORT_INTERVAL_NS
        except BaseException as error:
            with self._lock:
                self._worker_error = error
            self._stop_event.set()
        finally:
            if detector is not None:
                try:
                    detector.close()
                except BaseException as error:
                    with self._lock:
                        if self._worker_error is None:
                            self._worker_error = error
                with self._lock:
                    if self._detector is detector:
                        self._detector = None

    def _render_worker_loop(self) -> None:
        """Render only the newest completed result on an observer thread."""

        try:
            while not self._stop_event.is_set():
                self._render_condition.wait(timeout=0.05)
                self._render_condition.clear()
                while not self._stop_event.is_set():
                    with self._lock:
                        pending = self._pending_render
                        self._pending_render = None
                    if pending is None:
                        break
                    frame, snapshot = pending
                    render_start_ns = monotonic_ns()
                    rendered = CameraFrame(
                        sequence=frame.sequence,
                        timestamp_ns=frame.timestamp_ns,
                        image_bgr=render_target_observations(
                            frame.image_bgr,
                            snapshot.observations,
                            dropped_stale_age_ms=snapshot.dropped_stale_age_ms,
                            field_features=snapshot.field_features,
                        ),
                        metadata={
                            "perception_stale_dropped": snapshot.dropped_stale_age_ms
                            is not None,
                        },
                    )
                    render_ms = (monotonic_ns() - render_start_ns) / 1_000_000.0
                    with self._lock:
                        self._latest_frame = rendered
                        self._render_frame_count += 1
                        self._render_ms_total += render_ms
                        self._render_ms_max = max(self._render_ms_max, render_ms)
        except BaseException as error:
            with self._lock:
                self._worker_error = error
            self._stop_event.set()
