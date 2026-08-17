"""任务目标 perception 结果的叠加渲染和最新帧后台旁路。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Event, Lock, Thread
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.perception.detector import TargetPoseDetector
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

    构造函数只保存 detector factory，不创建 Hailo/推理资源。调用方必须显式
    ``start()``；第一次提交帧时才创建 detector。``submit()`` 只保留最新待处理
    帧，不会让相机或运动安全循环等待推理。``stop()`` 会在后台线程结束后
    关闭 detector。
    """

    def __init__(
        self,
        detector_factory: Callable[[], TargetPoseDetector | None],
    ) -> None:
        if not callable(detector_factory):
            raise TypeError("detector_factory must be callable.")
        self._detector_factory = detector_factory
        self._condition = Event()
        self._lock = Lock()
        self._pending_frame: CameraFrame | None = None
        self._latest_frame: CameraFrame | None = None
        self._worker_error: BaseException | None = None
        self._detector: TargetPoseDetector | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError("PerceptionFrameRenderer is already started.")
        self._stop_event.clear()
        self._condition.clear()
        with self._lock:
            self._pending_frame = None
            self._latest_frame = None
            self._worker_error = None
            self._detector = None
        self._thread = Thread(
            target=self._worker_loop,
            name="rescue-perception-video",
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

    def check_health(self) -> None:
        self._require_started()
        self._raise_worker_error()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        self._condition.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("PerceptionFrameRenderer worker did not stop.")
        self._thread = None
        self._started = False
        self._condition.clear()
        self._raise_worker_error()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("PerceptionFrameRenderer is not started.")

    def _raise_worker_error(self) -> None:
        error = self._worker_error
        if error is not None:
            raise RuntimeError("Perception video renderer failed.") from error

    def _worker_loop(self) -> None:
        detector: TargetPoseDetector | None = None
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
                    if detector is None:
                        detector = self._detector_factory()
                        if detector is None:
                            raise RuntimeError(
                                "Perception video mode requires an enabled "
                                "Hailo detector."
                            )
                        self._detector = detector
                    result = detector.detect_realtime(
                        frame,
                        frame.image_bgr,
                        result_timestamp_ns=max(
                            monotonic_ns(),
                            frame.timestamp_ns,
                        ),
                    )
                    rendered = CameraFrame(
                        sequence=frame.sequence,
                        timestamp_ns=frame.timestamp_ns,
                        image_bgr=render_target_observations(
                            frame.image_bgr,
                            result.observations,
                            dropped_stale_age_ms=result.dropped_stale_age_ms,
                        ),
                        metadata={
                            "perception_stale_dropped": result.stale_dropped,
                        },
                    )
                    with self._lock:
                        self._latest_frame = rendered
        except BaseException as error:
            with self._lock:
                self._worker_error = error
        finally:
            if detector is not None:
                try:
                    detector.close()
                except BaseException as error:
                    with self._lock:
                        if self._worker_error is None:
                            self._worker_error = error
