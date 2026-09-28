"""有界、非阻塞的 match 模型预标注采集旁路。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.localization.fusion import OdometryCalibration
from rescue_vision.motion.protocol import OdometryImu, SensorFlags
from rescue_vision.perception.detector import RealtimeDetectionResult


@dataclass(frozen=True, slots=True)
class MatchCaptureConfig:
    output_dir: Path | None = None
    frequency_hz: float = 1.0
    max_linear_speed_m_s: float = 0.10
    max_angular_speed_rad_s: float = 0.20
    max_telemetry_gap_ms: float = 200.0

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            raise ValueError(f"match_capture.output_dir must be Path or None, got {self.output_dir!r}")
        for name in ("frequency_hz", "max_linear_speed_m_s", "max_angular_speed_rad_s", "max_telemetry_gap_ms"):
            value = getattr(self, name)
            positive = name in ("frequency_hz", "max_telemetry_gap_ms")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f"match_capture.{name} must be finite and {'positive' if positive else 'non-negative'}, got {value!r}")


def capture_motion_allowed(
    timestamp_ns: int,
    samples: tuple[OdometryImu, ...],
    calibration: OdometryCalibration,
    config: MatchCaptureConfig,
) -> bool:
    """按主机单调时间夹住曝光起点，设备时间差计算实测车体速度。"""
    required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
    invalid = SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED
    for previous, current in zip(samples, samples[1:]):
        if not previous.received_timestamp_ns <= timestamp_ns < current.received_timestamp_ns:
            continue
        host_gap_ns = current.received_timestamp_ns - previous.received_timestamp_ns
        device_gap_ns = (current.sample_timestamp_us - previous.sample_timestamp_us) * 1000
        limit_ns = config.max_telemetry_gap_ms * 1_000_000
        if not (0 < host_gap_ns <= limit_ns and 0 < device_gap_ns <= limit_ns):
            return False
        if any(sample.sensor_flags & required != required or sample.sensor_flags & invalid
               for sample in (previous, current)):
            return False
        distance_per_count = 2 * math.pi / (1000 * calibration.encoder_counts_per_revolution)
        left = (current.left_encoder_count - previous.left_encoder_count) * calibration.left_wheel_radius_mm * distance_per_count
        right = (current.right_encoder_count - previous.right_encoder_count) * calibration.right_wheel_radius_mm * distance_per_count
        speed = abs((left + right) / 2 / (device_gap_ns / 1_000_000_000))
        angular = max(abs(previous.gyro_z_rad_s), abs(current.gyro_z_rad_s))
        return speed <= config.max_linear_speed_m_s and angular <= config.max_angular_speed_rad_s
    return False


def xanylabeling_document(
    frame: CameraFrame,
    result: RealtimeDetectionResult,
    image_name: str,
    keypoint_threshold: float,
) -> dict[str, object]:
    """只导出模型像素与分数，不使用 HSV、定位或 bbox 回退的关键点。"""
    shapes = []
    def shape(label: str, points: list[list[float]], kind: str, group: int, score: float) -> dict[str, object]:
        return dict(label=label, score=score, points=points, group_id=group,
                    description="model preannotation", difficult=False,
                    shape_type=kind, flags={}, attributes={}, kie_linking=[])

    for group, detection in enumerate(result.model_detections, start=1):
        box = detection.box
        shapes.append(shape(detection.model_class.value,
                            [[box.x_min, box.y_min], [box.x_max, box.y_max]],
                            "rectangle", group, detection.confidence))
        for name, keypoint in zip(("ground_anchor", "image_left_landmark", "image_right_landmark"), detection.keypoints):
            if keypoint.point is not None and keypoint.confidence > 0 and keypoint.confidence >= keypoint_threshold:
                shapes.append(shape(name, [[keypoint.point.u, keypoint.point.v]],
                                    "point", group, keypoint.confidence))
    return dict(version="4.0.4", flags={}, checked=False, shapes=shapes,
                imagePath=image_name, imageData=None,
                imageHeight=frame.image_bgr.shape[0], imageWidth=frame.image_bgr.shape[1])


class MatchDatasetCapture:
    """主循环仅追加遥测/提交引用；JPEG、JSON 和目录操作只在写线程执行。

    observe_motion 和 submit 由同一控制线程调用；图像由 CameraFrame 保持只读。
    两个待写请求加一个正在写入的请求限制内存，满队列直接丢弃。
    """

    def __init__(
        self,
        config: MatchCaptureConfig,
        calibration: OdometryCalibration,
        *,
        keypoint_threshold: float,
    ) -> None:
        if (isinstance(keypoint_threshold, bool)
                or not isinstance(keypoint_threshold, (int, float))
                or not math.isfinite(keypoint_threshold)
                or not 0 <= keypoint_threshold <= 1):
            raise ValueError(f"keypoint_threshold must be in [0, 1], got {keypoint_threshold!r}")
        self.config = config
        self.calibration = calibration
        self.keypoint_threshold = keypoint_threshold
        self.output_dir = config.output_dir or Path("data") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._samples: deque[OdometryImu] = deque(maxlen=512)
        self._queue: Queue[tuple[CameraFrame, RealtimeDetectionResult, tuple[OdometryImu, ...]]] = Queue(maxsize=2)
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_seen_ns = -1
        self._last_submitted_ns: int | None = None
        self.written = 0
        self.dropped = 0
        self.motion_skipped = 0
        self.worker_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("MatchDatasetCapture already started")
        self._thread = Thread(target=self._run, name="match-dataset-capture", daemon=True)
        try:
            self._thread.start()
        except Exception as exc:
            self.worker_error = repr(exc)
            self._thread = None

    def observe_motion(self, sample: OdometryImu) -> None:
        if self._samples and sample.received_timestamp_ns <= self._samples[-1].received_timestamp_ns:
            self._samples.clear()
        self._samples.append(sample)

    def submit(self, frame: CameraFrame, result: RealtimeDetectionResult) -> None:
        if self.worker_error is not None or self._stop.is_set() or self._thread is None:
            return
        if frame.timestamp_ns <= self._last_seen_ns:
            return
        self._last_seen_ns = frame.timestamp_ns
        if (result.stale_dropped or result.frame_sequence != frame.sequence
                or result.timing.capture_timestamp_ns != frame.timestamp_ns):
            return
        if (self._last_submitted_ns is not None and
                frame.timestamp_ns - self._last_submitted_ns < 1_000_000_000 / self.config.frequency_hz):
            return
        try:
            self._queue.put_nowait((frame, result, tuple(self._samples)))
        except Full:
            self.dropped += 1
        else:
            self._last_submitted_ns = frame.timestamp_ns

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self.worker_error = "capture writer did not stop within 2 s"

    def _run(self) -> None:
        directory_created = False
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    frame, result, samples = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                if not capture_motion_allowed(frame.timestamp_ns, samples, self.calibration, self.config):
                    self.motion_skipped += 1
                    continue
                if not directory_created:
                    self.output_dir.mkdir(parents=True, exist_ok=False)
                    directory_created = True
                stem = f"{frame.sequence:08d}_{frame.timestamp_ns}"
                image_path = self.output_dir / f"{stem}.jpg"
                label_path = self.output_dir / f"{stem}.json"
                document = xanylabeling_document(frame, result, image_path.name, self.keypoint_threshold)
                image_tmp = image_path.with_suffix(".jpg.tmp")
                label_tmp = label_path.with_suffix(".json.tmp")
                try:
                    success, encoded = cv2.imencode(".jpg", frame.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not success:
                        raise OSError(f"JPEG encoding failed: {image_path}")
                    image_tmp.write_bytes(encoded.tobytes())
                    label_tmp.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    image_tmp.replace(image_path)
                    label_tmp.replace(label_path)
                except Exception:
                    image_tmp.unlink(missing_ok=True)
                    label_tmp.unlink(missing_ok=True)
                    image_path.unlink(missing_ok=True)
                    raise
                self.written += 1
        except Exception as exc:
            self.worker_error = repr(exc)
        finally:
            # Release pending image references if writing failed.
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
