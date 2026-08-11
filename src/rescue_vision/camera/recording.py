"""将相机帧异步写入可回放记录目录。"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.exception_notes import add_exception_note


_STOP = object()
IMAGE_COORDINATE_SYSTEMS = {"raw_pixel", "undistorted_pixel"}
RECORDING_KINDS = {"camera", "supervised_manual_motion"}


class FrameRecorder:
    """将图像编码和写盘放到旁路线程；队列满时立即丢弃记录请求。"""

    def __init__(
        self,
        session_directory: str | Path,
        *,
        image_size: tuple[int, int],
        config_snapshot: Mapping[str, Any],
        session_tags: Mapping[str, str] | None = None,
        queue_capacity: int = 8,
        image_format: str = "png",
        image_coordinate_system: str = "raw_pixel",
        calibration_id: str | None = None,
        valid_pixel_ratio: float | None = None,
        undistort_fill_value: int | None = None,
        auxiliary_streams: Mapping[str, str] | None = None,
        recording_kind: str = "camera",
    ) -> None:
        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive.")
        if image_format not in {"png", "jpg"}:
            raise ValueError("image_format must be 'png' or 'jpg'.")
        if image_coordinate_system not in IMAGE_COORDINATE_SYSTEMS:
            raise ValueError(
                "image_coordinate_system must be 'raw_pixel' or "
                "'undistorted_pixel'."
            )
        if image_coordinate_system == "undistorted_pixel":
            if not isinstance(calibration_id, str) or not calibration_id.strip():
                raise ValueError(
                    "Undistorted recordings require a non-empty calibration_id."
                )
            calibration_id = calibration_id.strip()
            if (
                isinstance(valid_pixel_ratio, bool)
                or not isinstance(valid_pixel_ratio, (int, float))
                or not 0.0 < float(valid_pixel_ratio) <= 1.0
            ):
                raise ValueError(
                    "Undistorted recordings require valid_pixel_ratio in "
                    "(0, 1]."
                )
            valid_pixel_ratio = float(valid_pixel_ratio)
            if (
                isinstance(undistort_fill_value, bool)
                or not isinstance(undistort_fill_value, int)
                or not 0 <= undistort_fill_value <= 255
            ):
                raise ValueError(
                    "Undistorted recordings require integer "
                    "undistort_fill_value in [0, 255]."
                )
        elif (
            calibration_id is not None
            or valid_pixel_ratio is not None
            or undistort_fill_value is not None
        ):
            raise ValueError(
                "Raw recordings cannot declare intrinsics-derived metadata."
            )
        self.session_directory = Path(session_directory).expanduser().resolve()
        self.image_size = image_size
        self.config_snapshot = dict(config_snapshot)
        self.session_tags = dict(session_tags or {})
        if not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and value
            for key, value in self.session_tags.items()
        ):
            raise ValueError("session_tags must contain non-empty string pairs.")
        self.image_format = image_format
        self.image_coordinate_system = image_coordinate_system
        self.calibration_id = calibration_id
        self.valid_pixel_ratio = valid_pixel_ratio
        self.undistort_fill_value = undistort_fill_value
        self.auxiliary_streams = dict(auxiliary_streams or {})
        if not all(
            isinstance(name, str)
            and name
            and name.replace("_", "").isalnum()
            and isinstance(relative_path, str)
            and relative_path
            and Path(relative_path).name == relative_path
            for name, relative_path in self.auxiliary_streams.items()
        ):
            raise ValueError(
                "auxiliary_streams must map non-empty identifiers to relative "
                "filenames without directories."
            )
        if recording_kind not in RECORDING_KINDS:
            raise ValueError(
                "recording_kind must be 'camera' or "
                "'supervised_manual_motion'."
            )
        self.recording_kind = recording_kind
        if recording_kind == "camera" and self.auxiliary_streams:
            raise ValueError(
                "camera recording cannot declare auxiliary streams."
            )
        if recording_kind == "supervised_manual_motion" and (
            self.auxiliary_streams
            != {"manual_motion": "motion.jsonl"}
        ):
            raise ValueError(
                "supervised_manual_motion recording requires exactly the "
                "manual_motion auxiliary stream."
            )
        self._queue: queue.Queue[CameraFrame | object] = queue.Queue(queue_capacity)
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._started = False
        self.accepted_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0
        self._created_at: str | None = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Recorder is already started.")
        if self.session_directory.exists() and any(self.session_directory.iterdir()):
            raise FileExistsError(
                f"Recording directory is not empty: {self.session_directory}"
            )
        (self.session_directory / "frames").mkdir(parents=True, exist_ok=True)
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._write_session(completed=False)
        (self.session_directory / "annotations.json").write_text(
            json.dumps(
                {
                    "recording_id": self.session_directory.name,
                    "categories": [],
                    "items": [],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.session_directory / "frames.jsonl").touch()
        self._worker_error = None
        self._started = True
        self._thread = threading.Thread(
            target=self._worker,
            name="frame-recorder",
            daemon=True,
        )
        self._thread.start()

    def record(self, frame: CameraFrame) -> bool:
        """非阻塞提交；返回 False 表示记录队列已满。"""

        if not self._started:
            raise RuntimeError("Recorder is not started.")
        if self._worker_error is not None:
            raise RuntimeError("Recorder worker failed.") from self._worker_error
        actual_size = (int(frame.image_bgr.shape[1]), int(frame.image_bgr.shape[0]))
        if actual_size != self.image_size:
            raise ValueError(
                f"Frame image_size {actual_size} does not match recorder "
                f"{self.image_size}."
            )
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1
            return False
        self.accepted_frames += 1
        return True

    def stop(self) -> None:
        if not self._started:
            return
        assert self._thread is not None
        thread = self._thread
        if thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=1.0)
            except queue.Full:
                if self._worker_error is None:
                    self._worker_error = TimeoutError(
                        "Recorder stop sentinel could not be queued within 1.0 seconds."
                    )
        thread.join(timeout=5.0)
        if thread.is_alive() and self._worker_error is None:
            self._worker_error = TimeoutError(
                "Recorder worker did not stop within 5.0 seconds."
            )
        if not thread.is_alive():
            self._thread = None
        self._started = False
        self._write_session(
            completed=self._worker_error is None and self.written_frames > 0
        )
        if self._worker_error is not None:
            raise RuntimeError("Recorder worker failed.") from self._worker_error

    def _worker(self) -> None:
        try:
            manifest_path = self.session_directory / "frames.jsonl"
            with manifest_path.open("a", encoding="utf-8") as manifest:
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        return
                    assert isinstance(item, CameraFrame)
                    if self._worker_error is not None:
                        self.dropped_frames += 1
                        continue
                    extension = f".{self.image_format}"
                    parameters = (
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                        if self.image_format == "jpg"
                        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
                    )
                    ok, encoded = cv2.imencode(extension, item.image_bgr, parameters)
                    if not ok:
                        raise RuntimeError("OpenCV failed to encode a frame.")
                    payload = encoded.tobytes()
                    relative_path = Path("frames") / (
                        f"{item.sequence:08d}_{item.timestamp_ns}.{self.image_format}"
                    )
                    (self.session_directory / relative_path).write_bytes(payload)
                    record = {
                        "sequence": item.sequence,
                        "timestamp_ns": item.timestamp_ns,
                        "image_path": relative_path.as_posix(),
                        "metadata": dict(item.metadata),
                    }
                    manifest.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
                    manifest.flush()
                    self.written_frames += 1
        except BaseException as error:
            self._worker_error = error
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, CameraFrame):
                    self.dropped_frames += 1

    def _write_session(self, *, completed: bool) -> None:
        document = {
            "recording_id": self.session_directory.name,
            "recording_kind": self.recording_kind,
            "created_at": self._created_at,
            "completed_at": (
                datetime.now(timezone.utc).isoformat() if completed else None
            ),
            "completed": completed,
            "image_size": list(self.image_size),
            "image_format": self.image_format,
            "image_coordinate_system": self.image_coordinate_system,
            "calibration_id": self.calibration_id,
            "valid_pixel_ratio": self.valid_pixel_ratio,
            "undistort_fill_value": self.undistort_fill_value,
            "time_base": "application_monotonic_ns",
            "auxiliary_streams": {
                name: relative_path
                for name, relative_path in sorted(
                    self.auxiliary_streams.items()
                )
            },
            "config": self.config_snapshot,
            "tags": self.session_tags,
            "statistics": {
                "accepted_frames": self.accepted_frames,
                "written_frames": self.written_frames,
                "dropped_frames": self.dropped_frames,
            },
        }
        (self.session_directory / "session.json").write_text(
            json.dumps(
                document,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def __enter__(self) -> FrameRecorder:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.stop()
        except BaseException as cleanup_error:
            if isinstance(exc, BaseException):
                add_exception_note(
                    exc,
                    f"FrameRecorder cleanup also failed: {cleanup_error!r}",
                )
                return
            raise
