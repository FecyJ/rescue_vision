"""将相机帧异步写入可回放记录目录。"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from rescue_vision.camera.frame import CameraFrame


_STOP = object()


class FrameRecorder:
    """将图像编码和写盘放到旁路线程；队列满时立即丢弃记录请求。"""

    def __init__(
        self,
        session_directory: str | Path,
        *,
        image_size: tuple[int, int],
        config_snapshot: Mapping[str, Any],
        versions: Mapping[str, str],
        session_tags: Mapping[str, str] | None = None,
        queue_capacity: int = 8,
        image_format: str = "png",
    ) -> None:
        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive.")
        if image_format not in {"png", "jpg"}:
            raise ValueError("image_format must be 'png' or 'jpg'.")
        self.session_directory = Path(session_directory).expanduser().resolve()
        self.image_size = image_size
        self.config_snapshot = dict(config_snapshot)
        self.versions = dict(versions)
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
                    "schema_version": 1,
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
        self._queue.put(_STOP)
        assert self._thread is not None
        self._thread.join()
        self._thread = None
        self._started = False
        self._write_session(
            completed=self._worker_error is None and self.written_frames > 0
        )
        if self._worker_error is not None:
            raise RuntimeError("Recorder worker failed.") from self._worker_error

    def _worker(self) -> None:
        manifest_path = self.session_directory / "frames.jsonl"
        with manifest_path.open("a", encoding="utf-8") as manifest:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                assert isinstance(item, CameraFrame)
                if self._worker_error is not None:
                    continue
                try:
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
                        "schema_version": 1,
                        "sequence": item.sequence,
                        "timestamp_ns": item.timestamp_ns,
                        "image_path": relative_path.as_posix(),
                        "image_sha256": hashlib.sha256(payload).hexdigest(),
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

    def _write_session(self, *, completed: bool) -> None:
        document = {
            "schema_version": 1,
            "recording_id": self.session_directory.name,
            "created_at": self._created_at,
            "completed_at": (
                datetime.now(timezone.utc).isoformat() if completed else None
            ),
            "completed": completed,
            "image_size": list(self.image_size),
            "image_format": self.image_format,
            "time_base": "application_monotonic_ns",
            "config": self.config_snapshot,
            "versions": self.versions,
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
        self.stop()
