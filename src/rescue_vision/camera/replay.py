"""不依赖相机硬件的确定性帧源。"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame


class _FiniteSource:
    def __init__(self) -> None:
        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Frame source is not started.")

    @staticmethod
    def _validate_timeout(timeout: float) -> None:
        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}.")

    def stop(self) -> None:
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


class ImageDirectorySource(_FiniteSource):
    """按文件名稳定排序回放一个图片目录。"""

    SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(
        self,
        directory: str | Path,
        *,
        fps: float = 20.0,
        start_timestamp_ns: int = 0,
    ) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if start_timestamp_ns < 0:
            raise ValueError("start_timestamp_ns must be non-negative.")
        self.directory = Path(directory).expanduser().resolve()
        self.paths = sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUFFIXES
        )
        if not self.paths:
            raise ValueError(f"No supported images found in {self.directory}.")
        first = cv2.imread(str(self.paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"Cannot read image {self.paths[0]}.")
        self._image_size = (int(first.shape[1]), int(first.shape[0]))
        self.period_ns = round(1_000_000_000 / fps)
        self.start_timestamp_ns = start_timestamp_ns
        self._index = 0

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Frame source is already started.")
        self._index = 0
        self._started = True

    def read(self, timeout: float = 1.0) -> CameraFrame:
        self._validate_timeout(timeout)
        self._require_started()
        if self._index >= len(self.paths):
            raise EOFError("End of image directory.")
        path = self.paths[self._index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image {path}.")
        actual_size = (int(image.shape[1]), int(image.shape[0]))
        if actual_size != self._image_size:
            raise ValueError(
                f"Image {path.name} size {actual_size} differs from "
                f"{self._image_size}."
            )
        index = self._index
        self._index += 1
        return CameraFrame(
            sequence=index,
            timestamp_ns=self.start_timestamp_ns + index * self.period_ns,
            image_bgr=image,
            metadata={"source": "image_directory", "filename": path.name},
        )


class VideoFileSource(_FiniteSource):
    """按视频帧序和标称 FPS 确定性地产生时间戳。"""

    def __init__(
        self,
        path: str | Path,
        *,
        fps_override: float | None = None,
        start_timestamp_ns: int = 0,
    ) -> None:
        super().__init__()
        if fps_override is not None and fps_override <= 0:
            raise ValueError("fps_override must be positive.")
        if start_timestamp_ns < 0:
            raise ValueError("start_timestamp_ns must be non-negative.")
        self.path = Path(path).expanduser().resolve()
        self.fps_override = fps_override
        self.start_timestamp_ns = start_timestamp_ns
        self._capture: cv2.VideoCapture | None = None
        self._image_size: tuple[int, int] | None = None
        self._period_ns = 0
        self._index = 0

    @property
    def image_size(self) -> tuple[int, int]:
        if self._image_size is None:
            raise RuntimeError("Video source must be started before image_size is known.")
        return self._image_size

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Frame source is already started.")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"Cannot open video {self.path}.")
        fps = self.fps_override or float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise ValueError("Video FPS is missing; provide fps_override.")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Video reports an invalid frame size.")
        self._capture = capture
        self._image_size = (width, height)
        self._period_ns = round(1_000_000_000 / fps)
        self._index = 0
        self._started = True

    def read(self, timeout: float = 1.0) -> CameraFrame:
        self._validate_timeout(timeout)
        self._require_started()
        assert self._capture is not None
        ok, image = self._capture.read()
        if not ok:
            raise EOFError("End of video.")
        index = self._index
        self._index += 1
        return CameraFrame(
            sequence=index,
            timestamp_ns=self.start_timestamp_ns + index * self._period_ns,
            image_bgr=image,
            metadata={"source": "video", "filename": self.path.name},
        )

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        super().stop()


class RecordingSource(_FiniteSource):
    """严格按照记录清单中的顺序、序号和时间戳回放。"""

    def __init__(self, session_directory: str | Path) -> None:
        super().__init__()
        self.session_directory = Path(session_directory).expanduser().resolve()
        session = json.loads(
            (self.session_directory / "session.json").read_text(encoding="utf-8")
        )
        schema_version = session.get("schema_version")
        if schema_version not in {3, 4}:
            raise ValueError(
                "Recording session schema_version must be 3 or 4."
            )
        if schema_version == 4 and not isinstance(
            session.get("auxiliary_streams"), dict
        ):
            raise ValueError(
                "Recording session schema v4 auxiliary_streams must be a "
                "mapping."
            )
        if schema_version == 4 and session.get("recording_kind") not in {
            "camera",
            "supervised_manual_motion",
        }:
            raise ValueError(
                "Recording session schema v4 recording_kind is invalid."
            )
        image_size = session.get("image_size")
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise ValueError("Recording session image_size must be [width, height].")
        self._image_size = (int(image_size[0]), int(image_size[1]))
        coordinate_system = session.get("image_coordinate_system")
        if coordinate_system not in {"raw_pixel", "undistorted_pixel"}:
            raise ValueError(
                "Recording session image_coordinate_system must be "
                "'raw_pixel' or 'undistorted_pixel'."
            )
        fingerprint = session.get("intrinsics_fingerprint_sha256")
        if coordinate_system == "undistorted_pixel" and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or fingerprint != fingerprint.lower()
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint.lower()
            )
        ):
            raise ValueError(
                "Undistorted recording is missing its intrinsics fingerprint."
            )
        if coordinate_system == "raw_pixel" and fingerprint is not None:
            raise ValueError(
                "Raw recording must not declare an intrinsics fingerprint."
            )
        valid_pixel_ratio = session.get("valid_pixel_ratio")
        if coordinate_system == "undistorted_pixel" and (
            isinstance(valid_pixel_ratio, bool)
            or not isinstance(valid_pixel_ratio, (int, float))
            or not 0.0 < float(valid_pixel_ratio) <= 1.0
        ):
            raise ValueError(
                "Undistorted recording has invalid valid_pixel_ratio."
            )
        if coordinate_system == "raw_pixel" and valid_pixel_ratio is not None:
            raise ValueError(
                "Raw recording must not declare valid_pixel_ratio."
            )
        undistort_fill_value = session.get("undistort_fill_value")
        if coordinate_system == "undistorted_pixel" and (
            isinstance(undistort_fill_value, bool)
            or not isinstance(undistort_fill_value, int)
            or not 0 <= undistort_fill_value <= 255
        ):
            raise ValueError(
                "Undistorted recording has invalid undistort_fill_value."
            )
        if coordinate_system == "raw_pixel" and undistort_fill_value is not None:
            raise ValueError(
                "Raw recording must not declare undistort_fill_value."
            )
        self.image_coordinate_system = coordinate_system
        self.intrinsics_fingerprint_sha256 = fingerprint
        self.valid_pixel_ratio = (
            float(valid_pixel_ratio)
            if valid_pixel_ratio is not None
            else None
        )
        self.undistort_fill_value = undistort_fill_value
        self.records = [
            json.loads(line)
            for line in (self.session_directory / "frames.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self._index = 0

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Frame source is already started.")
        self._index = 0
        self._started = True

    def read(self, timeout: float = 1.0) -> CameraFrame:
        self._validate_timeout(timeout)
        self._require_started()
        if self._index >= len(self.records):
            raise EOFError("End of recording.")
        record = self.records[self._index]
        self._index += 1
        if record.get("schema_version") != 1:
            raise ValueError("Frame record schema_version must be 1.")
        image_path = self.session_directory / str(record["image_path"])
        payload = image_path.read_bytes()
        expected_hash = record.get("image_sha256")
        actual_hash = hashlib.sha256(payload).hexdigest()
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ValueError(
                f"Recorded frame hash mismatch for {image_path}; "
                "the recording is incomplete or corrupted."
            )
        image = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(f"Cannot read recorded frame {image_path}.")
        actual_size = (int(image.shape[1]), int(image.shape[0]))
        if actual_size != self._image_size:
            raise ValueError(
                f"Recorded frame size {actual_size} differs from "
                f"session {self._image_size}."
            )
        metadata = dict(record.get("metadata", {}))
        metadata["image_coordinate_system"] = self.image_coordinate_system
        if self.intrinsics_fingerprint_sha256 is not None:
            metadata["intrinsics_fingerprint_sha256"] = (
                self.intrinsics_fingerprint_sha256
            )
            metadata["valid_pixel_ratio"] = self.valid_pixel_ratio
            metadata["undistort_fill_value"] = self.undistort_fill_value
        return CameraFrame(
            sequence=int(record["sequence"]),
            timestamp_ns=int(record["timestamp_ns"]),
            image_bgr=image,
            metadata=metadata,
        )
