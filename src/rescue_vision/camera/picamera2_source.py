"""带逐帧传感器元数据的 Picamera2 最新帧源。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from rescue_vision.camera.frame import CameraFrame, MetadataValue


class Picamera2Source:
    """后台持续获取 request，只向调用方交付最新完整帧。"""

    def __init__(
        self,
        image_size: tuple[int, int] = (2304, 1296),
        fps: int = 20,
        lens_position: float = 1.0,
        *,
        camera_factory: Callable[[], Any] | None = None,
    ) -> None:
        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if not np.isfinite(lens_position) or lens_position < 0:
            raise ValueError(
                f"lens_position must be finite and non-negative, got "
                f"{lens_position}."
            )
        self._image_size = image_size
        self.fps = fps
        self.lens_position = lens_position
        self._camera_factory = camera_factory
        self._camera: Any | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest_frame: CameraFrame | None = None
        self._delivered_sequence = -1
        self._reader_error: BaseException | None = None
        self._running = False

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def start(self) -> None:
        with self._condition:
            if self._running or self._camera is not None:
                raise RuntimeError("Camera source is already started.")
            self._latest_frame = None
            self._delivered_sequence = -1
            self._reader_error = None

        camera, manual_focus = self._create_camera()
        try:
            configuration = camera.create_video_configuration(
                main={"size": self.image_size, "format": "RGB888"},
                controls={"FrameRate": self.fps},
                buffer_count=4,
            )
            camera.configure(configuration)
            camera.set_controls(manual_focus)
            camera.start()
        except BaseException:
            camera.close()
            raise

        self._camera = camera
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="picamera2-reader",
            daemon=True,
        )
        self._thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._latest_frame is not None
                or self._reader_error is not None,
                timeout=5.0,
            )
        if not ready:
            self.stop()
            raise TimeoutError("等待 Picamera2 第一帧超时")
        if self._reader_error is not None:
            error = self._reader_error
            self.stop()
            raise RuntimeError("Picamera2 采集启动失败") from error

    def _create_camera(self) -> tuple[Any, dict[str, Any]]:
        if self._camera_factory is not None:
            return self._camera_factory(), {"LensPosition": self.lens_position}

        from libcamera import controls
        from picamera2 import Picamera2

        return Picamera2(), {
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": self.lens_position,
        }

    def read(self, timeout: float = 1.0) -> CameraFrame:
        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}.")
        with self._condition:
            if not self._running:
                raise RuntimeError("Camera source is not started.")
            ready = self._condition.wait_for(
                lambda: (
                    self._latest_frame is not None
                    and self._latest_frame.sequence > self._delivered_sequence
                )
                or self._reader_error is not None,
                timeout=timeout,
            )
            if not ready:
                raise TimeoutError("等待 Picamera2 帧超时")
            if self._reader_error is not None:
                raise RuntimeError("Picamera2 采集进程异常退出") from self._reader_error
            frame = self._latest_frame
            assert frame is not None
            self._delivered_sequence = frame.sequence
            return frame

    def stop(self) -> None:
        camera = self._camera
        if camera is None:
            self._running = False
            return
        self._running = False
        stop_error = self._bounded_call(camera.stop, timeout=3.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive() and stop_error is None:
                stop_error = TimeoutError("Picamera2 reader thread did not stop.")
        close_error = (
            self._bounded_call(camera.close, timeout=2.0)
            if not isinstance(stop_error, TimeoutError)
            else None
        )
        self._thread = None
        self._camera = None
        resource_error = stop_error or close_error
        if resource_error is not None:
            raise RuntimeError("Picamera2 resource shutdown failed.") from resource_error

    @staticmethod
    def _bounded_call(
        action: Callable[[], Any],
        *,
        timeout: float,
    ) -> BaseException | None:
        errors: list[BaseException] = []
        completed = threading.Event()

        def run() -> None:
            try:
                action()
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        if not completed.wait(timeout):
            return TimeoutError(f"Camera operation exceeded {timeout:.1f} seconds.")
        return errors[0] if errors else None

    def _reader_loop(self) -> None:
        assert self._camera is not None
        sequence = 0
        try:
            while self._running:
                request = self._camera.capture_request()
                try:
                    image_bgr = np.asarray(
                        request.make_array("main"),
                        dtype=np.uint8,
                    ).copy()
                    raw_metadata = request.get_metadata()
                    timestamp_ns = time.monotonic_ns()
                finally:
                    request.release()
                frame = CameraFrame(
                    sequence=sequence,
                    timestamp_ns=timestamp_ns,
                    image_bgr=image_bgr,
                    metadata=self._normalize_metadata(raw_metadata),
                )
                with self._condition:
                    self._latest_frame = frame
                    self._condition.notify_all()
                sequence += 1
        except BaseException as error:
            with self._condition:
                if self._running:
                    self._reader_error = error
                self._condition.notify_all()

    def _normalize_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> dict[str, MetadataValue]:
        result: dict[str, MetadataValue] = {
            "source": "picamera2",
            "configured_fps": self.fps,
            "configured_lens_position": self.lens_position,
        }
        scalar_fields = {
            "SensorTimestamp": "sensor_timestamp_ns",
            "ExposureTime": "exposure_time_us",
            "AnalogueGain": "analogue_gain",
            "DigitalGain": "digital_gain",
            "ColourTemperature": "colour_temperature_k",
            "LensPosition": "lens_position",
            "FrameDuration": "frame_duration_us",
        }
        for source_name, output_name in scalar_fields.items():
            value = metadata.get(source_name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[output_name] = value
        colour_gains = metadata.get("ColourGains")
        if (
            isinstance(colour_gains, (tuple, list))
            and len(colour_gains) == 2
        ):
            result["colour_gain_red"] = float(colour_gains[0])
            result["colour_gain_blue"] = float(colour_gains[1])
        return result

    def __enter__(self) -> Picamera2Source:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
