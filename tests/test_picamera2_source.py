from __future__ import annotations

import threading

import numpy as np
import pytest

import rescue_vision.camera.picamera2_source as source_module
from rescue_vision.camera.picamera2_source import Picamera2Source


class FakeRequest:
    def __init__(self) -> None:
        self.released = False

    def make_array(self, stream: str) -> np.ndarray:
        assert stream == "main"
        return np.full((6, 8, 3), 17, dtype=np.uint8)

    def get_metadata(self) -> dict:
        return {
            "SensorTimestamp": 123456789,
            "ExposureTime": 12000,
            "AnalogueGain": 1.5,
            "DigitalGain": 1.0,
            "ColourTemperature": 4500,
            "ColourGains": (1.2, 1.7),
            "LensPosition": 0.95,
            "FrameDuration": 50000,
        }

    def release(self) -> None:
        self.released = True


class FakeCamera:
    def __init__(self) -> None:
        self.configuration = None
        self.controls = None
        self.started = False
        self.closed = False
        self.stopped = threading.Event()
        self.requests = 0

    def create_video_configuration(self, **kwargs):
        self.configuration = kwargs
        return {"configuration": kwargs}

    def configure(self, configuration) -> None:
        assert "configuration" in configuration

    def set_controls(self, controls) -> None:
        self.controls = controls

    def start(self) -> None:
        self.started = True

    def capture_request(self) -> FakeRequest:
        if self.requests == 0:
            self.requests += 1
            return FakeRequest()
        self.stopped.wait(timeout=2.0)
        raise RuntimeError("camera stopped")

    def stop(self) -> None:
        self.stopped.set()

    def close(self) -> None:
        self.closed = True


def test_picamera2_source_returns_latest_frame_with_sensor_metadata() -> None:
    camera = FakeCamera()
    source = Picamera2Source(
        image_size=(8, 6),
        fps=20,
        lens_position=0.95,
        camera_factory=lambda: camera,
    )
    source.start()
    with pytest.raises(RuntimeError, match="already started"):
        source.start()
    frame = source.read(timeout=0.1)
    assert frame.sequence == 0
    assert frame.image_bgr.shape == (6, 8, 3)
    assert frame.image_bgr.flags.writeable is False
    assert frame.metadata == {
        "source": "picamera2",
        "configured_fps": 20,
        "configured_lens_position": 0.95,
        "sensor_timestamp_ns": 123456789,
        "exposure_time_us": 12000,
        "analogue_gain": 1.5,
        "digital_gain": 1.0,
        "colour_temperature_k": 4500,
        "lens_position": 0.95,
        "frame_duration_us": 50000,
        "colour_gain_red": 1.2,
        "colour_gain_blue": 1.7,
        "timestamp_source": "host_frame_received_monotonic",
        "frame_received_timestamp_ns": frame.metadata["frame_received_timestamp_ns"],
    }
    with pytest.raises(TimeoutError, match="Picamera2 帧超时"):
        source.read(timeout=0.01)
    source.stop()
    source.stop()
    assert camera.closed
    assert camera.configuration["main"] == {
        "size": (8, 6),
        "format": "RGB888",
    }
    assert camera.controls == {"LensPosition": 0.95}


def test_picamera2_source_rejects_read_before_start() -> None:
    source = Picamera2Source(
        image_size=(8, 6),
        camera_factory=FakeCamera,
    )
    with pytest.raises(RuntimeError, match="not started"):
        source.read()


def test_picamera2_maps_recent_sensor_timestamp_to_monotonic(monkeypatch) -> None:
    monkeypatch.setattr(source_module.time, "CLOCK_BOOTTIME", 7, raising=False)
    monkeypatch.setattr(source_module.time, "clock_gettime_ns", lambda _clock: 2_000_000_000)
    monkeypatch.setattr(source_module.time, "monotonic_ns", lambda: 1_000_000_000)

    timestamp, source = Picamera2Source._capture_timestamp(
        {"SensorTimestamp": 2_500_000_000},
        1_600_000_000,
    )

    assert timestamp == 1_500_000_000
    assert source == "sensor_start_of_frame_monotonic"


def test_picamera2_stop_still_closes_after_stop_timeout(monkeypatch) -> None:
    camera = FakeCamera()
    source = Picamera2Source(
        image_size=(8, 6),
        camera_factory=lambda: camera,
    )
    source._camera = camera
    source._running = True

    def bounded(action, *, timeout):
        if action.__name__ == "stop":
            return TimeoutError("stop timeout")
        action()
        return None

    monkeypatch.setattr(source, "_bounded_call", bounded)
    with pytest.raises(RuntimeError, match="shutdown"):
        source.stop()
    assert camera.closed
