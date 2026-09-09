from __future__ import annotations

import io
import threading

import numpy as np
import pytest

import rescue_vision.camera.rpicam_source as source_module
from rescue_vision.camera.rpicam_source import RpicamSource


class BlockingFrameStream:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)
        self.closed_event = threading.Event()

    def readinto(self, target) -> int:
        count = self._buffer.readinto(target)
        if count:
            return count
        self.closed_event.wait(timeout=2.0)
        return 0

    def close(self) -> None:
        self.closed_event.set()


class FakeProcess:
    def __init__(self, payload: bytes) -> None:
        self.stdout = BlockingFrameStream(payload)
        self.returncode = None

    def poll(self):
        return self.returncode

    def send_signal(self, _signal) -> None:
        self.returncode = 0
        self.stdout.closed_event.set()

    def wait(self, timeout=None) -> int:
        return int(self.returncode or 0)

    def terminate(self) -> None:
        self.send_signal(None)

    def kill(self) -> None:
        self.send_signal(None)


def fake_payload(width: int, height: int) -> bytes:
    return np.zeros(width * height * 3 // 2, dtype=np.uint8).tobytes()


def test_read_before_start_and_negative_timeout_are_rejected() -> None:
    camera = RpicamSource(image_size=(4, 4))
    with pytest.raises(RuntimeError, match="not started"):
        camera.read()
    with pytest.raises(ValueError, match="non-negative"):
        camera.read(timeout=-1)


def test_yuv420_rejects_odd_image_dimensions() -> None:
    with pytest.raises(ValueError, match="must be even"):
        RpicamSource(image_size=(5, 4))


def test_read_exact_uses_first_byte_arrival_time(monkeypatch) -> None:
    class ChunkedStream:
        def __init__(self) -> None:
            self.payload = bytearray(b"abcd")

        def readinto(self, target) -> int:
            if not self.payload:
                return 0
            count = min(2, len(self.payload))
            target[:count] = self.payload[:count]
            del self.payload[:count]
            return count

    timestamps = iter((100, 200))
    monkeypatch.setattr(source_module.time, "monotonic_ns", lambda: next(timestamps))
    result = RpicamSource._read_exact(ChunkedStream(), 4)

    assert result == (bytearray(b"abcd"), 100)


def test_process_start_failure_leaves_source_stopped(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise FileNotFoundError("rpicam-vid")

    monkeypatch.setattr(source_module.subprocess, "Popen", fail)
    camera = RpicamSource(image_size=(4, 4))
    with pytest.raises(FileNotFoundError):
        camera.start()
    camera.stop()
    with pytest.raises(RuntimeError, match="not started"):
        camera.read()


def test_repeated_start_stop_and_read_timeout(monkeypatch) -> None:
    processes: list[FakeProcess] = []

    def create(*args, **kwargs):
        process = FakeProcess(fake_payload(4, 4))
        processes.append(process)
        return process

    monkeypatch.setattr(source_module.subprocess, "Popen", create)
    camera = RpicamSource(image_size=(4, 4))
    camera.start()
    with pytest.raises(RuntimeError, match="already started"):
        camera.start()
    frame = camera.read(timeout=0.1)
    assert frame.sequence == 0
    assert isinstance(frame.timestamp_ns, int)
    assert frame.image_bgr.shape == (4, 4, 3)
    assert frame.metadata["timestamp_source"] == "host_frame_first_byte_monotonic"
    assert isinstance(frame.metadata["frame_received_timestamp_ns"], int)
    with pytest.raises(TimeoutError, match="相机帧超时"):
        camera.read(timeout=0.01)
    camera.stop()
    camera.stop()

    camera.start()
    assert camera.read(timeout=0.1).sequence == 0
    camera.stop()
    assert len(processes) == 2
