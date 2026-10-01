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
    # Existing fixtures use 4 px images: one PiSP Y row still occupies 128 bytes.
    assert width == 4
    return np.zeros(128 * height * 3 // 2, dtype=np.uint8).tobytes()


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


def test_select_fixed_focus_camera_without_lens_option(monkeypatch) -> None:
    commands = []
    def create(command, **kwargs):
        commands.append(command)
        return FakeProcess(fake_payload(4, 4))
    monkeypatch.setattr(source_module.subprocess, 'Popen', create)
    with RpicamSource(image_size=(4, 4), camera_index=1, lens_position=None) as camera:
        frame = camera.read()
        assert frame.metadata['camera_index'] == 1
        assert frame.metadata['image_coordinate_system'] == 'raw_pixel'
    assert commands[0][commands[0].index('--camera') + 1] == '1'
    assert '--lens-position' not in commands[0]


def test_rpicam_resolves_physical_port_before_starting_process(monkeypatch) -> None:
    commands = []
    resolved_ports = []
    def resolve(port):
        resolved_ports.append(port)
        return 1
    def create(command, **kwargs):
        commands.append(command)
        return FakeProcess(fake_payload(4, 4))
    monkeypatch.setattr(source_module, 'resolve_csi_camera_index', resolve)
    monkeypatch.setattr(source_module.subprocess, 'Popen', create)
    camera = RpicamSource(image_size=(4, 4), csi_port=0, lens_position=None)
    assert resolved_ports == []
    with camera:
        sample = camera.read(timeout=0.1)
        assert sample.metadata['csi_port'] == 0
        assert sample.metadata['camera_index'] == 1
    assert resolved_ports == [0]
    assert commands[0][commands[0].index('--camera') + 1] == '1'


@pytest.mark.parametrize(('width', 'stride'), [(1640, 1664), (800, 896), (2304, 2304)])
def test_pisp_yuv_padding_preserves_colors_and_next_frame(monkeypatch, width, stride) -> None:
    import cv2

    height = 8
    images = []
    buffers = []
    for colors in (((255, 0, 0), (0, 255, 0)), ((0, 0, 255), (255, 255, 255))):
        bgr = np.empty((height, width, 3), dtype=np.uint8)
        bgr[:, :width // 2] = colors[0]
        bgr[:, width // 2:] = colors[1]
        packed = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        images.append(cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_I420))
        flat = packed.ravel()
        y_size = width * height
        chroma_size = y_size // 4
        planes = [flat[:y_size].reshape(height, width),
                  flat[y_size:y_size+chroma_size].reshape(height // 2, width // 2),
                  flat[y_size+chroma_size:].reshape(height // 2, width // 2)]
        padded_planes = []
        for plane, row_stride in zip(planes, (stride, stride // 2, stride // 2)):
            padded = np.full((plane.shape[0], row_stride), 197, dtype=np.uint8)
            padded[:, :plane.shape[1]] = plane
            padded_planes.append(padded.tobytes())
        buffers.append(b''.join(padded_planes))
    process = FakeProcess(b''.join(buffers))
    monkeypatch.setattr(source_module.subprocess, 'Popen', lambda *a, **kw: process)
    with RpicamSource(image_size=(width, height), lens_position=None) as camera:
        with camera._condition:
            assert camera._condition.wait_for(lambda: camera._latest_sequence == 1, timeout=1)
        frame = camera.read(timeout=0.1)
        assert camera.frame_size == len(buffers[0])
        assert frame.sequence == 1
        assert frame.metadata['yuv_stride_bytes'] == stride
        assert frame.image_bgr.shape == (height, width, 3)
        np.testing.assert_array_equal(frame.image_bgr, images[1])
