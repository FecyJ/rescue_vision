from __future__ import annotations

import importlib
from pathlib import Path
from threading import Event
from time import monotonic_ns

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.perception.types import ModelDetection, PoseKeypoint, UndistortedBoundingBox


@pytest.fixture
def preview_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'manual_tests'))
    return importlib.import_module('dual_camera_preview')


def frame(sequence: int) -> CameraFrame:
    return CameraFrame(sequence, monotonic_ns(), np.zeros((80, 120, 3), dtype=np.uint8),
                       {'image_coordinate_system': 'raw_pixel'})


def test_slow_inference_keeps_latest_per_camera_and_serves_both(preview_module) -> None:
    entered = Event()
    release = Event()
    completed = Event()
    calls = []
    def process(stream, sample):
        calls.append((stream, sample.sequence))
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        if len(calls) == 3:
            completed.set()
        return sample
    worker = preview_module.PreviewWorker(process)
    worker.start()
    try:
        worker.submit(0, frame(0))
        assert entered.wait(2)
        for seq in range(1, 30):
            worker.submit(0, frame(seq))
            worker.submit(1, frame(seq))
        assert len(worker.pending) == 2
        assert worker.latest() == {}
        release.set()
        assert completed.wait(2)
    finally:
        release.set()
        worker.stop()
    assert calls == [(0, 0), (1, 29), (0, 29)]
    assert {stream: sample.sequence for stream, sample in worker.latest().items()} == {0: 29, 1: 29}
    assert not worker.thread.is_alive()


def test_worker_reports_processing_failure(preview_module) -> None:
    failed = Event()
    def process(stream, sample):
        failed.set()
        raise ValueError('bad model tensor')
    worker = preview_module.PreviewWorker(process)
    worker.start()
    worker.submit(0, frame(0))
    assert failed.wait(2)
    worker.thread.join(2)
    with pytest.raises(RuntimeError, match='perception') as error:
        worker.latest()
    assert isinstance(error.value.__cause__, ValueError)
    worker.stop()


def test_near_preview_keeps_raw_identity_capture_time_and_input(preview_module) -> None:
    module = importlib.import_module('dual_camera_perception')
    sample = frame(7)
    detections = [ModelDetection(3, 0.9, UndistortedBoundingBox(20, 20, 70, 65),
                                (PoseKeypoint(UndistortedPixel(45, 60), 0.9),
                                 PoseKeypoint(None, 0), PoseKeypoint(None, 0)))]
    output = module.render_near_model(sample, detections, detection_threshold=0.5, k0_threshold=0.5)
    assert output.sequence == 7
    assert output.timestamp_ns == sample.timestamp_ns
    assert output.metadata['image_coordinate_system'] == 'raw_pixel'
    assert output.metadata['blue_danger_detections'] == 1
    assert output.metadata['result_timestamp_ns'] >= sample.timestamp_ns
    assert not sample.image_bgr.any()
    assert output.image_bgr.any()
