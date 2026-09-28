from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from threading import Event
import time

import cv2
import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.data.match_capture import (
    MatchCaptureConfig, MatchDatasetCapture, capture_motion_allowed, xanylabeling_document,
)
from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.localization.fusion import OdometryCalibration
from rescue_vision.motion.protocol import OdometryImu, SensorFlags
from rescue_vision.perception.detector import RealtimeDetectionResult
from rescue_vision.perception.timing import PerceptionTiming
from rescue_vision.perception.types import ModelDetection, PoseKeypoint, UndistortedBoundingBox

CALIBRATION = OdometryCalibration(1000, 50.0, 50.0)
VALID = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID


def odometry(ms: int, *, count: int = 0, angular: float = 0.0) -> OdometryImu:
    return OdometryImu(
        uart_sequence=ms, received_timestamp_ns=ms * 1_000_000,
        telemetry_sequence=ms, sample_timestamp_us=ms * 1000,
        left_encoder_count=count, right_encoder_count=count,
        gyro_x_urad_s=0, gyro_y_urad_s=0, gyro_z_urad_s=round(angular * 1_000_000),
        accel_x_mm_s2=0, accel_y_mm_s2=0, accel_z_mm_s2=9800,
        imu_temperature_cdeg=2500, sensor_flags=VALID,
    )


def sample(ms: int, sequence: int = 1, delay_ms: int = 400):
    timestamp = ms * 1_000_000
    frame = CameraFrame(sequence, timestamp, np.full((24, 32, 3), 80, dtype=np.uint8))
    timing = PerceptionTiming(timestamp, timestamp, timestamp,
                              timestamp + delay_ms * 1_000_000, timestamp + delay_ms * 1_000_000)
    missing = PoseKeypoint(None, 0.0)
    detections = tuple(ModelDetection(
        class_id, 0.9, UndistortedBoundingBox(1.0, 2.0, 25.0, 20.0),
        (PoseKeypoint(UndistortedPixel(10.0, 12.0), 0.8),
         PoseKeypoint(UndistortedPixel(4.0, 15.0), 0.7) if class_id == 5 else missing,
         PoseKeypoint(UndistortedPixel(22.0, 15.0), 0.1) if class_id == 5 else missing),
    ) for class_id in range(6))
    return frame, RealtimeDetectionResult(sequence, (), None, timing, model_detections=detections)


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_xanylabeling_all_six_classes_and_model_keypoints():
    frame, result = sample(1000)
    document = xanylabeling_document(frame, result, 'one.jpg', 0.5)
    assert document['imagePath'] == 'one.jpg'
    assert (document['imageWidth'], document['imageHeight']) == (32, 24)
    assert document['imageData'] is None and document['checked'] is False
    shapes = document['shapes']
    assert [s['label'] for s in shapes if s['shape_type'] == 'rectangle'] == [
        'green_supply', 'black_core', 'orange_injured', 'blue_danger', 'center_cross', 'safe_zone']
    assert len([s for s in shapes if s['label'] == 'ground_anchor']) == 6
    zone = [s for s in shapes if s['group_id'] == 6]
    assert [s['label'] for s in zone] == ['safe_zone', 'ground_anchor', 'image_left_landmark']
    assert zone[-1]['points'] == [[4.0, 15.0]]
    assert zone[-1]['score'] == 0.7


@pytest.mark.parametrize('count,angular,allowed', [(0, 0, True), (10, 0, True),
    (100, 0, False), (-100, 0, False), (0, 0.2, True), (0, 0.21, False), (0, -0.21, False)])
def test_speed_gates(count, angular, allowed):
    samples = (odometry(990), odometry(1090, count=count, angular=angular))
    assert capture_motion_allowed(1_000_000_000, samples, CALIBRATION, MatchCaptureConfig()) is allowed


@pytest.mark.parametrize('change', [dict(sensor_flags=SensorFlags(0)),
    dict(sensor_flags=VALID | SensorFlags.GYRO_SATURATED),
    dict(sensor_flags=VALID | SensorFlags.SAMPLE_OVERRUN),
    dict(sample_timestamp_us=990_000), dict(sample_timestamp_us=900_000),
    dict(sample_timestamp_us=1_500_000), dict(received_timestamp_ns=1_500_000_000)])
def test_missing_invalid_or_discontinuous_telemetry_skips(change):
    config = MatchCaptureConfig()
    current = replace(odometry(1090), **change)
    assert not capture_motion_allowed(1_000_000_000, (odometry(990), current), CALIBRATION, config)
    assert not capture_motion_allowed(1_000_000_000, (), CALIBRATION, config)
    assert not capture_motion_allowed(1_500_000_000, (odometry(990), odometry(1090)), CALIBRATION, config)


def test_delayed_result_uses_capture_motion_not_current_stop(tmp_path):
    collector = MatchDatasetCapture(MatchCaptureConfig(output_dir=tmp_path/'set', frequency_hz=10),
                                    CALIBRATION, keypoint_threshold=0.5)
    collector.start()
    try:
        for item in (odometry(990), odometry(1090, count=100),
                     odometry(1390, count=100), odometry(1490, count=100)):
            collector.observe_motion(item)
        collector.submit(*sample(1000))
        wait_until(lambda: collector.motion_skipped == 1)
        collector.submit(*sample(1400, 2))
        wait_until(lambda: collector.written == 1)
    finally:
        collector.stop()
    assert collector.worker_error is None
    files = list((tmp_path/'set').iterdir())
    assert len(files) == 2
    label = next(p for p in files if p.suffix == '.json')
    document = json.loads(label.read_text())
    image = label.with_suffix('.jpg')
    assert document['imagePath'] == image.name
    assert image.stem == '00000002_1400000000'
    np.testing.assert_array_equal(cv2.imread(str(image)), sample(1400, 2)[0].image_bgr)


def test_slow_writer_bounded_queue_and_fast_control_polling(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = cv2.imencode
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'imencode', blocked)
    collector = MatchDatasetCapture(MatchCaptureConfig(output_dir=tmp_path/'set', frequency_hz=10),
                                    CALIBRATION, keypoint_threshold=0.5)
    collector.start()
    try:
        for ms in range(990, 2200, 10):
            collector.observe_motion(odometry(ms))
        collector.submit(*sample(1000))
        assert entered.wait(1)
        # 5 ms polling and 300 ms frame intervals, 400 ms inference latency.
        for sequence, ms in enumerate((1300, 1600, 1900), start=2):
            for _ in range(4):
                collector.submit(*sample(ms, sequence))
                time.sleep(0.005)
        assert collector._queue.qsize() == 2
        assert collector.dropped == 1
        assert collector.written == 0
    finally:
        release.set()
        collector.stop()
    assert collector.written == 3
    assert collector.worker_error is None


def test_frequency_dedup_stale_and_pair_mismatch(tmp_path):
    collector = MatchDatasetCapture(MatchCaptureConfig(output_dir=tmp_path/'set'), CALIBRATION, keypoint_threshold=0.5)
    collector.start()
    try:
        for ms in range(990, 2200, 10):
            collector.observe_motion(odometry(ms))
        for sequence, ms in enumerate((1000, 1000, 1300, 1600, 2000), start=1):
            collector.submit(*sample(ms, sequence))
        frame, result = sample(3000, 6)
        collector.submit(frame, replace(result, dropped_stale_age_ms=1000))
        frame, result = sample(4000, 7)
        collector.submit(frame, replace(result, frame_sequence=100))
    finally:
        collector.stop()
    assert collector.written == 2


def test_disk_failure_is_reported_and_does_not_escape_submit(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(Path, 'write_bytes', fail)
    collector = MatchDatasetCapture(MatchCaptureConfig(output_dir=tmp_path/'set'), CALIBRATION, keypoint_threshold=0.5)
    collector.start()
    try:
        collector.observe_motion(odometry(990))
        collector.observe_motion(odometry(1090))
        collector.submit(*sample(1000))
        wait_until(lambda: collector.worker_error is not None)
        collector.submit(*sample(2000, 2))
    finally:
        collector.stop()
    assert 'disk full' in collector.worker_error
    assert list((tmp_path/'set').iterdir()) == []


def test_output_defaults_to_timestamp_under_data():
    collector = MatchDatasetCapture(MatchCaptureConfig(), CALIBRATION, keypoint_threshold=0.5)
    assert collector.output_dir.parent == Path('data')
    assert len(collector.output_dir.name) == 22


def test_missing_model_keypoint_is_not_filled_from_bbox_and_empty_is_negative():
    frame, result = sample(1000)
    missing = PoseKeypoint(None, 0.0)
    detection = replace(result.model_detections[0], keypoints=(missing, missing, missing))
    document = xanylabeling_document(frame, replace(result, model_detections=(detection,)), 'one.jpg', 0.5)
    assert [shape['shape_type'] for shape in document['shapes']] == ['rectangle']
    document = xanylabeling_document(frame, replace(result, model_detections=()), 'one.jpg', 0.5)
    assert document['shapes'] == []


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path/'existing.jpg'
    marker.write_bytes(b'existing')
    collector = MatchDatasetCapture(MatchCaptureConfig(output_dir=tmp_path), CALIBRATION, keypoint_threshold=0.5)
    collector.start()
    try:
        collector.observe_motion(odometry(990))
        collector.observe_motion(odometry(1090))
        collector.submit(*sample(1000))
        wait_until(lambda: collector.worker_error is not None)
    finally:
        collector.stop()
    assert 'FileExistsError' in collector.worker_error
    assert marker.read_bytes() == b'existing'
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize('entry', ['match', 'match_nb', 'match_cc', 'match_strategy', 'grab_transport'])
@pytest.mark.parametrize('enabled', [False, True])
def test_cli_capture_dataset_is_opt_in(monkeypatch, entry, enabled):
    import importlib
    import sys
    from unittest.mock import Mock
    from rescue_vision.app import match_runtime

    module = importlib.import_module(f'rescue_vision.app.{entry}')
    run = Mock()
    monkeypatch.setattr(match_runtime, '_run_hardware', run)
    if hasattr(module, '_run_hardware'):
        monkeypatch.setattr(module, '_run_hardware', run)
    argv = [entry, '--config', 'configs/runtime.match.yaml', '--supervised-physical-stop-ready']
    if enabled:
        argv.append('--capture-dataset')
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    run.assert_called_once()
    assert run.call_args.kwargs['capture_dataset'] is enabled
