from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rescue_vision.app import teach_replay
from rescue_vision.app.teach_replay import (
    GyroHeadingTracker,
    ReplaySegment,
    TeachSample,
    build_replay_segments,
    gyro_corrected_wheel_speeds,
    load_teach_log,
    write_teach_log,
)
from rescue_vision.localization import OdometryCalibration
from rescue_vision.motion import OdometryImu, SensorFlags


CALIBRATION = OdometryCalibration(
    encoder_counts_per_revolution=100,
    left_wheel_radius_mm=100.0,
    right_wheel_radius_mm=200.0,
    gyro_z_sign=-1,
)
VALID_FLAGS = int(
    SensorFlags.IMU_VALID
    | SensorFlags.LEFT_ENCODER_VALID
    | SensorFlags.RIGHT_ENCODER_VALID
)


def sample(
    timestamp_us: int,
    left: int,
    right: int,
    *,
    flags: int = VALID_FLAGS,
) -> TeachSample:
    return TeachSample(
        received_timestamp_ns=timestamp_us * 1000,
        telemetry_sequence=(timestamp_us // 10_000) & 0xFFFF,
        sample_timestamp_us=timestamp_us,
        left_encoder_count=left,
        right_encoder_count=right,
        gyro_x_urad_s=1,
        gyro_y_urad_s=2,
        gyro_z_urad_s=3,
        accel_x_mm_s2=4,
        accel_y_mm_s2=5,
        accel_z_mm_s2=6,
        imu_temperature_cdeg=2500,
        sensor_flags=flags,
    )


def odometry(
    timestamp_us: int,
    gyro_z_urad_s: int,
    *,
    flags: SensorFlags = SensorFlags(VALID_FLAGS),
) -> OdometryImu:
    return OdometryImu(
        uart_sequence=timestamp_us // 10_000,
        received_timestamp_ns=timestamp_us * 1000,
        telemetry_sequence=(timestamp_us // 10_000) & 0xFFFF,
        sample_timestamp_us=timestamp_us,
        left_encoder_count=0,
        right_encoder_count=0,
        gyro_x_urad_s=0,
        gyro_y_urad_s=0,
        gyro_z_urad_s=gyro_z_urad_s,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=0,
        imu_temperature_cdeg=2500,
        sensor_flags=flags,
    )


def test_log_round_trip_preserves_every_raw_field(tmp_path) -> None:
    path = tmp_path / "teach.jsonl"
    expected = [sample(100_000, 10, 20), sample(160_000, 20, 30)]

    write_teach_log(path, CALIBRATION, expected)

    assert load_teach_log(path, CALIBRATION) == expected
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_log_rejects_different_runtime_calibration(tmp_path) -> None:
    path = tmp_path / "teach.jsonl"
    write_teach_log(path, CALIBRATION, [sample(0, 0, 0), sample(50_000, 1, 1)])
    changed = OdometryCalibration(100, 101.0, 200.0, -1)

    with pytest.raises(ValueError, match="does not match"):
        load_teach_log(path, changed)


def test_segments_average_counts_and_speed_scale_preserves_distance() -> None:
    samples = [
        sample(0, 0, 0),
        sample(20_000, 1, 1),
        sample(60_000, 10, -5),
        sample(120_000, 20, -10),
    ]

    segments = build_replay_segments(
        samples,
        CALIBRATION,
        interval_s=0.05,
        speed_scale=0.5,
    )

    assert len(segments) == 2
    assert sum(item.duration_s for item in segments) == pytest.approx(0.24)
    assert sum(item.left_m_s * item.duration_s for item in segments) == pytest.approx(
        20 * 2 * math.pi * 0.1 / 100
    )
    assert sum(item.right_m_s * item.duration_s for item in segments) == pytest.approx(
        -10 * 2 * math.pi * 0.2 / 100
    )


def test_segments_merge_a_tiny_final_remainder() -> None:
    segments = build_replay_segments(
        [sample(0, 0, 0), sample(50_000, 5, 5), sample(51_000, 6, 6)],
        CALIBRATION,
        interval_s=0.05,
    )

    assert len(segments) == 1
    assert segments[0].duration_s == pytest.approx(0.051)


def test_gyro_heading_is_integrated_and_unchanged_by_speed_scale() -> None:
    samples = [
        sample(0, 0, 0),
        sample(50_000, 1, 1),
        sample(100_000, 2, 2),
    ]
    samples = [replace(item, gyro_z_urad_s=1_000_000) for item in samples]

    segments = build_replay_segments(
        samples,
        CALIBRATION,
        speed_scale=0.5,
        use_gyro=True,
    )

    assert sum(item.duration_s for item in segments) == pytest.approx(0.2)
    assert sum(item.heading_delta_rad or 0.0 for item in segments) == pytest.approx(
        -0.1
    )


def test_gyro_correction_changes_wheel_difference_within_limits() -> None:
    controller = SimpleNamespace(
        limits=SimpleNamespace(
            wheel_track_m=0.2,
            max_angular_velocity_rad_s=1.0,
            max_wheel_velocity_m_s=1.0,
        )
    )
    segment = ReplaySegment(0.05, 0.5, 0.5, 0.0)

    left, right = gyro_corrected_wheel_speeds(
        controller,
        segment,
        target_heading_rad=0.1,
        measured_heading_rad=0.0,
    )

    assert left == pytest.approx(0.48)
    assert right == pytest.approx(0.52)


def test_live_gyro_tracker_integrates_robot_heading_sign() -> None:
    tracker = GyroHeadingTracker(gyro_z_sign=-1)

    tracker.observe(odometry(0, 1_000_000))
    tracker.observe(odometry(100_000, 1_000_000))

    assert tracker.ready
    assert tracker.heading_rad == pytest.approx(-0.1)


def test_live_gyro_tracker_rebases_after_a_short_invalid_sample() -> None:
    tracker = GyroHeadingTracker(gyro_z_sign=1)
    invalid = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID

    tracker.observe(odometry(0, 1_000_000))
    tracker.observe(odometry(10_000, 1_000_000, flags=invalid))
    assert not tracker.available
    tracker.observe(odometry(20_000, 1_000_000))

    assert tracker.available
    assert tracker.heading_rad == 0.0


def test_live_gyro_tracker_rejects_prolonged_unavailability() -> None:
    tracker = GyroHeadingTracker(gyro_z_sign=1)
    invalid = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID

    tracker.observe(odometry(0, 0, flags=invalid))
    with pytest.raises(RuntimeError, match="more than 200 ms"):
        tracker.observe(odometry(210_000, 0, flags=invalid))


def test_live_gyro_tracker_rejects_missing_telemetry() -> None:
    tracker = GyroHeadingTracker(gyro_z_sign=1)
    tracker.observe(odometry(0, 0))

    tracker.require_recent(200_000_000)
    with pytest.raises(RuntimeError, match="telemetry has been missing"):
        tracker.require_recent(200_000_001)


def test_invalid_recorded_imu_is_skipped_without_blocking_encoder_replay() -> None:
    encoder_only_flags = int(
        SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
    )
    samples = [
        sample(0, 0, 0, flags=encoder_only_flags),
        sample(50_000, 1, 1, flags=encoder_only_flags),
    ]

    encoder_segments = build_replay_segments(samples, CALIBRATION, use_gyro=False)
    gyro_segments = build_replay_segments(samples, CALIBRATION, use_gyro=True)

    assert encoder_segments[0].heading_delta_rad is None
    assert gyro_segments[0].heading_delta_rad == 0.0


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (
            [
                sample(0, 0, 0),
                sample(20_000, 1, 1, flags=int(SensorFlags.IMU_VALID)),
                sample(60_000, 2, 2),
            ],
            "Encoder invalid",
        ),
        (
            [
                sample(0, 0, 0),
                sample(
                    20_000,
                    1,
                    1,
                    flags=VALID_FLAGS | int(SensorFlags.SAMPLE_OVERRUN),
                ),
                sample(60_000, 2, 2),
            ],
            "Sample overrun",
        ),
        ([sample(10, 0, 0), sample(10, 1, 1)], "timestamps must increase"),
    ],
)
def test_segments_reject_bad_intermediate_telemetry(samples, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_replay_segments(samples, CALIBRATION)


def test_loader_rejects_non_integer_sample_field(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    write_teach_log(path, CALIBRATION, [sample(0, 0, 0), sample(50_000, 1, 1)])
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["left_encoder_count"] = "zero"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid sample"):
        load_teach_log(path, CALIBRATION)


def test_existing_log_confirmation_happens_before_uart_opens(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "teach.jsonl"
    write_teach_log(path, CALIBRATION, [sample(0, 0, 0), sample(50_000, 1, 1)])

    class Channel:
        started = False
        starts = 0

        def __enter__(self):
            self.started = True
            self.starts += 1
            return self

        def __exit__(self, *_args):
            self.started = False

    channel = Channel()
    controller = SimpleNamespace(
        limits=SimpleNamespace(max_wheel_velocity_m_s=1.0)
    )
    config = SimpleNamespace(
        uart=SimpleNamespace(enabled=True, build_channel=lambda: channel),
        motion=SimpleNamespace(
            enabled=True,
            odometry=SimpleNamespace(build_calibration=lambda: CALIBRATION),
            build_controller=lambda built_channel: controller,
        ),
    )
    monkeypatch.setattr(teach_replay, "load_runtime_config", lambda _path: config)

    def confirm(_path):
        assert channel.started is False
        return False

    monkeypatch.setattr(teach_replay, "_confirm_replay", confirm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-teach-replay",
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--replay",
            str(path),
            "--supervised-physical-stop-ready",
        ],
    )

    teach_replay.main()

    assert channel.starts == 0


def test_new_recording_closes_uart_before_confirmation(tmp_path, monkeypatch) -> None:
    class Channel:
        started = False
        starts = 0

        def __enter__(self):
            self.started = True
            self.starts += 1
            return self

        def __exit__(self, *_args):
            self.started = False

    class Controller:
        emergency_stop_latched = False
        limits = SimpleNamespace(max_wheel_velocity_m_s=1.0)
        brakes = 0

        def synchronize(self, *, timeout_s):
            assert channel.started is True
            assert timeout_s == 1.0

        def soft_brake(self):
            self.brakes += 1

    channel = Channel()
    controller = Controller()
    config = SimpleNamespace(
        uart=SimpleNamespace(enabled=True, build_channel=lambda: channel),
        motion=SimpleNamespace(
            enabled=True,
            synchronization_timeout_s=1.0,
            odometry=SimpleNamespace(build_calibration=lambda: CALIBRATION),
            build_controller=lambda built_channel: controller,
        ),
    )
    monkeypatch.setattr(teach_replay, "load_runtime_config", lambda _path: config)
    monkeypatch.setattr(
        teach_replay,
        "record_samples",
        lambda _controller: [sample(0, 0, 0), sample(50_000, 1, 1)],
    )

    def confirm(path):
        assert path.exists()
        assert channel.started is False
        return False

    monkeypatch.setattr(teach_replay, "_confirm_replay", confirm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-teach-replay",
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--log-dir",
            str(tmp_path),
            "--supervised-physical-stop-ready",
        ],
    )

    teach_replay.main()

    assert channel.starts == 1
    assert controller.brakes == 1
