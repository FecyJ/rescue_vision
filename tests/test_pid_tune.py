from __future__ import annotations

import math

import pytest

from manual_tests.pid_tune import (
    StraightRunAccumulator,
    WheelScales,
    WheelSpeedSample,
    parse_speeds,
    parse_telemetry_line,
    turn_duration_s,
    update_wheel_scales,
)


def _sample(left: float, right: float) -> WheelSpeedSample:
    return WheelSpeedSample(
        timestamp_ms=0,
        left_speed_m_s=left,
        right_speed_m_s=right,
    )


def _feed(
    accumulator: StraightRunAccumulator,
    left: float,
    right: float,
    dt_s: float,
    intervals: int,
) -> None:
    for index in range(intervals + 1):
        accumulator.submit(_sample(left, right), now_s=index * dt_s)


def test_parse_telemetry_line_parses_measured_speeds() -> None:
    sample = parse_telemetry_line("t12345,0.19,0.20,0.20,0.20,90,45")

    assert sample is not None
    assert sample.timestamp_ms == 12345
    assert sample.left_speed_m_s == pytest.approx(0.19)
    assert sample.right_speed_m_s == pytest.approx(0.20)


def test_parse_telemetry_line_ignores_non_telemetry_and_empty() -> None:
    assert parse_telemetry_line("PID=3.500,0.250,0.000") is None
    assert parse_telemetry_line("OK m=0.20,0.20") is None
    assert parse_telemetry_line("") is None
    assert parse_telemetry_line("\n") is None


def test_parse_telemetry_line_rejects_bad_numbers_and_short_lines() -> None:
    assert parse_telemetry_line("t1,abc,0.20") is None
    assert parse_telemetry_line("t1,0.10") is None
    assert parse_telemetry_line("t1,0.10,nan") is None


def test_parse_speeds_returns_validated_tuple() -> None:
    assert parse_speeds("0.05,0.10,0.15,0.20") == (0.05, 0.10, 0.15, 0.20)


@pytest.mark.parametrize("text", ["", ",", "0.02", "0.25", "0.1,abc"])
def test_parse_speeds_rejects_invalid_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_speeds(text)


def test_turn_duration_matches_differential_geometry() -> None:
    # angular = 2 * wheel_speed / track; turn 90 deg at 0.1 m/s on 0.2 m track.
    duration = turn_duration_s(
        wheel_track_m=0.2,
        turn_speed_m_s=0.1,
        angle_rad=math.pi / 2.0,
    )

    assert duration == pytest.approx(math.pi / 2.0)


def test_accumulator_integrates_distance_and_heading() -> None:
    accumulator = StraightRunAccumulator(wheel_track_m=0.5)
    _feed(accumulator, left=0.3, right=0.1, dt_s=0.1, intervals=10)

    assert accumulator.distance_left_m == pytest.approx(0.3)
    assert accumulator.distance_right_m == pytest.approx(0.1)
    assert accumulator.distance_m == pytest.approx(0.2)
    assert accumulator.heading_deviation_rad == pytest.approx(0.4)


def test_accumulator_skips_large_gaps() -> None:
    accumulator = StraightRunAccumulator(wheel_track_m=0.5, max_dt_s=0.5)
    accumulator.submit(_sample(0.2, 0.2), now_s=0.0)
    accumulator.submit(_sample(0.2, 0.2), now_s=1.0)  # gap > max_dt_s -> re-anchor
    accumulator.submit(_sample(0.2, 0.2), now_s=1.1)

    assert accumulator.distance_left_m == pytest.approx(0.2 * 0.1)


def test_update_wheel_scales_leaves_tracking_wheels_unchanged() -> None:
    left, right = update_wheel_scales(
        1.0,
        1.0,
        commanded_speed_m_s=0.10,
        avg_left_speed_m_s=0.10,
        avg_right_speed_m_s=0.10,
    )

    assert left == pytest.approx(1.0)
    assert right == pytest.approx(1.0)


def test_update_wheel_scales_lowers_the_faster_wheel() -> None:
    left, right = update_wheel_scales(
        1.0,
        1.0,
        commanded_speed_m_s=0.10,
        avg_left_speed_m_s=0.12,
        avg_right_speed_m_s=0.10,
    )

    assert left == pytest.approx(1.0 / 1.2)
    assert right == pytest.approx(1.0)


def test_update_wheel_scales_clamps_large_corrections() -> None:
    left, _ = update_wheel_scales(
        1.0,
        1.0,
        commanded_speed_m_s=0.10,
        avg_left_speed_m_s=0.30,
        avg_right_speed_m_s=0.10,
    )

    assert left == pytest.approx(0.8)  # clamped to _MIN_SCALE_CORRECTION


def test_update_wheel_scales_ignores_near_zero_measured_speed() -> None:
    left, right = update_wheel_scales(
        1.0,
        1.0,
        commanded_speed_m_s=0.10,
        avg_left_speed_m_s=0.0,
        avg_right_speed_m_s=0.10,
    )

    assert left == pytest.approx(1.0)
    assert right == pytest.approx(1.0)


def test_update_wheel_scales_rejects_non_positive_command() -> None:
    with pytest.raises(ValueError):
        update_wheel_scales(
            1.0,
            1.0,
            commanded_speed_m_s=0.0,
            avg_left_speed_m_s=0.1,
            avg_right_speed_m_s=0.1,
        )


def test_wheel_scales_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        WheelScales(0.0, 1.0)
    with pytest.raises(ValueError):
        WheelScales(1.0, -1.0)
