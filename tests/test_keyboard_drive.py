"""键盘驾驶的纯逻辑单元测试（无硬件、无相机、无 Hailo）。"""

from __future__ import annotations

import numpy as np
import pytest

from manual_tests.keyboard_drive import (
    KeyDriveState,
    _command_label,
    _draw_gyro_overlay,
    compute_twist,
    decode_keys,
    gripper_angles_for,
)
from rescue_vision.motion import GripperCalibration, OdometryImu, SensorFlags


@pytest.mark.parametrize(
    ("buffer", "expected_events", "expected_leftover"),
    [
        (b"", [], b""),
        (b"w", ["forward"], b""),
        (b"W", ["forward"], b""),
        (b"s", ["backward"], b""),
        (b"a", ["turn_left"], b""),
        (b"d", ["turn_right"], b""),
        (b"q", ["quit"], b""),
        (b"Q", ["quit"], b""),
        (b" ", ["stop"], b""),
        (b"z", ["gripper_open"], b""),
        (b"Z", ["gripper_open"], b""),
        (b"x", ["gripper_transport"], b""),
        (b"X", ["gripper_transport"], b""),
        (b"c", ["gripper_close"], b""),
        (b"C", ["gripper_close"], b""),
        (b"wAsD", ["forward", "turn_left", "backward", "turn_right"], b""),
        (b"\x1b[A", ["forward"], b""),
        (b"\x1b[B", ["backward"], b""),
        (b"\x1b[C", ["turn_right"], b""),
        (b"\x1b[D", ["turn_left"], b""),
        (b"\x1b", [], b"\x1b"),
        (b"\x1b[", [], b"\x1b["),
        (b"w\x1b", ["forward"], b"\x1b"),
        (b"\x1b[Z", [], b""),
        # Esc 后跟非 '[' 字节：Esc 视为 quit，后续字节继续按普通键解析。
        (b"\x1bX", ["quit", "gripper_transport"], b""),
        (b"!!", [], b""),
    ],
)
def test_decode_keys(buffer, expected_events, expected_leftover):
    events, leftover = decode_keys(buffer)
    assert events == expected_events
    assert leftover == expected_leftover


def test_decode_keys_rejects_non_bytes():
    with pytest.raises(TypeError):
        decode_keys("w")  # type: ignore[arg-type]


def test_decode_keys_incomplete_csi_resolves_across_polls():
    # 两次轮询拼接：先收到 Esc，再收到 [A，应合成一次 forward。
    first_events, leftover = decode_keys(b"\x1b")
    assert first_events == []
    assert leftover == b"\x1b"
    second_events, second_leftover = decode_keys(leftover + b"[A")
    assert second_events == ["forward"]
    assert second_leftover == b""


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (frozenset(), (0.0, 0.0)),
        (frozenset({"forward"}), (0.10, 0.0)),
        (frozenset({"backward"}), (-0.10, 0.0)),
        (frozenset({"turn_left"}), (0.0, 0.60)),
        (frozenset({"turn_right"}), (0.0, -0.60)),
        (frozenset({"forward", "turn_left"}), (0.10, 0.60)),
        (frozenset({"forward", "turn_right"}), (0.10, -0.60)),
        # 相对的两个键同按：该轴归零。
        (frozenset({"forward", "backward"}), (0.0, 0.0)),
        (frozenset({"turn_left", "turn_right"}), (0.0, 0.0)),
        # 未知键被忽略。
        (frozenset({"forward", "ignored"}), (0.10, 0.0)),
    ],
)
def test_compute_twist(keys, expected):
    result = compute_twist(
        keys,
        linear_speed_m_s=0.10,
        angular_speed_rad_s=0.60,
    )
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (frozenset(), "stop"),
        (frozenset({"forward"}), "forward"),
        (frozenset({"backward"}), "backward"),
        (frozenset({"turn_left"}), "left"),
        (frozenset({"turn_right"}), "right"),
        (frozenset({"forward", "turn_left"}), "forward-left"),
        (frozenset({"backward", "turn_right"}), "backward-right"),
        (frozenset({"forward", "backward"}), "stop"),
    ],
)
def test_command_label(keys, expected):
    assert _command_label(keys) == expected


class TestKeyDriveState:
    def test_hold_and_timeout(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["forward"], now_ns=0)
        assert state.active_keys(now_ns=50) == frozenset({"forward"})
        assert state.active_keys(now_ns=100) == frozenset({"forward"})
        assert state.active_keys(now_ns=101) == frozenset()

    def test_auto_repeat_keeps_held(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["forward"], now_ns=0)
        state.apply(["forward"], now_ns=80)
        assert state.active_keys(now_ns=150) == frozenset({"forward"})

    def test_stop_is_edge_triggered_and_clears_drive(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["forward", "stop"], now_ns=0)
        assert state.consume_stop() is True
        assert state.consume_stop() is False
        assert state.active_keys(now_ns=0) == frozenset()

    def test_quit_is_edge_triggered(self):
        state = KeyDriveState()
        state.apply(["quit"], now_ns=0)
        assert state.consume_quit() is True
        assert state.consume_quit() is False

    def test_unknown_events_are_ignored(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["bogus"], now_ns=0)
        assert state.active_keys(now_ns=0) == frozenset()

    def test_gripper_consume(self):
        state = KeyDriveState()
        assert state.consume_gripper() is None
        state.apply(["gripper_open"], now_ns=0)
        assert state.consume_gripper() == "open"
        assert state.consume_gripper() is None

    def test_gripper_last_press_wins(self):
        state = KeyDriveState()
        state.apply(["gripper_open", "gripper_close"], now_ns=0)
        assert state.consume_gripper() == "close"

    def test_hold_timeout_must_be_nonnegative_int(self):
        with pytest.raises(ValueError):
            KeyDriveState(hold_timeout_ns=-1)
        with pytest.raises(ValueError):
            KeyDriveState(hold_timeout_ns=True)  # type: ignore[arg-type]


def _gripper_calibration(*, with_transport: bool = True) -> GripperCalibration:
    return GripperCalibration(
        open_left_angle_deg=20.0,
        open_right_angle_deg=160.0,
        closed_left_angle_deg=80.0,
        closed_right_angle_deg=100.0,
        full_travel_time_s=1.0,
        angle_sum_deg=180.0,
        transport_left_angle_deg=50.0 if with_transport else None,
        transport_right_angle_deg=130.0 if with_transport else None,
    )


class TestGripperAngles:
    def test_open(self):
        assert gripper_angles_for("open", _gripper_calibration()) == (
            20.0,
            160.0,
        )

    def test_close(self):
        assert gripper_angles_for("close", _gripper_calibration()) == (
            80.0,
            100.0,
        )

    def test_transport(self):
        assert gripper_angles_for("transport", _gripper_calibration()) == (
            50.0,
            130.0,
        )

    def test_transport_missing_raises(self):
        calibration = _gripper_calibration(with_transport=False)
        with pytest.raises(ValueError):
            gripper_angles_for("transport", calibration)

    def test_unknown_action_raises(self):
        with pytest.raises(ValueError):
            gripper_angles_for("bogus", _gripper_calibration())

    def test_rejects_non_calibration(self):
        with pytest.raises(TypeError):
            gripper_angles_for("open", object())  # type: ignore[arg-type]


def test_draw_gyro_overlay_shows_latest_raw_values() -> None:
    image = np.zeros((80, 320, 3), dtype=np.uint8)
    odometry = OdometryImu(
        uart_sequence=1,
        received_timestamp_ns=1_000_000_000,
        telemetry_sequence=2,
        sample_timestamp_us=1234,
        left_encoder_count=0,
        right_encoder_count=0,
        gyro_x_urad_s=100_000,
        gyro_y_urad_s=-200_000,
        gyro_z_urad_s=300_000,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=9807,
        imu_temperature_cdeg=2500,
        sensor_flags=SensorFlags.IMU_VALID,
    )

    result = _draw_gyro_overlay(image, odometry, 1_025_000_000)

    assert result is image
    assert np.any(image != 0)
