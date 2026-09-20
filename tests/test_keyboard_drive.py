"""键盘驾驶的纯逻辑单元测试（无硬件、无相机、无 Hailo）。"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from manual_tests.keyboard_drive import (
    KeyboardDriveLogWriter,
    KeyboardDriveCommand,
    KeyboardDriveTelemetry,
    LiveReplayFeedback,
    KeyDriveState,
    _command_label,
    _draw_gyro_overlay,
    apply_keyboard_drive_target,
    build_replay_reference,
    closed_loop_wheel_targets,
    compute_twist,
    decode_keys,
    gripper_angles_for,
    interpolate_replay_reference,
    load_keyboard_drive_log,
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
        (frozenset({"backward", "turn_left"}), (-0.10, 0.60)),
        (frozenset({"backward", "turn_right"}), (-0.10, -0.60)),
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
    def test_linear_direction_is_latched(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["forward"], now_ns=0)
        assert state.active_keys(now_ns=50) == frozenset({"forward"})
        assert state.active_keys(now_ns=10_000) == frozenset({"forward"})

    def test_turn_auto_repeat_combines_with_latched_linear(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["forward"], now_ns=0)
        state.apply(["turn_right"], now_ns=80)
        assert state.active_keys(now_ns=150) == frozenset(
            {"forward", "turn_right"}
        )
        assert state.active_keys(now_ns=181) == frozenset({"forward"})

    def test_opposite_direction_replaces_axis(self):
        state = KeyDriveState(hold_timeout_ns=100)
        state.apply(["forward", "backward"], now_ns=0)
        state.apply(["turn_left", "turn_right"], now_ns=0)
        assert state.active_keys(now_ns=0) == frozenset(
            {"backward", "turn_right"}
        )

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


def _fake_controller(*, wheel_track_m: float = 0.30):
    return SimpleNamespace(
        limits=SimpleNamespace(
            wheel_track_m=wheel_track_m,
            max_linear_velocity_m_s=0.4,
            max_angular_velocity_rad_s=1.2,
            max_wheel_velocity_m_s=0.5,
            min_wheel_velocity_m_s=0.02,
            max_linear_acceleration_m_s2=1.0,
            max_linear_deceleration_m_s2=1.0,
            max_angular_acceleration_rad_s2=10,
            max_angular_deceleration_rad_s2=10,
            left_wheel_speed_weight=1.0,
            right_wheel_speed_weight=1.0,
        )
    )


def _fake_calibration(*, left_radius_mm: float = 50.0):
    return SimpleNamespace(
        encoder_counts_per_revolution=1000,
        left_wheel_radius_mm=left_radius_mm,
        right_wheel_radius_mm=50.0,
        gyro_z_sign=1,
    )


def _log_odometry(sequence: int, sample_us: int, count: int) -> OdometryImu:
    return OdometryImu(
        uart_sequence=sequence,
        received_timestamp_ns=1_000 + sample_us * 1000,
        telemetry_sequence=sequence,
        sample_timestamp_us=sample_us,
        left_encoder_count=count,
        right_encoder_count=count,
        gyro_x_urad_s=0,
        gyro_y_urad_s=0,
        gyro_z_urad_s=100_000,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=9807,
        imu_temperature_cdeg=2500,
        sensor_flags=(
            SensorFlags.LEFT_ENCODER_VALID
            | SensorFlags.RIGHT_ENCODER_VALID
            | SensorFlags.IMU_VALID
        ),
    )


def test_keyboard_drive_log_round_trip(tmp_path) -> None:
    path = tmp_path / "drive.jsonl"
    controller = _fake_controller()
    calibration = _fake_calibration()
    with KeyboardDriveLogWriter(
        path, controller, calibration, started_ns=1_000
    ) as writer:
        assert writer.record(
            1_000,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
            target_wheel_speeds_m_s=(0.0, 0.0),
        )
        assert writer.record(
            11_000,
            linear_velocity_m_s=0.1,
            angular_velocity_rad_s=-0.6,
            target_wheel_speeds_m_s=(0.19, 0.01),
        )
        assert not writer.record(
            12_000,
            linear_velocity_m_s=0.1,
            angular_velocity_rad_s=-0.6,
            target_wheel_speeds_m_s=(0.19, 0.01),
        )
        assert writer.record(
            21_000,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
            target_wheel_speeds_m_s=(0.0, 0.0),
            force=True,
        )
        writer.record_telemetry(_log_odometry(1, 10, 0))
        writer.record_telemetry(_log_odometry(2, 20, 10))

    recording = load_keyboard_drive_log(path, controller, calibration)
    commands = recording.commands
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert header["version"] == 3
    assert header["motion"]["max_linear_acceleration_m_s2"] == 1.0
    assert header["motion"]["max_linear_deceleration_m_s2"] == 1.0
    assert header["motion"]["max_angular_acceleration_rad_s2"] == 10
    assert header["motion"]["max_angular_deceleration_rad_s2"] == 10
    assert [command.elapsed_ns for command in commands] == [0, 10_000, 20_000]
    assert commands[1].linear_velocity_m_s == pytest.approx(0.1)
    assert commands[1].angular_velocity_rad_s == pytest.approx(-0.6)
    assert commands[1].left_target_m_s == pytest.approx(0.19)
    assert len(recording.telemetry) == 2


def test_apply_keyboard_drive_target_does_not_reuse_pre_command_time() -> None:
    class Controller:
        target_wheel_speeds_m_s = (0.07, 0.13)

        def __init__(self) -> None:
            self.update_now_values = []

        def drive_wheel_limited(self, linear, angular):
            # The real method updates internally before installing the target.
            return linear, angular

        def update(self, *, now_ns=None):
            self.update_now_values.append(now_ns)

    class Log:
        def __init__(self) -> None:
            self.records = []

        def record(self, now_ns, **values):
            self.records.append((now_ns, values))

    controller = Controller()
    command_log = Log()

    applied = apply_keyboard_drive_target(
        controller,
        command_log,  # type: ignore[arg-type]
        requested_linear_m_s=0.1,
        requested_angular_rad_s=0.2,
    )

    assert applied == (0.1, 0.2)
    assert controller.update_now_values == [None]
    assert command_log.records[0][1]["target_wheel_speeds_m_s"] == (0.07, 0.13)


def test_keyboard_drive_log_rejects_config_mismatch(tmp_path) -> None:
    path = tmp_path / "drive.jsonl"
    controller = _fake_controller()
    calibration = _fake_calibration()
    with KeyboardDriveLogWriter(
        path, controller, calibration, started_ns=0
    ) as writer:
        writer.record(
            0,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
            target_wheel_speeds_m_s=(0.0, 0.0),
        )
        writer.record(
            1,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
            target_wheel_speeds_m_s=(0.0, 0.0),
            force=True,
        )
        writer.record_telemetry(_log_odometry(1, 10, 0))
        writer.record_telemetry(_log_odometry(2, 20, 0))

    with pytest.raises(ValueError, match="does not match"):
        load_keyboard_drive_log(
            path,
            _fake_controller(wheel_track_m=0.31),
            calibration,
        )


def test_keyboard_drive_log_rejects_nonzero_final_command(tmp_path) -> None:
    path = tmp_path / "drive.jsonl"
    controller = _fake_controller()
    calibration = _fake_calibration()
    with KeyboardDriveLogWriter(
        path, controller, calibration, started_ns=0
    ) as writer:
        writer.record(
            0,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
            target_wheel_speeds_m_s=(0.0, 0.0),
        )
        writer.record(
            1,
            linear_velocity_m_s=0.1,
            angular_velocity_rad_s=0.0,
            target_wheel_speeds_m_s=(0.1, 0.1),
        )
        writer.record_telemetry(_log_odometry(1, 10, 0))
        writer.record_telemetry(_log_odometry(2, 20, 10))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record for record in records if record["record_type"] == "command"][-1][
        "left_target_m_s"
    ] == 0.1

    with pytest.raises(ValueError, match="end with a zero-speed"):
        load_keyboard_drive_log(path, controller, calibration)


def test_replay_reference_interpolates_encoder_distance_and_gyro_heading() -> None:
    calibration = _fake_calibration()
    valid_flags = int(
        SensorFlags.LEFT_ENCODER_VALID
        | SensorFlags.RIGHT_ENCODER_VALID
        | SensorFlags.IMU_VALID
    )
    samples = (
        KeyboardDriveTelemetry(0, 1, 0, 100, 200, 100_000, valid_flags),
        KeyboardDriveTelemetry(
            100_000_000, 2, 100_000, 200, 300, 100_000, valid_flags
        ),
        KeyboardDriveTelemetry(
            200_000_000, 3, 200_000, 300, 400, 100_000, valid_flags
        ),
    )

    points = build_replay_reference(samples, calibration)
    midpoint = interpolate_replay_reference(points, 0.15)

    assert midpoint.left_distance_m == pytest.approx(0.15 * math.pi * 0.1)
    assert midpoint.right_distance_m == pytest.approx(0.15 * math.pi * 0.1)
    assert midpoint.heading_rad == pytest.approx(0.015)


def test_closed_loop_targets_correct_position_and_heading_error() -> None:
    controller = _fake_controller()
    feedback = LiveReplayFeedback(_fake_calibration())
    feedback.left_distance_m = 0.08
    feedback.right_distance_m = 0.09
    feedback.heading_rad = 0.05
    feedback.heading_available = True
    reference = SimpleNamespace(
        left_distance_m=0.10,
        right_distance_m=0.10,
        heading_rad=0.10,
    )
    feedforward = KeyboardDriveCommand(0, 0.1, 0.0, 0.1, 0.1)

    left, right, left_error, right_error = closed_loop_wheel_targets(
        controller,
        reference,
        feedback,
        feedforward,
    )

    assert left_error == pytest.approx(0.02)
    assert right_error == pytest.approx(0.01)
    assert right > left  # 正航向误差应增加逆时针差速。
    assert max(abs(left), abs(right)) <= controller.limits.max_wheel_velocity_m_s


def test_live_replay_feedback_uses_encoder_baseline_and_integrates_gyro() -> None:
    calibration = _fake_calibration()
    feedback = LiveReplayFeedback(calibration)
    first = _log_odometry(1, 10_000, 100)
    second = _log_odometry(2, 110_000, 200)

    feedback.observe(first)
    feedback.observe(second)

    expected_distance = 100 * math.pi * 0.1 / 1000.0
    assert feedback.left_distance_m == pytest.approx(expected_distance)
    assert feedback.right_distance_m == pytest.approx(expected_distance)
    assert feedback.heading_rad == pytest.approx(0.01)
    assert feedback.heading_available is True
    feedback.require_recent(second.received_timestamp_ns + 199_000_000)
    with pytest.raises(RuntimeError, match="stale"):
        feedback.require_recent(second.received_timestamp_ns + 201_000_000)


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
