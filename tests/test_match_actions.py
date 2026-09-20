from __future__ import annotations

from dataclasses import replace
import math
import sys

import numpy as np
import pytest

from rescue_vision.app import (
    GripperPosture,
    MatchSequence,
    MatchState,
    MatchPreflight,
)
from rescue_vision.config import MatchRuntimeConfig, load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.world import TeamColor
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    FieldFeatureDetectionResult,
    FieldPoseKeypoint,
    PerceptionSnapshot,
    RoiColorSegmentation,
    SafeZoneColor,
    SafeZoneObservation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import MultiTargetTracker, TrackingConfig
from rescue_vision.mission import SafetySignals
from rescue_vision.app.session_log import (
    _begin_time_named_log,
    _end_time_named_log,
)


def runtime_config(**overrides: object) -> MatchRuntimeConfig:
    values: dict[str, object] = {"enabled": True}
    values.update(overrides)
    return MatchRuntimeConfig(**values)  # type: ignore[arg-type]


def heading(heading: float = -math.pi / 2.0, **_: object) -> float:
    return heading


def observation(
    frame_sequence: int,
    timestamp_ns: int,
    ground: GroundPoint,
    *,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
    box_x: float = 10.0,
) -> TargetObservation:
    box = UndistortedBoundingBox(box_x, 10.0, box_x + 10.0, 20.0)
    segmentation = RoiColorSegmentation(
        candidate_class=target_class,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=box,
        mask=np.full((10, 10), 255, dtype=np.uint8),
        color_fraction=1.0,
        dominance=1.0,
    )
    return TargetObservation(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        image_size=(100, 100),
        model_target_class=target_class,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(target_class, 1.0),
        detection_confidence=0.95,
        box=box,
        color_segmentation=segmentation,
        k0=UndistortedPixel(box_x + 5.0, 15.0),
        k0_confidence=0.95,
        ground_point=ground,
        quality=frozenset(),
    )


def snapshot(
    frame_sequence: int,
    timestamp_ns: int,
    *observations: TargetObservation,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        observations=tuple(observations),
        field_features=None,
    )


def safe_zone_snapshot(
    frame_sequence: int,
    timestamp_ns: int,
    k0: GroundPoint,
    k1: GroundPoint,
) -> PerceptionSnapshot:
    box = UndistortedBoundingBox(10.0, 10.0, 90.0, 90.0)

    def keypoint(ground: GroundPoint) -> FieldPoseKeypoint:
        return FieldPoseKeypoint(UndistortedPixel(50.0, 50.0), ground, 0.9)

    zone = SafeZoneObservation(
        box,
        keypoint(k0),
        keypoint(k1),
        keypoint(GroundPoint(0.0, 0.0)),
        SafeZoneColor.RED,
        0.9,
        frozenset(),
    )
    features = FieldFeatureDetectionResult(
        frame_sequence,
        timestamp_ns,
        timestamp_ns,
        (100, 100),
        (zone,),
        None,
    )
    return PerceptionSnapshot(
        frame_sequence,
        timestamp_ns,
        timestamp_ns,
        (),
        features,
    )


def make_sequence(
    *,
    config: MatchRuntimeConfig | None = None,
    safe_zone_fallback_target_field: FieldPoint | None = None,
    initial_field_position: FieldPoint | None = None,
) -> MatchSequence:
    return MatchSequence(
        config or runtime_config(),
        tracker=MultiTargetTracker(
            TrackingConfig(
                confirmation_hits=2,
                max_association_ground_mm=250.0,
                min_association_iou=0.1,
                max_coast_ms=600.0,
                confidence_decay_per_second=0.8,
                min_confidence=0.15,
            )
        ),
        gripper_full_travel_time_s=1.0,
        team_color=TeamColor.RED,
        initial_field_position=initial_field_position,
    )


def start_sequence(sequence: MatchSequence) -> None:
    ready = sequence.preflight(
        0,
        MatchPreflight(True, True, True, True, True, True),
    )
    assert ready.state is MatchState.PREFLIGHT
    sequence.start(1)




def test_match_state_banner_is_prominent_and_includes_reason(capsys) -> None:
    from rescue_vision.app.match import _print_state_banner

    _print_state_banner(MatchState.SEARCH_CLUSTER, "waiting_for_cluster")

    output = capsys.readouterr().out
    assert "state=search_cluster" in output
    assert "reason=waiting_for_cluster" in output
    assert output.count("=") >= 40


def test_match_log_file_uses_prefix_but_lines_do_not(tmp_path) -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        stream, stdout_before, stderr_before = _begin_time_named_log(
            tmp_path,
            file_prefix="match_",
            line_prefix="",
        )
        assert stream is not None
        print("state=search_cluster", flush=True)
        log_files = list(tmp_path.glob("match_*.log"))
        assert len(log_files) == 1
        lines = log_files[0].read_text(encoding="utf-8").splitlines()
        assert lines
        assert all(not line.startswith("match_") for line in lines)
        _end_time_named_log(stream, stdout_before, stderr_before)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def test_startup_turn_and_forward_have_independent_half_second_settles() -> None:
    sequence = make_sequence()
    start_sequence(sequence)

    turning = sequence.step(
        2,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=0.0,
    )
    assert turning.state is MatchState.STARTUP_TURN_RIGHT
    assert turning.angular_velocity_rad_s < 0.0

    turned = sequence.step(
        3,
        perception=None,
        heading_rad=heading(heading=-math.pi / 2.0 - math.pi / 4.0),
        cumulative_distance_m=0.0,
    )
    assert turned.state is MatchState.STARTUP_TURN_SETTLE
    assert turned.linear_velocity_m_s == 0.0
    waiting = sequence.step(
        400_000_003,
        perception=None,
        heading_rad=heading(heading=-math.pi / 2.0 - math.pi / 4.0),
        cumulative_distance_m=0.0,
    )
    assert waiting.reason == "waiting_settle"
    forward = sequence.step(
        500_000_004,
        perception=None,
        heading_rad=heading(heading=-math.pi / 2.0 - math.pi / 4.0),
        cumulative_distance_m=0.0,
    )
    assert forward.state is MatchState.STARTUP_FORWARD
    moving = sequence.step(
        500_000_005,
        perception=None,
        heading_rad=heading(heading=-math.pi / 2.0 - math.pi / 4.0),
        cumulative_distance_m=0.0,
    )
    assert moving.linear_velocity_m_s == pytest.approx(0.12)
    complete = sequence.step(
        600_000_005,
        perception=None,
        heading_rad=heading(heading=-math.pi / 2.0 - math.pi / 4.0),
        cumulative_distance_m=1.2,
    )
    assert complete.state is MatchState.STARTUP_FORWARD_SETTLE
    search = sequence.step(
        1_100_000_006,
        perception=None,
        heading_rad=heading(heading=-math.pi / 2.0 - math.pi / 4.0),
        cumulative_distance_m=1.2,
    )
    assert search.state is MatchState.SEARCH_CLUSTER
    assert search.angular_velocity_rad_s < 0.0


def test_startup_straight_pid_outputs_right_correction_for_right_wheel_excess() -> None:
    sequence = make_sequence(
        config=runtime_config(
            startup_straight_pid_kp_rad_s_per_m_s=1.5,
            startup_straight_pid_max_angular_velocity_rad_s=0.08,
        )
    )
    start_sequence(sequence)
    sequence._started = True
    sequence.state = MatchState.STARTUP_FORWARD

    result = sequence.step(
        10,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=0.0,
        left_speed_feedback_m_s=0.10,
        right_speed_feedback_m_s=0.12,
    )

    assert result.state is MatchState.STARTUP_FORWARD
    assert result.angular_velocity_rad_s == pytest.approx(-0.03)


















def test_breakup_moves_forward_one_meter_then_backward_twenty_centimeters() -> None:
    sequence = make_sequence()
    sequence.config = replace(sequence.config, breakup_forward_distance_m=1.0,
                              breakup_backward_distance_m=0.2)
    # This fixture exercises the legacy fixed-action helper directly; formal
    # match execution supplies a BreakupPlan before entering the state.
    sequence._dynamic_breakup_enabled = False
    start_sequence(sequence)
    sequence._started = True
    sequence.state = MatchState.BREAKUP_FORWARD
    sequence._breakup_static_map = load_runtime_config("configs/runtime.match.yaml").world.static_map
    sequence._fallback_field_position = FieldPoint(800.0, 0.0)

    moving_forward = sequence.step(
        10,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=0.0,
    )
    assert moving_forward.linear_velocity_m_s == pytest.approx(0.40)

    open_start = sequence.step(
        20,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=1.0,
    )
    assert open_start.state is MatchState.OPEN_GRIPPER_SETTLE
    assert open_start.linear_velocity_m_s == 0.0
    assert open_start.gripper_posture is GripperPosture.OPEN

    backward_start = sequence.step(
        1_000_000_020,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=1.0,
    )
    assert backward_start.state is MatchState.BREAKUP_BACKWARD
    assert backward_start.linear_velocity_m_s == 0.0
    assert backward_start.gripper_posture is GripperPosture.OPEN

    moving_backward = sequence.step(
        1_000_000_030,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=1.0,
    )
    assert moving_backward.linear_velocity_m_s == pytest.approx(-0.08)
    assert moving_backward.gripper_posture is GripperPosture.OPEN

    close_start = sequence.step(
        1_000_000_040,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=0.8,
    )
    assert close_start.state is MatchState.CLOSE_GRIPPER_SETTLE
    assert close_start.linear_velocity_m_s == 0.0
    assert close_start.gripper_posture is GripperPosture.CLOSED

    close_spin = sequence.step(
        2_000_000_040,
        perception=None,
        heading_rad=heading(),
        cumulative_distance_m=0.8,
    )
    assert close_spin.state is MatchState.CLOSE_GRIPPER_SPIN
    assert close_spin.gripper_posture is GripperPosture.CLOSED




def test_no_isolated_green_keeps_attempt_budget_and_loops_to_search() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._started = True
    sequence.state = MatchState.CHECK_ISOLATED_GREEN
    result = sequence.step(
        10,
        perception=snapshot(
            1,
            10,
            observation(1, 10, GroundPoint(500.0, 0.0)),
            observation(
                1,
                10,
                GroundPoint(300.0, 120.0),
                target_class=TargetClass.BLACK_CORE,
                box_x=30.0,
            ),
        ),
        heading_rad=heading(),
        cumulative_distance_m=0.0,
        safety=SafetySignals.nominal(10),
    )
    assert result.state is MatchState.SEARCH_CLUSTER
    assert sequence._breakup_attempts == []
























@pytest.mark.parametrize("failure_phase", [None, "uart_open", "uart_synchronize"])
def test_hardware_entry_reaches_control_without_building_fusion(
    monkeypatch, tmp_path, failure_phase,
) -> None:
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from rescue_vision.app import cluster_breakup, manual_capture, match_runtime
    from rescue_vision.config import AppConfig
    from rescue_vision.motion import CarSystemStatus, OdometryImu, SensorFlags
    from rescue_vision import perception as perception_module
    from rescue_vision.communication import UartError
    import sys

    config = load_runtime_config("configs/runtime.match.yaml")
    channel = MagicMock()
    controller = MagicMock()
    controller.motion_synchronized = True
    controller.drain_messages.return_value = []
    status = MagicMock(spec=CarSystemStatus)
    status.watchdog_armed = True
    status.emergency_stop_latched = False

    def synchronize(**kwargs):
        consume = kwargs["on_message"]
        consume(status)
        for sample_us, count in ((1_000_000, 0), (1_020_000, 8)):
            consume(OdometryImu(
                uart_sequence=count, telemetry_sequence=count,
                received_timestamp_ns=1_000_000_000 + count,
                sample_timestamp_us=sample_us,
                left_encoder_count=count, right_encoder_count=count,
                gyro_x_urad_s=0, gyro_y_urad_s=0, gyro_z_urad_s=500_000,
                accel_x_mm_s2=0, accel_y_mm_s2=0, accel_z_mm_s2=9800,
                imu_temperature_cdeg=2500,
                sensor_flags=SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID,
            ))

    controller.synchronize.side_effect = synchronize
    monkeypatch.setattr(type(config.uart), "build_channel", lambda self: channel)
    monkeypatch.setattr(type(config.motion), "build_controller", lambda self, ch: controller)
    fusion_factory = MagicMock(side_effect=AssertionError("Fusion must not be constructed"))
    monkeypatch.setattr(AppConfig, "build_odometry_imu_fusion", fusion_factory)
    monkeypatch.setattr(AppConfig, "build_visual_localization_pipeline", fusion_factory)
    pipeline = SimpleNamespace(source=object(), prepare=lambda frame: frame, ground_projector=SimpleNamespace(supports_robot_projection=True))
    monkeypatch.setattr(manual_capture, "build_camera_pipeline", lambda cfg: pipeline)
    camera = MagicMock()
    monkeypatch.setattr(cluster_breakup, "CameraPerceptionPump", lambda *args: camera)
    renderer = MagicMock()
    renderer.latest_snapshot.return_value = snapshot(1, 10)
    renderer.latest_fresh_snapshot.return_value = snapshot(1, 10)
    renderer.latest.return_value = None
    monkeypatch.setattr(perception_module, "PerceptionFrameRenderer", lambda factory, **kwargs: renderer)
    def release_start_gate(**kwargs):
        kwargs["service"]()
        return match_runtime.time.monotonic_ns()

    start_gate = MagicMock(side_effect=release_start_gate)
    monkeypatch.setattr(match_runtime, "_wait_for_enter_start", start_gate)
    received = []

    def finish(self, timestamp_ns, **kwargs):
        received.append(kwargs)
        self.state = MatchState.FINISH_STOP
        return self._decision(timestamp_ns, 0.0, 0.0, "test_complete")

    monkeypatch.setattr(MatchSequence, "step", finish)
    if failure_phase is not None:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        original_error = UartError("injected startup UART failure")
        original_error.__cause__ = OSError("injected device failure")
        if failure_phase == "uart_open":
            channel.start.side_effect = original_error
        else:
            controller.synchronize.side_effect = original_error
            controller.soft_brake.side_effect = OSError("injected brake failure")
            channel.stop.side_effect = OSError("injected close failure")
        with pytest.raises(UartError) as raised:
            match_runtime._run_hardware(
                Path("configs/runtime.match.yaml"),
                supervised_stop_ready=True, log_dir=tmp_path,
            )
        assert raised.value is original_error
        text = next(tmp_path.glob("match_*.log")).read_text()
        assert "Traceback" in text
        assert "injected startup UART failure" in text
        assert "injected device failure" in text
        assert f"runtime_phase={failure_phase}" in text
        # Exception notes are displayed by traceback on Python 3.11+.
        assert f"match runtime_phase={failure_phase}" in original_error.__notes__
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        assert received == []
        if failure_phase == "uart_synchronize":
            notes = "\n".join(original_error.__notes__)
            assert "injected brake failure" in notes
            assert "injected close failure" in notes
            camera.stop.assert_called_once()
            channel.stop.assert_called_once()
        else:
            camera.start_in_background.assert_not_called()
        return
    match_runtime._run_hardware(
        Path("configs/runtime.match.yaml"),
        supervised_stop_ready=True, log_dir=None,
    )
    fusion_factory.assert_not_called()
    start_gate.assert_called_once()
    assert len(received) == 1
    assert received[0]["heading_rad"] == pytest.approx(-math.pi / 2 - 0.01)
    assert received[0]["cumulative_distance_m"] > 0
    camera.stop.assert_called_once()
    channel.stop.assert_called_once()


def test_start_gate_services_hardware_until_blank_line(monkeypatch) -> None:
    from unittest.mock import Mock

    from rescue_vision.app import match_runtime

    enter_checks = iter((False, False, True))
    monkeypatch.setattr(
        match_runtime,
        "_stdin_enter_pressed",
        lambda timeout_s: next(enter_checks),
    )
    monkeypatch.setattr(match_runtime.time, "monotonic_ns", lambda: 123_456_789)
    service = Mock()

    released_ns = match_runtime._wait_for_enter_start(
        service=service,
        should_stop=lambda: False,
    )

    assert released_ns == 123_456_789
    assert service.call_count == 2


def test_start_gate_ignores_nonblank_terminal_line(monkeypatch, capsys) -> None:
    from io import StringIO

    from rescue_vision.app import match_runtime

    terminal_input = StringIO("start\n\n")
    monkeypatch.setattr(match_runtime.sys, "stdin", terminal_input)
    monkeypatch.setattr(
        match_runtime.select,
        "select",
        lambda readable, writable, exceptional, timeout: (
            readable,
            writable,
            exceptional,
        ),
    )

    assert not match_runtime._stdin_enter_pressed(0.0)
    assert "请直接按 Enter" in capsys.readouterr().out
    assert match_runtime._stdin_enter_pressed(0.0)
