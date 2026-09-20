"""``match_nb`` 相对动作序列的纯逻辑回归。"""

from __future__ import annotations

import math
import sys

import pytest

from rescue_vision.app import (
    GripperPosture,
    MatchNBSequence,
    MatchPreflight,
    MatchState,
)
from rescue_vision.config import (
    MatchRuntimeConfig,
    NBOpeningStraight,
    NBOpeningTurn,
    load_runtime_config,
)
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import normalize_angle
from rescue_vision.tracking import MultiTargetTracker, TrackingConfig
from rescue_vision.world import TeamColor
from test_gripper_width_sequence import motion_sample


START_FIELD_POSITION = FieldPoint(1350.0, 1350.0)
START_HEADING_RAD = -math.pi / 2.0
CONTROL_PERIOD_S = 0.02


def runtime_config(**overrides: object) -> MatchRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "nb_opening_actions": (
            NBOpeningTurn(-0.40, 0.50),
            NBOpeningStraight(0.30, 0.40),
            NBOpeningTurn(0.30, 0.60),
            NBOpeningStraight(0.25, 0.30),
            NBOpeningStraight(-0.10, 0.20),
        ),
        "nb_opening_gripper_after_action": 2,
        "nb_opening_gripper_left_deg": 50.0,
        "nb_opening_gripper_right_deg": 130.0,
        "nb_opening_turn_tolerance_rad": 0.02,
        "nb_opening_distance_tolerance_m": 0.01,
        "nb_opening_settle_time_s": 0.06,
        "nb_opening_turn_timeout_s": 8.0,
    }
    values.update(overrides)
    return MatchRuntimeConfig(**values)  # type: ignore[arg-type]


def make_nb(
    *,
    config: MatchRuntimeConfig | None = None,
    initial_field_position: FieldPoint = START_FIELD_POSITION,
) -> MatchNBSequence:
    return MatchNBSequence(
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


def start_nb(sequence: MatchNBSequence) -> None:
    checks = MatchPreflight(True, True, True, True, True, True)
    assert sequence.preflight(0, checks).state is MatchState.PREFLIGHT
    assert sequence.start(1).state is MatchState.NB_OPENING_SEQUENCE


def drive_opening(
    sequence: MatchNBSequence,
    *,
    fixed_heading: float | None = None,
    max_seconds: float = 60.0,
) -> list[dict[str, object]]:
    """用带编码器/IMU延迟的最小车辆模型回放开场。"""

    heading = START_HEADING_RAD
    distance_m = 0.0
    measured_linear = 0.0
    measured_angular = 0.0
    encoder_count = 0
    timestamp_ns = 1
    step_ns = round(CONTROL_PERIOD_S * 1_000_000_000)
    records: list[dict[str, object]] = []
    for _ in range(round(max_seconds / CONTROL_PERIOD_S)):
        heading = normalize_angle(
            heading + measured_angular * CONTROL_PERIOD_S
        )
        distance_m += measured_linear * CONTROL_PERIOD_S
        encoder_count += round(measured_linear * CONTROL_PERIOD_S * 10_000)
        sequence.observe_grasp_motion(
            motion_sample(
                timestamp_ns,
                count=encoder_count,
                gyro=round(measured_angular * 1_000_000),
            )
        )
        left_feedback = measured_linear - measured_angular * 0.235 / 2.0
        right_feedback = measured_linear + measured_angular * 0.235 / 2.0
        decision = sequence.step(
            timestamp_ns,
            perception=None,
            heading_rad=heading if fixed_heading is None else fixed_heading,
            cumulative_distance_m=distance_m,
            left_speed_feedback_m_s=left_feedback,
            right_speed_feedback_m_s=right_feedback,
        )
        records.append(
            {
                "state": decision.state,
                "reason": decision.reason,
                "linear": decision.linear_velocity_m_s,
                "angular": decision.angular_velocity_rad_s,
                "posture": decision.gripper_posture,
                "angles": decision.gripper_angles_deg,
                "position": sequence.estimated_field_position,
            }
        )
        if decision.state in {MatchState.SEARCH_CLUSTER, MatchState.TERMINAL_STOP}:
            break
        measured_linear = decision.linear_velocity_m_s
        measured_angular = decision.angular_velocity_rad_s
        timestamp_ns += step_ns
    return records


def reasons(records: list[dict[str, object]]) -> list[str]:
    return [str(record["reason"]) for record in records]


def test_nb_config_contains_relative_actions() -> None:
    config = load_runtime_config("configs/runtime.match_nb.yaml").match
    assert config.nb_opening_actions == (
        NBOpeningTurn(-1.0, 2.0),
        NBOpeningStraight(1.3, 1.5),
        NBOpeningTurn(0.89, 2.0),
        NBOpeningStraight(0.9, 1.5),
        NBOpeningStraight(-0.9, 1.5),
    )
    assert config.nb_opening_gripper_after_action == 2
    assert config.nb_opening_turn_tolerance_rad == pytest.approx(0.08)
    assert config.nb_opening_distance_tolerance_m == pytest.approx(0.03)
    assert config.nb_opening_effective_linear_deceleration_m_s2 == pytest.approx(2.0)


def test_nb_defaults_match_shipped_relative_opening() -> None:
    defaults = MatchRuntimeConfig(enabled=True)
    shipped = load_runtime_config("configs/runtime.match_nb.yaml").match
    for name in (
        "nb_opening_execution_response_s",
        "nb_opening_max_telemetry_age_ms",
        "nb_opening_fine_linear_speed_m_s",
        "nb_opening_fine_angular_velocity_rad_s",
        "nb_opening_stop_wheel_speed_m_s",
        "nb_opening_stop_angular_velocity_rad_s",
        "nb_opening_heading_tolerance_rad",
        "nb_opening_heading_kp_rad_s",
        "nb_opening_heading_max_angular_velocity_rad_s",
        "nb_opening_correction_max_distance_m",
        "nb_opening_correction_max_angle_rad",
        "nb_opening_correction_timeout_s",
    ):
        assert getattr(defaults, name) == getattr(shipped, name), name


def test_nb_uses_conservative_linear_deceleration_without_vehicle_calibration() -> None:
    sequence = make_nb()
    start_nb(sequence)

    limits = sequence.motion_acceleration_limits

    assert limits is not None
    assert limits.linear_deceleration_m_s2 == pytest.approx(2.0)


def test_nb_opening_runs_configured_actions_in_order() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    seen = reasons(records)

    ordered = [
        "nb_opening_turn_1_",
        "nb_opening_straight_2_",
        "nb_opening_gripper_opened",
        "nb_opening_turn_3_",
        "nb_opening_straight_4_",
        "nb_opening_straight_5_",
        "nb_opening_sequence_complete",
    ]
    indices = [
        next(i for i, reason in enumerate(seen) if reason.startswith(prefix))
        for prefix in ordered
    ]
    assert indices == sorted(indices)
    assert records[-1]["state"] is MatchState.SEARCH_CLUSTER


def test_nb_never_drives_forward_before_turn_finishes() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)

    first_straight = next(
        i for i, reason in enumerate(reasons(records))
        if reason.startswith("nb_opening_straight_2_")
    )
    assert all(
        record["linear"] == 0.0 for record in records[:first_straight]
    )
    straight_records = [
        record
        for record in records[first_straight:]
        if str(record["reason"]).startswith("nb_opening_straight_")
    ]
    assert straight_records
    assert max(abs(float(record["angular"])) for record in straight_records) <= 0.25


def test_nb_uses_each_action_speed_and_signed_straight_distance() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)

    turn_1 = [r for r in records if str(r["reason"]).startswith("nb_opening_turn_1_")]
    turn_3 = [r for r in records if str(r["reason"]).startswith("nb_opening_turn_3_")]
    straight_2 = [r for r in records if str(r["reason"]).startswith("nb_opening_straight_2_")]
    straight_4 = [r for r in records if str(r["reason"]).startswith("nb_opening_straight_4_")]
    straight_5 = [r for r in records if str(r["reason"]).startswith("nb_opening_straight_5_")]
    assert turn_1 and turn_3 and straight_2 and straight_4 and straight_5
    assert min(float(r["angular"]) for r in turn_1) == pytest.approx(-0.50)
    assert max(float(r["angular"]) for r in turn_3) == pytest.approx(0.60)
    assert max(float(r["linear"]) for r in straight_2) == pytest.approx(0.40)
    assert max(float(r["linear"]) for r in straight_4) == pytest.approx(0.30)
    assert min(float(r["linear"]) for r in straight_5) == pytest.approx(-0.20)


def test_nb_opens_gripper_after_configured_action_and_keeps_it_open() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    opened = reasons(records).index("nb_opening_gripper_opened")

    assert all(
        record["posture"] is GripperPosture.CLOSED
        for record in records[:opened]
    )
    assert all(
        record["posture"] is GripperPosture.OPEN
        and record["angles"] == pytest.approx((50.0, 130.0))
        for record in records[opened:]
    )


def test_nb_relative_route_does_not_depend_on_field_position() -> None:
    sequence = make_nb(initial_field_position=FieldPoint(-900.0, 200.0))
    start_nb(sequence)
    records = drive_opening(sequence)

    assert records[-1]["state"] is MatchState.SEARCH_CLUSTER


def test_nb_turn_timeout_stops_without_translation() -> None:
    sequence = make_nb(
        config=runtime_config(
            nb_opening_actions=(NBOpeningTurn(0.5, 0.5),),
            nb_opening_gripper_after_action=None,
            nb_opening_turn_timeout_s=0.2,
        )
    )
    start_nb(sequence)
    records = drive_opening(sequence, fixed_heading=START_HEADING_RAD)

    assert records[-1]["state"] is MatchState.TERMINAL_STOP
    assert str(records[-1]["reason"]).startswith("nb_opening_turn_1_timeout")
    assert all(record["linear"] == 0.0 for record in records)


def test_nb_waits_for_heading_before_turn_command() -> None:
    sequence = make_nb()
    start_nb(sequence)
    waiting = sequence.step(
        2,
        perception=None,
        heading_rad=None,
        cumulative_distance_m=0.0,
    )

    assert waiting.reason.startswith("nb_opening_turn_waiting_heading:")
    assert waiting.linear_velocity_m_s == 0.0
    assert waiting.angular_velocity_rad_s == 0.0


def test_nb_missing_heading_timeout_stops_without_translation() -> None:
    sequence = make_nb(
        config=runtime_config(
            nb_opening_actions=(NBOpeningTurn(0.5, 0.5),),
            nb_opening_gripper_after_action=None,
            nb_opening_turn_timeout_s=0.2,
        )
    )
    start_nb(sequence)

    waiting = sequence.step(
        2,
        perception=None,
        heading_rad=None,
        cumulative_distance_m=0.0,
    )
    decision = sequence.step(
        2 + round(0.2 * 1_000_000_000),
        perception=None,
        heading_rad=None,
        cumulative_distance_m=0.0,
    )

    assert waiting.reason.startswith("nb_opening_turn_waiting_heading:")
    assert decision.state is MatchState.TERMINAL_STOP
    assert decision.reason.startswith("nb_opening_turn_timeout_stop")
    assert decision.linear_velocity_m_s == 0.0
    assert decision.angular_velocity_rad_s == 0.0


def test_nb_diagnostic_reports_action_progress() -> None:
    sequence = make_nb()
    start_nb(sequence)
    sequence.step(
        1,
        perception=None,
        heading_rad=START_HEADING_RAD,
        cumulative_distance_m=0.0,
    )
    diagnostic = sequence.nb_opening_diagnostic
    assert diagnostic is not None
    assert "action=1/5" in diagnostic
    assert "angle=-0.400rad" in diagnostic
    assert "speed=0.500rad/s" in diagnostic


def test_nb_cli_uses_current_match_subclass(monkeypatch) -> None:
    import rescue_vision.app.match_nb as match_nb_module

    received: dict[str, object] = {}
    monkeypatch.setattr(
        match_nb_module,
        "_run_hardware",
        lambda *args, **kwargs: received.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["rescue-vision-match-nb", "--config", "configs/runtime.match_nb.yaml"],
    )

    match_nb_module.main()

    factory = received["sequence_factory"]
    assert factory.__self__ is MatchNBSequence
    assert factory.__name__ == "from_app_config"
    assert received["mode_name"] == "match_nb"
