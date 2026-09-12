"""``match_nb`` 简单开场的纯逻辑回归。"""

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
from rescue_vision.config import MatchRuntimeConfig, load_runtime_config
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import normalize_angle
from rescue_vision.tracking import MultiTargetTracker, TrackingConfig
from rescue_vision.world import TeamColor

START_FIELD_POSITION = FieldPoint(1350.0, 1350.0)
START_HEADING_RAD = -math.pi / 2.0
CONTROL_PERIOD_S = 0.02


def runtime_config(**overrides: object) -> MatchRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "nb_opening_first_target_field": FieldPoint(100.0, 800.0),
        "nb_opening_first_speed_m_s": 0.60,
        "nb_opening_gripper_left_deg": 50.0,
        "nb_opening_gripper_right_deg": 130.0,
        "nb_opening_second_target_field": FieldPoint(100.0, -900.0),
        "nb_opening_second_speed_m_s": 0.60,
        "nb_opening_reverse_target_field": FieldPoint(100.0, 0.0),
        "nb_opening_reverse_speed_m_s": 0.60,
        "nb_opening_align_tolerance_mm": 30.0,
        "nb_opening_heading_tolerance_rad": 0.08,
        "nb_opening_heading_kp_rad_s": 3.0,
        "nb_opening_heading_max_angular_velocity_rad_s": 0.30,
        "nb_opening_align_angular_velocity_rad_s": 1.20,
        "nb_opening_settle_time_s": 0.30,
        "nb_opening_align_timeout_s": 8.0,
    }
    values.update(overrides)
    return MatchRuntimeConfig(**values)  # type: ignore[arg-type]


def make_nb(*, config: MatchRuntimeConfig | None = None) -> MatchNBSequence:
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
        initial_field_position=START_FIELD_POSITION,
    )


def start_nb(sequence: MatchNBSequence) -> None:
    checks = MatchPreflight(True, True, True, True, True, True)
    assert sequence.preflight(0, checks).state is MatchState.PREFLIGHT
    assert sequence.start(1).state is MatchState.NB_OPENING_TO_FIRST


def drive_opening(
    sequence: MatchNBSequence,
    *,
    fixed_heading: float | None = None,
) -> list[dict[str, object]]:
    """按决策积分理想差速运动，直到开场结束或停车。"""

    heading = START_HEADING_RAD
    distance_m = 0.0
    timestamp_ns = 1
    step_ns = round(CONTROL_PERIOD_S * 1_000_000_000)
    records: list[dict[str, object]] = []
    for _ in range(round(60.0 / CONTROL_PERIOD_S)):
        decision = sequence.step(
            timestamp_ns,
            perception=None,
            heading_rad=heading if fixed_heading is None else fixed_heading,
            cumulative_distance_m=distance_m,
        )
        position = sequence.estimated_field_position
        records.append(
            {
                "state": decision.state,
                "reason": decision.reason,
                "linear": decision.linear_velocity_m_s,
                "angular": decision.angular_velocity_rad_s,
                "posture": decision.gripper_posture,
                "angles": decision.gripper_angles_deg,
                "position": position,
            }
        )
        if decision.state in {MatchState.SEARCH_CLUSTER, MatchState.TERMINAL_STOP}:
            break
        heading = normalize_angle(
            heading + decision.angular_velocity_rad_s * CONTROL_PERIOD_S
        )
        distance_m += decision.linear_velocity_m_s * CONTROL_PERIOD_S
        timestamp_ns += step_ns
    return records


def reasons(records: list[dict[str, object]]) -> list[str]:
    return [str(record["reason"]) for record in records]


def test_nb_config_contains_requested_waypoints() -> None:
    config = load_runtime_config("configs/runtime.match_nb.yaml").match
    assert config.nb_opening_first_target_field == FieldPoint(100.0, 800.0)
    assert config.nb_opening_second_target_field == FieldPoint(100.0, -900.0)
    assert config.nb_opening_reverse_target_field == FieldPoint(100.0, 0.0)
    assert config.nb_opening_gripper_left_deg == pytest.approx(50.0)
    assert config.nb_opening_gripper_right_deg == pytest.approx(130.0)


def test_nb_defaults_match_shipped_opening() -> None:
    defaults = MatchRuntimeConfig(enabled=True)
    shipped = load_runtime_config("configs/runtime.match_nb.yaml").match
    for name in (
        "nb_opening_first_target_field",
        "nb_opening_second_target_field",
        "nb_opening_reverse_target_field",
        "nb_opening_first_speed_m_s",
        "nb_opening_second_speed_m_s",
        "nb_opening_reverse_speed_m_s",
        "nb_opening_gripper_left_deg",
        "nb_opening_gripper_right_deg",
        "nb_opening_align_tolerance_mm",
        "nb_opening_heading_tolerance_rad",
        "nb_opening_heading_kp_rad_s",
        "nb_opening_heading_max_angular_velocity_rad_s",
        "nb_opening_align_angular_velocity_rad_s",
        "nb_opening_settle_time_s",
        "nb_opening_align_timeout_s",
    ):
        assert getattr(defaults, name) == getattr(shipped, name), name


def test_nb_runtime_is_a_copy_of_current_match_runtime() -> None:
    """除入口选择外，NB 配置必须跟随当前正式 match，不保留旧策略参数。"""

    assert load_runtime_config("configs/runtime.match_nb.yaml") == load_runtime_config(
        "configs/runtime.match.yaml"
    )


def test_nb_opening_runs_requested_actions_in_order() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    seen = reasons(records)

    ordered = [
        "nb_opening_turn_to_first",
        "nb_opening_to_first",
        "nb_opening_first_reached_open_gripper",
        "nb_opening_gripper_opened",
        "nb_opening_turn_to_second",
        "nb_opening_to_second",
        "nb_opening_second_reached_start_reverse",
        "nb_opening_reverse",
        "nb_opening_reverse_reached_start_search",
    ]
    indices = [seen.index(reason) for reason in ordered]
    assert indices == sorted(indices)
    assert records[-1]["state"] is MatchState.SEARCH_CLUSTER


def test_nb_turns_only_before_the_two_forward_legs() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    seen = reasons(records)

    assert seen.count("nb_opening_turn_to_first") > 0
    assert seen.count("nb_opening_turn_to_second") > 0
    assert not any("reverse_turn" in reason for reason in seen)
    for record in records:
        if str(record["reason"]).startswith("nb_opening_turn_to_"):
            assert record["linear"] == 0.0


def test_nb_opens_gripper_at_first_waypoint_and_keeps_it_open() -> None:
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


def test_nb_reverse_is_negative_and_has_no_extra_stop_calibration() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    reverse = [r for r in records if r["reason"] == "nb_opening_reverse"]

    assert reverse
    assert all(float(r["linear"]) < 0.0 for r in reverse)
    assert not any(str(r["reason"]).endswith("_align") for r in records)


def test_nb_reaches_requested_waypoints() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    target_by_reason = {
        "nb_opening_first_reached_open_gripper": FieldPoint(100.0, 800.0),
        "nb_opening_second_reached_start_reverse": FieldPoint(100.0, -900.0),
        "nb_opening_reverse_reached_start_search": FieldPoint(100.0, 0.0),
    }
    tolerance = sequence.config.nb_opening_align_tolerance_mm + 15.0
    for reason, target in target_by_reason.items():
        record = records[reasons(records).index(reason)]
        position = record["position"]
        assert isinstance(position, FieldPoint)
        assert math.hypot(position.x - target.x, position.y - target.y) <= tolerance


def test_nb_turn_timeout_stops_without_translation() -> None:
    sequence = make_nb(config=runtime_config(nb_opening_align_timeout_s=0.2))
    start_nb(sequence)
    records = drive_opening(sequence, fixed_heading=START_HEADING_RAD)

    assert records[-1]["state"] is MatchState.TERMINAL_STOP
    assert records[-1]["reason"] == "nb_opening_turn_timeout_stop"
    assert all(record["linear"] == 0.0 for record in records)


def test_nb_diagnostic_reports_current_target() -> None:
    sequence = make_nb()
    start_nb(sequence)
    sequence.step(1, perception=None, heading_rad=START_HEADING_RAD, cumulative_distance_m=0.0)
    diagnostic = sequence.nb_opening_diagnostic
    assert diagnostic is not None
    assert "phase=turn_and_move_to_first" in diagnostic
    assert "target=(+100,+800)mm" in diagnostic
    assert "position=(+1350,+1350)mm" in diagnostic


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
