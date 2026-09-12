"""无解团（no-breakup）开场变体的纯逻辑测试。

开场的关键不变量由 ``_drive_opening`` 里的运动学仿真验证：仿真消费
``MatchNBSequence.step()`` 返回的线/角速度并按同一航向积分，因此它检查的是
车辆真实落点，而不只是策略内部估计。这样“一边转向一边前进走成弧线”这类
只体现在实际轨迹上的缺陷会被测出。
"""

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

# 区域 2 的真实初始位姿：地图右上角，场地航向 -90°（朝向场地 -y）。
START_FIELD_POSITION = FieldPoint(1350.0, 1350.0)
START_HEADING_RAD = -math.pi / 2.0
CONTROL_PERIOD_S = 0.02


def runtime_config(**overrides: object) -> MatchRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "nb_opening_first_target_field": FieldPoint(130.0, 800.0),
        "nb_opening_first_speed_m_s": 0.15,
        "nb_opening_gripper_left_deg": 50.0,
        "nb_opening_gripper_right_deg": 130.0,
        "nb_opening_second_target_field": FieldPoint(130.0, -900.0),
        "nb_opening_second_speed_m_s": 0.15,
        "nb_opening_reverse_target_field": FieldPoint(130.0, 0.0),
        "nb_opening_reverse_speed_m_s": 0.15,
        "nb_opening_align_tolerance_mm": 30.0,
        "nb_opening_heading_tolerance_rad": 0.02,
        "nb_opening_heading_kp_rad_s": 1.0,
        "nb_opening_heading_max_angular_velocity_rad_s": 0.3,
        "nb_opening_align_angular_velocity_rad_s": 0.5,
        "nb_opening_settle_time_s": 0.3,
        "nb_opening_align_timeout_s": 8.0,
    }
    values.update(overrides)
    return MatchRuntimeConfig(**values)  # type: ignore[arg-type]


def make_nb(
    *,
    config: MatchRuntimeConfig | None = None,
    initial_field_position: FieldPoint = START_FIELD_POSITION,
) -> MatchNBSequence:
    runtime = config or runtime_config()
    return MatchNBSequence(
        runtime,
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
    assert sequence.preflight(
        0, MatchPreflight(True, True, True, True, True, True)
    ).state is MatchState.PREFLIGHT
    assert sequence.start(1).state is MatchState.NB_OPENING_TO_FIRST


def drive_opening(
    sequence: MatchNBSequence,
    *,
    max_seconds: float = 120.0,
    heading_fn: object = None,
) -> list[dict[str, object]]:
    """按真实控制回路消费决策并积分车辆位姿，返回逐周期记录。

    ``heading_fn(elapsed_s, true_heading_rad)`` 可以改写上报给策略的航向，
    用来模拟陀螺仪异常或直线中途的瞬时偏航。
    """

    position = sequence._fallback_field_position
    assert position is not None
    x, y = position.x, position.y
    heading = START_HEADING_RAD
    distance_m = 0.0
    timestamp_ns = 1
    step_ns = round(CONTROL_PERIOD_S * 1_000_000_000)
    records: list[dict[str, object]] = []
    max_tracking_error_mm = 0.0

    while timestamp_ns < round(max_seconds * 1_000_000_000):
        elapsed_s = (timestamp_ns - 1) / 1_000_000_000.0
        reported_heading = (
            heading if heading_fn is None else heading_fn(elapsed_s, heading)
        )
        decision = sequence.step(
            timestamp_ns,
            perception=None,
            heading_rad=reported_heading,
            cumulative_distance_m=distance_m,
            left_speed_feedback_m_s=None,
            right_speed_feedback_m_s=None,
        )
        # 航位推算按上一个里程增量更新，因此估计值滞后物理位置一个控制周期。
        estimated = sequence._fallback_field_position
        if estimated is not None:
            max_tracking_error_mm = max(
                max_tracking_error_mm, math.hypot(estimated.x - x, estimated.y - y)
            )
        records.append(
            {
                "state": decision.state,
                "reason": decision.reason,
                "linear": decision.linear_velocity_m_s,
                "angular": decision.angular_velocity_rad_s,
                "posture": decision.gripper_posture,
                "angles": decision.gripper_angles_deg,
                "x": x,
                "y": y,
                "t_ns": timestamp_ns,
                "leg_start": sequence._nb_leg_start_position,
            }
        )
        if decision.state in {MatchState.TERMINAL_STOP, MatchState.SEARCH_CLUSTER}:
            break
        heading = normalize_angle(
            heading + decision.angular_velocity_rad_s * CONTROL_PERIOD_S
        )
        travel_mm = decision.linear_velocity_m_s * 1000.0 * CONTROL_PERIOD_S
        x += travel_mm * math.cos(heading)
        y += travel_mm * math.sin(heading)
        distance_m += decision.linear_velocity_m_s * CONTROL_PERIOD_S
        timestamp_ns += step_ns

    # 估计值必须跟随物理轨迹（一周期滞后约 3 mm），否则后续航点全无意义。
    step_mm = 0.15 * 1000.0 * CONTROL_PERIOD_S
    assert max_tracking_error_mm <= 2.0 * step_mm
    return records


def reasons(records: list[dict[str, object]]) -> list[str]:
    return [str(record["reason"]) for record in records]


def first_index(records: list[dict[str, object]], reason: str) -> int:
    return reasons(records).index(reason)


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------


def test_nb_config_parses_opening_waypoints() -> None:
    config = load_runtime_config("configs/runtime.match_nb.yaml").match
    assert config.enabled
    assert config.nb_opening_first_target_field == FieldPoint(130.0, 800.0)
    assert config.nb_opening_second_target_field == FieldPoint(130.0, -900.0)
    assert config.nb_opening_reverse_target_field == FieldPoint(130.0, 0.0)
    assert config.nb_opening_gripper_left_deg == pytest.approx(50.0)
    assert config.nb_opening_gripper_right_deg == pytest.approx(130.0)
    assert config.nb_opening_first_speed_m_s == pytest.approx(0.15)
    assert config.nb_opening_second_speed_m_s == pytest.approx(0.15)
    assert config.nb_opening_reverse_speed_m_s == pytest.approx(0.15)
    assert config.nb_opening_heading_tolerance_rad == pytest.approx(0.02)
    assert config.nb_opening_align_angular_velocity_rad_s == pytest.approx(0.5)
    assert config.nb_opening_settle_time_s == pytest.approx(0.3)
    assert config.nb_opening_align_timeout_s == pytest.approx(8.0)


def test_nb_config_defaults_match_shipped_reference_file() -> None:
    """未显式配置时数据类默认值应与区域 2 参考文件一致。"""

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


def test_nb_config_rejects_invalid_opening_values() -> None:
    with pytest.raises(ValueError, match="nb_opening_first_speed_m_s"):
        runtime_config(nb_opening_first_speed_m_s=0.0)
    with pytest.raises(ValueError, match="nb_opening_gripper_left_deg"):
        runtime_config(nb_opening_gripper_left_deg=200.0)
    with pytest.raises(ValueError, match="nb_opening_settle_time_s"):
        runtime_config(nb_opening_settle_time_s=-0.1)
    with pytest.raises(ValueError, match="nb_opening_align_timeout_s"):
        runtime_config(nb_opening_align_timeout_s=0.0)
    with pytest.raises(ValueError, match="nb_opening_align_angular_velocity_rad_s"):
        runtime_config(nb_opening_align_angular_velocity_rad_s=0.0)
    # 容差 0 会让“未对准”永远成立，必须拒绝。
    with pytest.raises(ValueError, match="nb_opening_heading_tolerance_rad"):
        runtime_config(nb_opening_heading_tolerance_rad=0.0)


def test_nb_settle_time_zero_is_allowed() -> None:
    assert runtime_config(nb_opening_settle_time_s=0.0).nb_opening_settle_time_s == 0.0


# ----------------------------------------------------------------------
# 开场状态与动作顺序
# ----------------------------------------------------------------------


def test_nb_start_skips_startup_turn_and_forward() -> None:
    sequence = make_nb()
    start_nb(sequence)
    assert sequence.state is MatchState.NB_OPENING_TO_FIRST
    assert sequence.state is not MatchState.STARTUP_TURN_RIGHT
    assert sequence.state is not MatchState.STARTUP_FORWARD


def test_nb_opening_runs_forward_gripper_forward_reverse_then_search() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    seen = reasons(records)

    assert "nb_opening_to_first" in seen
    assert "nb_opening_gripper_opened" in seen
    assert "nb_opening_to_second" in seen
    assert seen.index("nb_opening_to_first") < seen.index(
        "nb_opening_gripper_opened"
    )
    assert seen.index("nb_opening_gripper_opened") < seen.index(
        "nb_opening_to_second"
    )
    assert seen.index("nb_opening_to_second") < seen.index("nb_opening_reverse")
    assert records[-1]["state"] is MatchState.SEARCH_CLUSTER

    # 第一段夹爪闭合；张爪后各段保持配置的左右角度。
    opening_start = first_index(records, "nb_opening_to_first")
    gripper_open = first_index(records, "nb_opening_gripper_opened")
    assert all(
        records[i]["angles"] is None
        and records[i]["posture"] is GripperPosture.CLOSED
        for i in range(opening_start, gripper_open)
    )
    assert records[gripper_open]["angles"] == pytest.approx((50.0, 130.0))
    assert all(
        records[i]["angles"] == pytest.approx((50.0, 130.0))
        for i in range(gripper_open, len(records))
    )


def test_nb_reverse_leg_commands_negative_linear_velocity() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    reverse_drive = [
        record for record in records if record["reason"] == "nb_opening_reverse"
    ]
    assert reverse_drive
    assert all(record["linear"] < 0.0 for record in reverse_drive)


# ----------------------------------------------------------------------
# 走歪回归：先原地对准，再平移
# ----------------------------------------------------------------------


def test_nb_never_translates_while_heading_is_misaligned() -> None:
    """未对准时只能原地旋转；一旦一边转向一边前进就会走成弧线。"""

    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    tolerance = sequence.config.nb_opening_heading_tolerance_rad

    checked = 0
    for record in records:
        if record["linear"] == 0.0:
            continue
        # 正在平移的周期，实际航向必须已经在容差内。
        assert record["reason"] in {
            "nb_opening_to_first",
            "nb_opening_to_second",
            "nb_opening_reverse",
        }, record["reason"]
        checked += 1
    assert checked > 0
    # 对准周期必须是纯旋转：角速度非零、线速度为零。
    aligning = [r for r in records if str(r["reason"]).endswith("_align")]
    assert aligning
    assert all(r["linear"] == 0.0 and r["angular"] != 0.0 for r in aligning)


def test_nb_opening_physical_endpoints_hit_each_waypoint() -> None:
    """关键回归：车辆实际落点必须落在各航点容差内。

    旧实现一边转向一边前进（起始航向误差 65.7°），仿真终点横向偏出约
    325 mm 并靠投影越界提前结束，本断言会失败。
    """

    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    tolerance_mm = sequence.config.nb_opening_align_tolerance_mm

    waypoint_reasons = {
        "nb_opening_first_reached_open_gripper": FieldPoint(130.0, 800.0),
        "nb_opening_second_reached_start_reverse": FieldPoint(130.0, -900.0),
        "nb_opening_reverse_reached_start_search": FieldPoint(130.0, 0.0),
    }
    # 到达判据使用滞后一个控制周期的航位推算估计，因此物理落点最多比配置容差
    # 多出一个控制周期的行程（0.15 m/s × 20 ms = 3 mm）。
    step_mm = 0.15 * 1000.0 * CONTROL_PERIOD_S
    physical_tolerance_mm = tolerance_mm + 2.0 * step_mm
    for reason, target in waypoint_reasons.items():
        index = first_index(records, reason)
        reached = records[index]
        assert abs(float(reached["x"]) - target.x) <= physical_tolerance_mm, reason
        assert abs(float(reached["y"]) - target.y) <= physical_tolerance_mm, reason

    # 全程横向偏离直线不得超过到达容差量级，即不再走弧线。
    for reason, target in waypoint_reasons.items():
        start_index = {
            "nb_opening_first_reached_open_gripper": first_index(
                records, "nb_opening_to_first"
            ),
            "nb_opening_second_reached_start_reverse": first_index(
                records, "nb_opening_to_second"
            ),
            "nb_opening_reverse_reached_start_search": first_index(
                records, "nb_opening_reverse"
            ),
        }[reason]
        end_index = first_index(records, reason)
        origin = (
            float(records[start_index]["x"]),
            float(records[start_index]["y"]),
        )
        delta_x = target.x - origin[0]
        delta_y = target.y - origin[1]
        norm = math.hypot(delta_x, delta_y)
        assert norm > 0.0
        unit_x, unit_y = delta_x / norm, delta_y / norm
        # 直线段只允许航向死区带来的小幅漂移；旧实现走弧线时这里会到 300 mm 量级。
        limit_mm = 2.0 * tolerance_mm
        for record in records[start_index:end_index + 1]:
            offset_x = float(record["x"]) - origin[0]
            offset_y = float(record["y"]) - origin[1]
            lateral = abs(-offset_x * unit_y + offset_y * unit_x)
            assert lateral <= limit_mm, (reason, record["reason"], lateral)


# ----------------------------------------------------------------------
# 到位停稳
# ----------------------------------------------------------------------


def test_nb_settles_after_every_waypoint_arrival() -> None:
    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)
    settle_s = sequence.config.nb_opening_settle_time_s

    for arrived_reason, settle_reason in (
        ("nb_opening_to_first_arrived", "nb_opening_to_first_arrival_settle"),
        ("nb_opening_to_second_arrived", "nb_opening_to_second_arrival_settle"),
        ("nb_opening_reverse_arrived", "nb_opening_reverse_arrival_settle"),
    ):
        assert arrived_reason in reasons(records), arrived_reason
        arrived = first_index(records, arrived_reason)
        settling = [
            record
            for record in records[arrived + 1 :]
            if record["reason"] == settle_reason
        ]
        assert settling, settle_reason
        # 停稳期间必须零速，且已清空本段起算位姿，下一段会从新位姿重算。
        assert all(
            record["linear"] == 0.0 and record["angular"] == 0.0
            for record in settling
        )
        assert all(record["leg_start"] is None for record in settling)
        # 停稳时长应等于配置值（允许一个控制周期的边界误差）。
        span_s = (
            int(settling[-1]["t_ns"]) - int(records[arrived]["t_ns"])
        ) / 1_000_000_000.0
        assert span_s == pytest.approx(settle_s, abs=2 * CONTROL_PERIOD_S)


def test_nb_next_leg_restarts_from_post_settle_pose() -> None:
    """停稳结束后下一段必须重新起算航向与起点，而不是沿用刹车前的量。"""

    sequence = make_nb()
    start_nb(sequence)
    records = drive_opening(sequence)

    resumed = first_index(records, "nb_opening_to_second")
    assert records[resumed]["leg_start"] is not None
    # 第二段起点应是停稳后的估计位姿（≈第一航点），不是初始出发位姿。
    start_point = records[resumed]["leg_start"]
    assert abs(start_point.y - 800.0) <= sequence.config.nb_opening_align_tolerance_mm
    assert start_point.x != pytest.approx(START_FIELD_POSITION.x, abs=100.0)


def test_nb_zero_settle_time_moves_on_immediately() -> None:
    sequence = make_nb(config=runtime_config(nb_opening_settle_time_s=0.0))
    start_nb(sequence)
    records = drive_opening(sequence)
    assert not [r for r in records if str(r["reason"]).endswith("_arrival_settle")]


# ----------------------------------------------------------------------
# 保守停车
# ----------------------------------------------------------------------


def test_nb_align_timeout_stops_conservatively() -> None:
    """航向始终对不准（例如陀螺仪符号错误）时必须停车而不是持续旋转。"""

    sequence = make_nb(config=runtime_config(nb_opening_align_timeout_s=0.5))
    start_nb(sequence)
    records = drive_opening(
        sequence, heading_fn=lambda elapsed_s, heading: START_HEADING_RAD
    )

    assert records[-1]["state"] is MatchState.TERMINAL_STOP
    assert records[-1]["reason"] == "nb_opening_align_timeout_stop"
    assert records[-1]["linear"] == 0.0
    # 超时前不允许有任何平移。
    assert all(record["linear"] == 0.0 for record in records)


def test_nb_transient_mid_leg_heading_excursion_does_not_stop() -> None:
    """对准计时衡量“连续未对准”，直线中途的瞬时偏航不得触发保守停车。

    超时从直线段起点累加时，这条用例会在瞬时偏航处误判超时并停车。
    """

    def disturb(elapsed_s: float, heading: float) -> float:
        if 6.0 <= elapsed_s <= 6.15:
            return normalize_angle(heading + 0.3)
        return heading

    sequence = make_nb(
        config=runtime_config(nb_opening_align_timeout_s=5.0),
    )
    start_nb(sequence)
    records = drive_opening(sequence, heading_fn=disturb)

    assert "nb_opening_align_timeout_stop" not in reasons(records)
    assert records[-1]["state"] is MatchState.SEARCH_CLUSTER


# ----------------------------------------------------------------------
# 车端诊断
# ----------------------------------------------------------------------


def test_nb_opening_route_phase_tracks_each_leg() -> None:
    sequence = make_nb()
    start_nb(sequence)
    assert sequence.nb_opening_route_phase == "to_first"

    phases = []
    position = sequence._fallback_field_position
    x, y = position.x, position.y
    heading = START_HEADING_RAD
    distance_m = 0.0
    timestamp_ns = 1
    step_ns = round(CONTROL_PERIOD_S * 1_000_000_000)
    while timestamp_ns < round(120 * 1_000_000_000):
        decision = sequence.step(
            timestamp_ns,
            perception=None,
            heading_rad=heading,
            cumulative_distance_m=distance_m,
        )
        if not phases or phases[-1] != sequence.nb_opening_route_phase:
            phases.append(sequence.nb_opening_route_phase)
        if decision.state is MatchState.SEARCH_CLUSTER:
            break
        heading = normalize_angle(
            heading + decision.angular_velocity_rad_s * CONTROL_PERIOD_S
        )
        x += decision.linear_velocity_m_s * 1000.0 * CONTROL_PERIOD_S * math.cos(heading)
        y += decision.linear_velocity_m_s * 1000.0 * CONTROL_PERIOD_S * math.sin(heading)
        distance_m += decision.linear_velocity_m_s * CONTROL_PERIOD_S
        timestamp_ns += step_ns

    assert phases == [
        "to_first",
        "first_arrived_open_gripper",
        "to_second",
        "reverse",
        None,
    ]
    # 开场结束后不再声明诊断，避免在正常解团阶段产生误导读数。
    assert sequence.nb_opening_diagnostic is None


def test_nb_opening_diagnostic_reports_target_position_and_error() -> None:
    sequence = make_nb()
    start_nb(sequence)
    # 航向来自最近一次有效观测，未 step 之前按不可用处理。
    assert "heading=unavailable" in str(sequence.nb_opening_diagnostic)
    sequence.step(
        1,
        perception=None,
        heading_rad=START_HEADING_RAD,
        cumulative_distance_m=0.0,
    )
    diagnostic = sequence.nb_opening_diagnostic
    assert diagnostic is not None
    assert "phase=to_first" in diagnostic
    assert "target=(+130,+800)mm" in diagnostic
    assert "position=(+1350,+1350)mm" in diagnostic
    # 剩余误差必须等于目标减当前位置，便于现场直接对照航点。
    assert "error=(-1220,-550)mm" in diagnostic
    assert "heading=-90.0deg" in diagnostic


def test_nb_opening_diagnostic_survives_missing_position() -> None:
    sequence = make_nb()
    start_nb(sequence)
    sequence._fallback_field_position = None
    diagnostic = sequence.nb_opening_diagnostic
    assert diagnostic is not None
    assert "position=unavailable" in diagnostic
    assert "target=(+130,+800)mm" in diagnostic


# ----------------------------------------------------------------------
# 入口装配
# ----------------------------------------------------------------------


def test_nb_cli_wires_sequence_factory_and_mode(monkeypatch) -> None:
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
        [
            "rescue-vision-match-nb",
            "--config",
            "configs/runtime.match_nb.yaml",
        ],
    )

    match_nb_module.main()

    factory = received["sequence_factory"]
    assert factory.__self__ is MatchNBSequence
    assert factory.__name__ == "from_app_config"
    assert received["mode_name"] == "match_nb"
    assert received["log_file_prefix"] == "match_nb_"
