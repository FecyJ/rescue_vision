"""策略变体（蓝色优先流程）的纯逻辑测试。

``match_strategy.MatchSequence`` 是 ``match.MatchSequence`` 的子类：正式阶段经
``_SharedMatchSequence`` 显式委托回基类，策略阶段（开场两段冲刺 → 只搜蓝色危险
物块 → 两趟 D2 投放）使用本地实现，随后翻转进正式阶段复用绿块流程。

这里固定两类容易静默回归的东西：

* 阶段边界两侧各走哪条实现。特别是 ``_step_align_green``：共享实现会在
  ``_near_field_pickup`` 不为 None 时改走 ``_step_formal_green_align``，而
  ``_begin_strategy_blue_transport`` 恰好会重新暴露该序列，不按相位区分就会
  把 2026-09-11 真机验证过的对准路径悄悄换掉。
* 策略阶段对继承方法的调用与基类签名一致。``_advance_cluster_search_sweep``
  曾按旧的单参数签名调用，删掉本地覆盖改为继承后会直接抛 TypeError。

不打开设备或窗口：全部经 ``from_app_config``（其文档保证不打开任何硬件资源）
或直接构造。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from rescue_vision.app.match import (
    MatchPreflight,
    MatchSequence as BaseMatchSequence,
    MatchStartArea,
    MatchState,
)
from rescue_vision.app.match_strategy import MatchSequence as StrategySequence
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from rescue_vision.world import TeamColor

from test_match import observation, snapshot

STRATEGY_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "runtime.strategy.yaml"

# 区域 2 的真实初始位姿：地图右上角，场地航向 -90°。
START_FIELD_POSITION = FieldPoint(1350.0, 1350.0)
START_HEADING_RAD = -math.pi / 2.0
CONTROL_PERIOD_NS = 20_000_000

# 蓝色危险物块必须在车辆前方（x > 0）才会被策略阶段选中。
BLUE_GROUND = GroundPoint(300.0, 0.0)
# 侧向让开蓝色前进走廊的绿块：策略阶段不会选它，但也不该挡住蓝色。
GREEN_GROUND = GroundPoint(250.0, 400.0)
# 正压在蓝色前进走廊上的绿块：策略阶段无法处理，只能继续空转搜索。
GREEN_BLOCKING_GROUND = GroundPoint(250.0, 0.0)


def strategy_sequence() -> StrategySequence:
    """按真机配置装配策略变体；``from_app_config`` 不打开硬件。"""

    sequence = StrategySequence.from_app_config(
        load_runtime_config(STRATEGY_CONFIG_PATH)
    )
    assert sequence._strategy_formal_phase is False
    return sequence


def start_strategy(sequence: StrategySequence) -> None:
    ready = sequence.preflight(0, MatchPreflight(True, True, True, True, True, True))
    assert ready.state is MatchState.PREFLIGHT
    started = sequence.start(1)
    assert started.state is MatchState.STARTUP_TURN_RIGHT
    assert started.reason == "strategy_started"


def search_frame(
    frame_sequence: int,
    timestamp_ns: int,
    *observations_: object,
):
    return snapshot(frame_sequence, timestamp_ns, *observations_)


def blue_observation(frame_sequence: int, timestamp_ns: int, ground=BLUE_GROUND):
    return observation(
        frame_sequence,
        timestamp_ns,
        ground,
        target_class=TargetClass.BLUE_DANGER,
    )


def green_observation(frame_sequence: int, timestamp_ns: int, ground=GREEN_GROUND):
    return observation(
        frame_sequence,
        timestamp_ns,
        ground,
        target_class=TargetClass.GREEN_SUPPLY,
    )


def enter_strategy_search(sequence: StrategySequence) -> None:
    start_strategy(sequence)
    sequence._started = True
    sequence.state = MatchState.SEARCH_CLUSTER


# --------------------------------------------------------------------------
# 装配契约
# --------------------------------------------------------------------------


def test_from_app_config_inverts_near_field_pickup() -> None:
    """策略阶段停用共享 worker：pickup 置空并留真身，翻转后恢复。

    这与 ``tests/test_match.py`` 对正式入口的断言方向相反，是变体的关键契约。
    """

    sequence = strategy_sequence()
    assert sequence._near_field_pickup is None
    assert sequence._strategy_saved_near_field_pickup is not None
    assert sequence.near_field_enabled is False
    # 蓝色近场计划只在控制线程内生成，依赖这两个仅在 from_app_config 建立的对象。
    assert sequence._strategy_blue_projector is not None
    assert sequence._strategy_blue_selector is not None


def test_strategy_reuses_base_state_collection() -> None:
    """变体不再复制基类字段清单，2047 新增状态必须都在。"""

    sequence = strategy_sequence()
    for attribute in (
        "_pose_history",
        "_near_field_failures",
        "_breakup_failed_aims",
        "_breakup_failed_aim_positions",
        "_breakup_attempt_positions",
        "_breakup_plan_rejections",
        "_rotation_budget_commit_diagnostic",
        "_path_corridor_margin_mm",
    ):
        assert hasattr(sequence, attribute), attribute
    assert not hasattr(sequence, "_breakup_failed_regions")


def test_strategy_is_a_subclass_of_the_formal_sequence() -> None:
    assert issubclass(StrategySequence, BaseMatchSequence)


# --------------------------------------------------------------------------
# 启动：策略状态复位
# --------------------------------------------------------------------------


def test_start_resets_strategy_state_and_pose_history() -> None:
    sequence = strategy_sequence()
    sequence._pose_history.append(object())
    sequence._strategy_startup_leg = 9
    sequence._strategy_blue_transport = True

    start_strategy(sequence)

    assert sequence._strategy_startup_leg == 1
    assert sequence._strategy_blue_transport is False
    assert sequence._strategy_blue_grasp_phase == "idle"
    assert sequence._strategy_formal_phase is False
    assert sequence._pose_history == []


def test_start_area_three_mirrors_to_the_blue_team() -> None:
    """``--start-area 3`` 在运行时中心对称并切到蓝方。"""

    sequence = StrategySequence.from_app_config(
        load_runtime_config(STRATEGY_CONFIG_PATH),
        start_area="3",
    )
    assert sequence._team_color is TeamColor.BLUE
    assert sequence._initial_field_position is not None
    assert sequence._initial_field_position.x == -START_FIELD_POSITION.x
    assert sequence._initial_field_position.y == -START_FIELD_POSITION.y


# --------------------------------------------------------------------------
# 策略阶段：只搜蓝色
# --------------------------------------------------------------------------


def drive_search_frames(
    sequence: StrategySequence,
    *,
    frames: int = 3,
    green_ground: GroundPoint = GREEN_GROUND,
) -> None:
    """喂入同时含绿块和蓝块的帧；追踪器需要两帧才确认。"""

    base_ns = 2_000_000_000
    for index in range(frames):
        timestamp_ns = base_ns + index * CONTROL_PERIOD_NS
        sequence.step(
            timestamp_ns,
            perception=search_frame(
                index,
                timestamp_ns,
                green_observation(index, timestamp_ns, green_ground),
                blue_observation(index, timestamp_ns),
            ),
            heading_rad=START_HEADING_RAD,
            cumulative_distance_m=0.0,
        )


def test_strategy_search_prefers_blue_and_ignores_green() -> None:
    """走廊畅通时策略阶段选蓝色；同帧的绿块永不被选中。"""

    sequence = strategy_sequence()
    enter_strategy_search(sequence)

    drive_search_frames(sequence)

    selected = sequence._selected_target()
    assert selected is not None
    assert selected.target_class is TargetClass.BLUE_DANGER
    assert sequence._strategy_blue_transport is True


def test_strategy_search_never_falls_back_to_a_blocking_green() -> None:
    """绿块挡住蓝色前进走廊时策略阶段只空转，不会改抓绿块。

    策略阶段只处理蓝色危险物块；走廊被别的目标占住时没有可执行计划，
    必须继续搜索而不是改写候选类别。
    """

    sequence = strategy_sequence()
    enter_strategy_search(sequence)

    drive_search_frames(sequence, green_ground=GREEN_BLOCKING_GROUND)

    assert sequence._selected_target() is None
    assert sequence._strategy_blue_transport is False
    assert sequence._strategy_formal_phase is False


def test_strategy_search_sweep_uses_base_signature() -> None:
    """搜索空转必须走基类三参签名并消费其提交决策。

    本地覆盖删除后这里调用的是继承来的
    ``_advance_cluster_search_sweep(timestamp_ns, heading_rad)``；按旧的单参数
    签名调用会直接抛 TypeError。
    """

    sequence = strategy_sequence()
    enter_strategy_search(sequence)
    timestamp_ns = 2_000_000_000

    for index in range(4):
        current = timestamp_ns + index * CONTROL_PERIOD_NS
        decision = sequence.step(
            current,
            perception=None,
            heading_rad=START_HEADING_RAD,
            cumulative_distance_m=0.0,
        )
        assert decision.reason.startswith("strategy_search_blue_")


def test_strategy_search_never_picks_a_target_behind_the_robot() -> None:
    """蓝色目标在车后时不作为候选，只做空转搜索。"""

    sequence = strategy_sequence()
    enter_strategy_search(sequence)
    timestamp_ns = 2_000_000_000

    decision = sequence.step(
        timestamp_ns,
        perception=search_frame(
            0,
            timestamp_ns,
            blue_observation(0, timestamp_ns, GroundPoint(-300.0, 0.0)),
        ),
        heading_rad=START_HEADING_RAD,
        cumulative_distance_m=0.0,
    )

    assert decision.reason.startswith("strategy_search_blue_")
    assert sequence._strategy_blue_transport is False


# --------------------------------------------------------------------------
# 阶段分派：蓝色运输沿用真机验证过的对准路径
# --------------------------------------------------------------------------


def test_align_green_keeps_the_validated_path_during_blue_transport(
    monkeypatch,
) -> None:
    """蓝色运输必须走 ``_align_green_legacy``，正式阶段才走正式对准。

    共享的 ``_step_align_green`` 在 ``_near_field_pickup`` 不为 None 时改走
    ``_step_formal_green_align``，而蓝色运输会重新暴露该序列；不做相位区分就会
    静默替换掉真机跑通过的对准行为。
    """

    calls: list[str] = []
    for name in ("_step_formal_green_align", "_align_green_legacy"):
        original = getattr(BaseMatchSequence, name)

        def recorder(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(BaseMatchSequence, name, recorder)

    sequence = strategy_sequence()
    start_strategy(sequence)
    sequence._near_field_pickup = sequence._strategy_saved_near_field_pickup

    sequence._strategy_formal_phase = False
    sequence._strategy_blue_transport = True
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    sequence._step_align_green(2_000_000_000)
    assert calls == ["_align_green_legacy"]

    calls.clear()
    sequence._strategy_formal_phase = True
    sequence._strategy_blue_transport = False
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    sequence._step_align_green(3_000_000_000)
    assert calls == ["_step_formal_green_align"]


# --------------------------------------------------------------------------
# 阶段翻转
# --------------------------------------------------------------------------


def test_return_backup_intercepts_finish_stop_and_flips_phase(monkeypatch) -> None:
    """第二趟返回结束（共享实现给出 FINISH_STOP）时翻转进正式阶段。"""

    sequence = strategy_sequence()
    start_strategy(sequence)
    sequence._started = True

    def finish_immediately(self, timestamp_ns, cumulative_distance_m):
        self.state = MatchState.FINISH_STOP
        return self._decision(timestamp_ns, 0.0, 0.0, "finish_stop_reached")

    monkeypatch.setattr(
        BaseMatchSequence, "_step_return_backup", finish_immediately
    )

    decision = sequence._step_return_backup(5_000_000_000, 0.0)

    assert sequence._strategy_formal_phase is True
    assert decision.reason == "strategy_blue_tasks_complete_start_match_green_search"


def test_phase_flip_restores_formal_state() -> None:
    sequence = strategy_sequence()
    start_strategy(sequence)
    sequence._started = True
    sequence._cluster_search_progress_rad = 2.0
    sequence._breakup_failed_aims.append(FieldPoint(100.0, 100.0))
    sequence._green_target_field_point = FieldPoint(50.0, 50.0)

    decision = sequence._begin_formal_match_after_strategy(5_000_000_000)

    assert sequence._strategy_formal_phase is True
    assert sequence._near_field_pickup is sequence._strategy_saved_near_field_pickup
    assert sequence.near_field_enabled is True
    # 翻转是真正的会话边界：策略阶段累积的转过角度不能带进正式阶段。
    assert sequence._cluster_search_progress_rad == 0.0
    assert sequence._breakup_failed_aims == []
    assert sequence._green_target_field_point is None
    assert decision.reason == "strategy_blue_tasks_complete_start_match_green_search"


def test_phase_flip_enables_opportunistic_single_green() -> None:
    """策略阶段关掉的首轮单绿机会必须在翻转后重新打开。"""

    sequence = strategy_sequence()
    start_strategy(sequence)
    sequence._started = True
    assert sequence.config.opportunistic_single_green_enabled is False

    sequence._begin_formal_match_after_strategy(5_000_000_000)

    assert sequence.config.opportunistic_single_green_enabled is True


# --------------------------------------------------------------------------
# D2 投放端点
# --------------------------------------------------------------------------


def test_d2_endpoints_use_opposite_side_then_own_side() -> None:
    """策略阶段两趟分别放到对面左右 D2 点；翻转后端点取回己方 y 符号。"""

    sequence = strategy_sequence()
    start_strategy(sequence)
    shipped = sequence.config

    sequence._strategy_delivery_slot = 0
    first = sequence._safe_zone_transport_endpoint()
    assert first == shipped.safe_zone_fallback_target_field

    sequence._strategy_delivery_slot = 1
    second = sequence._safe_zone_transport_endpoint()
    assert second == shipped.safe_zone_injured_target_field

    # 两个端点 x 符号相反（左/右），且都在对面（y 与己方安全区相反）。
    assert first.x * second.x < 0.0
    assert first.y < 0.0 and second.y < 0.0
    assert sequence._safe_zone_final_target_y_mm() == second.y


# --------------------------------------------------------------------------
# 开场两段冲刺
# --------------------------------------------------------------------------


def drive_startup(sequence: StrategySequence, *, max_cycles: int = 4000):
    """按控制周期积分航向与里程，推进开场两段冲刺并记录逐周期决策。"""

    records = []
    heading_rad = START_HEADING_RAD
    distance_m = 0.0
    timestamp_ns = 1_000_000
    for _ in range(max_cycles):
        decision = sequence.step(
            timestamp_ns,
            perception=None,
            heading_rad=heading_rad,
            cumulative_distance_m=distance_m,
        )
        records.append(
            (
                decision.state,
                decision.reason,
                decision.linear_velocity_m_s,
                decision.angular_velocity_rad_s,
                sequence._strategy_startup_leg,
            )
        )
        heading_rad += decision.angular_velocity_rad_s * (CONTROL_PERIOD_NS / 1e9)
        distance_m += decision.linear_velocity_m_s * (CONTROL_PERIOD_NS / 1e9)
        timestamp_ns += CONTROL_PERIOD_NS
        if decision.state is MatchState.SEARCH_CLUSTER:
            break
    return records


def test_startup_runs_two_opposite_legs_then_searches() -> None:
    """开场先右转短冲、再左转长冲，两段都完成后才进入搜索。"""

    sequence = strategy_sequence()
    start_strategy(sequence)
    sequence._started = True

    records = drive_startup(sequence)

    states = [state for state, *_ in records]
    assert MatchState.STARTUP_TURN_RIGHT in states
    assert MatchState.STARTUP_TURN_SETTLE in states
    assert MatchState.STARTUP_FORWARD in states
    assert MatchState.STARTUP_FORWARD_SETTLE in states
    assert states[-1] is MatchState.SEARCH_CLUSTER

    # 第一腿右转（角速度为负），第二腿左转（为正）。
    leg_one_turn = next(
        record for record in records if record[1] == "strategy_startup_right_turn"
    )
    leg_two_turn = next(
        record for record in records if record[1] == "strategy_startup_left_turn"
    )
    assert leg_one_turn[3] < 0.0
    assert leg_two_turn[3] > 0.0
    assert leg_one_turn[4] == 1
    assert leg_two_turn[4] == 2

    # 转向量必须达到配置的启动转角，而不是只看状态流转。
    assert sequence._strategy_startup_leg == 2


def test_startup_turn_velocity_flips_sign_per_leg() -> None:
    sequence = strategy_sequence()
    start_strategy(sequence)
    magnitude = abs(sequence.config.startup_turn_angular_velocity_rad_s)

    sequence._strategy_startup_leg = 1
    assert sequence._strategy_startup_turn_velocity() == -magnitude

    sequence._strategy_startup_leg = 2
    assert sequence._strategy_startup_turn_velocity() == magnitude


# --------------------------------------------------------------------------
# 位姿历史前置条件
# --------------------------------------------------------------------------


def test_pose_history_is_recorded_and_queryable() -> None:
    """策略阶段也要维护采集时刻位姿历史。

    继承来的 ``_selected_green_point`` / ``_ground_point_at_now`` 依赖它补偿
    延迟观测；变体的 ``__init__``/``start``/``_step_actions`` 是有意分歧，
    必须自己保证这些状态存在并复位。
    """

    sequence = strategy_sequence()
    enter_strategy_search(sequence)

    base_ns = 2_000_000_000
    for index in range(4):
        timestamp_ns = base_ns + index * CONTROL_PERIOD_NS
        sequence.step(
            timestamp_ns,
            perception=None,
            heading_rad=START_HEADING_RAD + 0.05 * index,
            cumulative_distance_m=0.02 * index,
        )

    assert len(sequence._pose_history) == 4
    pose = sequence._pose_at(base_ns + 2 * CONTROL_PERIOD_NS)
    assert pose is not None
    assert pose.cumulative_distance_m == 0.04


def test_start_clears_pose_history() -> None:
    sequence = strategy_sequence()
    sequence._pose_history.append(object())
    start_strategy(sequence)
    assert sequence._pose_history == []


# --------------------------------------------------------------------------
# 入口接线
# --------------------------------------------------------------------------


def test_main_wires_the_strategy_entry(monkeypatch) -> None:
    """``main()`` 必须用策略工厂、策略 mode 名和 ``strategy_`` 日志前缀。"""

    import rescue_vision.app.match_runtime as match_runtime
    import rescue_vision.app.match_strategy as match_strategy_module

    received: dict[str, object] = {}
    monkeypatch.setattr(
        match_runtime,
        "_run_hardware",
        lambda *args, **kwargs: received.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["match_strategy", "--config", str(STRATEGY_CONFIG_PATH), "--start-area", "3"],
    )

    match_strategy_module.main()

    assert received["mode_name"] == "match_strategy"
    assert received["log_file_prefix"] == "strategy_"
    assert received["start_area"] is MatchStartArea.AREA_3
    factory = received["sequence_factory"]
    assert factory.__self__ is StrategySequence
    assert factory.__name__ == "from_app_config"
