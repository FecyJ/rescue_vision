from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import yaml

from rescue_vision.app.cluster_breakup import (
    BreakupDecision,
    BreakupState,
    GripperPosture,
)
from rescue_vision.app.simulation_20_point import (
    Simulation20PointSequence,
    Simulation20PointState,
    SimulationHealth,
    SimulationPreflight,
    _begin_time_named_log,
    _end_time_named_log,
    _publish_remote_simulation_state,
)
from rescue_vision.config import Simulation20PointRuntimeConfig, load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.localization import FieldPose2D, FusedPoseEstimate
from rescue_vision.mission import (
    ActivityState,
    MissionConfig,
    MissionStateMachine,
    TerminationReason,
    TransportStatus,
)
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    PerceptionSnapshot,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import MultiTargetTracker, TrackingConfig
from rescue_vision.world import (
    HazardState,
    RegionKind,
    StaticRegion,
    WorldModel,
    WorldModelConfig,
)


def runtime_config(**overrides: object) -> Simulation20PointRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "target_delivery_count": 4,
        "prepush_offset_mm": 150.0,
        "robot_footprint_radius_mm": 100.0,
        "target_half_extent_mm": 20.0,
        "safety_margin_mm": 20.0,
        "delivery_inset_mm": 20.0,
        "max_position_uncertainty_mm": 200.0,
        "max_heading_uncertainty_rad": 0.5,
        "target_max_age_ms": 500.0,
        "scan_min_heading_span_rad": 0.5,
        "scan_angular_velocity_rad_s": 0.2,
        "navigation_speed_m_s": 0.1,
        "navigation_angular_kp_rad_s": 1.0,
        "navigation_max_angular_velocity_rad_s": 0.4,
        "navigation_position_tolerance_mm": 30.0,
        "navigation_heading_tolerance_rad": 0.1,
        "approach_speed_m_s": 0.05,
        "alignment_kp_rad_s": 1.0,
        "alignment_max_angular_velocity_rad_s": 0.3,
        "engage_distance_mm": 80.0,
        "engage_lateral_tolerance_mm": 30.0,
        "contact_corridor_length_mm": 120.0,
        "push_speed_m_s": 0.05,
        "push_angular_kp_rad_s": 0.5,
        "push_max_angular_velocity_rad_s": 0.2,
        "retreat_speed_m_s": 0.05,
        "retreat_distance_m": 0.1,
        "retreat_clear_distance_mm": 100.0,
        "delivery_confirm_frames": 2,
        "disengage_confirm_frames": 2,
        "max_breakup_attempts_per_delivery": 1,
        "max_breakup_attempts_total": 2,
        "breakup_settle_confirm_frames": 2,
        "min_green_clearance_mm": 40.0,
        "corridor_sample_step_mm": 30.0,
        "completed_target_exclusion_mm": 100.0,
    }
    values.update(overrides)
    return Simulation20PointRuntimeConfig(**values)  # type: ignore[arg-type]


def observation(
    frame_sequence: int,
    timestamp_ns: int,
    ground_point: GroundPoint,
    *,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
) -> TargetObservation:
    box = UndistortedBoundingBox(40.0, 40.0, 60.0, 60.0)
    segmentation = RoiColorSegmentation(
        candidate_class=target_class,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=box,
        mask=np.full((20, 20), 255, dtype=np.uint8),
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
        k0=UndistortedPixel(50.0, 50.0),
        k0_confidence=0.95,
        ground_point=ground_point,
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


class _ImmediateGreenBreakup:
    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        return BreakupDecision(
            timestamp_ns,
            BreakupState.GREEN_FOUND,
            0.0,
            0.0,
            GripperPosture.CLOSED,
            "green_found",
        )


class _ApproachBreakup:
    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        return BreakupDecision(
            timestamp_ns,
            BreakupState.APPROACH_CLUSTER,
            0.1,
            0.0,
            GripperPosture.CLOSED,
            "approach_cluster",
        )


class _FaultingBreakup:
    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        return BreakupDecision(
            timestamp_ns,
            BreakupState.FAULT,
            0.0,
            0.0,
            GripperPosture.CLOSED,
            "cluster_lost",
        )


class _FixedBreakup:
    def __init__(self, state: BreakupState) -> None:
        self.state = state

    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        posture = (
            GripperPosture.OPEN
            if self.state
            in {BreakupState.BREAKUP_RELEASE, BreakupState.BREAKUP_OPEN_RETREAT}
            else GripperPosture.CLOSED
        )
        return BreakupDecision(
            timestamp_ns,
            self.state,
            0.0 if self.state is BreakupState.SCAN_GREEN else 0.1,
            0.1 if self.state is BreakupState.SCAN_GREEN else 0.0,
            posture,
            "fixed_breakup_action",
        )


def make_sequence(
    *,
    config: Simulation20PointRuntimeConfig | None = None,
    breakup: object | None = None,
    breakup_factory: object | None = None,
    confirmation_hits: int = 1,
    match_duration_s: float = 1000.0,
    no_motion_timeout_s: float = 15.0,
) -> Simulation20PointSequence:
    regions = (
        StaticRegion(
            "field",
            RegionKind.FIELD,
            (
                FieldPoint(-2000.0, -2000.0),
                FieldPoint(2000.0, -2000.0),
                FieldPoint(2000.0, 2000.0),
                FieldPoint(-2000.0, 2000.0),
            ),
        ),
        StaticRegion(
            "own-material",
            RegionKind.OWN_MATERIAL,
            (
                FieldPoint(-300.0, 1200.0),
                FieldPoint(300.0, 1200.0),
                FieldPoint(300.0, 1600.0),
                FieldPoint(-300.0, 1600.0),
            ),
        ),
    )
    return Simulation20PointSequence(
        config or runtime_config(),
        tracker=MultiTargetTracker(
            TrackingConfig(confirmation_hits, 2000.0, 0.1, 1000.0, 0.1, 0.01)
        ),
        world_model=WorldModel(
            WorldModelConfig(500.0, 500.0, 0.6, 0.15, 0.5),
            regions,
        ),
        mission=MissionStateMachine(
            MissionConfig(
                match_duration_s,
                no_motion_timeout_s,
                10.0,
                (
                    TargetClass.ORANGE_INJURED,
                    TargetClass.BLACK_CORE,
                    TargetClass.GREEN_SUPPLY,
                ),
            )
        ),
        breakup=breakup or _ImmediateGreenBreakup(),
        breakup_factory=breakup_factory,
    )


def pose(
    *,
    x: float = -1000.0,
    y: float = 0.0,
    heading: float = 0.0,
    position_uncertainty_mm: float = 20.0,
    heading_uncertainty_rad: float = 0.02,
):
    return FusedPoseEstimate(
        pose=FieldPose2D(FieldPoint(x, y), heading),
        estimate_timestamp_ns=1,
        position_uncertainty_mm=position_uncertainty_mm,
        heading_uncertainty_rad=heading_uncertainty_rad,
        confidence=0.9,
        anchor_source="configured_start",
        quality=frozenset(),
    )


def test_simulation_remote_state_submits_latest_localization() -> None:
    class _RecordingRemote:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def check_health(self) -> None:
            self.events.append(("health", None))

        def submit_localization(
            self,
            estimate: FusedPoseEstimate | None,
            timestamp_ns: int,
        ) -> None:
            self.events.append(("localization", (estimate, timestamp_ns)))

        def submit(self, frame: object) -> None:
            self.events.append(("perception", frame))

    remote = _RecordingRemote()
    estimate = pose(x=-1350.0, y=-1200.0, heading=1.5)
    rendered = object()

    _publish_remote_simulation_state(
        remote,
        pose=estimate,
        timestamp_ns=200,
        rendered=rendered,
    )
    # 定位旁路暂不可用时发布 None，观察端按 robot_localized=false 处理。
    _publish_remote_simulation_state(
        remote,
        pose=None,
        timestamp_ns=300,
        rendered=None,
    )

    assert remote.events == [
        ("health", None),
        ("localization", (estimate, 200)),
        ("perception", rendered),
        ("health", None),
        ("localization", (None, 300)),
    ]


def test_time_named_log_tees_output_and_restores_streams(tmp_path: Path) -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        stream, stdout_before, stderr_before = _begin_time_named_log(tmp_path)
        assert stream is not None
        print("teed-log-line", flush=True)
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        assert re.fullmatch(r"\d{8}_\d{4}\.log", log_files[0].name)
        content = log_files[0].read_text(encoding="utf-8")
        assert "logging to " in content
        assert "teed-log-line" in content
        _end_time_named_log(stream, stdout_before, stderr_before)
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # 不传目录时不动标准流、不创建任何文件。
    assert _begin_time_named_log(None) == (None, sys.stdout, sys.stderr)
    assert list(tmp_path.glob("*.log")) == log_files


def start_sequence(sequence: Simulation20PointSequence) -> None:
    sequence.preflight(
        0,
        SimulationPreflight(
            telemetry_fresh=True,
            watchdog_armed=True,
            emergency_stop_clear=True,
            zero_speed_command_accepted=True,
            camera_observation_fresh=True,
            observe_only_remote=True,
        ),
    )
    sequence.start(1)


def test_simulation_config_is_explicit_and_strict(tmp_path: Path) -> None:
    config = load_runtime_config("configs/runtime.simulation-20min.yaml")
    assert config.simulation_20_point.enabled
    assert config.simulation_20_point.target_delivery_count == 4
    assert config.remote.access_mode.value == "observe_only"
    assert config.localization.fusion.max_interpolated_overrun_samples == 1
    # 试验配置的定距里程计超期预算会被现场频繁调整；只验证可解析且语义合法。
    overrun_budget = config.motion.odometry.max_consecutive_overrun_samples
    assert overrun_budget is None or (
        isinstance(overrun_budget, int) and overrun_budget >= 0
    )
    # 带符号角速度：右转搜索/扫描直接用负值表达，不再有方向关键字。
    assert (
        config.motion.cluster_breakup.search_angular_velocity_rad_s
        == pytest.approx(-0.45)
    )
    assert (
        config.simulation_20_point.scan_angular_velocity_rad_s
        == pytest.approx(-0.30)
    )
    assert config.build_simulation_20_point_sequence().state is Simulation20PointState.BOOT

    raw = yaml.safe_load(
        Path("configs/runtime.example.yaml").read_text(encoding="utf-8")
    )
    raw["simulation_20_point"] = {"unexpected": 1}
    invalid = tmp_path / "invalid-runtime.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys in simulation_20_point"):
        load_runtime_config(invalid)


def test_preflight_failure_latches_terminal_stop() -> None:
    sequence = make_sequence()
    decision = sequence.preflight(
        0,
        SimulationPreflight(True, False, True, True, True, True),
    )
    assert decision.state is Simulation20PointState.TERMINAL_STOP
    assert decision.linear_velocity_m_s == 0.0


def test_existing_breakup_is_followed_by_track_reset_and_full_scan() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    green = lambda frame, timestamp: snapshot(
        frame,
        timestamp,
        observation(frame, timestamp, GroundPoint(600.0, 0.0)),
    )

    reset = sequence.step(
        2,
        perception=green(1, 2),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert reset.state is Simulation20PointState.RESET_TARGET_TRACKS
    settle_1 = sequence.step(
        3,
        perception=green(2, 3),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    settle_2 = sequence.step(
        4,
        perception=green(3, 4),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert settle_1.state is Simulation20PointState.RESET_TARGET_TRACKS
    assert settle_2.state is Simulation20PointState.SCAN_GREEN

    scanning = sequence.step(
        5,
        perception=green(4, 5),
        pose=pose(heading=0.2),
        cumulative_distance_m=0.0,
    )
    assert scanning.state is Simulation20PointState.SCAN_GREEN
    evaluated = sequence.step(
        6,
        perception=green(5, 6),
        pose=pose(heading=0.8),
        cumulative_distance_m=0.0,
    )
    assert evaluated.state is Simulation20PointState.EVALUATE_EASY_GREEN
    selected = sequence.step(
        7,
        perception=green(6, 7),
        pose=pose(heading=0.8),
        cumulative_distance_m=0.0,
    )
    assert selected.state is Simulation20PointState.SELECT_GREEN
    assert selected.selected_track_id == 1


def test_unknown_or_dangerous_target_cannot_become_easy_green() -> None:
    sequence = make_sequence(
        config=runtime_config(scan_min_heading_span_rad=0.1),
    )
    start_sequence(sequence)
    dangerous = snapshot(
        1,
        2,
        observation(1, 2, GroundPoint(600.0, 0.0)),
        observation(
            1,
            2,
            GroundPoint(520.0, 50.0),
            target_class=TargetClass.BLUE_DANGER,
        ),
    )
    sequence.step(2, perception=dangerous, pose=pose(), cumulative_distance_m=0.0)
    sequence.step(
        3,
        perception=dangerous,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    sequence.step(
        4,
        perception=snapshot(
            2,
            4,
            observation(2, 4, GroundPoint(600.0, 0.0)),
            observation(
                2,
                4,
                GroundPoint(520.0, 50.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    # 危险/未知目标占满候选视野时不锁存任何安全保持：继续原地扫描，扫描完成
    # 后因为没有易搬运绿色进入重复解团评估；无解团驱动时保持零速等待。
    decisions = []
    for frame_sequence, (timestamp_ns, heading) in enumerate(
        ((5, 0.2), (6, 0.4), (7, 0.6), (8, 0.8), (9, 1.0)),
        start=3,
    ):
        current = snapshot(
            frame_sequence,
            timestamp_ns,
            observation(frame_sequence, timestamp_ns, GroundPoint(600.0, 0.0)),
            observation(
                frame_sequence,
                timestamp_ns,
                GroundPoint(520.0, 50.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        )
        decisions.append(
            sequence.step(
                timestamp_ns,
                perception=current,
                pose=pose(heading=heading),
                cumulative_distance_m=0.0,
            )
        )
    assert all(
        decision.state is not Simulation20PointState.TERMINAL_STOP
        for decision in decisions
    )
    held = decisions[-1]
    assert held.state is Simulation20PointState.PLAN_REBREAKUP
    assert held.reason == "rebreakup_driver_unavailable"
    assert held.linear_velocity_m_s == 0.0


def test_side_path_branch_error_holds_and_recovers_automatically() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    held = sequence.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), branch_error="slow_inference"),
    )
    # 单次旁路故障不再锁存终止停车：保持当前行为状态输出零速，故障清除后
    # 下一周期自动继续，任何运动请求都不能覆盖该保持。
    assert held.state is Simulation20PointState.LEAVE_START
    assert held.reason == "side_path_recovering:slow_inference"
    assert held.linear_velocity_m_s == 0.0
    assert held.angular_velocity_rad_s == 0.0

    resumed = sequence.step(
        3,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert resumed.state is Simulation20PointState.RESET_TARGET_TRACKS


def test_emergency_stop_and_remote_gate_remain_terminal() -> None:
    stopped = make_sequence()
    start_sequence(stopped)
    estop = stopped.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), emergency_stop_clear=False),
    )
    assert estop.state is Simulation20PointState.TERMINAL_STOP
    assert estop.reason == "emergency_stop_latched"
    assert estop.linear_velocity_m_s == 0.0

    remote_violation = make_sequence()
    start_sequence(remote_violation)
    escalated = remote_violation.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), observe_only_remote=False),
    )
    assert escalated.state is Simulation20PointState.TERMINAL_STOP
    assert escalated.reason == "remote_is_not_observe_only"
    assert escalated.linear_velocity_m_s == 0.0


def test_evaluate_pose_uncertain_falls_through_to_rebreakup_evaluation() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.EVALUATE_EASY_GREEN

    # 位姿存在但不确定度不合格：无法验证任何易搬运绿色，按“无易搬运绿色”
    # 进入重复解团评估，不在 EVALUATE 无限保持等待。
    evaluated = sequence.step(
        2,
        perception=snapshot(1, 2),
        pose=pose(position_uncertainty_mm=300.0),
        cumulative_distance_m=0.0,
    )
    assert evaluated.state is Simulation20PointState.PLAN_REBREAKUP
    assert evaluated.reason == "no_easy_green_pose_uncertain"
    assert evaluated.linear_velocity_m_s == 0.0
    assert evaluated.angular_velocity_rad_s == 0.0

    # 下一周期由 PLAN_REBREAKUP 自身的上限检查决定继续解团或保持零速；
    # 本测试装配没有解团工厂，保持零速等待而不是终止。
    planned = sequence.step(
        3,
        perception=snapshot(2, 3),
        pose=pose(position_uncertainty_mm=300.0),
        cumulative_distance_m=0.0,
    )
    assert planned.state is Simulation20PointState.PLAN_REBREAKUP
    assert planned.reason == "rebreakup_driver_unavailable"
    assert planned.linear_velocity_m_s == 0.0


def test_rebreakup_proceeds_without_clearance_progress_gate() -> None:
    # 重复解团不强求每次取得可测量的分离进展：两次 PLAN_REBREAKUP 之间目标团
    # 几何完全不变，第二次评估仍执行受控再解团，只由次数上限停止。
    sequence = make_sequence(
        config=runtime_config(
            max_breakup_attempts_per_delivery=2,
            max_breakup_attempts_total=3,
        ),
        breakup=_ApproachBreakup(),
        breakup_factory=_ApproachBreakup,
    )
    start_sequence(sequence)

    def clustered(frame_sequence: int, timestamp_ns: int) -> PerceptionSnapshot:
        return snapshot(
            frame_sequence,
            timestamp_ns,
            observation(frame_sequence, timestamp_ns, GroundPoint(500.0, 0.0)),
            observation(frame_sequence, timestamp_ns, GroundPoint(500.0, 50.0)),
        )

    # 先让跟踪器确认两个紧贴的绿色目标；注入的解团实例会抢占 step() 分派，
    # 完成一次受控再解团后将其清空，等价于解团序列结束后回到 PLAN_REBREAKUP。
    sequence.step(2, perception=clustered(1, 2), pose=pose(), cumulative_distance_m=0.0)
    sequence._breakup = None
    sequence.state = Simulation20PointState.PLAN_REBREAKUP

    planned = sequence.step(
        3, perception=clustered(2, 3), pose=pose(), cumulative_distance_m=0.0
    )
    assert planned.state is Simulation20PointState.CENTER_REBREAKUP_CLUSTER
    assert planned.reason == "begin_controlled_rebreakup"
    assert planned.linear_velocity_m_s == 0.0

    # 目标团几何不变，第二次评估仍放行，不因“无净空改善”保持。
    sequence._breakup = None
    sequence.state = Simulation20PointState.PLAN_REBREAKUP
    repeated = sequence.step(
        4, perception=clustered(3, 4), pose=pose(), cumulative_distance_m=0.0
    )
    assert repeated.state is Simulation20PointState.CENTER_REBREAKUP_CLUSTER
    assert repeated.reason == "begin_controlled_rebreakup"
    assert sequence._current_breakup_attempts == 2

    # 次数上限仍然生效：第三次评估保持零速。
    sequence._breakup = None
    sequence.state = Simulation20PointState.PLAN_REBREAKUP
    limited = sequence.step(
        5, perception=clustered(4, 5), pose=pose(), cumulative_distance_m=0.0
    )
    assert limited.state is Simulation20PointState.PLAN_REBREAKUP
    assert limited.reason == "breakup_attempt_limit_per_delivery"
    assert limited.linear_velocity_m_s == 0.0


def test_world_update_failure_holds_without_latching_and_recovers() -> None:
    sequence = make_sequence()
    start_sequence(sequence)

    # 快照时间戳超出控制时间戳是瞬时内部同步不一致，不是终止原因。
    held = sequence.step(
        3,
        perception=snapshot(1, 4),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert held.state is Simulation20PointState.LEAVE_START
    assert held.reason.startswith("world_update_recovering:")
    assert held.linear_velocity_m_s == 0.0
    assert held.angular_velocity_rad_s == 0.0

    resumed = sequence.step(
        5,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert resumed.state is Simulation20PointState.RESET_TARGET_TRACKS


def test_breakup_fault_resets_tracks_and_continues_scan() -> None:
    sequence = make_sequence(breakup=_FaultingBreakup())
    start_sequence(sequence)

    faulted = sequence.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    # 瞬时解团故障不再锁存终止停车：丢弃碰撞前轨迹并回到常规扫描路径。
    assert faulted.state is Simulation20PointState.RESET_TARGET_TRACKS
    assert faulted.reason == "breakup_fault_reset_tracks:cluster_lost"
    assert faulted.linear_velocity_m_s == 0.0
    assert faulted.angular_velocity_rad_s == 0.0

    sequence.step(
        3,
        perception=snapshot(1, 3),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    resumed = sequence.step(
        4,
        perception=snapshot(2, 4),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert resumed.state is Simulation20PointState.SCAN_GREEN
    assert resumed.angular_velocity_rad_s == pytest.approx(0.2)


def test_breakup_approach_is_not_category_gated() -> None:
    sequence = make_sequence(
        breakup=_ApproachBreakup(),
        confirmation_hits=2,
    )
    start_sequence(sequence)
    first = snapshot(
        1,
        2,
        observation(1, 2, GroundPoint(500.0, 0.0)),
    )
    approaching = sequence.step(
        2,
        perception=first,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert approaching.state is Simulation20PointState.APPROACH_CLUSTER
    assert approaching.linear_velocity_m_s == pytest.approx(0.1)


def test_blue_during_breakup_does_not_trigger_safety_hold() -> None:
    sequence = make_sequence(breakup=_ApproachBreakup())
    start_sequence(sequence)
    danger_1 = snapshot(
        1,
        2,
        observation(
            1,
            2,
            GroundPoint(500.0, 0.0),
        ),
        observation(
            1,
            2,
            GroundPoint(520.0, 40.0),
            target_class=TargetClass.BLUE_DANGER,
        ),
    )
    approaching = sequence.step(
        2,
        perception=danger_1,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert approaching.state is Simulation20PointState.APPROACH_CLUSTER
    assert approaching.linear_velocity_m_s == pytest.approx(0.1)
    continuing = sequence.step(
        3,
        perception=danger_1,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert continuing.state is Simulation20PointState.APPROACH_CLUSTER
    assert continuing.linear_velocity_m_s == pytest.approx(0.1)


def test_blue_only_view_follows_normal_scan_without_special_branch() -> None:
    sequence = make_sequence(
        config=runtime_config(breakup_settle_confirm_frames=1),
    )
    start_sequence(sequence)
    danger = snapshot(
        1,
        2,
        observation(
            1,
            2,
            GroundPoint(500.0, 0.0),
        ),
    )
    sequence.step(
        2,
        perception=danger,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    sequence.step(
        3,
        perception=snapshot(
            2,
            3,
            observation(
                2,
                3,
                GroundPoint(500.0, 0.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    searching = sequence.step(
        4,
        perception=snapshot(
            3,
            4,
            observation(
                3,
                4,
                GroundPoint(500.0, 0.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        pose=pose(heading=0.2),
        cumulative_distance_m=0.0,
    )
    assert searching.state is Simulation20PointState.SCAN_GREEN
    assert searching.reason == "scan_green"
    assert searching.linear_velocity_m_s == 0.0
    assert searching.angular_velocity_rad_s != 0.0


def test_precontact_blue_track_cancels_green_selection_without_transport() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._breakup = None
    sequence._selected_track_id = 1
    sequence.state = Simulation20PointState.APPROACH_GREEN

    decision = sequence.step(
        2,
        perception=snapshot(
            1,
            2,
            observation(
                1,
                2,
                GroundPoint(500.0, 0.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    assert decision.state is Simulation20PointState.EVALUATE_EASY_GREEN
    assert decision.selected_track_id is None
    assert sequence._transport.engaged_track_ids == ()
    assert decision.linear_velocity_m_s == 0.0


def test_blue_in_prepush_corridor_cancels_plan_for_reselection() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._breakup = None
    sequence._selected_track_id = 1
    sequence.state = Simulation20PointState.NAVIGATE_PREPUSH

    decision = sequence.step(
        2,
        perception=snapshot(
            1,
            2,
            observation(1, 2, GroundPoint(1200.0, 0.0)),
            observation(
                1,
                2,
                GroundPoint(800.0, 0.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    assert decision.state is Simulation20PointState.EVALUATE_EASY_GREEN
    assert decision.reason == "navigate_corridor_blocked_reselect_green"
    assert decision.selected_track_id is None
    assert decision.linear_velocity_m_s == 0.0


def test_four_green_deliveries_latch_finish_stop_and_twenty_points() -> None:
    sequence = make_sequence(
        config=runtime_config(
            scan_min_heading_span_rad=0.5,
            delivery_confirm_frames=1,
            disengage_confirm_frames=1,
            retreat_distance_m=0.05,
            breakup_settle_confirm_frames=1,
        )
    )
    start_sequence(sequence)
    timestamp_ns = 1
    frame_sequence = 0

    def feed(
        field_point: FieldPoint,
        *,
        x: float = -1000.0,
        y: float = 0.0,
        heading: float = 0.0,
        distance_m: float = 0.0,
    ):
        nonlocal timestamp_ns, frame_sequence
        timestamp_ns += 10_000_000
        frame_sequence += 1
        return sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame_sequence,
                timestamp_ns,
                observation(
                    frame_sequence,
                    timestamp_ns,
                    GroundPoint(
                        np.cos(heading) * (field_point.x - x)
                        + np.sin(heading) * (field_point.y - y),
                        -np.sin(heading) * (field_point.x - x)
                        + np.cos(heading) * (field_point.y - y),
                    ),
                ),
            ),
            pose=pose(x=x, y=y, heading=heading),
            cumulative_distance_m=distance_m,
        )

    for target_x in (-500.0, -200.0, 200.0, 500.0):
        target = FieldPoint(target_x, 0.0)
        if sequence.state is Simulation20PointState.LEAVE_START:
            assert feed(target).state is Simulation20PointState.RESET_TARGET_TRACKS
        while sequence.state is Simulation20PointState.RESET_TARGET_TRACKS:
            feed(target)
        assert sequence.state is Simulation20PointState.SCAN_GREEN
        feed(target, heading=0.0)
        assert feed(target, heading=0.6).state is Simulation20PointState.EVALUATE_EASY_GREEN
        assert feed(target).state is Simulation20PointState.SELECT_GREEN
        assert feed(target).state is Simulation20PointState.PLAN_PREPUSH
        assert feed(target).state is Simulation20PointState.NAVIGATE_PREPUSH

        plan = sequence._selected_plan
        assert plan is not None
        prepush_heading = np.arctan2(
            plan.prepush_field.y,
            plan.prepush_field.x + 1000.0,
        )
        assert (
            feed(target, heading=float(prepush_heading)).state
            is Simulation20PointState.NAVIGATE_PREPUSH
        )
        assert feed(
            target,
            x=plan.prepush_field.x,
            y=plan.prepush_field.y,
            heading=float(prepush_heading),
        ).state is Simulation20PointState.ALIGN_GREEN
        push_heading = float(np.arctan2(1400.0, -target_x))
        approaching = feed(
            target,
            x=plan.prepush_field.x,
            y=plan.prepush_field.y,
            heading=push_heading,
        )
        assert approaching.state is Simulation20PointState.APPROACH_GREEN
        assert approaching.gripper_posture is GripperPosture.TRANSPORT
        contact_x = target.x - float(np.cos(push_heading)) * 80.0
        contact_y = target.y - float(np.sin(push_heading)) * 80.0
        engaging = feed(
            target,
            x=contact_x,
            y=contact_y,
            heading=push_heading,
        )
        assert engaging.state is Simulation20PointState.ENGAGE_GREEN
        assert engaging.gripper_posture is GripperPosture.TRANSPORT
        engaged = feed(
            target,
            x=contact_x,
            y=contact_y,
            heading=push_heading,
        )
        assert engaged.state is Simulation20PointState.PUSH_TO_MATERIAL_ZONE
        assert engaged.gripper_posture is GripperPosture.CLOSED

        destination = FieldPoint(0.0, 1400.0)
        assert feed(
            destination,
            x=0.0,
            y=1000.0,
            heading=np.pi / 2.0,
        ).state is Simulation20PointState.VERIFY_DELIVERY
        assert feed(
            destination,
            x=0.0,
            y=1000.0,
            heading=np.pi / 2.0,
            distance_m=0.1,
        ).state is Simulation20PointState.DISENGAGE_AND_RETREAT
        feed(destination, x=0.0, y=1000.0, heading=np.pi / 2.0, distance_m=0.1)
        cleared = feed(
            destination,
            x=0.0,
            y=1000.0,
            heading=np.pi / 2.0,
            distance_m=0.2,
        )
        assert cleared.state is Simulation20PointState.UPDATE_PROGRESS
        progress = feed(
            destination,
            x=-1000.0,
            y=0.0,
            distance_m=0.2,
        )
        if target_x != 500.0:
            assert progress.state is Simulation20PointState.RESET_TARGET_TRACKS
        else:
            assert progress.state is Simulation20PointState.FINISH_STOP

    assert sequence.valid_green_deliveries == 4
    assert sequence.score_points == 20
    stopped = sequence.step(
        timestamp_ns + 1,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.2,
    )
    assert stopped.state is Simulation20PointState.FINISH_STOP
    assert stopped.linear_velocity_m_s == 0.0


class _ParkedCenterBreakup:
    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        return BreakupDecision(
            timestamp_ns,
            BreakupState.CENTER_CLUSTER,
            0.0,
            0.0,
            GripperPosture.CLOSED,
            "center_cluster_parked",
        )


def test_match_timeout_during_breakup_latches_terminal_stop() -> None:
    sequence = make_sequence(
        breakup=_ApproachBreakup(),
        match_duration_s=0.05,
    )
    start_sequence(sequence)
    green = lambda frame, timestamp: snapshot(
        frame,
        timestamp,
        observation(frame, timestamp, GroundPoint(600.0, 0.0)),
    )

    first = sequence.step(
        2,
        perception=green(1, 2),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert first.state is Simulation20PointState.APPROACH_CLUSTER
    expired = sequence.step(
        60_000_002,
        perception=green(2, 60_000_002),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert expired.state is Simulation20PointState.TERMINAL_STOP
    assert expired.linear_velocity_m_s == 0.0
    assert expired.mission_decision is not None
    assert (
        expired.mission_decision.termination_reason
        is TerminationReason.MATCH_TIMEOUT
    )


def test_no_motion_timeout_during_breakup_latches_terminal_stop() -> None:
    sequence = make_sequence(breakup=_ParkedCenterBreakup())
    start_sequence(sequence)

    parked = sequence.step(
        2,
        perception=snapshot(1, 2, observation(1, 2, GroundPoint(600.0, 0.0))),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert parked.state is Simulation20PointState.CENTER_CLUSTER
    assert parked.linear_velocity_m_s == 0.0
    stalled = sequence.step(
        16_000_000_002,
        perception=snapshot(
            2,
            16_000_000_002,
            observation(2, 16_000_000_002, GroundPoint(600.0, 0.0)),
        ),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert stalled.state is Simulation20PointState.TERMINAL_STOP
    assert stalled.mission_decision is not None
    assert (
        stalled.mission_decision.termination_reason
        is TerminationReason.NO_MOTION_TIMEOUT
    )


def test_stale_vision_during_breakup_keeps_fixed_action() -> None:
    sequence = make_sequence(breakup=_ApproachBreakup())
    start_sequence(sequence)
    green = lambda frame, timestamp: snapshot(
        frame,
        timestamp,
        observation(frame, timestamp, GroundPoint(600.0, 0.0)),
    )

    approaching = sequence.step(
        2,
        perception=green(1, 2),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert approaching.state is Simulation20PointState.APPROACH_CLUSTER
    stale = sequence.step(
        600_000_002,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    # Non-terminal mission holds must not interrupt the fixed encoder-driven
    # breakup approach; only rule-level terminations stop it.
    assert stale.state is Simulation20PointState.APPROACH_CLUSTER
    assert stale.linear_velocity_m_s == pytest.approx(0.1)


@pytest.mark.parametrize(
    "breakup_state, simulation_state",
    (
        (BreakupState.BREAKUP_PUSH, Simulation20PointState.BREAKUP_PUSH),
        (BreakupState.BREAKUP_RELEASE, Simulation20PointState.BREAKUP_RELEASE),
        (
            BreakupState.BREAKUP_OPEN_RETREAT,
            Simulation20PointState.BREAKUP_OPEN_RETREAT,
        ),
        (BreakupState.BREAKUP_CLOSE, Simulation20PointState.BREAKUP_CLOSE),
        (BreakupState.RETREAT, Simulation20PointState.RETREAT_FROM_CLUSTER),
    ),
)
def test_camera_health_staleness_does_not_interrupt_fixed_breakup_action(
    breakup_state: BreakupState,
    simulation_state: Simulation20PointState,
) -> None:
    sequence = make_sequence(breakup=_FixedBreakup(breakup_state))
    start_sequence(sequence)
    entered = sequence.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert entered.state is simulation_state

    continuing = sequence.step(
        3,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.01,
        health=replace(
            SimulationHealth(),
            camera_fresh=False,
            localization_fresh=False,
        ),
    )

    assert continuing.state is simulation_state
    assert continuing.linear_velocity_m_s == pytest.approx(0.1)


def test_camera_health_staleness_still_holds_vision_guided_breakup_approach() -> None:
    sequence = make_sequence(breakup=_ApproachBreakup())
    start_sequence(sequence)
    sequence.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    held = sequence.step(
        3,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), camera_fresh=False),
    )

    # 视觉引导的解团接近在相机陈旧时保持零速但不锁存；新鲜后自动恢复接近。
    assert held.state is Simulation20PointState.APPROACH_CLUSTER
    assert held.reason == "camera_observation_recovering"
    assert held.linear_velocity_m_s == 0.0

    recovered = sequence.step(
        4,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert recovered.state is Simulation20PointState.APPROACH_CLUSTER
    assert recovered.linear_velocity_m_s == pytest.approx(0.1)


def test_stale_health_does_not_interrupt_incomplete_delivery_retreat() -> None:
    sequence = make_sequence(config=runtime_config(retreat_distance_m=0.1))
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.DISENGAGE_AND_RETREAT
    sequence._retreat_base_distance_m = 1.0
    sequence._transport = TransportStatus(
        engaged_track_ids=(1,),
        contact_started_ns=1,
    )

    retreating = sequence.step(
        2,
        perception=None,
        pose=None,
        cumulative_distance_m=1.05,
        health=replace(
            SimulationHealth(),
            camera_fresh=False,
            localization_fresh=False,
        ),
    )

    assert retreating.state is Simulation20PointState.DISENGAGE_AND_RETREAT
    assert retreating.reason == "retreat_after_delivery"
    assert retreating.linear_velocity_m_s == pytest.approx(-0.05)


def test_delivery_retreat_ignores_nonterminal_mission_hold_before_distance() -> None:
    sequence = make_sequence(config=runtime_config(retreat_distance_m=0.1))
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.DISENGAGE_AND_RETREAT
    sequence._retreat_base_distance_m = 1.0
    sequence._transport = TransportStatus(
        engaged_track_ids=(1,),
        contact_started_ns=1,
    )

    retreating = sequence.step(
        2,
        perception=None,
        pose=None,
        cumulative_distance_m=1.05,
    )

    assert retreating.mission_decision is not None
    assert retreating.mission_decision.activity is ActivityState.SAFETY_HOLD
    assert retreating.state is Simulation20PointState.DISENGAGE_AND_RETREAT
    assert retreating.linear_velocity_m_s == pytest.approx(-0.05)


def test_delivery_retreat_requires_fresh_evidence_after_fixed_distance() -> None:
    sequence = make_sequence(config=runtime_config(retreat_distance_m=0.1))
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.DISENGAGE_AND_RETREAT
    sequence._retreat_base_distance_m = 1.0

    held = sequence.step(
        2,
        perception=None,
        pose=None,
        cumulative_distance_m=1.1,
        health=replace(SimulationHealth(), camera_fresh=False),
    )

    # 最低退离距离走完后必须等新鲜证据，但保持当前状态零速等待，不锁存
    # 需要人工恢复的安全保持；证据恢复后自动继续收尾判定。
    assert held.state is Simulation20PointState.DISENGAGE_AND_RETREAT
    assert held.reason == "camera_observation_recovering"
    assert held.linear_velocity_m_s == 0.0


def test_rule_termination_still_stops_incomplete_delivery_retreat() -> None:
    sequence = make_sequence(
        config=runtime_config(retreat_distance_m=0.1),
        match_duration_s=0.05,
    )
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.DISENGAGE_AND_RETREAT
    sequence._retreat_base_distance_m = 1.0

    stopped = sequence.step(
        100_000_001,
        perception=None,
        pose=None,
        cumulative_distance_m=1.05,
    )

    assert stopped.state is Simulation20PointState.TERMINAL_STOP
    assert stopped.linear_velocity_m_s == 0.0
    assert stopped.mission_decision is not None
    assert (
        stopped.mission_decision.termination_reason
        is TerminationReason.MATCH_TIMEOUT
    )


def test_scan_waits_without_latching_hold_and_resumes_when_pose_recovers() -> None:
    sequence = make_sequence(breakup=_FixedBreakup(BreakupState.SCAN_GREEN))
    start_sequence(sequence)

    waiting = sequence.step(
        2,
        perception=None,
        pose=None,
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), localization_fresh=False),
    )

    assert waiting.state is Simulation20PointState.SCAN_GREEN
    assert waiting.reason == "scan_waiting_for_field_pose"
    assert waiting.linear_velocity_m_s == 0.0
    assert waiting.angular_velocity_rad_s == 0.0

    resumed = sequence.step(
        3,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    assert resumed.state is Simulation20PointState.SCAN_GREEN
    assert resumed.reason == "fixed_breakup_action"
    assert resumed.angular_velocity_rad_s == pytest.approx(0.1)


def test_track_reset_and_scan_ignore_missing_field_position_mission_hold() -> None:
    sequence = make_sequence(breakup=_ImmediateGreenBreakup())
    start_sequence(sequence)

    entered_reset = sequence.step(
        2,
        perception=None,
        pose=None,
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), localization_fresh=False),
    )
    assert entered_reset.state is Simulation20PointState.RESET_TARGET_TRACKS

    for frame_sequence, timestamp_ns in ((1, 3), (2, 4)):
        waiting = sequence.step(
            timestamp_ns,
            perception=snapshot(frame_sequence, timestamp_ns),
            pose=None,
            cumulative_distance_m=0.0,
            health=replace(SimulationHealth(), localization_fresh=False),
        )

    assert waiting.state is Simulation20PointState.SCAN_GREEN
    assert waiting.reason == "scan_waiting_for_field_pose"

    still_waiting = sequence.step(
        5,
        perception=snapshot(3, 5),
        pose=None,
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), localization_fresh=False),
    )

    assert still_waiting.state is Simulation20PointState.SCAN_GREEN
    assert still_waiting.reason == "scan_waiting_for_field_pose"
    assert still_waiting.mission_decision is not None
    assert still_waiting.mission_decision.reason == "robot_field_position_missing"
    assert still_waiting.linear_velocity_m_s == 0.0
    assert still_waiting.angular_velocity_rad_s == 0.0


def test_track_reset_counts_each_fresh_frame_only_once() -> None:
    sequence = make_sequence(breakup=_ImmediateGreenBreakup())
    start_sequence(sequence)
    sequence.step(2, perception=None, pose=pose(), cumulative_distance_m=0.0)
    first = snapshot(1, 3)

    settle_first = sequence.step(
        3,
        perception=first,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    repeated = sequence.step(
        4,
        perception=first,
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    assert settle_first.state is Simulation20PointState.RESET_TARGET_TRACKS
    assert repeated.state is Simulation20PointState.RESET_TARGET_TRACKS
    assert repeated.reason == "settle_after_breakup"

    next_frame = sequence.step(
        5,
        perception=snapshot(2, 5),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert next_frame.state is Simulation20PointState.SCAN_GREEN


def test_scan_waits_for_fresh_world_vision_without_latching_hold() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.SCAN_GREEN

    waiting = sequence.step(
        600_000_002,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    assert waiting.state is Simulation20PointState.SCAN_GREEN
    assert waiting.reason == "scan_waiting_for_fresh_vision"
    assert waiting.mission_decision is not None
    assert waiting.mission_decision.reason == "stale_vision"
    assert waiting.linear_velocity_m_s == 0.0
    assert waiting.angular_velocity_rad_s == 0.0

    resumed = sequence.step(
        600_000_003,
        perception=snapshot(1, 600_000_003),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert resumed.state is Simulation20PointState.SCAN_GREEN
    assert resumed.reason == "scan_green"
    assert resumed.angular_velocity_rad_s == pytest.approx(0.2)


def test_rule_termination_still_stops_scan_vision_recovery_wait() -> None:
    sequence = make_sequence(match_duration_s=0.05)
    start_sequence(sequence)
    sequence._breakup = None
    sequence.state = Simulation20PointState.SCAN_GREEN

    stopped = sequence.step(
        100_000_001,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
    )

    assert stopped.state is Simulation20PointState.TERMINAL_STOP
    assert stopped.linear_velocity_m_s == 0.0
    assert stopped.mission_decision is not None
    assert (
        stopped.mission_decision.termination_reason
        is TerminationReason.MATCH_TIMEOUT
    )
