"""单车 20 分模拟赛的受限行为编排与车端入口。

本模块只负责编排现有目标观测、跟踪、世界模型、规则状态机和运动控制。
中心目标团的解团动作仍由 :class:`ClusterBreakupSequence` 提供；本模块不复制
那套定距流程。所有像素和地面点都沿用 perception/geometry 的显式坐标类型，
目标进入场地后才转换为 ``FieldPoint``。
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Protocol, TextIO

from rescue_vision.app.cluster_breakup import (
    BreakupDecision,
    BreakupState,
    ClusterBreakupSequence,
    GripperPosture,
)
from rescue_vision.config import (
    AppConfig,
    Simulation20PointRuntimeConfig,
    load_runtime_config,
)
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import (
    FieldPose2D,
    FusedPoseEstimate,
    FusionQuality,
    normalize_angle,
)
from rescue_vision.mission import (
    AbstractAction,
    ActivityState,
    DeliveryDestination,
    DeliveryEvidence,
    MissionDecision,
    MissionStateMachine,
    SafetySignals,
    TransportStatus,
)
from rescue_vision.perception import PerceptionSnapshot, TargetClass
from rescue_vision.tracking import MultiTargetTracker, TrackStatus
from rescue_vision.world import (
    HazardState,
    RegionKind,
    StaticRegion,
    WorldModel,
    WorldSnapshot,
    WorldTarget,
    WorldUncertainty,
)


class Simulation20PointState(str, Enum):
    """20 分受限流程的应用状态；不替代 mission 包的规则阶段。"""

    BOOT = "boot"
    PREFLIGHT = "preflight"
    LEAVE_START = "leave_start"
    SEARCH_CLUSTER = "search_cluster"
    CENTER_CLUSTER = "center_cluster"
    APPROACH_CLUSTER = "approach_cluster"
    BREAKUP_PUSH = "breakup_push"
    BREAKUP_RELEASE = "breakup_release"
    BREAKUP_OPEN_RETREAT = "breakup_open_retreat"
    BREAKUP_CLOSE = "breakup_close"
    RETREAT_FROM_CLUSTER = "retreat_from_cluster"
    RESET_TARGET_TRACKS = "reset_target_tracks"
    SCAN_GREEN = "scan_green"
    EVALUATE_EASY_GREEN = "evaluate_easy_green"
    SELECT_GREEN = "select_green"
    PLAN_PREPUSH = "plan_prepush"
    NAVIGATE_PREPUSH = "navigate_prepush"
    ALIGN_GREEN = "align_green"
    APPROACH_GREEN = "approach_green"
    ENGAGE_GREEN = "engage_green"
    PUSH_TO_MATERIAL_ZONE = "push_to_material_zone"
    VERIFY_DELIVERY = "verify_delivery"
    DISENGAGE_AND_RETREAT = "disengage_and_retreat"
    UPDATE_PROGRESS = "update_progress"
    PLAN_REBREAKUP = "plan_rebreakup"
    CENTER_REBREAKUP_CLUSTER = "center_rebreakup_cluster"
    TERMINAL_STOP = "terminal_stop"
    FINISH_STOP = "finish_stop"


_FIXED_BREAKUP_STATES = frozenset(
    {
        Simulation20PointState.BREAKUP_PUSH,
        Simulation20PointState.BREAKUP_RELEASE,
        Simulation20PointState.BREAKUP_OPEN_RETREAT,
        Simulation20PointState.BREAKUP_CLOSE,
        Simulation20PointState.RETREAT_FROM_CLUSTER,
    }
)

# Bounded retry window for transient startup evidence before preflight;
# the emergency stop and observe_only gates are never retried.
_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000

# Bounded restart backoff for a disconnected observe_only remote; observation
# health never participates in motion gating.
_REMOTE_RESTART_BACKOFF_NS = 2_000_000_000


@dataclass(frozen=True, slots=True)
class SimulationPreflight:
    """启动前必须同时成立的车端证据。"""

    telemetry_fresh: bool
    watchdog_armed: bool
    emergency_stop_clear: bool
    zero_speed_command_accepted: bool
    camera_observation_fresh: bool
    observe_only_remote: bool

    def __post_init__(self) -> None:
        for name in (
            "telemetry_fresh",
            "watchdog_armed",
            "emergency_stop_clear",
            "zero_speed_command_accepted",
            "camera_observation_fresh",
            "observe_only_remote",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    @property
    def ready(self) -> bool:
        return all(
            (
                self.telemetry_fresh,
                self.watchdog_armed,
                self.emergency_stop_clear,
                self.zero_speed_command_accepted,
                self.camera_observation_fresh,
                self.observe_only_remote,
            )
        )


@dataclass(frozen=True, slots=True)
class SimulationHealth:
    """主循环从各有界旁路读取的最新健康快照。"""

    control_ready: bool = True
    command_accepted: bool = True
    watchdog_armed: bool = True
    emergency_stop_clear: bool = True
    camera_fresh: bool = True
    localization_fresh: bool = True
    observe_only_remote: bool = True
    branch_error: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "control_ready",
            "command_accepted",
            "watchdog_armed",
            "emergency_stop_clear",
            "camera_fresh",
            "localization_fresh",
            "observe_only_remote",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.branch_error is not None and (
            not isinstance(self.branch_error, str) or not self.branch_error.strip()
        ):
            raise ValueError("branch_error must be a non-empty string or None.")

    @property
    def hard_fault_reason(self) -> str | None:
        # Only irrecoverable safety/rule gates terminate: a latched emergency
        # stop and remote privilege loss. Side-path, synchronization and
        # watchdog faults are transient and go through wait_reason instead.
        if not self.emergency_stop_clear:
            return "emergency_stop_latched"
        if not self.observe_only_remote:
            return "remote_is_not_observe_only"
        return None

    @property
    def wait_reason(self) -> str | None:
        if self.branch_error is not None:
            return f"side_path_recovering:{self.branch_error}"
        if not self.control_ready:
            return "motion_control_recovering"
        if not self.command_accepted:
            return "motion_command_recovering"
        if not self.watchdog_armed:
            return "watchdog_rearming"
        return None


class _SimulationRemoteTransport(Protocol):
    def check_health(self) -> None: ...

    def submit_localization(
        self,
        estimate: FusedPoseEstimate | None,
        timestamp_ns: int,
    ) -> None: ...

    def submit(self, frame: object) -> None: ...


def _publish_remote_simulation_state(
    remote_transport: _SimulationRemoteTransport | None,
    *,
    pose: FusedPoseEstimate | None,
    timestamp_ns: int,
    rendered: object | None,
) -> None:
    """提交本周期最新定位和可选 perception 帧，不复制定位源。

    定位旁路暂不可用时提交 ``None``，观察端按 ``robot_localized=false``
    发布，不沿用过期坐标。
    """

    if remote_transport is None:
        return
    remote_transport.check_health()
    remote_transport.submit_localization(pose, timestamp_ns)
    if rendered is not None:
        remote_transport.submit(rendered)


class _TeeStream:
    """把标准流同时写到控制台和日志文件；flush 同步刷新两侧。"""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        primary_count = self._primary.write(text)
        self._secondary.write(text)
        return primary_count

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()


def _begin_time_named_log(
    log_dir: Path | None,
) -> tuple[TextIO | None, TextIO, TextIO]:
    """可选地把 stdout/stderr tee 到 ``log_dir/<YYYYmmdd_HHMM>.log``。

    返回 ``(日志流, 原 stdout, 原 stderr)``；``log_dir`` 为 ``None`` 时不动
    标准流并返回 ``(None, sys.stdout, sys.stderr)``。日志文件按行缓冲追加，
    同分钟重跑会继续写入同一文件；进程异常退出时已写入内容仍在文件中。
    """

    if log_dir is None:
        return None, sys.stdout, sys.stderr
    if not isinstance(log_dir, Path):
        raise TypeError("log_dir must be a Path or None.")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{time.strftime('%Y%m%d_%H%M')}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(original_stdout, stream)
    sys.stderr = _TeeStream(original_stderr, stream)
    print(f"logging to {path}", flush=True)
    return stream, original_stdout, original_stderr


def _end_time_named_log(
    log_stream: TextIO | None,
    original_stdout: TextIO,
    original_stderr: TextIO,
) -> None:
    """恢复标准流并关闭按时间命名的日志文件。"""

    if log_stream is None:
        return
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_stream.close()


@dataclass(frozen=True, slots=True)
class GreenTransportPlan:
    """一枚绿色普通物资的目标场地点、预推点和安全推送走廊。"""

    track_id: int
    target_field: FieldPoint
    destination_field: FieldPoint
    push_direction_x: float
    push_direction_y: float
    prepush_field: FieldPoint
    clearance_mm: float


@dataclass(frozen=True, slots=True)
class SimulationDecision:
    timestamp_ns: int
    state: Simulation20PointState
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    gripper_posture: GripperPosture
    reason: str
    selected_track_id: int | None
    valid_green_deliveries: int
    score_points: int
    mission_decision: MissionDecision | None = None
    world_snapshot: WorldSnapshot | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if not isinstance(self.state, Simulation20PointState):
            raise ValueError("state must be a Simulation20PointState.")
        for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not isinstance(self.gripper_posture, GripperPosture):
            raise ValueError("gripper_posture must be a GripperPosture.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string.")
        if self.selected_track_id is not None and (
            isinstance(self.selected_track_id, bool)
            or not isinstance(self.selected_track_id, int)
            or self.selected_track_id <= 0
        ):
            raise ValueError("selected_track_id must be a positive integer or None.")
        if (
            isinstance(self.valid_green_deliveries, bool)
            or not isinstance(self.valid_green_deliveries, int)
            or not 0 <= self.valid_green_deliveries <= 4
        ):
            raise ValueError("valid_green_deliveries must be an integer in [0, 4].")
        if (
            isinstance(self.score_points, bool)
            or not isinstance(self.score_points, int)
            or not 0 <= self.score_points <= 20
        ):
            raise ValueError("score_points must be an integer in [0, 20].")
        if self.score_points != self.valid_green_deliveries * 5:
            raise ValueError("score_points must equal five points per green delivery.")
        if self.mission_decision is not None and not isinstance(
            self.mission_decision, MissionDecision
        ):
            raise ValueError("mission_decision must be a MissionDecision or None.")
        if self.world_snapshot is not None and not isinstance(
            self.world_snapshot, WorldSnapshot
        ):
            raise ValueError("world_snapshot must be a WorldSnapshot or None.")


class _BreakupDriver(Protocol):
    def step(
        self,
        *,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        perception: PerceptionSnapshot | None,
    ) -> BreakupDecision: ...


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _distance(first: FieldPoint, second: FieldPoint) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _field_from_robot(pose: FieldPose2D, point: GroundPoint) -> FieldPoint:
    """Convert one robot-ground point to the field frame using the pose."""

    cosine = math.cos(pose.heading_rad)
    sine = math.sin(pose.heading_rad)
    return FieldPoint(
        pose.position.x + cosine * point.x - sine * point.y,
        pose.position.y + sine * point.x + cosine * point.y,
    )


def _angle_to(start: FieldPoint, end: FieldPoint) -> float:
    return math.atan2(end.y - start.y, end.x - start.x)


def _polygon_centroid(region: StaticRegion) -> FieldPoint:
    points = region.polygon_field
    area_twice = sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:] + points[:1])
    )
    if math.isclose(area_twice, 0.0, abs_tol=1e-9):
        return FieldPoint(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )
    factor = 1.0 / (3.0 * area_twice)
    return FieldPoint(
        factor
        * sum(
            (first.x + second.x)
            * (first.x * second.y - second.x * first.y)
            for first, second in zip(points, points[1:] + points[:1])
        ),
        factor
        * sum(
            (first.y + second.y)
            * (first.x * second.y - second.x * first.y)
            for first, second in zip(points, points[1:] + points[:1])
        ),
    )


def _distance_to_segment(point: FieldPoint, start: FieldPoint, end: FieldPoint) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return _distance(point, start)
    ratio = _clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared,
        0.0,
        1.0,
    )
    projection = FieldPoint(start.x + ratio * dx, start.y + ratio * dy)
    return _distance(point, projection)


def _inside_inset(region: StaticRegion, point: FieldPoint, inset_mm: float) -> bool:
    if not region.contains(point):
        return False
    points = region.polygon_field
    edge_distance = min(
        _distance_to_segment(point, first, second)
        for first, second in zip(points, points[1:] + points[:1])
    )
    return edge_distance >= inset_mm


class Simulation20PointSequence:
    """可重放的 20 分模拟赛行为状态机。

    ``step()`` 只读取已经完成的最新快照并返回轻量控制意图；它不等待相机、
    推理、网络或 UART。车端入口负责把这些意图提交给已有的
    ``MotionController``，并在旁路故障时传入 ``SimulationHealth``。
    """

    def __init__(
        self,
        config: Simulation20PointRuntimeConfig,
        *,
        tracker: MultiTargetTracker,
        world_model: WorldModel,
        mission: MissionStateMachine,
        breakup: _BreakupDriver,
        breakup_factory: Callable[[], _BreakupDriver] | None = None,
    ) -> None:
        if not isinstance(config, Simulation20PointRuntimeConfig):
            raise TypeError("config must be a Simulation20PointRuntimeConfig.")
        if not config.enabled:
            raise ValueError("Simulation20PointSequence requires enabled config.")
        if not isinstance(tracker, MultiTargetTracker):
            raise TypeError("tracker must be a MultiTargetTracker.")
        if not isinstance(world_model, WorldModel):
            raise TypeError("world_model must be a WorldModel.")
        if not isinstance(mission, MissionStateMachine):
            raise TypeError("mission must be a MissionStateMachine.")
        if not callable(getattr(breakup, "step", None)):
            raise TypeError("breakup must provide step().")
        if breakup_factory is not None and not callable(breakup_factory):
            raise TypeError("breakup_factory must be callable or None.")
        self.config = config
        self._tracker = tracker
        self._world_model = world_model
        self._mission = mission
        self._breakup: _BreakupDriver | None = breakup
        self._breakup_factory = breakup_factory
        self.state = Simulation20PointState.BOOT
        self._started = False
        self._start_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._last_motion_timestamp_ns = 0
        self._last_visual_timestamp_ns = 0
        self._last_tracker_frame_sequence: int | None = None
        self._ignore_frames_through: int | None = None
        self._world_snapshot: WorldSnapshot | None = None
        self._last_mission_decision: MissionDecision | None = None
        self._selected_track_id: int | None = None
        self._selected_plan: GreenTransportPlan | None = None
        self._transport = TransportStatus()
        self._delivery_confirm_count = 0
        self._disengage_confirm_count = 0
        self._delivery_sequence = 0
        self._completed_target_points: list[FieldPoint] = []
        self._current_breakup_attempts = 0
        self._total_breakup_attempts = 1
        self._settle_confirm_count = 0
        self._last_settle_frame_sequence: int | None = None
        self._scan_start_heading: float | None = None
        self._scan_last_heading: float | None = None
        self._scan_heading_span = 0.0
        self._nav_stage = "heading"
        self._retreat_base_distance_m: float | None = None
        self._rebreakup_mode = False
        self._cycle_mission_decision: MissionDecision | None = None

    @classmethod
    def from_app_config(cls, config: AppConfig) -> Simulation20PointSequence:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig.")
        runtime = config.simulation_20_point
        if not runtime.enabled:
            raise ValueError("simulation_20_point.enabled must be true.")
        gripper = config.motion.gripper.build_calibration()
        if gripper is None:
            raise RuntimeError("20-point flow requires enabled gripper calibration.")

        def make_breakup() -> ClusterBreakupSequence:
            return ClusterBreakupSequence(
                config.motion.cluster_breakup,
                gripper_full_travel_time_s=gripper.full_travel_time_s,
            )

        return cls(
            runtime,
            tracker=config.tracking.build_tracker(),
            world_model=config.world.build_model(),
            mission=config.mission.build_state_machine(),
            breakup=make_breakup(),
            breakup_factory=make_breakup,
        )

    @property
    def valid_green_deliveries(self) -> int:
        return min(self._mission.progress.delivered_green_supply, 4)

    @property
    def score_points(self) -> int:
        return self.valid_green_deliveries * 5

    @property
    def selected_track_id(self) -> int | None:
        return self._selected_track_id

    @property
    def world_snapshot(self) -> WorldSnapshot | None:
        return self._world_snapshot

    @property
    def mission(self) -> MissionStateMachine:
        return self._mission

    def preflight(
        self,
        timestamp_ns: int,
        checks: SimulationPreflight,
    ) -> SimulationDecision:
        self._validate_timestamp(timestamp_ns)
        if self.state is not Simulation20PointState.BOOT:
            raise RuntimeError("preflight() can only be called from BOOT.")
        if not isinstance(checks, SimulationPreflight):
            raise TypeError("checks must be a SimulationPreflight.")
        if not checks.ready:
            self.state = Simulation20PointState.TERMINAL_STOP
            return self._decision(timestamp_ns, 0.0, 0.0, "preflight_failed")
        self.state = Simulation20PointState.PREFLIGHT
        self._last_timestamp_ns = timestamp_ns
        return self._decision(timestamp_ns, 0.0, 0.0, "preflight_ready")

    def start(self, timestamp_ns: int) -> SimulationDecision:
        self._validate_timestamp(timestamp_ns)
        if self.state is not Simulation20PointState.PREFLIGHT:
            raise RuntimeError("start() requires a successful PREFLIGHT.")
        mission_decision = self._mission.start(timestamp_ns)
        self._last_mission_decision = mission_decision
        self._started = True
        self._start_timestamp_ns = timestamp_ns
        self._last_timestamp_ns = timestamp_ns
        self._last_motion_timestamp_ns = timestamp_ns
        self.state = Simulation20PointState.LEAVE_START
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "mission_started_one_button",
            mission_decision=mission_decision,
        )

    def step(
        self,
        timestamp_ns: int,
        *,
        perception: PerceptionSnapshot | None,
        pose: FusedPoseEstimate | None,
        cumulative_distance_m: float | None,
        safety: SafetySignals | None = None,
        health: SimulationHealth | None = None,
    ) -> SimulationDecision:
        """消费一个已经完成的快照并返回本周期控制意图。"""

        self._validate_timestamp(timestamp_ns)
        if not self._started:
            return self._decision(timestamp_ns, 0.0, 0.0, "waiting_one_button_start")
        if pose is not None and not isinstance(pose, FusedPoseEstimate):
            raise TypeError("pose must be a FusedPoseEstimate or None.")
        if perception is not None and not isinstance(perception, PerceptionSnapshot):
            raise TypeError("perception must be a PerceptionSnapshot or None.")
        if cumulative_distance_m is not None and not math.isfinite(
            float(cumulative_distance_m)
        ):
            raise ValueError("cumulative_distance_m must be finite when present.")
        if safety is None:
            safety = SafetySignals.nominal(self._last_motion_timestamp_ns)
        if not isinstance(safety, SafetySignals):
            raise TypeError("safety must be a SafetySignals or None.")
        if health is None:
            health = SimulationHealth()
        if not isinstance(health, SimulationHealth):
            raise TypeError("health must be a SimulationHealth or None.")

        if self.state in {
            Simulation20PointState.TERMINAL_STOP,
            Simulation20PointState.FINISH_STOP,
        }:
            return self._decision(timestamp_ns, 0.0, 0.0, "stop_is_latched")
        hard_fault = health.hard_fault_reason
        if hard_fault is not None:
            return self._terminal(timestamp_ns, hard_fault)
        health_wait = health.wait_reason
        if health_wait is not None:
            return self._hold(timestamp_ns, health_wait)
        vision_independent_action = self._vision_independent_action_active(
            cumulative_distance_m
        )
        scan_recovery_state = self.state in {
            Simulation20PointState.RESET_TARGET_TRACKS,
            Simulation20PointState.SCAN_GREEN,
        }
        if (
            not health.camera_fresh
            and self.state is not Simulation20PointState.LEAVE_START
            and not vision_independent_action
        ):
            return self._hold(timestamp_ns, "camera_observation_recovering")
        if not health.localization_fresh and self.state not in {
            Simulation20PointState.LEAVE_START,
            Simulation20PointState.SEARCH_CLUSTER,
            Simulation20PointState.CENTER_CLUSTER,
            Simulation20PointState.APPROACH_CLUSTER,
            Simulation20PointState.RESET_TARGET_TRACKS,
            Simulation20PointState.SCAN_GREEN,
        } and not vision_independent_action:
            return self._hold(timestamp_ns, "localization_recovering")
        direct_safety = self._direct_safety_reason(safety)
        if direct_safety is not None:
            return self._terminal(timestamp_ns, direct_safety)

        try:
            snapshot = self._update_world(timestamp_ns, perception, pose)
        except (RuntimeError, ValueError) as exc:
            # A single inconsistent snapshot (timestamp skew, malformed
            # observation) must not latch a permanent stop: keep the current
            # state at zero speed and retry the world update next cycle.
            return self._hold(timestamp_ns, f"world_update_recovering:{exc}")

        ignore_nonterminal_mission_hold = (
            vision_independent_action
            or scan_recovery_state
            or self._breakup is not None
            or self.state
            in {
                Simulation20PointState.RESET_TARGET_TRACKS,
                Simulation20PointState.UPDATE_PROGRESS,
            }
        )
        mission_decision = self._evaluate_mission(snapshot, safety)
        if mission_decision.terminal:
            return self._terminal(
                timestamp_ns,
                f"mission_terminated:{mission_decision.reason}",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if (
            mission_decision.activity
            in {ActivityState.SAFETY_HOLD, ActivityState.AVOIDING}
            and not ignore_nonterminal_mission_hold
        ):
            # Holds never latch a separate state: the current behaviour state
            # is kept at zero speed and the next cycle re-evaluates mission
            # and health, so recovery is automatic when evidence returns.
            # Breakup actions and the minimum delivery-retreat distance
            # intentionally ignore non-terminal mission activity instead of
            # interrupting their fixed encoder-driven actions.
            return self._hold(timestamp_ns, mission_decision.reason)
        if self.state is Simulation20PointState.RESET_TARGET_TRACKS:
            return self._step_reset_tracks(timestamp_ns, perception, pose, snapshot)
        if self._breakup is not None:
            return self._step_breakup(
                timestamp_ns,
                perception,
                pose,
                cumulative_distance_m,
                snapshot,
            )
        if self.state is Simulation20PointState.UPDATE_PROGRESS:
            # Delivery evidence has already been accepted by mission. Reset
            # the post-delivery track set before a coasting copy of the old
            # target can be interpreted as a new suspected hazard.
            return self._step_update_progress(timestamp_ns, pose, snapshot)

        if self.state is Simulation20PointState.SCAN_GREEN:
            return self._step_scan(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.EVALUATE_EASY_GREEN:
            return self._step_evaluate(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.SELECT_GREEN:
            return self._step_select(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.PLAN_PREPUSH:
            return self._step_plan_prepush(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.NAVIGATE_PREPUSH:
            return self._step_navigate(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.ALIGN_GREEN:
            return self._step_align(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.APPROACH_GREEN:
            return self._step_approach(timestamp_ns, pose, snapshot, mission_decision)
        if self.state is Simulation20PointState.ENGAGE_GREEN:
            return self._step_engage(
                timestamp_ns, pose, snapshot, cumulative_distance_m, safety
            )
        if self.state is Simulation20PointState.PUSH_TO_MATERIAL_ZONE:
            return self._step_push(timestamp_ns, pose, snapshot, safety)
        if self.state is Simulation20PointState.VERIFY_DELIVERY:
            return self._step_verify(timestamp_ns, pose, snapshot, safety)
        if self.state is Simulation20PointState.DISENGAGE_AND_RETREAT:
            return self._step_disengage(
                timestamp_ns, snapshot, cumulative_distance_m, safety
            )
        if self.state is Simulation20PointState.PLAN_REBREAKUP:
            return self._step_plan_rebreakup(timestamp_ns, snapshot)
        if self.state is Simulation20PointState.CENTER_REBREAKUP_CLUSTER:
            return self._step_breakup(
                timestamp_ns,
                perception,
                pose,
                cumulative_distance_m,
                snapshot,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            f"unhandled_state:{self.state.value}",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _update_world(
        self,
        timestamp_ns: int,
        perception: PerceptionSnapshot | None,
        pose: FusedPoseEstimate | None,
    ) -> WorldSnapshot:
        if perception is not None and perception.capture_timestamp_ns > timestamp_ns:
            raise ValueError("perception capture timestamp is in the future.")
        if (
            perception is not None
            and perception.dropped_stale_age_ms is None
            and (
                self._last_tracker_frame_sequence is None
                or perception.frame_sequence > self._last_tracker_frame_sequence
            )
            and (
                self._ignore_frames_through is None
                or perception.frame_sequence > self._ignore_frames_through
            )
        ):
            self._tracker.update(
                perception.capture_timestamp_ns,
                perception.observations,
            )
            self._last_tracker_frame_sequence = perception.frame_sequence
            self._last_visual_timestamp_ns = perception.capture_timestamp_ns
        elif perception is not None and perception.dropped_stale_age_ms is None:
            self._last_visual_timestamp_ns = max(
                self._last_visual_timestamp_ns,
                perception.capture_timestamp_ns,
            )

        robot_field_point: FieldPoint | None = None
        if pose is not None and pose.pose is not None:
            robot_field_point = pose.pose.position
        target_field_points: dict[int, FieldPoint] = {}
        if pose is not None and pose.pose is not None:
            for track in self._tracker.tracks:
                if track.ground_point is not None:
                    target_field_points[track.track_id] = _field_from_robot(
                        pose.pose,
                        track.ground_point,
                    )
        visual_timestamp_ns = min(self._last_visual_timestamp_ns, timestamp_ns)
        snapshot = self._world_model.update(
            timestamp_ns=timestamp_ns,
            visual_timestamp_ns=visual_timestamp_ns,
            tracks=self._tracker.tracks,
            robot_field_point=robot_field_point,
            target_field_points=target_field_points,
        )
        self._world_snapshot = snapshot
        return snapshot

    def _step_breakup(
        self,
        timestamp_ns: int,
        perception: PerceptionSnapshot | None,
        pose: FusedPoseEstimate | None,
        cumulative_distance_m: float | None,
        snapshot: WorldSnapshot,
    ) -> SimulationDecision:
        breakup = self._breakup
        if breakup is None:
            raise RuntimeError("breakup driver is missing while breakup is active.")
        breakup_decision = breakup.step(
            timestamp_ns=timestamp_ns,
            cumulative_distance_m=cumulative_distance_m,
            perception=perception,
        )
        if not isinstance(breakup_decision, BreakupDecision):
            raise TypeError("breakup.step() must return a BreakupDecision.")
        if breakup_decision.state is BreakupState.FAULT:
            # A transient breakup fault (lost cluster, odometry stall, phase
            # timeout) aborts the current fixed action instead of ending the
            # round: reset the pre-collision tracks and fall back to the
            # canonical scan path, which re-plans a bounded rebreakup only
            # when no easy green exists. Rule terminations still apply.
            self._begin_reset_tracks(perception)
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"breakup_fault_reset_tracks:{breakup_decision.reason}",
                posture=breakup_decision.gripper_posture,
                world_snapshot=snapshot,
            )
        if breakup_decision.state is BreakupState.GREEN_FOUND:
            self._begin_reset_tracks(perception)
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "breakup_green_found_reset_tracks",
                posture=breakup_decision.gripper_posture,
                world_snapshot=snapshot,
            )
        if breakup_decision.state is BreakupState.SCAN_GREEN:
            self.state = Simulation20PointState.SCAN_GREEN
            if pose is None or pose.pose is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "scan_waiting_for_field_pose",
                    posture=breakup_decision.gripper_posture,
                    world_snapshot=snapshot,
                )
            if self._scan_span_reached(pose):
                self._begin_reset_tracks(perception)
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "breakup_scan_span_complete_reset_tracks",
                    posture=breakup_decision.gripper_posture,
                    world_snapshot=snapshot,
                )
        else:
            self.state = {
                BreakupState.WAIT_ODOMETRY: Simulation20PointState.LEAVE_START,
                BreakupState.LEAVE_START: Simulation20PointState.LEAVE_START,
                BreakupState.SEARCH_CLUSTER: Simulation20PointState.SEARCH_CLUSTER,
                BreakupState.CENTER_CLUSTER: Simulation20PointState.CENTER_CLUSTER,
                BreakupState.APPROACH_CLUSTER: Simulation20PointState.APPROACH_CLUSTER,
                BreakupState.BREAKUP_PUSH: Simulation20PointState.BREAKUP_PUSH,
                BreakupState.BREAKUP_RELEASE: Simulation20PointState.BREAKUP_RELEASE,
                BreakupState.BREAKUP_OPEN_RETREAT: (
                    Simulation20PointState.BREAKUP_OPEN_RETREAT
                ),
                BreakupState.BREAKUP_CLOSE: Simulation20PointState.BREAKUP_CLOSE,
                BreakupState.RETREAT: Simulation20PointState.RETREAT_FROM_CLUSTER,
                BreakupState.SCAN_GREEN: Simulation20PointState.SCAN_GREEN,
            }[breakup_decision.state]
            if self._rebreakup_mode and self.state not in {
                Simulation20PointState.RETREAT_FROM_CLUSTER,
                Simulation20PointState.BREAKUP_PUSH,
                Simulation20PointState.BREAKUP_RELEASE,
                Simulation20PointState.BREAKUP_OPEN_RETREAT,
                Simulation20PointState.BREAKUP_CLOSE,
            }:
                self.state = Simulation20PointState.CENTER_REBREAKUP_CLUSTER
        return self._decision(
            timestamp_ns,
            breakup_decision.linear_velocity_m_s,
            breakup_decision.angular_velocity_rad_s,
            breakup_decision.reason,
            posture=breakup_decision.gripper_posture,
            world_snapshot=snapshot,
        )

    def _begin_reset_tracks(self, perception: PerceptionSnapshot | None) -> None:
        self._ignore_frames_through = self._last_tracker_frame_sequence
        if perception is not None:
            self._ignore_frames_through = max(
                self._ignore_frames_through or -1,
                perception.frame_sequence,
            )
        self._tracker.reset()
        self._world_model.reset()
        self._last_tracker_frame_sequence = None
        self._last_visual_timestamp_ns = 0
        self._world_snapshot = None
        self._settle_confirm_count = 0
        self._last_settle_frame_sequence = None
        self._breakup = None
        self._rebreakup_mode = False
        self._selected_track_id = None
        self._selected_plan = None
        self._transport = TransportStatus()
        self._delivery_confirm_count = 0
        self._disengage_confirm_count = 0
        self._scan_start_heading = None
        self._scan_last_heading = None
        self._scan_heading_span = 0.0
        self.state = Simulation20PointState.RESET_TARGET_TRACKS

    def _step_reset_tracks(
        self,
        timestamp_ns: int,
        perception: PerceptionSnapshot | None,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
    ) -> SimulationDecision:
        del snapshot
        if (
            perception is not None
            and perception.dropped_stale_age_ms is None
            and (
                self._ignore_frames_through is None
                or perception.frame_sequence > self._ignore_frames_through
            )
            and perception.frame_sequence != self._last_settle_frame_sequence
        ):
            self._settle_confirm_count += 1
            self._last_settle_frame_sequence = perception.frame_sequence
        if self._settle_confirm_count < self.config.breakup_settle_confirm_frames:
            return self._decision(timestamp_ns, 0.0, 0.0, "settle_after_breakup")
        self.state = Simulation20PointState.SCAN_GREEN
        self._scan_start_heading = None
        self._scan_last_heading = None
        self._scan_heading_span = 0.0
        if pose is None or pose.pose is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "scan_waiting_for_field_pose",
            )
        return self._decision(
            timestamp_ns,
            0.0,
            self._scan_velocity(),
            "scan_green_after_track_reset",
        )

    def _scan_span_reached(self, pose: FusedPoseEstimate | None) -> bool:
        if pose is None or pose.pose is None:
            return False
        heading = pose.pose.heading_rad
        if self._scan_last_heading is None:
            self._scan_start_heading = heading
            self._scan_last_heading = heading
            return False
        self._scan_heading_span += abs(normalize_angle(heading - self._scan_last_heading))
        self._scan_last_heading = heading
        return self._scan_heading_span >= self.config.scan_min_heading_span_rad

    def _scan_velocity(self) -> float:
        # 带符号角速度：左转为正，右转为负，符号由配置直接决定。
        return self.config.scan_angular_velocity_rad_s

    def _step_scan(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        if WorldUncertainty.STALE_VISION in snapshot.uncertainties:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "scan_waiting_for_fresh_vision",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if pose is None or pose.pose is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "scan_waiting_for_field_pose",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if self._scan_span_reached(pose):
            self.state = Simulation20PointState.EVALUATE_EASY_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "scan_heading_span_complete",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            self._scan_velocity(),
            "scan_green",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_evaluate(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        if pose is None or pose.pose is None:
            return self._hold(timestamp_ns, "candidate_selection_pose_uncertain")
        if not self._pose_usable(pose):
            # A pose that exists but exceeds the uncertainty gates can never
            # verify an easy green, so holding in EVALUATE would deadlock a
            # dead-reckoning-only run: treat it as "no easy green" and let
            # PLAN_REBREAKUP decide under its own attempt/geometry/progress
            # bounds. Candidate selection and navigation still require a
            # qualified pose.
            self.state = Simulation20PointState.PLAN_REBREAKUP
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "no_easy_green_pose_uncertain",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        plans = self._candidate_plans(snapshot, pose.pose)
        if plans:
            self._selected_plan = plans[0]
            self._selected_track_id = plans[0].track_id
            self.state = Simulation20PointState.SELECT_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"green_candidate_selected:{plans[0].track_id}",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        self.state = Simulation20PointState.PLAN_REBREAKUP
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "no_easy_green_candidate",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_select(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        if self._selected_track_id is None or not self._pose_usable(pose):
            self.state = Simulation20PointState.EVALUATE_EASY_GREEN
            return self._decision(timestamp_ns, 0.0, 0.0, "green_selection_cancelled")
        plan = self._find_plan(snapshot, pose.pose, self._selected_track_id)
        if plan is None:
            self._selected_track_id = None
            self._selected_plan = None
            self.state = Simulation20PointState.EVALUATE_EASY_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "selected_green_no_longer_safe",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        self._selected_plan = plan
        self.state = Simulation20PointState.PLAN_PREPUSH
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "green_selected_lock_track",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_plan_prepush(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        if self._selected_track_id is None or not self._pose_usable(pose):
            return self._hold(timestamp_ns, "prepush_pose_uncertain")
        plan = self._find_plan(snapshot, pose.pose, self._selected_track_id)
        if plan is None:
            self._selected_track_id = None
            self._selected_plan = None
            self.state = Simulation20PointState.EVALUATE_EASY_GREEN
            return self._decision(timestamp_ns, 0.0, 0.0, "prepush_plan_blocked")
        self._selected_plan = plan
        self._nav_stage = "heading"
        self.state = Simulation20PointState.NAVIGATE_PREPUSH
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "prepush_plan_ready",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_navigate(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        if self._selected_track_id is None:
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "navigate_target_missing_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if not self._pose_usable(pose):
            return self._hold(timestamp_ns, "navigate_pose_uncertain")
        plan = self._find_plan(snapshot, pose.pose, self._selected_track_id)
        if plan is None:
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "navigate_corridor_blocked_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        self._selected_plan = plan
        current = pose.pose.position
        distance = _distance(current, plan.prepush_field)
        desired_heading = _angle_to(current, plan.prepush_field)
        heading_error = normalize_angle(desired_heading - pose.pose.heading_rad)
        if self._nav_stage == "heading":
            if abs(heading_error) <= self.config.navigation_heading_tolerance_rad:
                self._nav_stage = "straight"
            else:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    _clamp(
                        self.config.navigation_angular_kp_rad_s * heading_error,
                        -self.config.navigation_max_angular_velocity_rad_s,
                        self.config.navigation_max_angular_velocity_rad_s,
                    ),
                    "rotate_to_prepush",
                    mission_decision=mission_decision,
                    world_snapshot=snapshot,
                )
        if distance <= self.config.navigation_position_tolerance_mm:
            self.state = Simulation20PointState.ALIGN_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "arrived_prepush_align_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        return self._decision(
            timestamp_ns,
            self.config.navigation_speed_m_s,
            _clamp(
                self.config.navigation_angular_kp_rad_s * heading_error,
                -self.config.navigation_max_angular_velocity_rad_s,
                self.config.navigation_max_angular_velocity_rad_s,
            ),
            "navigate_to_prepush",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_align(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        target = self._selected_target(snapshot)
        if target is None:
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "align_target_missing_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if target.ground_point is None:
            return self._hold(timestamp_ns, "align_target_missing")
        if (
            target.target_class is not TargetClass.GREEN_SUPPLY
            or target.hazard_state is not HazardState.CLEAR
        ):
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "align_target_became_hazard_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if not self._target_is_fresh(target, timestamp_ns):
            return self._hold(timestamp_ns, "align_target_stale")
        angle = math.atan2(target.ground_point.y, target.ground_point.x)
        if abs(angle) <= self.config.navigation_heading_tolerance_rad:
            self.state = Simulation20PointState.APPROACH_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_centerline_aligned",
                posture=GripperPosture.TRANSPORT,
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            _clamp(
                self.config.alignment_kp_rad_s * angle,
                -self.config.alignment_max_angular_velocity_rad_s,
                self.config.alignment_max_angular_velocity_rad_s,
            ),
            "align_green_centerline",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_approach(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        mission_decision: MissionDecision | None,
    ) -> SimulationDecision:
        del pose
        target = self._selected_target(snapshot)
        if target is None:
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "approach_target_missing_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if target.ground_point is None:
            return self._hold(timestamp_ns, "approach_target_missing")
        if (
            target.target_class is not TargetClass.GREEN_SUPPLY
            or target.hazard_state is not HazardState.CLEAR
        ):
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "approach_target_became_hazard_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if not self._target_is_fresh(target, timestamp_ns):
            return self._hold(timestamp_ns, "approach_target_stale")
        conflict = self._contact_conflict(snapshot, target)
        if conflict:
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "approach_corridor_blocked_reselect_green",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        ground = target.ground_point
        angle = math.atan2(ground.y, ground.x)
        if ground.x <= self.config.engage_distance_mm:
            self.state = Simulation20PointState.ENGAGE_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_contact_distance_reached",
                posture=GripperPosture.TRANSPORT,
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        return self._decision(
            timestamp_ns,
            self.config.approach_speed_m_s,
            _clamp(
                self.config.alignment_kp_rad_s * angle,
                -self.config.alignment_max_angular_velocity_rad_s,
                self.config.alignment_max_angular_velocity_rad_s,
            ),
            "approach_selected_green",
            posture=GripperPosture.TRANSPORT,
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_engage(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        cumulative_distance_m: float | None,
        safety: SafetySignals,
    ) -> SimulationDecision:
        del pose
        target = self._selected_target(snapshot)
        if target is None:
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "engage_target_missing_no_transport",
                world_snapshot=snapshot,
            )
        if target.ground_point is None:
            return self._hold(timestamp_ns, "engage_target_missing")
        if (
            target.target_class is not TargetClass.GREEN_SUPPLY
            or target.hazard_state is not HazardState.CLEAR
        ):
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "engage_target_became_hazard_no_transport",
                world_snapshot=snapshot,
            )
        if self._contact_conflict(snapshot, target):
            self._cancel_selected_green()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "engage_corridor_blocked_no_transport",
                world_snapshot=snapshot,
            )
        if not self._geometric_contact_is_valid(target):
            self.state = Simulation20PointState.APPROACH_GREEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "engage_evidence_not_ready",
                posture=GripperPosture.TRANSPORT,
            )
        assert self._selected_track_id is not None
        self._transport = TransportStatus(
            engaged_track_ids=(self._selected_track_id,),
            contact_started_ns=timestamp_ns,
        )
        # Contact establishment is a boundary event like delivery evidence:
        # re-evaluate within the same cycle so the rule machine judges the
        # engagement with the fresh TransportStatus instead of the stale
        # pre-contact decision from the top of step().
        mission_decision = self._mission.step(
            snapshot,
            transport=self._transport,
            safety=safety,
        )
        self._last_mission_decision = mission_decision
        self._cycle_mission_decision = mission_decision
        if mission_decision.terminal or mission_decision.action not in {
            AbstractAction.PUSH,
            AbstractAction.DELIVER,
        }:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "mission_rejected_geometric_contact",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        self._retreat_base_distance_m = cumulative_distance_m
        self.state = Simulation20PointState.PUSH_TO_MATERIAL_ZONE
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "single_green_contact_engaged",
            posture=GripperPosture.CLOSED,
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_push(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        safety: SafetySignals,
    ) -> SimulationDecision:
        target = self._selected_target(snapshot)
        if target is None or target.ground_point is None:
            return self._hold(timestamp_ns, "pushing_target_missing")
        if (
            target.target_class is not TargetClass.GREEN_SUPPLY
            or target.hazard_state is not HazardState.CLEAR
        ):
            return self._hold(timestamp_ns, "pushing_target_not_clear")
        if self._contact_conflict(snapshot, target):
            return self._hold(timestamp_ns, "second_target_during_push")
        if not self._target_is_fresh(target, timestamp_ns):
            return self._hold(timestamp_ns, "pushing_target_stale")
        if pose is None or pose.pose is None:
            return self._hold(timestamp_ns, "pushing_pose_missing")
        mission_decision = self._cycle_mission_decision
        assert mission_decision is not None
        if mission_decision.action not in {AbstractAction.PUSH, AbstractAction.DELIVER}:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "mission_did_not_allow_push",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if self._target_fully_entered(snapshot, target):
            self.state = Simulation20PointState.VERIFY_DELIVERY
            self._delivery_confirm_count = 0
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "material_zone_reached_verify_delivery",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        plan = self._find_plan(snapshot, pose, self._selected_track_id)
        if plan is None:
            if self._selected_plan is None or target.field_point is None:
                return self._hold(timestamp_ns, "pushing_corridor_blocked")
            if not self._segment_safe(
                snapshot,
                target.field_point,
                self._selected_plan.destination_field,
                ignored_track_id=target.track_id,
            ):
                return self._hold(timestamp_ns, "pushing_corridor_blocked")
        angle = math.atan2(target.ground_point.y, target.ground_point.x)
        return self._decision(
            timestamp_ns,
            self.config.push_speed_m_s,
            _clamp(
                self.config.push_angular_kp_rad_s * angle,
                -self.config.push_max_angular_velocity_rad_s,
                self.config.push_max_angular_velocity_rad_s,
            ),
            "push_single_green_to_material_zone",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_verify(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
        safety: SafetySignals,
    ) -> SimulationDecision:
        target = self._selected_target(snapshot)
        if target is None or not self._target_is_fresh(target, timestamp_ns):
            return self._hold(timestamp_ns, "delivery_target_missing")
        mission_decision = self._cycle_mission_decision
        assert mission_decision is not None
        if not self._target_fully_entered(snapshot, target):
            self._delivery_confirm_count = 0
            return self._decision(
                timestamp_ns,
                self.config.push_speed_m_s,
                0.0,
                "delivery_not_fully_entered",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        self._delivery_confirm_count += 1
        if self._delivery_confirm_count < self.config.delivery_confirm_frames:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "delivery_evidence_confirming",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if pose is None or pose.pose is None:
            return self._hold(timestamp_ns, "delivery_pose_missing")
        self._delivery_sequence += 1
        assert self._selected_track_id is not None
        evidence = DeliveryEvidence(
            delivery_id=f"simulation-20-point-delivery-{self._delivery_sequence}",
            track_ids=(self._selected_track_id,),
            destination=DeliveryDestination.OWN_MATERIAL,
            fully_entered=True,
        )
        mission_decision = self._mission.step(
            snapshot,
            transport=self._transport,
            safety=safety,
            delivery=evidence,
        )
        # The boundary event must reach the rule machine together with the
        # same-cycle snapshot; repeating the plain evaluation above plus this
        # delivery submission matches the historical contract and stays
        # idempotent because both calls share one timestamp.
        self._last_mission_decision = mission_decision
        self._cycle_mission_decision = mission_decision
        if mission_decision.terminal:
            return self._terminal(
                timestamp_ns,
                f"delivery_rejected:{mission_decision.reason}",
                mission_decision=mission_decision,
                world_snapshot=snapshot,
            )
        if self._selected_plan is not None:
            self._completed_target_points.append(self._selected_plan.target_field)
        self._retreat_base_distance_m = None
        self._disengage_confirm_count = 0
        self.state = Simulation20PointState.DISENGAGE_AND_RETREAT
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "delivery_evidence_accepted",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_disengage(
        self,
        timestamp_ns: int,
        snapshot: WorldSnapshot,
        cumulative_distance_m: float | None,
        safety: SafetySignals,
    ) -> SimulationDecision:
        if self._retreat_base_distance_m is None:
            self._retreat_base_distance_m = cumulative_distance_m
        if cumulative_distance_m is None or self._retreat_base_distance_m is None:
            return self._hold(timestamp_ns, "retreat_requires_encoder_distance")
        travelled = abs(cumulative_distance_m - self._retreat_base_distance_m)
        target = self._selected_target(snapshot)
        if travelled >= self.config.retreat_distance_m:
            if target is None or target.ground_point is None:
                return self._hold(timestamp_ns, "retreat_clear_evidence_missing")
            if target.ground_point.x < self.config.retreat_clear_distance_mm:
                self._disengage_confirm_count = 0
            else:
                self._disengage_confirm_count += 1
            if self._disengage_confirm_count >= self.config.disengage_confirm_frames:
                self._transport = TransportStatus()
                self.state = Simulation20PointState.UPDATE_PROGRESS
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "retreat_clear_transport",
                    world_snapshot=snapshot,
                )
        mission_decision = self._cycle_mission_decision
        assert mission_decision is not None
        return self._decision(
            timestamp_ns,
            -self.config.retreat_speed_m_s,
            0.0,
            "retreat_after_delivery",
            mission_decision=mission_decision,
            world_snapshot=snapshot,
        )

    def _step_update_progress(
        self,
        timestamp_ns: int,
        pose: FusedPoseEstimate | None,
        snapshot: WorldSnapshot,
    ) -> SimulationDecision:
        if self.valid_green_deliveries >= self.config.target_delivery_count:
            self.state = Simulation20PointState.FINISH_STOP
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "four_green_deliveries_finish_20_points",
                world_snapshot=snapshot,
            )
        self._current_breakup_attempts = 0
        self._selected_track_id = None
        self._selected_plan = None
        self._begin_reset_tracks(None)
        if pose is None or pose.pose is None:
            return self._hold(timestamp_ns, "next_scan_requires_field_pose")
        return self._decision(timestamp_ns, 0.0, 0.0, "scan_next_green_delivery")

    def _step_plan_rebreakup(
        self,
        timestamp_ns: int,
        snapshot: WorldSnapshot,
    ) -> SimulationDecision:
        if self._current_breakup_attempts >= self.config.max_breakup_attempts_per_delivery:
            return self._hold(timestamp_ns, "breakup_attempt_limit_per_delivery")
        if self._total_breakup_attempts >= self.config.max_breakup_attempts_total:
            return self._hold(timestamp_ns, "breakup_attempt_limit_total")
        if self._breakup_factory is None:
            return self._hold(timestamp_ns, "rebreakup_driver_unavailable")
        if not self._rebreakup_plan_is_safe(snapshot):
            return self._hold(timestamp_ns, "no_safe_rebreakup_plan")
        self._breakup = self._breakup_factory()
        self._rebreakup_mode = True
        self._current_breakup_attempts += 1
        self._total_breakup_attempts += 1
        self.state = Simulation20PointState.CENTER_REBREAKUP_CLUSTER
        return self._decision(timestamp_ns, 0.0, 0.0, "begin_controlled_rebreakup")

    def _rebreakup_plan_is_safe(self, snapshot: WorldSnapshot) -> bool:
        points = [
            target
            for target in snapshot.targets
            if target.ground_point is not None and target.field_point is not None
        ]
        if len(points) < 2:
            return False
        contact_distance = (
            2.0 * self.config.target_half_extent_mm
            + self.config.min_green_clearance_mm
        )
        return any(
            _distance(first.field_point, second.field_point) <= contact_distance
            for index, first in enumerate(points)
            for second in points[index + 1 :]
        )

    def _candidate_plans(
        self,
        snapshot: WorldSnapshot,
        pose: FieldPose2D,
    ) -> tuple[GreenTransportPlan, ...]:
        plans = [
            plan
            for target in snapshot.targets
            if target.target_class is TargetClass.GREEN_SUPPLY
            for plan in (self._find_plan(snapshot, pose, target.track_id),)
            if plan is not None
        ]
        plans.sort(
            key=lambda item: (
                _distance(pose.position, item.prepush_field),
                -item.clearance_mm,
                item.track_id,
            )
        )
        return tuple(plans)

    def _find_plan(
        self,
        snapshot: WorldSnapshot,
        pose: FieldPose2D,
        track_id: int | None,
    ) -> GreenTransportPlan | None:
        if track_id is None:
            return None
        target = snapshot.target(track_id)
        destination = self._destination(snapshot)
        if target is None or destination is None:
            return None
        if not self._candidate_target_is_safe(snapshot, target):
            return None
        assert target.field_point is not None
        push_dx = destination.x - target.field_point.x
        push_dy = destination.y - target.field_point.y
        norm = math.hypot(push_dx, push_dy)
        if norm <= 1e-6:
            return None
        push_dx /= norm
        push_dy /= norm
        prepush = FieldPoint(
            target.field_point.x - push_dx * self.config.prepush_offset_mm,
            target.field_point.y - push_dy * self.config.prepush_offset_mm,
        )
        if not self._segment_safe(
            snapshot,
            pose.position,
            prepush,
            ignored_track_id=target.track_id,
        ):
            return None
        if not self._segment_safe(
            snapshot,
            target.field_point,
            destination,
            ignored_track_id=target.track_id,
        ):
            return None
        clearance = self._green_clearance(snapshot, target)
        if clearance < self.config.min_green_clearance_mm:
            return None
        return GreenTransportPlan(
            track_id=target.track_id,
            target_field=target.field_point,
            destination_field=destination,
            push_direction_x=push_dx,
            push_direction_y=push_dy,
            prepush_field=prepush,
            clearance_mm=clearance,
        )

    def _candidate_target_is_safe(
        self,
        snapshot: WorldSnapshot,
        target: WorldTarget,
    ) -> bool:
        if target.target_class is not TargetClass.GREEN_SUPPLY:
            return False
        if target.hazard_state is not HazardState.CLEAR:
            return False
        if not target.ever_confirmed or target.track_status is not TrackStatus.CONFIRMED:
            return False
        if target.field_point is None or target.ground_point is None:
            return False
        if any(
            _distance(target.field_point, completed)
            <= self.config.completed_target_exclusion_mm
            for completed in self._completed_target_points
        ):
            return False
        if snapshot.target_region_kinds(target.track_id) is None:
            return False
        if snapshot.target_region_kinds(target.track_id) & {
            RegionKind.OPPONENT_SAFE,
            RegionKind.OWN_MATERIAL,
        }:
            return False
        if not self._target_is_fresh(target, snapshot.timestamp_ns):
            return False
        return True

    def _green_clearance(self, snapshot: WorldSnapshot, target: WorldTarget) -> float:
        assert target.field_point is not None
        clearance = float("inf")
        for other in snapshot.targets:
            if other.track_id == target.track_id:
                continue
            if other.field_point is None:
                return -float("inf")
            clearance = min(
                clearance,
                _distance(target.field_point, other.field_point)
                - 2.0 * self.config.target_half_extent_mm
                - self.config.safety_margin_mm,
            )
        if math.isinf(clearance):
            return 1_000_000.0
        return clearance

    def _segment_safe(
        self,
        snapshot: WorldSnapshot,
        start: FieldPoint,
        end: FieldPoint,
        *,
        ignored_track_id: int | None,
    ) -> bool:
        length = _distance(start, end)
        sample_count = max(
            1,
            math.ceil(length / self.config.corridor_sample_step_mm),
        )
        field_regions = tuple(
            region for region in snapshot.regions if region.kind is RegionKind.FIELD
        )
        opponent_regions = tuple(
            region
            for region in snapshot.regions
            if region.kind is RegionKind.OPPONENT_SAFE
        )
        if not field_regions:
            return False
        obstacle_radius = (
            self.config.robot_footprint_radius_mm
            + self.config.target_half_extent_mm
            + self.config.safety_margin_mm
        )
        for index in range(sample_count + 1):
            ratio = index / sample_count
            point = FieldPoint(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
            )
            if not any(region.contains(point) for region in field_regions):
                return False
            if any(region.contains(point) for region in opponent_regions):
                return False
            for target in snapshot.targets:
                if target.track_id == ignored_track_id:
                    continue
                if target.field_point is None:
                    return False
                if target.hazard_state is not HazardState.CLEAR or (
                    target.target_class is TargetClass.UNKNOWN
                ):
                    radius = obstacle_radius + self.config.min_green_clearance_mm
                else:
                    radius = obstacle_radius
                if _distance(point, target.field_point) < radius:
                    return False
        return True

    def _destination(self, snapshot: WorldSnapshot) -> FieldPoint | None:
        regions = tuple(
            region for region in snapshot.regions if region.kind is RegionKind.OWN_MATERIAL
        )
        if not regions:
            return None
        return _polygon_centroid(regions[0])

    def _target_fully_entered(
        self,
        snapshot: WorldSnapshot,
        target: WorldTarget,
    ) -> bool:
        if target.field_point is None:
            return False
        inset = self.config.delivery_inset_mm + self.config.target_half_extent_mm
        return any(
            _inside_inset(region, target.field_point, inset)
            for region in snapshot.regions
            if region.kind is RegionKind.OWN_MATERIAL
        )

    def _target_is_fresh(self, target: WorldTarget, timestamp_ns: int) -> bool:
        return (
            timestamp_ns >= target.last_seen_timestamp_ns
            and (timestamp_ns - target.last_seen_timestamp_ns) / 1_000_000.0
            <= self.config.target_max_age_ms
        )

    def _selected_target(self, snapshot: WorldSnapshot) -> WorldTarget | None:
        if self._selected_track_id is None:
            return None
        return snapshot.target(self._selected_track_id)

    def _cancel_selected_green(self) -> None:
        """Drop a pre-contact green plan so the next frame can re-evaluate candidates."""

        self._selected_track_id = None
        self._selected_plan = None
        self.state = Simulation20PointState.EVALUATE_EASY_GREEN

    def _pose_usable(self, estimate: FusedPoseEstimate | None) -> bool:
        if estimate is None or estimate.pose is None:
            return False
        if FusionQuality.STALE in estimate.quality:
            return False
        if estimate.position_uncertainty_mm is None or estimate.heading_uncertainty_rad is None:
            return False
        return (
            estimate.position_uncertainty_mm <= self.config.max_position_uncertainty_mm
            and estimate.heading_uncertainty_rad <= self.config.max_heading_uncertainty_rad
        )

    def _contact_conflict(
        self,
        snapshot: WorldSnapshot,
        selected: WorldTarget,
    ) -> bool:
        if selected.ground_point is None:
            return True
        for other in snapshot.targets:
            if other.track_id == selected.track_id or other.ground_point is None:
                continue
            if (
                0.0 <= other.ground_point.x <= self.config.contact_corridor_length_mm
                and abs(other.ground_point.y - selected.ground_point.y)
                <= 2.0 * self.config.target_half_extent_mm
                + self.config.engage_lateral_tolerance_mm
            ):
                return True
        return False

    def _geometric_contact_is_valid(self, target: WorldTarget) -> bool:
        if target.ground_point is None:
            return False
        return (
            target.track_status is TrackStatus.CONFIRMED
            and target.target_class is TargetClass.GREEN_SUPPLY
            and target.hazard_state is HazardState.CLEAR
            and 0.0 < target.ground_point.x <= self.config.engage_distance_mm
            and abs(target.ground_point.y) <= self.config.engage_lateral_tolerance_mm
        )

    def _evaluate_mission(
        self,
        snapshot: WorldSnapshot,
        safety: SafetySignals,
    ) -> MissionDecision:
        """Advance the rule state machine exactly once per control cycle.

        Terminal outcomes are enforced by ``step()`` for every state. A
        non-terminal ``SAFETY_HOLD``/``AVOIDING`` activity only produces a
        zero-speed hold, never a latched state, when it originates from a
        regular behaviour state; fixed encoder-driven breakup/reset/update
        actions keep running so that brief vision gaps do not interrupt them,
        while rule-level terminations still stop motion.
        """

        decision = self._mission.step(
            snapshot,
            transport=self._transport,
            safety=safety,
        )
        self._last_mission_decision = decision
        self._cycle_mission_decision = decision
        if decision.terminal:
            return decision
        return decision

    def _vision_independent_action_active(
        self,
        cumulative_distance_m: float | None,
    ) -> bool:
        """固定动作未完成时允许短时缺少视觉/定位快照。

        解团的固定动作由各自的编码器/定时阶段和超时约束。交付后退离只在最低
        编码器距离尚未走完时属于固定动作；达到距离后必须恢复新鲜视觉与定位，
        再确认目标已经脱离，不能靠旧目标位置完成交付收尾。
        """

        if self.state in _FIXED_BREAKUP_STATES:
            return True
        if self.state is not Simulation20PointState.DISENGAGE_AND_RETREAT:
            return False
        if cumulative_distance_m is None or self._retreat_base_distance_m is None:
            return True
        return (
            abs(cumulative_distance_m - self._retreat_base_distance_m)
            < self.config.retreat_distance_m
        )

    @staticmethod
    def _direct_safety_reason(safety: SafetySignals) -> str | None:
        if safety.external_stop_requested:
            return "external_stop"
        if safety.safety_accident:
            return "safety_accident"
        if safety.lost_control:
            return "lost_control"
        if safety.human_touched_after_start:
            return "human_touched_after_start"
        if safety.target_carried_on_robot:
            return "illegal_carry"
        if safety.active_attack:
            return "active_attack"
        return None

    def _validate_timestamp(self, timestamp_ns: int) -> None:
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                f"timestamp_ns moved backwards from {self._last_timestamp_ns} "
                f"to {timestamp_ns}."
            )
        self._last_timestamp_ns = timestamp_ns

    def _decision(
        self,
        timestamp_ns: int,
        linear: float,
        angular: float,
        reason: str,
        *,
        posture: GripperPosture = GripperPosture.CLOSED,
        mission_decision: MissionDecision | None = None,
        world_snapshot: WorldSnapshot | None = None,
    ) -> SimulationDecision:
        if linear != 0.0 or angular != 0.0:
            self._last_motion_timestamp_ns = timestamp_ns
        return SimulationDecision(
            timestamp_ns=timestamp_ns,
            state=self.state,
            linear_velocity_m_s=float(linear),
            angular_velocity_rad_s=float(angular),
            gripper_posture=posture,
            reason=reason,
            selected_track_id=self._selected_track_id,
            valid_green_deliveries=self.valid_green_deliveries,
            score_points=self.score_points,
            mission_decision=mission_decision,
            world_snapshot=world_snapshot,
        )

    def _hold(self, timestamp_ns: int, reason: str) -> SimulationDecision:
        """保持当前流程状态并输出零速；证据恢复后下一周期自动继续。"""

        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            reason,
            mission_decision=self._cycle_mission_decision,
            world_snapshot=self._world_snapshot,
        )

    def _terminal(
        self,
        timestamp_ns: int,
        reason: str,
        *,
        mission_decision: MissionDecision | None = None,
        world_snapshot: WorldSnapshot | None = None,
    ) -> SimulationDecision:
        self.state = Simulation20PointState.TERMINAL_STOP
        self._transport = TransportStatus()
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            reason,
            mission_decision=mission_decision,
            world_snapshot=world_snapshot or self._world_snapshot,
        )


def _run_hardware(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    jpeg_quality: int = 80,
    observer_image_interval_s: float = 1.0,
    log_dir: Path | None = None,
) -> None:
    """装配真实旁路；不在模块导入阶段访问相机、Hailo 或串口。"""

    # 传入 --log-dir 时把之后所有终端输出（含配置错误和旁路 worker 打印）
    # tee 到按时间命名的日志文件；正常退出时在 finally 中恢复标准流。
    log_stream, original_stdout, original_stderr = _begin_time_named_log(log_dir)

    from rescue_vision.app.cluster_breakup import (
        CameraPerceptionPump,
        EncoderTravelTracker,
        OdometryFusionPump,
        RemotePerceptionTransport,
        _submit_breakup_odometry,
    )
    from rescue_vision.app.manual_capture import build_camera_pipeline
    from rescue_vision.communication import (
        RemoteAccessMode,
        RemoteRole,
        TeamColor as RemoteTeamColor,
    )
    from rescue_vision.motion import (
        CarCommandReply,
        CarSystemStatus,
        CommandResult,
        MotionController,
        MotionSynchronizationError,
        OdometryImu,
    )
    from rescue_vision.perception import PerceptionFrameRenderer

    config = load_runtime_config(config_path)
    if not config.simulation_20_point.enabled:
        raise RuntimeError("simulation_20_point.enabled must be true.")
    if not supervised_stop_ready:
        raise RuntimeError(
            "A physical emergency stop and continuous supervision are required "
            "until the STM32 watchdog has been verified."
        )
    if config.remote.enabled and (
        config.remote.role is not RemoteRole.SERVER
        or config.remote.access_mode is not RemoteAccessMode.OBSERVE_ONLY
    ):
        raise RuntimeError(
            "20-point simulation remote access must be server/observe_only."
        )
    channel = config.uart.build_channel()
    assert channel is not None
    controller = config.motion.build_controller(channel)
    assert controller is not None
    gripper = config.motion.gripper.build_calibration()
    if gripper is None:
        raise RuntimeError("20-point simulation requires gripper calibration.")
    if gripper.transport_angles_deg is None:
        raise RuntimeError(
            "20-point simulation requires transport gripper angles."
        )

    class _MotionChannelContext:
        """Ensure the soft brake is sent before UART shutdown on every exit."""

        def __enter__(self):
            channel.start()
            return channel

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                controller.soft_brake()
            finally:
                channel.stop()
            return False

    odometry_calibration = config.motion.odometry.build_calibration()
    if odometry_calibration is None:
        raise RuntimeError("20-point simulation requires odometry calibration.")
    encoder_tracker = EncoderTravelTracker(
        odometry_calibration,
        max_wheel_velocity_m_s=config.motion.max_wheel_velocity_m_s,
        max_consecutive_overrun_samples=(
            config.motion.odometry.max_consecutive_overrun_samples
        ),
    )
    fusion = config.build_odometry_imu_fusion()
    if fusion is None:
        raise RuntimeError("20-point simulation requires odometry/IMU fusion.")
    fusion_pump = OdometryFusionPump(fusion)
    pipeline = build_camera_pipeline(config)
    visual_localization = config.build_visual_localization_pipeline(
        ground_projector=pipeline.ground_projector,
        fusion=fusion,
    )
    renderer = PerceptionFrameRenderer(
        lambda: config.build_target_pose_detector(
            ground_projector=pipeline.ground_projector
        )
    )
    camera_pump = CameraPerceptionPump(pipeline.source, pipeline.prepare, renderer)
    remote_transport = None
    if config.remote.enabled:
        server = config.remote.build_server()
        assert server is not None
        remote_transport = RemotePerceptionTransport(
            server,
            pipeline,
            config,
            jpeg_quality=jpeg_quality,
            min_publish_interval_s=observer_image_interval_s,
            map_team_color=RemoteTeamColor(config.world.team_color.value),
        )
    sequence = Simulation20PointSequence.from_app_config(config)
    latest_status: CarSystemStatus | None = None
    latest_snapshot: PerceptionSnapshot | None = None
    stop_requested = False
    camera_started = False
    camera_start_thread: Thread | None = None
    fusion_started = False
    active_motion_since_ns: int | None = None
    last_reported_state: Simulation20PointState | None = None
    next_progress_ns = 0
    next_resynchronization_attempt_ns = 0
    remote_restart_thread: Thread | None = None
    next_remote_restart_attempt_ns = 0
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    def consume(message: object) -> None:
        nonlocal latest_status
        if isinstance(message, OdometryImu):
            _submit_breakup_odometry(encoder_tracker, fusion_pump, message)
        elif isinstance(message, CarSystemStatus):
            latest_status = message

    def service_uart_during_camera_startup() -> None:
        """在相机/Hailo 预热期间继续排空车端遥测。"""

        controller.update(now_ns=time.monotonic_ns())
        for message in controller.drain_messages():
            consume(message)
        if latest_status is not None and latest_status.emergency_stop_latched:
            raise RuntimeError(
                "STM32 emergency stop is latched during camera startup."
            )

    signal.signal(signal.SIGTERM, request_stop)
    try:
        fusion_pump.start()
        fusion_started = True
        if remote_transport is not None:
            remote_transport.start()
        with _MotionChannelContext():
            # Hailo/相机预热与运动通道同步并行；等待预热期间主线程仍排空
            # UART，避免 100 Hz 遥测在启动门禁处挤满接收队列。
            camera_start_thread = camera_pump.start_in_background()
            # A transient UART blip must not abort startup: retry the initial
            # SOFT_BRAKE synchronization within the bounded preflight window,
            # aborting immediately on a latched emergency stop.
            sync_deadline_ns = time.monotonic_ns() + _PREFLIGHT_RETRY_WINDOW_NS
            while True:
                try:
                    controller.synchronize(on_message=consume)
                    break
                except MotionSynchronizationError:
                    if (
                        latest_status is not None
                        and latest_status.emergency_stop_latched
                    ):
                        raise RuntimeError(
                            "STM32 emergency stop is latched during camera "
                            "startup."
                        )
                    if time.monotonic_ns() >= sync_deadline_ns:
                        raise
                    time.sleep(0.02)
            camera_pump.wait_until_started(
                camera_start_thread,
                on_wait=service_uart_during_camera_startup,
            )
            camera_started = True
            controller.query_state()
            fusion_pump.wait_until_ready(on_wait=service_uart_during_camera_startup)
            # The first completed perception snapshot is a start gate; the
            # controller remains synchronized and stopped while waiting for it.
            while latest_snapshot is None and not stop_requested:
                controller.update(now_ns=time.monotonic_ns())
                for message in controller.drain_messages():
                    consume(message)
                camera_pump.check_health()
                latest_snapshot = renderer.latest_snapshot()
                if (
                    latest_snapshot is not None
                    and latest_snapshot.field_features is not None
                    and visual_localization is not None
                ):
                    visual_localization.submit(latest_snapshot.field_features)
                time.sleep(0.005)
            if latest_snapshot is None:
                raise RuntimeError("No fresh perception snapshot before start.")

            def preflight_checks() -> SimulationPreflight:
                return SimulationPreflight(
                    telemetry_fresh=encoder_tracker.distance_m is not None,
                    watchdog_armed=bool(
                        latest_status is not None and latest_status.watchdog_armed
                    ),
                    emergency_stop_clear=not bool(
                        latest_status is not None
                        and latest_status.emergency_stop_latched
                    ),
                    zero_speed_command_accepted=controller.motion_synchronized,
                    camera_observation_fresh=True,
                    observe_only_remote=not config.remote.enabled
                    or config.remote.access_mode is RemoteAccessMode.OBSERVE_ONLY,
                )

            # Transient startup evidence (telemetry, watchdog report, zero
            # speed synchronization) is retried within a bounded window while
            # UART keeps draining; a latched emergency stop or a non
            # observe_only remote gate is a safety/rule failure and is never
            # retried.
            checks = preflight_checks()
            preflight_deadline_ns = (
                time.monotonic_ns() + _PREFLIGHT_RETRY_WINDOW_NS
            )
            while (
                not checks.ready
                and time.monotonic_ns() < preflight_deadline_ns
                and checks.emergency_stop_clear
                and checks.observe_only_remote
            ):
                controller.update(now_ns=time.monotonic_ns())
                for message in controller.drain_messages():
                    consume(message)
                if controller.needs_synchronization:
                    try:
                        controller.synchronize(on_message=consume)
                    except MotionSynchronizationError:
                        # Bounded by the retry window above; SOFT_BRAKE has
                        # already zeroed the controller state.
                        pass
                time.sleep(0.005)
                checks = preflight_checks()
            preflight_decision = sequence.preflight(time.monotonic_ns(), checks)
            if preflight_decision.state is Simulation20PointState.TERMINAL_STOP:
                raise RuntimeError(preflight_decision.reason)
            sequence.start(time.monotonic_ns())
            last_posture: GripperPosture | None = None
            while not stop_requested:
                loop_start_ns = time.monotonic_ns()
                if (
                    controller.needs_synchronization
                    and loop_start_ns >= next_resynchronization_attempt_ns
                ):
                    try:
                        controller.synchronize(on_message=consume)
                    except MotionSynchronizationError:
                        # SOFT_BRAKE has already zeroed the controller state.
                        # Keep the process alive and retry after a bounded
                        # interval instead of converting a transient missing
                        # reply into a permanent application stop.
                        next_resynchronization_attempt_ns = (
                            time.monotonic_ns() + 500_000_000
                        )
                    else:
                        next_resynchronization_attempt_ns = 0
                now_ns = time.monotonic_ns()
                branch_error: str | None = None
                controller.update(now_ns=now_ns)
                try:
                    for message in controller.drain_messages():
                        if (
                            isinstance(message, CarCommandReply)
                            and message.result is not CommandResult.ACCEPTED
                        ):
                            # The controller already handled the rejection
                            # internally (old sequence or other -> request
                            # synchronization, emergency stop -> latch). A
                            # single rejected reply must not end the round;
                            # the bounded resynchronization retry recovers.
                            print(
                                "STM32 rejected command: "
                                f"{message.command_type.name.lower()}="
                                f"{message.result.name.lower()}",
                                flush=True,
                            )
                            continue
                        try:
                            consume(message)
                        except Exception as exc:
                            if branch_error is None:
                                branch_error = f"odometry_submit:{exc}"
                except MotionSynchronizationError as exc:
                    # A rejected SOFT_BRAKE reply has already zeroed the
                    # controller state; retry synchronization after the
                    # bounded backoff instead of exiting the process.
                    print(f"motion_synchronization_retry={exc}", flush=True)
                    next_resynchronization_attempt_ns = (
                        time.monotonic_ns() + 500_000_000
                    )
                try:
                    camera_pump.check_health()
                except Exception as exc:
                    if branch_error is None:
                        branch_error = f"camera_pump:{exc}"
                try:
                    fresh_snapshot = renderer.latest_snapshot()
                except Exception as exc:
                    if branch_error is None:
                        branch_error = f"perception_renderer:{exc}"
                    fresh_snapshot = None
                latest_snapshot = fresh_snapshot or latest_snapshot
                if (
                    fresh_snapshot is not None
                    and fresh_snapshot.field_features is not None
                    and visual_localization is not None
                ):
                    visual_localization.submit(fresh_snapshot.field_features)
                try:
                    pose = fusion_pump.latest_estimate(now_ns)
                except Exception as exc:
                    if branch_error is None:
                        branch_error = f"fusion_pump:{exc}"
                    pose = None
                decision = sequence.step(
                    now_ns,
                    perception=latest_snapshot,
                    pose=pose,
                    cumulative_distance_m=encoder_tracker.distance_m,
                    health=SimulationHealth(
                        control_ready=(
                            latest_status is not None
                            and latest_status.protocol_ready
                            and controller.motion_synchronized
                            and not controller.link_degraded
                            and not controller.emergency_stop_latched
                        ),
                        command_accepted=True,
                        watchdog_armed=latest_status is not None
                        and latest_status.watchdog_armed,
                        emergency_stop_clear=not (
                            latest_status is not None
                            and latest_status.emergency_stop_latched
                        ),
                        camera_fresh=latest_snapshot is not None
                        and (
                            now_ns - latest_snapshot.capture_timestamp_ns
                        )
                        <= round(config.processing.max_observation_age_ms * 1_000_000),
                        localization_fresh=pose is not None
                        and pose.pose is not None,
                        observe_only_remote=not config.remote.enabled
                        or config.remote.access_mode is RemoteAccessMode.OBSERVE_ONLY,
                        branch_error=branch_error,
                    ),
                )
                if remote_transport is not None:
                    # Observation is optional and must not stop motion: a
                    # broken observer connection only logs and schedules a
                    # bounded background restart so a new observer can
                    # connect again.
                    try:
                        _publish_remote_simulation_state(
                            remote_transport,
                            pose=pose,
                            timestamp_ns=now_ns,
                            rendered=renderer.latest(),
                        )
                    except Exception as exc:
                        print(f"remote_observation_error={exc}", flush=True)
                        if (
                            remote_restart_thread is None
                            and now_ns >= next_remote_restart_attempt_ns
                        ):
                            next_remote_restart_attempt_ns = (
                                now_ns + _REMOTE_RESTART_BACKOFF_NS
                            )

                            def restart_remote_observation() -> None:
                                try:
                                    remote_transport.stop()
                                    remote_transport.start()
                                except Exception as restart_exc:
                                    print(
                                        "remote_observation_restart_failed="
                                        f"{restart_exc}",
                                        flush=True,
                                    )

                            remote_restart_thread = Thread(
                                target=restart_remote_observation,
                                name="rescue-remote-observation-restart",
                                daemon=True,
                            )
                            remote_restart_thread.start()
                    if (
                        remote_restart_thread is not None
                        and not remote_restart_thread.is_alive()
                    ):
                        remote_restart_thread = None
                if decision.gripper_posture is not last_posture:
                    if decision.gripper_posture is GripperPosture.OPEN:
                        angles = (
                            gripper.open_left_angle_deg,
                            gripper.open_right_angle_deg,
                        )
                    elif decision.gripper_posture is GripperPosture.TRANSPORT:
                        transport_angles = gripper.transport_angles_deg
                        if transport_angles is None:
                            raise RuntimeError(
                                "Transport gripper posture is not configured."
                            )
                        angles = transport_angles
                    else:
                        angles = (
                            gripper.closed_left_angle_deg,
                            gripper.closed_right_angle_deg,
                        )
                    controller.set_gripper_angles(*angles)
                    last_posture = decision.gripper_posture
                motion_requested = (
                    decision.linear_velocity_m_s != 0.0
                    or decision.angular_velocity_rad_s != 0.0
                )
                if motion_requested:
                    if active_motion_since_ns is None:
                        active_motion_since_ns = now_ns
                else:
                    active_motion_since_ns = None
                motor_blocked_reason: str | None = None
                if (
                    active_motion_since_ns is not None
                    and now_ns - active_motion_since_ns >= 750_000_000
                    and latest_status is not None
                    and now_ns - latest_status.received_timestamp_ns
                    <= 500_000_000
                    and not latest_status.motor_output_enabled
                ):
                    # Keep the command stream alive at zero speed and wait for
                    # the STM32 to re-enable motor output instead of ending
                    # the round on a single disabled state; the 15 s no-motion
                    # rule still bounds an unrecovered failure.
                    motor_blocked_reason = (
                        f"stop_reason={latest_status.stop_reason.name.lower()} "
                        f"watchdog_armed={latest_status.watchdog_armed} "
                        "last_motion_command_age_ms="
                        f"{latest_status.last_motion_command_age_ms}"
                    )
                if motor_blocked_reason is None:
                    controller.drive_wheel_limited(
                        decision.linear_velocity_m_s,
                        decision.angular_velocity_rad_s,
                    )
                else:
                    controller.drive_wheel_limited(0.0, 0.0)
                if decision.state is not last_reported_state or now_ns >= next_progress_ns:
                    status_text = "status=none"
                    if latest_status is not None:
                        status_text = (
                            "status=("
                            f"motor_output={latest_status.motor_output_enabled},"
                            f"watchdog={latest_status.watchdog_armed},"
                            f"stop_reason={latest_status.stop_reason.name.lower()},"
                            f"motion_age_ms={latest_status.last_motion_command_age_ms})"
                        )
                    localization_quality = (
                        ""
                        if pose is None
                        else ",".join(sorted(item.value for item in pose.quality))
                    )
                    position_sigma_text = (
                        "none"
                        if pose is None or pose.position_uncertainty_mm is None
                        else f"{pose.position_uncertainty_mm:.1f}"
                    )
                    heading_sigma_text = (
                        "none"
                        if pose is None or pose.heading_uncertainty_rad is None
                        else f"{pose.heading_uncertainty_rad:.3f}"
                    )
                    localization_text = (
                        "localization=("
                        f"pose={pose is not None and pose.pose is not None},"
                        f"quality={localization_quality or 'unknown'},"
                        f"position_sigma_mm={position_sigma_text},"
                        f"heading_sigma_rad={heading_sigma_text},"
                        "estimate_timestamp_ns="
                        f"{pose.estimate_timestamp_ns if pose is not None else 'none'},"
                        "continuity_loss_reason="
                        f"{fusion_pump.continuity_loss_reason or 'none'})"
                    )
                    perception_text = "perception=none"
                    if latest_snapshot is not None:
                        perception_text = (
                            "perception=("
                            "age_ms="
                            f"{(now_ns - latest_snapshot.capture_timestamp_ns) / 1_000_000.0:.1f},"
                            "stale_dropped="
                            f"{latest_snapshot.dropped_stale_age_ms is not None},"
                            f"frame_sequence={latest_snapshot.frame_sequence})"
                        )
                    print(
                        f"state={decision.state.value} reason={decision.reason} "
                        f"deliveries={decision.valid_green_deliveries} "
                        f"score={decision.score_points} "
                        f"{encoder_tracker.diagnostic()} "
                        f"target_wheel_m_s={controller.target_wheel_speeds_m_s} "
                        f"commanded_wheel_m_s={controller.commanded_wheel_speeds_m_s} "
                        f"motor_blocked={motor_blocked_reason or 'none'} "
                        f"{perception_text} "
                        f"{status_text} {localization_text}",
                        flush=True,
                    )
                    last_reported_state = decision.state
                    next_progress_ns = now_ns + 1_000_000_000
                if decision.state in {
                    Simulation20PointState.TERMINAL_STOP,
                    Simulation20PointState.FINISH_STOP,
                }:
                    break
                time.sleep(0.005)
            controller.soft_brake()
    finally:
        try:
            if camera_start_thread is not None or camera_started:
                camera_pump.stop()
        finally:
            try:
                controller.soft_brake()
            except Exception:
                pass
            try:
                if remote_restart_thread is not None and (
                    remote_restart_thread.is_alive()
                ):
                    # Wait for a pending observe-only restart so transport
                    # shutdown does not race its stop()/start() pair.
                    remote_restart_thread.join(timeout=2.0)
                if remote_transport is not None:
                    remote_transport.stop()
            finally:
                try:
                    if fusion_started:
                        fusion_pump.stop()
                finally:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                    _end_time_named_log(
                        log_stream,
                        original_stdout,
                        original_stderr,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the restricted four-green 20-point simulation flow."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm a physical emergency stop and continuous supervision.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--observer-image-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for a time-named log: stdout/stderr are also "
            "written to <dir>/<YYYYmmdd_HHMM>.log (append on same-minute "
            "restarts), e.g. --log-dir logs."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if not math.isfinite(args.observer_image_interval_seconds) or (
        args.observer_image_interval_seconds <= 0.0
    ):
        parser.error("--observer-image-interval-seconds must be positive")
    _run_hardware(
        args.config,
        supervised_stop_ready=args.supervised_physical_stop_ready,
        jpeg_quality=args.jpeg_quality,
        observer_image_interval_s=args.observer_image_interval_seconds,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
