"""实验性门前清障：场地触发几何、暂存任务和可中断动作编排。

不创建硬件。MatchSequence 提供采集位姿、真实运动反馈和现有动作控制器；
原载荷暂存和清障物释放均不生成交付事件，完成后直接交还普通搜索。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

from rescue_vision.config.gate_clearance import GateClearanceConfig
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import normalize_angle
from rescue_vision.perception import TargetClass
from rescue_vision.world.static_map import PhysicalRegionKind, StaticFieldMap, TeamColor

if TYPE_CHECKING:
    from rescue_vision.app.match import MatchDecision, MatchSequence


@dataclass(frozen=True, slots=True)
class GateObject:
    target_class: TargetClass
    point: FieldPoint
    radius_mm: float


@dataclass(frozen=True, slots=True)
class ClearanceAction:
    name: str
    kind: str  # point / heading / straight / reverse / gripper
    opened: bool
    point: FieldPoint | None = None
    value: float = 0.0
    target_heading_rad: float | None = None


@dataclass(slots=True)
class GateClearanceSession:
    actions: tuple[ClearanceAction, ...]
    trigger_capture_ns: int
    attempt_started_ns: int
    index: int = 0
    action_started_ns: int | None = None
    gripper_started_ns: int | None = None
    initial_stop_confirmed: bool = False
    released: bool = False
    sweep_realign_pending: bool = False
    sweep_realign_count: int = 0
    failure: str | None = None

    @property
    def action(self) -> ClearanceAction:
        return self.actions[self.index]


def gate_obstructions(objects: tuple[GateObject, ...], static_map: StaticFieldMap,
                      team: TeamColor,
                      config: GateClearanceConfig) -> tuple[GateObject, ...]:
    """按己方两个分区的真实前边界检查 K0；不使用画面颜色猜区域。"""
    material = PhysicalRegionKind.BLUE_MATERIAL if team is TeamColor.BLUE else PhysicalRegionKind.RED_MATERIAL
    injured = PhysicalRegionKind.BLUE_INJURED if team is TeamColor.BLUE else PhysicalRegionKind.RED_INJURED
    sign = -1 if team is TeamColor.BLUE else 1
    result = []
    for obj in objects:
        for region in static_map.regions:
            if region.kind not in (material, injured):
                continue
            xs = [p.x for p in region.polygon_field]
            x_min = min(xs) + config.lateral_inset_mm
            x_max = max(xs) - config.lateral_inset_mm
            if x_min >= x_max:
                raise ValueError(
                    "gate_clearance.lateral_inset_mm removes the complete gate width: "
                    f"region={region.region_id!r}, bounds=({min(xs)!r},{max(xs)!r}), "
                    f"inset={config.lateral_inset_mm!r}."
                )
            edge = (
                min(sign * p.y for p in region.polygon_field)
                - config.front_edge_inset_mm
            )
            depth = edge - sign * obj.point.y
            if not (
                x_min <= obj.point.x <= x_max
                and 0 < depth <= config.front_depth_mm
            ):
                continue
            wrong = (obj.target_class is TargetClass.BLUE_DANGER
                     or region.kind is material and obj.target_class is TargetClass.ORANGE_INJURED
                     or region.kind is injured and obj.target_class in (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE))
            if wrong:
                result.append(obj)
                break
    return tuple(result)


def make_clearance_session(config: GateClearanceConfig, position: FieldPoint,
                           team: TeamColor, capture_ns: int, *,
                           attempt_started_ns: int | None = None) -> GateClearanceSession:
    """S 点是横扫轴心；清障物在中场原点 200 mm 半径内释放后直接搜索。"""
    sign = -1 if team is TeamColor.BLUE else 1
    left = FieldPoint(-config.side_x_mm, sign * config.sweep_y_mm)
    right = FieldPoint(config.side_x_mm, sign * config.sweep_y_mm)
    sweep_start, end = sorted((left, right), key=lambda p: math.hypot(p.x-position.x, p.y-position.y))
    # 暂存轴心向外让出退出距离；后退后准确落在 S1/S2 开始完整横扫。
    direction = -1 if sweep_start.x < 0 else 1
    start = FieldPoint(sweep_start.x + direction*config.release_reverse_m*1000, sweep_start.y)
    outward = math.pi if start.x < 0 else 0.0
    sweep_heading = 0.0 if start.x < 0 else math.pi
    center_heading = math.atan2(-end.y, -end.x)
    center_start_radius_mm = math.hypot(end.x, end.y)
    # 直行动作使用 5 mm 完成容差；目标再向原点内收 5 mm，保证动作完成时
    # 机器人轴心已经进入配置的 200 mm 最大半径，而不是停在其外侧。
    center_motion_tolerance_m = 0.005
    center_target_radius_mm = max(
        0.0,
        config.center_stop_radius_mm - center_motion_tolerance_m * 1000.0,
    )
    center_distance_m = (
        center_start_radius_mm - center_target_radius_mm
    ) / 1000.0
    center_stop = FieldPoint(
        end.x * center_target_radius_mm / center_start_radius_mm,
        end.y * center_target_radius_mm / center_start_radius_mm,
    )
    stash_heading = math.atan2(start.y - position.y, start.x - position.x)
    stash_distance_m = math.hypot(start.x - position.x, start.y - position.y) / 1000.0
    actions = (
        ClearanceAction("align_stash_heading", "heading", False, value=stash_heading),
        ClearanceAction(
            "to_stash_s",
            "straight",
            False,
            point=start,
            value=stash_distance_m,
            target_heading_rad=stash_heading,
        ),
        ClearanceAction("face_outward", "heading", False, value=outward),
        ClearanceAction("stash_open", "gripper", True),
        ClearanceAction("stash_reverse_120mm", "reverse", True, value=-config.release_reverse_m),
        # 暂存退出时仍保持张爪；必须完成一次独立合爪和机械等待后才允许横扫。
        ClearanceAction("sweep_close", "gripper", False),
        ClearanceAction("align_sweep_heading", "heading", False, value=sweep_heading),
        # 左中、中心、右中三点共线。先完成上一步绝对航向对正，再锁定该航向直行；
        # end 只保留给扫掠风险几何，不能交给点跟随器重新选择切入角。
        ClearanceAction(
            "sweep_via_midpoint",
            "straight",
            False,
            point=end,
            value=2.0 * config.side_x_mm / 1000.0,
            target_heading_rad=sweep_heading,
        ),
        ClearanceAction("align_field_center", "heading", False, value=center_heading),
        ClearanceAction(
            "center_forward_200mm_radius",
            "straight",
            False,
            point=center_stop,
            value=center_distance_m,
            target_heading_rad=center_heading,
        ),
        ClearanceAction("center_open", "gripper", True),
        ClearanceAction(
            "center_reverse_120mm",
            "reverse",
            True,
            value=-config.release_reverse_m,
            target_heading_rad=center_heading,
        ),
    )
    return GateClearanceSession(
        actions,
        capture_ns,
        capture_ns if attempt_started_ns is None else attempt_started_ns,
    )


def _objects(sequence: MatchSequence, now_ns: int) -> tuple[GateObject, ...]:
    from rescue_vision.app.breakup_planner import physical_radii
    snapshot = sequence._latest_perception
    geometry = sequence._breakup_target_geometry
    if snapshot is None or geometry is None or not sequence._fresh_perception(snapshot, now_ns):
        return ()
    result = []
    for target in sequence._tracker.tracks:
        if (target.frame_sequence != snapshot.frame_sequence
                or target.last_seen_timestamp_ns != snapshot.capture_timestamp_ns):
            continue
        point = sequence._field_point_for_target(target)
        if point is not None:
            radius = physical_radii(geometry.geometry_for(target.target_class))[1]
            result.append(GateObject(target.target_class, point, radius))
    return tuple(result)


def maybe_begin_clearance(sequence: MatchSequence, now_ns: int) -> MatchDecision | None:
    from rescue_vision.app.match import MatchState
    cfg = sequence.config.gate_clearance
    if (not cfg.enabled or sequence._gate_clearance_attempted
            or not sequence._transport_target_classes or sequence._near_field_pickup is None
            or sequence._breakup_static_map is None):
        return None
    # 只在 D1 两帧视觉校准成功后、D2 到达前检查。D1 前的远场投影误差
    # 不能再截获运输，D2 末段也不能把已经投放的物体当作本趟载荷搬走。
    calibrated_route = (
        sequence._safe_zone_calibration_pose is not None
        and (
            sequence.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
            and sequence._safe_zone_phase == "align_d2_line"
            or sequence.state is MatchState.TRANSPORT_FORWARD
            and sequence._safe_zone_phase == "forward_d2_line"
        )
    )
    if not calibrated_route:
        return None
    position = sequence.estimated_field_position
    snapshot = sequence._latest_perception
    if position is None or snapshot is None:
        return None
    obstructions = gate_obstructions(
        _objects(sequence, now_ns),
        sequence._breakup_static_map,
        sequence._team_color,
        cfg,
    )
    if not obstructions:
        return None
    sequence._gate_clearance = make_clearance_session(cfg, position, sequence._team_color,
        snapshot.capture_timestamp_ns,
        attempt_started_ns=now_ns)
    sequence._gate_clearance_attempted = True
    sequence._finish_noncontact()
    sequence._action_settle_phase = None
    sequence._action_settle_until_ns = None
    sequence._grasp_task = None
    sequence._greedy_active = False
    sequence.state = MatchState.GATE_CLEARANCE
    evidence = ";".join(
        f"{item.target_class.value}@({item.point.x:.1f},{item.point.y:.1f})"
        for item in obstructions
    )
    return sequence._decision(
        now_ns,
        0.0,
        0.0,
        "gate_clearance_triggered:"
        f"capture_ns={snapshot.capture_timestamp_ns},objects={evidence},"
        f"front_edge_inset_mm={cfg.front_edge_inset_mm:.1f},"
        f"front_depth_mm={cfg.front_depth_mm:.1f},"
        f"lateral_inset_mm={cfg.lateral_inset_mm:.1f}",
        soft_brake=True,
    )


def _action_diagnostic(sequence: MatchSequence, action: ClearanceAction) -> str:
    position = sequence.estimated_field_position
    current = "none" if position is None else f"({position.x:.1f},{position.y:.1f})"
    target = "none" if action.point is None else f"({action.point.x:.1f},{action.point.y:.1f})"
    target_heading_rad = action.value if action.kind == "heading" else action.target_heading_rad
    heading = (
        "none" if target_heading_rad is None
        else f"{math.degrees(target_heading_rad):.1f}deg"
    )
    return f"field_position={current},target_field={target},target_heading={heading}"


def _advance(sequence: MatchSequence, now_ns: int) -> None:
    session = sequence._gate_clearance
    assert session is not None
    if session.action.name == "stash_open":
        session.released = True
        sequence._transport_target_classes = ()
        sequence._cargo_capture_floor_ns = None
        sequence._counted_pickup_session = None
    session.index += 1
    session.action_started_ns = None
    session.gripper_started_ns = None
    sequence._finish_noncontact()


def _action_index(session: GateClearanceSession, name: str) -> int:
    return next(index for index, action in enumerate(session.actions) if action.name == name)


def _fail(sequence: MatchSequence, now_ns: int, reason: str) -> MatchDecision:
    """暂存前失败继续投放；暂存后失败放弃本次载荷并回到普通搜索。"""
    from rescue_vision.app.cluster_breakup import GripperPosture
    from rescue_vision.app.match import MatchState
    session = sequence._gate_clearance
    assert session is not None
    sequence._finish_noncontact()
    session.failure = reason
    if not session.released:
        sequence._gate_clearance = None
        return sequence._start_safe_zone_transport(now_ns, transport_opened=False,
            posture=GripperPosture.CLOSED, reason=f"gate_clearance_aborted_before_stash:{reason}")
    return _finish_to_search(sequence, now_ns, f"gate_clearance_failed_search:{reason}")


def _finish_to_search(sequence: MatchSequence, now_ns: int, reason: str) -> MatchDecision:
    """张爪结束清障；不返回暂存点，也不建立定向回取任务。"""
    from rescue_vision.app.cluster_breakup import GripperPosture
    from rescue_vision.app.match import MatchState
    sequence._finish_noncontact()
    sequence._gate_clearance = None
    sequence._gate_clearance_attempted = False
    sequence._transport_target_classes = ()
    sequence._cargo_capture_floor_ns = None
    sequence._counted_pickup_session = None
    sequence._begin_cluster_search()
    sequence._reset_rotation_budget()
    sequence.state = MatchState.SEARCH_CLUSTER
    return sequence._decision(
        now_ns,
        0.0,
        0.0,
        reason,
        posture=GripperPosture.OPEN,
        soft_brake=True,
    )


def _sweep_risk(sequence: MatchSequence, now_ns: int) -> str | None:
    """新危险侵入仍检查：横扫不允许把危险实体推入安全区或推出场界。"""
    session = sequence._gate_clearance
    assert session is not None
    bounds = sequence._physical_field_bounds()
    end = next(action.point for action in session.actions if action.name == "sweep_via_midpoint")
    assert end is not None
    direction = 1 if end.x > 0 else -1
    footprint = sequence.config.robot_footprint_radius_mm
    # 夹爪及被推物块均在路径前方；向场地中央搬运前的最大前伸包络。
    reach = max(footprint, sequence.config.breakup_gripper_offset_mm)
    for obj in _objects(sequence, now_ns):
        if obj.target_class is not TargetClass.BLUE_DANGER:
            continue
        if abs(obj.point.y-end.y) > reach + obj.radius_mm:
            continue
        pushed = FieldPoint(end.x + direction*reach, obj.point.y)
        radius = obj.radius_mm
        if not (bounds[0]+radius <= pushed.x <= bounds[1]-radius
                and bounds[2]+radius <= pushed.y <= bounds[3]-radius):
            return "danger_sweep_out_of_field"
        for region in sequence._breakup_static_map.regions:
            if region.kind is PhysicalRegionKind.FIELD:
                continue
            if "material" not in region.kind.value and "injured" not in region.kind.value:
                continue
            xs = [p.x for p in region.polygon_field]
            ys = [p.y for p in region.polygon_field]
            # 整个平移线段的实体包络，而不是只有终点。
            if (max(obj.point.x, pushed.x)+radius >= min(xs)
                    and min(obj.point.x, pushed.x)-radius <= max(xs)
                    and min(ys)-radius <= obj.point.y <= max(ys)+radius):
                return "danger_sweep_intersects_safe_zone"
    return None


def step_clearance(sequence: MatchSequence, now_ns: int) -> MatchDecision:
    from rescue_vision.app.cluster_breakup import GripperPosture
    cfg = sequence.config.gate_clearance
    session = sequence._gate_clearance
    assert session is not None
    deadline = session.attempt_started_ns + round(cfg.attempt_timeout_s*1e9)
    if now_ns >= deadline:
        action_name = "initial_stop" if not session.initial_stop_confirmed else session.action.name
        return _fail(sequence, now_ns, f"{action_name}:deadline")
    if not session.initial_stop_confirmed:
        stationary_since = sequence._stationary_motion.stationary_since(now_ns)
        if stationary_since is None:
            position = sequence.estimated_field_position
            current = "none" if position is None else f"({position.x:.1f},{position.y:.1f})"
            return sequence._decision(
                now_ns,
                0.0,
                0.0,
                "gate_clearance:initial_stop_waiting,"
                f"field_position={current},deadline_ns={deadline},"
                + sequence._stationary_motion.diagnostic(now_ns),
                posture=GripperPosture.CLOSED,
                soft_brake=True,
            )
        position = sequence.estimated_field_position
        if position is None:
            return sequence._decision(
                now_ns,
                0.0,
                0.0,
                "gate_clearance:initial_stop_pose_unavailable,"
                f"stationary_since_ns={stationary_since},deadline_ns={deadline}",
                posture=GripperPosture.CLOSED,
                soft_brake=True,
            )
        replanned = make_clearance_session(
            cfg,
            position,
            sequence._team_color,
            session.trigger_capture_ns,
            attempt_started_ns=session.attempt_started_ns,
        )
        session.actions = replanned.actions
        session.initial_stop_confirmed = True
        return sequence._decision(
            now_ns,
            0.0,
            0.0,
            "gate_clearance:initial_stop_complete,"
            f"stationary_since_ns={stationary_since},deadline_ns={deadline},"
            + _action_diagnostic(sequence, session.action),
            posture=GripperPosture.CLOSED,
            soft_brake=True,
        )
    action = session.action
    if session.sweep_realign_pending:
        stationary_since = sequence._stationary_motion.stationary_since(now_ns)
        if stationary_since is None:
            return sequence._decision(
                now_ns,
                0.0,
                0.0,
                "gate_clearance:sweep_realign_waiting_stationary,"
                f"attempt={session.sweep_realign_count},deadline_ns={deadline},"
                + sequence._stationary_motion.diagnostic(now_ns),
                posture=GripperPosture.OPEN,
                soft_brake=True,
            )
        session.sweep_realign_pending = False
        session.index = _action_index(session, "align_sweep_heading")
        session.action_started_ns = None
        session.gripper_started_ns = None
        sequence._finish_noncontact()
        return sequence._decision(
            now_ns,
            0.0,
            0.0,
            "gate_clearance:sweep_realign_ready,"
            f"attempt={session.sweep_realign_count},stationary_since_ns={stationary_since},"
            f"deadline_ns={deadline}",
            posture=GripperPosture.OPEN,
            soft_brake=True,
        )
    if session.action_started_ns is None:
        session.action_started_ns = now_ns
    posture = GripperPosture.OPEN if action.opened else GripperPosture.CLOSED
    if action.kind == "gripper":
        # 必须真正停稳才开始机械等待；零命令不能冒充静止。
        if sequence._stationary_motion.stationary_since(now_ns) is None:
            return sequence._decision(now_ns, 0, 0,
                f"gate_clearance:{action.name}:await_stationary,deadline_ns={deadline},"
                + _action_diagnostic(sequence, action) + ","
                + sequence._stationary_motion.diagnostic(now_ns), posture=posture, soft_brake=True)
        if session.gripper_started_ns is None:
            session.gripper_started_ns = now_ns
        if now_ns-session.gripper_started_ns >= sequence._gripper_full_travel_time_ns:
            _advance(sequence, now_ns)
        return sequence._decision(
            now_ns, 0, 0,
            f"gate_clearance:{action.name},deadline_ns={deadline},"
            + _action_diagnostic(sequence, action),
            posture=posture, soft_brake=True,
        )
    if action.name == "sweep_via_midpoint":
        risk = _sweep_risk(sequence, now_ns)
        if risk is not None:
            return _fail(sequence, now_ns, risk)
        position = sequence.estimated_field_position
        if position is not None and action.point is not None:
            cross_track_error_mm = abs(position.y - action.point.y)
            if cross_track_error_mm > cfg.sweep_cross_track_tolerance_mm:
                return _fail(
                    sequence,
                    now_ns,
                    "sweep_cross_track_outside:"
                    f"error_mm={cross_track_error_mm:.1f},"
                    f"limit_mm={cfg.sweep_cross_track_tolerance_mm:.1f}",
                )
        heading = sequence._latest_heading_rad
        if action.target_heading_rad is not None and heading is not None:
            heading_error = normalize_angle(action.target_heading_rad - heading)
            if abs(heading_error) > cfg.sweep_heading_tolerance_rad:
                sequence._finish_noncontact()
                session.sweep_realign_pending = True
                session.sweep_realign_count += 1
                session.action_started_ns = None
                return sequence._decision(
                    now_ns,
                    0.0,
                    0.0,
                    "gate_clearance:sweep_heading_drift_brake,"
                    f"error_rad={heading_error:.4f},"
                    f"limit_rad={cfg.sweep_heading_tolerance_rad:.4f},"
                    f"attempt={session.sweep_realign_count},deadline_ns={deadline}",
                    posture=GripperPosture.OPEN,
                    soft_brake=True,
                )
    turn = action.kind == "heading"
    heading = sequence._latest_heading_rad
    target = normalize_angle(action.value-heading) if turn and heading is not None else action.value
    if action.name == "sweep_via_midpoint" and action.point is not None:
        position = sequence.estimated_field_position
        if position is not None:
            target = max(1e-6, abs(action.point.x - position.x) / 1000.0)
    speed = (sequence.config.safe_zone_fallback_max_angular_velocity_rad_s if turn
             else cfg.sweep_speed_m_s if action.name == "sweep_via_midpoint" else cfg.transit_speed_m_s)
    command = sequence._noncontact_motion(
        now_ns,
        "gate_" + action.name,
        target=target,
        speed=speed,
        turn=turn,
        point=action.point if action.kind == "point" else None,
        tolerance=(
            cfg.sweep_heading_tolerance_rad
            if action.name == "align_sweep_heading"
            else 0.005
            if action.name == "center_forward_200mm_radius"
            else None
        ),
        target_heading_rad=action.target_heading_rad,
    )
    if command.timed_out:
        return _fail(sequence, now_ns, action.name+":"+command.reason)
    if command.complete:
        if action.name == "center_reverse_120mm":
            return _finish_to_search(
                sequence,
                now_ns,
                "gate_clearance_complete_search",
            )
        _advance(sequence, now_ns)
        return sequence._decision(
            now_ns, 0, 0,
            f"gate_clearance:{action.name}:complete,"
            + _action_diagnostic(sequence, action),
            posture=posture,
        )
    decision = sequence._noncontact_decision(now_ns, command, posture)
    return replace(
        decision,
        reason=(f"{decision.reason},gate_attempt_deadline_ns={deadline},"
                + _action_diagnostic(sequence, action)),
    )
