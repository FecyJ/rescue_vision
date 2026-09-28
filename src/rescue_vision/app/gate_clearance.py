"""实验性门前路线：识别门前异类物块后执行一次定距交付动作。

不创建硬件。MatchSequence 提供当前感知、定位、真实运动反馈和动作执行器；
交付结束后复用正式安全区倒车退出流程。
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
    kind: str  # point / heading / straight / reverse
    point: FieldPoint | None = None
    value: float = 0.0
    target_heading_rad: float | None = None


@dataclass(slots=True)
class GateClearanceSession:
    actions: tuple[ClearanceAction, ...]
    start_point: FieldPoint
    lane_point: FieldPoint
    lateral_end_point: FieldPoint
    lateral_return_point: FieldPoint
    d2_push_end_point: FieldPoint
    trigger_capture_ns: int
    attempt_started_ns: int
    index: int = 0
    initial_stop_confirmed: bool = False

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
            edge = min(sign * p.y for p in region.polygon_field) - config.front_edge_inset_mm
            depth = edge - sign * obj.point.y
            if not (x_min <= obj.point.x <= x_max and 0 < depth <= config.front_depth_mm):
                continue
            wrong = (
                obj.target_class is TargetClass.BLUE_DANGER
                or region.kind is material and obj.target_class is TargetClass.ORANGE_INJURED
                or region.kind is injured and obj.target_class in (
                    TargetClass.GREEN_SUPPLY,
                    TargetClass.BLACK_CORE,
                )
            )
            if wrong:
                result.append(obj)
                break
    return tuple(result)


def make_clearance_session(
    config: GateClearanceConfig,
    position: FieldPoint,
    team: TeamColor,
    capture_ns: int,
    *,
    target_classes: tuple[TargetClass, ...] = (),
    attempt_started_ns: int | None = None,
) -> GateClearanceSession:
    """生成从 D1 到门前横移点，再移交正式 D2 投递的动作序列。

    显式坐标以红方为基准；蓝方通过绕场地原点旋转 180°获得中心对称路线。
    """
    orange = target_classes == (TargetClass.ORANGE_INJURED,)
    red_x_mm = config.lane_x_abs_mm if orange else -config.lane_x_abs_mm
    team_sign = -1 if team is TeamColor.BLUE else 1
    lane = FieldPoint(team_sign * red_x_mm, team_sign * config.lane_y_abs_mm)

    # 绿/黑从红方 D1 朝 +x 清行，橙色朝 -x；蓝方中心对称转换。
    red_lateral_heading = math.pi if orange else 0.0
    lateral_heading = normalize_angle(red_lateral_heading + (math.pi if team_sign < 0 else 0.0))
    lateral_direction = 1.0 if math.cos(lateral_heading) >= 0.0 else -1.0
    lateral_end = FieldPoint(
        lane.x + lateral_direction * config.lateral_forward_distance_m * 1000.0,
        lane.y,
    )
    lateral_return = FieldPoint(
        lateral_end.x - lateral_direction * config.lateral_reverse_distance_m * 1000.0,
        lane.y,
    )
    d2_push_end = FieldPoint(
        lateral_return.x,
        lateral_return.y + team_sign * config.d2_push_distance_m * 1000.0,
    )

    actions = (
        ClearanceAction("to_lane", "point", point=lane),
        ClearanceAction("align_lateral_heading", "heading", value=lateral_heading),
        ClearanceAction(
            "lateral_forward_900mm",
            "straight",
            value=config.lateral_forward_distance_m,
            target_heading_rad=lateral_heading,
        ),
        ClearanceAction(
            "lateral_reverse_600mm",
            "reverse",
            value=-config.lateral_reverse_distance_m,
            target_heading_rad=lateral_heading,
        ),
    )
    return GateClearanceSession(
        actions=actions,
        start_point=position,
        lane_point=lane,
        lateral_end_point=lateral_end,
        lateral_return_point=lateral_return,
        d2_push_end_point=d2_push_end,
        trigger_capture_ns=capture_ns,
        attempt_started_ns=capture_ns if attempt_started_ns is None else attempt_started_ns,
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
    # 新动作必须从 D1 开始，因此仅在两帧校准完成、D2 直线尚未启动时拦截。
    calibrated_d1 = (
        sequence._safe_zone_calibration_pose is not None
        and sequence.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
        and sequence._safe_zone_phase == "align_d2_line"
    )
    if not calibrated_d1:
        return None
    position = sequence.estimated_field_position
    snapshot = sequence._latest_perception
    if position is None or snapshot is None:
        return None
    obstructions = gate_obstructions(
        _objects(sequence, now_ns), sequence._breakup_static_map, sequence._team_color, cfg
    )
    if not obstructions:
        return None
    sequence._gate_clearance = make_clearance_session(
        cfg,
        position,
        sequence._team_color,
        snapshot.capture_timestamp_ns,
        target_classes=sequence._transport_target_classes,
        attempt_started_ns=now_ns,
    )
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
        f"lane=({sequence._gate_clearance.lane_point.x:.1f},"
        f"{sequence._gate_clearance.lane_point.y:.1f})",
        soft_brake=True,
    )


def _action_diagnostic(sequence: MatchSequence, action: ClearanceAction) -> str:
    position = sequence.estimated_field_position
    current = "none" if position is None else f"({position.x:.1f},{position.y:.1f})"
    target = "none" if action.point is None else f"({action.point.x:.1f},{action.point.y:.1f})"
    target_heading_rad = action.value if action.kind == "heading" else action.target_heading_rad
    heading = "none" if target_heading_rad is None else f"{math.degrees(target_heading_rad):.1f}deg"
    return f"field_position={current},target_field={target},target_heading={heading}"


def _advance(sequence: MatchSequence) -> None:
    session = sequence._gate_clearance
    assert session is not None
    session.index += 1
    sequence._finish_noncontact()


def _segment_risk(sequence: MatchSequence, now_ns: int,
                  start: FieldPoint, end: FieldPoint) -> str | None:
    """拒绝可能把蓝色危险物推入任一安全区或场界外的线段。"""
    session = sequence._gate_clearance
    assert session is not None
    bounds = sequence._physical_field_bounds()
    reach = max(
        sequence.config.robot_footprint_radius_mm,
        sequence.config.breakup_gripper_offset_mm,
    )
    dx = end.x - start.x
    dy = end.y - start.y
    segment_length_sq = dx * dx + dy * dy
    if segment_length_sq <= 1e-9:
        return None
    for obj in _objects(sequence, now_ns):
        if obj.target_class is not TargetClass.BLUE_DANGER:
            continue
        fraction = max(0.0, min(1.0, (
            (obj.point.x - start.x) * dx + (obj.point.y - start.y) * dy
        ) / segment_length_sq))
        closest = FieldPoint(start.x + fraction * dx, start.y + fraction * dy)
        if math.hypot(obj.point.x - closest.x, obj.point.y - closest.y) > reach + obj.radius_mm:
            continue
        norm = math.sqrt(segment_length_sq)
        for direction in (-1.0, 1.0):
            pushed = FieldPoint(
                obj.point.x + direction * dx / norm * reach,
                obj.point.y + direction * dy / norm * reach,
            )
            radius = obj.radius_mm
            if not (bounds[0] + radius <= pushed.x <= bounds[1] - radius
                    and bounds[2] + radius <= pushed.y <= bounds[3] - radius):
                return "danger_route_out_of_field"
            swept_min_x = min(obj.point.x, pushed.x) - radius
            swept_max_x = max(obj.point.x, pushed.x) + radius
            swept_min_y = min(obj.point.y, pushed.y) - radius
            swept_max_y = max(obj.point.y, pushed.y) + radius
            for region in sequence._breakup_static_map.regions:
                if region.kind is PhysicalRegionKind.FIELD:
                    continue
                if "material" not in region.kind.value and "injured" not in region.kind.value:
                    continue
                xs = [p.x for p in region.polygon_field]
                ys = [p.y for p in region.polygon_field]
                if (swept_max_x >= min(xs) and swept_min_x <= max(xs)
                        and swept_max_y >= min(ys) and swept_min_y <= max(ys)):
                    return "danger_route_intersects_safe_zone"
    return None


def _action_risk(sequence: MatchSequence, now_ns: int, action: ClearanceAction) -> str | None:
    session = sequence._gate_clearance
    assert session is not None
    segment = {
        "to_lane": (session.start_point, session.lane_point),
        "lateral_forward_900mm": (session.lane_point, session.lateral_end_point),
        "lateral_reverse_600mm": (session.lateral_end_point, session.lateral_return_point),
    }.get(action.name)
    return None if segment is None else _segment_risk(sequence, now_ns, *segment)


def _fail(sequence: MatchSequence, now_ns: int, reason: str) -> MatchDecision:
    """门前横移移交正式 D2 投递前失败时，保持载荷并恢复普通运输。"""
    from rescue_vision.app.cluster_breakup import GripperPosture

    sequence._finish_noncontact()
    sequence._gate_clearance = None
    sequence._safe_zone_phase = "idle"
    return sequence._start_safe_zone_transport(
        now_ns,
        transport_opened=False,
        posture=GripperPosture.CLOSED,
        reason=f"gate_clearance_aborted_resume_transport:{reason}",
    )


def _handoff_to_normal_d2_delivery(sequence: MatchSequence, now_ns: int) -> MatchDecision:
    """横移后交给正式 D2 开爪、对准、闭爪推入及倒车状态机。"""
    from rescue_vision.app.match import MatchState
    from rescue_vision.app.cluster_breakup import GripperPosture

    session = sequence._gate_clearance
    assert session is not None
    risk = _segment_risk(
        sequence, now_ns, session.lateral_return_point, session.d2_push_end_point
    )
    if risk is not None:
        return _fail(sequence, now_ns, risk)
    sequence._finish_noncontact()
    sequence._gate_clearance = None
    sequence._gate_clearance_push_distance_m = sequence.config.gate_clearance.d2_push_distance_m
    sequence._gate_clearance_attempted = True
    sequence._transport_opened = False
    sequence._gripper_phase_started_ns = None
    sequence._safe_zone_stop_since_ns = None
    sequence._safe_zone_phase = "stopping_before_d2_opening"
    sequence._transport_forward_base_distance_m = None
    sequence._transport_forward_distance_m = None
    sequence.state = MatchState.TRANSPORT_RELEASE
    position = sequence.estimated_field_position
    current = "none" if position is None else f"({position.x:.1f},{position.y:.1f})"
    return sequence._decision(
        now_ns,
        0.0,
        0.0,
        "gate_clearance:lateral_reverse_600mm:complete_handoff_to_normal_d2,"
        f"field_position={current},target_field=({session.d2_push_end_point.x:.1f},"
        f"{session.d2_push_end_point.y:.1f}),d2_push_mm="
        f"{sequence.config.gate_clearance.d2_push_distance_m * 1000:.1f}",
        posture=GripperPosture.CLOSED,
        soft_brake=True,
    )


def step_clearance(sequence: MatchSequence, now_ns: int) -> MatchDecision:
    from rescue_vision.app.cluster_breakup import GripperPosture

    cfg = sequence.config.gate_clearance
    session = sequence._gate_clearance
    assert session is not None
    deadline = session.attempt_started_ns + round(cfg.attempt_timeout_s * 1e9)
    if now_ns >= deadline:
        return _fail(sequence, now_ns, f"{session.action.name}:deadline")
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
            target_classes=sequence._transport_target_classes,
            attempt_started_ns=session.attempt_started_ns,
        )
        session.actions = replanned.actions
        session.start_point = replanned.start_point
        session.lane_point = replanned.lane_point
        session.lateral_end_point = replanned.lateral_end_point
        session.lateral_return_point = replanned.lateral_return_point
        session.d2_push_end_point = replanned.d2_push_end_point
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
    risk = _action_risk(sequence, now_ns, action)
    if risk is not None:
        return _fail(sequence, now_ns, risk)

    turn = action.kind == "heading"
    heading = sequence._latest_heading_rad
    target = normalize_angle(action.value - heading) if turn and heading is not None else action.value
    speed = (
        sequence.config.safe_zone_fallback_max_angular_velocity_rad_s
        if turn
        else cfg.transit_speed_m_s
    )
    command = sequence._noncontact_motion(
        now_ns,
        "gate_" + action.name,
        target=target,
        speed=speed,
        turn=turn,
        point=action.point if action.kind == "point" else None,
        target_heading_rad=action.target_heading_rad,
    )
    if command.timed_out:
        return _fail(sequence, now_ns, action.name + ":" + command.reason)
    if command.complete:
        if action.name == "lateral_reverse_600mm":
            return _handoff_to_normal_d2_delivery(sequence, now_ns)
        _advance(sequence)
        return sequence._decision(
            now_ns,
            0.0,
            0.0,
            f"gate_clearance:{action.name}:complete,"
            + _action_diagnostic(sequence, action),
            posture=GripperPosture.CLOSED,
        )
    decision = sequence._noncontact_decision(now_ns, command, GripperPosture.CLOSED)
    return replace(
        decision,
        reason=f"{decision.reason},gate_attempt_deadline_ns={deadline},"
        + _action_diagnostic(sequence, action),
    )
