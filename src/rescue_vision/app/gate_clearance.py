"""实验性门前清障：场地触发几何、暂存任务和可中断动作编排。

不创建硬件。MatchSequence 提供采集位姿、真实运动反馈和现有动作控制器；
重新夹取交还普通近场流程，临时放置不生成交付事件。
"""
from __future__ import annotations

from collections import Counter
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
    from rescue_vision.app.near_field_grasp import NearFieldGraspPlan


@dataclass(frozen=True, slots=True)
class GateObject:
    target_class: TargetClass
    point: FieldPoint
    radius_mm: float


@dataclass(frozen=True, slots=True)
class ClearanceAction:
    name: str
    kind: str  # point / heading / reverse / gripper / reacquire
    opened: bool
    point: FieldPoint | None = None
    value: float = 0.0


@dataclass(slots=True)
class GateClearanceSession:
    actions: tuple[ClearanceAction, ...]
    original_classes: tuple[TargetClass, ...]
    stash: FieldPoint
    trigger_capture_ns: int
    attempt_started_ns: int
    index: int = 0
    action_started_ns: int | None = None
    gripper_started_ns: int | None = None
    released: bool = False
    reacquiring: bool = False
    recovery_attempted: bool = False
    failure: str | None = None

    @property
    def action(self) -> ClearanceAction:
        return self.actions[self.index]


def gate_obstructions(objects: tuple[GateObject, ...], static_map: StaticFieldMap,
                      team: TeamColor, depth_mm: float) -> tuple[GateObject, ...]:
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
            edge = min(sign * p.y for p in region.polygon_field)
            depth = edge - sign * obj.point.y
            if not (min(xs) <= obj.point.x <= max(xs) and 0 < depth <= depth_mm):
                continue
            wrong = (obj.target_class is TargetClass.BLUE_DANGER
                     or region.kind is material and obj.target_class is TargetClass.ORANGE_INJURED
                     or region.kind is injured and obj.target_class in (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE))
            if wrong:
                result.append(obj)
                break
    return tuple(result)


def make_clearance_session(config: GateClearanceConfig, position: FieldPoint,
                           team: TeamColor, cargo: tuple[TargetClass, ...],
                           capture_ns: int, stash_offset_mm: float, *,
                           attempt_started_ns: int | None = None) -> GateClearanceSession:
    """S 点是横扫轴心；暂存轴心外移退出距离，暂存 K0 再按夹取终点外移。"""
    sign = -1 if team is TeamColor.BLUE else 1
    left = FieldPoint(-config.side_x_mm, sign * config.sweep_y_mm)
    right = FieldPoint(config.side_x_mm, sign * config.sweep_y_mm)
    sweep_start, end = sorted((left, right), key=lambda p: math.hypot(p.x-position.x, p.y-position.y))
    # 暂存轴心向外让出退出距离；后退后准确落在 S1/S2 开始完整横扫。
    direction = -1 if sweep_start.x < 0 else 1
    start = FieldPoint(sweep_start.x + direction*config.release_reverse_m*1000, sweep_start.y)
    outward = math.pi if start.x < 0 else 0.0
    sweep_heading = 0.0 if start.x < 0 else math.pi
    center = FieldPoint(end.x, sign * config.center_release_y_mm)
    stash = FieldPoint(start.x + math.cos(outward)*stash_offset_mm, start.y)
    actions = (
        ClearanceAction("to_stash_s", "point", False, start),
        ClearanceAction("face_outward", "heading", False, value=outward),
        ClearanceAction("stash_open", "gripper", True),
        ClearanceAction("stash_reverse_120mm", "reverse", True, value=-config.release_reverse_m),
        ClearanceAction("turn_180", "heading", True, value=sweep_heading),
        # 中点与两端共线，整个线段一次执行，不在中点停车。
        ClearanceAction("sweep_via_midpoint", "point", True, end),
        ClearanceAction("collect_close", "gripper", False),
        ClearanceAction("to_field_center", "point", False, center),
        ClearanceAction("face_field_center", "heading", False, value=-sign*math.pi/2),
        ClearanceAction("center_open", "gripper", True),
        ClearanceAction("center_reverse_120mm", "reverse", True, value=-config.release_reverse_m),
        ClearanceAction("return_to_stash_s", "point", False, start),
        ClearanceAction("face_stash", "heading", False, value=outward),
        ClearanceAction("reacquire", "reacquire", False),
    )
    return GateClearanceSession(
        actions,
        cargo,
        stash,
        capture_ns,
        capture_ns if attempt_started_ns is None else attempt_started_ns,
    )


def reacquire_plan_matches_cargo(
    session: GateClearanceSession,
    plan: NearFieldGraspPlan | None,
) -> bool:
    """回取必须覆盖原载荷的完整类别多重集，不能夹回一部分就去投放。"""
    if plan is None:
        return False
    planned = Counter(member.observation.target_class for member in plan.members)
    return planned == Counter(session.original_classes)


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
    # 末端推进已经开始后不再将已投放物体当作本趟载荷搬走。
    if (sequence.state not in {MatchState.TRANSPORT_ALIGN_RED_ZONE, MatchState.TRANSPORT_FORWARD, MatchState.TRANSPORT_RELEASE}
            or sequence._safe_zone_phase in sequence._D2_ACCELERATION_LIMIT_PHASES):
        return None
    position = sequence.estimated_field_position
    snapshot = sequence._latest_perception
    if position is None or snapshot is None:
        return None
    if not gate_obstructions(_objects(sequence, now_ns), sequence._breakup_static_map,
                             sequence._team_color, cfg.front_depth_mm):
        return None
    near = sequence._near_field_grasp_config
    offset = (near.orange_target_final_x_mm if sequence._transport_target_classes == (TargetClass.ORANGE_INJURED,)
              else near.target_final_x_mm) if near is not None else 110.0
    sequence._gate_clearance = make_clearance_session(cfg, position, sequence._team_color,
        sequence._transport_target_classes, snapshot.capture_timestamp_ns, offset,
        attempt_started_ns=now_ns)
    sequence._gate_clearance_attempted = True
    sequence._finish_noncontact()
    sequence._action_settle_phase = None
    sequence._action_settle_until_ns = None
    sequence._grasp_task = None
    sequence._greedy_active = False
    sequence.state = MatchState.GATE_CLEARANCE
    return sequence._decision(now_ns, 0.0, 0.0, "gate_clearance_triggered", soft_brake=True)


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


def _fail(sequence: MatchSequence, now_ns: int, reason: str) -> MatchDecision:
    """一次失败转为取回任务；不重新扫同一片门前区域。"""
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
    if not session.recovery_attempted:
        # 中途可能已围入杂物：先朝中场释放，绝不闭爪带着杂物去取原载荷。
        session.recovery_attempted = True
        session.index = 7 if session.index < 10 else 11
        session.action_started_ns = None
        return sequence._decision(now_ns, 0, 0, f"gate_clearance_recover_stash:{reason}", soft_brake=True)
    # 已有一次实际回取动作或位置失效，退出本次定点任务；下一次搜索由普通流程处理。
    sequence._gate_clearance = None
    sequence._transport_target_classes = ()
    sequence._cargo_capture_floor_ns = None
    sequence._begin_cluster_search()
    sequence._reset_rotation_budget()
    sequence.state = MatchState.SEARCH_CLUSTER
    return sequence._decision(now_ns, 0, sequence.config.cluster_search_angular_velocity_rad_s,
                              f"gate_clearance_exit_reselect:{reason}", posture=GripperPosture.OPEN)


def _sweep_risk(sequence: MatchSequence, now_ns: int) -> str | None:
    """新危险侵入仍检查：横扫不允许把危险实体推入安全区或推出场界。"""
    session = sequence._gate_clearance
    assert session is not None
    bounds = sequence._physical_field_bounds()
    end = session.actions[5].point
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


def reacquire_path_clear(sequence: MatchSequence, heading: float, distance_m: float,
                         plan: NearFieldGraspPlan | None) -> bool | None:
    """S 点外侧回取检查实际夹爪走廊，避免通用圆形余量把已知安全侧向路径封死。"""
    from rescue_vision.app.breakup_planner import safe_zone_intersection
    position = sequence.estimated_field_position
    if position is None or plan is None or sequence._breakup_static_map is None:
        return None
    if not math.isfinite(distance_m) or distance_m < 0:
        return False
    # 已完整验证的 selector 走廊包含实体前伸和危险物检查，转为当前场地坐标。
    c, s = math.cos(heading), math.sin(heading)
    endpoint = FieldPoint(position.x+c*distance_m*1000, position.y+s*distance_m*1000)
    bounds = sequence._physical_field_bounds()
    radius = sequence.config.robot_footprint_radius_mm
    for point in (position, endpoint):
        if not (bounds[0]+radius <= point.x <= bounds[1]-radius
                and bounds[2]+radius <= point.y <= bounds[3]-radius):
            return False
    # 校验整个轴心至前伸末端的走廊。横向宽度由已提交计划决定。
    # 不允许使用该特例向安全区内夹取，车体与安全区重叠本身不是违规推物证据。
    relative = tuple(p for region in plan.regions for p in region)
    angle = plan.alignment_angle_rad
    rotated = tuple((p.x*math.cos(angle)+p.y*math.sin(angle),
                     -p.x*math.sin(angle)+p.y*math.cos(angle)) for p in relative)
    half_width = max(abs(p[1]) for p in rotated)
    front = max(p[0] for p in rotated) - plan.forward_distance_mm
    tip = FieldPoint(endpoint.x+c*front, endpoint.y+s*front)
    intersects = safe_zone_intersection(
        sequence._breakup_static_map,
        position,
        tip,
        half_width,
    )
    return None if intersects is None else not intersects


def step_clearance(sequence: MatchSequence, now_ns: int) -> MatchDecision:
    from rescue_vision.app.cluster_breakup import GripperPosture
    from rescue_vision.app.near_field_grasp import NearFieldHandoffPrior
    cfg = sequence.config.gate_clearance
    session = sequence._gate_clearance
    assert session is not None
    action = session.action
    if session.action_started_ns is None:
        session.action_started_ns = now_ns
    elapsed = now_ns-session.action_started_ns
    posture = GripperPosture.OPEN if action.opened else GripperPosture.CLOSED
    deadline = session.attempt_started_ns + round(cfg.attempt_timeout_s*1e9)
    if now_ns >= deadline:
        return _fail(sequence, now_ns, f"{action.name}:deadline")
    if action.kind == "gripper":
        # 必须真正停稳才开始机械等待；零命令不能冒充静止。
        if sequence._stationary_motion.stationary_since(now_ns) is None:
            return sequence._decision(now_ns, 0, 0,
                f"gate_clearance:{action.name}:await_stationary,deadline_ns={deadline},"
                + sequence._stationary_motion.diagnostic(now_ns), posture=posture, soft_brake=True)
        if session.gripper_started_ns is None:
            session.gripper_started_ns = now_ns
        if now_ns-session.gripper_started_ns >= sequence._gripper_full_travel_time_ns:
            _advance(sequence, now_ns)
        return sequence._decision(now_ns, 0, 0, f"gate_clearance:{action.name}", posture=posture, soft_brake=True)
    if action.kind == "reacquire":
        candidates = []
        snapshot = sequence._latest_perception
        since = sequence._stationary_motion.stationary_since(now_ns)
        if (snapshot is not None and since is not None
                and snapshot.capture_timestamp_ns >= since
                and sequence.grasp_scene_capture_valid(snapshot, now_ns)):
            for target in sequence._tracker.tracks:
                point = sequence._field_point_for_target(target)
                if (point is not None and target.frame_sequence == snapshot.frame_sequence
                        and target.target_class in session.original_classes
                        and math.hypot(point.x-session.stash.x, point.y-session.stash.y) <= cfg.reacquire_radius_mm):
                    candidates.append(target)
        if candidates:
            target = min(candidates, key=lambda t: math.hypot(t.ground_point.x, t.ground_point.y))
            point = sequence._current_ground_point_for_track(target, now_ns)
            if point is not None:
                session.reacquiring = True
                return sequence._begin_near_field_grasp(now_ns, handoff_prior=NearFieldHandoffPrior(
                    target.target_class, point, target.track_id))
        wait_ms = cfg.observation_timeout_ms
        if elapsed >= wait_ms*1e6:
            return _fail(sequence, now_ns, "stash_not_seen")
        age = None if snapshot is None else (now_ns-snapshot.capture_timestamp_ns)/1e6
        return sequence._decision(now_ns, 0, 0,
            f"gate_clearance:reacquire,confirmation=0/1,capture_age_ms={age},stationary_since_ns={since},"
            f"preparation_age_ms={elapsed/1e6},deadline_ns={session.action_started_ns+round(wait_ms*1e6)}",
            posture=posture, soft_brake=True)
    if action.name == "sweep_via_midpoint":
        risk = _sweep_risk(sequence, now_ns)
        if risk is not None:
            return _fail(sequence, now_ns, risk)
    turn = action.kind == "heading"
    heading = sequence._latest_heading_rad
    target = normalize_angle(action.value-heading) if turn and heading is not None else action.value
    speed = (sequence.config.safe_zone_fallback_max_angular_velocity_rad_s if turn
             else cfg.sweep_speed_m_s if action.name == "sweep_via_midpoint" else cfg.transit_speed_m_s)
    command = sequence._noncontact_motion(now_ns, "gate_"+action.name, target=target,
        speed=speed, turn=turn, point=action.point)
    if command.timed_out:
        return _fail(sequence, now_ns, action.name+":"+command.reason)
    if command.complete:
        _advance(sequence, now_ns)
        return sequence._decision(now_ns, 0, 0, f"gate_clearance:{action.name}:complete", posture=posture)
    decision = sequence._noncontact_decision(now_ns, command, posture)
    return replace(
        decision,
        reason=f"{decision.reason},gate_attempt_deadline_ns={deadline}",
    )
