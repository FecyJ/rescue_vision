"""Local contact-based breakup plans. Geometry is in mm; no hardware or tracking state."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from rescue_vision.perception.target_ground_geometry import BoxTargetGeometry, TargetGeometry
from rescue_vision.world.static_map import StaticFieldMap, TeamColor

if TYPE_CHECKING:
    from rescue_vision.config import MatchRuntimeConfig


@dataclass(frozen=True, slots=True)
class BreakupTarget:
    track_id: int
    capture_timestamp_ns: int
    target_class: TargetClass
    center: GroundPoint
    contact_radius_mm: float
    safety_radius_mm: float

    def __post_init__(self) -> None:
        for name in ('track_id', 'capture_timestamp_ns'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == 'track_id' else 0):
                raise ValueError(f'{name} has invalid value {value!r}')
        if not isinstance(self.target_class, TargetClass) or not isinstance(self.center, GroundPoint):
            raise ValueError(f'Invalid breakup target: {self!r}')
        if not all(math.isfinite(v) for v in (self.center.x, self.center.y)):
            raise ValueError(f'Invalid center: {self.center!r}')
        for name in ("contact_radius_mm", "safety_radius_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 < self.contact_radius_mm <= self.safety_radius_mm:
            raise ValueError(f'Invalid radii: {self.contact_radius_mm!r}, {self.safety_radius_mm!r}')


def physical_radii(geometry: TargetGeometry) -> tuple[float, float]:
    """Inscribed contact disk and circumscribed safety disk, independent of yaw."""
    if isinstance(geometry, BoxTargetGeometry):
        return min(geometry.length_mm, geometry.width_mm) / 2, math.hypot(geometry.length_mm, geometry.width_mm) / 2
    return geometry.edge_mm / math.sqrt(12), geometry.edge_mm / math.sqrt(3)


@dataclass(frozen=True, slots=True)
class BreakupPlan:
    member_ids: tuple[int, ...]
    contact_ids: tuple[int, ...]
    aim_id: int
    aim: GroundPoint
    heading_rad: float
    approach_distance_mm: float
    forward_distance_mm: float
    backward_distance_mm: float
    penetration_mm: float
    capture_timestamp_ns: int
    attempt: int
    member_field_points: tuple[FieldPoint, ...]
    aim_field: FieldPoint
    sweep_half_width_mm: float
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.member_ids, tuple) or not isinstance(self.contact_ids, tuple):
            raise ValueError("member_ids and contact_ids must be tuples.")
        if (not self.member_ids or any(
            isinstance(member_id, bool) or not isinstance(member_id, int) or member_id <= 0
            for member_id in self.member_ids
        ) or len(set(self.member_ids)) != len(self.member_ids)):
            raise ValueError("member_ids must contain unique members.")
        if (not self.contact_ids or any(
            isinstance(member_id, bool) or not isinstance(member_id, int) or member_id <= 0
            for member_id in self.contact_ids
        ) or not set(self.contact_ids).issubset(self.member_ids)):
            raise ValueError("contact_ids must be a non-empty subset of member_ids.")
        if self.aim_id not in self.member_ids or self.aim_id not in self.contact_ids:
            raise ValueError("aim_id must identify a contact member.")
        if not isinstance(self.aim, GroundPoint) or not isinstance(self.aim_field, FieldPoint):
            raise ValueError("aim and aim_field must use explicit coordinate types.")
        if not isinstance(self.member_field_points, tuple) or not isinstance(self.rejection_reasons, tuple):
            raise ValueError("member_field_points and rejection_reasons must be tuples.")
        if len(self.member_field_points) != len(self.member_ids):
            raise ValueError("member_field_points must align with member_ids.")
        if not all(isinstance(point, FieldPoint) for point in self.member_field_points):
            raise ValueError("member_field_points must contain FieldPoint values.")
        for name in (
            "heading_rad", "approach_distance_mm", "forward_distance_mm",
            "backward_distance_mm", "penetration_mm", "sweep_half_width_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name != "heading_rad" and value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if isinstance(self.capture_timestamp_ns, bool) or not isinstance(self.capture_timestamp_ns, int) or self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be a non-negative integer.")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer.")
        if any(not isinstance(reason, str) or not reason.strip() for reason in self.rejection_reasons):
            raise ValueError("rejection_reasons must contain non-empty strings.")


def field_point(point: GroundPoint, origin: FieldPoint, heading: float) -> FieldPoint:
    c, s = math.cos(heading), math.sin(heading)
    return FieldPoint(origin.x + c * point.x - s * point.y, origin.y + s * point.x + c * point.y)


def segment_intersects_box(start: FieldPoint, end: FieldPoint, bounds: tuple[float, float, float, float]) -> bool:
    low, high = 0.0, 1.0
    xmin, xmax, ymin, ymax = bounds
    for origin, delta, lower, upper in ((start.x, end.x-start.x, xmin, xmax), (start.y, end.y-start.y, ymin, ymax)):
        if abs(delta) < 1e-9:
            if origin < lower or origin > upper:
                return False
        else:
            entry, leave = sorted(((lower-origin)/delta, (upper-origin)/delta))
            low, high = max(low, entry), min(high, leave)
            if low > high:
                return False
    return True


def safe_zone_intersection(static_map: StaticFieldMap, start: FieldPoint, end: FieldPoint, margin_mm: float) -> bool | None:
    for color in (TeamColor.RED, TeamColor.BLUE):
        polygon = static_map.safe_zone_polygon_field(color)
        if polygon is None:
            return None
        bounds = (min(p.x for p in polygon)-margin_mm, max(p.x for p in polygon)+margin_mm,
                  min(p.y for p in polygon)-margin_mm, max(p.y for p in polygon)+margin_mm)
        if segment_intersects_box(start, end, bounds):
            return True
    return False


def segment_clear(static_map: StaticFieldMap, start: FieldPoint, end: FieldPoint,
                  bounds: tuple[float, float, float, float], margin_mm: float) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return all(xmin+margin_mm < p.x < xmax-margin_mm and ymin+margin_mm < p.y < ymax-margin_mm
               for p in (start, end)) and safe_zone_intersection(static_map, start, end, margin_mm) is False


def robot_clearance_mm(config: MatchRuntimeConfig, front_mm: float) -> float:
    """Radius enclosing the body and jaw reach, plus one safety allowance."""
    return max(config.robot_footprint_radius_mm, config.breakup_gripper_offset_mm, front_mm) + config.safety_margin_mm


def connected_groups(targets: tuple[BreakupTarget, ...], distance_mm: float) -> tuple[tuple[BreakupTarget, ...], ...]:
    remaining = {t.track_id: t for t in targets}
    groups = []
    while remaining:
        seed = remaining.pop(min(remaining))
        group, pending = [seed], [seed]
        while pending:
            current = pending.pop()
            neighbors = [t for t in remaining.values() if math.hypot(t.center.x-current.center.x, t.center.y-current.center.y) <= distance_mm]
            for item in neighbors:
                remaining.pop(item.track_id)
                pending.append(item)
                group.append(item)
        groups.append(tuple(sorted(group, key=lambda t: t.track_id)))
    return tuple(groups)


def same_local_group(a: tuple[FieldPoint, ...], b: tuple[FieldPoint, ...], tolerance_mm: float) -> bool:
    """Spatial majority match survives tracker resets, but not a different distant group."""
    if not a or not b:
        return False
    def matched(source: tuple[FieldPoint, ...], dest: tuple[FieldPoint, ...]) -> int:
        return sum(any(math.hypot(p.x-q.x, p.y-q.y) <= tolerance_mm for q in dest) for p in source)
    return matched(a, b) >= math.ceil(len(a)/2) and matched(b, a) >= math.ceil(len(b)/2)


def plan_breakup(
    targets: tuple[BreakupTarget, ...], *, config: MatchRuntimeConfig,
    origin: FieldPoint, heading_rad: float, static_map: StaticFieldMap,
    field_bounds: tuple[float, float, float, float], front_mm: float,
    allowed_classes: frozenset[TargetClass], attempt: int = 1,
    previous_aim: FieldPoint | None = None, previous_penetration_mm: float = 0.0,
    required_ids: frozenset[int] | None = None, approach: bool = True,
    rejection_reasons: tuple[str, ...] = (),
    priority_ids: frozenset[int] = frozenset(),
    required_aim_id: int | None = None,
    non_contact_ids: frozenset[int] = frozenset(),
    rejections: list[str] | None = None,
) -> tuple[BreakupPlan, ...]:
    """Rank real K0 contact rays. Check entire group displacement, including blue.

    Conservative disk/swept-segment checks are predictions, not a physical sliding bound.
    Unknown objects may belong to a group and its safety checks, never define a contact ray.
    field_bounds are physical field edges, without prior jaw/body insets.

    ``rejections`` 是可选的诊断收集器：每个被淘汰的候选按组/瞄准点追加一行
    具体原因和当时的关键数值。没有它，现场只能看到"选不出计划"这一个结论，
    无法判断是接触半径、最小推进深度还是路径净空拦掉的。
    """
    if not math.isfinite(heading_rad) or not math.isfinite(front_mm) or front_mm <= 0:
        raise ValueError(f'Invalid heading/front: {heading_rad!r}, {front_mm!r}')
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f'Invalid attempt: {attempt!r}')

    def skip(reason: str, aim_id: int | None = None, **facts: object) -> None:
        if rejections is None:
            return
        detail = " ".join(f"{name}={value}" for name, value in facts.items())
        rejections.append(
            f"aim={aim_id if aim_id is not None else 'none'} reason={reason}"
            + (f" {detail}" if detail else "")
        )

    ranked = []
    # 已交付（安全区内）成员不再作为成组和瞄准点候选，但仍留在 ``targets``
    # 里参加下面的推移净空检查——不能被碰撞，也不能被忽略。
    contact_targets = tuple(
        target for target in targets if target.track_id not in non_contact_ids
    )
    for group in connected_groups(contact_targets, config.cluster_group_ground_mm):
        ids = tuple(t.track_id for t in group)
        if len(group) < config.cluster_min_detections or not any(t.target_class in allowed_classes for t in group):
            skip("group_too_small_or_no_allowed_class",
                 size=len(group), ids=ids,
                 min_detections=config.cluster_min_detections)
            continue
        if required_ids is not None and frozenset(ids) != required_ids:
            skip("group_not_required_ids", ids=ids, required=sorted(required_ids))
            continue
        points = tuple(field_point(t.center, origin, heading_rad) for t in group)
        for aim in group:
            if required_aim_id is not None and aim.track_id != required_aim_id:
                continue
            if aim.center.x <= 0:
                skip("aim_behind_robot", aim.track_id, x=f"{aim.center.x:.1f}")
                continue
            bearing = math.atan2(aim.center.y, aim.center.x)
            c, s = math.cos(bearing), math.sin(bearing)
            # Only inscribed disks establish contact; inflated uncertainty never invents a hit.
            contact = []
            for item in group:
                x, y = c*item.center.x+s*item.center.y, -s*item.center.x+c*item.center.y
                if x > front_mm and abs(y) < item.contact_radius_mm:
                    half = math.sqrt(max(0.0, item.contact_radius_mm**2-y*y))
                    contact.append((item, x-half, x+half))
            if not contact:
                skip("no_contact_disk_crosses_jaw_line", aim.track_id,
                     front_mm=f"{front_mm:.1f}",
                     nearest_x=f"{min(c*t.center.x+s*t.center.y for t in group):.1f}",
                     max_contact_radius=f"{max(t.contact_radius_mm for t in group):.1f}")
                continue
            near = min(x0 for _, x0, _ in contact)
            approach_mm = max(0.0, near-config.cluster_breakup_standoff_mm) if approach else 0.0
            gap = max(0.0, near-approach_mm-front_mm)
            cap = config.breakup_penetration_mm if attempt == 1 else config.breakup_retry_penetration_mm
            # Stop after opening a local pocket; never chase the far end of a long chain.
            local = [(t, x0, x1) for t, x0, x1 in contact if x0 <= near+cap]
            penetration = min(cap, max(x1 for _, _, x1 in local)-near)
            max_penetration = min(config.breakup_forward_distance_m*1000-gap,
                                  config.breakup_backward_distance_m*1000-config.breakup_retreat_clearance_mm-config.breakup_braking_margin_mm,
                                  penetration)
            aim_field = field_point(aim.center, origin, heading_rad)
            repeated = previous_aim is not None and math.hypot(aim_field.x-previous_aim.x, aim_field.y-previous_aim.y) <= aim.contact_radius_mm
            direction = heading_rad + bearing
            def endpoint(start: FieldPoint, distance: float) -> FieldPoint:
                return FieldPoint(start.x+distance*math.cos(direction), start.y+distance*math.sin(direction))
            blocked_path = "none"
            def path_clear(start: FieldPoint, end: FieldPoint, margin: float, label: str) -> bool:
                nonlocal blocked_path
                if segment_clear(static_map, start, end, field_bounds, margin):
                    return True
                xmin, xmax, ymin, ymax = field_bounds
                boundary = any(not (xmin+margin < p.x < xmax-margin
                                    and ymin+margin < p.y < ymax-margin) for p in (start, end))
                blocked_path = (f"{label}:{'field_boundary' if boundary else 'safe_zone_or_missing_map'}"
                                f":start=({start.x:.1f},{start.y:.1f})"
                                f":end=({end.x:.1f},{end.y:.1f}):margin_mm={margin:.1f}")
                return False
            def clear(depth: float) -> bool:
                forward = gap+depth
                end = endpoint(origin, approach_mm+forward+config.breakup_braking_margin_mm)
                retreat_end = endpoint(origin, approach_mm+gap-config.breakup_retreat_clearance_mm-config.breakup_braking_margin_mm)
                margin = robot_clearance_mm(config, front_mm)
                if not path_clear(origin, end, margin, "robot_forward") or not path_clear(end, retreat_end, margin, "robot_retreat"):
                    return False
                # All members may transmit a push. Do not discard blue/unknown or peripheral members.
                displacement = depth+config.breakup_push_margin_mm+config.breakup_braking_margin_mm
                affected = [(t, field_point(t.center, origin, heading_rad)) for t in targets
                            if t.track_id in ids or (
                                0 < c*t.center.x+s*t.center.y <= near+depth+t.safety_radius_mm
                                and abs(-s*t.center.x+c*t.center.y) <= config.robot_footprint_radius_mm+t.safety_radius_mm)]
                return all(path_clear(p, endpoint(p, max(displacement,
                               near+depth-(c*t.center.x+s*t.center.y)+t.safety_radius_mm)),
                               t.safety_radius_mm+config.breakup_push_margin_mm,
                               f"target_{t.track_id}_{t.target_class.value}")
                           for t, p in affected)
            if max_penetration <= 0 or not clear(0):
                skip("local_push_not_clear_or_no_penetration", aim.track_id,
                     near_mm=f"{near:.1f}", gap_mm=f"{gap:.1f}",
                     penetration_mm=f"{penetration:.1f}",
                     max_penetration_mm=f"{max_penetration:.1f}", blocked_path=blocked_path,
                     physical_field_bounds=field_bounds)
                continue
            depth = max_penetration
            if not clear(max_penetration):
                low, high = 0.0, max_penetration
                for _ in range(16):
                    mid = (low+high)/2
                    if clear(mid):
                        low = mid
                    else:
                        high = mid
                depth = low
            # 局部接触集自身的纵向跨度就是推穿它所需的全部行程：比固定门槛浅的
            # 团只要能整段推穿就成立，不该被门槛永久淘汰。40 mm 正四面体黑核的
            # 内切接触盘只有 2r≈23 mm，取固定门槛（10+20=30 mm）时它永远选不出
            # 瞄准点；刹车余量已经由行程上限和扫掠外延各扣一次，不该再当成
            # 物料深度要求。门槛取两者较小值后，浅团必须整段推穿才算成立。
            minimum_depth = min(config.breakup_min_penetration_mm
                                + config.breakup_braking_margin_mm, penetration)
            if depth < minimum_depth or (repeated and depth <= previous_penetration_mm+1e-6):
                skip("penetration_below_minimum", aim.track_id,
                     depth_mm=f"{depth:.1f}", minimum_mm=f"{minimum_depth:.1f}",
                     repeated=repeated, blocked_path=blocked_path)
                continue
            forward = gap+depth
            hit_ids = tuple(t.track_id for t, x0, _ in contact if x0 <= near+depth)
            if aim.track_id not in hit_ids:
                # An aim member that would remain beyond the selected local
                # penetration is not a real impact point; skip it instead of
                # falling back to an empty-space ray.
                skip("aim_outside_local_penetration", aim.track_id,
                     hit_ids=hit_ids, depth_mm=f"{depth:.1f}")
                continue
            density = sum(math.hypot(t.center.x-aim.center.x, t.center.y-aim.center.y) <= config.cluster_group_ground_mm for t in group)
            plan = BreakupPlan(ids, hit_ids, aim.track_id, aim.center, direction, approach_mm,
                               forward, depth+config.breakup_retreat_clearance_mm+config.breakup_braking_margin_mm, depth,
                               min(t.capture_timestamp_ns for t in group), attempt, points, aim_field,
                               config.robot_footprint_radius_mm, rejection_reasons)
            blocked_priority = not bool(priority_ids.intersection(ids))
            ranked.append(((blocked_priority, repeated, -len(hit_ids), -density,
                            approach_mm+forward, abs(bearing), ids, aim.track_id), plan))
    return tuple(plan for _, plan in sorted(ranked, key=lambda item: item[0]))
