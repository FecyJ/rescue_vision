"""带身份的场地目标聚类；可复用，尚未接入正式流程。"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rescue_vision.geometry.types import FieldPoint
from rescue_vision.perception import TargetClass
from rescue_vision.world import HazardState


@dataclass(frozen=True, slots=True)
class FieldClusterMember:
    """一个目标在重复解团选团时冻结的身份和场地位置。"""

    track_id: int
    target_class: TargetClass
    hazard_state: HazardState
    field_point: FieldPoint
    from_memory: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id <= 0
        ):
            raise ValueError("track_id must be a positive integer.")
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass.")
        if not isinstance(self.hazard_state, HazardState):
            raise ValueError("hazard_state must be a HazardState.")
        if not isinstance(self.field_point, FieldPoint):
            raise ValueError("field_point must be a FieldPoint.")
        if not isinstance(self.from_memory, bool):
            raise ValueError("from_memory must be a boolean.")


@dataclass(frozen=True, slots=True)
class FieldTargetCluster:
    """由 FieldPoint 邻接关系形成的带身份连通分量。"""

    members: tuple[FieldClusterMember, ...]
    centroid_field: FieldPoint

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("members must contain at least two targets.")
        if len({member.track_id for member in self.members}) != len(self.members):
            raise ValueError("members must contain unique track IDs.")
        if not isinstance(self.centroid_field, FieldPoint):
            raise ValueError("centroid_field must be a FieldPoint.")

    @property
    def member_ids(self) -> tuple[int, ...]:
        return tuple(member.track_id for member in self.members)

    @property
    def green_members(self) -> tuple[FieldClusterMember, ...]:
        return tuple(
            member
            for member in self.members
            if member.target_class is TargetClass.GREEN_SUPPLY
        )

    @property
    def from_memory(self) -> bool:
        return all(member.from_memory for member in self.members)


def connected_field_clusters(
    members: Sequence[FieldClusterMember],
    *,
    neighbor_distance_mm: float,
    edge_allowed: Callable[[FieldClusterMember, FieldClusterMember], bool]
    | None = None,
) -> tuple[FieldTargetCluster, ...]:
    """按 FieldPoint 欧氏距离返回至少两个成员的确定性连通分量。"""

    threshold = float(neighbor_distance_mm)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("neighbor_distance_mm must be finite and positive.")
    values = tuple(members)
    if not all(isinstance(member, FieldClusterMember) for member in values):
        raise TypeError("members must contain only FieldClusterMember values.")
    if len({member.track_id for member in values}) != len(values):
        raise ValueError("members must contain unique track IDs.")
    if len(values) < 2:
        return ()

    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first_index, first in enumerate(values):
        for second_index in range(first_index + 1, len(values)):
            second = values[second_index]
            if edge_allowed is not None and not edge_allowed(first, second):
                continue
            if (
                math.hypot(
                    first.field_point.x - second.field_point.x,
                    first.field_point.y - second.field_point.y,
                )
                > threshold
            ):
                continue
            first_root = find(first_index)
            second_root = find(second_index)
            if first_root != second_root:
                parent[second_root] = first_root

    grouped: dict[int, list[FieldClusterMember]] = {}
    for index, member in enumerate(values):
        grouped.setdefault(find(index), []).append(member)

    clusters = []
    for component in grouped.values():
        if len(component) < 2:
            continue
        ordered = tuple(sorted(component, key=lambda item: item.track_id))
        clusters.append(
            FieldTargetCluster(
                members=ordered,
                centroid_field=FieldPoint(
                    sum(member.field_point.x for member in ordered) / len(ordered),
                    sum(member.field_point.y for member in ordered) / len(ordered),
                ),
            )
        )
    clusters.sort(key=lambda cluster: cluster.member_ids)
    return tuple(clusters)
