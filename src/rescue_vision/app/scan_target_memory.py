"""SCAN_GREEN 扫描期间的目标短时记忆。

扫描旋转时每个周期把"已确认且双坐标齐全"的目标场地点存档；此后任意时刻都
可以用当前位姿把记忆条目重投影回机器人地面系，让视野外目标继续参与候选
规划、走廊阻塞与再解团成团判断。成团证据只来自"同一周期同时在线"的近距
目标对，跨趟位姿漂移产生的重复条目因此永远不会伪造出待解团。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D
from rescue_vision.perception import ClassProbabilities, TargetClass
from rescue_vision.tracking import TrackStatus
from rescue_vision.world import HazardState, WorldTarget

# 记忆 id 从远大于 tracker 自增 id 的基数开始，保证合成 WorldTarget 的
# track_id 永不与实时轨迹冲突（WorldSnapshot 要求 track_id 唯一）。
_MEMORY_ID_BASE = 1_000_000


def ground_from_field(pose: FieldPose2D, point: FieldPoint) -> GroundPoint:
    """场地点按位姿逆变换回机器人地面系。

    使用场地位置差和逆航向旋转；测试以解析正变换核对往返结果。
    """

    dx = point.x - pose.position.x
    dy = point.y - pose.position.y
    cosine = math.cos(pose.heading_rad)
    sine = math.sin(pose.heading_rad)
    return GroundPoint(
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
    )


def _distance(first: FieldPoint, second: FieldPoint) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


@dataclass(frozen=True, slots=True)
class ScanMemoryEntry:
    """一个扫描记忆条目：记录时刻的目标属性与场地位置。

    ``last_seen_timestamp_ns`` 保持该目标最后一次真实被观测的时刻，不随
    机器人旋转刷新；记忆老化完全由它决定。
    """

    memory_id: int
    target_class: TargetClass
    class_probabilities: ClassProbabilities
    confidence: float
    hazard_state: HazardState
    field_point: FieldPoint
    last_seen_timestamp_ns: int


class ScanTargetMemory:
    """SCAN_GREEN 期间可靠目标的短时场地记忆。

    尚未接入正式流程。生命周期由未来调用方控制：进入扫描后每周期 ``record``
    实时快照，离开扫描决策族后 ``clear``。本类不做位姿或视觉新鲜度门控，
    调用方必须先确认采集时刻位姿及观测新鲜度。
    """

    def __init__(
        self,
        *,
        merge_radius_mm: float,
        max_age_s: float,
        pair_distance_mm: float,
    ) -> None:
        for name, value in (
            ("merge_radius_mm", merge_radius_mm),
            ("max_age_s", max_age_s),
            ("pair_distance_mm", pair_distance_mm),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if merge_radius_mm > pair_distance_mm:
            raise ValueError(
                "merge_radius_mm must not exceed pair_distance_mm, "
                "otherwise a re-observation can hijack a neighbour's entry "
                "when its own entry has expired "
                f"(merge={merge_radius_mm}, pair={pair_distance_mm})."
            )
        self._merge_radius_mm = float(merge_radius_mm)
        self._max_age_ns = round(float(max_age_s) * 1_000_000_000)
        self._pair_distance_mm = float(pair_distance_mm)
        self._entries: dict[int, ScanMemoryEntry] = {}
        self._pairs: dict[tuple[int, int], int] = {}
        self._next_memory_id = _MEMORY_ID_BASE

    @property
    def entries(self) -> tuple[ScanMemoryEntry, ...]:
        return tuple(self._entries.values())

    @property
    def memory_ids(self) -> frozenset[int]:
        return frozenset(self._entries)

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    def entry(self, memory_id: int) -> ScanMemoryEntry | None:
        return self._entries.get(memory_id)

    def is_fresh(self, memory_id: int, timestamp_ns: int) -> bool:
        entry = self._entries.get(memory_id)
        if entry is None:
            return False
        age_ns = timestamp_ns - entry.last_seen_timestamp_ns
        return 0 <= age_ns <= self._max_age_ns

    def has_pair_evidence(self, first_id: int, second_id: int) -> bool:
        return self._pair_key(first_id, second_id) in self._pairs

    def clear(self) -> None:
        self._entries.clear()
        self._pairs.clear()

    def record(
        self,
        targets: tuple[WorldTarget, ...],
        timestamp_ns: int,
    ) -> None:
        """把当前实时快照中的可靠目标并入记忆并刷新成团证据。

        只记录 CONFIRMED 且 ever_confirmed、地面/场地点齐全的目标；类别
        不限——非绿目标同样作为走廊障碍物与成团成员参与后续决策。同一
        条目在一个周期内只允许被一个观测认领，相邻的真实目标对不会被
        合并成单条目。
        """

        reliable = tuple(
            target
            for target in targets
            if target.track_status is TrackStatus.CONFIRMED
            and target.ever_confirmed
            and target.ground_point is not None
            and target.field_point is not None
        )
        claimed: set[int] = set()
        recorded_ids: list[int] = []
        for target in reliable:
            match_id = self._match_entry(target.field_point, claimed)
            if match_id is None:
                match_id = self._next_memory_id
                self._next_memory_id += 1
            # 新建条目同样计入本周期已认领：同一周期内相邻目标（例如贴合
            # 的成团对）不能抢占彼此的条目，否则近距对会塌缩成单条目。
            claimed.add(match_id)
            # 目标是静态的：用最新观测直接替换位置与属性，刷新真实观测时刻。
            self._entries[match_id] = ScanMemoryEntry(
                memory_id=match_id,
                target_class=target.target_class,
                class_probabilities=target.class_probabilities,
                confidence=target.confidence,
                hazard_state=target.hazard_state,
                field_point=target.field_point,
                last_seen_timestamp_ns=target.last_seen_timestamp_ns,
            )
            recorded_ids.append(match_id)
        for index, first_id in enumerate(recorded_ids):
            for second_id in recorded_ids[index + 1 :]:
                first = self._entries[first_id].field_point
                second = self._entries[second_id].field_point
                if _distance(first, second) <= self._pair_distance_mm:
                    self._pairs[
                        self._pair_key(first_id, second_id)
                    ] = timestamp_ns
        self._prune(timestamp_ns)

    def augmented_targets(
        self,
        live_targets: tuple[WorldTarget, ...],
        pose: FieldPose2D,
        timestamp_ns: int,
    ) -> tuple[WorldTarget, ...]:
        """返回可并入实时快照的记忆合成目标。

        距任一实时 CONFIRMED 目标场地点不超过合并半径的条目被抑制（实时
        观测已代表该目标）；``ground_point`` 按当前位姿重投影，``field_point``
        与 ``last_seen_timestamp_ns`` 保持记录时的真实值，不伪造新鲜度。
        超过老化上限的条目不再返回。
        """

        live_points = tuple(
            target.field_point
            for target in live_targets
            if target.field_point is not None
            and target.track_status is TrackStatus.CONFIRMED
        )
        synthesized: list[WorldTarget] = []
        for entry in self._entries.values():
            if (timestamp_ns - entry.last_seen_timestamp_ns) > self._max_age_ns:
                continue
            if any(
                _distance(entry.field_point, point) <= self._merge_radius_mm
                for point in live_points
            ):
                continue
            synthesized.append(
                WorldTarget(
                    track_id=entry.memory_id,
                    track_status=TrackStatus.CONFIRMED,
                    ever_confirmed=True,
                    target_class=entry.target_class,
                    class_probabilities=entry.class_probabilities,
                    confidence=entry.confidence,
                    hazard_state=entry.hazard_state,
                    ground_point=ground_from_field(pose, entry.field_point),
                    field_point=entry.field_point,
                    last_seen_timestamp_ns=entry.last_seen_timestamp_ns,
                )
            )
        return tuple(synthesized)

    def _match_entry(self, point: FieldPoint, claimed: set[int]) -> int | None:
        # 严格小于合并半径才认领：恰好在成团接触距离上的相邻条目不互相
        # 吸收，保证近距对在记忆里保持两个独立条目。
        best_id: int | None = None
        best_distance = self._merge_radius_mm
        for entry in self._entries.values():
            if entry.memory_id in claimed:
                continue
            distance = _distance(point, entry.field_point)
            if distance < best_distance:
                best_id = entry.memory_id
                best_distance = distance
        return best_id

    def _prune(self, timestamp_ns: int) -> None:
        stale_ids = {
            memory_id
            for memory_id, entry in self._entries.items()
            if (timestamp_ns - entry.last_seen_timestamp_ns) > self._max_age_ns
        }
        for memory_id in stale_ids:
            del self._entries[memory_id]
        for key, evidence_ns in list(self._pairs.items()):
            first_id, second_id = key
            if (
                first_id not in self._entries
                or second_id not in self._entries
                or (timestamp_ns - evidence_ns) > self._max_age_ns
            ):
                del self._pairs[key]

    @staticmethod
    def _pair_key(first_id: int, second_id: int) -> tuple[int, int]:
        if first_id > second_id:
            first_id, second_id = second_id, first_id
        return (first_id, second_id)
