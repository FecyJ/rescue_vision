"""ScanTargetMemory 的纯逻辑单元测试。"""

from __future__ import annotations

import math

import pytest

def _field_from_robot(pose, point):
    return FieldPoint(
        pose.position.x + math.cos(pose.heading_rad) * point.x - math.sin(pose.heading_rad) * point.y,
        pose.position.y + math.sin(pose.heading_rad) * point.x + math.cos(pose.heading_rad) * point.y,
    )


from rescue_vision.app.scan_target_memory import (
    _MEMORY_ID_BASE,
    ScanTargetMemory,
    ground_from_field,
)
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D
from rescue_vision.perception import ClassProbabilities, TargetClass
from rescue_vision.tracking import TrackStatus
from rescue_vision.world import HazardState, WorldTarget

_NOW_NS = 10_000_000_000
_MISSING = object()


def make_memory(
    *,
    merge_radius_mm: float = 80.0,
    max_age_s: float = 45.0,
    pair_distance_mm: float = 100.0,
) -> ScanTargetMemory:
    return ScanTargetMemory(
        merge_radius_mm=merge_radius_mm,
        max_age_s=max_age_s,
        pair_distance_mm=pair_distance_mm,
    )


def world_target(
    track_id: int,
    field_point: FieldPoint,
    *,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
    hazard_state: HazardState = HazardState.CLEAR,
    track_status: TrackStatus = TrackStatus.CONFIRMED,
    ever_confirmed: bool = True,
    ground_point: object = _MISSING,
    last_seen_timestamp_ns: int = _NOW_NS,
) -> WorldTarget:
    return WorldTarget(
        track_id=track_id,
        track_status=track_status,
        ever_confirmed=ever_confirmed,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(target_class, 0.9),
        confidence=0.9,
        hazard_state=hazard_state,
        ground_point=(
            GroundPoint(500.0, 0.0)
            if ground_point is _MISSING
            else ground_point  # type: ignore[arg-type]
        ),
        field_point=field_point,
        last_seen_timestamp_ns=last_seen_timestamp_ns,
    )


def test_record_rejects_unreliable_targets() -> None:
    memory = make_memory()
    memory.record(
        (
            world_target(1, FieldPoint(0.0, 0.0), track_status=TrackStatus.TENTATIVE),
            world_target(
                2, FieldPoint(100.0, 0.0), ever_confirmed=False
            ),
            world_target(3, FieldPoint(200.0, 0.0), ground_point=None),
            world_target(4, None),  # type: ignore[arg-type]  # field_point=None
        ),
        _NOW_NS,
    )
    assert memory.entries == ()
    assert memory.pair_count == 0


def test_record_empty_input_is_noop() -> None:
    memory = make_memory()
    memory.record((), _NOW_NS)
    assert memory.entries == ()


def test_record_merges_within_radius_and_creates_beyond() -> None:
    memory = make_memory()
    memory.record(
        (world_target(1, FieldPoint(0.0, 0.0), last_seen_timestamp_ns=_NOW_NS),),
        _NOW_NS,
    )
    assert len(memory.entries) == 1
    first_id = memory.entries[0].memory_id
    # 同一块在合并半径内被重新观测：位置替换为最新值、刷新观测时刻。
    memory.record(
        (
            world_target(
                7,
                FieldPoint(30.0, 0.0),
                last_seen_timestamp_ns=_NOW_NS + 1_000_000,
            ),
        ),
        _NOW_NS + 1_000_000,
    )
    assert len(memory.entries) == 1
    updated = memory.entries[0]
    assert updated.memory_id == first_id
    assert updated.field_point == FieldPoint(30.0, 0.0)
    assert updated.last_seen_timestamp_ns == _NOW_NS + 1_000_000
    # 超出合并半径的观测新建条目，且记忆 id 不与 tracker 小整数冲突。
    memory.record((world_target(8, FieldPoint(500.0, 0.0)),), _NOW_NS + 2_000_000)
    assert len(memory.entries) == 2
    assert all(entry.memory_id >= _MEMORY_ID_BASE for entry in memory.entries)
    assert len(memory.memory_ids) == 2


def test_adjacent_pair_keeps_separate_entries_and_evidence() -> None:
    memory = make_memory(pair_distance_mm=100.0)
    memory.record(
        (
            world_target(1, FieldPoint(0.0, 0.0)),
            world_target(2, FieldPoint(60.0, 0.0)),
        ),
        _NOW_NS,
    )
    # 贴合成团对（60 mm < 100 mm）必须保持两个独立条目并记录成团证据，
    # 第二个目标不能抢占第一个目标刚创建的条目。
    assert len(memory.entries) == 2
    first_id, second_id = (entry.memory_id for entry in memory.entries)
    assert memory.has_pair_evidence(first_id, second_id)
    assert memory.has_pair_evidence(second_id, first_id)


def test_cross_pass_duplicate_never_gains_pair_evidence() -> None:
    memory = make_memory(merge_radius_mm=80.0, max_age_s=45.0)
    memory.record((world_target(1, FieldPoint(0.0, 0.0)),), _NOW_NS)
    # 一整圈后同一位姿漂移 120 mm 重新观测同一块：超出合并半径产生重复
    # 条目，但两趟从不同时在线，不允许产生成团证据。
    memory.record(
        (world_target(9, FieldPoint(120.0, 0.0)),),
        _NOW_NS + 20_000_000_000,
    )
    assert len(memory.entries) == 2
    first_id, second_id = (entry.memory_id for entry in memory.entries)
    assert not memory.has_pair_evidence(first_id, second_id)


def test_stale_entries_and_pairs_are_pruned() -> None:
    memory = make_memory(max_age_s=45.0)
    memory.record(
        (
            world_target(1, FieldPoint(0.0, 0.0)),
            world_target(2, FieldPoint(60.0, 0.0)),
        ),
        _NOW_NS,
    )
    first_id, second_id = (entry.memory_id for entry in memory.entries)
    assert memory.has_pair_evidence(first_id, second_id)
    # 46 s 后两块都未再被观测：条目与成团证据一并剪除。
    memory.record((), _NOW_NS + 46_000_000_000)
    assert memory.entries == ()
    assert memory.pair_count == 0
    assert not memory.has_pair_evidence(first_id, second_id)


def test_is_fresh_uses_max_age_window() -> None:
    memory = make_memory(max_age_s=45.0)
    memory.record((world_target(1, FieldPoint(0.0, 0.0)),), _NOW_NS)
    memory_id = memory.entries[0].memory_id
    assert memory.is_fresh(memory_id, _NOW_NS)
    assert memory.is_fresh(memory_id, _NOW_NS + 45_000_000_000)
    assert not memory.is_fresh(memory_id, _NOW_NS + 45_000_000_001)
    assert not memory.is_fresh(memory_id, _NOW_NS - 1)
    assert not memory.is_fresh(_MEMORY_ID_BASE + 999, _NOW_NS)


def test_augmented_targets_reproject_to_current_robot_frame() -> None:
    memory = make_memory()
    memory.record((world_target(1, FieldPoint(0.0, 0.0)),), _NOW_NS)
    entry = memory.entries[0]
    pose = FieldPose2D(position=FieldPoint(1000.0, -500.0), heading_rad=math.pi / 2)
    augmented = memory.augmented_targets((), pose, _NOW_NS)
    assert len(augmented) == 1
    target = augmented[0]
    assert target.track_id == entry.memory_id
    assert target.track_status is TrackStatus.CONFIRMED
    assert target.ever_confirmed
    assert target.field_point == entry.field_point
    assert target.last_seen_timestamp_ns == entry.last_seen_timestamp_ns
    # 场地点 (0, 0) 在该位姿下的机器人系坐标：R(-π/2)·(-1000, 500) = (500, 1000)。
    assert target.ground_point is not None
    assert target.ground_point.x == pytest.approx(500.0, abs=1e-6)
    assert target.ground_point.y == pytest.approx(1000.0, abs=1e-6)
    # 与 _field_from_robot 互为逆变换（往返对照）。
    restored = _field_from_robot(pose, target.ground_point)
    assert restored.x == pytest.approx(entry.field_point.x, abs=1e-6)
    assert restored.y == pytest.approx(entry.field_point.y, abs=1e-6)


def test_ground_from_field_round_trip_arbitrary_pose() -> None:
    pose = FieldPose2D(
        position=FieldPoint(-321.5, 987.25), heading_rad=2.234
    )
    point = FieldPoint(456.0, -789.0)
    ground = ground_from_field(pose, point)
    restored = _field_from_robot(pose, ground)
    assert math.isclose(restored.x, point.x, abs_tol=1e-9)
    assert math.isclose(restored.y, point.y, abs_tol=1e-9)


def test_augmented_targets_suppressed_near_confirmed_live_target() -> None:
    memory = make_memory()
    memory.record((world_target(1, FieldPoint(0.0, 0.0)),), _NOW_NS)
    pose = FieldPose2D(position=FieldPoint(0.0, 0.0), heading_rad=0.0)
    live_confirmed = (world_target(50, FieldPoint(50.0, 0.0)),)
    live_tentative = (
        world_target(51, FieldPoint(50.0, 0.0), track_status=TrackStatus.TENTATIVE),
    )
    # 实时 CONFIRMED 目标已代表该位置：记忆条目被抑制。
    assert memory.augmented_targets(live_confirmed, pose, _NOW_NS) == ()
    # 未确认的实时观测不能抹掉记忆条目（保守保留为障碍物）。
    assert len(memory.augmented_targets(live_tentative, pose, _NOW_NS)) == 1


def test_augmented_targets_drop_stale_entries() -> None:
    memory = make_memory(max_age_s=45.0)
    memory.record((world_target(1, FieldPoint(0.0, 0.0)),), _NOW_NS)
    pose = FieldPose2D(position=FieldPoint(0.0, 0.0), heading_rad=0.0)
    assert len(memory.augmented_targets((), pose, _NOW_NS + 44_000_000_000)) == 1
    assert memory.augmented_targets((), pose, _NOW_NS + 46_000_000_000) == ()


def test_clear_resets_memory() -> None:
    memory = make_memory()
    memory.record(
        (world_target(1, FieldPoint(0.0, 0.0)), world_target(2, FieldPoint(60.0, 0.0))),
        _NOW_NS,
    )
    assert memory.entries and memory.pair_count == 1
    memory.clear()
    assert memory.entries == ()
    assert memory.memory_ids == frozenset()
    assert memory.pair_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"merge_radius_mm": 0.0, "max_age_s": 45.0, "pair_distance_mm": 100.0},
        {"merge_radius_mm": 80.0, "max_age_s": -1.0, "pair_distance_mm": 100.0},
        {"merge_radius_mm": 80.0, "max_age_s": 45.0, "pair_distance_mm": 0.0},
        {"merge_radius_mm": float("nan"), "max_age_s": 45.0, "pair_distance_mm": 100.0},
        # merge 超过成团距离会劫持邻居条目，必须拒绝。
        {"merge_radius_mm": 120.0, "max_age_s": 45.0, "pair_distance_mm": 100.0},
    ],
)
def test_constructor_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_memory(**kwargs)  # type: ignore[arg-type]
