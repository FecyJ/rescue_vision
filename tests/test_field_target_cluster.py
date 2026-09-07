"""FieldPoint 带身份连通聚类的纯逻辑测试。"""

from __future__ import annotations

import pytest

from rescue_vision.app.field_target_cluster import (
    FieldClusterMember,
    connected_field_clusters,
)
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.perception import TargetClass
from rescue_vision.world import HazardState


def member(
    track_id: int,
    x: float,
    y: float,
    *,
    target_class: TargetClass = TargetClass.BLACK_CORE,
    from_memory: bool = False,
) -> FieldClusterMember:
    return FieldClusterMember(
        track_id,
        target_class,
        HazardState.CLEAR,
        FieldPoint(x, y),
        from_memory,
    )


def test_connected_field_clusters_preserve_identity_and_chain_neighbors() -> None:
    clusters = connected_field_clusters(
        (
            member(3, 180.0, 0.0),
            member(1, 0.0, 0.0, target_class=TargetClass.GREEN_SUPPLY),
            member(2, 90.0, 0.0),
            member(4, 500.0, 0.0),
        ),
        neighbor_distance_mm=100.0,
    )

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.member_ids == (1, 2, 3)
    assert tuple(item.track_id for item in cluster.green_members) == (1,)
    assert cluster.centroid_field == FieldPoint(90.0, 0.0)


def test_edge_policy_can_reject_close_memory_duplicates() -> None:
    clusters = connected_field_clusters(
        (member(1, 0.0, 0.0, from_memory=True), member(2, 50.0, 0.0, from_memory=True)),
        neighbor_distance_mm=100.0,
        edge_allowed=lambda _first, _second: False,
    )

    assert clusters == ()


def test_connected_field_clusters_validate_threshold_and_unique_ids() -> None:
    with pytest.raises(ValueError, match="neighbor_distance_mm"):
        connected_field_clusters((), neighbor_distance_mm=0.0)
    with pytest.raises(ValueError, match="unique track IDs"):
        connected_field_clusters(
            (member(1, 0.0, 0.0), member(1, 10.0, 0.0)),
            neighbor_distance_mm=100.0,
        )
