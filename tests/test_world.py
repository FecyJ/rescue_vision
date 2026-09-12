from __future__ import annotations

from dataclasses import replace

import pytest

from rescue_vision.geometry.types import (
    FieldPoint,
    GroundPoint,
    UndistortedPixel,
)
from rescue_vision.perception import (
    ClassProbabilities,
    TargetClass,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import TrackStatus, TrackedTarget
from rescue_vision.world import (
    HazardState,
    OpponentOccupancy,
    RegionKind,
    StaticRegion,
    WorldModel,
    WorldModelConfig,
    WorldUncertainty,
)


def config(**overrides: object) -> WorldModelConfig:
    values = {
        "max_visual_age_ms": 200.0,
        "opponent_max_age_ms": 500.0,
        "danger_confirm_threshold": 0.6,
        "danger_suspect_threshold": 0.2,
    }
    values.update(overrides)
    return WorldModelConfig(**values)  # type: ignore[arg-type]


def square_region(
    region_id: str = "opponent",
    kind: RegionKind = RegionKind.OPPONENT_SAFE,
) -> StaticRegion:
    return StaticRegion(
        region_id,
        kind,
        (
            FieldPoint(0.0, 0.0),
            FieldPoint(100.0, 0.0),
            FieldPoint(100.0, 100.0),
            FieldPoint(0.0, 100.0),
        ),
    )


def track(
    *,
    track_id: int = 1,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
    confidence: float = 0.8,
    status: TrackStatus = TrackStatus.CONFIRMED,
    ground_point: GroundPoint | None = GroundPoint(100.0, 0.0),
) -> TrackedTarget:
    return TrackedTarget(
        track_id=track_id,
        status=status,
        ever_confirmed=status is TrackStatus.CONFIRMED,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(
            target_class,
            confidence,
        ),
        confidence=confidence,
        box=UndistortedBoundingBox(0.0, 0.0, 10.0, 10.0),
        k0=(
            UndistortedPixel(5.0, 9.0)
            if ground_point is not None
            else None
        ),
        k0_confidence=0.9 if ground_point is not None else 0.0,
        ground_point=ground_point,
        quality=frozenset(),
        first_seen_timestamp_ns=0,
        last_seen_timestamp_ns=0,
        state_timestamp_ns=0,
        frame_sequence=0,
        hit_count=2,
        missed_count=0,
    )


def opponent(timestamp_ns: int) -> OpponentOccupancy:
    return OpponentOccupancy(
        "opponent-1",
        (
            FieldPoint(200.0, 200.0),
            FieldPoint(300.0, 200.0),
            FieldPoint(300.0, 300.0),
            FieldPoint(200.0, 300.0),
        ),
        0.9,
        timestamp_ns,
    )


def test_static_region_contains_inside_edge_and_rejects_degenerate() -> None:
    region = square_region()
    assert region.contains(FieldPoint(50.0, 50.0))
    assert region.contains(FieldPoint(0.0, 20.0))
    assert not region.contains(FieldPoint(101.0, 50.0))
    with pytest.raises(ValueError, match="non-zero area"):
        StaticRegion(
            "line",
            RegionKind.FIELD,
            (
                FieldPoint(0.0, 0.0),
                FieldPoint(1.0, 1.0),
                FieldPoint(2.0, 2.0),
            ),
        )


def test_world_marks_confirmed_and_suspected_hazards() -> None:
    model = WorldModel(config(), [])
    snapshot = model.update(
        timestamp_ns=100,
        visual_timestamp_ns=100,
        tracks=[
            track(
                track_id=1,
                target_class=TargetClass.BLUE_DANGER,
                confidence=0.8,
            ),
            replace(track(track_id=2, target_class=TargetClass.BLUE_DANGER),
                    status=TrackStatus.TENTATIVE, ever_confirmed=False),
            track(track_id=3),
        ],
    )
    assert snapshot.target(1).hazard_state is HazardState.CONFIRMED  # type: ignore[union-attr]
    assert snapshot.target(2).hazard_state is HazardState.SUSPECTED  # type: ignore[union-attr]
    assert snapshot.target(3).hazard_state is HazardState.CLEAR  # type: ignore[union-attr]
    assert {
        target.track_id
        for target in snapshot.hazards_within_ground_distance(150.0)
    } == {1, 2}


def test_confirmed_danger_history_is_not_weakened_while_coasting() -> None:
    model = WorldModel(config(), [])
    danger = replace(
        track(
            target_class=TargetClass.BLUE_DANGER,
            confidence=0.8,
        ),
        status=TrackStatus.COASTING,
        ever_confirmed=True,
    )
    snapshot = model.update(
        timestamp_ns=1,
        visual_timestamp_ns=1,
        tracks=[danger],
    )
    assert snapshot.target(1).hazard_state is HazardState.CONFIRMED  # type: ignore[union-attr]


def test_world_exposes_visual_and_coordinate_uncertainty() -> None:
    model = WorldModel(config(), [])
    snapshot = model.update(
        timestamp_ns=300_000_001,
        visual_timestamp_ns=0,
        tracks=[
            replace(
                track(target_class=TargetClass.BLUE_DANGER),
                status=TrackStatus.TENTATIVE,
                ever_confirmed=False,
                ground_point=None,
            )
        ],
    )
    assert snapshot.uncertainties >= {
        WorldUncertainty.STALE_VISION,
        WorldUncertainty.MISSING_ROBOT_FIELD_POSITION,
        WorldUncertainty.TARGET_WITHOUT_GROUND_POINT,
        WorldUncertainty.UNCONFIRMED_TARGET,
    }


def test_world_maps_robot_and_targets_to_field_regions() -> None:
    region = square_region()
    model = WorldModel(config(), [region])
    snapshot = model.update(
        timestamp_ns=1,
        visual_timestamp_ns=1,
        tracks=[track()],
        robot_field_point=FieldPoint(50.0, 50.0),
        target_field_points={1: FieldPoint(70.0, 70.0)},
    )
    assert snapshot.robot_in_region(RegionKind.OPPONENT_SAFE)
    assert snapshot.target(1).field_point == FieldPoint(70.0, 70.0)  # type: ignore[union-attr]
    assert snapshot.target_region_kinds(1) == frozenset(
        {RegionKind.OPPONENT_SAFE}
    )
    assert snapshot.target_in_opponent_occupancy(1) is False
    assert WorldUncertainty.MISSING_ROBOT_FIELD_POSITION not in snapshot.uncertainties


def test_target_area_queries_preserve_unknown_and_opponent_occupancy() -> None:
    model = WorldModel(config(), [])
    snapshot = model.update(
        timestamp_ns=1,
        visual_timestamp_ns=1,
        tracks=[track(track_id=1), track(track_id=2)],
        target_field_points={1: FieldPoint(250.0, 250.0)},
        opponent_occupancies=[opponent(1)],
    )
    assert snapshot.target_region_kinds(1) == frozenset()
    assert snapshot.target_in_opponent_occupancy(1) is True
    assert snapshot.target_region_kinds(2) is None
    assert snapshot.target_in_opponent_occupancy(2) is None
    with pytest.raises(KeyError, match="Unknown track_id"):
        snapshot.target_region_kinds(99)


def test_opponent_occupancy_is_retained_then_expires() -> None:
    model = WorldModel(config(opponent_max_age_ms=100.0), [])
    first = model.update(
        timestamp_ns=0,
        visual_timestamp_ns=0,
        tracks=[],
        opponent_occupancies=[opponent(0)],
    )
    retained = model.update(
        timestamp_ns=50_000_000,
        visual_timestamp_ns=50_000_000,
        tracks=[],
    )
    expired = model.update(
        timestamp_ns=100_000_001,
        visual_timestamp_ns=100_000_001,
        tracks=[],
    )
    assert len(first.opponent_occupancies) == 1
    assert len(retained.opponent_occupancies) == 1
    assert expired.opponent_occupancies == ()
    assert WorldUncertainty.STALE_OPPONENT in expired.uncertainties


def test_world_rejects_bad_time_ids_and_regions() -> None:
    with pytest.raises(ValueError, match="unique"):
        WorldModel(config(), [square_region("same"), square_region("same")])
    model = WorldModel(config(), [])
    with pytest.raises(ValueError, match="unknown track IDs"):
        model.update(
            timestamp_ns=1,
            visual_timestamp_ns=1,
            tracks=[],
            target_field_points={1: FieldPoint(0.0, 0.0)},
        )
    model.update(timestamp_ns=2, visual_timestamp_ns=2, tracks=[])
    with pytest.raises(ValueError, match="backwards"):
        model.update(timestamp_ns=1, visual_timestamp_ns=1, tracks=[])


def test_world_rejects_regressing_or_duplicate_external_observations() -> None:
    model = WorldModel(config(), [])
    model.update(
        timestamp_ns=10,
        visual_timestamp_ns=10,
        tracks=[],
        opponent_occupancies=[opponent(10)],
    )
    with pytest.raises(ValueError, match="visual_timestamp_ns moved backwards"):
        model.update(
            timestamp_ns=11,
            visual_timestamp_ns=9,
            tracks=[],
        )
    with pytest.raises(ValueError, match="timestamp moved backwards"):
        model.update(
            timestamp_ns=12,
            visual_timestamp_ns=10,
            tracks=[],
            opponent_occupancies=[opponent(9)],
        )

    duplicate = opponent(13)
    with pytest.raises(ValueError, match="unique opponent IDs"):
        model.update(
            timestamp_ns=13,
            visual_timestamp_ns=13,
            tracks=[],
            opponent_occupancies=[duplicate, duplicate],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_visual_age_ms", 0.0),
        ("opponent_max_age_ms", float("inf")),
        ("danger_confirm_threshold", 1.1),
        ("danger_suspect_threshold", -0.1),
    ],
)
def test_world_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        config(**{field: value})


def test_world_config_rejects_inverted_danger_thresholds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        config(
            danger_confirm_threshold=0.3,
            danger_suspect_threshold=0.4,
        )
