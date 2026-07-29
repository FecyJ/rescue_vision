from __future__ import annotations

import numpy as np
import pytest

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import (
    MultiTargetTracker,
    TrackStatus,
    TrackingConfig,
)


def config(**overrides: object) -> TrackingConfig:
    values = {
        "confirmation_hits": 2,
        "max_association_ground_mm": 200.0,
        "min_association_iou": 0.1,
        "max_coast_ms": 500.0,
        "confidence_decay_per_second": 1.0,
        "min_confidence": 0.1,
    }
    values.update(overrides)
    return TrackingConfig(**values)  # type: ignore[arg-type]


def observation(
    timestamp_ns: int,
    *,
    sequence: int = 0,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
    confidence: float = 0.8,
    box: UndistortedBoundingBox | None = None,
    ground_point: GroundPoint | None = GroundPoint(500.0, 0.0),
) -> TargetObservation:
    k0 = (
        UndistortedPixel(20.0, 30.0)
        if ground_point is not None
        else None
    )
    return TargetObservation(
        frame_sequence=sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 1_000_000,
        image_size=(100, 80),
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(
            target_class,
            confidence,
        ),
        detection_confidence=confidence,
        box=box or UndistortedBoundingBox(10.0, 10.0, 30.0, 40.0),
        k0=k0,
        k0_confidence=0.9 if k0 is not None else 0.0,
        ground_point=ground_point,
        quality=frozenset(),
    )


def test_tracker_confirms_and_associates_by_ground_distance() -> None:
    tracker = MultiTargetTracker(config())
    first = tracker.update(1_000_000_000, [observation(1_000_000_000)])
    second = tracker.update(
        1_050_000_000,
        [
            observation(
                1_050_000_000,
                sequence=1,
                ground_point=GroundPoint(540.0, 10.0),
            )
        ],
    )

    assert first[0].status is TrackStatus.TENTATIVE
    assert second[0].track_id == first[0].track_id
    assert second[0].status is TrackStatus.CONFIRMED
    assert second[0].ground_point == GroundPoint(540.0, 10.0)
    assert second[0].hit_count == 2


def test_tracker_uses_iou_without_ground_mapping() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    first = tracker.update(
        1_000_000_000,
        [observation(1_000_000_000, ground_point=None)],
    )
    second = tracker.update(
        1_050_000_000,
        [
            observation(
                1_050_000_000,
                sequence=1,
                ground_point=None,
                box=UndistortedBoundingBox(12.0, 11.0, 32.0, 41.0),
            )
        ],
    )
    assert second[0].track_id == first[0].track_id


def test_incompatible_classes_and_distant_targets_get_new_ids() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    first_id = tracker.update(
        1_000_000_000,
        [observation(1_000_000_000)],
    )[0].track_id
    tracks = tracker.update(
        1_050_000_000,
        [
            observation(
                1_050_000_000,
                target_class=TargetClass.BLACK_CORE,
            ),
            observation(
                1_050_000_000,
                ground_point=GroundPoint(900.0, 0.0),
                box=UndistortedBoundingBox(50.0, 10.0, 70.0, 40.0),
            ),
        ],
    )
    assert len(tracks) == 3
    assert {track.track_id for track in tracks} > {first_id}


def test_short_occlusion_coasts_decays_and_expires() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    original = tracker.update(
        1_000_000_000,
        [observation(1_000_000_000)],
    )[0]
    coasting = tracker.update(1_200_000_000, [])[0]

    assert coasting.status is TrackStatus.COASTING
    assert coasting.ever_confirmed
    assert coasting.confidence < original.confidence
    assert coasting.age_since_seen_ms == pytest.approx(200.0)
    assert tracker.update(1_501_000_000, []) == ()


def test_unknown_can_associate_with_known_track_without_erasing_history() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    track_id = tracker.update(
        1_000_000_000,
        [observation(1_000_000_000)],
    )[0].track_id
    updated = tracker.update(
        1_050_000_000,
        [
            observation(
                1_050_000_000,
                target_class=TargetClass.UNKNOWN,
                confidence=1.0,
            )
        ],
    )[0]
    assert updated.track_id == track_id
    assert updated.class_probabilities.green_supply > 0.0


def test_tracker_rejects_mixed_or_backwards_timestamps() -> None:
    tracker = MultiTargetTracker(config())
    with pytest.raises(ValueError, match="does not match"):
        tracker.update(2, [observation(1)])
    tracker.update(2, [])
    with pytest.raises(ValueError, match="backwards"):
        tracker.update(1, [])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confirmation_hits", 0),
        ("max_association_ground_mm", 0.0),
        ("min_association_iou", 1.1),
        ("max_coast_ms", float("inf")),
        ("confidence_decay_per_second", 0.0),
        ("min_confidence", -0.1),
    ],
)
def test_tracking_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        config(**{field: value})
