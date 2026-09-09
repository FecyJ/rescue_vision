from __future__ import annotations

import math

import numpy as np
import pytest

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    RoiColorSegmentation,
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
    observation_box = box or UndistortedBoundingBox(10.0, 10.0, 30.0, 40.0)
    mask_shape = (
        math.ceil(observation_box.y_max) - math.floor(observation_box.y_min),
        math.ceil(observation_box.x_max) - math.floor(observation_box.x_min),
    )
    if target_class is TargetClass.UNKNOWN:
        candidate_class = TargetClass.UNKNOWN
        status = ColorSegmentationStatus.INSUFFICIENT
        mask = np.zeros(mask_shape, dtype=np.uint8)
        color_fraction = 0.0
        dominance = 0.0
    else:
        candidate_class = target_class
        status = ColorSegmentationStatus.ACCEPTED
        mask = np.full(mask_shape, 255, dtype=np.uint8)
        color_fraction = 1.0
        dominance = 1.0
    return TargetObservation(
        frame_sequence=sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 1_000_000,
        image_size=(100, 80),
        model_target_class=target_class,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(
            target_class,
            confidence,
        ),
        detection_confidence=confidence,
        box=observation_box,
        color_segmentation=RoiColorSegmentation(
            candidate_class=candidate_class,
            status=status,
            roi_box=UndistortedBoundingBox(
                float(math.floor(observation_box.x_min)),
                float(math.floor(observation_box.y_min)),
                float(math.ceil(observation_box.x_max)),
                float(math.ceil(observation_box.y_max)),
            ),
            mask=mask,
            color_fraction=color_fraction,
            dominance=dominance,
        ),
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


def test_same_frame_near_identical_detection_is_deduplicated() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    duplicate = observation(
        1_000_000_000,
        confidence=0.7,
        ground_point=GroundPoint(506.0, 4.0),
        box=UndistortedBoundingBox(11.0, 10.0, 31.0, 40.0),
    )
    tracks = tracker.update(
        1_000_000_000,
        [observation(1_000_000_000, confidence=0.9), duplicate],
    )
    assert len(tracks) == 1
    assert tracks[0].confidence == pytest.approx(0.9)


def test_nearby_distinct_blocks_are_not_deduplicated() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    tracks = tracker.update(
        1_000_000_000,
        [
            observation(1_000_000_000),
            observation(
                1_000_000_000,
                ground_point=GroundPoint(525.0, 0.0),
            ),
        ],
    )
    assert len(tracks) == 2


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


def test_soft_class_association_keeps_spatial_identity() -> None:
    tracker = MultiTargetTracker(config(confirmation_hits=1))
    tracker.enable_soft_class_association()
    first = tracker.update(
        1_000_000_000,
        [
            observation(
                1_000_000_000,
                ground_point=GroundPoint(500.0, -30.0),
                box=UndistortedBoundingBox(10.0, 10.0, 30.0, 40.0),
            ),
            observation(
                1_000_000_000,
                ground_point=GroundPoint(500.0, 30.0),
                box=UndistortedBoundingBox(70.0, 10.0, 90.0, 40.0),
            ),
        ],
    )
    ids_by_y = {round(item.ground_point.y): item.track_id for item in first}

    second = tracker.update(
        1_050_000_000,
        [
            observation(
                1_050_000_000,
                target_class=TargetClass.BLACK_CORE,
                ground_point=GroundPoint(501.0, 30.0),
                box=UndistortedBoundingBox(70.0, 10.0, 90.0, 40.0),
                sequence=1,
            ),
            observation(
                1_050_000_000,
                ground_point=GroundPoint(499.0, -30.0),
                box=UndistortedBoundingBox(10.0, 10.0, 30.0, 40.0),
                sequence=1,
            ),
        ],
    )
    assert {
        round(item.ground_point.y): item.track_id for item in second
    } == ids_by_y


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
