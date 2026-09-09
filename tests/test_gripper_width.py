from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rescue_vision.app.gripper_width import (
    _fit_preview_image,
)
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    GripperWidthEstimatorConfig,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
    average_gripper_width_measurements,
    estimate_gripper_width,
)


def _observation(
    *,
    center_x_mm: float = 230.0,
    center_y_mm: float = 0.0,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
    status: ColorSegmentationStatus = ColorSegmentationStatus.ACCEPTED,
    mask: np.ndarray | None = None,
) -> TargetObservation:
    box = UndistortedBoundingBox(10.0, 20.0, 14.0, 24.0)
    owned_mask = np.asarray(
        mask
        if mask is not None
        else (
            np.array(
                [
                    [0, 255, 0, 0],
                    [255, 255, 255, 0],
                    [0, 255, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
            if status is ColorSegmentationStatus.ACCEPTED
            else np.zeros((4, 4), dtype=np.uint8)
        )
    )
    candidate = (
        target_class
        if status is ColorSegmentationStatus.ACCEPTED
        else TargetClass.UNKNOWN
    )
    segmentation = RoiColorSegmentation(
        candidate_class=candidate,
        status=status,
        roi_box=box,
        mask=owned_mask,
        color_fraction=0.5 if status is ColorSegmentationStatus.ACCEPTED else 0.0,
        dominance=1.0 if status is ColorSegmentationStatus.ACCEPTED else 0.0,
    )
    return TargetObservation(
        frame_sequence=9,
        capture_timestamp_ns=1_000,
        result_timestamp_ns=1_001,
        image_size=(30, 40),
        model_target_class=target_class,
        target_class=candidate,
        class_probabilities=ClassProbabilities.from_top_class(candidate, 1.0),
        detection_confidence=0.9,
        box=box,
        color_segmentation=segmentation,
        k0=UndistortedPixel(12.0, 22.0),
        k0_confidence=0.9,
        ground_point=GroundPoint(center_x_mm, center_y_mm),
        quality=frozenset(),
    )


def test_estimator_projects_mask_and_adds_clearance() -> None:
    measurement = estimate_gripper_width(
        _observation(),
        GroundProjector(np.eye(3)),
    )

    assert measurement is not None
    # Pixel centers have v values 20.5, 21.5 and 22.5 for the mask above.
    assert measurement.left_y_mm == pytest.approx(22.5)
    assert measurement.right_y_mm == pytest.approx(20.5)
    assert measurement.center_x_mm == pytest.approx(230.0)
    assert measurement.front_x_mm == pytest.approx(237.5)
    assert measurement.center_to_front_mm == pytest.approx(7.5)
    assert measurement.forward_distance_mm == pytest.approx(237.5)
    assert measurement.width_mm == pytest.approx(2.0)
    assert measurement.opening_width_mm == pytest.approx(6.0)
    assert measurement.left_edge_pixel == UndistortedPixel(11.5, 22.5)
    assert measurement.right_edge_pixel == UndistortedPixel(11.5, 20.5)


def test_estimator_keeps_extrema_from_disconnected_mask_components() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
    mask[3, 3] = 255
    measurement = estimate_gripper_width(
        _observation(mask=mask),
        GroundProjector(np.eye(3)),
    )

    assert measurement is not None
    assert measurement.left_y_mm == pytest.approx(23.5)
    assert measurement.right_y_mm == pytest.approx(20.5)
    assert measurement.front_x_mm == pytest.approx(238.5)


@pytest.mark.parametrize("center_y_mm", [-5.0, 5.0, 8.0])
def test_estimator_uses_strict_center_gate(center_y_mm: float) -> None:
    assert (
        estimate_gripper_width(
            _observation(center_y_mm=center_y_mm),
            GroundProjector(np.eye(3)),
        )
        is None
    )


def test_estimator_rejects_unaccepted_color_segmentation() -> None:
    assert (
        estimate_gripper_width(
            _observation(status=ColorSegmentationStatus.INSUFFICIENT),
            GroundProjector(np.eye(3)),
        )
        is None
    )


def test_estimator_config_validates_clearance_and_pixel_count() -> None:
    with pytest.raises(ValueError, match="clearance_mm"):
        GripperWidthEstimatorConfig(clearance_mm=-1.0)
    with pytest.raises(ValueError, match="min_mask_pixels"):
        GripperWidthEstimatorConfig(min_mask_pixels=0)


def test_estimator_returns_none_without_ground_center() -> None:
    observation = _observation()
    # TargetObservation permits a missing ground point when K0 is present.
    observation = TargetObservation(
        frame_sequence=observation.frame_sequence,
        capture_timestamp_ns=observation.capture_timestamp_ns,
        result_timestamp_ns=observation.result_timestamp_ns,
        image_size=observation.image_size,
        model_target_class=observation.model_target_class,
        target_class=observation.target_class,
        class_probabilities=observation.class_probabilities,
        detection_confidence=observation.detection_confidence,
        box=observation.box,
        color_segmentation=observation.color_segmentation,
        k0=observation.k0,
        k0_confidence=observation.k0_confidence,
        ground_point=None,
        quality=observation.quality,
    )
    assert estimate_gripper_width(observation, GroundProjector(np.eye(3))) is None


def test_average_uses_only_valid_measurements() -> None:
    first = estimate_gripper_width(
        _observation(),
        GroundProjector(np.eye(3)),
    )
    assert first is not None
    second = replace(
        first,
        frame_sequence=10,
        center_y_mm=2.0,
        left_y_mm=24.5,
        right_y_mm=19.5,
        width_mm=5.0,
        opening_width_mm=9.0,
        left_edge_pixel=UndistortedPixel(12.5, 23.5),
        right_edge_pixel=UndistortedPixel(10.5, 20.5),
    )

    averaged = average_gripper_width_measurements(
        [first, second],
        clearance_mm=4.0,
    )

    assert averaged.frame_sequence == 10
    assert averaged.center_x_mm == pytest.approx(230.0)
    assert averaged.front_x_mm == pytest.approx(237.5)
    assert averaged.center_to_front_mm == pytest.approx(7.5)
    assert averaged.forward_distance_mm == pytest.approx(237.5)
    assert averaged.center_y_mm == pytest.approx(1.0)
    assert averaged.left_y_mm == pytest.approx(23.5)
    assert averaged.right_y_mm == pytest.approx(20.0)
    assert averaged.width_mm == pytest.approx(3.5)
    assert averaged.opening_width_mm == pytest.approx(7.5)


def test_preview_is_scaled_to_a_normal_window_size() -> None:
    image = np.zeros((1296, 2304, 3), dtype=np.uint8)

    preview = _fit_preview_image(image)

    assert preview.shape == (720, 1280, 3)
