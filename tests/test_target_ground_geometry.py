from __future__ import annotations

from dataclasses import replace
import math
from time import monotonic_ns

import cv2
import numpy as np
import pytest

from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import (
    GroundPoint,
    RobotPoint3D,
    UndistortedPixel,
)
from rescue_vision.perception import (
    BoxTargetGeometry,
    ClassProbabilities,
    ColorSegmentationStatus,
    GroundGeometryMethod,
    GroundGeometryQuality,
    ObservationQuality,
    RegularTetrahedronTargetGeometry,
    RoiColorSegmentation,
    StaleGroundGeometryError,
    TargetClass,
    TargetGroundGeometryConfig,
    TargetGroundGeometryEstimator,
    TargetObservation,
    UndistortedBoundingBox,
)


def projector(*, oblique: bool = False) -> GroundProjector:
    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    if oblique:
        camera_position = np.asarray((0.0, 0.0, 250.0))
        camera_z = np.asarray((500.0, 0.0, 0.0)) - camera_position
        camera_z /= np.linalg.norm(camera_z)
        camera_x = np.asarray((0.0, -1.0, 0.0))
        camera_y = np.cross(camera_z, camera_x)
        rotation = np.vstack((camera_x, camera_y, camera_z))
        translation = -rotation @ camera_position
        ground_to_image = camera_matrix @ np.column_stack(
            (rotation[:, 0], rotation[:, 1], translation)
        )
        image_to_ground = np.linalg.inv(ground_to_image)
        image_to_ground /= image_to_ground[2, 2]
    else:
        rotation = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        )
        translation = np.array([0.0, 0.0, 500.0])
        image_to_ground = np.array(
            [
                [0.0, -0.625, 150.0],
                [-0.625, 0.0, 200.0],
                [0.0, 0.0, 1.0],
            ]
        )
    return GroundProjector(
        image_to_ground,
        new_camera_matrix=camera_matrix,
        rotation_robot_to_camera=rotation,
        translation_robot_to_camera_mm=translation,
    )


def config(
    *,
    green_size_mm: float = 40.0,
    enabled: bool = True,
) -> TargetGroundGeometryConfig:
    return TargetGroundGeometryConfig(
        enabled=enabled,
        green_supply=BoxTargetGeometry(
            green_size_mm,
            green_size_mm,
            green_size_mm,
        ),
        black_core=RegularTetrahedronTargetGeometry(40.0),
        orange_injured=BoxTargetGeometry(80.0, 40.0, 40.0),
        blue_danger=BoxTargetGeometry(40.0, 40.0, 40.0),
        coarse_center_step_mm=4.0,
        coarse_yaw_step_deg=15.0,
        refine_center_step_mm=1.0,
        refine_center_radius_mm=5.0,
        refine_top_candidates=3,
        refine_yaw_step_deg=3.0,
        refine_yaw_radius_deg=15.0,
        search_radius_margin_mm=4.0,
        silhouette_weight=0.65,
        contour_weight=0.25,
        contact_weight=0.10,
        contour_distance_scale_px=4.0,
        contact_distance_scale_px=6.0,
        max_contact_residual_px=8.0,
        ambiguity_score_delta=0.02,
        max_center_uncertainty_mm=8.0,
        min_fit_score=0.55,
        min_silhouette_iou=0.45,
    )


def local_vertices(
    target_class: TargetClass,
    *,
    green_size_mm: float = 40.0,
) -> np.ndarray:
    if target_class is TargetClass.BLACK_CORE:
        edge = 40.0
        radius = edge / math.sqrt(3.0)
        return np.asarray(
            (
                (radius, 0.0, 0.0),
                (-radius / 2.0, edge / 2.0, 0.0),
                (-radius / 2.0, -edge / 2.0, 0.0),
                (0.0, 0.0, edge * math.sqrt(2.0 / 3.0)),
            )
        )
    if target_class is TargetClass.ORANGE_INJURED:
        length, width, height = 80.0, 40.0, 40.0
    else:
        length = width = height = green_size_mm
    base = np.asarray(
        (
            (length / 2.0, width / 2.0, 0.0),
            (-length / 2.0, width / 2.0, 0.0),
            (-length / 2.0, -width / 2.0, 0.0),
            (length / 2.0, -width / 2.0, 0.0),
        )
    )
    top = base.copy()
    top[:, 2] = height
    return np.vstack((base, top))


def observation(
    target_class: TargetClass,
    ground_projector: GroundProjector,
    *,
    center: GroundPoint = GroundPoint(80.0, 10.0),
    yaw_rad: float = 0.0,
    green_size_mm: float = 40.0,
    timestamp_ns: int = 1_000_000_000,
    with_k0: bool = True,
) -> TargetObservation:
    vertices = local_vertices(
        target_class,
        green_size_mm=green_size_mm,
    )
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    xy = vertices[:, :2] @ rotation.T
    xy[:, 0] += center.x
    xy[:, 1] += center.y
    robot_points = tuple(
        RobotPoint3D(float(x), float(y), float(z))
        for (x, y), z in zip(xy, vertices[:, 2], strict=True)
    )
    pixels = ground_projector.project_robot_points(robot_points)
    points = np.asarray([(point.u, point.v) for point in pixels])
    x_min = max(0, math.floor(float(np.min(points[:, 0]))) - 2)
    y_min = max(0, math.floor(float(np.min(points[:, 1]))) - 2)
    x_max = min(640, math.ceil(float(np.max(points[:, 0]))) + 3)
    y_max = min(480, math.ceil(float(np.max(points[:, 1]))) + 3)
    box = UndistortedBoundingBox(
        float(x_min),
        float(y_min),
        float(x_max),
        float(y_max),
    )
    local = points - np.asarray((x_min, y_min))
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.rint(cv2.convexHull(local.astype(np.float32))).astype(np.int32),
        255,
    )
    base_count = 3 if target_class is TargetClass.BLACK_CORE else 4
    k0_index = max(
        range(base_count),
        key=lambda index: pixels[index].v,
    )
    k0 = pixels[k0_index] if with_k0 else None
    anchor = (
        GroundPoint(
            robot_points[k0_index].x,
            robot_points[k0_index].y,
        )
        if with_k0
        else None
    )
    return TargetObservation(
        frame_sequence=3,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 10_000_000,
        image_size=(640, 480),
        model_target_class=target_class,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(
            target_class,
            1.0,
        ),
        detection_confidence=0.95,
        box=box,
        color_segmentation=RoiColorSegmentation(
            candidate_class=target_class,
            status=ColorSegmentationStatus.ACCEPTED,
            roi_box=box,
            mask=mask,
            color_fraction=float(cv2.countNonZero(mask)) / mask.size,
            dominance=1.0,
        ),
        k0=k0,
        k0_confidence=0.9 if with_k0 else 0.0,
        ground_point=anchor,
        quality=(
            frozenset()
            if with_k0
            else frozenset({ObservationQuality.K0_UNAVAILABLE})
        ),
    )


@pytest.mark.parametrize(
    "target_class",
    [
        TargetClass.GREEN_SUPPLY,
        TargetClass.BLACK_CORE,
        TargetClass.ORANGE_INJURED,
        TargetClass.BLUE_DANGER,
    ],
)
def test_estimator_recovers_four_configured_target_centers(
    target_class: TargetClass,
) -> None:
    ground_projector = projector()
    expected = GroundPoint(80.0, 10.0)
    source = observation(target_class, ground_projector, center=expected)
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=200.0,
    )

    result = estimator.estimate(
        [source],
        result_timestamp_ns=1_050_000_000,
    )[0]

    assert result.method is GroundGeometryMethod.MODEL_FIT
    assert result.center_ground is not None
    assert (result.center_ground.x, result.center_ground.y) == pytest.approx(
        (expected.x, expected.y),
        abs=1.5,
    )
    assert result.fit_score is not None and result.fit_score >= 0.55
    assert result.silhouette_iou is not None
    assert result.silhouette_iou >= 0.45
    assert result.footprint_ground


def test_configured_size_changes_recovered_footprint() -> None:
    ground_projector = projector()
    source = observation(
        TargetClass.GREEN_SUPPLY,
        ground_projector,
        green_size_mm=60.0,
    )
    estimator = TargetGroundGeometryEstimator(
        config(green_size_mm=60.0),
        ground_projector=ground_projector,
        max_observation_age_ms=200.0,
    )

    result = estimator.estimate(
        [source],
        result_timestamp_ns=1_050_000_000,
    )[0]

    assert result.center_ground is not None
    lengths = [
        math.hypot(
            result.footprint_ground[index].x
            - result.footprint_ground[(index + 1) % 4].x,
            result.footprint_ground[index].y
            - result.footprint_ground[(index + 1) % 4].y,
        )
        for index in range(4)
    ]
    assert lengths == pytest.approx([60.0] * 4)


@pytest.mark.parametrize(
    "target_class",
    [TargetClass.BLACK_CORE, TargetClass.ORANGE_INJURED],
)
def test_estimator_recovers_center_with_oblique_camera(
    target_class: TargetClass,
) -> None:
    ground_projector = projector(oblique=True)
    expected = GroundPoint(500.0, 25.0)
    source = observation(
        target_class,
        ground_projector,
        center=expected,
        yaw_rad=math.radians(23.0),
    )
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=200.0,
    )

    result = estimator.estimate(
        [source],
        result_timestamp_ns=1_050_000_000,
    )[0]

    assert result.center_ground is not None
    assert (result.center_ground.x, result.center_ground.y) == pytest.approx(
        (expected.x, expected.y),
        abs=2.0,
    )


def test_estimator_can_fit_without_k0_but_marks_the_degradation() -> None:
    ground_projector = projector()
    source = observation(
        TargetClass.BLACK_CORE,
        ground_projector,
        with_k0=False,
    )
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=200.0,
    )

    result = estimator.estimate(
        [source],
        result_timestamp_ns=1_050_000_000,
    )[0]

    assert result.center_ground is not None
    assert GroundGeometryQuality.K0_UNAVAILABLE in result.quality


def test_estimator_meets_realtime_budget_without_a_frozen_clock() -> None:
    ground_projector = projector()
    source = observation(TargetClass.ORANGE_INJURED, ground_projector)
    capture_timestamp_ns = monotonic_ns()
    source = replace(
        source,
        capture_timestamp_ns=capture_timestamp_ns,
        result_timestamp_ns=capture_timestamp_ns,
    )
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=150.0,
    )

    result = estimator.estimate_realtime([source])

    assert not result.stale_dropped
    assert len(result.estimates) == 1
    assert result.estimates[0].center_ground is not None


def test_estimator_rejects_stale_and_incomplete_geometry() -> None:
    ground_projector = projector()
    source = observation(TargetClass.GREEN_SUPPLY, ground_projector)
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=20.0,
    )
    with pytest.raises(StaleGroundGeometryError):
        estimator.estimate(
            [source],
            result_timestamp_ns=1_050_000_000,
        )
    realtime = estimator.estimate_realtime(
        [source],
        result_timestamp_ns=1_050_000_000,
    )
    assert realtime.stale_dropped
    assert realtime.estimates == ()

    planar_only = GroundProjector(np.eye(3))
    with pytest.raises(ValueError, match="full camera extrinsics"):
        TargetGroundGeometryEstimator(
            config(),
            ground_projector=planar_only,
            max_observation_age_ms=100.0,
        )
    with pytest.raises(ValueError, match="enabled config"):
        TargetGroundGeometryEstimator(
            config(enabled=False),
            ground_projector=ground_projector,
            max_observation_age_ms=100.0,
        )


def test_estimator_degrades_unknown_and_bad_masks_without_guessing() -> None:
    ground_projector = projector()
    source = observation(TargetClass.GREEN_SUPPLY, ground_projector)
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=200.0,
    )
    unknown_segmentation = RoiColorSegmentation(
        candidate_class=TargetClass.GREEN_SUPPLY,
        status=ColorSegmentationStatus.INSUFFICIENT,
        roi_box=source.color_segmentation.roi_box,
        mask=source.color_segmentation.mask,
        color_fraction=source.color_segmentation.color_fraction,
        dominance=1.0,
    )
    unknown = replace(
        source,
        target_class=TargetClass.UNKNOWN,
        class_probabilities=ClassProbabilities.from_top_class(
            TargetClass.UNKNOWN,
            1.0,
        ),
        color_segmentation=unknown_segmentation,
    )
    unknown_result = estimator.estimate(
        [unknown],
        result_timestamp_ns=1_050_000_000,
    )[0]
    assert unknown_result.center_ground is None
    assert GroundGeometryQuality.UNKNOWN_CLASS in unknown_result.quality

    bad_mask = np.zeros_like(source.color_segmentation.mask)
    bad_mask[2:5, 2:5] = 255
    bad_segmentation = RoiColorSegmentation(
        candidate_class=TargetClass.GREEN_SUPPLY,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=source.color_segmentation.roi_box,
        mask=bad_mask,
        color_fraction=9 / bad_mask.size,
        dominance=1.0,
    )
    bad = replace(source, color_segmentation=bad_segmentation)
    bad_result = estimator.estimate(
        [bad],
        result_timestamp_ns=1_050_000_000,
    )[0]
    assert bad_result.center_ground is None
    assert (
        GroundGeometryQuality.FIT_LOW_CONFIDENCE
        in bad_result.quality
    )


def test_estimator_requires_one_frame_and_accepts_empty_batch() -> None:
    ground_projector = projector()
    source = observation(TargetClass.GREEN_SUPPLY, ground_projector)
    estimator = TargetGroundGeometryEstimator(
        config(),
        ground_projector=ground_projector,
        max_observation_age_ms=200.0,
    )
    assert estimator.estimate([]) == ()
    with pytest.raises(ValueError, match="same frame"):
        estimator.estimate(
            [source, replace(source, frame_sequence=4)],
            result_timestamp_ns=1_050_000_000,
        )
