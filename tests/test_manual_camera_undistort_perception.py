from __future__ import annotations

from manual_tests.camera_undistort_perception import (
    _format_field_feature_coordinates,
)
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    CenterCrossConfirmation,
    CenterCrossObservation,
    FieldFeatureDetectionResult,
    FieldPoseKeypoint,
    SafeZoneColor,
    SafeZoneObservation,
    UndistortedBoundingBox,
)


def keypoint(u: float, v: float, x: float, y: float) -> FieldPoseKeypoint:
    return FieldPoseKeypoint(
        UndistortedPixel(u, v),
        GroundPoint(x, y),
        0.9,
    )


def test_field_feature_coordinates_include_center_and_safe_zone_keypoints() -> None:
    center = CenterCrossObservation(
        box=UndistortedBoundingBox(0.0, 0.0, 100.0, 100.0),
        intersection=keypoint(50.0, 60.0, 300.0, -40.0),
        axes=(),
        confidence=0.9,
        quality=frozenset(),
        confirmation=CenterCrossConfirmation.CANDIDATE,
    )
    safe_zone = SafeZoneObservation(
        box=UndistortedBoundingBox(100.0, 0.0, 200.0, 100.0),
        ground_anchor=keypoint(120.0, 20.0, 400.0, 10.0),
        image_left_landmark=keypoint(110.0, 80.0, 350.0, 120.0),
        image_right_landmark=keypoint(190.0, 80.0, 350.0, -120.0),
        physical_color=SafeZoneColor.RED,
        confidence=0.9,
        quality=frozenset(),
    )
    result = FieldFeatureDetectionResult(
        frame_sequence=1,
        capture_timestamp_ns=100,
        result_timestamp_ns=200,
        image_size=(640, 480),
        safe_zones=(safe_zone,),
        center_cross=center,
    )

    text = _format_field_feature_coordinates(result)

    assert "center_cross_k0[pixel=(u=50.0,v=60.0) ground=(x=300.0,y=-40.0)mm]" in text
    assert "red:K0[pixel=(u=120.0,v=20.0) ground=(x=400.0,y=10.0)mm]" in text
    assert "K1[pixel=(u=110.0,v=80.0) ground=(x=350.0,y=120.0)mm]" in text
    assert "K2[pixel=(u=190.0,v=80.0) ground=(x=350.0,y=-120.0)mm]" in text


def test_field_feature_coordinates_show_missing_features_as_dash() -> None:
    assert _format_field_feature_coordinates(None) == (
        "center_cross_k0=- safe_zone_keypoints=-"
    )
