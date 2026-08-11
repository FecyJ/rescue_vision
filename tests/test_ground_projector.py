from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from rescue_vision.geometry.camera_model import CameraCalibration, CameraModelType
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.geometry.types import (
    BevPixel,
    GroundPoint,
    RobotPoint3D,
    UndistortedPixel,
)


def make_projector() -> GroundProjector:
    return GroundProjector(
        np.array([[2.0, 0.0, 10.0], [0.0, 3.0, -5.0], [0.0, 0.0, 1.0]]),
        BevConfig(
            x_min=-100.0,
            x_max=300.0,
            y_min=-200.0,
            y_max=200.0,
            mm_per_pixel=10.0,
        ),
    )


def test_empty_single_and_batch_pixel_ground_roundtrip() -> None:
    projector = make_projector()
    assert projector.pixels_to_ground([]) == []
    assert projector.ground_to_pixels([]) == []

    pixels = [UndistortedPixel(1.0, 2.0), UndistortedPixel(7.5, -3.0)]
    ground = projector.pixels_to_ground(pixels)
    assert [(point.x, point.y) for point in ground] == pytest.approx(
        [(12.0, 1.0), (25.0, -14.0)]
    )
    roundtrip = projector.ground_to_pixels(ground)
    assert np.allclose(
        [(point.u, point.v) for point in roundtrip],
        [(point.u, point.v) for point in pixels],
    )
    assert projector.pixel_to_ground(pixels[0]) == ground[0]


def test_bev_corners_orientation_and_roundtrip() -> None:
    projector = make_projector()
    config = projector.bev_config
    assert config is not None
    points = [
        GroundPoint(config.x_max, config.y_max),
        GroundPoint(config.x_max, config.y_min),
        GroundPoint(config.x_min, config.y_max),
        GroundPoint(config.x_min, config.y_min),
    ]
    bev = projector.ground_to_bev_pixels(points)
    assert [(point.u, point.v) for point in bev] == pytest.approx(
        [
            (0.0, 0.0),
            (config.width, 0.0),
            (0.0, config.height),
            (config.width, config.height),
        ]
    )
    ground = projector.bev_pixels_to_ground(bev)
    assert [(point.x, point.y) for point in ground] == pytest.approx(
        [(point.x, point.y) for point in points]
    )
    assert projector.bev_pixel_to_ground(BevPixel(0.0, 0.0)) == points[0]


def test_invalid_homography_and_bev_are_rejected() -> None:
    with pytest.raises(ValueError, match="invertible"):
        GroundProjector(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="divisible"):
        BevConfig(0, 101, -100, 100, 10)


def test_robot_3d_projection_and_horizontal_plane_intersection() -> None:
    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    translation = np.array([0.0, 0.0, 500.0])
    image_to_ground = np.array(
        [[0.0, -0.625, 150.0], [-0.625, 0.0, 200.0], [0.0, 0.0, 1.0]]
    )
    projector = GroundProjector(
        image_to_ground,
        new_camera_matrix=camera_matrix,
        rotation_robot_to_camera=rotation,
        translation_robot_to_camera_mm=translation,
    )

    point = RobotPoint3D(100.0, 25.0, 40.0)
    pixel = projector.project_robot_point(point)
    recovered = projector.pixel_to_horizontal_plane(pixel, z_mm=40.0)

    assert projector.supports_robot_projection
    assert projector.project_robot_points([]) == []
    assert (recovered.x, recovered.y, recovered.z) == pytest.approx(
        (point.x, point.y, point.z)
    )
    batch = [
        point,
        RobotPoint3D(80.0, -10.0, 0.0),
    ]
    projected = projector.project_robot_points(batch)
    rvec, _ = cv2.Rodrigues(rotation)
    expected, _ = cv2.projectPoints(
        np.asarray([(item.x, item.y, item.z) for item in batch]),
        rvec,
        translation,
        camera_matrix,
        np.zeros(5),
    )
    assert [(item.u, item.v) for item in projected] == pytest.approx(
        expected.reshape(-1, 2)
    )
    ground_pixel = projector.ground_to_pixel(GroundPoint(100.0, 25.0))
    physical_ground_pixel = projector.project_robot_point(
        RobotPoint3D(100.0, 25.0, 0.0)
    )
    assert (ground_pixel.u, ground_pixel.v) == pytest.approx(
        (physical_ground_pixel.u, physical_ground_pixel.v)
    )


def test_partial_or_invalid_robot_projection_is_rejected() -> None:
    with pytest.raises(ValueError, match="provided together"):
        GroundProjector(
            np.eye(3),
            new_camera_matrix=np.eye(3),
        )
    with pytest.raises(ValueError, match="proper rotation"):
        GroundProjector(
            np.eye(3),
            new_camera_matrix=np.eye(3),
            rotation_robot_to_camera=np.diag([1.0, 1.0, -1.0]),
            translation_robot_to_camera_mm=np.zeros(3),
        )
    projector = GroundProjector(
        np.eye(3),
        new_camera_matrix=np.eye(3),
        rotation_robot_to_camera=np.eye(3),
        translation_robot_to_camera_mm=np.zeros(3),
    )
    with pytest.raises(ValueError, match="front"):
        projector.project_robot_point(RobotPoint3D(0.0, 0.0, -1.0))
    horizontal_ray_projector = GroundProjector(
        np.eye(3),
        new_camera_matrix=np.eye(3),
        rotation_robot_to_camera=np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        translation_robot_to_camera_mm=np.asarray((0.0, 0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="parallel"):
        horizontal_ray_projector.pixel_to_horizontal_plane(
            UndistortedPixel(0.0, 0.0),
            z_mm=1.0,
        )


def test_ground_mapping_rejects_intrinsic_mismatch(tmp_path) -> None:
    calibration = CameraCalibration(
        model=CameraModelType.PINHOLE,
        image_size=(32, 24),
        K=np.eye(3),
        D=np.zeros(5),
        new_K=np.eye(3),
    )
    document = {
        "schema_version": 3,
        "quality": {"usable": True, "physically_valid": True},
        "image_size": [32, 24],
        "intrinsics": {
            "model_type": "pinhole",
            "fingerprint_sha256": calibration.fingerprint(),
        },
        "image_to_ground": np.eye(3).tolist(),
        "bev": {
            "x_min_mm": 0,
            "x_max_mm": 100,
            "y_min_mm": -50,
            "y_max_mm": 50,
            "mm_per_pixel": 10,
        },
    }
    path = tmp_path / "ground.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert GroundProjector.from_json(
        path, camera_calibration=calibration
    ).bev_config is not None

    document["extrinsics"] = {
        "rotation_robot_to_camera": np.eye(3).tolist(),
        "translation_robot_to_camera_mm": [0.0, 0.0, 1.0],
        "physically_valid": True,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    assert GroundProjector.from_json(
        path,
        camera_calibration=calibration,
    ).supports_robot_projection

    document["intrinsics"]["model_type"] = "fisheye"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        GroundProjector.from_json(path, camera_calibration=calibration)

    document["intrinsics"]["model_type"] = "pinhole"
    document["intrinsics"]["fingerprint_sha256"] = "wrong"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        GroundProjector.from_json(path, camera_calibration=calibration)

    document["intrinsics"]["fingerprint_sha256"] = calibration.fingerprint()
    document["quality"]["usable"] = False
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="quality.usable"):
        GroundProjector.from_json(path, camera_calibration=calibration)
