from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from rescue_vision.geometry.camera_model import (
    IMAGE_BORDER_FILL_VALUE,
    CameraCalibration,
    CameraModel,
    CameraModelType,
)
from rescue_vision.geometry.types import RawPixel


def calibration(model: CameraModelType) -> CameraCalibration:
    distortion = {
        CameraModelType.PINHOLE: np.array([0.08, -0.03, 0.002, -0.001, 0.01]),
        CameraModelType.PINHOLE_RATIONAL: np.array(
            [0.08, -0.03, 0.002, -0.001, 0.01, 0.005, -0.002, 0.001]
        ),
        CameraModelType.FISHEYE: np.array([0.04, -0.01, 0.002, -0.0005]),
    }[model]
    return CameraCalibration(
        model=model,
        image_size=(32, 24),
        K=np.array([[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]]),
        D=distortion,
        new_K=np.array(
            [[18.0, 0.0, 15.0], [0.0, 19.0, 11.0], [0.0, 0.0, 1.0]]
        ),
    )


@pytest.mark.parametrize("model", list(CameraModelType))
def test_all_models_undistort_points_and_image(model: CameraModelType) -> None:
    parameters = calibration(model)
    camera = CameraModel(parameters)
    raw = np.array([[[3.0, 4.0]], [[20.0, 10.0]]], dtype=np.float64)
    points = camera.undistort_pixels([RawPixel(3.0, 4.0), RawPixel(20.0, 10.0)])
    if model is CameraModelType.FISHEYE:
        expected = cv2.fisheye.undistortPoints(
            raw,
            parameters.K,
            parameters.D,
            R=np.eye(3),
            P=parameters.new_K,
        )
    else:
        expected = cv2.undistortPoints(
            raw,
            parameters.K,
            parameters.D,
            R=np.eye(3),
            P=parameters.new_K,
        )
    assert np.allclose(
        np.array([(point.u, point.v) for point in points]),
        expected.reshape(-1, 2),
    )
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    assert camera.undistort_image(image).shape == image.shape


@pytest.mark.parametrize("model", list(CameraModelType))
def test_undistort_empty_and_single_point(model: CameraModelType) -> None:
    camera = CameraModel(calibration(model))
    assert camera.undistort_pixels([]) == []
    single = camera.undistort_pixel(RawPixel(3.0, 4.0))
    batch = camera.undistort_pixels([RawPixel(3.0, 4.0)])
    assert single == batch[0]
    assert (single.u, single.v) != pytest.approx((3.0, 4.0))


def test_calibration_fingerprint_changes_with_model() -> None:
    assert calibration(CameraModelType.PINHOLE).fingerprint() != calibration(
        CameraModelType.PINHOLE_RATIONAL
    ).fingerprint()


def test_unusable_calibration_is_rejected(tmp_path) -> None:
    document = {
        "model_type": "pinhole",
        "image_size": [32, 24],
        "camera_matrix": calibration(CameraModelType.PINHOLE).K.tolist(),
        "distortion": [0, 0, 0, 0, 0],
        "new_camera_matrix": calibration(CameraModelType.PINHOLE).new_K.tolist(),
        "quality": {"usable": False},
    }
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="marked unusable"):
        CameraCalibration.from_json(path)
    loaded = CameraCalibration.from_json(path, allow_unusable=True)
    assert loaded.model is CameraModelType.PINHOLE


def test_image_size_mismatch_is_rejected() -> None:
    camera = CameraModel(calibration(CameraModelType.PINHOLE))
    with pytest.raises(ValueError, match="does not match calibration"):
        camera.undistort_image(np.zeros((10, 10, 3), dtype=np.uint8))


def test_undistort_fills_invalid_pixels_with_letterbox_value() -> None:
    parameters = calibration(CameraModelType.PINHOLE)
    shifted = CameraCalibration(
        model=parameters.model,
        image_size=parameters.image_size,
        K=parameters.K,
        D=parameters.D,
        new_K=np.array(
            [[20.0, 0.0, -20.0], [0.0, 20.0, -20.0], [0.0, 0.0, 1.0]]
        ),
    )
    camera = CameraModel(shifted)
    result = camera.undistort_image(
        np.zeros((24, 32, 3), dtype=np.uint8)
    )
    invalid = camera.valid_mask == 0

    assert np.any(invalid)
    assert np.all(result[invalid] == IMAGE_BORDER_FILL_VALUE)


def test_invalid_distortion_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        CameraCalibration(
            model=CameraModelType.FISHEYE,
            image_size=(32, 24),
            K=np.eye(3),
            D=np.zeros(5),
            new_K=np.eye(3),
        )
