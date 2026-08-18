from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from rescue_vision.calibration.calibrate_intrinsics import (
    CalibrationFit,
    MINIMUM_VIEWS,
    PATTERN_SIZE,
    create_object_template,
    make_folds,
    make_runtime_calibration_document,
    view_rmse,
)
from rescue_vision.calibration.capture_chessboard_images import (
    camera_binding_metadata,
    detect_chessboard,
    make_preview,
)
from rescue_vision.geometry.camera_model import CameraCalibration


def test_intrinsic_object_template_uses_mm_and_expected_corner_order() -> None:
    template = create_object_template(15.0)
    assert template.shape == (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3)
    assert template[0] == pytest.approx([0.0, 0.0, 0.0])
    assert template[1] == pytest.approx([15.0, 0.0, 0.0])
    assert template[PATTERN_SIZE[0]] == pytest.approx([0.0, 15.0, 0.0])


def test_intrinsic_folds_are_complete_deterministic_and_validate_size() -> None:
    first = make_folds(MINIMUM_VIEWS, 5, 123)
    second = make_folds(MINIMUM_VIEWS, 5, 123)
    assert [fold.tolist() for fold in first] == [fold.tolist() for fold in second]
    assert sorted(np.concatenate(first).tolist()) == list(range(MINIMUM_VIEWS))
    with pytest.raises(ValueError, match="valid views"):
        make_folds(MINIMUM_VIEWS - 1, 5, 123)


def test_view_rmse_uses_euclidean_pixel_error() -> None:
    observed = np.array([[0.0, 0.0], [1.0, 1.0]])
    projected = np.array([[3.0, 4.0], [1.0, 1.0]])
    assert view_rmse(observed, projected) == pytest.approx(np.sqrt(12.5))


def test_runtime_calibration_document_uses_minimal_current_schema() -> None:
    fit = CalibrationFit(
        model_type="pinhole",
        rms_px=0.5,
        K=np.eye(3),
        D=np.zeros(5),
        rvecs=(),
        tvecs=(),
        solver_variant="test",
    )
    quality = {
        "usable": True,
        "grade": "good",
        "selection_score_px": 0.5,
    }

    document = make_runtime_calibration_document(
        "intrinsics_test",
        fit,
        np.eye(3),
        quality,
        1.25,
        {
            "camera_model": "imx708_wide",
            "sensor_pixel_array_size": [4608, 2592],
            "scaler_crop": [0, 0, 4608, 2592],
        },
    )

    assert set(document) == {
        "calibration_id",
        "model_type",
        "image_size",
        "camera_matrix",
        "distortion",
        "new_camera_matrix",
        "lens_position",
        "camera_model",
        "sensor_pixel_array_size",
        "scaler_crop",
        "quality",
    }
    assert document["calibration_id"] == "intrinsics_test"
    assert document["quality"] == quality
    assert document["lens_position"] == pytest.approx(1.25)
    assert document["camera_model"] == "imx708_wide"
    assert (
        CameraCalibration.from_dict(document).calibration_id
        == "intrinsics_test"
    )


def test_camera_binding_metadata_normalizes_libcamera_values() -> None:
    camera = SimpleNamespace(
        camera_properties={
            "Model": "imx708_wide",
            "PixelArraySize": SimpleNamespace(width=4608, height=2592),
        },
        capture_metadata=lambda: {
            "ScalerCrop": SimpleNamespace(x=0, y=0, width=4608, height=2592)
        },
    )

    assert camera_binding_metadata(camera) == {
        "camera_model": "imx708_wide",
        "sensor_pixel_array_size": [4608, 2592],
        "scaler_crop": [0, 0, 4608, 2592],
    }


def test_capture_preview_and_blank_detection_are_hardware_free() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    found, corners, sharpness = detect_chessboard(image)
    assert not found
    assert corners is None
    assert sharpness == 0.0
    preview = make_preview(image, 0, 50, 1.0, "ready", 0.5)
    assert preview.shape == (12, 16, 3)


def test_small_board_detection_retries_on_an_enlarged_image() -> None:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    square_px = 3
    origin_u, origin_v = 40, 30
    for row in range(9):
        for column in range(12):
            if (row + column) % 2 == 0:
                image[
                    origin_v + row * square_px : origin_v + (row + 1) * square_px,
                    origin_u + column * square_px : origin_u + (column + 1) * square_px,
                ] = 0

    found, corners, _sharpness = detect_chessboard(
        image,
        max_detection_scale=2.0,
    )

    assert found
    assert corners is not None
    assert corners.shape == (88, 1, 2)
