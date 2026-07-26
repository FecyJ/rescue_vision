from __future__ import annotations

import numpy as np
import pytest

from rescue_vision.calibration.calibrate_intrinsics import (
    MINIMUM_VIEWS,
    PATTERN_SIZE,
    create_object_template,
    make_folds,
    view_rmse,
)
from rescue_vision.calibration.capture_chessboard_images import (
    detect_chessboard,
    make_preview,
)


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


def test_capture_preview_and_blank_detection_are_hardware_free() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    found, corners, sharpness = detect_chessboard(image)
    assert not found
    assert corners is None
    assert sharpness == 0.0
    preview = make_preview(image, 0, 50, 1.0, "ready", 0.5)
    assert preview.shape == (12, 16, 3)
