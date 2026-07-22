from __future__ import annotations

import json

import numpy as np
import pytest

from rescue_vision.calibration.calibrate_extrinsics_ground import load_intrinsics
from rescue_vision.geometry.camera_model import CameraModelType


@pytest.mark.parametrize(
    ("model", "distortion"),
    [
        ("pinhole", [0, 0, 0, 0, 0]),
        ("pinhole_rational", [0] * 8),
        ("fisheye", [0, 0, 0, 0]),
    ],
)
def test_ground_calibration_loads_all_intrinsic_models(
    tmp_path, model: str, distortion: list[int]
) -> None:
    document = {
        "model_type": model,
        "image_size": [32, 24],
        "camera_matrix": np.eye(3).tolist(),
        "distortion": distortion,
        "new_camera_matrix": np.eye(3).tolist(),
        "quality": {"usable": True},
    }
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded, original = load_intrinsics(path)
    assert loaded.model is CameraModelType(model)
    assert original == document


def test_ground_calibration_rejects_unusable_intrinsics(tmp_path) -> None:
    document = {
        "model_type": "pinhole",
        "image_size": [32, 24],
        "camera_matrix": np.eye(3).tolist(),
        "distortion": [0, 0, 0, 0, 0],
        "new_camera_matrix": np.eye(3).tolist(),
        "quality": {"usable": False},
    }
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="marked unusable"):
        load_intrinsics(path)
