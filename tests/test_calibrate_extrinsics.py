from __future__ import annotations

import json

import numpy as np
import pytest

import cv2

import rescue_vision.calibration.calibrate_extrinsics_ground as ground_module
from rescue_vision.calibration.calibrate_extrinsics_ground import (
    estimate_planar_pose,
    load_configured_intrinsics,
    load_intrinsics,
    pose_ground_homography,
    validate_correspondence_source,
)
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
        "calibration_id": "test",
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
        "calibration_id": "test",
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


def test_ground_calibration_uses_intrinsics_path_from_runtime_config(
    tmp_path,
) -> None:
    intrinsics_document = {
        "calibration_id": "runtime-selected",
        "model_type": "pinhole",
        "image_size": [32, 24],
        "camera_matrix": np.eye(3).tolist(),
        "distortion": [0, 0, 0, 0, 0],
        "new_camera_matrix": np.eye(3).tolist(),
        "quality": {"usable": True},
    }
    intrinsics_path = tmp_path / "selected_calibration.json"
    intrinsics_path.write_text(
        json.dumps(intrinsics_document),
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "camera:\n"
        "  backend: rpicam_vid\n"
        "  image_size: [32, 24]\n"
        "  fps: 20\n"
        "  lens_position: 1.0\n"
        "geometry:\n"
        "  intrinsics_enabled: true\n"
        "  intrinsics_path: selected_calibration.json\n",
        encoding="utf-8",
    )

    resolved_path, camera_model, loaded_document = load_configured_intrinsics(
        config_path
    )

    assert resolved_path == intrinsics_path.resolve()
    assert camera_model.calibration.calibration_id == "runtime-selected"
    assert camera_model.image_size == (32, 24)
    assert loaded_document == intrinsics_document


def test_ground_calibration_rejects_disabled_runtime_intrinsics(tmp_path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "camera:\n"
        "  backend: rpicam_vid\n"
        "  image_size: [32, 24]\n"
        "  fps: 20\n"
        "  lens_position: 1.0\n"
        "geometry:\n"
        "  intrinsics_enabled: false\n"
        "  intrinsics_path: null\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intrinsics_enabled=false"):
        load_configured_intrinsics(config_path)


def test_planar_pose_and_homography_recover_known_geometry() -> None:
    ground = np.array(
        [[-200.0, -150.0], [200.0, -150.0], [200.0, 150.0], [-200.0, 150.0]]
    )
    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(rotation)
    translation = np.array([[0.0], [0.0], [1000.0]])
    objects = np.column_stack((ground, np.zeros(len(ground))))
    pixels, _ = cv2.projectPoints(
        objects,
        rvec,
        translation,
        camera_matrix,
        np.zeros((4, 1)),
    )

    _, estimated_t, estimated_r, camera_position, rmse = estimate_planar_pose(
        ground,
        pixels.reshape(-1, 2),
        camera_matrix,
    )
    assert rmse < 2e-6
    assert camera_position[2, 0] > 0.0
    ground_to_image, image_to_ground = pose_ground_homography(
        estimated_r,
        estimated_t,
        camera_matrix,
    )
    assert ground_to_image @ image_to_ground == pytest.approx(np.eye(3))


def test_planar_pose_rejects_all_physically_invalid_candidates(
    monkeypatch,
) -> None:
    def invalid_pose(*_args, **_kwargs):
        return (
            True,
            [np.zeros((3, 1))],
            [np.array([[0.0], [0.0], [1000.0]])],
            None,
        )

    monkeypatch.setattr(ground_module.cv2, "solvePnPGeneric", invalid_pose)
    with pytest.raises(RuntimeError, match="physically invalid"):
        estimate_planar_pose(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.eye(3),
        )


def test_pose_homography_rejects_zero_normalization_scale() -> None:
    with pytest.raises(ValueError, match="normalization scale"):
        pose_ground_homography(
            np.eye(3),
            np.array([1.0, 2.0, 0.0]),
            np.eye(3),
        )


def test_correspondences_are_bound_to_image_name_and_size(tmp_path) -> None:
    image = tmp_path / "ground.png"
    cv2.imwrite(str(image), np.zeros((6, 8, 3), dtype=np.uint8))
    correspondences = tmp_path / "correspondences.json"
    correspondences.write_text(
        json.dumps(
            {
                "image": image.name,
                "image_size": [8, 6],
                "points": [],
            }
        ),
        encoding="utf-8",
    )
    validate_correspondence_source(correspondences, image)
    cv2.imwrite(str(image), np.zeros((7, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="recollect"):
        validate_correspondence_source(correspondences, image)
