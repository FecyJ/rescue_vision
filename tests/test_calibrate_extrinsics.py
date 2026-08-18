from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cv2

import rescue_vision.calibration.calibrate_extrinsics_ground as ground_module
from rescue_vision.calibration.calibrate_extrinsics_ground import (
    BoardCalibrationSpec,
    BoardImageSpec,
    board_points_field_mm,
    detect_board_observation,
    estimate_planar_pose,
    field_points_to_robot_ground,
    fit_station_robust_homography,
    ground_calibration_frame_metadata,
    load_configured_intrinsics,
    load_board_calibration,
    load_intrinsics,
    pose_ground_homography,
    validate_capture_session,
)
from rescue_vision.geometry.camera_model import (
    CameraCalibration,
    CameraModel,
    CameraModelType,
)


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


def _board_calibration_document() -> dict:
    coordinates = [(100.0, 200.0), (300.0, 200.0), (100.0, 400.0), (200.0, 300.0)]
    return {
        "coordinate_frame": ground_calibration_frame_metadata(),
        "board": {
            "pattern_size_internal_corners": [11, 8],
            "physical_square_count": [12, 9],
            "square_size_mm": 15.0,
            "reference": "lower_left_inner_corner",
            "reference_corner_marked": True,
            "edge_margin_mm": {
                "left": 20.0,
                "right": 20.0,
                "bottom": 18.0,
                "top": 18.0,
            },
            "detected_corner_order": "reference_first",
        },
        "images": [
            {
                "name": f"station_{index}",
                "image": f"images/board_{index}.png",
                "reference_inner_corner_global_mm": [
                    coordinates[index - 1][0],
                    coordinates[index - 1][1],
                ],
                "role": "holdout" if index == 4 else "fit",
            }
            for index in range(1, 5)
        ],
    }


def test_board_calibration_loads_global_positions_and_margins(tmp_path) -> None:
    path = tmp_path / "board_calibration.json"
    document = _board_calibration_document()
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_board_calibration(path)

    assert loaded.square_size_mm == pytest.approx(15.0)
    assert loaded.edge_margin_mm == pytest.approx((20.0, 20.0, 18.0, 18.0))
    assert len(loaded.images) == 4
    assert loaded.images[0].image_path == (tmp_path / "images/board_1.png").resolve()
    assert loaded.images[-1].role == "holdout"


def test_outer_corner_reference_uses_left_and_bottom_margins(tmp_path) -> None:
    path = tmp_path / "board_calibration.json"
    document = _board_calibration_document()
    document["board"]["reference"] = "lower_left_outer_corner"
    for item in document["images"]:
        inner_x, inner_y = item.pop("reference_inner_corner_global_mm")
        item["reference_outer_corner_global_mm"] = [inner_x - 20.0, inner_y - 18.0]
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_board_calibration(path)

    assert loaded.reference == "lower_left_outer_corner"
    assert loaded.images[0].reference_corner_field_mm == pytest.approx(
        [80.0, 182.0]
    )
    assert loaded.images[0].reference_inner_corner_field_mm == pytest.approx(
        [100.0, 200.0]
    )


def test_board_calibration_requires_explicit_reference_order(tmp_path) -> None:
    path = tmp_path / "board_calibration.json"
    document = _board_calibration_document()
    document["board"]["detected_corner_order"] = "opencv_default"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="detected_corner_order"):
        load_board_calibration(path)


def test_board_calibration_requires_a_marked_reference_corner(tmp_path) -> None:
    path = tmp_path / "board_calibration.json"
    document = _board_calibration_document()
    document["board"]["reference_corner_marked"] = False
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="reference_corner_marked"):
        load_board_calibration(path)


def test_board_calibration_rejects_collinear_fit_stations(tmp_path) -> None:
    path = tmp_path / "board_calibration.json"
    document = _board_calibration_document()
    for index, item in enumerate(document["images"][:3]):
        item["reference_inner_corner_global_mm"] = [float(index * 100), 200.0]
    document["images"][3]["reference_inner_corner_global_mm"] = [50.0, 200.0]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="span two field dimensions"):
        load_board_calibration(path)


def test_board_points_expand_long_x_and_short_y_edges() -> None:
    field = board_points_field_mm((100.0, 200.0), 15.0)

    assert field.shape == (88, 2)
    assert field[0] == pytest.approx([100.0, 200.0])
    assert field[1] == pytest.approx([115.0, 200.0])
    assert field[11] == pytest.approx([100.0, 215.0])
    assert field[-1] == pytest.approx([250.0, 305.0])


def test_field_to_robot_mapping_preserves_declared_right_handed_axes() -> None:
    field = np.array([[100.0, 200.0], [-50.0, 0.0]])

    robot = field_points_to_robot_ground(field)

    assert np.allclose(robot, [[200.0, -100.0], [0.0, 50.0]])


def test_detect_board_observation_expands_all_corners_without_clicks(tmp_path) -> None:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    square_px = 25
    origin_u, origin_v = 100, 100
    for row in range(9):
        for column in range(12):
            if (row + column) % 2 == 0:
                cv2.rectangle(
                    image,
                    (
                        origin_u + column * square_px,
                        origin_v + row * square_px,
                    ),
                    (
                        origin_u + (column + 1) * square_px,
                        origin_v + (row + 1) * square_px,
                    ),
                    (0, 0, 0),
                    thickness=-1,
                )
    image_path = tmp_path / "board.png"
    cv2.imwrite(str(image_path), image)
    calibration = CameraCalibration(
        calibration_id="synthetic",
        model=CameraModelType.PINHOLE,
        image_size=(640, 480),
        K=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        new_K=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
    )
    board = BoardCalibrationSpec(
        source_path=tmp_path / "board_calibration.json",
        square_size_mm=15.0,
        edge_margin_mm=(20.0, 20.0, 20.0, 20.0),
        detected_corner_order="reference_first",
        images=(),
    )
    observation = detect_board_observation(
        BoardImageSpec(
            image_path=image_path,
            reference_inner_corner_field_mm=(10.0, 20.0),
            role="fit",
            name="test",
        ),
        board,
        CameraModel(calibration),
    )

    assert observation.raw_pixels.shape == (88, 2)
    assert observation.undistorted_pixels.shape == (88, 2)
    assert observation.field_points_mm[0] == pytest.approx([10.0, 20.0])
    assert observation.robot_points_mm[0] == pytest.approx([20.0, -10.0])


def test_multiple_board_positions_recover_known_pose() -> None:
    camera_matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(rotation)
    translation = np.array([[0.0], [0.0], [1800.0]])

    field_references = [
        (-600.0, 500.0),
        (300.0, 500.0),
        (-600.0, 1100.0),
        (300.0, 1100.0),
    ]
    robot_points = np.concatenate(
        [
            field_points_to_robot_ground(
                board_points_field_mm(reference, 15.0)
            )
            for reference in field_references
        ],
        axis=0,
    )
    object_points = np.column_stack(
        (robot_points, np.zeros(len(robot_points), dtype=np.float64))
    )
    pixels, _ = cv2.projectPoints(
        object_points,
        rvec,
        translation,
        camera_matrix,
        np.zeros((4, 1)),
    )
    pixels = pixels.reshape(-1, 2)

    _, estimated_t, estimated_r, camera_position, rmse = estimate_planar_pose(
        robot_points,
        pixels,
        camera_matrix,
    )

    assert estimated_t == pytest.approx(translation, abs=1e-6)
    assert estimated_r == pytest.approx(rotation, abs=1e-6)
    assert camera_position.reshape(3) == pytest.approx(
        [0.0, 0.0, 1800.0], abs=1e-6
    )
    assert rmse < 1e-6


def _synthetic_observation(
    name: str,
    origin: tuple[float, float],
    ground_to_image: np.ndarray,
    *,
    corrupt: bool = False,
) -> ground_module.BoardObservation:
    local = np.asarray(
        [[x, y] for y in np.linspace(0.0, 120.0, 4) for x in np.linspace(0.0, 160.0, 5)],
        dtype=np.float64,
    )
    ground = local + np.asarray(origin, dtype=np.float64)
    pixels = ground_module.transform_points(ground, ground_to_image)
    if corrupt:
        pixels = pixels + np.column_stack(
            (np.linspace(80.0, 180.0, len(pixels)), np.linspace(-120.0, 90.0, len(pixels)))
        )
    spec = BoardImageSpec(
        image_path=Path(f"/{name}.png"),
        reference_inner_corner_field_mm=origin,
        role="fit",
        name=name,
        reference_corner_field_mm=origin,
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    return ground_module.BoardObservation(
        spec=spec,
        image=image,
        undistorted_image=image,
        raw_pixels=pixels.copy(),
        undistorted_pixels=pixels,
        field_points_mm=ground.copy(),
        robot_points_mm=ground,
        sharpness=100.0,
    )


def test_station_robust_fit_rejects_one_systematically_bad_station() -> None:
    ground_to_image = np.array(
        [[0.7, 0.03, 500.0], [-0.02, 0.65, 300.0], [0.00002, 0.00004, 1.0]],
        dtype=np.float64,
    )
    observations = [
        _synthetic_observation("left", (-500.0, 400.0), ground_to_image),
        _synthetic_observation("right", (300.0, 400.0), ground_to_image),
        _synthetic_observation("far", (-100.0, 1200.0), ground_to_image),
        _synthetic_observation("bad", (0.0, 800.0), ground_to_image, corrupt=True),
    ]

    result = fit_station_robust_homography(
        observations,
        threshold_px=2.0,
        minimum_station_inlier_ratio=0.6,
        maximum_corners_per_station=12,
    )

    assert set(result.valid_station_names) == {"left", "right", "far"}
    assert result.station_metrics["bad"]["valid"] is False
    assert len(result.pose_ground_points_mm) <= 36
    assert result.worst_leave_one_out_mean_error_mm < 1e-3


def test_capture_session_rejects_lens_position_mismatch(tmp_path) -> None:
    path = tmp_path / "board_calibration.json"
    document = _board_calibration_document()
    path.write_text(json.dumps(document), encoding="utf-8")
    board = load_board_calibration(path)
    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "image_size": [640, 480],
                "lens_position": 1.5,
                "pixel_format": "RGB888",
                "camera_model": "imx708_wide",
                "sensor_pixel_array_size": [4608, 2592],
                "scaler_crop": [0, 0, 4608, 2592],
                "square_size_mm": 15.0,
                "long_margin_mm": 20.0,
                "short_margin_mm": 18.0,
                "pattern_size_internal_corners": [11, 8],
                "physical_square_count": [12, 9],
                "detected_corner_order": "reference_first",
            }
        ),
        encoding="utf-8",
    )
    calibration = CameraCalibration(
        calibration_id="session_test",
        model=CameraModelType.PINHOLE,
        image_size=(640, 480),
        K=np.eye(3),
        D=np.zeros(5),
        new_K=np.eye(3),
        lens_position=1.0,
        camera_model="imx708_wide",
        sensor_pixel_array_size=(4608, 2592),
        scaler_crop=(0, 0, 4608, 2592),
    )

    with pytest.raises(ValueError, match="lens_position"):
        validate_capture_session(board, calibration)

    session = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    session["lens_position"] = 1.0
    (tmp_path / "session.json").write_text(json.dumps(session), encoding="utf-8")
    assert validate_capture_session(board, calibration)["lens_position"] == 1.0
