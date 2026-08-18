from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from rescue_vision.calibration.calibrate_extrinsics_ground import (
    BoardCalibrationSpec,
    BoardImageSpec,
    detect_board_observation,
    estimate_planar_pose,
    field_points_to_robot_ground,
    load_board_calibration,
)
from rescue_vision.calibration.capture_extrinsics_charuco import (
    CapturedCharucoImage,
    build_charuco_calibration_document,
)
from rescue_vision.calibration.charuco_board import (
    CharucoDetection,
    charuco_detection_jitter_px,
    charuco_points_field_mm,
    create_charuco_board,
    detect_charuco_board,
)
from rescue_vision.geometry.camera_model import (
    CameraCalibration,
    CameraModel,
    CameraModelType,
)


def test_charuco_detector_keeps_ids_for_a_partial_board_view() -> None:
    board = create_charuco_board((12, 9), 40.0, 25.0, "DICT_4X4_100")
    image = board.generateImage((600, 450), marginSize=20)
    partial = image[:, 80:]

    detection = detect_charuco_board(partial, board, minimum_corners=6)

    assert detection is not None
    assert 6 <= len(detection.charuco_ids) < 88
    assert len(detection.charuco_ids) == len(detection.charuco_corners)
    assert len(set(detection.charuco_ids.tolist())) == len(detection.charuco_ids)


def test_charuco_temporal_jitter_uses_common_corner_ids() -> None:
    def detected(offset: float) -> CharucoDetection:
        return CharucoDetection(
            charuco_corners=np.asarray([[10.0 + offset, 20.0], [30.0, 40.0 + offset]]),
            charuco_ids=np.asarray([2, 7]),
            marker_corners=(),
            marker_ids=np.empty((0,), dtype=np.int64),
            detection_scale=1.0,
        )

    jitter = charuco_detection_jitter_px(
        [detected(0.0), detected(0.5), detected(-0.5)],
        minimum_common_corners=2,
    )

    assert jitter == pytest.approx(0.5)


def test_charuco_ids_expand_to_board_coordinates() -> None:
    board = create_charuco_board((12, 9), 15.0, 10.0, "DICT_4X4_100")

    points = charuco_points_field_mm(
        board,
        [0, 1, 11],
        (100.0, 200.0),
        (20.0, 20.0, 18.0, 18.0),
        0,
    )

    assert np.allclose(
        points,
        [
            [135.0, 233.0],
            [150.0, 233.0],
            [135.0, 248.0],
        ],
    )


def test_charuco_board_coordinates_apply_explicit_rotation() -> None:
    board = create_charuco_board((12, 9), 15.0, 10.0, "DICT_4X4_100")

    points = charuco_points_field_mm(
        board,
        [0],
        (100.0, 200.0),
        (20.0, 20.0, 18.0, 18.0),
        90,
    )

    assert np.allclose(points, [[67.0, 235.0]])

def test_partial_charuco_points_recover_planar_pose() -> None:
    board = create_charuco_board((12, 9), 15.0, 10.0, "DICT_4X4_100")
    ids = [0, 1, 2, 11, 12, 13, 22, 23, 24, 35]
    field = charuco_points_field_mm(
        board,
        ids,
        (-100.0, 400.0),
        (20.0, 20.0, 18.0, 18.0),
        0,
    )
    ground = field_points_to_robot_ground(field)
    camera_matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(rotation)
    translation = np.array([[0.0], [0.0], [1800.0]])
    objects = np.column_stack((ground, np.zeros(len(ground))))
    pixels, _ = cv2.projectPoints(
        objects,
        rvec,
        translation,
        camera_matrix,
        np.zeros((4, 1)),
    )
    pixels = pixels.reshape(-1, 2)

    _, estimated_t, _estimated_r, _camera_position, rmse = estimate_planar_pose(
        ground,
        pixels,
        camera_matrix,
        prefer_iterative=True,
    )

    assert estimated_t == pytest.approx(translation, abs=1e-6)
    assert rmse < 1e-6


def test_charuco_capture_document_loads_in_ground_solver(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    station_coordinates = [(100.0, 200.0), (300.0, 200.0), (100.0, 400.0), (200.0, 300.0)]
    records = [
        CapturedCharucoImage(
            name=f"station_{index:02d}",
            image_path=images_dir / f"charuco_{index:02d}.png",
            board_origin_outer_corner_global_mm=station_coordinates[index - 1],
            role="holdout" if index == 4 else "fit",
            charuco_corner_count=12,
            marker_count=8,
        )
        for index in range(1, 5)
    ]
    document = build_charuco_calibration_document(
        tmp_path,
        squares_x=12,
        squares_y=9,
        square_size_mm=15.0,
        marker_size_mm=10.0,
        dictionary_name="DICT_4X4_100",
        minimum_charuco_corners=8,
        board_rotation_degrees=0,
        long_margin_mm=20.0,
        short_margin_mm=18.0,
        images=records,
    )
    path = tmp_path / "board_calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_board_calibration(path)

    assert loaded.board_type == "charuco"
    assert loaded.chessboard_size_squares == (12, 9)
    assert loaded.dictionary_name == "DICT_4X4_100"
    assert loaded.minimum_charuco_corners == 8
    assert loaded.images[0].reference_corner_field_mm == pytest.approx(
        [100.0, 200.0]
    )


def test_ground_observation_uses_visible_charuco_ids(tmp_path) -> None:
    board_model = create_charuco_board((12, 9), 15.0, 10.0, "DICT_4X4_100")
    image = board_model.generateImage((600, 450), marginSize=20)
    image_path = tmp_path / "charuco.png"
    assert cv2.imwrite(str(image_path), image)
    calibration = CameraCalibration(
        calibration_id="charuco_test",
        model=CameraModelType.PINHOLE,
        image_size=(600, 450),
        K=np.array([[500.0, 0.0, 300.0], [0.0, 500.0, 225.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        new_K=np.array([[500.0, 0.0, 300.0], [0.0, 500.0, 225.0], [0.0, 0.0, 1.0]]),
    )
    board_spec = BoardCalibrationSpec(
        source_path=tmp_path / "board_calibration.json",
        square_size_mm=15.0,
        edge_margin_mm=(20.0, 20.0, 18.0, 18.0),
        detected_corner_order="reference_first",
        images=(),
        board_type="charuco",
        chessboard_size_squares=(12, 9),
        marker_size_mm=10.0,
        dictionary_name="DICT_4X4_100",
        minimum_charuco_corners=8,
        reference="opencv_board_origin_outer_corner",
        board_rotation_degrees=0,
    )
    observation = detect_board_observation(
        BoardImageSpec(
            image_path=image_path,
            reference_inner_corner_field_mm=(120.0, 218.0),
            reference_corner_field_mm=(100.0, 200.0),
            role="fit",
            name="charuco",
        ),
        board_spec,
        CameraModel(calibration),
    )

    assert observation.corner_ids is not None
    assert len(observation.corner_ids) == len(observation.raw_pixels)
    assert len(observation.corner_ids) == 88
