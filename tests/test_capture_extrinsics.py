from __future__ import annotations

import json
from pathlib import Path

import pytest

from rescue_vision.calibration.calibrate_extrinsics_ground import (
    load_board_calibration,
)
from rescue_vision.calibration.capture_extrinsics_ground import (
    CapturedBoardImage,
    build_board_calibration_document,
    capture_role,
    edge_margin_document,
    parse_args,
    parse_global_coordinate,
    prompt_capture_count,
    prompt_edge_margins,
)


def test_prompt_margins_and_count_are_interactive_but_hardware_free() -> None:
    values = iter(["20", "18"])
    assert prompt_edge_margins(input_fn=lambda _prompt: next(values)) == (
        20.0,
        18.0,
    )

    count_values = iter(["2", "six", "4"])
    assert prompt_capture_count(input_fn=lambda _prompt: next(count_values)) == 4


def test_parse_args_terminal_flag_defaults_off() -> None:
    assert parse_args([]).terminal is False
    assert parse_args(["--terminal"]).terminal is True


def test_coordinate_parser_accepts_space_and_comma_forms() -> None:
    assert parse_global_coordinate("300 800") == (300.0, 800.0)
    assert parse_global_coordinate("-120.5, 60") == (-120.5, 60.0)
    with pytest.raises(ValueError, match="两个数字"):
        parse_global_coordinate("300")


def test_edge_margins_expand_to_solver_schema() -> None:
    assert edge_margin_document(20.0, 18.0) == {
        "left": 20.0,
        "right": 20.0,
        "bottom": 18.0,
        "top": 18.0,
    }
    with pytest.raises(ValueError, match="non-negative"):
        edge_margin_document(-1.0, 18.0)


def test_capture_role_reserves_last_image_for_holdout() -> None:
    assert [capture_role(index, 4) for index in range(4)] == [
        "fit",
        "fit",
        "fit",
        "holdout",
    ]
    with pytest.raises(ValueError, match="index"):
        capture_role(4, 4)


def test_capture_document_loads_directly_in_ground_solver(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    coordinates = [(100.0, 200.0), (300.0, 200.0), (100.0, 400.0), (200.0, 300.0)]
    records = [
        CapturedBoardImage(
            name=f"station_{index:02d}",
            image_path=images_dir / f"board_{index:02d}.png",
            reference_outer_corner_global_mm=coordinates[index - 1],
            role="holdout" if index == 4 else "fit",
            sharpness=100.0 + index,
        )
        for index in range(1, 5)
    ]
    document = build_board_calibration_document(
        tmp_path,
        square_size_mm=15.0,
        long_margin_mm=20.0,
        short_margin_mm=18.0,
        detected_corner_order="reference_first",
        images=records,
    )
    path = tmp_path / "board_calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_board_calibration(path)

    assert loaded.reference == "lower_left_outer_corner"
    assert loaded.edge_margin_mm == pytest.approx((20.0, 20.0, 18.0, 18.0))
    assert loaded.images[0].reference_inner_corner_field_mm == pytest.approx(
        [120.0, 218.0]
    )
    assert document["images"][0]["image"] == "images/board_01.png"
