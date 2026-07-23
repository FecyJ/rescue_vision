from __future__ import annotations

import numpy as np
import pytest

from rescue_vision.perception.hailo_yolo26_pose import (
    HailoYolo26PoseBackend,
    letterbox_bgr,
    map_hef_outputs_to_onnx_inputs,
    parse_yolo26_pose_output,
)


def test_letterbox_and_coordinate_round_trip() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, transform = letterbox_bgr(image, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert transform.scale == pytest.approx(3.2)
    point = transform.point_to_original(
        50 * transform.scale + transform.x_offset,
        25 * transform.scale + transform.y_offset,
    )
    assert point.u == pytest.approx(50)
    assert point.v == pytest.approx(25)


@pytest.mark.parametrize(("kpt_shape", "width"), [((1, 3), 9), ((3, 3), 15)])
def test_pose_parser_uses_class_id_and_only_k0(kpt_shape, width) -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, transform = letterbox_bgr(image, (640, 640))
    output = np.zeros((1, 2, width), dtype=np.float32)
    output[0, 0, :9] = [32, 192, 320, 352, 0.9, 3, 160, 320, 0.8]
    if width == 15:
        output[0, 0, 9:] = [999, 999, 1, -999, -999, 1]
    parsed = parse_yolo26_pose_output(
        output,
        kpt_shape=kpt_shape,
        transform=transform,
        score_threshold=0.25,
        max_detections=10,
    )
    assert len(parsed) == 1
    assert parsed[0].model_class_id == 3
    assert parsed[0].k0.u == pytest.approx(50)
    assert parsed[0].k0.v == pytest.approx(50)


def test_tensor_mapping_transposes_nhwc() -> None:
    tensor = np.zeros((1, 2, 3, 4), dtype=np.float32)
    mapped = map_hef_outputs_to_onnx_inputs(
        {"hef/output": tensor},
        {"hef/output": ["onnx/input", [4, 2, 3]]},
    )
    assert mapped["onnx/input"].shape == (1, 4, 2, 3)


def test_tensor_mapping_rejects_missing_or_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="Missing"):
        map_hef_outputs_to_onnx_inputs({}, {"missing": ["input", [1, 1, 1]]})
    with pytest.raises(ValueError, match="float32"):
        map_hef_outputs_to_onnx_inputs(
            {"output": np.zeros((1, 1, 1, 1), dtype=np.uint8)},
            {"output": ["input", [1, 1, 1]]},
        )


def test_backend_rejects_hef_checksum_before_hardware_import(tmp_path) -> None:
    hef = tmp_path / "model.hef"
    onnx = tmp_path / "postprocess.onnx"
    mapping = tmp_path / "mapping.json"
    hef.write_bytes(b"not-a-real-hef")
    onnx.write_bytes(b"not-a-real-onnx")
    mapping.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        HailoYolo26PoseBackend(
            hef_path=hef,
            postprocess_onnx_path=onnx,
            output_mapping_path=mapping,
            model_version="test",
            model_sha256="0" * 64,
            class_count=4,
            max_detections=10,
            score_threshold=0.25,
        )
