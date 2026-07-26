from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rescue_vision.config.runtime import load_runtime_config
from rescue_vision.geometry.camera_model import CameraCalibration, CameraModelType


def write_intrinsics(path, *, usable: bool = True) -> CameraCalibration:
    calibration = CameraCalibration(
        model=CameraModelType.PINHOLE,
        image_size=(32, 24),
        K=np.eye(3),
        D=np.zeros(5),
        new_K=np.eye(3),
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "model_type": "pinhole",
                "image_size": [32, 24],
                "camera_matrix": calibration.K.tolist(),
                "distortion": calibration.D.reshape(-1).tolist(),
                "new_camera_matrix": calibration.new_K.tolist(),
                "quality": {"usable": usable},
            }
        ),
        encoding="utf-8",
    )
    return calibration


def config_text(
    *,
    image_size: str = "[32, 24]",
    intrinsics_enabled: bool = True,
    ground_mapping_enabled: bool = True,
    extra: str = "",
) -> str:
    return f"""schema_version: 4
camera:
  backend: rpicam_vid
  image_size: {image_size}
  fps: 20
  lens_position: 1.0
geometry:
  intrinsics_enabled: {str(intrinsics_enabled).lower()}
  intrinsics_path: intrinsics.json
  ground_mapping_enabled: {str(ground_mapping_enabled).lower()}
  ground_mapping_path: ground.json
recording:
  queue_capacity: 4
  image_format: png
processing:
  max_observation_age_ms: 150.0
tracking:
  confirmation_hits: 2
  max_association_ground_mm: 250.0
  min_association_iou: 0.1
  max_coast_ms: 600.0
  confidence_decay_per_second: 0.8
  min_confidence: 0.15
world:
  max_visual_age_ms: 250.0
  opponent_max_age_ms: 500.0
  danger_confirm_threshold: 0.6
  danger_suspect_threshold: 0.15
  unknown_suspect_threshold: 0.5
  regions: []
mission:
  match_duration_s: 180.0
  no_motion_timeout_s: 15.0
  opponent_contact_timeout_s: 10.0
  danger_avoid_distance_mm: 500.0
  target_priority: [orange_injured, black_core, green_supply]
hailo:
  enabled: false
  hef_path: null
  postprocess_onnx_path: null
  output_mapping_path: null
  model_version: null
  hef_sha256: null
  raw_classes: []
  class_mapping: {{}}
  backend_score_threshold: 0.01
  detection_threshold: 0.25
  semantic_threshold: 0.5
  k0_threshold: 0.5
  max_detections: 100
{extra}"""


def test_strict_config_and_geometry_build(tmp_path) -> None:
    calibration = write_intrinsics(tmp_path / "intrinsics.json")
    (tmp_path / "ground.json").write_text(
        json.dumps(
            {
                    "schema_version": 3,
                    "quality": {
                        "usable": True,
                        "physically_valid": True,
                    },
                "image_size": [32, 24],
                "intrinsics": {
                    "model_type": "pinhole",
                    "fingerprint_sha256": calibration.fingerprint(),
                },
                "image_to_ground": np.eye(3).tolist(),
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(), encoding="utf-8")
    config = load_runtime_config(path)
    geometry = config.build_geometry()
    assert geometry is not None
    assert geometry.camera_model.image_size == (32, 24)
    assert geometry.ground_projector is not None


def test_intrinsics_can_be_enabled_without_ground_mapping(tmp_path) -> None:
    write_intrinsics(tmp_path / "intrinsics.json")
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text(ground_mapping_enabled=False),
        encoding="utf-8",
    )

    config = load_runtime_config(path)
    camera_model = config.build_camera_model()
    geometry = config.build_geometry()

    assert camera_model is not None
    assert geometry is not None
    assert geometry.camera_model.image_size == (32, 24)
    assert geometry.ground_projector is None


def test_ground_mapping_requires_enabled_intrinsics(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text(intrinsics_enabled=False, ground_mapping_enabled=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires.*intrinsics_enabled=true"):
        load_runtime_config(path)


def test_both_geometry_stages_can_be_disabled(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text(
            intrinsics_enabled=False,
            ground_mapping_enabled=False,
        ),
        encoding="utf-8",
    )
    config = load_runtime_config(path)
    assert config.build_camera_model() is None
    assert config.build_geometry() is None


def test_unknown_config_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(extra="unexpected: true\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_runtime_config(path)


def test_runtime_resolution_mismatch_is_rejected(tmp_path) -> None:
    write_intrinsics(tmp_path / "intrinsics.json")
    (tmp_path / "ground.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(image_size="[64, 48]"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match intrinsics"):
        load_runtime_config(path).build_geometry()


def test_unusable_intrinsics_fail_at_startup(tmp_path) -> None:
    write_intrinsics(tmp_path / "intrinsics.json", usable=False)
    (tmp_path / "ground.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="marked unusable"):
        load_runtime_config(path).build_geometry()


def test_hailo_class_mapping_and_relative_paths(tmp_path) -> None:
    text = config_text().replace(
        """  enabled: false
  hef_path: null
  postprocess_onnx_path: null
  output_mapping_path: null
  model_version: null
  hef_sha256: null
  raw_classes: []
  class_mapping: {}""",
        """  enabled: true
  hef_path: bundle/model.hef
  postprocess_onnx_path: bundle/postprocess.onnx
  output_mapping_path: bundle/mapping.json
  model_version: model-v1
  hef_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  raw_classes: [raw_a, raw_b]
  class_mapping:
    raw_a: green_supply
    raw_b: blue_danger""",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_runtime_config(path)
    assert config.hailo.hef_path == (tmp_path / "bundle/model.hef").resolve()
    assert config.hailo.model_class_mapping()[0].value == "green_supply"
    assert config.hailo.model_class_mapping()[1].value == "blue_danger"


def test_hailo_mapping_must_be_exhaustive(tmp_path) -> None:
    text = config_text().replace(
        "  raw_classes: []\n  class_mapping: {}",
        "  raw_classes: [raw_a]\n  class_mapping: {}",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        load_runtime_config(path)


def test_backend_threshold_must_not_hide_detector_candidates(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  backend_score_threshold: 0.01",
            "  backend_score_threshold: 0.30",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="backend_score_threshold"):
        load_runtime_config(path)


def test_runtime_example_matches_strict_schema() -> None:
    example = (
        Path(__file__).resolve().parents[1] / "configs" / "runtime.example.yaml"
    )
    config = load_runtime_config(example)
    assert config.schema_version == 4


def test_p1_config_builds_algorithms_and_regions(tmp_path) -> None:
    text = config_text().replace(
        "  regions: []",
        """  regions:
    - region_id: own-material
      kind: own_material
      polygon_field_mm:
        - [0.0, 0.0]
        - [100.0, 0.0]
        - [100.0, 100.0]
        - [0.0, 100.0]""",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")

    config = load_runtime_config(path)

    assert config.tracking.build_tracker().tracks == ()
    assert config.world.build_model().update(
        timestamp_ns=0,
        visual_timestamp_ns=0,
        tracks=[],
    ).regions[0].region_id == "own-material"
    assert config.mission.build_state_machine().phase.value == "wait_start"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "  min_association_iou: 0.1",
            "  min_association_iou: 1.1",
            "min_association_iou",
        ),
        (
            "  danger_suspect_threshold: 0.15",
            "  danger_suspect_threshold: 0.7",
            "must not exceed",
        ),
        (
            "  target_priority: [orange_injured, black_core, green_supply]",
            "  target_priority: [green_supply]",
            "target_priority",
        ),
    ],
)
def test_p1_config_rejects_invalid_values(
    tmp_path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text().replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_runtime_config(path)
