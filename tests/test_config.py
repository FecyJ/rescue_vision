from __future__ import annotations

import json

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


def config_text(*, image_size: str = "[32, 24]", extra: str = "") -> str:
    return f"""schema_version: 1
camera:
  backend: rpicam_vid
  image_size: {image_size}
  fps: 20
  lens_position: 1.0
geometry:
  enabled: true
  intrinsics_path: intrinsics.json
  ground_mapping_path: ground.json
recording:
  queue_capacity: 4
  image_format: png
processing:
  max_observation_age_ms: 150.0
{extra}"""


def test_strict_config_and_geometry_build(tmp_path) -> None:
    calibration = write_intrinsics(tmp_path / "intrinsics.json")
    (tmp_path / "ground.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
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
