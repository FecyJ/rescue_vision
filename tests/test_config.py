from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from rescue_vision.config.runtime import load_runtime_config
from rescue_vision.geometry.camera_model import CameraCalibration, CameraModelType
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.geometry.types import FieldPoint, robot_frame_metadata
from rescue_vision.world import (
    PhysicalRegionKind,
    PhysicalStaticRegion,
    RegionKind,
    StaticFieldMap,
    TeamColor,
)


def write_intrinsics(path, *, usable: bool = True) -> CameraCalibration:
    calibration = CameraCalibration(
        calibration_id="test",
        model=CameraModelType.PINHOLE,
        image_size=(32, 24),
        K=np.eye(3),
        D=np.zeros(5),
        new_K=np.eye(3),
    )
    path.write_text(
        json.dumps(
            {
                "calibration_id": calibration.calibration_id,
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
    return f"""camera:
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
uart:
  enabled: false
  device: null
  read_timeout_ms: 100.0
  write_timeout_ms: 100.0
  receive_queue_capacity: 256
remote:
  enabled: false
  role: server
  host: 0.0.0.0
  port: 8765
  access_mode: observe_only
  connect_timeout_ms: 2000.0
  io_timeout_ms: 100.0
  control_queue_capacity: 32
  observation_queue_capacity: 2
  max_header_bytes: 4096
  max_payload_bytes: 2097152
motion:
  enabled: false
  wheel_track_m: null
  max_linear_velocity_m_s: 0.25
  max_angular_velocity_rad_s: 1.0
  max_wheel_velocity_m_s: 0.30
  max_linear_acceleration_m_s2: 0.50
  max_linear_deceleration_m_s2: 0.75
  max_angular_acceleration_rad_s2: 4.0
  max_angular_deceleration_rad_s2: 6.0
  min_wheel_velocity_m_s: 0.02
  max_remote_command_valid_for_ms: 500
  synchronization_timeout_s: 1.0
  gripper:
    enabled: false
    open_left_angle_deg: null
    open_right_angle_deg: null
    closed_left_angle_deg: null
    closed_right_angle_deg: null
    transport_left_angle_deg: null
    transport_right_angle_deg: null
    full_travel_time_s: null
    angle_sum_deg: null
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
  team_color: unknown
  static_map:
    center_cross:
      intersection_field_mm: [0.0, 0.0]
      terminals:
        positive_x: plain_boundary
        negative_x: plain_boundary
        positive_y: red_safe_zone
        negative_y: blue_safe_zone
    regions:
      - region_id: red-material
        kind: red_material
        polygon_field_mm: [[-100.0, 500.0], [0.0, 500.0], [0.0, 600.0], [-100.0, 600.0]]
      - region_id: red-injured
        kind: red_injured
        polygon_field_mm: [[0.0, 500.0], [100.0, 500.0], [100.0, 600.0], [0.0, 600.0]]
      - region_id: blue-material
        kind: blue_material
        polygon_field_mm: [[0.0, -600.0], [100.0, -600.0], [100.0, -500.0], [0.0, -500.0]]
      - region_id: blue-injured
        kind: blue_injured
        polygon_field_mm: [[-100.0, -600.0], [0.0, -600.0], [0.0, -500.0], [-100.0, -500.0]]
      - region_id: start-1
        kind: start_zone
        polygon_field_mm: [[-600.0, 540.0], [-540.0, 540.0], [-540.0, 600.0], [-600.0, 600.0]]
mission:
  match_duration_s: 180.0
  no_motion_timeout_s: 15.0
  opponent_contact_timeout_s: 10.0
  target_priority: [orange_injured, black_core, green_supply]
perception:
  detection_threshold: 0.25
  k0_threshold: 0.5
  color_classifier:
    ranges:
      green_supply:
        - lower: [35, 70, 71]
          upper: [84, 255, 255]
      black_core:
        - lower: [0, 0, 0]
          upper: [179, 255, 70]
      orange_injured:
        - lower: [0, 90, 80]
          upper: [20, 255, 255]
        - lower: [170, 90, 80]
          upper: [179, 255, 255]
      blue_danger:
        - lower: [85, 50, 71]
          upper: [110, 255, 255]
    min_color_fraction: 0.15
    min_color_dominance: 0.70
    min_dominance_margin: 0.20
    morphology_kernel_size: 3
    open_iterations: 1
    close_iterations: 1
    min_component_area_fraction: 0.002
  target_ground_geometry:
    enabled: false
    objects:
      green_supply:
        shape: box
        length_mm: 40.0
        width_mm: 40.0
        height_mm: 40.0
      black_core:
        shape: regular_tetrahedron
        edge_mm: 40.0
      orange_injured:
        shape: box
        length_mm: 80.0
        width_mm: 40.0
        height_mm: 40.0
      blue_danger:
        shape: box
        length_mm: 40.0
        width_mm: 40.0
        height_mm: 40.0
    fitting:
      coarse_center_step_mm: 5.0
      coarse_yaw_step_deg: 10.0
      refine_center_step_mm: 1.0
      refine_center_radius_mm: 6.0
      refine_top_candidates: 3
      refine_yaw_step_deg: 2.0
      refine_yaw_radius_deg: 10.0
      search_radius_margin_mm: 8.0
      silhouette_weight: 0.65
      contour_weight: 0.25
      contact_weight: 0.10
      contour_distance_scale_px: 4.0
      contact_distance_scale_px: 6.0
      max_contact_residual_px: 12.0
      ambiguity_score_delta: 0.03
      max_center_uncertainty_mm: 12.0
      min_fit_score: 0.60
      min_silhouette_iou: 0.45
hailo:
  enabled: false
  hef_path: null
  postprocess_onnx_path: null
  output_mapping_path: null
  raw_classes: []
  backend_score_threshold: 0.01
  max_detections: 100
{extra}"""


def test_strict_config_and_geometry_build(tmp_path) -> None:
    calibration = write_intrinsics(tmp_path / "intrinsics.json")
    (tmp_path / "ground.json").write_text(
        json.dumps(
            {
                "quality": {
                    "usable": True,
                    "physically_valid": True,
                },
                "image_size": [32, 24],
                "model_type": "pinhole",
                "calibration_id": calibration.calibration_id,
                "coordinate_frame": robot_frame_metadata(),
                "image_to_ground": np.eye(3).tolist(),
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(), encoding="utf-8")
    config = load_runtime_config(path)
    assert not config.motion.cluster_breakup.enabled
    geometry = config.build_geometry()
    assert geometry is not None
    assert geometry.camera_model.image_size == (32, 24)
    assert geometry.ground_projector is not None


def test_center_cross_localizer_requires_enabled_and_ground_mapping(
    tmp_path,
) -> None:
    calibration = write_intrinsics(tmp_path / "intrinsics.json")
    (tmp_path / "ground.json").write_text(
        json.dumps(
            {
                "quality": {"usable": True, "physically_valid": True},
                "image_size": [32, 24],
                "model_type": "pinhole",
                "calibration_id": calibration.calibration_id,
                "coordinate_frame": robot_frame_metadata(),
                "image_to_ground": np.eye(3).tolist(),
            }
        ),
        encoding="utf-8",
    )
    localization = """
localization:
  enabled: true
  ray_min_forward_distance_mm: 500.0
  ray_max_forward_distance_mm: 1800.0
  ray_max_lateral_distance_mm: 120.0
  ray_angle_tolerance_deg: 10.0
  min_anchor_confidence: 0.25
  max_prior_heading_innovation_deg: 20.0
  position_uncertainty_floor_mm: 20.0
  heading_uncertainty_floor_deg: 3.0
"""
    enabled_text = config_text(extra=localization)
    path = tmp_path / "runtime.yaml"
    path.write_text(enabled_text, encoding="utf-8")
    loaded = load_runtime_config(path)
    geometry = loaded.build_geometry()
    assert geometry is not None
    assert loaded.build_center_cross_localizer(
        ground_projector=geometry.ground_projector
    ) is not None
    tracker = loaded.build_static_field_landmark_tracker()
    corner_localizer = loaded.build_safe_zone_corner_localizer()
    assert tracker.config.confirmation_hits == 2
    assert corner_localizer.config.min_baseline_mm == pytest.approx(150.0)
    assert loaded.build_center_cross_localizer(ground_projector=None) is None

    disabled_localization_path = tmp_path / "disabled_localization.yaml"
    disabled_localization_path.write_text(
        config_text(
            extra=localization.replace(
                "  enabled: true",
                "  enabled: false",
                1,
            )
        ),
        encoding="utf-8",
    )
    disabled_localization = load_runtime_config(disabled_localization_path)
    assert disabled_localization.build_center_cross_localizer(
        ground_projector=geometry.ground_projector
    ) is None

    disabled_ground_path = tmp_path / "disabled_ground.yaml"
    disabled_ground_path.write_text(
        enabled_text.replace(
            "  ground_mapping_enabled: true",
            "  ground_mapping_enabled: false",
        ),
        encoding="utf-8",
    )
    disabled_ground = load_runtime_config(disabled_ground_path)
    assert disabled_ground.build_center_cross_localizer(
        ground_projector=geometry.ground_projector
    ) is None


def test_odometry_imu_fusion_builds_from_single_motion_calibration(tmp_path) -> None:
    text = config_text(
        extra="""
localization:
  enabled: true
  fusion:
    enabled: true
    imu_calibration:
      gyro_bias_rad_s: [0.001, -0.002, 0.01]
      sensor_to_robot_rotation:
        - [0.0, 0.0, -1.0]
        - [0.0, 1.0, 0.0]
        - [1.0, 0.0, 0.0]
    initial_pose:
      x_mm: 10.0
      y_mm: -20.0
      heading_deg: 90.0
      position_uncertainty_mm: 50.0
      heading_uncertainty_deg: 5.0
      confidence: 0.6
"""
    )
    text = text.replace("uart:\n  enabled: false", "uart:\n  enabled: true")
    text = text.replace("  device: null", "  device: /dev/ttyAMA0", 1)
    text = text.replace(
        "motion:\n  enabled: false\n  wheel_track_m: null",
        """motion:
  enabled: true
  wheel_track_m: 0.2
  odometry:
    enabled: true
    encoder_counts_per_revolution: 4096
    left_wheel_radius_mm: 32.0
    right_wheel_radius_mm: 31.8
    gyro_z_sign: -1""",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")

    loaded = load_runtime_config(path)
    estimator = loaded.build_odometry_imu_fusion()

    assert estimator is not None
    assert estimator.calibration.encoder_counts_per_revolution == 4096
    assert estimator.calibration.gyro_z_sign == -1
    assert estimator.config.imu_frame_calibration.gyro_bias_rad_s == (
        0.001,
        -0.002,
        0.01,
    )
    assert estimator.config.imu_frame_calibration.sensor_to_robot_rotation == (
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    assert estimator.config.initial_pose.position == FieldPoint(10.0, -20.0)
    assert estimator.config.initial_pose.heading_rad == pytest.approx(np.pi / 2)


def test_odometry_imu_fusion_can_build_without_visual_localization(tmp_path) -> None:
    text = config_text(
        ground_mapping_enabled=False,
        extra="""
localization:
  enabled: false
  fusion:
    enabled: true
    initial_pose:
      x_mm: -1350.0
      y_mm: -1350.0
      heading_deg: 90.0
      position_uncertainty_mm: 100.0
      heading_uncertainty_deg: 10.0
      confidence: 0.5
""",
    )
    text = text.replace("uart:\n  enabled: false", "uart:\n  enabled: true")
    text = text.replace("  device: null", "  device: /dev/ttyAMA0", 1)
    text = text.replace(
        "motion:\n  enabled: false\n  wheel_track_m: null",
        """motion:
  enabled: true
  wheel_track_m: 0.2
  odometry:
    enabled: true
    encoder_counts_per_revolution: 4096
    left_wheel_radius_mm: 32.0
    right_wheel_radius_mm: 31.8
    gyro_z_sign: -1""",
    )
    path = tmp_path / "runtime-wheel-imu.yaml"
    path.write_text(text, encoding="utf-8")

    loaded = load_runtime_config(path)

    assert not loaded.localization.center_cross.enabled
    assert loaded.build_odometry_imu_fusion() is not None


def test_enabled_fusion_requires_odometry_calibration(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text(extra="localization:\n  enabled: true\n  fusion:\n    enabled: true\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="motion.odometry"):
        load_runtime_config(path)


def test_motion_odometry_overrun_budget_parsing(tmp_path) -> None:
    def write_config(overrun_line: str, name: str) -> Path:
        text = config_text()
        text = text.replace("uart:\n  enabled: false", "uart:\n  enabled: true")
        text = text.replace("  device: null", "  device: /dev/ttyAMA0", 1)
        text = text.replace(
            "motion:\n  enabled: false\n  wheel_track_m: null",
            f"""motion:
  enabled: true
  wheel_track_m: 0.2
  odometry:
    enabled: true
    encoder_counts_per_revolution: 4096
    left_wheel_radius_mm: 32.0
    right_wheel_radius_mm: 31.8
    gyro_z_sign: -1{overrun_line}""",
        )
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    loaded = load_runtime_config(write_config("", "default.yaml"))
    assert loaded.motion.odometry.max_consecutive_overrun_samples == 1

    loaded = load_runtime_config(
        write_config("\n    max_consecutive_overrun_samples: 3", "budget.yaml")
    )
    assert loaded.motion.odometry.max_consecutive_overrun_samples == 3

    loaded = load_runtime_config(
        write_config(
            "\n    max_consecutive_overrun_samples: null",
            "disabled.yaml",
        )
    )
    assert loaded.motion.odometry.max_consecutive_overrun_samples is None

    with pytest.raises(
        ValueError,
        match="max_consecutive_overrun_samples",
    ):
        load_runtime_config(write_config("\n    max_consecutive_overrun_samples: -1", "invalid.yaml"))


def test_disabled_hailo_does_not_create_target_pose_detector(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text(
            intrinsics_enabled=False,
            ground_mapping_enabled=False,
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.build_target_pose_detector() is None


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


def test_hailo_v3_classes_and_relative_paths(tmp_path) -> None:
    text = config_text().replace(
        """  enabled: false
  hef_path: null
  postprocess_onnx_path: null
  output_mapping_path: null
  raw_classes: []""",
        """  enabled: true
  hef_path: bundle/model.hef
  postprocess_onnx_path: bundle/postprocess.onnx
  output_mapping_path: bundle/mapping.json
  raw_classes: [green_supply, black_core, orange_injured, blue_danger, center_cross, safe_zone]""",
    )
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_runtime_config(path)
    assert config.hailo.hef_path == (tmp_path / "bundle/model.hef").resolve()
    assert config.hailo.raw_classes[-2:] == ("center_cross", "safe_zone")


def test_hailo_classes_must_match_v3_order(tmp_path) -> None:
    text = config_text().replace(
        "  raw_classes: []",
        "  raw_classes: [green_supply]",
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


def test_color_classifier_config_is_loaded_from_perception(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(), encoding="utf-8")

    config = load_runtime_config(path)

    assert config.perception.detection_threshold == pytest.approx(0.25)
    assert config.perception.k0_threshold == pytest.approx(0.5)
    classifier = config.perception.color_classifier
    assert classifier.green_supply[0].lower == (35, 70, 71)
    assert len(classifier.orange_injured) == 2
    assert classifier.morphology_kernel_size == 3
    target_geometry = config.perception.target_ground_geometry
    assert target_geometry.enabled is False
    assert target_geometry.green_supply.length_mm == pytest.approx(40.0)
    assert target_geometry.black_core.edge_mm == pytest.approx(40.0)
    assert target_geometry.orange_injured.length_mm == pytest.approx(80.0)
    assert config.world.static_map.safe_zone_dimensions_mm(
        TeamColor.RED
    ) == pytest.approx((200.0, 100.0))
    assert config.world.static_map.start_zone_dimensions_mm() == (
        (60.0, 60.0),
    )


def test_target_ground_geometry_estimator_is_strictly_configured(
    tmp_path,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(), encoding="utf-8")
    disabled = load_runtime_config(path)
    assert (
        disabled.perception.build_target_ground_geometry_estimator(
            max_observation_age_ms=150.0,
            ground_projector=None,
        )
        is None
    )

    path.write_text(
        config_text().replace(
            "  target_ground_geometry:\n    enabled: false",
            "  target_ground_geometry:\n    enabled: true",
        ),
        encoding="utf-8",
    )
    enabled = load_runtime_config(path)
    with pytest.raises(RuntimeError, match="requires ground mapping"):
        enabled.perception.build_target_ground_geometry_estimator(
            max_observation_age_ms=150.0,
            ground_projector=None,
        )

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    projector = GroundProjector(
        np.eye(3),
        new_camera_matrix=camera_matrix,
        rotation_robot_to_camera=rotation,
        translation_robot_to_camera_mm=np.asarray((0.0, 0.0, 500.0)),
    )
    estimator = enabled.perception.build_target_ground_geometry_estimator(
        max_observation_age_ms=150.0,
        ground_projector=projector,
    )
    assert estimator is not None
    assert estimator.config.black_core.edge_mm == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "        edge_mm: 40.0",
            "        edge_mm: 0.0",
            "edge_mm",
        ),
        (
            "        shape: regular_tetrahedron",
            "        shape: sphere",
            "shape",
        ),
        (
            "      silhouette_weight: 0.65",
            "      silhouette_weight: 0.70",
            "must equal",
        ),
        (
            "      refine_center_step_mm: 1.0",
            "      refine_center_step_mm: 10.0",
            "must not exceed",
        ),
    ],
)
def test_target_ground_geometry_config_rejects_invalid_values(
    tmp_path,
    old,
    new,
    message,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text().replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            """      blue_danger:
        - lower: [85, 50, 71]
          upper: [110, 255, 255]
""",
            """      unexpected_color:
        - lower: [85, 50, 71]
          upper: [110, 255, 255]
""",
            "ranges keys",
        ),
        ("lower: [35, 70, 71]", "lower: [180, 70, 71]", "HSV component"),
        (
            """      blue_danger:
        - lower: [85, 50, 71]
          upper: [110, 255, 255]""",
            """      blue_danger:
        - lower: [35, 70, 71]
          upper: [84, 255, 255]""",
            "must not overlap",
        ),
        (
            "    morphology_kernel_size: 3",
            "    morphology_kernel_size: 4",
            "positive odd",
        ),
    ],
)
def test_color_classifier_rejects_invalid_schema(
    tmp_path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text().replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_runtime_config(path)


def test_removed_hailo_semantic_threshold_is_rejected(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  backend_score_threshold: 0.01",
            "  backend_score_threshold: 0.01\n  semantic_threshold: 0.5",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic_threshold"):
        load_runtime_config(path)


def test_runtime_example_matches_strict_schema() -> None:
    example = (
        Path(__file__).resolve().parents[1] / "configs" / "runtime.example.yaml"
    )
    config = load_runtime_config(example)
    assert config.hailo.enabled is False


def test_optional_sections_use_safe_defaults(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "camera:\n"
        "  backend: rpicam_vid\n"
        "  image_size: [32, 24]\n"
        "  fps: 20\n"
        "  lens_position: 1.0\n",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert not config.uart.enabled
    assert not config.remote.enabled
    assert not config.motion.enabled
    assert not config.hailo.enabled
    assert not config.perception.target_ground_geometry.enabled
    assert config.perception.detection_threshold == pytest.approx(0.25)


def test_perception_supports_partial_nested_overrides(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "camera:\n"
        "  backend: rpicam_vid\n"
        "  image_size: [32, 24]\n"
        "  fps: 20\n"
        "  lens_position: 1.0\n"
        "perception:\n"
        "  detection_threshold: 0.4\n"
        "  color_classifier:\n"
        "    min_color_fraction: 0.22\n",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.perception.detection_threshold == pytest.approx(0.4)
    assert config.perception.k0_threshold == pytest.approx(0.5)
    assert config.perception.color_classifier.min_color_fraction == pytest.approx(0.22)
    assert config.perception.color_classifier.morphology_kernel_size == 3


def test_uart_config_builds_channel_without_opening_device(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  enabled: false\n  device: null",
            "  enabled: true\n  device: /dev/serial0",
            1,
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)
    channel = config.uart.build_channel()

    assert channel is not None
    assert channel.device == "/dev/serial0"
    assert channel.baudrate == 230400
    assert channel.max_frame_bytes == 64
    assert not channel.started


def test_uart_fixed_protocol_parameters_are_not_runtime_schema(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  read_timeout_ms: 100.0",
            "  baudrate: 57600\n  read_timeout_ms: 100.0",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys in uart.*baudrate"):
        load_runtime_config(path)


def test_enabled_uart_requires_device(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "uart:\n  enabled: false",
            "uart:\n  enabled: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires uart.device"):
        load_runtime_config(path)


def test_motion_config_builds_controller_without_opening_uart(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text()
        .replace(
            "uart:\n  enabled: false\n  device: null",
            "uart:\n  enabled: true\n  device: /dev/serial0",
        )
        .replace(
            "motion:\n  enabled: false\n  wheel_track_m: null",
            "motion:\n  enabled: true\n  wheel_track_m: 0.2",
        )
        .replace(
            """  gripper:
    enabled: false
    open_left_angle_deg: null
    open_right_angle_deg: null
    closed_left_angle_deg: null
    closed_right_angle_deg: null
    transport_left_angle_deg: null
    transport_right_angle_deg: null
    full_travel_time_s: null
    angle_sum_deg: null""",
            """  gripper:
    enabled: true
    open_left_angle_deg: 20.0
    open_right_angle_deg: 180.0
    closed_left_angle_deg: 90.0
    closed_right_angle_deg: 110.0
    transport_left_angle_deg: 50.0
    transport_right_angle_deg: 150.0
    full_travel_time_s: 1.5
    angle_sum_deg: 200.0""",
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)
    channel = config.uart.build_channel()
    controller = config.motion.build_controller(channel)
    executor = config.motion.build_remote_executor(controller)
    gripper_executor = config.motion.build_remote_gripper_executor(controller)

    assert channel is not None
    assert not channel.started
    assert controller is not None
    assert controller.limits.wheel_track_m == pytest.approx(0.2)
    assert controller.limits.max_linear_acceleration_m_s2 == pytest.approx(0.5)
    assert controller.limits.max_linear_deceleration_m_s2 == pytest.approx(0.75)
    assert controller.limits.max_angular_acceleration_rad_s2 == pytest.approx(4.0)
    assert controller.limits.max_angular_deceleration_rad_s2 == pytest.approx(6.0)
    assert controller.limits.min_wheel_velocity_m_s == pytest.approx(0.02)
    assert controller.limits.left_wheel_speed_weight == pytest.approx(1.0)
    assert controller.limits.right_wheel_speed_weight == pytest.approx(1.0)
    assert controller.limits.stall_guard_enabled is True
    assert controller.limits.stall_guard_timeout_ms == 300
    assert controller.limits.stall_guard_min_command_speed_m_s == pytest.approx(
        0.02
    )
    assert controller.limits.stall_guard_stationary_encoder_delta_count == 0
    assert config.motion.synchronization_timeout_s == pytest.approx(1.0)
    assert executor is not None
    assert executor.controller is controller
    assert gripper_executor is not None
    assert gripper_executor.controller is controller
    assert gripper_executor.calibration.full_travel_time_s == pytest.approx(1.5)
    assert gripper_executor.calibration.angle_sum_deg == pytest.approx(200.0)
    assert gripper_executor.calibration.transport_angles_deg == pytest.approx(
        (50.0, 150.0)
    )


def test_motion_synchronization_timeout_is_configurable(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  synchronization_timeout_s: 1.0",
            "  synchronization_timeout_s: 2.5",
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.motion.synchronization_timeout_s == pytest.approx(2.5)


def test_motion_synchronization_timeout_defaults_to_one_second(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace("  synchronization_timeout_s: 1.0\n", ""),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.motion.synchronization_timeout_s == pytest.approx(1.0)


def test_motion_synchronization_timeout_must_be_positive(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  synchronization_timeout_s: 1.0",
            "  synchronization_timeout_s: 0.0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.synchronization_timeout_s"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "name",
    [
        "max_linear_acceleration_m_s2",
        "max_linear_deceleration_m_s2",
        "max_angular_acceleration_rad_s2",
        "max_angular_deceleration_rad_s2",
    ],
)
def test_motion_acceleration_limits_must_be_positive(
    tmp_path,
    name: str,
) -> None:
    configured_values = {
        "max_linear_acceleration_m_s2": "0.50",
        "max_linear_deceleration_m_s2": "0.75",
        "max_angular_acceleration_rad_s2": "4.0",
        "max_angular_deceleration_rad_s2": "6.0",
    }
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            f"  {name}: {configured_values[name]}",
            f"  {name}: 0.0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=name):
        load_runtime_config(path)


def test_motion_config_rejects_removed_wheel_acceleration_key(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  max_linear_acceleration_m_s2: 0.50",
            "  max_wheel_acceleration_m_s2: 0.50",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_wheel_acceleration_m_s2"):
        load_runtime_config(path)


def test_motion_min_wheel_velocity_must_not_exceed_max(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  max_wheel_velocity_m_s: 0.30",
            "  max_wheel_velocity_m_s: 0.01",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_wheel_velocity_m_s"):
        load_runtime_config(path)


def test_motion_wheel_speed_weights_are_parsed(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text()
        .replace(
            "  max_linear_acceleration_m_s2: 0.50",
            "  max_linear_acceleration_m_s2: 0.50\n"
            "  left_wheel_speed_weight: 1.05\n"
            "  right_wheel_speed_weight: 0.95",
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.motion.left_wheel_speed_weight == pytest.approx(1.05)
    assert config.motion.right_wheel_speed_weight == pytest.approx(0.95)


def test_motion_wheel_speed_weight_must_be_positive(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  max_linear_acceleration_m_s2: 0.50",
            "  max_linear_acceleration_m_s2: 0.50\n"
            "  left_wheel_speed_weight: 0.0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.left_wheel_speed_weight"):
        load_runtime_config(path)


def test_gripper_angle_sum_must_be_within_servo_range(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "    angle_sum_deg: null",
            "    angle_sum_deg: 360.1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion.gripper.angle_sum_deg"):
        load_runtime_config(path)


def test_enabled_motion_requires_uart_and_real_wheel_track(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "motion:\n  enabled: false",
            "motion:\n  enabled: true",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wheel_track"):
        load_runtime_config(path)

    path.write_text(
        config_text()
        .replace(
            "motion:\n  enabled: false\n  wheel_track_m: null",
            "motion:\n  enabled: true\n  wheel_track_m: 0.2",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="uart.enabled=true"):
        load_runtime_config(path)


def test_remote_server_config_builds_without_opening_network(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "remote:\n  enabled: false",
            "remote:\n  enabled: true",
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)
    server = config.remote.build_server()

    assert server is not None
    assert server.host == "0.0.0.0"
    assert server.port == 8765
    assert not server.started
    assert config.remote.access_mode.value == "observe_only"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("authentication_key_path", "remote.key"),
        ("handshake_timeout_ms", "2000.0"),
    ],
)
def test_removed_remote_security_fields_are_rejected(
    tmp_path,
    field_name: str,
    field_value: str,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  connect_timeout_ms: 2000.0",
            "  connect_timeout_ms: 2000.0\n"
            f"  {field_name}: {field_value}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys"):
        load_runtime_config(path)


def test_remote_access_mode_is_strict(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "  access_mode: observe_only",
            "  access_mode: competition_control",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="access_mode"):
        load_runtime_config(path)


def test_p1_config_builds_algorithms_and_regions(tmp_path) -> None:
    text = config_text().replace("  team_color: unknown", "  team_color: red")
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")

    config = load_runtime_config(path)

    assert config.tracking.build_tracker().tracks == ()
    assert config.world.build_model().update(
        timestamp_ns=0,
        visual_timestamp_ns=0,
        tracks=[],
    ).regions[0].region_id == "red-material"
    assert config.mission.build_state_machine().phase.value == "wait_start"


def test_static_physical_regions_derive_team_relative_mission_regions(
    tmp_path,
) -> None:
    text = config_text().replace("  team_color: unknown", "  team_color: blue")
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")

    config = load_runtime_config(path)
    mapped = {region.region_id: region.kind for region in config.world.mission_regions()}

    assert config.world.team_color is TeamColor.BLUE
    assert mapped == {
        "red-material": RegionKind.OPPONENT_SAFE,
        "red-injured": RegionKind.OPPONENT_SAFE,
        "blue-material": RegionKind.OWN_MATERIAL,
        "blue-injured": RegionKind.OWN_INJURED,
    }


def test_legacy_world_regions_and_nonzero_center_origin_are_rejected(
    tmp_path,
) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        config_text().replace(
            "  team_color: unknown",
            "  regions: []\n  team_color: unknown",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys in world"):
        load_runtime_config(legacy)

    shifted = tmp_path / "shifted.yaml"
    shifted.write_text(
        config_text().replace(
            "intersection_field_mm: [0.0, 0.0]",
            "intersection_field_mm: [1.0, 0.0]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="field frame origin"):
        load_runtime_config(shifted)


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


def test_near_field_grasp_config_weights_and_strict_keys(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(config_text(extra="near_field_grasp:\n  orange_priority_weight: 0.7\n  target_final_x_mm: 105\n  greedy_target_final_x_mm: 95\n"), encoding="utf-8")
    config = load_runtime_config(path)
    assert config.near_field_grasp.orange_priority_weight == 0.7
    assert config.near_field_grasp.target_final_x_mm == 105
    assert config.near_field_grasp.greedy_target_final_x_mm == 95
    assert config.near_field_grasp.orange_target_final_x_mm == 142
    assert config.near_field_grasp.max_targets == 3
    assert config.near_field_grasp.stopped_scene_new_target_max_bbox_iou == 0.2
    path.write_text(
        config_text(
            extra=(
                "near_field_grasp:\n"
                "  stopped_scene_new_target_max_bbox_iou: 1.01\n"
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stopped_scene_new_target_max_bbox_iou"):
        load_runtime_config(path)
    path.write_text(config_text(extra="near_field_grasp:\n  typo_weight: 1\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_runtime_config(path)


def test_pickup_terminal_speed_gain_is_configurable_and_positive(tmp_path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text(extra="match:\n  pickup_terminal_speed_gain_s_inv: 0.5\n"),
        encoding="utf-8",
    )
    assert load_runtime_config(path).match.pickup_terminal_speed_gain_s_inv == 0.5

    path.write_text(
        config_text(extra="match:\n  pickup_terminal_speed_gain_s_inv: 0\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pickup_terminal_speed_gain_s_inv"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "removed_key",
    (
        "opportunistic_single_green_realign_standoff_mm",
        "green_grab_offset_mm",
        "green_preclose_recheck_range_mm",
        "green_preclose_recheck_hold_ms",
        "green_preclose_max_carried_blocks",
    ),
)
def test_removed_match_near_field_keys_are_rejected(tmp_path, removed_key) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        config_text().replace(
            "tracking:\n",
            f"match:\n  {removed_key}: 1\ntracking:\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys"):
        load_runtime_config(path)


def test_strategy_safe_zone_corner_gate_is_not_tighter_than_pipeline() -> None:
    """strategy 的安全区三点年龄门必须覆盖实测感知延迟，且不超过快照门。

    logs/strategy/strategy_20260911_1021.log 实测该阶段 capture_to_result p50
    中位数约 343 ms、观测年龄最大 638 ms；沿用默认 250 ms 会让每一次三点校准
    都在年龄门被拒（现场表现为 calibration_failure=observation_stale）。上限取
    processing.max_observation_age_ms，因为主循环的快照门已经按该值丢弃旧帧，
    写更大的门限不会生效。
    """

    config = load_runtime_config("configs/runtime.strategy.yaml")
    corners = config.build_safe_zone_corner_localizer().config

    assert corners.max_observation_age_ms <= config.processing.max_observation_age_ms
    assert corners.max_observation_age_ms >= 500.0
