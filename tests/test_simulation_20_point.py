from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from rescue_vision.app.cluster_breakup import (
    BreakupDecision,
    BreakupState,
    GripperPosture,
)
from rescue_vision.app.simulation_20_point import (
    Simulation20PointSequence,
    Simulation20PointState,
    SimulationHealth,
    SimulationPreflight,
)
from rescue_vision.config import Simulation20PointRuntimeConfig, load_runtime_config
from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.localization import FieldPose2D, FusedPoseEstimate
from rescue_vision.mission import MissionConfig, MissionStateMachine
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    PerceptionSnapshot,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import MultiTargetTracker, TrackingConfig
from rescue_vision.world import (
    HazardState,
    RegionKind,
    StaticRegion,
    WorldModel,
    WorldModelConfig,
)


def runtime_config(**overrides: object) -> Simulation20PointRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "target_delivery_count": 4,
        "prepush_offset_mm": 150.0,
        "robot_footprint_radius_mm": 100.0,
        "target_half_extent_mm": 20.0,
        "safety_margin_mm": 20.0,
        "delivery_inset_mm": 20.0,
        "max_position_uncertainty_mm": 200.0,
        "max_heading_uncertainty_rad": 0.5,
        "target_max_age_ms": 500.0,
        "scan_min_heading_span_rad": 0.5,
        "scan_angular_velocity_rad_s": 0.2,
        "navigation_speed_m_s": 0.1,
        "navigation_angular_kp_rad_s": 1.0,
        "navigation_max_angular_velocity_rad_s": 0.4,
        "navigation_position_tolerance_mm": 30.0,
        "navigation_heading_tolerance_rad": 0.1,
        "approach_speed_m_s": 0.05,
        "alignment_kp_rad_s": 1.0,
        "alignment_max_angular_velocity_rad_s": 0.3,
        "engage_distance_mm": 80.0,
        "engage_lateral_tolerance_mm": 30.0,
        "contact_corridor_length_mm": 120.0,
        "push_speed_m_s": 0.05,
        "push_angular_kp_rad_s": 0.5,
        "push_max_angular_velocity_rad_s": 0.2,
        "retreat_speed_m_s": 0.05,
        "retreat_distance_m": 0.1,
        "retreat_clear_distance_mm": 100.0,
        "delivery_confirm_frames": 2,
        "disengage_confirm_frames": 2,
        "max_breakup_attempts_per_delivery": 1,
        "max_breakup_attempts_total": 2,
        "breakup_settle_confirm_frames": 2,
        "min_breakup_progress_mm": 10.0,
        "min_green_clearance_mm": 40.0,
        "corridor_sample_step_mm": 30.0,
        "completed_target_exclusion_mm": 100.0,
    }
    values.update(overrides)
    return Simulation20PointRuntimeConfig(**values)  # type: ignore[arg-type]


def observation(
    frame_sequence: int,
    timestamp_ns: int,
    ground_point: GroundPoint,
    *,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
) -> TargetObservation:
    box = UndistortedBoundingBox(40.0, 40.0, 60.0, 60.0)
    segmentation = RoiColorSegmentation(
        candidate_class=target_class,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=box,
        mask=np.full((20, 20), 255, dtype=np.uint8),
        color_fraction=1.0,
        dominance=1.0,
    )
    return TargetObservation(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        image_size=(100, 100),
        model_target_class=target_class,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(target_class, 1.0),
        detection_confidence=0.95,
        box=box,
        color_segmentation=segmentation,
        k0=UndistortedPixel(50.0, 50.0),
        k0_confidence=0.95,
        ground_point=ground_point,
        quality=frozenset(),
    )


def snapshot(
    frame_sequence: int,
    timestamp_ns: int,
    *observations: TargetObservation,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        observations=tuple(observations),
    )


class _ImmediateGreenBreakup:
    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        return BreakupDecision(
            timestamp_ns,
            BreakupState.GREEN_FOUND,
            0.0,
            0.0,
            GripperPosture.CLOSED,
            "green_found",
        )


class _ApproachBreakup:
    def step(self, *, timestamp_ns, cumulative_distance_m, perception):
        del cumulative_distance_m, perception
        return BreakupDecision(
            timestamp_ns,
            BreakupState.APPROACH_CLUSTER,
            0.1,
            0.0,
            GripperPosture.CLOSED,
            "approach_cluster",
        )


def make_sequence(
    *,
    config: Simulation20PointRuntimeConfig | None = None,
    breakup: object | None = None,
    confirmation_hits: int = 1,
) -> Simulation20PointSequence:
    regions = (
        StaticRegion(
            "field",
            RegionKind.FIELD,
            (
                FieldPoint(-2000.0, -2000.0),
                FieldPoint(2000.0, -2000.0),
                FieldPoint(2000.0, 2000.0),
                FieldPoint(-2000.0, 2000.0),
            ),
        ),
        StaticRegion(
            "own-material",
            RegionKind.OWN_MATERIAL,
            (
                FieldPoint(-300.0, 1200.0),
                FieldPoint(300.0, 1200.0),
                FieldPoint(300.0, 1600.0),
                FieldPoint(-300.0, 1600.0),
            ),
        ),
    )
    return Simulation20PointSequence(
        config or runtime_config(),
        tracker=MultiTargetTracker(
            TrackingConfig(confirmation_hits, 2000.0, 0.1, 1000.0, 0.1, 0.01)
        ),
        world_model=WorldModel(
            WorldModelConfig(500.0, 500.0, 0.6, 0.15, 0.5),
            regions,
        ),
        mission=MissionStateMachine(
            MissionConfig(
                1000.0,
                15.0,
                10.0,
                500.0,
                (
                    TargetClass.ORANGE_INJURED,
                    TargetClass.BLACK_CORE,
                    TargetClass.GREEN_SUPPLY,
                ),
            )
        ),
        breakup=breakup or _ImmediateGreenBreakup(),
        scan_direction="left",
    )


def pose(*, x: float = -1000.0, y: float = 0.0, heading: float = 0.0):
    return FusedPoseEstimate(
        pose=FieldPose2D(FieldPoint(x, y), heading),
        estimate_timestamp_ns=1,
        position_uncertainty_mm=20.0,
        heading_uncertainty_rad=0.02,
        confidence=0.9,
        anchor_source="configured_start",
        quality=frozenset(),
    )


def start_sequence(sequence: Simulation20PointSequence) -> None:
    sequence.preflight(
        0,
        SimulationPreflight(
            telemetry_fresh=True,
            watchdog_armed=True,
            emergency_stop_clear=True,
            zero_speed_command_accepted=True,
            camera_observation_fresh=True,
            observe_only_remote=True,
        ),
    )
    sequence.start(1)


def test_simulation_config_is_explicit_and_strict(tmp_path: Path) -> None:
    config = load_runtime_config("configs/runtime.simulation-20min.yaml")
    assert config.simulation_20_point.enabled
    assert config.simulation_20_point.target_delivery_count == 4
    assert config.remote.access_mode.value == "observe_only"
    assert config.build_simulation_20_point_sequence().state is Simulation20PointState.BOOT

    raw = yaml.safe_load(
        Path("configs/runtime.example.yaml").read_text(encoding="utf-8")
    )
    raw["simulation_20_point"] = {"unexpected": 1}
    invalid = tmp_path / "invalid-runtime.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys in simulation_20_point"):
        load_runtime_config(invalid)


def test_preflight_failure_latches_terminal_stop() -> None:
    sequence = make_sequence()
    decision = sequence.preflight(
        0,
        SimulationPreflight(True, False, True, True, True, True),
    )
    assert decision.state is Simulation20PointState.TERMINAL_STOP
    assert decision.linear_velocity_m_s == 0.0


def test_existing_breakup_is_followed_by_track_reset_and_full_scan() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    green = lambda frame, timestamp: snapshot(
        frame,
        timestamp,
        observation(frame, timestamp, GroundPoint(600.0, 0.0)),
    )

    reset = sequence.step(
        2,
        perception=green(1, 2),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert reset.state is Simulation20PointState.RESET_TARGET_TRACKS
    settle_1 = sequence.step(
        3,
        perception=green(2, 3),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    settle_2 = sequence.step(
        4,
        perception=green(3, 4),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert settle_1.state is Simulation20PointState.RESET_TARGET_TRACKS
    assert settle_2.state is Simulation20PointState.SCAN_GREEN

    scanning = sequence.step(
        5,
        perception=green(4, 5),
        pose=pose(heading=0.2),
        cumulative_distance_m=0.0,
    )
    assert scanning.state is Simulation20PointState.SCAN_GREEN
    evaluated = sequence.step(
        6,
        perception=green(5, 6),
        pose=pose(heading=0.8),
        cumulative_distance_m=0.0,
    )
    assert evaluated.state is Simulation20PointState.EVALUATE_EASY_GREEN
    selected = sequence.step(
        7,
        perception=green(6, 7),
        pose=pose(heading=0.8),
        cumulative_distance_m=0.0,
    )
    assert selected.state is Simulation20PointState.SELECT_GREEN
    assert selected.selected_track_id == 1


def test_unknown_or_dangerous_target_cannot_become_easy_green() -> None:
    sequence = make_sequence(
        config=runtime_config(scan_min_heading_span_rad=0.1),
    )
    start_sequence(sequence)
    dangerous = snapshot(
        1,
        2,
        observation(1, 2, GroundPoint(600.0, 0.0)),
        observation(
            1,
            2,
            GroundPoint(520.0, 50.0),
            target_class=TargetClass.BLUE_DANGER,
        ),
    )
    sequence.step(2, perception=dangerous, pose=pose(), cumulative_distance_m=0.0)
    sequence.step(
        3,
        perception=dangerous,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    sequence.step(
        4,
        perception=snapshot(
            2,
            4,
            observation(2, 4, GroundPoint(600.0, 0.0)),
            observation(
                2,
                4,
                GroundPoint(520.0, 50.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        ),
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    held = None
    for frame_sequence, (timestamp_ns, heading) in enumerate(
        ((5, 0.2), (6, 0.4), (7, 0.6), (8, 0.8), (9, 1.0)),
        start=3,
    ):
        current = snapshot(
            frame_sequence,
            timestamp_ns,
            observation(frame_sequence, timestamp_ns, GroundPoint(600.0, 0.0)),
            observation(
                frame_sequence,
                timestamp_ns,
                GroundPoint(520.0, 50.0),
                target_class=TargetClass.BLUE_DANGER,
            ),
        )
        decision = sequence.step(
            timestamp_ns,
            perception=current,
            pose=pose(heading=heading),
            cumulative_distance_m=0.0,
        )
        if decision.state is Simulation20PointState.SAFETY_HOLD:
            held = decision
            break
    assert held is not None
    assert held.state is Simulation20PointState.SAFETY_HOLD
    assert held.linear_velocity_m_s == 0.0


def test_side_path_failure_cannot_be_overridden_by_motion_request() -> None:
    sequence = make_sequence()
    start_sequence(sequence)
    decision = sequence.step(
        2,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.0,
        health=replace(SimulationHealth(), branch_error="slow_inference"),
    )
    assert decision.state is Simulation20PointState.TERMINAL_STOP
    assert decision.linear_velocity_m_s == 0.0


def test_breakup_safety_waits_for_track_confirmation_before_approach() -> None:
    sequence = make_sequence(
        breakup=_ApproachBreakup(),
        confirmation_hits=2,
    )
    start_sequence(sequence)
    first = snapshot(
        1,
        2,
        observation(1, 2, GroundPoint(500.0, 0.0)),
    )
    safety_check = sequence.step(
        2,
        perception=first,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert safety_check.state is Simulation20PointState.BREAKUP_SAFETY_CHECK
    assert safety_check.linear_velocity_m_s == 0.0

    confirmed = snapshot(
        2,
        3,
        observation(2, 3, GroundPoint(500.0, 0.0)),
    )
    approaching = sequence.step(
        3,
        perception=confirmed,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert approaching.state is Simulation20PointState.APPROACH_CLUSTER
    assert approaching.linear_velocity_m_s == pytest.approx(0.1)


def test_breakup_safety_hold_does_not_oscillate_back_into_breakup() -> None:
    sequence = make_sequence(breakup=_ApproachBreakup())
    start_sequence(sequence)
    danger_1 = snapshot(
        1,
        2,
        observation(
            1,
            2,
            GroundPoint(500.0, 0.0),
            target_class=TargetClass.BLUE_DANGER,
        ),
    )
    sequence.step(
        2,
        perception=danger_1,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    danger_2 = snapshot(
        2,
        3,
        observation(
            2,
            3,
            GroundPoint(500.0, 0.0),
            target_class=TargetClass.BLUE_DANGER,
        ),
    )
    held = sequence.step(
        3,
        perception=danger_2,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert held.state is Simulation20PointState.SAFETY_HOLD
    still_held = sequence.step(
        4,
        perception=danger_2,
        pose=pose(),
        cumulative_distance_m=0.0,
    )
    assert still_held.state is Simulation20PointState.SAFETY_HOLD
    assert still_held.linear_velocity_m_s == 0.0


def test_four_green_deliveries_latch_finish_stop_and_twenty_points() -> None:
    sequence = make_sequence(
        config=runtime_config(
            scan_min_heading_span_rad=0.5,
            delivery_confirm_frames=1,
            disengage_confirm_frames=1,
            retreat_distance_m=0.05,
            breakup_settle_confirm_frames=1,
        )
    )
    start_sequence(sequence)
    timestamp_ns = 1
    frame_sequence = 0

    def feed(
        field_point: FieldPoint,
        *,
        x: float = -1000.0,
        y: float = 0.0,
        heading: float = 0.0,
        distance_m: float = 0.0,
    ):
        nonlocal timestamp_ns, frame_sequence
        timestamp_ns += 10_000_000
        frame_sequence += 1
        return sequence.step(
            timestamp_ns,
            perception=snapshot(
                frame_sequence,
                timestamp_ns,
                observation(
                    frame_sequence,
                    timestamp_ns,
                    GroundPoint(
                        np.cos(heading) * (field_point.x - x)
                        + np.sin(heading) * (field_point.y - y),
                        -np.sin(heading) * (field_point.x - x)
                        + np.cos(heading) * (field_point.y - y),
                    ),
                ),
            ),
            pose=pose(x=x, y=y, heading=heading),
            cumulative_distance_m=distance_m,
        )

    for target_x in (-500.0, -200.0, 200.0, 500.0):
        target = FieldPoint(target_x, 0.0)
        if sequence.state is Simulation20PointState.LEAVE_START:
            assert feed(target).state is Simulation20PointState.RESET_TARGET_TRACKS
        while sequence.state is Simulation20PointState.RESET_TARGET_TRACKS:
            feed(target)
        assert sequence.state is Simulation20PointState.SCAN_GREEN
        feed(target, heading=0.0)
        assert feed(target, heading=0.6).state is Simulation20PointState.EVALUATE_EASY_GREEN
        assert feed(target).state is Simulation20PointState.SELECT_GREEN
        assert feed(target).state is Simulation20PointState.PLAN_PREPUSH
        assert feed(target).state is Simulation20PointState.NAVIGATE_PREPUSH

        plan = sequence._selected_plan
        assert plan is not None
        prepush_heading = np.arctan2(
            plan.prepush_field.y,
            plan.prepush_field.x + 1000.0,
        )
        assert (
            feed(target, heading=float(prepush_heading)).state
            is Simulation20PointState.NAVIGATE_PREPUSH
        )
        assert feed(
            target,
            x=plan.prepush_field.x,
            y=plan.prepush_field.y,
            heading=float(prepush_heading),
        ).state is Simulation20PointState.ALIGN_GREEN
        push_heading = float(np.arctan2(1400.0, -target_x))
        assert feed(
            target,
            x=plan.prepush_field.x,
            y=plan.prepush_field.y,
            heading=push_heading,
        ).state is Simulation20PointState.APPROACH_GREEN
        contact_x = target.x - float(np.cos(push_heading)) * 80.0
        contact_y = target.y - float(np.sin(push_heading)) * 80.0
        assert feed(
            target,
            x=contact_x,
            y=contact_y,
            heading=push_heading,
        ).state is Simulation20PointState.ENGAGE_GREEN
        assert feed(
            target,
            x=contact_x,
            y=contact_y,
            heading=push_heading,
        ).state is Simulation20PointState.PUSH_TO_MATERIAL_ZONE

        destination = FieldPoint(0.0, 1400.0)
        assert feed(
            destination,
            x=0.0,
            y=1000.0,
            heading=np.pi / 2.0,
        ).state is Simulation20PointState.VERIFY_DELIVERY
        assert feed(
            destination,
            x=0.0,
            y=1000.0,
            heading=np.pi / 2.0,
            distance_m=0.1,
        ).state is Simulation20PointState.DISENGAGE_AND_RETREAT
        feed(destination, x=0.0, y=1000.0, heading=np.pi / 2.0, distance_m=0.1)
        cleared = feed(
            destination,
            x=0.0,
            y=1000.0,
            heading=np.pi / 2.0,
            distance_m=0.2,
        )
        assert cleared.state is Simulation20PointState.UPDATE_PROGRESS
        progress = feed(
            destination,
            x=-1000.0,
            y=0.0,
            distance_m=0.2,
        )
        if target_x != 500.0:
            assert progress.state is Simulation20PointState.RESET_TARGET_TRACKS
        else:
            assert progress.state is Simulation20PointState.FINISH_STOP

    assert sequence.valid_green_deliveries == 4
    assert sequence.score_points == 20
    stopped = sequence.step(
        timestamp_ns + 1,
        perception=None,
        pose=pose(),
        cumulative_distance_m=0.2,
    )
    assert stopped.state is Simulation20PointState.FINISH_STOP
    assert stopped.linear_velocity_m_s == 0.0
