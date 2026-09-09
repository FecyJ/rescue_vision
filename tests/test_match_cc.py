from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from rescue_vision.app.match_cc import (
    find_cc_clusters,
    isolated_targets,
    point_to_origin_segment_distance_mm,
)
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    PerceptionSnapshot,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.config import load_runtime_config
from rescue_vision.app.match_cc import MatchCCSequence
from rescue_vision.app.match import MatchState


CC_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "runtime.cc.yaml"


def target(track_id: int, x_mm: float, y_mm: float, target_class=TargetClass.GREEN_SUPPLY):
    return SimpleNamespace(
        track_id=track_id,
        ground_point=GroundPoint(x_mm, y_mm),
        target_class=target_class,
    )


def snapshot(frame: int, timestamp_ns: int, points: tuple[GroundPoint, ...]) -> PerceptionSnapshot:
    observations = []
    for index, point in enumerate(points):
        left = 5.0 + index * 20.0
        box = UndistortedBoundingBox(left, 5.0, left + 10.0, 15.0)
        segmentation = RoiColorSegmentation(
            candidate_class=TargetClass.GREEN_SUPPLY,
            status=ColorSegmentationStatus.ACCEPTED,
            roi_box=box,
            mask=np.full((10, 10), 255, dtype=np.uint8),
            color_fraction=1.0,
            dominance=1.0,
        )
        observations.append(
            TargetObservation(
                frame_sequence=frame,
                capture_timestamp_ns=timestamp_ns,
                result_timestamp_ns=timestamp_ns,
                image_size=(100, 100),
                model_target_class=TargetClass.GREEN_SUPPLY,
                target_class=TargetClass.GREEN_SUPPLY,
                class_probabilities=ClassProbabilities.from_top_class(
                    TargetClass.GREEN_SUPPLY, 1.0
                ),
                detection_confidence=0.95,
                box=box,
                color_segmentation=segmentation,
                k0=UndistortedPixel(left + 5.0, 10.0),
                k0_confidence=0.95,
                ground_point=point,
                quality=frozenset(),
            )
        )
    return PerceptionSnapshot(frame, timestamp_ns, timestamp_ns, tuple(observations), None)


def test_cc_cluster_requires_every_member_to_have_two_close_neighbours() -> None:
    tracks = (
        target(1, 500.0, 0.0),
        target(2, 570.0, 0.0),
        target(3, 535.0, 60.0),
        target(4, 900.0, 0.0),
    )

    clusters = find_cc_clusters(tracks, neighbor_distance_mm=100.0)

    assert tuple(cluster.member_ids for cluster in clusters) == ((1, 2, 3),)


def test_cc_cluster_rejects_three_blocks_without_a_two_neighbour_anchor() -> None:
    tracks = (
        target(1, 0.0, 0.0),
        target(2, 90.0, 0.0),
        target(3, 201.0, 0.0),
    )

    assert find_cc_clusters(tracks, neighbor_distance_mm=100.0) == ()


def test_isolation_uses_distance_to_finite_origin_target_segment() -> None:
    selected = target(1, 300.0, 0.0)
    behind_target = target(2, 380.0, 0.0, TargetClass.BLACK_CORE)

    assert point_to_origin_segment_distance_mm(
        behind_target.ground_point, selected.ground_point
    ) == 80.0
    assert isolated_targets(
        (selected, behind_target),
        allowed_classes=frozenset((TargetClass.GREEN_SUPPLY,)),
        clearance_for_target=lambda _target: 30.0,
    ) == (selected,)


def test_isolation_rejects_any_other_graspable_block_in_corridor() -> None:
    selected = target(1, 300.0, 0.0)
    blocker = target(2, 150.0, 25.0, TargetClass.BLUE_DANGER)

    assert isolated_targets(
        (selected, blocker),
        allowed_classes=frozenset((TargetClass.GREEN_SUPPLY,)),
        clearance_for_target=lambda _target: 30.0,
    ) == ()


def test_runtime_cc_copies_only_shared_original_breakup_values_into_match() -> None:
    raw = yaml.safe_load(CC_CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw["match_cc"]["cluster_neighbor_distance_mm"] == 100.0
    assert raw["match"]["cluster_group_ground_mm"] == 100.0
    assert raw["match"]["cluster_min_detections"] == 3
    assert raw["match"]["cluster_align_hold_ms"] == 1000.0
    assert raw["match"]["breakup_forward_distance_m"] == 0.4
    assert raw["match"]["breakup_backward_distance_m"] == 0.4
    assert "green_alignment_tolerance_mm" not in raw["match"]
    assert set(raw["near_field_grasp"]) == {"clearance_mm", "min_mask_pixels"}
    assert load_runtime_config(CC_CONFIG_PATH).match_cc.enabled


def test_area_three_keeps_cc_orange_destination_absolute_x_positive() -> None:
    sequence = MatchCCSequence.from_app_config(
        load_runtime_config(CC_CONFIG_PATH),
        start_area="3",
    )

    destination = sequence.config.safe_zone_injured_target_field
    assert destination.x > 0.0
    assert destination.y < 0.0


def test_cc_sequence_uses_original_five_frame_reference_without_id_locking() -> None:
    sequence = MatchCCSequence.from_app_config(load_runtime_config(CC_CONFIG_PATH))
    sequence._started = True
    sequence._initial_field_position = FieldPoint(0.0, 0.0)
    sequence._fallback_field_position = FieldPoint(0.0, 0.0)
    sequence.state = MatchState.SEARCH_CLUSTER
    points = (
        GroundPoint(500.0, 0.0),
        GroundPoint(570.0, 0.0),
        GroundPoint(535.0, 60.0),
    )

    sequence.step(1_000_000_000, perception=snapshot(1, 1_000_000_000, points), heading_rad=0.0, cumulative_distance_m=0.0)
    found = sequence.step(1_010_000_000, perception=snapshot(2, 1_010_000_000, points), heading_rad=0.0, cumulative_distance_m=0.0)
    assert found.state is MatchState.ALIGN_CLUSTER_ONCE

    aligned_points = (
        GroundPoint(500.0, -5.0),
        GroundPoint(570.0, 0.0),
        GroundPoint(535.0, 5.0),
    )
    decision = found
    for frame in range(3, 7):
        timestamp_ns = 1_000_000_000 + frame * 10_000_000
        decision = sequence.step(
            timestamp_ns,
            perception=snapshot(frame, timestamp_ns, aligned_points),
            heading_rad=0.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.APPROACH_CLUSTER
    assert sequence._cluster_reference_distance_mm == pytest.approx(500.0)


def test_cc_open_command_precedes_backward_motion_by_twenty_ms() -> None:
    sequence = MatchCCSequence.from_app_config(load_runtime_config(CC_CONFIG_PATH))
    sequence._started = True
    sequence.state = MatchState.CC_BREAKUP_OPEN_GAP
    sequence._cc_command_ns = 1_000_000_000
    sequence._cc_motion_distance_m = 0.5

    waiting = sequence.step(
        1_019_000_000,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert waiting.state is MatchState.CC_BREAKUP_OPEN_GAP
    assert waiting.linear_velocity_m_s == 0.0

    released = sequence.step(
        1_020_000_000,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert released.state is MatchState.BREAKUP_BACKWARD
    moving = sequence.step(
        1_021_000_000,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert moving.linear_velocity_m_s == -0.3


def test_cc_reference_collection_restarts_search_after_one_second_without_cluster() -> None:
    sequence = MatchCCSequence.from_app_config(load_runtime_config(CC_CONFIG_PATH))
    sequence._started = True
    sequence.state = MatchState.ALIGN_CLUSTER_ONCE
    sequence._cluster_capture_heading_rad = 0.0

    waiting = sequence.step(
        1,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert waiting.reason == "cluster_collecting_reference_waiting_for_fresh_cluster"

    restarted = sequence.step(
        1_000_000_001,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert restarted.state is MatchState.SEARCH_CLUSTER
    assert restarted.reason == "cluster_reference_collection_timeout_restart_search"
