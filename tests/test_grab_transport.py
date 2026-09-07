from __future__ import annotations

import math

import numpy as np

from rescue_vision.app import (
    GripperPosture,
    GrabTransportSequence,
    MatchState,
    MatchPreflight,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    PerceptionSnapshot,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)


CONFIG_PATH = "configs/runtime.match.yaml"


def green_observation(
    frame_sequence: int,
    timestamp_ns: int,
    ground: GroundPoint,
    *,
    box_x: float = 10.0,
) -> TargetObservation:
    box = UndistortedBoundingBox(box_x, 10.0, box_x + 10.0, 20.0)
    segmentation = RoiColorSegmentation(
        candidate_class=TargetClass.GREEN_SUPPLY,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=box,
        mask=np.full((10, 10), 255, dtype=np.uint8),
        color_fraction=1.0,
        dominance=1.0,
    )
    return TargetObservation(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        image_size=(100, 100),
        model_target_class=TargetClass.GREEN_SUPPLY,
        target_class=TargetClass.GREEN_SUPPLY,
        class_probabilities=ClassProbabilities.from_top_class(
            TargetClass.GREEN_SUPPLY,
            1.0,
        ),
        detection_confidence=0.95,
        box=box,
        color_segmentation=segmentation,
        k0=UndistortedPixel(box_x + 5.0, 15.0),
        k0_confidence=0.95,
        ground_point=ground,
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
        field_features=None,
    )


def make_flow() -> GrabTransportSequence:
    config = load_runtime_config(CONFIG_PATH)
    return config.build_grab_transport_sequence()


def start_flow(
    flow: GrabTransportSequence,
) -> None:
    flow.preflight(
        0,
        MatchPreflight(True, True, True, True, True, True),
    )
    flow.start(1)


def test_direct_flow_starts_at_origin_and_skips_startup_motion() -> None:
    flow = make_flow()

    assert flow.estimated_field_position is not None
    assert flow.estimated_field_position.x == 0.0
    assert flow.estimated_field_position.y == 0.0
    assert flow.INITIAL_HEADING_RAD == math.pi / 2.0
    start_flow(flow)

    assert flow.state is MatchState.SEARCH_CLUSTER
    first_search = flow.step(
        2,
        perception=None,
        heading_rad=math.pi / 2.0,
        cumulative_distance_m=0.0,
    )

    assert first_search.state is MatchState.SEARCH_CLUSTER
    assert first_search.linear_velocity_m_s == 0.0
    assert first_search.angular_velocity_rad_s < 0.0
    assert first_search.gripper_posture is GripperPosture.CLOSED


def test_direct_flow_sends_a_confirmed_single_green_to_transport() -> None:
    flow = make_flow()
    start_flow(flow)

    for frame_sequence in (1, 2):
        decision = flow.step(
            frame_sequence + 1,
            perception=snapshot(
                frame_sequence,
                frame_sequence + 1,
                green_observation(
                    frame_sequence,
                    frame_sequence + 1,
                    GroundPoint(600.0, 0.0),
                ),
            ),
            heading_rad=math.pi / 2.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.selected_track_id is not None
    assert decision.gripper_posture is GripperPosture.TRANSPORT


def test_direct_flow_does_not_fall_back_to_breakup_for_a_cluster() -> None:
    flow = make_flow()
    start_flow(flow)

    for frame_sequence in (1, 2):
        decision = flow.step(
            frame_sequence + 1,
            perception=snapshot(
                frame_sequence,
                frame_sequence + 1,
                green_observation(
                    frame_sequence,
                    frame_sequence + 1,
                    GroundPoint(600.0, 0.0),
                    box_x=10.0,
                ),
                green_observation(
                    frame_sequence,
                    frame_sequence + 1,
                    GroundPoint(620.0, 0.0),
                    box_x=30.0,
                ),
            ),
            heading_rad=math.pi / 2.0,
            cumulative_distance_m=0.0,
        )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.linear_velocity_m_s == 0.0
    assert decision.reason.startswith("search_cluster_")
