from __future__ import annotations

from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossPoseObservation,
    FieldPose2D,
    FieldPositionObservation,
    FusedPoseEstimate,
    FusionQuality,
    VisualFusionResult,
    VisualLocalizationPipeline,
)
from rescue_vision.perception import FieldFeatureDetectionResult


def empty_result(sequence: int) -> FieldFeatureDetectionResult:
    return FieldFeatureDetectionResult(
        sequence,
        1_000_000 + sequence,
        1_100_000 + sequence,
        (640, 480),
        (),
        None,
    )


class Tracker:
    def confirm_center_cross(self, result):
        return result

    def update(self, result, *, pose=None):
        del result, pose


class Center:
    def localize(self, result, *, prior_pose=None):
        del prior_pose
        return CenterCrossPoseObservation(
            result.frame_sequence,
            result.capture_timestamp_ns,
            result.result_timestamp_ns,
            (),
            (),
            None,
            None,
            0.0,
            frozenset(),
        )

    def localize_position(self, result, *, prior_pose):
        del prior_pose
        return FieldPositionObservation(
            result.frame_sequence,
            result.capture_timestamp_ns,
            result.result_timestamp_ns,
            FieldPoint(10.0, 20.0),
            25.0,
            0.8,
            "center_cross_position",
        )


class SafeZone:
    def localize(self, result, *, prior_pose=None):
        del result, prior_pose
        return None


class Fusion:
    def __init__(self) -> None:
        self.position_submissions = 0

    def pose_at(self, timestamp_ns):
        return FusedPoseEstimate(
            FieldPose2D(FieldPoint(0.0, 0.0), 0.0),
            timestamp_ns,
            10.0,
            0.1,
            0.8,
            "initial",
            frozenset({FusionQuality.FUSED}),
        )

    def submit_visual(self, observation):
        raise AssertionError(f"unexpected full-pose observation {observation!r}")

    def submit_position_landmark(self, observation):
        self.position_submissions += 1
        return VisualFusionResult(True, 0, 0.0)


def test_pipeline_submits_each_frame_at_most_once() -> None:
    fusion = Fusion()
    pipeline = VisualLocalizationPipeline(
        center_cross_localizer=Center(),
        safe_zone_localizer=SafeZone(),
        landmark_tracker=Tracker(),
        fusion=fusion,
    )

    assert pipeline.submit(empty_result(1)).accepted
    assert pipeline.submit(empty_result(1)) is None
    assert pipeline.submit(empty_result(0)) is None
    assert fusion.position_submissions == 1
