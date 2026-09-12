"""同帧 v3 场地地标到连续融合器的单次消费管线。"""

from __future__ import annotations

from typing import Protocol

from rescue_vision.localization.center_cross import CenterCrossLocalizer
from rescue_vision.localization.fusion import FusedPoseEstimate, VisualFusionResult
from rescue_vision.localization.types import CenterCrossSelectionSource
from rescue_vision.localization.static_landmarks import (
    SafeZoneCornerLocalizer,
    StaticFieldLandmarkTracker,
    select_same_frame_pose_observation,
)
from rescue_vision.perception.field_feature_types import FieldFeatureDetectionResult


class _FusionSink(Protocol):
    def pose_at(self, timestamp_ns: int) -> FusedPoseEstimate: ...
    def submit_visual(self, observation: object) -> VisualFusionResult: ...
    def submit_position_landmark(self, observation: object) -> VisualFusionResult: ...


class VisualLocalizationPipeline:
    """保证每个模型帧至多提交一次视觉更新。"""

    def __init__(
        self,
        *,
        center_cross_localizer: CenterCrossLocalizer,
        safe_zone_localizer: SafeZoneCornerLocalizer,
        landmark_tracker: StaticFieldLandmarkTracker,
        fusion: _FusionSink,
    ) -> None:
        self._center = center_cross_localizer
        self._safe_zone = safe_zone_localizer
        self._tracker = landmark_tracker
        self._fusion = fusion
        self._last_frame_sequence: int | None = None

    @property
    def last_frame_sequence(self) -> int | None:
        return self._last_frame_sequence

    def submit(self, result: FieldFeatureDetectionResult) -> VisualFusionResult | None:
        if not isinstance(result, FieldFeatureDetectionResult):
            raise TypeError("result must be a FieldFeatureDetectionResult.")
        if self._last_frame_sequence is not None and result.frame_sequence <= self._last_frame_sequence:
            return None
        self._last_frame_sequence = result.frame_sequence
        prior_estimate = self._fusion.pose_at(result.capture_timestamp_ns)
        prior = prior_estimate.pose
        confirmed = self._tracker.confirm_center_cross(result)
        self._tracker.update(confirmed, pose=prior)
        safe_observation = self._safe_zone.localize(
            confirmed, prior_pose=prior
        ).observation
        center_observation = self._center.localize(confirmed, prior_pose=prior)
        selected_cross, selected_safe_zone = select_same_frame_pose_observation(
            center_observation,
            safe_observation,
            prior_pose=prior,
        )
        selected = selected_safe_zone or (
            selected_cross
            if (
                selected_cross is not None
                and selected_cross.selected_pose is not None
                and selected_cross.selection_source
                is not CenterCrossSelectionSource.PRIOR
            )
            else None
        )
        if selected is not None:
            return self._fusion.submit_visual(selected)
        if prior is None:
            return None
        position = self._center.localize_position(confirmed, prior_pose=prior)
        if position is None:
            return None
        return self._fusion.submit_position_landmark(position)
