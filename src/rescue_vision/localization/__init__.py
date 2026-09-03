"""Visual localization observations in the static competition field."""

from rescue_vision.localization.center_cross import CenterCrossLocalizer
from rescue_vision.localization.fusion import (
    FusedPoseEstimate,
    FusionConfig,
    FusionQuality,
    ImuFrameCalibration,
    OdometryCalibration,
    OdometryImuFusion,
    VisualAnchorHealth,
    VisualFusionResult,
)
from rescue_vision.localization.static_landmarks import (
    SafeZoneCornerLocalizer,
    SafeZoneCornerLocalizerConfig,
    SafeZoneCornerPoseObservation,
    StaticFieldLandmarkTracker,
    StaticLandmarkTrack,
    StaticLandmarkTrackingConfig,
    select_same_frame_pose_observation,
)
from rescue_vision.localization.types import (
    CenterCrossLocalizationQuality,
    CenterCrossLocalizerConfig,
    CenterCrossPoseCandidate,
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    CenterLineTerminalKind,
    CenterLineTerminalObservation,
    FieldPose2D,
    FieldPositionObservation,
    angular_distance,
    normalize_angle,
)
from rescue_vision.localization.visual_pipeline import VisualLocalizationPipeline

__all__ = [
    "CenterCrossLocalizationQuality",
    "CenterCrossLocalizer",
    "CenterCrossLocalizerConfig",
    "CenterCrossPoseCandidate",
    "CenterCrossPoseObservation",
    "CenterCrossSelectionSource",
    "CenterLineTerminalKind",
    "CenterLineTerminalObservation",
    "FieldPose2D",
    "FieldPositionObservation",
    "FusedPoseEstimate",
    "FusionConfig",
    "FusionQuality",
    "ImuFrameCalibration",
    "OdometryCalibration",
    "OdometryImuFusion",
    "SafeZoneCornerLocalizer",
    "SafeZoneCornerLocalizerConfig",
    "SafeZoneCornerPoseObservation",
    "StaticFieldLandmarkTracker",
    "StaticLandmarkTrack",
    "StaticLandmarkTrackingConfig",
    "VisualAnchorHealth",
    "VisualFusionResult",
    "VisualLocalizationPipeline",
    "angular_distance",
    "normalize_angle",
    "select_same_frame_pose_observation",
]
