"""Visual localization observations in the static competition field."""

from rescue_vision.localization.center_cross import CenterCrossLocalizer
from rescue_vision.localization.fusion import (
    FusedPoseEstimate,
    FusionConfig,
    FusionQuality,
    ImuFrameCalibration,
    OdometryCalibration,
    OdometryImuFusion,
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
    angular_distance,
    normalize_angle,
)

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
    "VisualFusionResult",
    "angular_distance",
    "normalize_angle",
    "select_same_frame_pose_observation",
]
