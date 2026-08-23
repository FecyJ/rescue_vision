"""Visual localization observations in the static competition field."""

from rescue_vision.localization.center_cross import CenterCrossLocalizer
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
    "angular_distance",
    "normalize_angle",
]
