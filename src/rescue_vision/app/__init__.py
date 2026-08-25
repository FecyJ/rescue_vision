"""可运行应用装配入口。"""

from rescue_vision.app.field_map import (
    LatestCenterCrossLocalization,
    MapRobotPose,
)
from rescue_vision.app.manual_capture import run_manual_capture_session

__all__ = [
    "LatestCenterCrossLocalization",
    "MapRobotPose",
    "run_manual_capture_session",
]
