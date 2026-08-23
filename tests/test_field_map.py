from __future__ import annotations

import time

import cv2
import numpy as np

from rescue_vision.app.field_map import (
    FieldMapSnapshotRenderer,
    LatestCenterCrossLocalization,
    MapRobotPose,
    MapTargetMarker,
)
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossPoseCandidate,
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    FieldPose2D,
)
from rescue_vision.world import (
    PhysicalRegionKind,
    PhysicalStaticRegion,
    TeamColor,
    default_static_field_map,
)
from rescue_vision.world.static_map import StaticFieldMap


def static_map() -> StaticFieldMap:
    landmarks = default_static_field_map()
    return StaticFieldMap(
        landmarks.center_cross,
        (
            PhysicalStaticRegion(
                "field",
                PhysicalRegionKind.FIELD,
                (
                    FieldPoint(-1500.0, -1500.0),
                    FieldPoint(1500.0, -1500.0),
                    FieldPoint(1500.0, 1500.0),
                    FieldPoint(-1500.0, 1500.0),
                ),
            ),
        ),
    )


def test_field_map_snapshot_encodes_static_map_pose_and_future_target() -> None:
    renderer = FieldMapSnapshotRenderer(static_map(), TeamColor.RED, max_dimension_px=400)
    robot = MapRobotPose(
        FieldPose2D(FieldPoint(100.0, 200.0), 0.5),
        capture_timestamp_ns=900,
        confidence=0.8,
        position_uncertainty_mm=40.0,
        heading_uncertainty_rad=0.1,
        source="red_safe_zone",
    )

    snapshot = renderer.render(
        timestamp_ns=1000,
        robot=robot,
        targets=(MapTargetMarker(7, FieldPoint(-200.0, 300.0), "green_supply"),),
    )

    decoded = cv2.imdecode(np.frombuffer(snapshot.png_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (400, 400, 3)
    assert snapshot.attributes.robot_localized
    assert snapshot.attributes.robot_x_mm == 100.0
    assert snapshot.attributes.robot_y_mm == 200.0
    assert snapshot.attributes.robot_heading_rad == 0.5
    assert snapshot.attributes.localization_capture_timestamp_ns == 900
    assert snapshot.attributes.localization_source == "red_safe_zone"
    assert snapshot.attributes.map_pixel_to_field(
        snapshot.attributes.field_to_map_pixel(FieldPoint(100.0, 200.0))
    ) == FieldPoint(100.0, 200.0)


def test_field_map_without_unique_pose_explicitly_reports_unlocalized() -> None:
    snapshot = FieldMapSnapshotRenderer(static_map(), TeamColor.UNKNOWN).render(
        timestamp_ns=1000
    )

    assert not snapshot.attributes.robot_localized
    assert snapshot.attributes.robot_x_mm is None
    assert snapshot.attributes.localization_capture_timestamp_ns is None


class FakeDetector:
    def detect(self, frame, image, *, valid_mask):
        del image, valid_mask
        return frame


class FakeLocalizer:
    def localize(self, frame):
        pose = FieldPose2D(FieldPoint(20.0, -30.0), 0.25)
        candidate = CenterCrossPoseCandidate(pose, 0, 25.0, 0.05)
        candidates = (candidate,) + tuple(
            CenterCrossPoseCandidate(
                FieldPose2D(FieldPoint(float(index), 0.0), index * 0.5),
                index,
                25.0,
                0.05,
            )
            for index in range(1, 4)
        )
        return CenterCrossPoseObservation(
            frame_sequence=frame.sequence,
            capture_timestamp_ns=frame.timestamp_ns,
            result_timestamp_ns=frame.timestamp_ns,
            candidates=candidates,
            terminals=(),
            selected_pose=pose,
            selection_source=CenterCrossSelectionSource.PRIOR,
            confidence=0.7,
            quality=frozenset(),
        )


def test_latest_localization_drops_stale_pose() -> None:
    localizer = LatestCenterCrossLocalization(
        FakeDetector(),  # type: ignore[arg-type]
        FakeLocalizer(),  # type: ignore[arg-type]
        valid_mask=np.full((3, 4), 255, np.uint8),
        max_pose_age_ms=10.0,
    )
    frame = CameraFrame(2, 1_000_000_000, np.zeros((3, 4, 3), np.uint8))
    localizer.start()
    try:
        localizer.submit(frame)
        deadline = time.monotonic() + 1.0
        robot = None
        while robot is None and time.monotonic() < deadline:
            robot = localizer.latest_robot_pose(frame.timestamp_ns)
            time.sleep(0.005)
        assert robot is not None
        assert robot.pose.position == FieldPoint(20.0, -30.0)
        assert localizer.latest_robot_pose(frame.timestamp_ns + 10_000_001) is None
    finally:
        localizer.stop()
