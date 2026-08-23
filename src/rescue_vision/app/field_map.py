"""Render the configured static field map with fresh global observations."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.communication import MapSnapshotAttributes
from rescue_vision.communication.remote_observations import TeamColor as RemoteTeamColor
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossLocalizer,
    CenterCrossPoseObservation,
    FieldPose2D,
)
from rescue_vision.perception import FieldFeatureDetector
from rescue_vision.world.static_map import (
    PhysicalRegionKind,
    StaticFieldMap,
    TeamColor,
)


@dataclass(frozen=True, slots=True)
class MapTargetMarker:
    """A confirmed target that a future world-model publisher may overlay."""

    track_id: int
    position: FieldPoint
    label: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 0
        ):
            raise ValueError("track_id must be a non-negative integer.")
        if not isinstance(self.position, FieldPoint):
            raise ValueError("position must be a FieldPoint.")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class MapRobotPose:
    pose: FieldPose2D
    capture_timestamp_ns: int
    confidence: float
    position_uncertainty_mm: float
    heading_uncertainty_rad: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.pose, FieldPose2D):
            raise ValueError("pose must be a FieldPose2D.")
        if (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be a non-negative integer.")
        for name in (
            "confidence",
            "position_uncertainty_mm",
            "heading_uncertainty_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if self.position_uncertainty_mm <= 0.0:
            raise ValueError("position_uncertainty_mm must be positive.")
        if self.heading_uncertainty_rad <= 0.0:
            raise ValueError("heading_uncertainty_rad must be positive.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class EncodedMapSnapshot:
    png_bytes: bytes
    attributes: MapSnapshotAttributes


def _map_bounds(static_map: StaticFieldMap) -> tuple[float, float, float, float]:
    field_regions = tuple(
        region
        for region in static_map.regions
        if region.kind is PhysicalRegionKind.FIELD
    )
    regions = field_regions or static_map.regions
    points = tuple(point for region in regions for point in region.polygon_field)
    if not points:
        raise ValueError("Static map rendering requires at least one configured region.")
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Static map bounds must have positive width and height.")
    return min_x, max_x, min_y, max_y


class FieldMapSnapshotRenderer:
    """Encode the canonical static map and optional localized entities as PNG."""

    _COLORS = {
        PhysicalRegionKind.FIELD: (245, 245, 245),
        PhysicalRegionKind.RED_MATERIAL: (120, 170, 255),
        PhysicalRegionKind.RED_INJURED: (70, 110, 235),
        PhysicalRegionKind.BLUE_MATERIAL: (255, 190, 120),
        PhysicalRegionKind.BLUE_INJURED: (235, 130, 70),
        PhysicalRegionKind.START_ZONE: (230, 150, 230),
    }

    def __init__(
        self,
        static_map: StaticFieldMap,
        team_color: TeamColor,
        *,
        max_dimension_px: int = 800,
    ) -> None:
        if not isinstance(static_map, StaticFieldMap):
            raise ValueError("static_map must be a StaticFieldMap.")
        if not isinstance(team_color, TeamColor):
            raise ValueError("team_color must be a TeamColor.")
        if (
            isinstance(max_dimension_px, bool)
            or not isinstance(max_dimension_px, int)
            or max_dimension_px < 200
        ):
            raise ValueError("max_dimension_px must be an integer of at least 200.")
        min_x, max_x, min_y, max_y = _map_bounds(static_map)
        aspect = (max_x - min_x) / (max_y - min_y)
        if aspect >= 1.0:
            width = max_dimension_px
            height = max(200, round(max_dimension_px / aspect))
        else:
            height = max_dimension_px
            width = max(200, round(max_dimension_px * aspect))
        self._static_map = static_map
        self._sequence = 0
        self._attributes_base = dict(
            width=width,
            height=height,
            field_min_x_mm=min_x,
            field_max_x_mm=max_x,
            field_min_y_mm=min_y,
            field_max_y_mm=max_y,
            team_color=RemoteTeamColor(team_color.value),
        )
        self._base_image = self._draw_static_map()

    def _attributes(
        self,
        timestamp_ns: int,
        robot: MapRobotPose | None,
    ) -> MapSnapshotAttributes:
        return MapSnapshotAttributes(
            snapshot_sequence=self._sequence,
            timestamp_ns=timestamp_ns,
            **self._attributes_base,
            robot_localized=robot is not None,
            robot_x_mm=None if robot is None else robot.pose.position.x,
            robot_y_mm=None if robot is None else robot.pose.position.y,
            robot_heading_rad=None if robot is None else robot.pose.heading_rad,
            localization_capture_timestamp_ns=(
                None if robot is None else robot.capture_timestamp_ns
            ),
            localization_confidence=None if robot is None else robot.confidence,
            localization_position_uncertainty_mm=(
                None if robot is None else robot.position_uncertainty_mm
            ),
            localization_heading_uncertainty_rad=(
                None if robot is None else robot.heading_uncertainty_rad
            ),
            localization_source=None if robot is None else robot.source,
        )

    def _pixel(
        self,
        attributes: MapSnapshotAttributes,
        point: FieldPoint,
    ) -> tuple[int, int]:
        pixel = attributes.field_to_map_pixel(point)
        return round(pixel.u), round(pixel.v)

    def _draw_static_map(self) -> np.ndarray:
        attributes = self._attributes(0, None)
        image = np.full((attributes.height, attributes.width, 3), 255, np.uint8)
        ordered = sorted(
            self._static_map.regions,
            key=lambda item: item.kind is not PhysicalRegionKind.FIELD,
        )
        for region in ordered:
            polygon = np.asarray(
                [self._pixel(attributes, point) for point in region.polygon_field],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [polygon], self._COLORS[region.kind])
            cv2.polylines(image, [polygon], True, (70, 70, 70), 2, cv2.LINE_AA)
        center = self._pixel(attributes, self._static_map.center_cross.intersection_field)
        arm = max(12, min(attributes.width, attributes.height) // 30)
        cv2.line(
            image,
            (center[0] - arm, center[1]),
            (center[0] + arm, center[1]),
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            (center[0], center[1] - arm),
            (center[0], center[1] + arm),
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        return image

    def render(
        self,
        *,
        timestamp_ns: int,
        robot: MapRobotPose | None = None,
        targets: tuple[MapTargetMarker, ...] = (),
    ) -> EncodedMapSnapshot:
        attributes = self._attributes(timestamp_ns, robot)
        image = self._base_image.copy()
        for target in targets:
            if not isinstance(target, MapTargetMarker):
                raise ValueError("targets must contain MapTargetMarker values.")
            center = self._pixel(attributes, target.position)
            cv2.circle(image, center, 7, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.putText(
                image,
                target.label,
                (center[0] + 9, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
        if robot is not None:
            center = self._pixel(attributes, robot.pose.position)
            scale_x = (attributes.width - 1) / (attributes.field_max_x_mm - attributes.field_min_x_mm)
            scale_y = (attributes.height - 1) / (attributes.field_max_y_mm - attributes.field_min_y_mm)
            radius = max(
                4,
                round(
                    robot.position_uncertainty_mm
                    * (scale_x + scale_y)
                    / 2.0
                ),
            )
            cv2.circle(image, center, radius, (0, 170, 0), 1, cv2.LINE_AA)
            length = max(18, min(attributes.width, attributes.height) // 18)
            tip = (
                round(center[0] + length * math.cos(robot.pose.heading_rad)),
                round(center[1] - length * math.sin(robot.pose.heading_rad)),
            )
            cv2.arrowedLine(
                image,
                center,
                tip,
                (0, 120, 0),
                4,
                cv2.LINE_AA,
                tipLength=0.35,
            )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("OpenCV failed to encode the field-map PNG.")
        self._sequence += 1
        return EncodedMapSnapshot(encoded.tobytes(), attributes)


class LatestCenterCrossLocalization:
    """Run center-cross localization on a bounded latest-frame side path."""

    def __init__(
        self,
        detector: FieldFeatureDetector,
        localizer: CenterCrossLocalizer,
        *,
        valid_mask: np.ndarray,
        max_pose_age_ms: float,
    ) -> None:
        self._detector = detector
        self._localizer = localizer
        if (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.dtype != np.uint8
            or valid_mask.ndim != 2
            or not np.any(valid_mask)
            or np.any((valid_mask != 0) & (valid_mask != 255))
        ):
            raise ValueError(
                "valid_mask must be a non-empty uint8 2D array containing only "
                "0 and 255."
            )
        self._valid_mask = valid_mask.copy()
        age_ms = float(max_pose_age_ms)
        if not math.isfinite(age_ms):
            raise ValueError("max_pose_age_ms must be finite.")
        self._max_pose_age_ns = round(age_ms * 1_000_000)
        if self._max_pose_age_ns <= 0:
            raise ValueError("max_pose_age_ms must be positive.")
        self._event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: CameraFrame | None = None
        self._latest: CenterCrossPoseObservation | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("LatestCenterCrossLocalization is already started.")
        self._stop.clear()
        self._event.clear()
        with self._lock:
            self._pending = None
            self._latest = None
            self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="rescue-map-localization",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: CameraFrame) -> None:
        self._raise_error()
        if self._thread is None:
            raise RuntimeError("LatestCenterCrossLocalization is not started.")
        with self._lock:
            self._pending = frame
        self._event.set()

    def latest_robot_pose(self, current_timestamp_ns: int) -> MapRobotPose | None:
        self._raise_error()
        if (
            isinstance(current_timestamp_ns, bool)
            or not isinstance(current_timestamp_ns, int)
            or current_timestamp_ns < 0
        ):
            raise ValueError("current_timestamp_ns must be a non-negative integer.")
        with self._lock:
            observation = self._latest
        if (
            observation is None
            or observation.selected_pose is None
            or current_timestamp_ns < observation.capture_timestamp_ns
            or current_timestamp_ns - observation.capture_timestamp_ns
            > self._max_pose_age_ns
        ):
            return None
        candidate = min(
            observation.candidates,
            key=lambda item: abs(item.pose.heading_rad - observation.selected_pose.heading_rad),
        )
        assert observation.selection_source is not None
        return MapRobotPose(
            observation.selected_pose,
            observation.capture_timestamp_ns,
            observation.confidence,
            candidate.position_uncertainty_mm,
            candidate.heading_uncertainty_rad,
            observation.selection_source.value,
        )

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        self._event.set()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError(
                "LatestCenterCrossLocalization worker did not stop."
            )
        self._thread = None
        self._raise_error()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Center-cross map localization failed.") from self._error

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._event.wait(0.1)
                self._event.clear()
                with self._lock:
                    frame = self._pending
                    self._pending = None
                if frame is None:
                    continue
                realtime_result = self._detector.detect_realtime(
                    frame,
                    frame.image_bgr,
                    valid_mask=self._valid_mask,
                )
                result = realtime_result.result
                if result is None:
                    # A slow field-feature pass is an expected real-time drop,
                    # not a worker fault.  Clear any previous pose so the map
                    # cannot keep displaying stale localization.
                    with self._lock:
                        self._latest = None
                    continue
                observation = self._localizer.localize(result)
                with self._lock:
                    self._latest = observation
        except BaseException as exc:
            self._error = exc
            self._stop.set()
