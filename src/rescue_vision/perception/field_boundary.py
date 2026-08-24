"""时序局部场界估计、三态掩膜与保守场外过滤。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.perception.field_feature_types import (
    BoundaryFeatureKind,
    FieldFeatureDetectionResult,
)
from rescue_vision.world.static_map import PhysicalRegionKind, StaticFieldMap


Uint8Array = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]


class FieldMaskState(IntEnum):
    """三态场界掩膜的稳定整数编码。"""

    OUTSIDE = 0
    UNCERTAIN = 127
    INSIDE = 255


@dataclass(frozen=True, slots=True)
class FieldBoundaryConfig:
    """局部场界拟合、时序确认和掩膜生成参数。"""

    enabled: bool
    hard_mask_enabled: bool
    min_candidate_confidence: float
    min_confirmations: int
    max_missed_frames: int
    line_angle_tolerance_deg: float
    line_distance_tolerance_mm: float
    ransac_inlier_distance_mm: float
    rectangle_tolerance_fraction: float
    boundary_band_mm: float
    segment_extension_mm: float
    min_filter_confidence: float
    max_mask_age_ms: float
    mask_blur_radius_px: int
    neutral_fill_bgr: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(
            self.hard_mask_enabled, bool
        ):
            raise ValueError("field boundary enabled flags must be booleans.")
        for name in (
            "min_candidate_confidence",
            "rectangle_tolerance_fraction",
            "min_filter_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        for name in ("min_confirmations", "max_missed_frames"):
            value = getattr(self, name)
            minimum = 1 if name == "min_confirmations" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in (
            "line_angle_tolerance_deg",
            "line_distance_tolerance_mm",
            "ransac_inlier_distance_mm",
            "boundary_band_mm",
            "segment_extension_mm",
            "max_mask_age_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.line_angle_tolerance_deg >= 45.0:
            raise ValueError("line_angle_tolerance_deg must be less than 45 degrees.")
        if (
            isinstance(self.mask_blur_radius_px, bool)
            or not isinstance(self.mask_blur_radius_px, int)
            or self.mask_blur_radius_px < 0
        ):
            raise ValueError("mask_blur_radius_px must be a non-negative integer.")
        if (
            not isinstance(self.neutral_fill_bgr, tuple)
            or len(self.neutral_fill_bgr) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in self.neutral_fill_bgr
            )
        ):
            raise ValueError("neutral_fill_bgr must contain three integers in [0, 255].")


@dataclass(frozen=True, slots=True)
class LocalFieldBoundary:
    """机器人地面系中的有限场界线段及指向场内的单位法向。"""

    start_ground: GroundPoint
    end_ground: GroundPoint
    interior_normal_ground: tuple[float, float]
    line_offset_mm: float
    capture_timestamp_ns: int
    confidence: float
    confirmations: int

    def __post_init__(self) -> None:
        if self.start_ground == self.end_ground:
            raise ValueError("local field boundary endpoints must be distinct.")
        nx, ny = self.interior_normal_ground
        if not all(math.isfinite(value) for value in (nx, ny, self.line_offset_mm)):
            raise ValueError("local field boundary line parameters must be finite.")
        if not math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-6):
            raise ValueError("interior_normal_ground must be a unit vector.")
        for point in (self.start_ground, self.end_ground):
            residual = nx * point.x + ny * point.y + self.line_offset_mm
            if not math.isclose(residual, 0.0, abs_tol=1e-5):
                raise ValueError("boundary endpoints must lie on the configured line.")
        if self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be non-negative.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if self.confirmations <= 0:
            raise ValueError("confirmations must be positive.")


@dataclass(frozen=True, slots=True)
class FieldBoundaryMask:
    """与一帧去畸变图和对应 BEV 对齐的只读三态掩膜。"""

    capture_timestamp_ns: int
    expires_timestamp_ns: int
    image_state: Uint8Array
    bev_state: Uint8Array
    boundaries: tuple[LocalFieldBoundary, ...]
    confidence: float
    filter_ready: bool
    hard_mask_ready: bool
    neutral_fill_bgr: tuple[int, int, int]
    blur_radius_px: int

    def __post_init__(self) -> None:
        if self.expires_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError("expires_timestamp_ns must not precede capture time.")
        for name in ("image_state", "bev_state"):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.uint8 or value.ndim != 2:
                raise ValueError(f"{name} must be a uint8 2D array.")
            if np.any(
                (value != FieldMaskState.OUTSIDE)
                & (value != FieldMaskState.UNCERTAIN)
                & (value != FieldMaskState.INSIDE)
            ):
                raise ValueError(f"{name} contains an invalid field mask state.")
            owned = np.ascontiguousarray(value).copy()
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if not isinstance(self.filter_ready, bool) or not isinstance(
            self.hard_mask_ready, bool
        ):
            raise ValueError("field boundary readiness flags must be booleans.")
        if self.hard_mask_ready and not self.filter_ready:
            raise ValueError("hard_mask_ready requires filter_ready.")
        if (
            not isinstance(self.neutral_fill_bgr, tuple)
            or len(self.neutral_fill_bgr) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in self.neutral_fill_bgr
            )
        ):
            raise ValueError("neutral_fill_bgr must contain three uint8 values.")
        if (
            isinstance(self.blur_radius_px, bool)
            or not isinstance(self.blur_radius_px, int)
            or self.blur_radius_px < 0
        ):
            raise ValueError("blur_radius_px must be a non-negative integer.")

    def usable_at(self, timestamp_ns: int) -> bool:
        return self.capture_timestamp_ns <= timestamp_ns <= self.expires_timestamp_ns

    def state_at_pixel(self, u: float, v: float) -> FieldMaskState:
        height, width = self.image_state.shape
        if not (math.isfinite(u) and math.isfinite(v)):
            return FieldMaskState.UNCERTAIN
        iu = int(round(u))
        iv = int(round(v))
        if not (0 <= iu < width and 0 <= iv < height):
            return FieldMaskState.UNCERTAIN
        return FieldMaskState(int(self.image_state[iv, iu]))

    def mask_for_inference(
        self,
        image_bgr: Uint8Array,
        *,
        timestamp_ns: int,
    ) -> Uint8Array:
        """只遮挡明确场外像素；不修改调用方持有的原图。"""

        if (
            not self.hard_mask_ready
            or not self.usable_at(timestamp_ns)
            or image_bgr.shape[:2] != self.image_state.shape
        ):
            return image_bgr
        outside = np.where(
            self.image_state == FieldMaskState.OUTSIDE,
            255,
            0,
        ).astype(np.uint8)
        if self.blur_radius_px > 0:
            kernel = 2 * self.blur_radius_px + 1
            alpha = cv2.GaussianBlur(outside, (kernel, kernel), 0).astype(np.float32)
            alpha /= 255.0
        else:
            alpha = (outside != 0).astype(np.float32)
        fill = np.asarray(self.neutral_fill_bgr, dtype=np.float32)
        result = (
            image_bgr.astype(np.float32) * (1.0 - alpha[:, :, None])
            + fill[None, None, :] * alpha[:, :, None]
        )
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)


@dataclass(slots=True)
class _BoundaryTrack:
    boundary: LocalFieldBoundary
    confirmations: int
    missed_frames: int = 0


def _angle_difference(first: FloatArray, second: FloatArray) -> float:
    cosine = abs(float(np.dot(first, second)))
    return math.acos(min(1.0, max(0.0, cosine)))


def _oriented_angle_difference(first: FloatArray, second: FloatArray) -> float:
    cosine = float(np.dot(first, second))
    return math.acos(min(1.0, max(-1.0, cosine)))


def _field_dimensions(static_map: StaticFieldMap) -> tuple[float, float]:
    fields = tuple(
        region
        for region in static_map.regions
        if region.kind is PhysicalRegionKind.FIELD
    )
    if len(fields) != 1:
        raise ValueError("field boundary estimation requires one field region.")
    points = fields[0].polygon_field
    return (
        max(point.x for point in points) - min(point.x for point in points),
        max(point.y for point in points) - min(point.y for point in points),
    )


def _fit_segment(
    points: FloatArray,
    *,
    inlier_distance_mm: float,
    interior_evidence: FloatArray,
) -> tuple[GroundPoint, GroundPoint, FloatArray, float] | None:
    """以最长端点对为种子执行确定性 RANSAC，再对内点做 TLS 拟合。"""

    if len(points) < 2:
        return None
    best_indices: npt.NDArray[np.bool_] | None = None
    best_span = -1.0
    for first in range(len(points) - 1):
        for second in range(first + 1, len(points)):
            direction = points[second] - points[first]
            length = float(np.linalg.norm(direction))
            if length <= 1e-6:
                continue
            direction /= length
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
            distances = np.abs((points - points[first]) @ normal)
            indices = distances <= inlier_distance_mm
            if int(np.count_nonzero(indices)) < 2:
                continue
            projections = (points[indices] - points[first]) @ direction
            span = float(np.ptp(projections))
            score = int(np.count_nonzero(indices)) * 1_000_000.0 + span
            if score > best_span:
                best_span = score
                best_indices = indices
    if best_indices is None:
        return None
    inliers = points[best_indices]
    center = np.mean(inliers, axis=0)
    _u, _s, vh = np.linalg.svd(inliers - center, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0 or (math.isclose(direction[0], 0.0) and direction[1] < 0.0):
        direction = -direction
    projections = (inliers - center) @ direction
    start = center + float(np.min(projections)) * direction
    end = center + float(np.max(projections)) * direction
    if float(np.linalg.norm(end - start)) <= 1e-6:
        return None
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    midpoint = (start + end) * 0.5
    evidence_score = float(np.mean((interior_evidence - midpoint) @ normal))
    if evidence_score < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, midpoint))
    return (
        GroundPoint(float(start[0]), float(start[1])),
        GroundPoint(float(end[0]), float(end[1])),
        normal,
        offset,
    )


class FieldBoundaryEstimator:
    """把逐帧低置信度候选变为时序确认的局部开放场界。"""

    def __init__(
        self,
        config: FieldBoundaryConfig,
        *,
        static_map: StaticFieldMap,
        ground_projector: GroundProjector,
    ) -> None:
        if not isinstance(config, FieldBoundaryConfig):
            raise ValueError("config must be a FieldBoundaryConfig.")
        if not config.enabled:
            raise ValueError("FieldBoundaryEstimator requires config.enabled=true.")
        if ground_projector.bev_config is None:
            raise ValueError("field boundary estimation requires BEV configuration.")
        self._config = config
        self._projector = ground_projector
        self._field_dimensions_mm = _field_dimensions(static_map)
        self._tracks: list[_BoundaryTrack] = []
        self._last_capture_timestamp_ns: int | None = None
        bev = ground_projector.bev_config
        assert bev is not None
        rows, columns = np.indices((bev.height, bev.width), dtype=np.float64)
        self._bev_x = bev.x_max - rows * bev.mm_per_pixel
        self._bev_y = bev.y_max - columns * bev.mm_per_pixel

    def _single_frame_boundaries(
        self,
        result: FieldFeatureDetectionResult,
    ) -> list[LocalFieldBoundary]:
        segments = [
            item
            for item in result.boundary_features
            if item.kind is BoundaryFeatureKind.FENCE_BASE_SEGMENT
            and item.points_ground is not None
            and item.confidence >= self._config.min_candidate_confidence
        ]
        if not segments:
            return []
        interior_points: list[tuple[float, float]] = [(0.0, 0.0)]
        for region in (*result.safe_zones, *result.start_zones):
            if region.polygon_ground is not None:
                interior_points.extend(
                    (point.x, point.y) for point in region.polygon_ground
                )
        if result.center_cross is not None:
            if result.center_cross.intersection_ground is not None:
                point = result.center_cross.intersection_ground
                interior_points.append((point.x, point.y))
            for axis in result.center_cross.axes:
                if axis.start_ground is not None and axis.end_ground is not None:
                    interior_points.extend(
                        (
                            (axis.start_ground.x, axis.start_ground.y),
                            (axis.end_ground.x, axis.end_ground.y),
                        )
                    )
        interior_evidence = np.asarray(interior_points, dtype=np.float64)
        clusters: list[list[FloatArray]] = []
        cluster_directions: list[FloatArray] = []
        cluster_confidences: list[list[float]] = []
        angle_tolerance = math.radians(self._config.line_angle_tolerance_deg)
        for item in sorted(segments, key=lambda value: value.confidence, reverse=True):
            assert item.points_ground is not None
            points = np.asarray(
                [(point.x, point.y) for point in item.points_ground],
                dtype=np.float64,
            )
            direction = points[1] - points[0]
            direction /= np.linalg.norm(direction)
            selected: int | None = None
            for index, existing in enumerate(cluster_directions):
                if _angle_difference(direction, existing) > angle_tolerance:
                    continue
                all_points = np.concatenate(clusters[index], axis=0)
                normal = np.asarray((-existing[1], existing[0]), dtype=np.float64)
                center_distance = abs(
                    float(
                        np.dot(
                            np.mean(points, axis=0)
                            - np.mean(all_points, axis=0),
                            normal,
                        )
                    )
                )
                if center_distance <= self._config.line_distance_tolerance_mm:
                    selected = index
                    break
            if selected is None:
                clusters.append([points])
                cluster_directions.append(direction)
                cluster_confidences.append([item.confidence])
            else:
                clusters[selected].append(points)
                cluster_confidences[selected].append(item.confidence)

        fitted: list[LocalFieldBoundary] = []
        for cluster, confidences in zip(clusters, cluster_confidences):
            fit = _fit_segment(
                np.concatenate(cluster, axis=0),
                inlier_distance_mm=self._config.ransac_inlier_distance_mm,
                interior_evidence=interior_evidence,
            )
            if fit is None:
                continue
            start, end, normal, offset = fit
            confidence = min(
                0.95,
                max(confidences)
                + 0.08 * (len(cluster) - 1),
            )
            fitted.append(
                LocalFieldBoundary(
                    start,
                    end,
                    (float(normal[0]), float(normal[1])),
                    offset,
                    result.capture_timestamp_ns,
                    confidence,
                    1,
                )
            )

        if not fitted:
            return []
        reference = np.asarray(fitted[0].interior_normal_ground, dtype=np.float64)
        allowed = []
        for boundary in fitted:
            normal = np.asarray(boundary.interior_normal_ground, dtype=np.float64)
            angle = _angle_difference(reference, normal)
            perpendicular_error = abs(angle - math.pi / 2.0)
            if min(angle, perpendicular_error) <= angle_tolerance:
                allowed.append(boundary)

        max_dimension = max(self._field_dimensions_mm) * (
            1.0 + self._config.rectangle_tolerance_fraction
        )
        constrained: list[LocalFieldBoundary] = []
        for boundary in allowed:
            normal = np.asarray(boundary.interior_normal_ground, dtype=np.float64)
            if any(
                _angle_difference(
                    normal,
                    np.asarray(existing.interior_normal_ground, dtype=np.float64),
                )
                <= angle_tolerance
                and abs(
                    normal[0]
                    * (
                        existing.start_ground.x
                        - boundary.start_ground.x
                    )
                    + normal[1]
                    * (
                        existing.start_ground.y
                        - boundary.start_ground.y
                    )
                ) > max_dimension
                for existing in constrained
            ):
                continue
            constrained.append(boundary)
        return constrained

    def _update_tracks(
        self,
        boundaries: list[LocalFieldBoundary],
    ) -> tuple[LocalFieldBoundary, ...]:
        angle_tolerance = math.radians(self._config.line_angle_tolerance_deg)
        unmatched = set(range(len(self._tracks)))
        updated: list[_BoundaryTrack] = []
        for boundary in boundaries:
            normal = np.asarray(boundary.interior_normal_ground, dtype=np.float64)
            best: int | None = None
            best_distance = float("inf")
            for index in unmatched:
                track = self._tracks[index]
                existing = np.asarray(
                    track.boundary.interior_normal_ground,
                    dtype=np.float64,
                )
                if _oriented_angle_difference(normal, existing) > angle_tolerance:
                    continue
                distance = abs(boundary.line_offset_mm - track.boundary.line_offset_mm)
                if (
                    distance <= self._config.line_distance_tolerance_mm
                    and distance < best_distance
                ):
                    best = index
                    best_distance = distance
            confirmations = 1
            if best is not None:
                confirmations = self._tracks[best].confirmations + 1
                unmatched.remove(best)
            confirmed = LocalFieldBoundary(
                boundary.start_ground,
                boundary.end_ground,
                boundary.interior_normal_ground,
                boundary.line_offset_mm,
                boundary.capture_timestamp_ns,
                boundary.confidence,
                confirmations,
            )
            updated.append(_BoundaryTrack(confirmed, confirmations))
        for index in unmatched:
            track = self._tracks[index]
            track.missed_frames += 1
            if track.missed_frames <= self._config.max_missed_frames:
                updated.append(track)
        self._tracks = updated
        return tuple(
            track.boundary
            for track in updated
            if track.missed_frames == 0
            and track.confirmations >= self._config.min_confirmations
        )

    def _bev_state(
        self,
        boundaries: tuple[LocalFieldBoundary, ...],
    ) -> Uint8Array:
        x = self._bev_x
        y = self._bev_y
        outside = np.zeros(x.shape, dtype=bool)
        inside = np.zeros(x.shape, dtype=bool)
        for boundary in boundaries:
            start = np.asarray(
                (boundary.start_ground.x, boundary.start_ground.y),
                dtype=np.float64,
            )
            end = np.asarray(
                (boundary.end_ground.x, boundary.end_ground.y),
                dtype=np.float64,
            )
            tangent = end - start
            length = float(np.linalg.norm(tangent))
            tangent /= length
            along = (x - start[0]) * tangent[0] + (y - start[1]) * tangent[1]
            applicable = (
                (along >= -self._config.segment_extension_mm)
                & (along <= length + self._config.segment_extension_mm)
            )
            nx, ny = boundary.interior_normal_ground
            signed = nx * x + ny * y + boundary.line_offset_mm
            outside |= applicable & (signed < -self._config.boundary_band_mm)
            inside |= applicable & (signed > self._config.boundary_band_mm)
        state = np.full(x.shape, FieldMaskState.UNCERTAIN, dtype=np.uint8)
        state[inside & ~outside] = FieldMaskState.INSIDE
        state[outside] = FieldMaskState.OUTSIDE
        return state

    def update(
        self,
        result: FieldFeatureDetectionResult,
        *,
        valid_mask: Uint8Array,
    ) -> FieldBoundaryMask:
        width, height = result.image_size
        if (
            self._last_capture_timestamp_ns is not None
            and result.capture_timestamp_ns <= self._last_capture_timestamp_ns
        ):
            raise ValueError(
                "field boundary frames must have strictly increasing capture timestamps."
            )
        if (
            valid_mask.dtype != np.uint8
            or valid_mask.shape != (height, width)
            or np.any((valid_mask != 0) & (valid_mask != 255))
            or not np.any(valid_mask)
        ):
            raise ValueError(
                "valid_mask must be a non-empty matching binary uint8 image."
            )
        self._last_capture_timestamp_ns = result.capture_timestamp_ns
        current = self._single_frame_boundaries(result)
        confirmed = self._update_tracks(current)
        bev_state = self._bev_state(confirmed)
        assert self._projector.bev_to_ground is not None
        bev_to_image = self._projector.ground_to_image @ self._projector.bev_to_ground
        image_state = cv2.warpPerspective(
            bev_state,
            bev_to_image,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(FieldMaskState.UNCERTAIN),
        )
        image_state[valid_mask == 0] = FieldMaskState.OUTSIDE
        confidence = (
            min(boundary.confidence for boundary in confirmed)
            if confirmed
            else 0.0
        )
        filter_ready = (
            bool(confirmed)
            and confidence >= self._config.min_filter_confidence
        )
        hard_ready = self._config.hard_mask_enabled and filter_ready
        return FieldBoundaryMask(
            capture_timestamp_ns=result.capture_timestamp_ns,
            expires_timestamp_ns=result.capture_timestamp_ns
            + round(self._config.max_mask_age_ms * 1_000_000.0),
            image_state=image_state,
            bev_state=bev_state,
            boundaries=confirmed,
            confidence=confidence,
            filter_ready=filter_ready,
            hard_mask_ready=hard_ready,
            neutral_fill_bgr=self._config.neutral_fill_bgr,
            blur_radius_px=self._config.mask_blur_radius_px,
        )
