"""用颜色、线段和角点产生保守的静态场地特征观测。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import monotonic_ns

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import BevPixel, GroundPoint, UndistortedPixel
from rescue_vision.perception.detector import StaleObservationError
from rescue_vision.perception.field_feature_types import (
    BoundaryFeatureKind,
    BoundaryFeatureObservation,
    CenterCrossObservation,
    FieldFeatureConfig,
    FieldFeatureDetectionResult,
    FieldFeatureQuality,
    LineSegmentObservation,
    RealtimeFieldFeatureResult,
    SafeZoneColor,
    SafeZoneHalfObservation,
    SafeZoneObservation,
    SafeZoneSide,
    StartZoneObservation,
)
from rescue_vision.perception.types import HsvRange


FloatPoint = npt.NDArray[np.float64]
Uint8Array = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _WorkingImage:
    image_bgr: Uint8Array
    valid_mask: Uint8Array
    is_bev: bool


@dataclass(frozen=True, slots=True)
class _MappedPoints:
    pixels: tuple[UndistortedPixel, ...]
    ground: tuple[GroundPoint, ...] | None


def _mask_for_ranges(
    image_hsv: Uint8Array,
    ranges: tuple[HsvRange, ...],
) -> Uint8Array:
    mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
    for hsv_range in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                image_hsv,
                np.asarray(hsv_range.lower, dtype=np.uint8),
                np.asarray(hsv_range.upper, dtype=np.uint8),
            ),
        )
    return mask


def _morphology(mask: Uint8Array, config: FieldFeatureConfig) -> Uint8Array:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel_size, config.morphology_kernel_size),
    )
    result = mask
    if config.open_iterations:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_OPEN,
            kernel,
            iterations=config.open_iterations,
        )
    if config.close_iterations:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config.close_iterations,
        )
    return result


def _line_length(segment: FloatPoint) -> float:
    return float(np.linalg.norm(segment[1] - segment[0]))


def _line_fraction(
    mask: Uint8Array,
    start: FloatPoint,
    end: FloatPoint,
    thickness: int,
) -> float:
    sample = np.zeros(mask.shape, dtype=np.uint8)
    cv2.line(
        sample,
        tuple(np.rint(start).astype(int)),
        tuple(np.rint(end).astype(int)),
        255,
        max(1, int(thickness)),
        cv2.LINE_8,
    )
    selected = sample != 0
    count = int(np.count_nonzero(selected))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(mask[selected])) / count


def _within_expected(
    actual: float,
    expected: float,
    tolerance_fraction: float,
) -> bool:
    return (
        expected * (1.0 - tolerance_fraction)
        <= actual
        <= expected * (1.0 + tolerance_fraction)
    )


def _hough_segments(
    mask: Uint8Array,
    *,
    min_length_fraction: float,
    max_gap_fraction: float,
) -> list[FloatPoint]:
    height, width = mask.shape
    diagonal = math.hypot(width, height)
    min_length = max(5, round(diagonal * min_length_fraction))
    max_gap = max(1, round(diagonal * max_gap_fraction))
    lines = cv2.HoughLinesP(
        mask,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(10, round(min_length * 0.25)),
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []
    segments = [
        np.asarray(((x1, y1), (x2, y2)), dtype=np.float64)
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
    ]
    segments.sort(key=_line_length, reverse=True)
    return segments


def _line_gap_count(mask: Uint8Array, segment: FloatPoint) -> int:
    """沿候选轴统计内部背景段，拒绝把实线围栏当作点划线。"""

    length = max(2, round(_line_length(segment)))
    samples = np.linspace(segment[0], segment[1], length + 1)
    foreground: list[bool] = []
    height, width = mask.shape
    for u_float, v_float in samples:
        u = min(width - 1, max(0, round(float(u_float))))
        v = min(height - 1, max(0, round(float(v_float))))
        neighborhood = mask[
            max(0, v - 1) : min(height, v + 2),
            max(0, u - 1) : min(width, u + 2),
        ]
        foreground.append(bool(np.any(neighborhood)))
    gaps = 0
    in_gap = False
    seen_foreground = False
    for value in foreground:
        if value:
            if in_gap and seen_foreground:
                gaps += 1
            seen_foreground = True
            in_gap = False
        elif seen_foreground:
            in_gap = True
    return gaps


def _perpendicular(
    first: FloatPoint,
    second: FloatPoint,
    tolerance_deg: float,
) -> bool:
    first_vector = first[1] - first[0]
    second_vector = second[1] - second[0]
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0.0:
        return False
    absolute_cosine = abs(float(np.dot(first_vector, second_vector) / denominator))
    return absolute_cosine <= math.sin(math.radians(tolerance_deg))


def _parallel(
    first: FloatPoint,
    second: FloatPoint,
    tolerance_deg: float,
) -> bool:
    first_vector = first[1] - first[0]
    second_vector = second[1] - second[0]
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0.0:
        return False
    absolute_cosine = abs(float(np.dot(first_vector, second_vector) / denominator))
    return absolute_cosine >= math.cos(math.radians(tolerance_deg))


def _point_to_line_distance(point: FloatPoint, line: FloatPoint) -> float:
    direction = line[1] - line[0]
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return float("inf")
    offset = point - line[0]
    return abs(float(direction[0] * offset[1] - direction[1] * offset[0])) / length


def _segment_intersection(
    first: FloatPoint,
    second: FloatPoint,
) -> FloatPoint | None:
    p = first[0]
    r = first[1] - first[0]
    q = second[0]
    s = second[1] - second[0]
    cross = float(r[0] * s[1] - r[1] * s[0])
    if math.isclose(cross, 0.0, abs_tol=1e-9):
        return None
    q_minus_p = q - p
    t = float((q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross)
    u = float((q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / cross)
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return np.asarray(p + t * r, dtype=np.float64)


class FieldFeatureDetector:
    """检测安全区、出发区、中心十字和低精度边界特征。"""

    def __init__(
        self,
        config: FieldFeatureConfig,
        *,
        max_observation_age_ms: float,
        ground_projector: GroundProjector | None = None,
    ) -> None:
        if not isinstance(config, FieldFeatureConfig):
            raise ValueError("config must be a FieldFeatureConfig.")
        if not config.enabled:
            raise ValueError(
                "FieldFeatureDetector requires config.enabled=true; runtime "
                "callers should use PerceptionConfig.build_field_feature_detector()."
            )
        converted_age = float(max_observation_age_ms)
        if not math.isfinite(converted_age) or converted_age <= 0.0:
            raise ValueError("max_observation_age_ms must be positive and finite.")
        self._config = config
        self._max_observation_age_ms = converted_age
        self._ground_projector = ground_projector

    def _working_image(
        self,
        image_bgr: Uint8Array,
        valid_mask: Uint8Array,
    ) -> _WorkingImage:
        projector = self._ground_projector
        if (
            projector is None
            or projector.bev_config is None
            or projector.image_to_bev is None
        ):
            return _WorkingImage(
                image_bgr=image_bgr,
                valid_mask=valid_mask,
                is_bev=False,
            )
        bev = projector.make_bev_image(image_bgr)
        valid = cv2.warpPerspective(
            valid_mask,
            projector.image_to_bev,
            (projector.bev_config.width, projector.bev_config.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return _WorkingImage(bev, valid, True)

    def _map_points(
        self,
        points: npt.ArrayLike,
        *,
        is_bev: bool,
    ) -> _MappedPoints:
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        projector = self._ground_projector
        if is_bev:
            if projector is None:
                raise RuntimeError("BEV points require a GroundProjector.")
            ground = tuple(
                projector.bev_pixels_to_ground(
                    [BevPixel(float(u), float(v)) for u, v in array]
                )
            )
            pixels = tuple(projector.ground_to_pixels(ground))
            return _MappedPoints(pixels, ground)
        pixels = tuple(UndistortedPixel(float(u), float(v)) for u, v in array)
        ground = (
            tuple(projector.pixels_to_ground(pixels))
            if projector is not None
            else None
        )
        return _MappedPoints(pixels, ground)

    def _line_observation(
        self,
        segment: FloatPoint,
        *,
        is_bev: bool,
    ) -> LineSegmentObservation:
        mapped = self._map_points(segment, is_bev=is_bev)
        return LineSegmentObservation(
            mapped.pixels[0],
            mapped.pixels[1],
            mapped.ground[0] if mapped.ground is not None else None,
            mapped.ground[1] if mapped.ground is not None else None,
        )

    def _safe_zone(
        self,
        color: SafeZoneColor,
        color_mask: Uint8Array,
        purple_mask: Uint8Array,
        dark_mask: Uint8Array,
        *,
        valid_area: int,
        is_bev: bool,
    ) -> SafeZoneObservation | None:
        nonzero = cv2.findNonZero(color_mask)
        if nonzero is None:
            return None
        area = int(np.count_nonzero(color_mask))
        area_fraction = area / valid_area
        if area_fraction < self._config.min_region_area_fraction:
            return None

        rectangle = cv2.minAreaRect(nonzero)
        width_px, depth_px = sorted((float(rectangle[1][0]), float(rectangle[1][1])), reverse=True)
        rectangle_area = width_px * depth_px
        if rectangle_area <= 0.0:
            return None
        rectangularity = area / rectangle_area
        if rectangularity < self._config.min_rectangularity:
            return None
        if is_bev:
            assert self._ground_projector is not None
            assert self._ground_projector.bev_config is not None
            mm_per_pixel = self._ground_projector.bev_config.mm_per_pixel
            if not (
                _within_expected(
                    width_px * mm_per_pixel,
                    self._config.safe_width_mm,
                    self._config.dimension_tolerance_fraction,
                )
                and _within_expected(
                    depth_px * mm_per_pixel,
                    self._config.safe_depth_mm,
                    self._config.dimension_tolerance_fraction,
                )
            ):
                return None

        box = cv2.boxPoints(rectangle).astype(np.float64)
        mapped_box = self._map_points(box, is_bev=is_bev)
        edges = [
            np.asarray((box[index], box[(index + 1) % 4]), dtype=np.float64)
            for index in range(4)
        ]
        lengths = [_line_length(edge) for edge in edges]
        longest = max(lengths)
        long_edge_indices = [
            index for index, length in enumerate(lengths) if length >= 0.85 * longest
        ]
        band = max(3, round(min(width_px, depth_px) * 0.12))
        entrance_index: int | None = None
        entrance_score = 0.0
        if len(long_edge_indices) >= 2:
            scores = [
                (_line_fraction(purple_mask, *edges[index], band), index)
                for index in long_edge_indices
            ]
            entrance_score, best_index = max(scores)
            if entrance_score >= self._config.entrance_color_fraction:
                entrance_index = best_index

        quality: set[FieldFeatureQuality] = set()
        if mapped_box.ground is None:
            quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
        if entrance_index is None:
            quality.update(
                {
                    FieldFeatureQuality.ENTRANCE_UNRESOLVED,
                    FieldFeatureQuality.SIDE_UNRESOLVED,
                }
            )
            entrance_observation = None
            divider_observation = None
            halves: tuple[SafeZoneHalfObservation, ...] = ()
            divider_score = 0.0
            quality.add(FieldFeatureQuality.DIVIDER_UNRESOLVED)
        else:
            entrance_segment = edges[entrance_index]
            opposite_segment = edges[(entrance_index + 2) % 4]
            entrance_observation = self._line_observation(
                entrance_segment,
                is_bev=is_bev,
            )
            entrance_midpoint = np.mean(entrance_segment, axis=0)
            opposite_midpoint = np.mean(opposite_segment, axis=0)
            divider_segment = np.asarray(
                (entrance_midpoint, opposite_midpoint),
                dtype=np.float64,
            )
            divider_score = _line_fraction(
                dark_mask,
                divider_segment[0],
                divider_segment[1],
                max(3, round(min(width_px, depth_px) * 0.08)),
            )
            if divider_score < self._config.divider_dark_fraction:
                divider_observation = None
                halves = ()
                quality.update(
                    {
                        FieldFeatureQuality.DIVIDER_UNRESOLVED,
                        FieldFeatureQuality.SIDE_UNRESOLVED,
                    }
                )
            else:
                divider_observation = self._line_observation(
                    divider_segment,
                    is_bev=is_bev,
                )
                if mapped_box.ground is None:
                    halves = ()
                    quality.add(FieldFeatureQuality.SIDE_UNRESOLVED)
                else:
                    entrance_a = box[entrance_index]
                    entrance_b = box[(entrance_index + 1) % 4]
                    back_for_a = box[(entrance_index + 3) % 4]
                    back_for_b = box[(entrance_index + 2) % 4]
                    ground_corners = mapped_box.ground
                    ground_a = ground_corners[entrance_index]
                    ground_b = ground_corners[(entrance_index + 1) % 4]
                    ground_center = GroundPoint(
                        sum(point.x for point in ground_corners) / 4.0,
                        sum(point.y for point in ground_corners) / 4.0,
                    )
                    ground_entrance_mid = GroundPoint(
                        (ground_a.x + ground_b.x) / 2.0,
                        (ground_a.y + ground_b.y) / 2.0,
                    )
                    direction = np.asarray(
                        (
                            ground_center.x - ground_entrance_mid.x,
                            ground_center.y - ground_entrance_mid.y,
                        ),
                        dtype=np.float64,
                    )
                    left = np.asarray((-direction[1], direction[0]), dtype=np.float64)
                    endpoint_delta = np.asarray(
                        (ground_a.x - ground_entrance_mid.x, ground_a.y - ground_entrance_mid.y),
                        dtype=np.float64,
                    )
                    if float(np.dot(endpoint_delta, left)) >= 0.0:
                        left_entrance, left_back = entrance_a, back_for_a
                        right_entrance, right_back = entrance_b, back_for_b
                    else:
                        left_entrance, left_back = entrance_b, back_for_b
                        right_entrance, right_back = entrance_a, back_for_a
                    left_work = np.asarray(
                        (
                            left_entrance,
                            entrance_midpoint,
                            opposite_midpoint,
                            left_back,
                        ),
                        dtype=np.float64,
                    )
                    right_work = np.asarray(
                        (
                            entrance_midpoint,
                            right_entrance,
                            right_back,
                            opposite_midpoint,
                        ),
                        dtype=np.float64,
                    )
                    left_mapped = self._map_points(left_work, is_bev=is_bev)
                    right_mapped = self._map_points(right_work, is_bev=is_bev)
                    assert left_mapped.ground is not None
                    assert right_mapped.ground is not None
                    halves = (
                        SafeZoneHalfObservation(
                            SafeZoneSide.APPROACH_LEFT,
                            left_mapped.pixels,
                            left_mapped.ground,
                        ),
                        SafeZoneHalfObservation(
                            SafeZoneSide.APPROACH_RIGHT,
                            right_mapped.pixels,
                            right_mapped.ground,
                        ),
                    )

        area_evidence = min(
            1.0,
            area_fraction / max(self._config.min_region_area_fraction * 4.0, 1e-9),
        )
        confidence = min(
            1.0,
            0.45 * min(1.0, rectangularity)
            + 0.20 * area_evidence
            + 0.175 * min(1.0, entrance_score)
            + 0.175 * min(1.0, divider_score),
        )
        return SafeZoneObservation(
            physical_color=color,
            polygon_undistorted=mapped_box.pixels,
            polygon_ground=mapped_box.ground,
            entrance=entrance_observation,
            divider=divider_observation,
            halves=halves,
            confidence=confidence,
            quality=frozenset(quality),
        )

    def _start_zones(
        self,
        mask: Uint8Array,
        *,
        valid_area: int,
        is_bev: bool,
    ) -> tuple[StartZoneObservation, ...]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        observations: list[tuple[float, StartZoneObservation]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_fraction = area / valid_area
            if area_fraction < self._config.min_region_area_fraction:
                continue
            rectangle = cv2.minAreaRect(contour)
            first, second = (float(rectangle[1][0]), float(rectangle[1][1]))
            rectangle_area = first * second
            if rectangle_area <= 0.0:
                continue
            rectangularity = area / rectangle_area
            if rectangularity < self._config.min_rectangularity:
                continue
            if is_bev:
                assert self._ground_projector is not None
                assert self._ground_projector.bev_config is not None
                scale = self._ground_projector.bev_config.mm_per_pixel
                if not (
                    _within_expected(
                        first * scale,
                        self._config.start_side_mm,
                        self._config.dimension_tolerance_fraction,
                    )
                    and _within_expected(
                        second * scale,
                        self._config.start_side_mm,
                        self._config.dimension_tolerance_fraction,
                    )
                ):
                    continue
            mapped = self._map_points(cv2.boxPoints(rectangle), is_bev=is_bev)
            quality = (
                frozenset({FieldFeatureQuality.NO_GROUND_PROJECTION})
                if mapped.ground is None
                else frozenset()
            )
            confidence = min(
                1.0,
                0.7 * min(1.0, rectangularity)
                + 0.3
                * min(
                    1.0,
                    area_fraction
                    / max(self._config.min_region_area_fraction * 4.0, 1e-9),
                ),
            )
            observations.append(
                (
                    area,
                    StartZoneObservation(
                        mapped.pixels,
                        mapped.ground,
                        confidence,
                        quality,
                    ),
                )
            )
        observations.sort(key=lambda item: item[0], reverse=True)
        return tuple(item[1] for item in observations)

    def _center_cross(
        self,
        dark_mask: Uint8Array,
        *,
        is_bev: bool,
    ) -> CenterCrossObservation | None:
        segments = _hough_segments(
            dark_mask,
            min_length_fraction=self._config.center_min_axis_span_fraction,
            max_gap_fraction=self._config.center_max_gap_fraction,
        )
        segments = [
            segment
            for segment in segments
            if _line_gap_count(dark_mask, segment)
            >= self._config.center_min_gap_count
        ]
        if not segments:
            return None
        best_pair: tuple[FloatPoint, FloatPoint, FloatPoint] | None = None
        best_score = -1.0
        for first_index, first in enumerate(segments[:20]):
            for second in segments[first_index + 1 : 20]:
                if not _perpendicular(
                    first,
                    second,
                    self._config.center_perpendicular_tolerance_deg,
                ):
                    continue
                intersection = _segment_intersection(first, second)
                if intersection is None:
                    continue
                score = _line_length(first) + _line_length(second)
                if score > best_score:
                    best_score = score
                    best_pair = (first, second, intersection)

        quality: set[FieldFeatureQuality] = set()
        if best_pair is None:
            quality.add(FieldFeatureQuality.PARTIAL)
            axis = self._line_observation(segments[0], is_bev=is_bev)
            if axis.start_ground is None:
                quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
            return CenterCrossObservation(
                axes=(axis,),
                intersection_undistorted=None,
                intersection_ground=None,
                confidence=0.30,
                quality=frozenset(quality),
            )

        first, second, intersection = best_pair
        axes = (
            self._line_observation(first, is_bev=is_bev),
            self._line_observation(second, is_bev=is_bev),
        )
        mapped_intersection = self._map_points((intersection,), is_bev=is_bev)
        if mapped_intersection.ground is None:
            quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
        return CenterCrossObservation(
            axes=axes,
            intersection_undistorted=mapped_intersection.pixels[0],
            intersection_ground=(
                mapped_intersection.ground[0]
                if mapped_intersection.ground is not None
                else None
            ),
            confidence=0.80,
            quality=frozenset(quality),
        )

    def _boundary_features(
        self,
        image_bgr: Uint8Array,
        valid_mask: Uint8Array,
        center_cross: CenterCrossObservation | None,
        *,
        is_bev: bool,
    ) -> tuple[BoundaryFeatureObservation, ...]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            self._config.boundary_canny_low_threshold,
            self._config.boundary_canny_high_threshold,
        )
        # Canny 会在去畸变填充区与有效图像的交界处产生强边缘。向内收缩两像素
        # 后再筛边缘，避免把这圈人工边界当作围栏或场角。
        interior_valid = cv2.erode(
            valid_mask,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        edges[interior_valid == 0] = 0
        segments = _hough_segments(
            edges,
            min_length_fraction=self._config.boundary_min_line_length_fraction,
            max_gap_fraction=0.03,
        )
        if not segments:
            return ()

        height, width = edges.shape
        diagonal = math.hypot(width, height)
        center_axes_work: list[FloatPoint] = []
        if center_cross is not None:
            for axis in center_cross.axes:
                if is_bev:
                    if (
                        self._ground_projector is None
                        or axis.start_ground is None
                        or axis.end_ground is None
                    ):
                        continue
                    bev_points = self._ground_projector.ground_to_bev_pixels(
                        (axis.start_ground, axis.end_ground)
                    )
                    center_axes_work.append(
                        np.asarray(
                            (
                                (bev_points[0].u, bev_points[0].v),
                                (bev_points[1].u, bev_points[1].v),
                            ),
                            dtype=np.float64,
                        )
                    )
                else:
                    center_axes_work.append(
                        np.asarray(
                            (
                                (
                                    axis.start_undistorted.u,
                                    axis.start_undistorted.v,
                                ),
                                (
                                    axis.end_undistorted.u,
                                    axis.end_undistorted.v,
                                ),
                            ),
                            dtype=np.float64,
                        )
                    )
        segments = [
            segment
            for segment in segments
            if not any(
                _parallel(segment, center_axis, 10.0)
                and _point_to_line_distance(
                    np.mean(segment, axis=0),
                    center_axis,
                )
                < 0.02 * diagonal
                for center_axis in center_axes_work
            )
        ]
        if not segments:
            return ()

        excluded_center: np.ndarray | None = None
        if center_cross is not None:
            if (
                is_bev
                and center_cross.intersection_ground is not None
                and self._ground_projector is not None
            ):
                center_bev = self._ground_projector.ground_to_bev_pixel(
                    center_cross.intersection_ground
                )
                excluded_center = np.asarray(
                    (center_bev.u, center_bev.v),
                    dtype=np.float64,
                )
            elif (
                not is_bev
                and center_cross.intersection_undistorted is not None
            ):
                excluded_center = np.asarray(
                    (
                        center_cross.intersection_undistorted.u,
                        center_cross.intersection_undistorted.v,
                    ),
                    dtype=np.float64,
                )

        observations: list[BoundaryFeatureObservation] = []
        corner_points: list[FloatPoint] = []
        for first_index, first in enumerate(segments[:12]):
            for second in segments[first_index + 1 : 12]:
                if not _perpendicular(
                    first,
                    second,
                    self._config.boundary_corner_tolerance_deg,
                ):
                    continue
                intersection = _segment_intersection(first, second)
                if intersection is None:
                    continue
                if excluded_center is not None and float(
                    np.linalg.norm(intersection - excluded_center)
                ) < 0.05 * diagonal:
                    continue
                if any(
                    float(np.linalg.norm(intersection - existing)) < 0.03 * diagonal
                    for existing in corner_points
                ):
                    continue
                corner_points.append(intersection)
                mapped = self._map_points((intersection,), is_bev=is_bev)
                quality = {FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY}
                if mapped.ground is None:
                    quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
                observations.append(
                    BoundaryFeatureObservation(
                        BoundaryFeatureKind.FIELD_CORNER,
                        mapped.pixels,
                        mapped.ground,
                        0.35,
                        frozenset(quality),
                    )
                )
                if len(observations) >= self._config.boundary_max_features:
                    return tuple(observations)

        for segment in segments:
            mapped = self._map_points(segment, is_bev=is_bev)
            quality = {FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY}
            if mapped.ground is None:
                quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
            observations.append(
                BoundaryFeatureObservation(
                    BoundaryFeatureKind.FENCE_BASE_SEGMENT,
                    mapped.pixels,
                    mapped.ground,
                    min(0.45, 0.2 + 0.25 * _line_length(segment) / diagonal),
                    frozenset(quality),
                )
            )
            if len(observations) >= self._config.boundary_max_features:
                break
        return tuple(observations)

    def detect(
        self,
        frame: CameraFrame,
        undistorted_image_bgr: Uint8Array,
        *,
        valid_mask: Uint8Array,
        result_timestamp_ns: int | None = None,
    ) -> FieldFeatureDetectionResult:
        if (
            not isinstance(undistorted_image_bgr, np.ndarray)
            or undistorted_image_bgr.dtype != np.uint8
            or undistorted_image_bgr.ndim != 3
            or undistorted_image_bgr.shape[2] != 3
        ):
            raise ValueError(
                "undistorted_image_bgr must be uint8 with shape "
                "(height, width, 3)."
            )
        image_size = (
            int(undistorted_image_bgr.shape[1]),
            int(undistorted_image_bgr.shape[0]),
        )
        frame_size = (int(frame.image_bgr.shape[1]), int(frame.image_bgr.shape[0]))
        if image_size != frame_size:
            raise ValueError(
                f"Undistorted image_size {image_size} does not match frame {frame_size}."
            )
        if (
            not isinstance(valid_mask, np.ndarray)
            or valid_mask.dtype != np.uint8
            or valid_mask.ndim != 2
            or valid_mask.shape != undistorted_image_bgr.shape[:2]
        ):
            raise ValueError(
                "valid_mask must be uint8 with shape (height, width) matching "
                "undistorted_image_bgr."
            )
        if np.any((valid_mask != 0) & (valid_mask != 255)):
            raise ValueError("valid_mask values must be either 0 or 255.")
        if not np.any(valid_mask):
            raise ValueError("valid_mask must contain at least one valid pixel.")

        working = self._working_image(undistorted_image_bgr, valid_mask)
        hsv = cv2.cvtColor(working.image_bgr, cv2.COLOR_BGR2HSV)
        masks = {
            "safe_red": _morphology(
                _mask_for_ranges(hsv, self._config.safe_red),
                self._config,
            ),
            "safe_blue": _morphology(
                _mask_for_ranges(hsv, self._config.safe_blue),
                self._config,
            ),
            "start_magenta": _morphology(
                _mask_for_ranges(hsv, self._config.start_magenta),
                self._config,
            ),
            "entrance_purple": _morphology(
                _mask_for_ranges(hsv, self._config.entrance_purple),
                self._config,
            ),
            "dark_marking": _morphology(
                _mask_for_ranges(hsv, self._config.dark_marking),
                self._config,
            ),
        }
        for mask in masks.values():
            mask[working.valid_mask == 0] = 0
        valid_area = int(np.count_nonzero(working.valid_mask))
        if valid_area == 0:
            raise ValueError("Ground projection produced an empty valid BEV.")

        safe_zones = tuple(
            observation
            for color, name in (
                (SafeZoneColor.RED, "safe_red"),
                (SafeZoneColor.BLUE, "safe_blue"),
            )
            if (
                observation := self._safe_zone(
                    color,
                    masks[name],
                    masks["entrance_purple"],
                    masks["dark_marking"],
                    valid_area=valid_area,
                    is_bev=working.is_bev,
                )
            )
            is not None
        )
        start_zones = self._start_zones(
            masks["start_magenta"],
            valid_area=valid_area,
            is_bev=working.is_bev,
        )

        colored = cv2.bitwise_or(masks["safe_red"], masks["safe_blue"])
        colored = cv2.bitwise_or(colored, masks["start_magenta"])
        colored = cv2.bitwise_or(colored, masks["entrance_purple"])
        exclusion = cv2.dilate(
            colored,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        center_mask = cv2.bitwise_and(
            masks["dark_marking"],
            cv2.bitwise_not(exclusion),
        )
        center_cross = self._center_cross(
            center_mask,
            is_bev=working.is_bev,
        )
        boundary_features = self._boundary_features(
            working.image_bgr,
            working.valid_mask,
            center_cross,
            is_bev=working.is_bev,
        )

        completed_timestamp_ns = (
            monotonic_ns() if result_timestamp_ns is None else result_timestamp_ns
        )
        if completed_timestamp_ns < frame.timestamp_ns:
            raise ValueError(
                "result_timestamp_ns must not be earlier than frame timestamp."
            )
        age_ms = (completed_timestamp_ns - frame.timestamp_ns) / 1_000_000.0
        if age_ms > self._max_observation_age_ms:
            raise StaleObservationError(age_ms, self._max_observation_age_ms)
        return FieldFeatureDetectionResult(
            frame_sequence=frame.sequence,
            capture_timestamp_ns=frame.timestamp_ns,
            result_timestamp_ns=completed_timestamp_ns,
            image_size=image_size,
            safe_zones=safe_zones,
            start_zones=start_zones,
            center_cross=center_cross,
            boundary_features=boundary_features,
        )

    def detect_realtime(
        self,
        frame: CameraFrame,
        undistorted_image_bgr: Uint8Array,
        *,
        valid_mask: Uint8Array,
        result_timestamp_ns: int | None = None,
    ) -> RealtimeFieldFeatureResult:
        """检测最新帧，并把过期结果转换为显式空结果。"""

        try:
            result = self.detect(
                frame,
                undistorted_image_bgr,
                valid_mask=valid_mask,
                result_timestamp_ns=result_timestamp_ns,
            )
        except StaleObservationError as exc:
            return RealtimeFieldFeatureResult(
                result=None,
                dropped_stale_age_ms=exc.age_ms,
            )
        return RealtimeFieldFeatureResult(result=result)
