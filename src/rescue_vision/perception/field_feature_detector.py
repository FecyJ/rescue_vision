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
from rescue_vision.world.static_map import StaticFieldMap, TeamColor


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


@dataclass(frozen=True, slots=True)
class _CenterAxisCandidate:
    segment: FloatPoint
    local_support: float
    gap_count: int


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
    """沿候选轴统计内部背景段；调用方先按允许线宽膨胀掩码。"""

    length = max(2, round(_line_length(segment)))
    samples = np.linspace(segment[0], segment[1], length + 1)
    height, width = mask.shape
    u = np.clip(np.rint(samples[:, 0]).astype(np.intp), 0, width - 1)
    v = np.clip(np.rint(samples[:, 1]).astype(np.intp), 0, height - 1)
    foreground = mask[v, u] != 0
    indices = np.flatnonzero(foreground)
    if indices.size < 2:
        return 0
    internal = foreground[indices[0] : indices[-1] + 1]
    return int(np.count_nonzero(internal[:-1] & ~internal[1:]))


def _line_sample_fraction(mask: Uint8Array, segment: FloatPoint) -> float:
    """在已按允许线宽膨胀的掩码上计算候选轴支撑比例。"""

    length = max(2, round(_line_length(segment)))
    samples = np.linspace(segment[0], segment[1], length + 1)
    height, width = mask.shape
    u = np.clip(np.rint(samples[:, 0]).astype(np.intp), 0, width - 1)
    v = np.clip(np.rint(samples[:, 1]).astype(np.intp), 0, height - 1)
    return float(np.count_nonzero(mask[v, u])) / len(samples)


def _line_side_contrast(
    gray: Uint8Array,
    valid_mask: Uint8Array,
    segment: FloatPoint,
) -> float:
    """比较候选底边两侧的局部外观，作为场内外变化的弱证据。"""

    direction = segment[1] - segment[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return 0.0
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64) / length
    offset = max(2.0, math.hypot(*gray.shape) * 0.01)
    samples = np.linspace(segment[0], segment[1], max(8, round(length / 4.0)))
    means: list[float] = []
    height, width = gray.shape
    for sign in (-1.0, 1.0):
        shifted = samples + sign * offset * normal
        u = np.rint(shifted[:, 0]).astype(np.intp)
        v = np.rint(shifted[:, 1]).astype(np.intp)
        selected = (
            (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        u = u[selected]
        v = v[selected]
        if len(u) == 0:
            return 0.0
        valid = valid_mask[v, u] != 0
        if int(np.count_nonzero(valid)) < max(3, len(u) // 2):
            return 0.0
        means.append(float(np.median(gray[v[valid], u[valid]])))
    return abs(means[0] - means[1])


def _segment_angle(segment: FloatPoint) -> float:
    direction = segment[1] - segment[0]
    return math.atan2(float(direction[1]), float(direction[0])) % math.pi


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _deduplicate_center_axes(
    candidates: list[_CenterAxisCandidate],
    *,
    diagonal: float,
    angle_tolerance_deg: float,
    limit: int = 32,
) -> list[_CenterAxisCandidate]:
    """折叠同一物理轴产生的平行 Hough 重复线。"""

    selected: list[_CenterAxisCandidate] = []
    angle_tolerance = math.radians(max(1.0, angle_tolerance_deg / 3.0))
    distance_tolerance = max(2.0, diagonal * 0.01)
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.local_support + 0.10 * min(item.gap_count, 5),
            _line_length(item.segment),
        ),
        reverse=True,
    ):
        midpoint = np.mean(candidate.segment, axis=0)
        if any(
            _angle_difference(
                _segment_angle(candidate.segment),
                _segment_angle(existing.segment),
            )
            <= angle_tolerance
            and _point_to_line_distance(midpoint, existing.segment)
            <= distance_tolerance
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _axis_balance(segment: FloatPoint, intersection: FloatPoint) -> float:
    length = _line_length(segment)
    if length == 0.0:
        return 0.0
    return min(
        float(np.linalg.norm(intersection - segment[0])),
        float(np.linalg.norm(segment[1] - intersection)),
    ) / length


def _mask_distance_at(distance: npt.NDArray[np.float32], point: FloatPoint) -> float:
    height, width = distance.shape
    u = min(width - 1, max(0, round(float(point[0]))))
    v = min(height - 1, max(0, round(float(point[1]))))
    return float(distance[v, u])


def _local_dark_line_mask(
    image_bgr: Uint8Array,
    valid_mask: Uint8Array,
    config: FieldFeatureConfig,
) -> Uint8Array:
    """提取比局部地面更暗的低饱和细线，适应灰色连续场地标线。"""

    height, width = valid_mask.shape
    diagonal = math.hypot(width, height)
    window = max(3, round(diagonal * config.center_local_window_fraction))
    if window % 2 == 0:
        window += 1
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    local_background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window)),
    )
    contrast = cv2.subtract(local_background, gray)
    selected = (
        (contrast >= config.center_local_contrast_threshold)
        & (hsv[:, :, 1] <= config.center_max_saturation)
        & (valid_mask != 0)
    )
    return np.where(selected, 255, 0).astype(np.uint8)


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
        static_map: StaticFieldMap,
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
        if not isinstance(static_map, StaticFieldMap):
            raise ValueError("static_map must be a StaticFieldMap.")
        red_dimensions = static_map.safe_zone_dimensions_mm(TeamColor.RED)
        blue_dimensions = static_map.safe_zone_dimensions_mm(TeamColor.BLUE)
        if red_dimensions is None or blue_dimensions is None:
            raise ValueError(
                "FieldFeatureDetector requires red and blue material/injured "
                "regions in static_map."
            )
        start_dimensions = static_map.start_zone_dimensions_mm()
        if not start_dimensions:
            raise ValueError(
                "FieldFeatureDetector requires at least one start_zone region "
                "in static_map."
            )
        start_sides: list[float] = []
        for width_mm, depth_mm in start_dimensions:
            if not math.isclose(width_mm, depth_mm, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("static_map start_zone regions must be square.")
            start_sides.append(width_mm)
        if not all(
            math.isclose(side, start_sides[0], rel_tol=1e-6, abs_tol=1e-6)
            for side in start_sides[1:]
        ):
            raise ValueError(
                "static_map start_zone regions must use one consistent size."
            )
        converted_age = float(max_observation_age_ms)
        if not math.isfinite(converted_age) or converted_age <= 0.0:
            raise ValueError("max_observation_age_ms must be positive and finite.")
        self._config = config
        self._safe_zone_dimensions = {
            SafeZoneColor.RED: red_dimensions,
            SafeZoneColor.BLUE: blue_dimensions,
        }
        self._start_side_mm = start_sides[0]
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
        # The configured close operation has already joined the two halves
        # across a narrow divider.  Extract its remaining components so
        # distant same-colored targets cannot enlarge the safe-zone rectangle.
        component_count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        )
        candidates: list[SafeZoneObservation] = []
        for label in range(1, component_count):
            if stats[label, cv2.CC_STAT_AREA] <= 0:
                continue
            component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            candidate = self._safe_zone_component(
                color,
                component_mask,
                purple_mask,
                dark_mask,
                valid_area=valid_area,
                is_bev=is_bev,
            )
            if candidate is not None:
                candidates.append(candidate)
        if candidates:
            return max(candidates, key=lambda item: item.confidence)
        return self._partial_safe_zone(
            color,
            color_mask,
            purple_mask,
            dark_mask,
            valid_area=valid_area,
            is_bev=is_bev,
        )

    def _partial_safe_zone(
        self,
        color: SafeZoneColor,
        color_mask: Uint8Array,
        purple_mask: Uint8Array,
        dark_mask: Uint8Array,
        *,
        valid_area: int,
        is_bev: bool,
    ) -> SafeZoneObservation | None:
        """Accept occluded colored halves only inside one purple enclosure."""

        minimum_total_area = max(
            1,
            math.ceil(self._config.min_region_area_fraction * valid_area),
        )
        component_count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        )
        usable_components = tuple(
            label
            for label in range(1, component_count)
            if stats[label, cv2.CC_STAT_AREA] > 0
        )
        if not usable_components:
            return None

        purple_contours, _ = cv2.findContours(
            purple_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        purple_contours = sorted(
            (
                contour
                for contour in purple_contours
                if cv2.contourArea(contour) >= minimum_total_area
            ),
            key=cv2.contourArea,
            reverse=True,
        )[:8]
        enclosure_groups = [
            (contour,)
            for contour in purple_contours
        ] + [
            (first, second)
            for first_index, first in enumerate(purple_contours)
            for second in purple_contours[first_index + 1 :]
        ]
        candidates: list[SafeZoneObservation] = []
        for enclosure_group in enclosure_groups:
            purple_area = sum(
                float(cv2.contourArea(contour))
                for contour in enclosure_group
            )
            enclosure_points = np.concatenate(enclosure_group, axis=0)
            enclosure_rectangle = cv2.minAreaRect(enclosure_points)
            enclosure_width, enclosure_height = enclosure_rectangle[1]
            enclosure_area = float(enclosure_width * enclosure_height)
            if enclosure_area <= 0.0:
                continue
            purple_fill = purple_area / enclosure_area
            if purple_fill < self._config.entrance_color_fraction:
                continue
            enclosure_geometry_confirmed = False
            if is_bev:
                assert self._ground_projector is not None
                assert self._ground_projector.bev_config is not None
                mm_per_pixel = self._ground_projector.bev_config.mm_per_pixel
                expected_width_mm, expected_depth_mm = (
                    self._safe_zone_dimensions[color]
                )
                enclosure_dimensions_mm = sorted(
                    (
                        enclosure_width * mm_per_pixel,
                        enclosure_height * mm_per_pixel,
                    ),
                    reverse=True,
                )
                if not (
                    _within_expected(
                        enclosure_dimensions_mm[0],
                        expected_width_mm,
                        self._config.dimension_tolerance_fraction,
                    )
                    and _within_expected(
                        enclosure_dimensions_mm[1],
                        expected_depth_mm,
                        self._config.dimension_tolerance_fraction,
                    )
                ):
                    continue
                enclosure_geometry_confirmed = True
            enclosure_box = cv2.boxPoints(enclosure_rectangle).astype(np.float32)
            enclosed_components = sorted(
                label
                for label in usable_components
                if cv2.pointPolygonTest(
                    enclosure_box,
                    (
                        float(centroids[label][0]),
                        float(centroids[label][1]),
                    ),
                    False,
                )
                >= 0.0
            )
            enclosed_components.sort(
                key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
                reverse=True,
            )
            if not enclosed_components:
                continue
            largest_component_area = int(
                stats[enclosed_components[0], cv2.CC_STAT_AREA]
            )
            significant_components = tuple(
                label
                for label in enclosed_components
                if stats[label, cv2.CC_STAT_AREA]
                >= largest_component_area
                * self._config.entrance_color_fraction
            )
            if not significant_components:
                continue
            selected_mask = np.where(
                np.isin(labels, significant_components),
                255,
                0,
            ).astype(np.uint8)
            selected_area = int(np.count_nonzero(selected_mask))
            if selected_area < minimum_total_area:
                continue
            purple_overlap = int(
                np.count_nonzero(cv2.bitwise_and(selected_mask, purple_mask))
            ) / selected_area
            if purple_overlap > self._config.entrance_color_fraction:
                continue
            nonzero = cv2.findNonZero(selected_mask)
            assert nonzero is not None
            visible_rectangle = cv2.minAreaRect(nonzero)
            visible_width, visible_height = visible_rectangle[1]
            visible_rectangle_area = float(visible_width * visible_height)
            if visible_rectangle_area <= 0.0:
                continue
            visible_rectangularity = selected_area / visible_rectangle_area
            if visible_rectangularity < self._config.min_rectangularity:
                continue
            visible_box = cv2.boxPoints(visible_rectangle).astype(np.float64)
            observation_box = (
                enclosure_box.astype(np.float64)
                if enclosure_geometry_confirmed
                else visible_box
            )
            mapped_box = self._map_points(observation_box, is_bev=is_bev)
            edges = [
                np.asarray(
                    (visible_box[index], visible_box[(index + 1) % 4]),
                    dtype=np.float64,
                )
                for index in range(4)
            ]
            edge_lengths = [_line_length(edge) for edge in edges]
            longest = max(edge_lengths)
            long_edge_indices = [
                index
                for index, length in enumerate(edge_lengths)
                if length >= 0.85 * longest
            ]
            divider_score = 0.0
            divider_segment: FloatPoint | None = None
            if len(long_edge_indices) >= 2:
                first_long_edge = edges[long_edge_indices[0]]
                second_long_edge = edges[long_edge_indices[1]]
                divider_segment = np.asarray(
                    (
                        np.mean(first_long_edge, axis=0),
                        np.mean(second_long_edge, axis=0),
                    ),
                    dtype=np.float64,
                )
                divider_score = _line_fraction(
                    dark_mask,
                    divider_segment[0],
                    divider_segment[1],
                    max(3, round(min(visible_width, visible_height) * 0.12)),
                )
            has_two_color_halves = len(significant_components) >= 2
            has_dark_divider = (
                divider_segment is not None
                and divider_score >= self._config.divider_dark_fraction
            )
            if (
                not has_two_color_halves
                and not has_dark_divider
                and not enclosure_geometry_confirmed
            ):
                continue
            divider_observation = (
                self._line_observation(divider_segment, is_bev=is_bev)
                if has_dark_divider and divider_segment is not None
                else None
            )
            component_areas = [
                int(stats[label, cv2.CC_STAT_AREA])
                for label in significant_components
            ]
            component_balance = min(component_areas) / max(component_areas)
            area_evidence = min(
                1.0,
                selected_area / max(2.0 * minimum_total_area, 1.0),
            )
            corroboration_score = (
                component_balance
                if has_two_color_halves
                else min(1.0, divider_score)
                if has_dark_divider
                else self._config.entrance_color_fraction
            )
            confidence = min(
                0.55,
                0.20
                + 0.15 * area_evidence
                + 0.10 * corroboration_score
                + 0.10 * min(1.0, purple_fill),
            )
            quality = {
                FieldFeatureQuality.PARTIAL,
                FieldFeatureQuality.ENTRANCE_UNRESOLVED,
                FieldFeatureQuality.SIDE_UNRESOLVED,
            }
            if divider_observation is None:
                quality.add(FieldFeatureQuality.DIVIDER_UNRESOLVED)
            if mapped_box.ground is None:
                quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
            candidates.append(
                SafeZoneObservation(
                    physical_color=color,
                    polygon_undistorted=mapped_box.pixels,
                    polygon_ground=mapped_box.ground,
                    entrance=None,
                    divider=divider_observation,
                    halves=(),
                    confidence=confidence,
                    quality=frozenset(quality),
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _safe_zone_component(
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
            expected_width_mm, expected_depth_mm = self._safe_zone_dimensions[color]
            if not (
                _within_expected(
                    width_px * mm_per_pixel,
                    expected_width_mm,
                    self._config.dimension_tolerance_fraction,
                )
                and _within_expected(
                    depth_px * mm_per_pixel,
                    expected_depth_mm,
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
                        self._start_side_mm,
                        self._config.dimension_tolerance_fraction,
                    )
                    and _within_expected(
                        second * scale,
                        self._start_side_mm,
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
        local_line_mask: Uint8Array,
        valid_mask: Uint8Array,
        *,
        is_bev: bool,
        anchor_points: tuple[FloatPoint, ...] = (),
    ) -> CenterCrossObservation | None:
        candidate_mask = cv2.bitwise_or(dark_mask, local_line_mask)
        raw_segments = _hough_segments(
            candidate_mask,
            min_length_fraction=self._config.center_min_axis_span_fraction,
            max_gap_fraction=self._config.center_max_gap_fraction,
        )
        height, width = candidate_mask.shape
        diagonal = math.hypot(width, height)
        distance_from_invalid = cv2.distanceTransform(
            np.where(valid_mask != 0, 255, 0).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        min_margin = diagonal * self._config.center_min_intersection_margin_fraction
        support_thickness = max(1, round(diagonal * 0.003))
        gap_mask = cv2.dilate(
            dark_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        support_mask = cv2.dilate(
            local_line_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (support_thickness, support_thickness),
            ),
            iterations=1,
        )
        candidates: list[_CenterAxisCandidate] = []
        for segment in raw_segments:
            local_support = _line_sample_fraction(support_mask, segment)
            gap_count = (
                0
                if local_support
                >= self._config.center_min_line_support_fraction
                else _line_gap_count(gap_mask, segment)
            )
            if (
                gap_count < self._config.center_min_gap_count
                and local_support
                < self._config.center_min_line_support_fraction
            ):
                continue
            if _mask_distance_at(
                distance_from_invalid,
                np.mean(segment, axis=0),
            ) < min_margin:
                continue
            candidates.append(
                _CenterAxisCandidate(segment, local_support, gap_count)
            )
        candidates = _deduplicate_center_axes(
            candidates,
            diagonal=diagonal,
            angle_tolerance_deg=(
                self._config.center_perpendicular_tolerance_deg
            ),
        )
        if not candidates:
            return None
        best_pair: tuple[
            _CenterAxisCandidate,
            _CenterAxisCandidate,
            FloatPoint,
        ] | None = None
        best_score = -1.0
        rejected_intersecting_pair = False
        for first_index, first_candidate in enumerate(candidates):
            for second_candidate in candidates[first_index + 1 :]:
                first = first_candidate.segment
                second = second_candidate.segment
                if not _perpendicular(
                    first,
                    second,
                    self._config.center_perpendicular_tolerance_deg,
                ):
                    continue
                intersection = _segment_intersection(first, second)
                if intersection is None:
                    continue
                anchor_aligned = False
                if anchor_points:
                    cosine_threshold = math.cos(
                        math.radians(
                            self._config.center_perpendicular_tolerance_deg
                            * 0.5
                        )
                    )
                    axis_directions = (
                        first[1] - first[0],
                        second[1] - second[0],
                    )
                    for anchor_point in anchor_points:
                        anchor_direction = anchor_point - intersection
                        anchor_norm = float(np.linalg.norm(anchor_direction))
                        if anchor_norm <= 1e-9:
                            continue
                        if any(
                            abs(
                                float(
                                    np.dot(anchor_direction, axis_direction)
                                    / (
                                        anchor_norm
                                        * np.linalg.norm(axis_direction)
                                    )
                                )
                            )
                            >= cosine_threshold
                            for axis_direction in axis_directions
                        ):
                            anchor_aligned = True
                            break
                first_balance = _axis_balance(first, intersection)
                second_balance = _axis_balance(second, intersection)
                required_balance = (
                    self._config.center_min_intersection_margin_fraction
                    if anchor_aligned
                    else self._config.center_min_axis_balance_fraction
                )
                if (
                    min(first_balance, second_balance) < required_balance
                    or _mask_distance_at(distance_from_invalid, intersection)
                    < min_margin
                ):
                    rejected_intersecting_pair = True
                    continue
                support = min(
                    first_candidate.local_support,
                    second_candidate.local_support,
                )
                score = (
                    _line_length(first) + _line_length(second)
                ) * (0.5 + support) * min(first_balance, second_balance)
                if anchor_aligned:
                    score += 2.0 * diagonal
                if score > best_score:
                    best_score = score
                    best_pair = (
                        first_candidate,
                        second_candidate,
                        intersection,
                    )

        quality: set[FieldFeatureQuality] = set()
        if best_pair is None:
            if rejected_intersecting_pair:
                return None
            quality.add(FieldFeatureQuality.PARTIAL)
            candidate = candidates[0]
            axis = self._line_observation(
                candidate.segment,
                is_bev=is_bev,
            )
            if axis.start_ground is None:
                quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
            confidence = min(
                0.50,
                0.25 + 0.5 * candidate.local_support,
            )
            return CenterCrossObservation(
                axes=(axis,),
                intersection_undistorted=None,
                intersection_ground=None,
                confidence=confidence,
                quality=frozenset(quality),
            )

        first_candidate, second_candidate, intersection = best_pair
        first = first_candidate.segment
        second = second_candidate.segment
        axes = (
            self._line_observation(first, is_bev=is_bev),
            self._line_observation(second, is_bev=is_bev),
        )
        mapped_intersection = self._map_points((intersection,), is_bev=is_bev)
        if mapped_intersection.ground is None:
            quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
        first_direction = first[1] - first[0]
        second_direction = second[1] - second[0]
        absolute_cosine = abs(
            float(
                np.dot(first_direction, second_direction)
                / (
                    np.linalg.norm(first_direction)
                    * np.linalg.norm(second_direction)
                )
            )
        )
        angle_quality = 1.0 - min(
            1.0,
            absolute_cosine
            / math.sin(
                math.radians(
                    self._config.center_perpendicular_tolerance_deg
                )
            ),
        )
        balance_quality = min(
            1.0,
            2.0 * min(
                _axis_balance(first, intersection),
                _axis_balance(second, intersection),
            ),
        )
        support_quality = min(
            1.0,
            (
                first_candidate.local_support
                + second_candidate.local_support
            )
            / max(
                2.0 * self._config.center_min_line_support_fraction,
                1e-9,
            ),
        )
        gap_quality = min(
            1.0,
            (
                first_candidate.gap_count
                + second_candidate.gap_count
            )
            / (2.0 * self._config.center_min_gap_count),
        )
        structure_quality = max(support_quality, gap_quality)
        confidence = min(
            0.95,
            0.45
            + 0.20 * angle_quality
            + 0.15 * balance_quality
            + 0.15 * structure_quality,
        )
        return CenterCrossObservation(
            axes=axes,
            intersection_undistorted=mapped_intersection.pixels[0],
            intersection_ground=(
                mapped_intersection.ground[0]
                if mapped_intersection.ground is not None
                else None
            ),
            confidence=confidence,
            quality=frozenset(quality),
        )

    def _boundary_features(
        self,
        image_bgr: Uint8Array,
        valid_mask: Uint8Array,
        center_cross: CenterCrossObservation | None,
        *,
        capture_timestamp_ns: int,
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
        support_segments = _hough_segments(
            edges,
            min_length_fraction=max(
                0.03,
                self._config.boundary_min_line_length_fraction * 0.30,
            ),
            max_gap_fraction=0.015,
        )

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
                        kind=BoundaryFeatureKind.FIELD_CORNER,
                        points_undistorted=mapped.pixels,
                        points_ground=mapped.ground,
                        capture_timestamp_ns=capture_timestamp_ns,
                        confidence=0.35,
                        quality=frozenset(quality),
                    )
                )
                if len(observations) >= self._config.boundary_max_features:
                    return tuple(observations)

        for segment in segments:
            mapped = self._map_points(segment, is_bev=is_bev)
            quality = {FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY}
            if mapped.ground is None:
                quality.add(FieldFeatureQuality.NO_GROUND_PROJECTION)
            interior_normal: tuple[float, float] | None = None
            line_offset: float | None = None
            if mapped.ground is not None:
                start = np.asarray(
                    (mapped.ground[0].x, mapped.ground[0].y),
                    dtype=np.float64,
                )
                end = np.asarray(
                    (mapped.ground[1].x, mapped.ground[1].y),
                    dtype=np.float64,
                )
                direction = end - start
                direction /= np.linalg.norm(direction)
                normal = np.asarray((-direction[1], direction[0]))
                midpoint = (start + end) * 0.5
                if float(np.dot(normal, -midpoint)) < 0.0:
                    normal = -normal
                interior_normal = (float(normal[0]), float(normal[1]))
                line_offset = -float(np.dot(normal, midpoint))
            vertical_support = sum(
                1
                for support in support_segments
                if _perpendicular(
                    segment,
                    support,
                    self._config.boundary_corner_tolerance_deg,
                )
                and min(
                    _point_to_line_distance(support[0], segment),
                    _point_to_line_distance(support[1], segment),
                ) <= 0.03 * diagonal
            )
            vertical_score = min(
                1.0,
                vertical_support
                / self._config.boundary_min_vertical_support_count,
            )
            contrast = _line_side_contrast(gray, interior_valid, segment)
            contrast_score = min(
                1.0,
                contrast / self._config.boundary_side_contrast_threshold,
            )
            confidence = min(
                0.75,
                0.2
                + 0.25 * _line_length(segment) / diagonal
                + 0.15 * vertical_score
                + 0.15 * contrast_score,
            )
            observations.append(
                BoundaryFeatureObservation(
                    kind=BoundaryFeatureKind.FENCE_BASE_SEGMENT,
                    points_undistorted=mapped.pixels,
                    points_ground=mapped.ground,
                    capture_timestamp_ns=capture_timestamp_ns,
                    confidence=confidence,
                    quality=frozenset(quality),
                    interior_normal_ground=interior_normal,
                    line_offset_mm=line_offset,
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
        include_boundary_features: bool = True,
    ) -> FieldFeatureDetectionResult:
        if not isinstance(include_boundary_features, bool):
            raise ValueError("include_boundary_features must be a boolean.")
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
        for safe_zone in safe_zones:
            if working.is_bev and safe_zone.polygon_ground is not None:
                assert self._ground_projector is not None
                zone_pixels = self._ground_projector.ground_to_bev_pixels(
                    safe_zone.polygon_ground
                )
                polygon = np.asarray(
                    [(pixel.u, pixel.v) for pixel in zone_pixels],
                    dtype=np.int32,
                )
            else:
                polygon = np.asarray(
                    [
                        (pixel.u, pixel.v)
                        for pixel in safe_zone.polygon_undistorted
                    ],
                    dtype=np.int32,
                )
            cv2.fillConvexPoly(colored, polygon, 255)
        exclusion = cv2.dilate(
            colored,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        center_mask = cv2.bitwise_and(
            masks["dark_marking"],
            cv2.bitwise_not(exclusion),
        )
        local_line_mask = cv2.bitwise_and(
            _local_dark_line_mask(
                working.image_bgr,
                working.valid_mask,
                self._config,
            ),
            cv2.bitwise_not(exclusion),
        )
        center_anchor_points: list[FloatPoint] = []
        for safe_zone in safe_zones:
            if working.is_bev and safe_zone.polygon_ground is not None:
                assert self._ground_projector is not None
                zone_pixels = self._ground_projector.ground_to_bev_pixels(
                    safe_zone.polygon_ground
                )
                center_anchor_points.append(
                    np.mean(
                        np.asarray(
                            [(pixel.u, pixel.v) for pixel in zone_pixels],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                )
            else:
                center_anchor_points.append(
                    np.mean(
                        np.asarray(
                            [
                                (pixel.u, pixel.v)
                                for pixel in safe_zone.polygon_undistorted
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                )
        center_cross = self._center_cross(
            center_mask,
            local_line_mask,
            working.valid_mask,
            is_bev=working.is_bev,
            anchor_points=tuple(center_anchor_points),
        )
        boundary_features = (
            self._boundary_features(
                working.image_bgr,
                working.valid_mask,
                center_cross,
                capture_timestamp_ns=frame.timestamp_ns,
                is_bev=working.is_bev,
            )
            if include_boundary_features
            else ()
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
        include_boundary_features: bool = True,
    ) -> RealtimeFieldFeatureResult:
        """检测最新帧，并把过期结果转换为显式空结果。"""

        try:
            result = self.detect(
                frame,
                undistorted_image_bgr,
                valid_mask=valid_mask,
                result_timestamp_ns=result_timestamp_ns,
                include_boundary_features=include_boundary_features,
            )
        except StaleObservationError as exc:
            return RealtimeFieldFeatureResult(
                result=None,
                dropped_stale_age_ms=exc.age_ms,
            )
        return RealtimeFieldFeatureResult(result=result)
