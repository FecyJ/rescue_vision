"""用已知目标形状、ROI 分割和相机几何估计目标地面中心与足迹。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from time import monotonic_ns

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import (
    GroundPoint,
    RobotPoint3D,
    UndistortedPixel,
)
from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    TargetClass,
    TargetObservation,
)


FloatArray = npt.NDArray[np.float64]
Uint8Array = npt.NDArray[np.uint8]


class TargetGeometryShape(str, Enum):
    """当前比赛目标使用的可配置几何形状。"""

    BOX = "box"
    REGULAR_TETRAHEDRON = "regular_tetrahedron"


@dataclass(frozen=True, slots=True)
class BoxTargetGeometry:
    """平放盒体尺寸，局部 x/y/z 分别对应长、宽、高，单位 mm。"""

    length_mm: float
    width_mm: float
    height_mm: float
    shape: TargetGeometryShape = TargetGeometryShape.BOX

    def __post_init__(self) -> None:
        if self.shape is not TargetGeometryShape.BOX:
            raise ValueError("BoxTargetGeometry shape must be box.")
        for name in ("length_mm", "width_mm", "height_mm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True, slots=True)
class RegularTetrahedronTargetGeometry:
    """以正三角形面接地的正四面体，棱长单位 mm。"""

    edge_mm: float
    shape: TargetGeometryShape = TargetGeometryShape.REGULAR_TETRAHEDRON

    def __post_init__(self) -> None:
        if self.shape is not TargetGeometryShape.REGULAR_TETRAHEDRON:
            raise ValueError(
                "RegularTetrahedronTargetGeometry shape must be "
                "regular_tetrahedron."
            )
        if not math.isfinite(self.edge_mm) or self.edge_mm <= 0.0:
            raise ValueError("edge_mm must be finite and positive.")

    @property
    def height_mm(self) -> float:
        return self.edge_mm * math.sqrt(2.0 / 3.0)


TargetGeometry = BoxTargetGeometry | RegularTetrahedronTargetGeometry


@dataclass(frozen=True, slots=True)
class TargetGroundGeometryConfig:
    """四类目标尺寸和传统视觉三维模板拟合门限。"""

    enabled: bool
    green_supply: TargetGeometry
    black_core: TargetGeometry
    orange_injured: TargetGeometry
    blue_danger: TargetGeometry
    coarse_center_step_mm: float
    coarse_yaw_step_deg: float
    refine_center_step_mm: float
    refine_center_radius_mm: float
    refine_top_candidates: int
    refine_yaw_step_deg: float
    refine_yaw_radius_deg: float
    search_radius_margin_mm: float
    silhouette_weight: float
    contour_weight: float
    contact_weight: float
    contour_distance_scale_px: float
    contact_distance_scale_px: float
    max_contact_residual_px: float
    ambiguity_score_delta: float
    max_center_uncertainty_mm: float
    min_fit_score: float
    min_silhouette_iou: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if (
            isinstance(self.refine_top_candidates, bool)
            or not isinstance(self.refine_top_candidates, int)
            or not 1 <= self.refine_top_candidates <= 8
        ):
            raise ValueError(
                "refine_top_candidates must be an integer in [1, 8]."
            )
        geometries = {
            TargetClass.GREEN_SUPPLY: self.green_supply,
            TargetClass.BLACK_CORE: self.black_core,
            TargetClass.ORANGE_INJURED: self.orange_injured,
            TargetClass.BLUE_DANGER: self.blue_danger,
        }
        if not all(
            isinstance(
                geometry,
                (BoxTargetGeometry, RegularTetrahedronTargetGeometry),
            )
            for geometry in geometries.values()
        ):
            raise ValueError(
                "Each target class must have a supported target geometry."
            )
        for name in (
            "coarse_center_step_mm",
            "coarse_yaw_step_deg",
            "refine_center_step_mm",
            "refine_center_radius_mm",
            "refine_yaw_step_deg",
            "refine_yaw_radius_deg",
            "search_radius_margin_mm",
            "contour_distance_scale_px",
            "contact_distance_scale_px",
            "max_contact_residual_px",
            "max_center_uncertainty_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "silhouette_weight",
            "contour_weight",
            "contact_weight",
            "ambiguity_score_delta",
            "min_fit_score",
            "min_silhouette_iou",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        weight_sum = (
            self.silhouette_weight
            + self.contour_weight
            + self.contact_weight
        )
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-9, rel_tol=0.0):
            raise ValueError(
                "silhouette_weight + contour_weight + contact_weight "
                "must equal 1.0."
            )
        if self.refine_center_step_mm > self.coarse_center_step_mm:
            raise ValueError(
                "refine_center_step_mm must not exceed "
                "coarse_center_step_mm."
            )
        if self.refine_yaw_step_deg > self.coarse_yaw_step_deg:
            raise ValueError(
                "refine_yaw_step_deg must not exceed coarse_yaw_step_deg."
            )

    def geometry_for(self, target_class: TargetClass) -> TargetGeometry:
        if target_class not in COLOR_TARGET_CLASSES:
            raise ValueError(
                f"No geometry is available for {target_class.value}."
            )
        return getattr(self, target_class.value)


class GroundGeometryMethod(str, Enum):
    UNAVAILABLE = "unavailable"
    MODEL_FIT = "model_fit"


class GroundGeometryQuality(str, Enum):
    COLOR_MASK_UNAVAILABLE = "color_mask_unavailable"
    K0_UNAVAILABLE = "k0_unavailable"
    FIT_LOW_CONFIDENCE = "fit_low_confidence"
    CONTACT_INCONSISTENT = "contact_inconsistent"
    CENTER_AMBIGUOUS = "center_ambiguous"


@dataclass(frozen=True, slots=True)
class TargetGroundGeometry:
    """一帧目标对应的机器人地面系中心、足迹和估计质量。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    target_class: TargetClass
    contact_anchor_ground: GroundPoint | None
    center_ground: GroundPoint | None
    footprint_ground: tuple[GroundPoint, ...]
    yaw_rad: float | None
    yaw_symmetry_rad: float | None
    center_uncertainty_mm: float | None
    fit_score: float | None
    silhouette_iou: float | None
    contact_residual_px: float | None
    method: GroundGeometryMethod
    quality: frozenset[GroundGeometryQuality]

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative.")
        if (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns < 0
            or isinstance(self.result_timestamp_ns, bool)
            or not isinstance(self.result_timestamp_ns, int)
            or self.result_timestamp_ns < self.capture_timestamp_ns
        ):
            raise ValueError(
                "Geometry timestamps must satisfy non-negative "
                "capture_timestamp_ns <= result_timestamp_ns."
            )
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass.")
        if self.contact_anchor_ground is not None:
            _validate_ground_point(
                self.contact_anchor_ground,
                "contact_anchor_ground",
            )
        if self.center_ground is not None:
            _validate_ground_point(self.center_ground, "center_ground")
        if not all(
            isinstance(point, GroundPoint) for point in self.footprint_ground
        ):
            raise ValueError(
                "footprint_ground must contain only GroundPoint values."
            )
        for index, point in enumerate(self.footprint_ground):
            _validate_ground_point(point, f"footprint_ground[{index}]")
        if (self.center_ground is None) != (not self.footprint_ground):
            raise ValueError(
                "center_ground and footprint_ground must be available together."
            )
        if self.center_ground is None:
            if self.yaw_rad is not None or self.yaw_symmetry_rad is not None:
                raise ValueError(
                    "Unavailable center must not expose yaw values."
                )
            if self.method is GroundGeometryMethod.MODEL_FIT:
                raise ValueError(
                    "MODEL_FIT method requires an accepted center."
                )
        else:
            if self.yaw_rad is None or self.yaw_symmetry_rad is None:
                raise ValueError(
                    "Available center requires yaw and yaw symmetry."
                )
            if self.method is not GroundGeometryMethod.MODEL_FIT:
                raise ValueError(
                    "Available center requires MODEL_FIT method."
                )
            if self.yaw_symmetry_rad <= 0.0:
                raise ValueError(
                    "Available center requires positive yaw symmetry."
                )
            if not 0.0 <= self.yaw_rad < self.yaw_symmetry_rad + 1e-12:
                raise ValueError(
                    "yaw_rad must be normalized into the symmetry interval."
                )
        for name in (
            "yaw_rad",
            "yaw_symmetry_rad",
            "center_uncertainty_mm",
            "contact_residual_px",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("fit_score", "silhouette_iou"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if not isinstance(self.method, GroundGeometryMethod):
            raise ValueError("method must be a GroundGeometryMethod.")
        if not all(
            isinstance(item, GroundGeometryQuality) for item in self.quality
        ):
            raise ValueError(
                "quality must contain only GroundGeometryQuality values."
            )


@dataclass(frozen=True, slots=True)
class RealtimeTargetGroundGeometryResult:
    estimates: tuple[TargetGroundGeometry, ...]
    dropped_stale_age_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.estimates, tuple) or not all(
            isinstance(item, TargetGroundGeometry)
            for item in self.estimates
        ):
            raise ValueError(
                "estimates must be a tuple of TargetGroundGeometry values."
            )
        if self.dropped_stale_age_ms is not None:
            if (
                not math.isfinite(self.dropped_stale_age_ms)
                or self.dropped_stale_age_ms < 0.0
            ):
                raise ValueError(
                    "dropped_stale_age_ms must be finite and non-negative."
                )
            if self.estimates:
                raise ValueError(
                    "A stale result must not contain geometry estimates."
                )

    @property
    def stale_dropped(self) -> bool:
        return self.dropped_stale_age_ms is not None


class StaleGroundGeometryError(ValueError):
    def __init__(self, age_ms: float, max_age_ms: float) -> None:
        self.age_ms = float(age_ms)
        self.max_age_ms = float(max_age_ms)
        super().__init__(
            f"Target ground geometry age {self.age_ms:.3f} ms exceeds "
            f"{self.max_age_ms:.3f} ms."
        )


@dataclass(frozen=True, slots=True)
class _Model:
    vertices: FloatArray
    base_vertex_count: int
    yaw_symmetry_rad: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    center: GroundPoint
    yaw_rad: float
    score: float
    silhouette_iou: float
    contact_residual_px: float | None


def _validate_ground_point(point: GroundPoint, location: str) -> None:
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        raise ValueError(f"{location} must be finite.")


def _target_model(geometry: TargetGeometry) -> _Model:
    if isinstance(geometry, BoxTargetGeometry):
        half_length = geometry.length_mm / 2.0
        half_width = geometry.width_mm / 2.0
        base = np.asarray(
            (
                (half_length, half_width, 0.0),
                (-half_length, half_width, 0.0),
                (-half_length, -half_width, 0.0),
                (half_length, -half_width, 0.0),
            ),
            dtype=np.float64,
        )
        top = base.copy()
        top[:, 2] = geometry.height_mm
        symmetry = (
            math.pi / 2.0
            if math.isclose(
                geometry.length_mm,
                geometry.width_mm,
                abs_tol=1e-9,
                rel_tol=0.0,
            )
            else math.pi
        )
        return _Model(
            vertices=np.vstack((base, top)),
            base_vertex_count=4,
            yaw_symmetry_rad=symmetry,
        )

    edge = geometry.edge_mm
    base_radius = edge / math.sqrt(3.0)
    base = np.asarray(
        (
            (base_radius, 0.0, 0.0),
            (-base_radius / 2.0, edge / 2.0, 0.0),
            (-base_radius / 2.0, -edge / 2.0, 0.0),
        ),
        dtype=np.float64,
    )
    apex = np.asarray(
        ((0.0, 0.0, geometry.height_mm),),
        dtype=np.float64,
    )
    return _Model(
        vertices=np.vstack((base, apex)),
        base_vertex_count=3,
        yaw_symmetry_rad=2.0 * math.pi / 3.0,
    )


def _rotated_vertices(
    model: _Model,
    center: GroundPoint,
    yaw_rad: float,
) -> tuple[RobotPoint3D, ...]:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    rotation = np.asarray(
        ((cosine, -sine), (sine, cosine)),
        dtype=np.float64,
    )
    xy = model.vertices[:, :2] @ rotation.T
    xy[:, 0] += center.x
    xy[:, 1] += center.y
    return tuple(
        RobotPoint3D(float(x), float(y), float(z))
        for (x, y), z in zip(
            xy,
            model.vertices[:, 2],
            strict=True,
        )
    )


def _point_segment_distance(
    point: UndistortedPixel,
    start: UndistortedPixel,
    end: UndistortedPixel,
) -> float:
    segment = np.asarray((end.u - start.u, end.v - start.v))
    offset = np.asarray((point.u - start.u, point.v - start.v))
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return float(np.linalg.norm(offset))
    ratio = min(1.0, max(0.0, float(np.dot(offset, segment)) / denominator))
    nearest = np.asarray((start.u, start.v)) + ratio * segment
    return float(
        np.linalg.norm(np.asarray((point.u, point.v)) - nearest)
    )


def _range_values(
    start: float,
    stop: float,
    step: float,
) -> tuple[float, ...]:
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] < stop - 1e-9:
        values.append(stop)
    return tuple(values)


def _symmetric_values(radius: float, step: float) -> tuple[float, ...]:
    count = int(math.floor(radius / step + 1e-9))
    values = {index * step for index in range(-count, count + 1)}
    if count * step < radius - 1e-9:
        values.update((-radius, radius))
    return tuple(sorted(values))


class TargetGroundGeometryEstimator:
    """对四类目标的 ROI 颜色掩码执行已知三维形状模板拟合。"""

    def __init__(
        self,
        config: TargetGroundGeometryConfig,
        *,
        ground_projector: GroundProjector,
        max_observation_age_ms: float,
    ) -> None:
        if not isinstance(config, TargetGroundGeometryConfig):
            raise ValueError(
                "config must be a TargetGroundGeometryConfig."
            )
        if not config.enabled:
            raise ValueError(
                "TargetGroundGeometryEstimator requires enabled config."
            )
        if not isinstance(ground_projector, GroundProjector):
            raise ValueError("ground_projector must be a GroundProjector.")
        if not ground_projector.supports_robot_projection:
            raise ValueError(
                "TargetGroundGeometryEstimator requires ground calibration "
                "with full camera extrinsics."
            )
        converted_age = float(max_observation_age_ms)
        if not math.isfinite(converted_age) or converted_age <= 0.0:
            raise ValueError(
                "max_observation_age_ms must be finite and positive."
            )
        self._config = config
        self._ground_projector = ground_projector
        self._max_observation_age_ms = converted_age

    @property
    def config(self) -> TargetGroundGeometryConfig:
        return self._config

    def _unavailable(
        self,
        observation: TargetObservation,
        result_timestamp_ns: int,
        quality: set[GroundGeometryQuality],
        *,
        candidate: _Candidate | None = None,
        uncertainty_mm: float | None = None,
    ) -> TargetGroundGeometry:
        return TargetGroundGeometry(
            frame_sequence=observation.frame_sequence,
            capture_timestamp_ns=observation.capture_timestamp_ns,
            result_timestamp_ns=result_timestamp_ns,
            target_class=observation.target_class,
            contact_anchor_ground=observation.ground_point,
            center_ground=None,
            footprint_ground=(),
            yaw_rad=None,
            yaw_symmetry_rad=None,
            center_uncertainty_mm=uncertainty_mm,
            fit_score=candidate.score if candidate is not None else None,
            silhouette_iou=(
                candidate.silhouette_iou
                if candidate is not None
                else None
            ),
            contact_residual_px=(
                candidate.contact_residual_px
                if candidate is not None
                else None
            ),
            method=GroundGeometryMethod.UNAVAILABLE,
            quality=frozenset(quality),
        )

    def _score_candidate(
        self,
        observation: TargetObservation,
        observed_mask: Uint8Array,
        observed_area: int,
        observed_distance: npt.NDArray[np.float32],
        model: _Model,
        center: GroundPoint,
        yaw_rad: float,
    ) -> _Candidate | None:
        try:
            pixels = self._ground_projector.project_robot_points(
                _rotated_vertices(model, center, yaw_rad)
            )
        except ValueError:
            return None
        roi = observation.color_segmentation.roi_box
        local_points = np.asarray(
            [
                (pixel.u - roi.x_min, pixel.v - roi.y_min)
                for pixel in pixels
            ],
            dtype=np.float32,
        )
        hull = cv2.convexHull(local_points).reshape(-1, 2)
        if len(hull) < 3 or not np.all(np.isfinite(hull)):
            return None
        predicted = np.zeros_like(observed_mask)
        cv2.fillConvexPoly(
            predicted,
            np.rint(hull).astype(np.int32),
            255,
        )
        predicted_area = int(cv2.countNonZero(predicted))
        if predicted_area == 0:
            return None
        intersection = int(
            cv2.countNonZero(cv2.bitwise_and(predicted, observed_mask))
        )
        union = observed_area + predicted_area - intersection
        silhouette_iou = intersection / union if union > 0 else 0.0

        predicted_boundary = np.zeros_like(observed_mask)
        cv2.polylines(
            predicted_boundary,
            [np.rint(hull).astype(np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
        boundary_locations = predicted_boundary > 0
        if np.any(boundary_locations):
            mean_distance = float(
                np.mean(observed_distance[boundary_locations])
            )
            contour_score = math.exp(
                -mean_distance
                / self._config.contour_distance_scale_px
            )
        else:
            contour_score = 0.0

        contact_residual: float | None = None
        if observation.k0 is not None:
            base_pixels = pixels[: model.base_vertex_count]
            contact_residual = min(
                _point_segment_distance(
                    observation.k0,
                    base_pixels[index],
                    base_pixels[(index + 1) % len(base_pixels)],
                )
                for index in range(len(base_pixels))
            )
            contact_score = math.exp(
                -contact_residual
                / self._config.contact_distance_scale_px
            )
            weight_sum = 1.0
        else:
            contact_score = 0.0
            weight_sum = (
                self._config.silhouette_weight
                + self._config.contour_weight
            )

        score = (
            self._config.silhouette_weight * silhouette_iou
            + self._config.contour_weight * contour_score
            + (
                self._config.contact_weight * contact_score
                if observation.k0 is not None
                else 0.0
            )
        ) / weight_sum
        return _Candidate(
            center=center,
            yaw_rad=yaw_rad % model.yaw_symmetry_rad,
            score=min(1.0, max(0.0, score)),
            silhouette_iou=silhouette_iou,
            contact_residual_px=contact_residual,
        )

    def _coarse_candidates(
        self,
        observation: TargetObservation,
        observed_mask: Uint8Array,
        observed_area: int,
        observed_distance: npt.NDArray[np.float32],
        model: _Model,
    ) -> list[_Candidate]:
        """从 K0/框底锚点的物理接触假设生成稀疏粗候选。

        K0 只可能是可见底面顶点或连续底边中点。直接枚举这些接触假设，
        比在整个外接圆内逐点扫 ``center_x / center_y / yaw`` 少一个数量级，
        同时仍由 ``search_radius_margin_mm`` 容纳锚点误差。
        """

        if observation.ground_point is not None:
            anchor = observation.ground_point
        else:
            box = observation.box
            anchor = self._ground_projector.pixel_to_ground(
                UndistortedPixel(
                    (box.x_min + box.x_max) / 2.0,
                    box.y_max,
                )
            )

        uncovered_margin = max(
            0.0,
            self._config.search_radius_margin_mm
            - self._config.refine_center_radius_mm,
        )
        if uncovered_margin <= 1e-9:
            correction_centers = (GroundPoint(0.0, 0.0),)
        else:
            diagonal = uncovered_margin / math.sqrt(2.0)
            correction_centers = tuple(
                GroundPoint(dx, dy)
                for dx, dy in (
                    (0.0, 0.0),
                    (uncovered_margin, 0.0),
                    (-uncovered_margin, 0.0),
                    (0.0, uncovered_margin),
                    (0.0, -uncovered_margin),
                    (diagonal, diagonal),
                    (diagonal, -diagonal),
                    (-diagonal, diagonal),
                    (-diagonal, -diagonal),
                )
            )
        yaw_step = math.radians(self._config.coarse_yaw_step_deg)
        yaws = _range_values(
            0.0,
            max(0.0, model.yaw_symmetry_rad - yaw_step),
            yaw_step,
        )
        candidates: list[_Candidate] = []
        seen: set[tuple[float, float, float]] = set()
        base_xy = model.vertices[: model.base_vertex_count, :2]
        contact_xy = np.vstack(
            (
                base_xy,
                (base_xy + np.roll(base_xy, -1, axis=0)) / 2.0,
            )
        )
        for yaw in yaws:
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            rotation = np.asarray(
                ((cosine, -sine), (sine, cosine)),
                dtype=np.float64,
            )
            rotated_contacts = contact_xy @ rotation.T
            for contact_x, contact_y in rotated_contacts:
                nominal = GroundPoint(
                    anchor.x - float(contact_x),
                    anchor.y - float(contact_y),
                )
                for correction in correction_centers:
                    center = GroundPoint(
                        nominal.x + correction.x,
                        nominal.y + correction.y,
                    )
                    key = (
                        round(center.x, 9),
                        round(center.y, 9),
                        round(yaw % model.yaw_symmetry_rad, 12),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = self._score_candidate(
                        observation,
                        observed_mask,
                        observed_area,
                        observed_distance,
                        model,
                        center,
                        yaw,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _score_refined_candidate(
        self,
        observation: TargetObservation,
        observed_mask: Uint8Array,
        observed_area: int,
        observed_distance: npt.NDArray[np.float32],
        model: _Model,
        center: GroundPoint,
        yaw_rad: float,
        coarse: _Candidate,
    ) -> _Candidate | None:
        if (
            math.hypot(
                center.x - coarse.center.x,
                center.y - coarse.center.y,
            )
            > self._config.refine_center_radius_mm + 1e-9
        ):
            return None
        return self._score_candidate(
            observation,
            observed_mask,
            observed_area,
            observed_distance,
            model,
            center,
            yaw_rad,
        )

    def _refined_candidates(
        self,
        observation: TargetObservation,
        observed_mask: Uint8Array,
        observed_area: int,
        observed_distance: npt.NDArray[np.float32],
        model: _Model,
        coarse: _Candidate,
    ) -> list[_Candidate]:
        """先分别收敛中心和朝向，再做一个最小联合邻域搜索。"""

        candidates: list[_Candidate] = [coarse]
        center_offsets = _symmetric_values(
            self._config.refine_center_radius_mm,
            self._config.refine_center_step_mm,
        )
        best_center = coarse
        for axis in (0, 1, 0, 1):
            axis_candidates: list[_Candidate] = []
            for offset in center_offsets:
                dx = offset if axis == 0 else 0.0
                dy = offset if axis == 1 else 0.0
                center = GroundPoint(
                    best_center.center.x + dx,
                    best_center.center.y + dy,
                )
                candidate = self._score_refined_candidate(
                    observation,
                    observed_mask,
                    observed_area,
                    observed_distance,
                    model,
                    center,
                    coarse.yaw_rad,
                    coarse,
                )
                if candidate is not None:
                    axis_candidates.append(candidate)
            candidates.extend(axis_candidates)
            if axis_candidates:
                best_center = max(
                    axis_candidates,
                    key=lambda item: item.score,
                )

        yaw_radius = math.radians(self._config.refine_yaw_radius_deg)
        yaw_step = math.radians(self._config.refine_yaw_step_deg)
        yaw_offsets = _range_values(-yaw_radius, yaw_radius, yaw_step)
        yaw_candidates: list[_Candidate] = []
        for yaw_offset in yaw_offsets:
            candidate = self._score_refined_candidate(
                observation,
                observed_mask,
                observed_area,
                observed_distance,
                model,
                best_center.center,
                coarse.yaw_rad + yaw_offset,
                coarse,
            )
            if candidate is not None:
                yaw_candidates.append(candidate)
        candidates.extend(yaw_candidates)
        best_yaw = max(
            yaw_candidates or [best_center],
            key=lambda item: item.score,
        )

        center_step = self._config.refine_center_step_mm
        for dx in (-center_step, 0.0, center_step):
            for dy in (-center_step, 0.0, center_step):
                center = GroundPoint(
                    best_yaw.center.x + dx,
                    best_yaw.center.y + dy,
                )
                for yaw_offset in (-yaw_step, 0.0, yaw_step):
                    candidate = self._score_refined_candidate(
                        observation,
                        observed_mask,
                        observed_area,
                        observed_distance,
                        model,
                        center,
                        best_yaw.yaw_rad + yaw_offset,
                        coarse,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _estimate_one(
        self,
        observation: TargetObservation,
        result_timestamp_ns: int,
    ) -> TargetGroundGeometry:
        quality: set[GroundGeometryQuality] = set()
        segmentation = observation.color_segmentation
        if not np.any(segmentation.mask):
            quality.add(GroundGeometryQuality.COLOR_MASK_UNAVAILABLE)
            return self._unavailable(
                observation,
                result_timestamp_ns,
                quality,
            )
        if observation.k0 is None or observation.ground_point is None:
            quality.add(GroundGeometryQuality.K0_UNAVAILABLE)

        observed_mask = np.ascontiguousarray(segmentation.mask)
        observed_area = int(cv2.countNonZero(observed_mask))
        observed_boundary = cv2.morphologyEx(
            observed_mask,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        if cv2.countNonZero(observed_boundary) == 0:
            quality.add(GroundGeometryQuality.COLOR_MASK_UNAVAILABLE)
            return self._unavailable(
                observation,
                result_timestamp_ns,
                quality,
            )
        observed_distance = cv2.distanceTransform(
            cv2.bitwise_not(observed_boundary),
            cv2.DIST_L2,
            3,
        )

        geometry = self._config.geometry_for(observation.target_class)
        model = _target_model(geometry)
        coarse_candidates = self._coarse_candidates(
            observation,
            observed_mask,
            observed_area,
            observed_distance,
            model,
        )
        if not coarse_candidates:
            quality.add(GroundGeometryQuality.FIT_LOW_CONFIDENCE)
            return self._unavailable(
                observation,
                result_timestamp_ns,
                quality,
            )
        selected_coarse: list[_Candidate] = []
        minimum_separation = self._config.coarse_center_step_mm / 2.0
        for candidate in sorted(
            coarse_candidates,
            key=lambda item: item.score,
            reverse=True,
        ):
            if any(
                math.hypot(
                    candidate.center.x - selected.center.x,
                    candidate.center.y - selected.center.y,
                )
                < minimum_separation
                for selected in selected_coarse
            ):
                continue
            selected_coarse.append(candidate)
            if (
                len(selected_coarse)
                >= self._config.refine_top_candidates
            ):
                break
        refined = [
            candidate
            for coarse in selected_coarse
            for candidate in self._refined_candidates(
                observation,
                observed_mask,
                observed_area,
                observed_distance,
                model,
                coarse,
            )
        ]
        candidates = refined or selected_coarse
        best = max(candidates, key=lambda item: item.score)
        plausible = [
            candidate
            for candidate in candidates
            if candidate.score
            >= best.score - self._config.ambiguity_score_delta
        ]
        uncertainty_mm = math.sqrt(
            sum(
                (candidate.center.x - best.center.x) ** 2
                + (candidate.center.y - best.center.y) ** 2
                for candidate in plausible
            )
            / len(plausible)
        )

        if (
            best.score < self._config.min_fit_score
            or best.silhouette_iou < self._config.min_silhouette_iou
        ):
            quality.add(GroundGeometryQuality.FIT_LOW_CONFIDENCE)
        if (
            best.contact_residual_px is not None
            and best.contact_residual_px
            > self._config.max_contact_residual_px
        ):
            quality.add(GroundGeometryQuality.CONTACT_INCONSISTENT)
        if uncertainty_mm > self._config.max_center_uncertainty_mm:
            quality.add(GroundGeometryQuality.CENTER_AMBIGUOUS)
        if quality & {
            GroundGeometryQuality.FIT_LOW_CONFIDENCE,
            GroundGeometryQuality.CONTACT_INCONSISTENT,
            GroundGeometryQuality.CENTER_AMBIGUOUS,
        }:
            return self._unavailable(
                observation,
                result_timestamp_ns,
                quality,
                candidate=best,
                uncertainty_mm=uncertainty_mm,
            )

        vertices = _rotated_vertices(model, best.center, best.yaw_rad)
        footprint = tuple(
            GroundPoint(vertex.x, vertex.y)
            for vertex in vertices[: model.base_vertex_count]
        )
        return TargetGroundGeometry(
            frame_sequence=observation.frame_sequence,
            capture_timestamp_ns=observation.capture_timestamp_ns,
            result_timestamp_ns=result_timestamp_ns,
            target_class=observation.target_class,
            contact_anchor_ground=observation.ground_point,
            center_ground=best.center,
            footprint_ground=footprint,
            yaw_rad=best.yaw_rad,
            yaw_symmetry_rad=model.yaw_symmetry_rad,
            center_uncertainty_mm=uncertainty_mm,
            fit_score=best.score,
            silhouette_iou=best.silhouette_iou,
            contact_residual_px=best.contact_residual_px,
            method=GroundGeometryMethod.MODEL_FIT,
            quality=frozenset(quality),
        )

    def estimate(
        self,
        observations: tuple[TargetObservation, ...]
        | list[TargetObservation],
        *,
        result_timestamp_ns: int | None = None,
    ) -> tuple[TargetGroundGeometry, ...]:
        """按输入顺序估计同一帧目标；过期结果不会返回。"""

        observation_tuple = tuple(observations)
        if not all(
            isinstance(item, TargetObservation)
            for item in observation_tuple
        ):
            raise ValueError(
                "observations must contain only TargetObservation values."
            )
        if not observation_tuple:
            return ()
        frame_keys = {
            (
                item.frame_sequence,
                item.capture_timestamp_ns,
                item.image_size,
            )
            for item in observation_tuple
        }
        if len(frame_keys) != 1:
            raise ValueError(
                "All target observations must belong to the same frame."
            )
        capture_timestamp_ns = observation_tuple[0].capture_timestamp_ns
        source_result_timestamp_ns = max(
            item.result_timestamp_ns for item in observation_tuple
        )
        before_fitting_timestamp_ns = (
            monotonic_ns() if result_timestamp_ns is None else result_timestamp_ns
        )
        if (
            isinstance(before_fitting_timestamp_ns, bool)
            or not isinstance(before_fitting_timestamp_ns, int)
            or before_fitting_timestamp_ns < source_result_timestamp_ns
        ):
            raise ValueError(
                "result_timestamp_ns must be an integer not earlier than "
                "the source observation result timestamp."
            )
        before_fitting_age_ms = (
            before_fitting_timestamp_ns - capture_timestamp_ns
        ) / 1_000_000.0
        if before_fitting_age_ms > self._max_observation_age_ms:
            raise StaleGroundGeometryError(
                before_fitting_age_ms,
                self._max_observation_age_ms,
            )
        provisional = tuple(
            self._estimate_one(item, item.result_timestamp_ns)
            for item in observation_tuple
        )
        completed_timestamp_ns = (
            monotonic_ns() if result_timestamp_ns is None else result_timestamp_ns
        )
        age_ms = (
            completed_timestamp_ns - capture_timestamp_ns
        ) / 1_000_000.0
        if age_ms > self._max_observation_age_ms:
            raise StaleGroundGeometryError(
                age_ms,
                self._max_observation_age_ms,
            )
        return tuple(
            replace(
                estimate,
                result_timestamp_ns=completed_timestamp_ns,
            )
            for estimate in provisional
        )

    def estimate_realtime(
        self,
        observations: tuple[TargetObservation, ...]
        | list[TargetObservation],
        *,
        result_timestamp_ns: int | None = None,
    ) -> RealtimeTargetGroundGeometryResult:
        try:
            estimates = self.estimate(
                observations,
                result_timestamp_ns=result_timestamp_ns,
            )
        except StaleGroundGeometryError as exc:
            return RealtimeTargetGroundGeometryResult(
                estimates=(),
                dropped_stale_age_ms=exc.age_ms,
            )
        return RealtimeTargetGroundGeometryResult(estimates=estimates)


__all__ = [
    "BoxTargetGeometry",
    "GroundGeometryMethod",
    "GroundGeometryQuality",
    "RealtimeTargetGroundGeometryResult",
    "RegularTetrahedronTargetGeometry",
    "StaleGroundGeometryError",
    "TargetGeometry",
    "TargetGeometryShape",
    "TargetGroundGeometry",
    "TargetGroundGeometryConfig",
    "TargetGroundGeometryEstimator",
]
