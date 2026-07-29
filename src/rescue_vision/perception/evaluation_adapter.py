"""统一目标观测到现有逐对象评测 JSONL 的适配。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rescue_vision.geometry.types import GroundPoint
from rescue_vision.perception.types import (
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)


@dataclass(frozen=True, slots=True)
class TargetAnnotation:
    object_id: str
    target_class: TargetClass
    box: UndistortedBoundingBox
    ground_point: GroundPoint | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty.")


def observations_to_evaluation_records(
    *,
    sample_id: str,
    annotations: Sequence[TargetAnnotation],
    observations: Sequence[TargetObservation],
    capture_timestamp_ns: int,
    result_timestamp_ns: int,
    iou_threshold: float = 0.5,
    tags: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """按类别无关 IoU 一对一匹配，并输出逐对象评测记录。"""

    if not sample_id.strip():
        raise ValueError("sample_id must be non-empty.")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")
    object_ids = [annotation.object_id for annotation in annotations]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("annotations must contain unique object_id values.")
    if result_timestamp_ns < capture_timestamp_ns:
        raise ValueError("result_timestamp_ns must not precede capture_timestamp_ns.")
    for observation in observations:
        if (
            observation.capture_timestamp_ns != capture_timestamp_ns
            or observation.result_timestamp_ns != result_timestamp_ns
        ):
            raise ValueError("All observations must share the supplied frame timestamps.")

    candidates = sorted(
        (
            (-annotation.box.iou(observation.box), truth_index, prediction_index)
            for truth_index, annotation in enumerate(annotations)
            for prediction_index, observation in enumerate(observations)
            if annotation.box.iou(observation.box) >= iou_threshold
        )
    )
    truth_matches: dict[int, int] = {}
    prediction_matches: set[int] = set()
    for _, truth_index, prediction_index in candidates:
        if truth_index in truth_matches or prediction_index in prediction_matches:
            continue
        truth_matches[truth_index] = prediction_index
        prediction_matches.add(prediction_index)

    common_tags = dict(tags or {})
    records: list[dict[str, object]] = []
    for truth_index, annotation in enumerate(annotations):
        prediction_index = truth_matches.get(truth_index)
        observation = (
            observations[prediction_index]
            if prediction_index is not None
            else None
        )
        records.append(
            _record(
                sample_id=sample_id,
                object_id=annotation.object_id,
                truth=annotation,
                observation=observation,
                capture_timestamp_ns=capture_timestamp_ns,
                result_timestamp_ns=result_timestamp_ns,
                tags=common_tags,
            )
        )

    for prediction_index, observation in enumerate(observations):
        if prediction_index in prediction_matches:
            continue
        records.append(
            _record(
                sample_id=sample_id,
                object_id=f"__prediction__{prediction_index:04d}",
                truth=None,
                observation=observation,
                capture_timestamp_ns=capture_timestamp_ns,
                result_timestamp_ns=result_timestamp_ns,
                tags=common_tags,
            )
        )
    return records


def _ground(point: GroundPoint | None) -> list[float] | None:
    return [point.x, point.y] if point is not None else None


def _record(
    *,
    sample_id: str,
    object_id: str,
    truth: TargetAnnotation | None,
    observation: TargetObservation | None,
    capture_timestamp_ns: int,
    result_timestamp_ns: int,
    tags: Mapping[str, str],
) -> dict[str, object]:
    confidence = observation.detection_confidence if observation is not None else None
    return {
        "sample_id": sample_id,
        "object_id": object_id,
        "ground_truth_class": truth.target_class.value if truth else None,
        "predicted_class": observation.target_class.value if observation else None,
        "confidence": confidence,
        "quality": (
            sorted(item.value for item in observation.quality)
            if observation is not None
            else []
        ),
        "k0_confidence": (
            observation.k0_confidence if observation is not None else None
        ),
        "ground_truth_ground_mm": _ground(truth.ground_point) if truth else None,
        "predicted_ground_mm": (
            _ground(observation.ground_point) if observation else None
        ),
        "capture_timestamp_ns": capture_timestamp_ns,
        "result_timestamp_ns": result_timestamp_ns,
        "tags": dict(tags),
    }
