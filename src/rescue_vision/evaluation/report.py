"""分类、地面点和端到端时延的统一离线评测。"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from rescue_vision.perception.types import TargetClass


BACKGROUND = "__background__"
MISSED = "__missed__"
DEFAULT_CLASSES = tuple(target_class.value for target_class in TargetClass)


def _ground_point(value: object, location: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{location} must be [x_mm, y_mm] or null.")
    return array


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    model_version: str,
    dataset_version: str,
    code_version: str,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> dict[str, Any]:
    """评测已经完成匹配的逐对象记录。

    ``ground_truth_class=null`` 表示误检，``predicted_class=null`` 表示漏检。
    """

    if not model_version or not dataset_version or not code_version:
        raise ValueError("model, dataset and code versions must be non-empty.")
    if not records:
        raise ValueError("Evaluation records must not be empty.")
    if (
        not classes
        or len(set(classes)) != len(classes)
        or any(not isinstance(label, str) or not label for label in classes)
        or BACKGROUND in classes
        or MISSED in classes
    ):
        raise ValueError("classes must contain unique, non-empty class names.")
    allowed_labels = set(classes)

    normalized: list[tuple[str | None, str | None]] = []
    ground_errors: list[float] = []
    sample_latencies_ms: dict[str, float] = {}
    failures: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for index, record in enumerate(records):
        if record.get("schema_version") != 1:
            raise ValueError(f"Record {index} schema_version must be 1.")
        sample_id = record.get("sample_id")
        object_id = record.get("object_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Record {index} has invalid sample_id.")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"Record {index} has invalid object_id.")
        key = (sample_id, object_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate evaluation key {key}.")
        seen_keys.add(key)

        truth = record.get("ground_truth_class")
        prediction = record.get("predicted_class")
        confidence = record.get("confidence")
        if truth is not None and (not isinstance(truth, str) or not truth):
            raise ValueError(f"Record {index} has invalid ground_truth_class.")
        if prediction is not None and (
            not isinstance(prediction, str) or not prediction
        ):
            raise ValueError(f"Record {index} has invalid predicted_class.")
        if truth is None and prediction is None:
            raise ValueError(f"Record {index} cannot have both classes null.")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"Record {index} confidence must be in [0, 1] or null.")
        for field_name, label in (
            ("ground_truth_class", truth),
            ("predicted_class", prediction),
        ):
            if label is not None and label not in allowed_labels:
                raise ValueError(
                    f"Record {index} {field_name} {label!r} is outside "
                    f"the configured classes {list(classes)!r}."
                )
        normalized.append((truth, prediction))

        truth_ground = _ground_point(
            record.get("ground_truth_ground_mm"),
            f"Record {index}.ground_truth_ground_mm",
        )
        predicted_ground = _ground_point(
            record.get("predicted_ground_mm"),
            f"Record {index}.predicted_ground_mm",
        )
        if truth_ground is not None and predicted_ground is not None:
            ground_errors.append(float(np.linalg.norm(predicted_ground - truth_ground)))

        capture_ns = record.get("capture_timestamp_ns")
        result_ns = record.get("result_timestamp_ns")
        if capture_ns is not None or result_ns is not None:
            if (
                isinstance(capture_ns, bool)
                or isinstance(result_ns, bool)
                or not isinstance(capture_ns, int)
                or not isinstance(result_ns, int)
                or result_ns < capture_ns
            ):
                raise ValueError(
                    f"Record {index} timestamps must be monotonic integer ns."
                )
            latency_ms = (result_ns - capture_ns) / 1_000_000.0
            previous_latency = sample_latencies_ms.setdefault(sample_id, latency_ms)
            if previous_latency != latency_ms:
                raise ValueError(
                    f"Sample {sample_id!r} has inconsistent frame timestamps."
                )

        if truth != prediction:
            if truth is None:
                failure_type = "false_positive"
            elif prediction is None:
                failure_type = "missed"
            else:
                failure_type = "misclassified"
            failures.append(
                {
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "type": failure_type,
                    "ground_truth_class": truth,
                    "predicted_class": prediction,
                    "confidence": confidence,
                    "tags": record.get("tags", {}),
                }
            )

    ordered_labels = list(classes)
    matrix_rows = ordered_labels + [BACKGROUND]
    matrix_columns = ordered_labels + [MISSED]
    matrix: dict[str, Counter[str]] = {
        row: Counter({column: 0 for column in matrix_columns})
        for row in matrix_rows
    }
    for truth, prediction in normalized:
        matrix[truth or BACKGROUND][prediction or MISSED] += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in ordered_labels:
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in normalized
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in normalized
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in normalized
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = None
        if precision is not None and recall is not None:
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
        per_class[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    latencies_ms = list(sample_latencies_ms.values())
    warnings: list[dict[str, str]] = []
    danger_label = TargetClass.BLUE_DANGER.value
    if not any(truth == danger_label for truth, _ in normalized):
        warnings.append(
            {
                "code": "danger_class_has_no_ground_truth",
                "class": danger_label,
                "message": "blue_danger has no ground-truth samples.",
            }
        )
    return {
        "schema_version": 1,
        "versions": {
            "model": model_version,
            "dataset": dataset_version,
            "code": code_version,
        },
        "record_count": len(records),
        "classes": ordered_labels,
        "per_class": per_class,
        "confusion_matrix": {
            "rows_ground_truth": matrix_rows,
            "columns_prediction": matrix_columns,
            "values": {
                row: dict(matrix[row])
                for row in matrix_rows
            },
        },
        "ground_error_mm": {
            "count": len(ground_errors),
            "mean": float(np.mean(ground_errors)) if ground_errors else None,
            "median": _percentile(ground_errors, 50),
            "p95": _percentile(ground_errors, 95),
            "max": max(ground_errors, default=None),
        },
        "end_to_end_latency_ms": {
            "count": len(latencies_ms),
            "mean": float(np.mean(latencies_ms)) if latencies_ms else None,
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "max": max(latencies_ms, default=None),
        },
        "failure_count": len(failures),
        "failures": failures,
        "warnings": warnings,
    }
