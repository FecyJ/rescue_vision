from __future__ import annotations

import pytest

from rescue_vision.evaluation.report import evaluate_records


def item(
    object_id: str,
    truth: str | None,
    prediction: str | None,
    *,
    truth_ground=None,
    predicted_ground=None,
) -> dict:
    return {
        "schema_version": 1,
        "sample_id": "sample-1",
        "object_id": object_id,
        "ground_truth_class": truth,
        "predicted_class": prediction,
        "confidence": 0.8,
        "ground_truth_ground_mm": truth_ground,
        "predicted_ground_mm": predicted_ground,
        "capture_timestamp_ns": 1_000_000_000,
        "result_timestamp_ns": 1_050_000_000,
        "tags": {"lighting": "bright"},
    }


def test_report_contains_per_class_confusion_ground_latency_and_failures() -> None:
    report = evaluate_records(
        [
            item("1", "hazard", "hazard", truth_ground=[0, 0], predicted_ground=[3, 4]),
            item("2", "hazard", None),
            item("3", "normal", "hazard"),
            item("4", None, "normal"),
        ],
        model_version="model-a",
        dataset_version="dataset-a",
        code_version="code-a",
    )
    hazard = report["per_class"]["hazard"]
    assert hazard["true_positive"] == 1
    assert hazard["false_positive"] == 1
    assert hazard["false_negative"] == 1
    assert hazard["precision"] == pytest.approx(0.5)
    assert hazard["recall"] == pytest.approx(0.5)
    assert report["ground_error_mm"]["mean"] == pytest.approx(5.0)
    assert report["end_to_end_latency_ms"]["p95"] == pytest.approx(50.0)
    assert report["failure_count"] == 3
    assert report["confusion_matrix"]["values"]["hazard"]["__missed__"] == 1
    assert report["versions"]["dataset"] == "dataset-a"


def test_report_rejects_non_monotonic_timestamps() -> None:
    record = item("1", "hazard", "hazard")
    record["result_timestamp_ns"] = 0
    with pytest.raises(ValueError, match="monotonic"):
        evaluate_records(
            [record],
            model_version="m",
            dataset_version="d",
            code_version="c",
        )
