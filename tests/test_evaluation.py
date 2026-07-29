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
            item(
                "1",
                "blue_danger",
                "blue_danger",
                truth_ground=[0, 0],
                predicted_ground=[3, 4],
            ),
            item("2", "blue_danger", None),
            item("3", "green_supply", "blue_danger"),
            item("4", None, "green_supply"),
        ],
    )
    danger = report["per_class"]["blue_danger"]
    assert danger["true_positive"] == 1
    assert danger["false_positive"] == 1
    assert danger["false_negative"] == 1
    assert danger["precision"] == pytest.approx(0.5)
    assert danger["recall"] == pytest.approx(0.5)
    assert report["ground_error_mm"]["mean"] == pytest.approx(5.0)
    assert report["end_to_end_latency_ms"]["p95"] == pytest.approx(50.0)
    assert report["end_to_end_latency_ms"]["count"] == 1
    assert report["failure_count"] == 3
    assert report["confusion_matrix"]["values"]["blue_danger"]["__missed__"] == 1


def test_report_rejects_non_monotonic_timestamps() -> None:
    record = item("1", "blue_danger", "blue_danger")
    record["result_timestamp_ns"] = 0
    with pytest.raises(ValueError, match="monotonic"):
        evaluate_records(
            [record],
        )


def test_report_rejects_empty_and_unknown_vocabulary() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_records(
            [],
        )
    with pytest.raises(ValueError, match="outside the configured classes"):
        evaluate_records(
            [item("1", "hazard", "hazard")],
        )


def test_report_keeps_danger_visible_when_dataset_has_no_danger_truth() -> None:
    report = evaluate_records(
        [item("1", "green_supply", None)],
    )
    assert "blue_danger" in report["classes"]
    assert report["per_class"]["blue_danger"]["true_positive"] == 0
    assert report["per_class"]["blue_danger"]["precision"] is None
    assert report["per_class"]["blue_danger"]["recall"] is None
    assert report["per_class"]["blue_danger"]["f1"] is None
    assert report["warnings"][0]["code"] == "danger_class_has_no_ground_truth"


def test_complete_failure_f1_is_zero() -> None:
    report = evaluate_records(
        [
            item("1", "blue_danger", None),
            item("2", None, "blue_danger"),
        ],
    )
    assert report["per_class"]["blue_danger"]["f1"] == 0.0
