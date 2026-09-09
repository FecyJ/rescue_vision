"""感知阶段时间契约与统一新鲜度计算。"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _timestamp(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class PerceptionTiming:
    """一帧感知从采集到发布的阶段时间，全部为单调时钟 ns。"""

    capture_timestamp_ns: int
    submitted_timestamp_ns: int | None
    processing_started_timestamp_ns: int
    inference_completed_timestamp_ns: int
    result_timestamp_ns: int

    def __post_init__(self) -> None:
        capture = _timestamp(self.capture_timestamp_ns, "capture_timestamp_ns")
        started = _timestamp(
            self.processing_started_timestamp_ns,
            "processing_started_timestamp_ns",
        )
        inference = _timestamp(
            self.inference_completed_timestamp_ns,
            "inference_completed_timestamp_ns",
        )
        result = _timestamp(self.result_timestamp_ns, "result_timestamp_ns")
        submitted = self.submitted_timestamp_ns
        if submitted is not None:
            submitted = _timestamp(submitted, "submitted_timestamp_ns")
            if submitted < capture:
                raise ValueError("submitted_timestamp_ns cannot precede capture_timestamp_ns.")
            if submitted > started:
                raise ValueError(
                    "submitted_timestamp_ns cannot follow processing_started_timestamp_ns."
                )
        if not capture <= started <= inference <= result:
            raise ValueError(
                "perception timestamps must satisfy capture <= processing start "
                "<= inference complete <= result."
            )

    @property
    def capture_to_submit_ns(self) -> int | None:
        if self.submitted_timestamp_ns is None:
            return None
        return self.submitted_timestamp_ns - self.capture_timestamp_ns

    @property
    def queue_wait_ns(self) -> int | None:
        if self.submitted_timestamp_ns is None:
            return None
        return self.processing_started_timestamp_ns - self.submitted_timestamp_ns

    @property
    def inference_ns(self) -> int:
        return self.inference_completed_timestamp_ns - self.processing_started_timestamp_ns

    @property
    def postprocess_ns(self) -> int:
        return self.result_timestamp_ns - self.inference_completed_timestamp_ns

    @property
    def capture_to_result_ns(self) -> int:
        return self.result_timestamp_ns - self.capture_timestamp_ns

    @property
    def capture_to_result_ms(self) -> float:
        return self.capture_to_result_ns / 1_000_000.0


def observation_age_ns(capture_timestamp_ns: int, now_ns: int) -> int | None:
    """返回观测年龄；未来时间返回 ``None``，避免静默接受错误时钟。"""

    capture = _timestamp(capture_timestamp_ns, "capture_timestamp_ns")
    now = _timestamp(now_ns, "now_ns")
    if now < capture:
        return None
    return now - capture


def is_observation_fresh(
    capture_timestamp_ns: int,
    now_ns: int,
    max_age_ms: float,
) -> bool:
    if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, (int, float)):
        raise ValueError(f"max_age_ms must be a positive finite number, got {max_age_ms!r}.")
    max_age = float(max_age_ms)
    if not math.isfinite(max_age) or max_age <= 0.0:
        raise ValueError(f"max_age_ms must be a positive finite number, got {max_age_ms!r}.")
    age_ns = observation_age_ns(capture_timestamp_ns, now_ns)
    return age_ns is not None and age_ns <= round(max_age * 1_000_000.0)
