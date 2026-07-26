"""不依赖硬件的轻量任务目标多目标跟踪。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    ObservationQuality,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    confirmation_hits: int
    max_association_ground_mm: float
    min_association_iou: float
    max_coast_ms: float
    confidence_decay_per_second: float
    min_confidence: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.confirmation_hits, bool)
            or not isinstance(self.confirmation_hits, int)
            or self.confirmation_hits <= 0
        ):
            raise ValueError("confirmation_hits must be a positive integer.")
        for name in (
            "max_association_ground_mm",
            "max_coast_ms",
            "confidence_decay_per_second",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        for name in ("min_association_iou", "min_confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")

    def build_tracker(self) -> MultiTargetTracker:
        return MultiTargetTracker(self)


@dataclass(frozen=True, slots=True)
class TrackedTarget:
    track_id: int
    status: TrackStatus
    ever_confirmed: bool
    target_class: TargetClass
    class_probabilities: ClassProbabilities
    confidence: float
    box: UndistortedBoundingBox
    k0: UndistortedPixel | None
    k0_confidence: float
    ground_point: GroundPoint | None
    quality: frozenset[ObservationQuality]
    first_seen_timestamp_ns: int
    last_seen_timestamp_ns: int
    state_timestamp_ns: int
    frame_sequence: int
    hit_count: int
    missed_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id <= 0
        ):
            raise ValueError("track_id must be a positive integer.")
        if not isinstance(self.status, TrackStatus):
            raise ValueError("status must be a TrackStatus.")
        if not isinstance(self.ever_confirmed, bool):
            raise ValueError("ever_confirmed must be a boolean.")
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass.")
        if not isinstance(self.class_probabilities, ClassProbabilities):
            raise ValueError(
                "class_probabilities must be a ClassProbabilities."
            )
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1].")
        if not isinstance(self.box, UndistortedBoundingBox):
            raise ValueError("box must be an UndistortedBoundingBox.")
        if self.k0 is not None and not isinstance(self.k0, UndistortedPixel):
            raise ValueError("k0 must be an UndistortedPixel or None.")
        if (
            not math.isfinite(self.k0_confidence)
            or not 0.0 <= self.k0_confidence <= 1.0
        ):
            raise ValueError("k0_confidence must be finite and in [0, 1].")
        if self.ground_point is not None:
            if not isinstance(self.ground_point, GroundPoint):
                raise ValueError("ground_point must be a GroundPoint or None.")
            if self.k0 is None:
                raise ValueError("ground_point requires an available k0.")
            if not (
                math.isfinite(self.ground_point.x)
                and math.isfinite(self.ground_point.y)
            ):
                raise ValueError("ground_point must be finite.")
        if not all(
            isinstance(item, ObservationQuality) for item in self.quality
        ):
            raise ValueError(
                "quality must contain only ObservationQuality values."
            )
        timestamps = (
            self.first_seen_timestamp_ns,
            self.last_seen_timestamp_ns,
            self.state_timestamp_ns,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in timestamps
        ) or not (
            self.first_seen_timestamp_ns
            <= self.last_seen_timestamp_ns
            <= self.state_timestamp_ns
        ):
            raise ValueError(
                "Track timestamps must be non-negative and ordered "
                "first_seen <= last_seen <= state."
            )
        for name, value, minimum in (
            ("frame_sequence", self.frame_sequence, 0),
            ("hit_count", self.hit_count, 1),
            ("missed_count", self.missed_count, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")

    @property
    def age_since_seen_ms(self) -> float:
        return (self.state_timestamp_ns - self.last_seen_timestamp_ns) / 1_000_000.0


def _target_class(probabilities: ClassProbabilities) -> TargetClass:
    values = {
        item: probabilities.probability(item)
        for item in TargetClass
    }
    return max(TargetClass, key=lambda item: values[item])


def _blend_probabilities(
    previous: ClassProbabilities,
    current: ClassProbabilities,
    previous_weight: float,
    current_weight: float,
) -> ClassProbabilities:
    total = previous_weight + current_weight
    values = {
        item: (
            previous.probability(item) * previous_weight
            + current.probability(item) * current_weight
        )
        / total
        for item in TargetClass
    }
    return ClassProbabilities(
        green_supply=values[TargetClass.GREEN_SUPPLY],
        black_core=values[TargetClass.BLACK_CORE],
        orange_injured=values[TargetClass.ORANGE_INJURED],
        blue_danger=values[TargetClass.BLUE_DANGER],
        unknown=values[TargetClass.UNKNOWN],
    )


class MultiTargetTracker:
    """用地面距离和图像 IoU 做确定性关联，并保留短时遮挡目标。"""

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._tracks: dict[int, TrackedTarget] = {}
        self._next_track_id = 1
        self._last_timestamp_ns: int | None = None

    @property
    def tracks(self) -> tuple[TrackedTarget, ...]:
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    def update(
        self,
        timestamp_ns: int,
        observations: tuple[TargetObservation, ...] | list[TargetObservation],
    ) -> tuple[TrackedTarget, ...]:
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                f"timestamp_ns moved backwards from {self._last_timestamp_ns} "
                f"to {timestamp_ns}."
            )
        observations = tuple(observations)
        for index, observation in enumerate(observations):
            if not isinstance(observation, TargetObservation):
                raise ValueError(
                    f"observations[{index}] must be a TargetObservation."
                )
            if observation.capture_timestamp_ns != timestamp_ns:
                raise ValueError(
                    f"observations[{index}].capture_timestamp_ns "
                    f"{observation.capture_timestamp_ns} does not match update "
                    f"timestamp_ns {timestamp_ns}."
                )

        decayed = {
            track_id: self._decay(track, timestamp_ns)
            for track_id, track in self._tracks.items()
        }
        matches = self._associate(decayed, observations)
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_observation_indices = {
            observation_index for _, observation_index in matches
        }

        updated: dict[int, TrackedTarget] = {}
        for track_id, observation_index in matches:
            updated[track_id] = self._update_track(
                decayed[track_id],
                observations[observation_index],
                timestamp_ns,
            )
        for track_id, track in decayed.items():
            if track_id in matched_track_ids:
                continue
            coasting = replace(
                track,
                status=TrackStatus.COASTING,
                state_timestamp_ns=timestamp_ns,
                missed_count=track.missed_count + 1,
            )
            if self._keep(coasting):
                updated[track_id] = coasting

        for index, observation in enumerate(observations):
            if index in matched_observation_indices:
                continue
            track = self._new_track(observation, timestamp_ns)
            if self._keep(track):
                updated[track.track_id] = track

        self._tracks = updated
        self._last_timestamp_ns = timestamp_ns
        return self.tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp_ns = None

    def _associate(
        self,
        tracks: dict[int, TrackedTarget],
        observations: tuple[TargetObservation, ...],
    ) -> tuple[tuple[int, int], ...]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in tracks.items():
            for observation_index, observation in enumerate(observations):
                score = self._association_score(track, observation)
                if score is not None:
                    candidates.append((score, track_id, observation_index))
        candidates.sort()

        used_tracks: set[int] = set()
        used_observations: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, track_id, observation_index in candidates:
            if (
                track_id in used_tracks
                or observation_index in used_observations
            ):
                continue
            used_tracks.add(track_id)
            used_observations.add(observation_index)
            matches.append((track_id, observation_index))
        return tuple(matches)

    def _association_score(
        self,
        track: TrackedTarget,
        observation: TargetObservation,
    ) -> float | None:
        if (
            track.target_class is not TargetClass.UNKNOWN
            and observation.target_class is not TargetClass.UNKNOWN
            and track.target_class is not observation.target_class
        ):
            return None
        iou = track.box.iou(observation.box)
        if track.ground_point is not None and observation.ground_point is not None:
            distance_mm = math.hypot(
                track.ground_point.x - observation.ground_point.x,
                track.ground_point.y - observation.ground_point.y,
            )
            if distance_mm > self._config.max_association_ground_mm:
                return None
            return (
                distance_mm / self._config.max_association_ground_mm
                + 0.25 * (1.0 - iou)
            )
        if iou < self._config.min_association_iou:
            return None
        return 1.0 - iou

    def _decay(
        self,
        track: TrackedTarget,
        timestamp_ns: int,
    ) -> TrackedTarget:
        elapsed_s = (timestamp_ns - track.state_timestamp_ns) / 1_000_000_000.0
        factor = math.exp(
            -self._config.confidence_decay_per_second * elapsed_s
        )
        return replace(
            track,
            confidence=track.confidence * factor,
            state_timestamp_ns=timestamp_ns,
        )

    def _new_track(
        self,
        observation: TargetObservation,
        timestamp_ns: int,
    ) -> TrackedTarget:
        track_id = self._next_track_id
        self._next_track_id += 1
        confidence = max(observation.class_probabilities.as_dict().values())
        status = (
            TrackStatus.CONFIRMED
            if self._config.confirmation_hits == 1
            else TrackStatus.TENTATIVE
        )
        return TrackedTarget(
            track_id=track_id,
            status=status,
            ever_confirmed=status is TrackStatus.CONFIRMED,
            target_class=observation.target_class,
            class_probabilities=observation.class_probabilities,
            confidence=confidence,
            box=observation.box,
            k0=observation.k0,
            k0_confidence=observation.k0_confidence,
            ground_point=observation.ground_point,
            quality=observation.quality,
            first_seen_timestamp_ns=timestamp_ns,
            last_seen_timestamp_ns=timestamp_ns,
            state_timestamp_ns=timestamp_ns,
            frame_sequence=observation.frame_sequence,
            hit_count=1,
            missed_count=0,
        )

    def _update_track(
        self,
        track: TrackedTarget,
        observation: TargetObservation,
        timestamp_ns: int,
    ) -> TrackedTarget:
        observation_confidence = max(
            observation.class_probabilities.as_dict().values()
        )
        confidence = 1.0 - (
            (1.0 - track.confidence) * (1.0 - observation_confidence)
        )
        probabilities = _blend_probabilities(
            track.class_probabilities,
            observation.class_probabilities,
            max(track.confidence, 1e-9),
            max(observation_confidence, 1e-9),
        )
        hit_count = track.hit_count + 1
        status = (
            TrackStatus.CONFIRMED
            if hit_count >= self._config.confirmation_hits
            else TrackStatus.TENTATIVE
        )
        return TrackedTarget(
            track_id=track.track_id,
            status=status,
            ever_confirmed=(
                track.ever_confirmed
                or status is TrackStatus.CONFIRMED
            ),
            target_class=_target_class(probabilities),
            class_probabilities=probabilities,
            confidence=confidence,
            box=observation.box,
            k0=observation.k0,
            k0_confidence=observation.k0_confidence,
            ground_point=observation.ground_point,
            quality=observation.quality,
            first_seen_timestamp_ns=track.first_seen_timestamp_ns,
            last_seen_timestamp_ns=timestamp_ns,
            state_timestamp_ns=timestamp_ns,
            frame_sequence=observation.frame_sequence,
            hit_count=hit_count,
            missed_count=0,
        )

    def _keep(self, track: TrackedTarget) -> bool:
        return (
            track.age_since_seen_ms <= self._config.max_coast_ms
            and track.confidence >= self._config.min_confidence
        )
