"""不依赖硬件的轻量任务目标多目标跟踪。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception.types import (
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
        # 远场任务 tracker 默认把两个已知类别的冲突当成不同目标，避免
        # 危险类别证据被普通物资历史吞掉。近场停车窗口由上层显式开启软
        # 类别关联：K0 空间连续性优先，类别只作为很小的二级代价。
        self._class_mismatch_penalty: float | None = None
        self._retain_low_confidence_tracks = False

    @property
    def config(self) -> TrackingConfig:
        """返回创建 tracker 时使用的生命周期和门限配置。"""

        return self._config

    def enable_soft_class_association(
        self,
        *,
        mismatch_penalty: float = 0.10,
    ) -> None:
        """允许近场类别抖动继续沿用同一空间轨迹。

        该设置只影响后续关联，不改变观测中的类别、质量或危险历史；
        ``GraspTargetTracker`` 仍会对明确危险观测保持不可选状态。
        """

        if (
            isinstance(mismatch_penalty, bool)
            or not isinstance(mismatch_penalty, (int, float))
            or not math.isfinite(float(mismatch_penalty))
            or float(mismatch_penalty) < 0.0
        ):
            raise ValueError(
                "mismatch_penalty must be finite and non-negative, "
                f"got {mismatch_penalty!r}."
            )
        self._class_mismatch_penalty = float(mismatch_penalty)

    def enable_low_confidence_retention(self) -> None:
        """保留近场窗口中的弱观测轨迹，避免障碍 ID 退化为临时编号。"""

        self._retain_low_confidence_tracks = True

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
        observations = self.deduplicate_observations(observations)

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

    @staticmethod
    def deduplicate_observations(
        observations: tuple[TargetObservation, ...],
    ) -> tuple[TargetObservation, ...]:
        """压掉同帧同类、同地面点且框高度重叠的重复检测。

        15 mm 小于两个 40 mm 实体可能具有的中心间距；类别或模型证据不同的
        检测绝不合并，避免把危险冲突证据静默删除。
        """

        ranked = sorted(
            enumerate(observations),
            key=lambda item: (
                len(item[1].quality),
                -item[1].detection_confidence,
                -item[1].k0_confidence,
                item[0],
            ),
        )
        retained: list[tuple[int, TargetObservation]] = []
        for index, observation in ranked:
            duplicate = False
            for _, accepted in retained:
                if (
                    observation.target_class is not accepted.target_class
                    or observation.model_target_class
                    is not accepted.model_target_class
                    or observation.ground_point is None
                    or accepted.ground_point is None
                ):
                    continue
                distance_mm = math.hypot(
                    observation.ground_point.x - accepted.ground_point.x,
                    observation.ground_point.y - accepted.ground_point.y,
                )
                if distance_mm <= 15.0 and observation.box.iou(accepted.box) >= 0.7:
                    duplicate = True
                    break
            if not duplicate:
                retained.append((index, observation))
        return tuple(observation for _, observation in sorted(retained))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp_ns = None

    def _associate(
        self,
        tracks: dict[int, TrackedTarget],
        observations: tuple[TargetObservation, ...],
    ) -> tuple[tuple[int, int], ...]:
        if not tracks or not observations:
            return ()

        track_items = tuple(sorted(tracks.items()))
        track_count = len(track_items)
        observation_count = len(observations)
        # Add dummy rows/columns so the assignment can leave either side
        # unmatched.  A valid gated pair is always cheaper than dropping both
        # endpoints; an invalid pair is much more expensive than doing so.
        unmatched_cost = 2.0
        invalid_cost = 8.0
        size = track_count + observation_count
        costs = [[0.0 for _ in range(size)] for _ in range(size)]
        valid_scores: dict[tuple[int, int], float] = {}
        for track_index, (_, track) in enumerate(track_items):
            for observation_index, observation in enumerate(observations):
                score = self._association_score(track, observation)
                costs[track_index][observation_index] = (
                    invalid_cost if score is None else score
                )
                if score is not None:
                    valid_scores[(track_index, observation_index)] = score
            for dummy_column in range(observation_count, size):
                costs[track_index][dummy_column] = unmatched_cost
        for dummy_row in range(track_count, size):
            for observation_index in range(observation_count):
                costs[dummy_row][observation_index] = unmatched_cost

        assignment = self._minimum_cost_assignment(tuple(tuple(row) for row in costs))
        matches = [
            (track_items[track_index][0], observation_index)
            for track_index, observation_index in enumerate(assignment)
            if (
                track_index < track_count
                and observation_index < observation_count
                and (track_index, observation_index) in valid_scores
            )
        ]
        return tuple(sorted(matches))

    @staticmethod
    def _minimum_cost_assignment(
        costs: tuple[tuple[float, ...], ...],
    ) -> tuple[int, ...]:
        """返回方阵最小代价分配；列顺序作为完全相等时的稳定 tie-break。"""

        size = len(costs)
        if size == 0:
            return ()
        if any(len(row) != size for row in costs):
            raise ValueError("costs must be a non-empty square matrix.")
        # Kuhn-Munkres / Hungarian algorithm, using 1-based work arrays as
        # customary.  The near-field detector has a small bounded target count,
        # so the O(n^3) assignment is cheap and avoids greedy ID swaps when two
        # K0 positions are close.
        u = [0.0] * (size + 1)
        v = [0.0] * (size + 1)
        matched_column_row = [0] * (size + 1)
        previous_column = [0] * (size + 1)
        for row in range(1, size + 1):
            matched_column_row[0] = row
            current_column = 0
            minimum = [math.inf] * (size + 1)
            used = [False] * (size + 1)
            while True:
                used[current_column] = True
                matched_row = matched_column_row[current_column]
                delta = math.inf
                next_column = 0
                for column in range(1, size + 1):
                    if used[column]:
                        continue
                    reduced = (
                        costs[matched_row - 1][column - 1]
                        - u[matched_row]
                        - v[column]
                    )
                    if reduced < minimum[column]:
                        minimum[column] = reduced
                        previous_column[column] = current_column
                    if minimum[column] < delta:
                        delta = minimum[column]
                        next_column = column
                if not math.isfinite(delta):
                    raise RuntimeError("Hungarian assignment has no augmenting path.")
                for column in range(size + 1):
                    if used[column]:
                        u[matched_column_row[column]] += delta
                        v[column] -= delta
                    else:
                        minimum[column] -= delta
                current_column = next_column
                if matched_column_row[current_column] == 0:
                    break
            while True:
                prior = previous_column[current_column]
                matched_column_row[current_column] = matched_column_row[prior]
                current_column = prior
                if current_column == 0:
                    break

        assignment = [-1] * size
        for column in range(1, size + 1):
            row = matched_column_row[column]
            if row != 0:
                assignment[row - 1] = column - 1
        if any(column < 0 for column in assignment):
            raise RuntimeError("Hungarian assignment left a row unmatched.")
        return tuple(assignment)

    def _association_score(
        self,
        track: TrackedTarget,
        observation: TargetObservation,
    ) -> float | None:
        class_mismatch = (
            track.target_class is not TargetClass.UNKNOWN
            and observation.target_class is not TargetClass.UNKNOWN
            and track.target_class is not observation.target_class
        )
        if (
            class_mismatch
            and self._class_mismatch_penalty is not None
            and (
                track.target_class
                in {TargetClass.ORANGE_INJURED, TargetClass.BLUE_DANGER}
                or observation.target_class
                in {TargetClass.ORANGE_INJURED, TargetClass.BLUE_DANGER}
            )
        ):
            # 近场可允许绿/黑物资之间的短暂类别抖动，但伤员和危险目标
            # 具有不同任务/安全语义，不能仅因空间接近而交换轨迹身份。
            return None
        if (
            class_mismatch
            and self._class_mismatch_penalty is None
        ):
            return None
        class_penalty = (
            0.0
            if not class_mismatch
            else self._class_mismatch_penalty
        )
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
                + class_penalty
            )
        if iou < self._config.min_association_iou:
            return None
        return 1.0 - iou + class_penalty

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
        confidence = observation.detection_confidence
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
        observation_confidence = observation.detection_confidence
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
            and (
                self._retain_low_confidence_tracks
                or track.confidence >= self._config.min_confidence
            )
        )
