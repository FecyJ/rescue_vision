"""远程调试控制与观察数据的稳定消息契约。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


class RemoteTopic(str, Enum):
    """稳定控制与观察 topic；传输层仍允许扩展自定义 topic。"""

    DEBUG_MOTION = "control/debug/motion"
    DEBUG_CAPTURE = "control/debug/capture"
    SESSION_STATUS = "observation/session/status"
    VIDEO_FRAME = "observation/video/frame"
    MAP_SNAPSHOT = "observation/map/snapshot"
    VEHICLE_STATE = "observation/vehicle/state"
    CAPTURE_STATUS = "observation/capture/status"


class MotionControlMode(str, Enum):
    """调试运动意图的解释方式。"""

    TWIST = "twist"
    TARGET_HEADING = "target_heading"


class HeadingReference(str, Enum):
    """目标朝向的零点；角度均为逆时针为正、单位 rad。"""

    FIELD = "field"
    SESSION_START = "session_start"


class CaptureAction(str, Enum):
    """远程采集动作；输出路径始终由车端运行配置决定。"""

    START = "start"
    STOP = "stop"
    SNAPSHOT = "snapshot"
    MARK_EVENT = "mark_event"


def _non_negative_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{location} must be a non-negative integer, got {value!r}."
        )
    return value


def _finite_float(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number, got {value!r}.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return converted


def _identifier(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    stripped = value.strip()
    if len(stripped.encode("utf-8")) > 128:
        raise ValueError(f"{location} must be at most 128 UTF-8 bytes.")
    return stripped


def _decode_json_object(payload: bytes, location: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError(
            f"{location} payload must be bytes, got {type(payload).__name__}."
        )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{location} payload must be valid UTF-8 JSON.") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise ValueError(f"{location} payload must be a JSON object.")
    return decoded


def _require_exact_keys(
    document: Mapping[str, Any],
    expected: set[str],
    location: str,
) -> None:
    actual = set(document)
    if actual != expected:
        raise ValueError(
            f"{location} keys must be exactly {sorted(expected)}, "
            f"got {sorted(actual)}."
        )


def _require_schema_version(
    value: object,
    expected: int,
    location: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected
    ):
        raise ValueError(
            f"Unsupported {location} schema_version {value!r}."
        )


@dataclass(frozen=True, slots=True)
class DebugMotionCommand:
    """调试用车体运动意图；deadman 字段承载切换式使能的当前状态。"""

    SCHEMA_VERSION: ClassVar[int] = 1

    command_id: str
    issued_timestamp_ns: int
    valid_for_ms: int
    deadman_enabled: bool
    control_mode: MotionControlMode
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    target_heading_rad: float | None = None
    heading_reference: HeadingReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_id",
            _identifier(self.command_id, "command_id"),
        )
        _non_negative_int(self.issued_timestamp_ns, "issued_timestamp_ns")
        if (
            isinstance(self.valid_for_ms, bool)
            or not isinstance(self.valid_for_ms, int)
            or not 1 <= self.valid_for_ms <= 5_000
        ):
            raise ValueError(
                "valid_for_ms must be an integer in [1, 5000], "
                f"got {self.valid_for_ms!r}."
            )
        if not isinstance(self.deadman_enabled, bool):
            raise ValueError("deadman_enabled must be a boolean.")
        if not isinstance(self.control_mode, MotionControlMode):
            raise ValueError(
                f"control_mode must be MotionControlMode, got "
                f"{self.control_mode!r}."
            )
        object.__setattr__(
            self,
            "linear_velocity_m_s",
            _finite_float(self.linear_velocity_m_s, "linear_velocity_m_s"),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_float(self.angular_velocity_rad_s, "angular_velocity_rad_s"),
        )
        if self.target_heading_rad is not None:
            heading = _finite_float(
                self.target_heading_rad,
                "target_heading_rad",
            )
            if not -math.pi <= heading <= math.pi:
                raise ValueError(
                    "target_heading_rad must be in [-pi, pi], "
                    f"got {heading!r}."
                )
            object.__setattr__(self, "target_heading_rad", heading)
        if self.heading_reference is not None and not isinstance(
            self.heading_reference,
            HeadingReference,
        ):
            raise ValueError(
                "heading_reference must be HeadingReference or None."
            )

        if self.control_mode is MotionControlMode.TWIST:
            if (
                self.target_heading_rad is not None
                or self.heading_reference is not None
            ):
                raise ValueError(
                    "TWIST control must not provide target heading fields."
                )
        elif (
            self.target_heading_rad is None
            or self.heading_reference is None
        ):
            raise ValueError(
                "TARGET_HEADING control requires target_heading_rad and "
                "heading_reference."
            )
        elif self.angular_velocity_rad_s != 0.0:
            raise ValueError(
                "TARGET_HEADING control must set angular_velocity_rad_s to "
                "zero; the car-side heading controller owns turn rate."
            )
        if not self.deadman_enabled and (
            self.linear_velocity_m_s != 0.0
            or self.angular_velocity_rad_s != 0.0
            or self.target_heading_rad is not None
            or self.heading_reference is not None
        ):
            raise ValueError(
                "A command with deadman disabled must request zero twist "
                "and no target heading."
            )

    def to_payload(self) -> bytes:
        document = {
            "schema_version": self.SCHEMA_VERSION,
            "command_id": self.command_id,
            "issued_timestamp_ns": self.issued_timestamp_ns,
            "valid_for_ms": self.valid_for_ms,
            "deadman_enabled": self.deadman_enabled,
            "control_mode": self.control_mode.value,
            "linear_velocity_m_s": self.linear_velocity_m_s,
            "angular_velocity_rad_s": self.angular_velocity_rad_s,
            "target_heading_rad": self.target_heading_rad,
            "heading_reference": (
                self.heading_reference.value
                if self.heading_reference is not None
                else None
            ),
        }
        return (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_payload(cls, payload: bytes) -> DebugMotionCommand:
        document = _decode_json_object(payload, "DebugMotionCommand")
        _require_exact_keys(
            document,
            {
                "schema_version",
                "command_id",
                "issued_timestamp_ns",
                "valid_for_ms",
                "deadman_enabled",
                "control_mode",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
                "target_heading_rad",
                "heading_reference",
            },
            "DebugMotionCommand",
        )
        _require_schema_version(
            document["schema_version"],
            cls.SCHEMA_VERSION,
            "DebugMotionCommand",
        )
        try:
            control_mode = MotionControlMode(document["control_mode"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unknown control_mode {document['control_mode']!r}."
            ) from exc
        try:
            heading_reference = (
                None
                if document["heading_reference"] is None
                else HeadingReference(document["heading_reference"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unknown heading_reference "
                f"{document['heading_reference']!r}."
            ) from exc
        return cls(
            command_id=document["command_id"],
            issued_timestamp_ns=document["issued_timestamp_ns"],
            valid_for_ms=document["valid_for_ms"],
            deadman_enabled=document["deadman_enabled"],
            control_mode=control_mode,
            linear_velocity_m_s=document["linear_velocity_m_s"],
            angular_velocity_rad_s=document["angular_velocity_rad_s"],
            target_heading_rad=document["target_heading_rad"],
            heading_reference=heading_reference,
        )


@dataclass(frozen=True, slots=True)
class DebugCaptureCommand:
    """调试录制控制；不允许远端指定车端文件路径。"""

    SCHEMA_VERSION: ClassVar[int] = 1

    request_id: str
    issued_timestamp_ns: int
    action: CaptureAction
    label: str | None = None
    session_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, "request_id"),
        )
        _non_negative_int(self.issued_timestamp_ns, "issued_timestamp_ns")
        if not isinstance(self.action, CaptureAction):
            raise ValueError(
                f"action must be CaptureAction, got {self.action!r}."
            )
        if self.label is not None:
            object.__setattr__(
                self,
                "label",
                _identifier(self.label, "label"),
            )
        if not isinstance(self.session_tags, Mapping):
            raise ValueError("session_tags must be a mapping.")
        tags: dict[str, str] = {}
        for key, value in self.session_tags.items():
            tags[_identifier(key, "session_tags key")] = _identifier(
                value,
                f"session_tags[{key!r}]",
            )
        object.__setattr__(self, "session_tags", MappingProxyType(tags))

        if self.action is CaptureAction.START:
            if self.label is not None:
                raise ValueError("START must not provide label.")
        elif self.session_tags:
            raise ValueError(
                f"{self.action.value} must not provide session_tags."
            )
        if self.action is CaptureAction.MARK_EVENT and self.label is None:
            raise ValueError("MARK_EVENT requires label.")
        if self.action is CaptureAction.STOP and self.label is not None:
            raise ValueError("STOP must not provide label.")

    def to_payload(self) -> bytes:
        document = {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": self.request_id,
            "issued_timestamp_ns": self.issued_timestamp_ns,
            "action": self.action.value,
            "label": self.label,
            "session_tags": dict(self.session_tags),
        }
        return (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_payload(cls, payload: bytes) -> DebugCaptureCommand:
        document = _decode_json_object(payload, "DebugCaptureCommand")
        _require_exact_keys(
            document,
            {
                "schema_version",
                "request_id",
                "issued_timestamp_ns",
                "action",
                "label",
                "session_tags",
            },
            "DebugCaptureCommand",
        )
        _require_schema_version(
            document["schema_version"],
            cls.SCHEMA_VERSION,
            "DebugCaptureCommand",
        )
        try:
            action = CaptureAction(document["action"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown capture action {document['action']!r}.") from exc
        return cls(
            request_id=document["request_id"],
            issued_timestamp_ns=document["issued_timestamp_ns"],
            action=action,
            label=document["label"],
            session_tags=document["session_tags"],
        )
