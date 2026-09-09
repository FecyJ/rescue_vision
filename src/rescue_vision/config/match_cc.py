"""CC 比赛流程的独立运行参数。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def _number(
    raw: Mapping[str, object],
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"match_cc.{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(
            f"match_cc.{name} must be finite and >= {minimum}, got {value!r}."
        )
    return result


def _positive_int(raw: Mapping[str, object], name: str, default: int) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"match_cc.{name} must be a positive integer, got {value!r}.")
    return value


def _signed_number(raw: Mapping[str, object], name: str, default: float) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"match_cc.{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or abs(result) < 0.001:
        raise ValueError(f"match_cc.{name} must be finite and non-zero, got {value!r}.")
    return result


@dataclass(frozen=True, slots=True)
class MatchCCRuntimeConfig:
    """CC 独立搜索、解团和单块抓取参数；启动与安全区运输仍由 ``match`` 提供。"""

    enabled: bool = False
    cluster_neighbor_distance_mm: float = 100.0
    isolated_line_clearance_mm: float = 30.0
    block_alignment_tolerance_mm: float = 5.0
    near_field_threshold_mm: float = 450.0
    green_target_final_x_mm: float = 150.0
    target_search_angular_velocity_rad_s: float = -0.30
    alignment_max_angular_velocity_rad_s: float = 0.18
    alignment_kp_rad_s_per_mm: float = 0.004
    target_approach_speed_m_s: float = 0.15
    command_interval_ms: float = 20.0
    alignment_stable_frames: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("match_cc.enabled must be a boolean.")
        for name in (
            "cluster_neighbor_distance_mm",
            "isolated_line_clearance_mm",
            "block_alignment_tolerance_mm",
            "near_field_threshold_mm",
            "green_target_final_x_mm",
            "alignment_max_angular_velocity_rad_s",
            "alignment_kp_rad_s_per_mm",
            "target_approach_speed_m_s",
            "command_interval_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"match_cc.{name} must be finite and positive.")
        for name in (
            "target_search_angular_velocity_rad_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or abs(value) < 0.001:
                raise ValueError(
                    f"match_cc.{name} must be a finite non-zero angular velocity."
                )
        if self.green_target_final_x_mm >= self.near_field_threshold_mm:
            raise ValueError(
                "match_cc.green_target_final_x_mm must be below near_field_threshold_mm."
            )
        for name in ("alignment_stable_frames",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"match_cc.{name} must be a positive integer.")


_KEYS = frozenset(MatchCCRuntimeConfig.__dataclass_fields__)


def parse_match_cc_config(value: object) -> MatchCCRuntimeConfig:
    """严格解析 ``match_cc`` YAML 节。"""

    if value is None:
        raw: Mapping[str, object] = {}
    elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
        raw = value
    else:
        raise ValueError("match_cc must be a mapping.")
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(f"match_cc contains unknown keys: {unknown!r}.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("match_cc.enabled must be a boolean.")
    defaults = MatchCCRuntimeConfig()
    return MatchCCRuntimeConfig(
        enabled=enabled,
        cluster_neighbor_distance_mm=_number(raw, "cluster_neighbor_distance_mm", defaults.cluster_neighbor_distance_mm),
        isolated_line_clearance_mm=_number(raw, "isolated_line_clearance_mm", defaults.isolated_line_clearance_mm),
        block_alignment_tolerance_mm=_number(raw, "block_alignment_tolerance_mm", defaults.block_alignment_tolerance_mm),
        near_field_threshold_mm=_number(raw, "near_field_threshold_mm", defaults.near_field_threshold_mm),
        green_target_final_x_mm=_number(raw, "green_target_final_x_mm", defaults.green_target_final_x_mm),
        target_search_angular_velocity_rad_s=_signed_number(raw, "target_search_angular_velocity_rad_s", defaults.target_search_angular_velocity_rad_s),
        alignment_max_angular_velocity_rad_s=_number(raw, "alignment_max_angular_velocity_rad_s", defaults.alignment_max_angular_velocity_rad_s),
        alignment_kp_rad_s_per_mm=_number(raw, "alignment_kp_rad_s_per_mm", defaults.alignment_kp_rad_s_per_mm),
        target_approach_speed_m_s=_number(raw, "target_approach_speed_m_s", defaults.target_approach_speed_m_s),
        command_interval_ms=_number(raw, "command_interval_ms", defaults.command_interval_ms),
        alignment_stable_frames=_positive_int(raw, "alignment_stable_frames", defaults.alignment_stable_frames),
    )
