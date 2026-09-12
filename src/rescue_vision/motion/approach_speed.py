"""Cruise acceleration with an unchanged terminal approach profile."""
from __future__ import annotations

import math


def approach_speed_m_s(
    remaining_m: float, baseline_m_s: float, scale: float,
    deceleration_m_s2: float, *, precision_approach: bool,
) -> float:
    """Reserve 150 ms reaction travel and decelerate before the terminal zone.

    Pickup retains the original last 50 mm proportional approach. Transport
    returns to its calibrated baseline at least 150 mm before its stop target.
    """
    terminal_zone_m = 0.05 if precision_approach else 0.15
    terminal_speed = min(baseline_m_s, 0.05) if precision_approach else baseline_m_s
    cruise = baseline_m_s * scale
    distance = max(0.0, remaining_m - terminal_zone_m - cruise * 0.15)
    braking_cap = math.sqrt(terminal_speed**2 + 2 * deceleration_m_s2 * distance)
    if precision_approach:
        proportional = max(0.005, remaining_m + (scale - 1) * max(0.0, remaining_m - 0.05))
        return min(cruise, proportional, braking_cap)
    return min(cruise, braking_cap)
