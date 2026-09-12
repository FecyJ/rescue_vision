from __future__ import annotations

import math

import pytest

from rescue_vision.motion.approach_speed import approach_speed_m_s


@pytest.mark.parametrize('dt', [0.005, 0.01])
@pytest.mark.parametrize('precision', [True, False])
def test_accelerated_run_reaches_endpoint_with_baseline_terminal_speed(dt, precision):
    def run(scale):
        x, v, elapsed = 0.0, 0.0, 0.0
        baseline = 0.15 if precision else 0.35
        terminal_max = 0.0
        while x < 1.0 and elapsed < 20:
            command = approach_speed_m_s(1 - x, baseline, scale, 0.5, precision_approach=precision)
            v += max(-0.5 * dt, min(0.5 * dt, command - v))
            x += v * dt
            elapsed += dt
            if x > (0.99 if precision else 0.9):
                terminal_max = max(terminal_max, v)
        assert x >= 1.0
        # Include physical braking tail after the encoder stop threshold.
        stop_x = x + v * v / (2 * 0.5)
        return elapsed, stop_x, terminal_max
    slow, fast = run(1.0), run(1.5)
    assert fast[0] < slow[0]
    assert fast[1] <= slow[1] + 0.004
    assert fast[2] <= slow[2] + 0.001


@pytest.mark.parametrize('remaining', [0.0, 0.001, 0.01, 0.03, 0.05])
def test_last_50mm_pickup_profile_is_unchanged(remaining):
    assert approach_speed_m_s(remaining, .15, 1.5, .5, precision_approach=True) == pytest.approx(max(.005, remaining))


def test_transport_cruises_faster_but_preserves_calibrated_stop_speed():
    assert approach_speed_m_s(1., .35, 1.5, .5, precision_approach=False) == pytest.approx(.525)
    assert approach_speed_m_s(.1, .35, 1.5, .5, precision_approach=False) == pytest.approx(.35)
