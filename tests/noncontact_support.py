"""Small deterministic wheel/IMU plant for match motion regressions."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

from rescue_vision.app.match import MatchState
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import FieldPose2D, normalize_angle
from test_gripper_width_sequence import motion_sample


class MotionPlant:
    def __init__(self, sequence, *, heading=0.0, dt=0.01, perception=None,
                 latency_s=0.4, frame_interval_s=0.3):
        self.sequence = sequence
        self.heading = heading
        self.distance = 0.0
        self.linear = self.angular = 0.0
        self.time_ns = 1_000_000_000
        self.dt = dt
        self.decision = None
        self.records = []
        self.perception = perception
        self.latency_ns = round(latency_s*1e9)
        self.interval_ns = round(frame_interval_s*1e9)
        self.next_capture_ns = self.time_ns
        self.frame = 0
        self.pending = deque()
        self.latest = None

    def tick(self):
        seq = self.sequence
        if self.decision is not None:
            self.linear += max(-self.dt, min(self.dt, self.decision.linear_velocity_m_s-self.linear))
            self.angular += max(-2*self.dt, min(2*self.dt, self.decision.angular_velocity_rad_s-self.angular))
        self.distance += self.linear*self.dt
        self.heading = normalize_angle(self.heading+self.angular*self.dt)
        self.time_ns += round(self.dt*1e9)
        if self.perception is not None and self.time_ns >= self.next_capture_ns:
            self.frame += 1
            pose = FieldPose2D(seq.estimated_field_position or FieldPoint(0, 0), self.heading)
            snap = self.perception(self.frame, self.time_ns, pose)
            if snap is not None:
                self.pending.append((self.time_ns+self.latency_ns,
                    replace(snap, result_timestamp_ns=self.time_ns+self.latency_ns, timing=None)))
            self.next_capture_ns += self.interval_ns
        if self.pending and self.pending[0][0] <= self.time_ns:
            _, self.latest = self.pending.popleft()
        seq.observe_grasp_motion(motion_sample(self.time_ns,
            count=round(self.distance*10000), gyro=round(self.angular*1e6)))
        self.decision = seq.step(self.time_ns, perception=self.latest,
            heading_rad=self.heading, cumulative_distance_m=self.distance,
            left_speed_feedback_m_s=self.linear-self.angular*0.235/2,
            right_speed_feedback_m_s=self.linear+self.angular*0.235/2)
        self.records.append(self.decision)
        return self.decision

    def until(self, predicate, *, seconds=15):
        for _ in range(round(seconds/self.dt)):
            decision = self.tick()
            assert decision.state is not MatchState.TERMINAL_STOP, decision.reason
            if predicate(decision):
                return decision
        raise AssertionError(f"No bounded progress: {self.decision}")
