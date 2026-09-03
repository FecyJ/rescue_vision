"""green_grab 纯逻辑状态机的无硬件测试。"""

from __future__ import annotations

import pytest

from rescue_vision.app.cluster_breakup import GripperPosture
from rescue_vision.app.green_grab import (
    GreenGrabSequence,
    GreenGrabState,
    GreenPixelTarget,
)
from rescue_vision.config import GreenGrabRuntimeConfig


def make_config(**overrides: object) -> GreenGrabRuntimeConfig:
    values: dict[str, object] = dict(
        enabled=True,
        search_angular_velocity_rad_s=0.30,
        approach_speed_m_s=0.08,
        align_tolerance_ratio=0.06,
        align_kp_rad_s=1.2,
        align_max_angular_velocity_rad_s=0.35,
        engage_bottom_fraction=0.85,
        confirm_frames=3,
        target_loss_timeout_ms=500.0,
    )
    values.update(overrides)
    return GreenGrabRuntimeConfig(**values)  # type: ignore[arg-type]


def make_sequence(**overrides: object) -> GreenGrabSequence:
    return GreenGrabSequence(
        make_config(**overrides),
        gripper_full_travel_time_s=1.0,
    )


def target(
    center_u: float = 100.0,
    bottom_v: float = 50.0,
    width: int = 200,
    height: int = 100,
) -> GreenPixelTarget:
    return GreenPixelTarget(
        center_u=center_u,
        bottom_v=bottom_v,
        image_width=width,
        image_height=height,
    )


def confirm_and_align(seq: GreenGrabSequence) -> None:
    """推进到绿色确认并进入 ALIGN：连续三帧居中目标。"""

    seq.step(1_000_000_000, target())
    seq.step(1_100_000_000, target())
    seq.step(1_200_000_000, target())


def test_search_rotates_when_no_green() -> None:
    seq = make_sequence()
    decision = seq.step(0, None)
    assert decision.state is GreenGrabState.SEARCH
    assert decision.angular_velocity_rad_s == 0.30
    assert decision.linear_velocity_m_s == 0.0
    assert decision.gripper_posture is GripperPosture.CLOSED


def test_confirm_requires_configured_frames_then_opens() -> None:
    seq = make_sequence()
    first = seq.step(1_000_000_000, target())
    second = seq.step(1_100_000_000, target())
    assert first.state is GreenGrabState.SEARCH
    assert first.gripper_posture is GripperPosture.CLOSED
    assert second.state is GreenGrabState.SEARCH
    third = seq.step(1_200_000_000, target())
    assert third.state is GreenGrabState.ALIGN
    assert third.gripper_posture is GripperPosture.OPEN


def test_confirm_resets_when_green_lost() -> None:
    seq = make_sequence()
    seq.step(1_000_000_000, target())
    seq.step(1_100_000_000, target())
    # 丢失一帧后确认计数清零。
    seq.step(1_200_000_000, None)
    decision = seq.step(1_300_000_000, target())
    assert decision.state is GreenGrabState.SEARCH
    assert decision.gripper_posture is GripperPosture.CLOSED


def test_align_turns_right_for_right_target() -> None:
    seq = make_sequence()
    confirm_and_align(seq)
    # 目标在图像右侧，ratio > 0，角速度应为负（右转）。
    decision = seq.step(1_300_000_000, target(center_u=180.0))
    assert decision.state is GreenGrabState.ALIGN
    assert decision.angular_velocity_rad_s < 0.0
    assert decision.linear_velocity_m_s == 0.0


def test_align_angular_is_clamped() -> None:
    seq = make_sequence()
    confirm_and_align(seq)
    decision = seq.step(1_300_000_000, target(center_u=200.0))
    assert decision.angular_velocity_rad_s == -0.35


def test_align_advances_when_centered() -> None:
    seq = make_sequence()
    confirm_and_align(seq)
    decision = seq.step(1_300_000_000, target(center_u=100.0))
    assert decision.state is GreenGrabState.APPROACH
    assert decision.gripper_posture is GripperPosture.OPEN


def test_approach_drives_forward_until_close() -> None:
    seq = make_sequence()
    confirm_and_align(seq)
    seq.step(1_300_000_000, target(center_u=100.0))  # -> APPROACH
    decision = seq.step(1_400_000_000, target(center_u=100.0, bottom_v=50.0))
    assert decision.state is GreenGrabState.APPROACH
    assert decision.linear_velocity_m_s == 0.08


def test_approach_grabs_when_close() -> None:
    seq = make_sequence()
    confirm_and_align(seq)
    seq.step(1_300_000_000, target(center_u=100.0))  # -> APPROACH
    # engage_bottom_fraction=0.85，height=100 -> 底边 >= 85 即抓取。
    decision = seq.step(1_400_000_000, target(center_u=100.0, bottom_v=90.0))
    assert decision.state is GreenGrabState.GRAB
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert decision.linear_velocity_m_s == 0.0


def test_grab_holds_then_done() -> None:
    seq = make_sequence()
    confirm_and_align(seq)
    seq.step(1_300_000_000, target(center_u=100.0))  # -> APPROACH
    grab = seq.step(1_400_000_000, target(center_u=100.0, bottom_v=90.0))
    assert grab.state is GreenGrabState.GRAB
    # 全行程 1.0 秒内保持合爪。
    holding = seq.step(1_800_000_000, target(center_u=100.0, bottom_v=90.0))
    assert holding.state is GreenGrabState.GRAB
    # 超过全行程时间后进入 DONE。
    done = seq.step(2_500_000_000, target(center_u=100.0, bottom_v=90.0))
    assert done.state is GreenGrabState.DONE


def test_lost_target_holds_then_resumes_search() -> None:
    seq = make_sequence()
    confirm_and_align(seq)  # 最后看到绿色在 1_200_000_000
    # 短暂丢失 < 500ms：保持 OPEN 零速。
    briefly = seq.step(1_300_000_000, None)
    assert briefly.state is GreenGrabState.ALIGN
    assert briefly.linear_velocity_m_s == 0.0
    assert briefly.gripper_posture is GripperPosture.OPEN
    # 丢失 >= 500ms：回到搜索并合爪。
    resumed = seq.step(1_800_000_000, None)
    assert resumed.state is GreenGrabState.SEARCH
    assert resumed.gripper_posture is GripperPosture.CLOSED
    assert resumed.angular_velocity_rad_s == 0.30


def test_step_rejects_backwards_timestamp() -> None:
    seq = make_sequence()
    seq.step(1_000_000_000, None)
    with pytest.raises(ValueError):
        seq.step(999_999_999, None)


def test_step_rejects_bad_green_type() -> None:
    seq = make_sequence()
    with pytest.raises(TypeError):
        seq.step(1_000_000_000, object())  # type: ignore[arg-type]


def test_green_pixel_target_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        GreenPixelTarget(
            center_u=-1.0,
            bottom_v=50.0,
            image_width=200,
            image_height=100,
        )
    with pytest.raises(ValueError):
        GreenPixelTarget(
            center_u=100.0,
            bottom_v=50.0,
            image_width=0,
            image_height=100,
        )
    with pytest.raises(ValueError):
        GreenPixelTarget(
            center_u=100.0,
            bottom_v=101.0,
            image_width=200,
            image_height=100,
        )


def test_config_rejects_bad_engage_fraction() -> None:
    with pytest.raises(ValueError):
        make_config(engage_bottom_fraction=1.5)


def test_config_rejects_zero_confirm_frames() -> None:
    with pytest.raises(ValueError):
        make_config(confirm_frames=0)
