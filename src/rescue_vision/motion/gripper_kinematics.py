"""双舵机夹爪的平面运动学。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from rescue_vision.geometry.types import GroundPoint


def _finite(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number, got {value!r}.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return converted


def _servo_angle(value: object, location: str) -> float:
    angle = _finite(value, location)
    if not 0.0 <= angle <= 180.0:
        raise ValueError(f"{location} must be in [0, 180], got {value!r}.")
    return angle


@dataclass(frozen=True, slots=True)
class GripperKinematics:
    """用户给定结构尺寸下的左右夹爪末端映射。

    坐标使用机器人地面系：``x`` 向前、``y`` 向左，单位 mm。相对角度以
    关闭姿态为 0；左夹爪向左、右夹爪向右的角度均为正。该模型对应：

    * 左转轴 ``(52.5, 75)``，末端向量 ``(105, -75)``；
    * 右转轴 ``(52.5, -75)``，末端向量 ``(105, 75)``。

    对称开口仍可使用 ``servo_angles_for_opening()``；需要分别贴合左右
    目标边界时使用 ``servo_angles_for_edge_positions()``。
    """

    pivot_x_mm: float = 52.5
    pivot_half_spacing_mm: float = 75.0
    tip_offset_x_mm: float = 105.0
    tip_offset_y_mm: float = 75.0

    def __post_init__(self) -> None:
        for name in (
            "pivot_x_mm",
            "pivot_half_spacing_mm",
            "tip_offset_x_mm",
            "tip_offset_y_mm",
        ):
            value = _finite(getattr(self, name), name)
            if name != "pivot_x_mm" and value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}.")
            object.__setattr__(self, name, value)

    @staticmethod
    def _relative_angle_rad(relative_angle_deg: float) -> float:
        angle = _finite(relative_angle_deg, "relative_angle_deg")
        if angle < 0.0:
            raise ValueError(
                "relative_angle_deg must be non-negative, "
                f"got {relative_angle_deg!r}."
            )
        return math.radians(angle)

    def left_tip_position(self, relative_angle_deg: float) -> GroundPoint:
        """返回左夹爪末端在机器人地面系中的位置。"""

        theta = self._relative_angle_rad(relative_angle_deg)
        cosine = math.cos(theta)
        sine = math.sin(theta)
        return GroundPoint(
            self.pivot_x_mm
            + self.tip_offset_x_mm * cosine
            + self.tip_offset_y_mm * sine,
            self.pivot_half_spacing_mm
            + self.tip_offset_x_mm * sine
            - self.tip_offset_y_mm * cosine,
        )

    def right_tip_position(self, relative_angle_deg: float) -> GroundPoint:
        """返回右夹爪末端在机器人地面系中的位置。"""

        theta = self._relative_angle_rad(relative_angle_deg)
        cosine = math.cos(theta)
        sine = math.sin(theta)
        return GroundPoint(
            self.pivot_x_mm
            + self.tip_offset_x_mm * cosine
            + self.tip_offset_y_mm * sine,
            -self.pivot_half_spacing_mm
            - self.tip_offset_x_mm * sine
            + self.tip_offset_y_mm * cosine,
        )

    def symmetric_opening_width_mm(self, relative_angle_deg: float) -> float:
        """返回左右对称打开时两末端的 ``y`` 方向开口宽度。"""

        left = self.left_tip_position(relative_angle_deg)
        right = self.right_tip_position(relative_angle_deg)
        width = left.y - right.y
        if width < -1e-9:
            raise ValueError(
                "relative angle produces a negative opening width, "
                f"angle={relative_angle_deg!r}, width={width!r}."
            )
        return max(0.0, width)

    def relative_angle_for_opening(
        self,
        opening_width_mm: float,
        *,
        max_relative_angle_deg: float,
    ) -> float:
        """反解目标开口对应的对称相对角度。

        反解只在 ``0..90°`` 的单调段内进行，避免大角度姿态出现多个解。
        ``max_relative_angle_deg`` 应来自左右舵机实际安全行程的较小值。
        """

        opening = _finite(opening_width_mm, "opening_width_mm")
        if opening < 0.0:
            raise ValueError(
                "opening_width_mm must be non-negative, "
                f"got {opening_width_mm!r}."
            )
        maximum_angle = _finite(
            max_relative_angle_deg,
            "max_relative_angle_deg",
        )
        if not 0.0 < maximum_angle <= 90.0:
            raise ValueError(
                "max_relative_angle_deg must be in (0, 90], "
                f"got {max_relative_angle_deg!r}."
            )
        maximum_opening = self.symmetric_opening_width_mm(maximum_angle)
        if opening > maximum_opening + 1e-9:
            raise ValueError(
                "opening_width_mm exceeds the configured gripper geometry: "
                f"requested={opening:.3f} mm, maximum={maximum_opening:.3f} mm."
            )
        if opening <= 1e-9:
            return 0.0

        lower = 0.0
        upper = maximum_angle
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            if self.symmetric_opening_width_mm(middle) < opening:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def servo_angles_for_opening(
        self,
        opening_width_mm: float,
        *,
        open_left_angle_deg: float,
        open_right_angle_deg: float,
        closed_left_angle_deg: float,
        closed_right_angle_deg: float,
    ) -> tuple[float, float]:
        """把目标开口反解为左右绝对舵机角度。

        按当前机械方向，左舵机命令角随向左打开而减小，右舵机命令角随向右
        打开而增大。因此结果满足 ``left + right = closed_left + closed_right``。
        """

        open_left = _servo_angle(open_left_angle_deg, "open_left_angle_deg")
        open_right = _servo_angle(open_right_angle_deg, "open_right_angle_deg")
        closed_left = _servo_angle(
            closed_left_angle_deg,
            "closed_left_angle_deg",
        )
        closed_right = _servo_angle(
            closed_right_angle_deg,
            "closed_right_angle_deg",
        )
        left_travel = closed_left - open_left
        right_travel = open_right - closed_right
        if left_travel <= 0.0:
            raise ValueError(
                "left servo direction is incompatible with outward opening: "
                f"open={open_left:g}, closed={closed_left:g}."
            )
        if right_travel <= 0.0:
            raise ValueError(
                "right servo direction is incompatible with outward opening: "
                f"closed={closed_right:g}, open={open_right:g}."
            )
        maximum_relative_angle = min(left_travel, right_travel)
        relative_angle = self.relative_angle_for_opening(
            opening_width_mm,
            max_relative_angle_deg=maximum_relative_angle,
        )
        return (
            closed_left - relative_angle,
            closed_right + relative_angle,
        )

    def servo_angles_for_edge_positions(
        self,
        left_tip_y_mm: float,
        right_tip_y_mm: float,
        *,
        open_left_angle_deg: float,
        open_right_angle_deg: float,
        closed_left_angle_deg: float,
        closed_right_angle_deg: float,
    ) -> tuple[float, float]:
        """把左右夹爪末端的独立 ``y`` 位置反解为绝对舵机角度。

        左末端只能从闭合位置 ``y=0`` 向机器人左侧增大，右末端只能向
        机器人右侧减小。因此调用方应传入 ``left_tip_y_mm >= 0`` 和
        ``right_tip_y_mm <= 0``。两侧允许使用不同的相对开角。
        """

        left_tip_y = _finite(left_tip_y_mm, "left_tip_y_mm")
        right_tip_y = _finite(right_tip_y_mm, "right_tip_y_mm")
        if left_tip_y < right_tip_y:
            raise ValueError(
                "left_tip_y_mm must be greater than or equal to right_tip_y_mm, "
                f"got {left_tip_y_mm} < {right_tip_y_mm}."
            )
        if left_tip_y < -1e-9:
            raise ValueError(
                "left_tip_y_mm must be non-negative for outward opening, "
                f"got {left_tip_y_mm}."
            )
        if right_tip_y > 1e-9:
            raise ValueError(
                "right_tip_y_mm must be non-positive for outward opening, "
                f"got {right_tip_y_mm}."
            )

        open_left = _servo_angle(open_left_angle_deg, "open_left_angle_deg")
        open_right = _servo_angle(open_right_angle_deg, "open_right_angle_deg")
        closed_left = _servo_angle(
            closed_left_angle_deg,
            "closed_left_angle_deg",
        )
        closed_right = _servo_angle(
            closed_right_angle_deg,
            "closed_right_angle_deg",
        )
        left_travel = closed_left - open_left
        right_travel = open_right - closed_right
        if not 0.0 < left_travel <= 90.0:
            raise ValueError(
                "left servo outward travel must be in (0, 90], "
                f"got {left_travel}."
            )
        if not 0.0 < right_travel <= 90.0:
            raise ValueError(
                "right servo outward travel must be in (0, 90], "
                f"got {right_travel}."
            )

        left_relative = self._relative_angle_for_left_tip_y(
            left_tip_y,
            left_travel,
        )
        right_relative = self._relative_angle_for_right_tip_y(
            right_tip_y,
            right_travel,
        )
        return (
            closed_left - left_relative,
            closed_right + right_relative,
        )

    def _relative_angle_for_left_tip_y(
        self,
        target_y_mm: float,
        max_relative_angle_deg: float,
    ) -> float:
        maximum_y = self.left_tip_position(max_relative_angle_deg).y
        if target_y_mm > maximum_y + 1e-9:
            raise ValueError(
                "left_tip_y_mm exceeds the configured gripper geometry: "
                f"requested={target_y_mm:.3f} mm, maximum={maximum_y:.3f} mm."
            )
        if target_y_mm <= 1e-9:
            return 0.0
        lower = 0.0
        upper = max_relative_angle_deg
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            if self.left_tip_position(middle).y < target_y_mm:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def _relative_angle_for_right_tip_y(
        self,
        target_y_mm: float,
        max_relative_angle_deg: float,
    ) -> float:
        target_opening = -target_y_mm
        maximum_opening = -self.right_tip_position(max_relative_angle_deg).y
        if target_opening > maximum_opening + 1e-9:
            raise ValueError(
                "right_tip_y_mm exceeds the configured gripper geometry: "
                f"requested={target_opening:.3f} mm, maximum={maximum_opening:.3f} mm."
            )
        if target_opening <= 1e-9:
            return 0.0
        lower = 0.0
        upper = max_relative_angle_deg
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            if -self.right_tip_position(middle).y < target_opening:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)
