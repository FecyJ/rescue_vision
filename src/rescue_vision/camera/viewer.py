"""用于命令行采集与回放的可选 OpenCV 画面查看器。"""

from __future__ import annotations

import math

import cv2
import numpy as np


class OpenCvFrameViewer:
    """显示可缩放的 BGR/灰度帧；Q 或 Esc 请求停止显示。"""

    def __init__(
        self,
        title: str,
        *,
        maximum_size: tuple[int, int] = (1280, 720),
    ) -> None:
        if not title:
            raise ValueError("title must be non-empty.")
        if (
            len(maximum_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in maximum_size
            )
        ):
            raise ValueError(
                f"maximum_size must be positive integers, got {maximum_size}."
            )
        self.title = title
        self.maximum_size = maximum_size
        self._opened = False
        self._display_size: tuple[int, int] | None = None

    def show(self, image: np.ndarray, *, delay_ms: int = 1) -> bool:
        """显示一帧；返回 False 表示用户按下 Q 或 Esc。"""

        if image.ndim not in {2, 3}:
            raise ValueError(
                f"image must have 2 or 3 dimensions, got {image.shape}."
            )
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError(
                "color image must have 3 BGR channels, got "
                f"{image.shape}."
            )
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"image must be non-empty, got {image.shape}.")
        if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or delay_ms < 1:
            raise ValueError(f"delay_ms must be a positive integer, got {delay_ms}.")

        if not self._opened:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            image_height, image_width = image.shape[:2]
            maximum_width, maximum_height = self.maximum_size
            scale = min(
                maximum_width / image_width,
                maximum_height / image_height,
                1.0,
            )
            self._display_size = (
                max(1, round(image_width * scale)),
                max(1, round(image_height * scale)),
            )
            cv2.resizeWindow(
                self.title,
                *self._display_size,
            )
            self._opened = True

        assert self._display_size is not None
        source_size = (int(image.shape[1]), int(image.shape[0]))
        preview = (
            image
            if source_size == self._display_size
            else cv2.resize(
                image,
                self._display_size,
                interpolation=cv2.INTER_AREA,
            )
        )
        cv2.imshow(self.title, preview)
        key = cv2.waitKey(delay_ms) & 0xFF
        return key not in {27, ord("q"), ord("Q")}

    def close(self) -> None:
        if not self._opened:
            return
        try:
            cv2.destroyWindow(self.title)
        finally:
            self._opened = False
            self._display_size = None

    def __enter__(self) -> OpenCvFrameViewer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def playback_delay_ms(
    previous_timestamp_ns: int | None,
    timestamp_ns: int,
    *,
    speed: float,
    maximum_delay_ms: int = 1000,
) -> int:
    """把相邻采集时间换算为回放等待时间，并限制异常长停顿。"""

    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed must be positive and finite, got {speed}.")
    if (
        isinstance(maximum_delay_ms, bool)
        or not isinstance(maximum_delay_ms, int)
        or maximum_delay_ms < 1
    ):
        raise ValueError("maximum_delay_ms must be a positive integer.")
    if previous_timestamp_ns is None:
        return 1
    delta_ns = max(0, timestamp_ns - previous_timestamp_ns)
    delay_ms = max(1, round(delta_ns / 1_000_000 / speed))
    return min(delay_ms, maximum_delay_ms)
