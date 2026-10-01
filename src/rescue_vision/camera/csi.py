"""Raspberry Pi 5 物理 CSI 接口到 libcamera 相机编号的解析。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def resolve_csi_camera_index(
    csi_port: int,
    *,
    camera_info: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """按 Pi 5 设备路径选择 CAM/DISP0 或 CAM/DISP1，不依赖枚举顺序。

    只枚举、不打开相机。camera_info 可注入用于无硬件验证。
    """
    if isinstance(csi_port, bool) or not isinstance(csi_port, int) or csi_port not in (0, 1):
        raise ValueError(f"csi_port must be 0 or 1, got {csi_port!r}.")
    if camera_info is None:
        from picamera2 import Picamera2
        camera_info = Picamera2.global_camera_info()
    bus = "i2c@88000" if csi_port == 0 else "i2c@80000"
    for camera in camera_info:
        if f"/rp1/{bus}/" in camera['Id']:
            return int(camera['Num'])
    raise RuntimeError(
        f"No camera found on CAM/DISP{csi_port}; enumerated cameras={list(camera_info)!r}. "
        "Check the sensor overlay in /boot/firmware/config.txt and the cable connection."
    )
