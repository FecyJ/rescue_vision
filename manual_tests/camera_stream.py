from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2

from rescue_vision.camera.rpicam_source import RpicamSource


def main() -> None:
    parser = argparse.ArgumentParser(description="实时预览相机，按 Enter 保存当前原始帧。")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/camera_snapshots"),
        help="图片保存目录（默认：output/camera_snapshots）",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera = RpicamSource(
        image_size=(2304, 1296),
        fps=30,
        lens_position=1.0,
    )

    frame_count = 0
    start_time = perf_counter()
    print(f"聚焦 camera 窗口：Enter 拍照，Q/Esc 退出；保存目录：{args.output_dir.resolve()}")

    try:
        camera.start()

        while True:
            frame = camera.read()

            # 在这里调用后续视觉模块
            image = frame.image_bgr

            # 仅用于调试显示，缩小可降低桌面显示开销
            preview = cv2.resize(
                image,
                (1152, 648),
            )

            cv2.imshow("camera", preview)

            frame_count += 1
            elapsed = perf_counter() - start_time

            if elapsed >= 1.0:
                print(
                    f"FPS: {frame_count / elapsed:.1f}, "
                    f"sequence: {frame.sequence}, "
                    f"timestamp_ns: {frame.timestamp_ns}"
                )

                frame_count = 0
                start_time = perf_counter()

            key = cv2.waitKey(1) & 0xFF
            if key in (10, 13):
                filename = (
                    f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}"
                    f"_{frame.sequence}.png"
                )
                path = args.output_dir / filename
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"保存图片失败：{path.resolve()}")
                print(
                    f"已保存：{path.resolve()}，sequence={frame.sequence}，"
                    f"timestamp_ns={frame.timestamp_ns}，shape={image.shape}"
                )
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        try:
            camera.stop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
