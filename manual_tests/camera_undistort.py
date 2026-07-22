"""真机去畸变人工验收；需要通过参数提供标定文件。"""

import argparse

from time import perf_counter

import cv2
import numpy as np

from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.geometry.camera_model import CameraModel

IMAGE_SIZE = (2304, 1296)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsics", required=True)
    args = parser.parse_args()
    camera_model = CameraModel.from_json(args.intrinsics)

    camera = RpicamSource(
        image_size=IMAGE_SIZE,
        fps=30,
        lens_position=1.0,
    )

    frame_count = 0
    start_time = perf_counter()

    try:
        camera.start()

        while True:
            frame = camera.read()

            # 在这里调用后续视觉模块
            image = frame.image_bgr

            # 去畸变
            undistorted_image = camera_model.undistort_image(image)

            # 仅用于调试显示，缩小可降低桌面显示开销
            preview = cv2.resize(
                undistorted_image,
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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
