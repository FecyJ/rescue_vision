# tests/test_camera_stream.py

from time import perf_counter

import cv2
import numpy as np

from rescue_vision.camera.source import PiCameraSource
from rescue_vision.geometry.camera_model import CameraModel, CameraCalibration


IMAGE_SIZE = (2304, 1296)  # (width, height)

CAMERA_MATRIX = np.array(
    [
        [982.1,   0.0, 1152.0],
        [  0.0, 982.1,  648.0],
        [  0.0,   0.0,    1.0],
    ],
    dtype=np.float64,
)

# OpenCV fisheye 模型的 4 个参数：
# k1, k2, k3, k4
#
# 这是一组人为设置的轻度桶形畸变，
# 仅用于验证去畸变代码和坐标转换流程。
DISTORTION = np.array(
    [
        0.17,
        0.15,
        -0.05,
        0.00,
    ],
    dtype=np.float64,
).reshape(4, 1)


def main() -> None:
    camera_calibration = CameraCalibration(
        K=CAMERA_MATRIX,
        D=DISTORTION,
        new_K=CAMERA_MATRIX,
        image_size=IMAGE_SIZE,
    )
    camera_model = CameraModel(camera_calibration)

    camera = PiCameraSource(
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
                    f"timestamp: {frame.timestamp}"
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