import cv2
import numpy as np

from rescue_vision.geometry.camera_model import CameraModel, CameraCalibration
from rescue_vision.geometry.ground_projector import (
    BevConfig,
    GroundProjector,
)
from rescue_vision.geometry.types import RawPixel

IMAGE_SIZE = (1536, 864)  # OpenCV 顺序：(width, height)

CAMERA_MATRIX = np.array(
    [
        [510.0,   0.0, 768.0],
        [  0.0, 510.0, 432.0],
        [  0.0,   0.0,   1.0],
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
        -0.0600,
         0.0120,
        -0.0020,
         0.0003,
    ],
    dtype=np.float64,
).reshape(4, 1)


def main():
    camera_config = CameraCalibration(
        K=CAMERA_MATRIX,
        D=DISTORTION,
        new_K=CAMERA_MATRIX,
        image_size=IMAGE_SIZE,
    )
    camera_model = CameraModel(camera_config)

    projector = GroundProjector(
        image_to_ground=np.array([
            [1.2, 0.1, -800.0],
            [0.0, -2.0, 1200.0],
            [0.0, -0.001, 1.0],
        ]),
        bev_config=BevConfig(
            x_min=100.0,
            x_max=2500.0,
            y_min=-1200.0,
            y_max=1200.0,
            mm_per_pixel=5.0,
        ),
    )


    raw_frame = cv2.imread("frame.png")

    undistorted_frame = camera_model.undistort_image(
        raw_frame
    )

    bev = projector.make_bev_image(undistorted_frame)
    cv2.imwrite("bev.jpg", bev)

    raw_contact = RawPixel(
        u=820.0,
        v=620.0,
    )

    undistorted_contact = camera_model.undistort_pixel(
        raw_contact
    )

    ground_contact = projector.pixel_to_ground(
        undistorted_contact
    )

    print(ground_contact)

if __name__ == "__main__":
    main()