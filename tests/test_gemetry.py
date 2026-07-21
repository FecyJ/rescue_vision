import cv2
import numpy as np

from rescue_vision.geometry.camera_model import CameraModel, CameraCalibration
from rescue_vision.geometry.ground_projector import (
    BevConfig,
    GroundProjector,
)
from rescue_vision.geometry.types import RawPixel, GroundPoint, UndistortedPixel

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


def draw_bev_source_region(
    undistorted_image: np.ndarray,
    projector,
) -> np.ndarray:
    config = projector.bev_config

    ground_corners = [
        GroundPoint(config.x_max, config.y_max),
        GroundPoint(config.x_max, config.y_min),
        GroundPoint(config.x_min, config.y_min),
        GroundPoint(config.x_min, config.y_max),
    ]

    image_corners = [
        projector.ground_to_pixel(point)
        for point in ground_corners
    ]

    polygon = np.array(
        [
            [round(point.u), round(point.v)]
            for point in image_corners
        ],
        dtype=np.int32,
    )

    output = undistorted_image.copy()

    cv2.polylines(
        output,
        [polygon],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    for index, point in enumerate(polygon):
        cv2.circle(
            output,
            tuple(point),
            radius=6,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.putText(
            output,
            str(index),
            tuple(point + [8, -8]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return output


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
    cv2.imwrite("undistorted.jpg", undistorted_frame)

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

    debug_image = draw_bev_source_region(
        undistorted_frame,
        projector,
    )

    cv2.imwrite(
        "bev_source_region.jpg",
        debug_image,
    )

if __name__ == "__main__":
    main()