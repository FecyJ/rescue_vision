"""Picamera2 最小真机人工验收脚本。"""

from time import sleep

import cv2
from picamera2 import Picamera2


IMAGE_SIZE = (2304, 1296)

camera = Picamera2()
started = False

try:
    config = camera.create_preview_configuration(
        main={
            "size": IMAGE_SIZE,
            "format": "YUV420",
        },
        sensor={
            "output_size": IMAGE_SIZE,
            "bit_depth": 10,
        },
        buffer_count=2,
    )

    camera.configure(config)

    print("Camera configuration:")
    print(camera.camera_configuration())

    camera.start()
    started = True

    sleep(1.0)

    image_yuv = camera.capture_array("main")

    image_bgr = cv2.cvtColor(
        image_yuv,
        cv2.COLOR_YUV420p2BGR,
    )

    print("YUV shape:", image_yuv.shape)
    print("BGR shape:", image_bgr.shape)

    cv2.imwrite(
        "captured_frame_2304x1296.jpg",
        image_bgr,
    )

finally:
    if started:
        camera.stop()

    camera.close()
