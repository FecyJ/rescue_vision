import cv2

from rescue_vision.camera.source import PiCameraSource

camera = PiCameraSource(
    image_size=(2304, 1296), 
    fps=30, 
    lens_position=1.0
)

try:
    camera.start()
    frame = camera.read()

    print(f"Frame {frame.sequence} captured at {frame.timestamp} ns, shape: {frame.image_bgr.shape}")

    cv2.imwrite("tmp/captured_frame.jpg", frame.image_bgr)
finally:
    camera.stop()