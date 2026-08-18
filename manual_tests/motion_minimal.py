from rescue_vision.config import load_runtime_config
import time

config = load_runtime_config("configs/runtime.yaml")

# build_channel() 和 build_controller() 只创建对象，尚未打开串口。
channel = config.uart.build_channel()
controller = config.motion.build_controller(channel)

with channel:
    try:
        controller.drive(
            linear_velocity_m_s=0.20,
            angular_velocity_rad_s=0.60,
        )
        time.sleep(1.0)
        controller.soft_brake()
    finally:
        controller.soft_brake()