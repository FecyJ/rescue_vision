import time
import serial

PORT = "/dev/ttyAMA10"
BAUDRATE = 115200
TEST_COUNT = 1000

success = 0
failed = 0

with serial.Serial(PORT, BAUDRATE, timeout=0.2) as uart:
    for sequence in range(TEST_COUNT):
        message = f"UART,{sequence:04d},ABCDEFGHIJ\r\n".encode()

        uart.reset_input_buffer()
        uart.write(message)
        uart.flush()

        received = uart.read(len(message))

        if received == message:
            success += 1
        else:
            failed += 1
            print(
                f"第 {sequence} 包失败："
                f"发送={message!r}，接收={received!r}"
            )

        time.sleep(0.002)

print(f"测试完成：成功 {success}，失败 {failed}")