# `communication`：协议无关 UART 行通道

本包负责 UART 字节收发、CRLF/LF 行分帧、接收时间戳、有界队列和资源
生命周期，不解释电机、舵机、遥测、IMU 或任务规则。当前 Rescue Car
协议适配器尚未实现；后续适配器消费这里的原始行并区分 `OK`、`ERR`、
轮速遥测和未来 IMU 遥测。

## 常用类

| 入口 | 用途 | 关键语义 |
| --- | --- | --- |
| `UartLineChannel` | 真实 8N1 UART 的后台读取与同步发送 | 单一读线程；必须 `start/stop` 或使用上下文管理 |
| `ReceivedUartLine` | 一条完整原始行 | `received_timestamp_ns` 为树莓派接收完成时的单调时钟 |
| `UartLineFramer` | 对任意字节分块执行 CRLF/LF 分帧 | 不解码、不按报文前缀分类 |
| `UartReceiveOverflowError` | 有界接收队列溢出 | 通道进入显式故障，不静默丢弃旧遥测或命令回复 |

## 从运行配置创建通道

本机实际设备名只写入不提交的 `configs/runtime.yaml`：

```yaml
uart:
  enabled: true
  device: /dev/serial0
  baudrate: 115200
  read_timeout_ms: 100.0
  write_timeout_ms: 100.0
  receive_queue_capacity: 256
  max_line_bytes: 512
```

生产装配从同一运行配置创建通道：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
channel = config.uart.build_channel()
if channel is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用 UART")

with channel:
    # 传输层只负责完整写出并添加 CRLF；命令合法性由后续电控协议层校验。
    channel.send_line(b"v")
    received = channel.receive_line(timeout=0.5)
    print(
        received.sequence,
        received.received_timestamp_ns,
        received.payload,
    )
```

`receive_line()` 返回原始 `bytes`。协议层应按自己的字符集严格解码，并
保留未知前缀，不能因当前固件只定义轮速遥测就丢弃未来 IMU 报文。
STM32 自身的毫秒计时应作为协议字段另外保存，不能替代树莓派
`received_timestamp_ns` 与相机帧对齐。

## 生命周期与故障

- `start()` 才会延迟导入 PySerial 并打开配置中的设备；导入配置和运行无
  硬件测试不会访问串口。
- 所有读取只发生在一个后台线程；上层不得再直接读取同一个串口对象。
- 多线程发送由写锁保持单次消息连续，但命令回复与主动遥测仍会交错；
  后续协议层必须统一解复用，不能假定“发送后的下一行就是回复”。
- 行超过 `max_line_bytes`、队列溢出、设备断开或读取异常都会使通道进入
  故障。调用 `check_health()`、`send()` 或 `receive_line()` 会得到异常。
- `stop()` 可重复调用，并在正常及异常路径关闭设备。车辆失联停车仍必须
  由后续 STM32 看门狗和电控协议层保证，不能把关闭串口当作停车证据。

## 当前边界

当前没有小车命令编码、遥测 dataclass、IMU schema、心跳、看门狗、急停
恢复、手动驾驶或运动脚本。它们属于 P1 后续阶段，不得把本模块描述为已
完成底盘通信或整车控制。
