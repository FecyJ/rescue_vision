# `communication`：UART 与直接 TCP 远程消息通道

本包提供两种协议无关传输基础：

- UART 字节收发、CRLF/LF 行分帧、接收时间戳和有界队列；
- 电脑与树莓派之间免密直连、带长度分帧和有界控制/观察队列的 TCP 消息
  通道。

本包不解释电机、舵机、轮速、IMU 或任务规则。Rescue Car 协议适配器和
远程调试意图到底盘动作的转换位于相邻
[`motion`](../motion/README.md) 包。

## 常用类和函数

| 入口 | 用途 | 关键语义 |
| --- | --- | --- |
| `UartLineChannel` | 真实 8N1 UART 后台读取与同步发送 | 必须 `start/stop` 或使用上下文管理 |
| `send()` | 写出任意非空 bytes | 不添加行结束符 |
| `send_line()` | 写出一条协议 payload | 禁止内含 CR/LF，自动添加 CRLF |
| `receive_line()` | 等待一条完整行 | 返回 `ReceivedUartLine`；超时抛出 `TimeoutError` |
| `check_health()` | 检查后台读取和设备状态 | 通道未启动或已经故障时抛出异常 |
| `ReceivedUartLine` | 一条完整原始行 | 带 UART 序号和树莓派接收单调时间 ns |
| `UartLineFramer` | 对任意字节分块执行 CRLF/LF 分帧 | 不解码、不解释报文前缀 |
| `RemoteTcpServer` | 树莓派 TCP 服务端 | 单监听端点；`accept()` 返回一个消息连接 |
| `connect_remote_client()` | 仓库内参考客户端 | 只用于互操作和人工检查 |
| `RemoteMessageConnection` | 一个 TCP 会话的双向消息通道 | 启动两个后台线程，严格验证线路序号 |
| `send_control()` | 提交可靠控制 | 仅 `debug_control` 客户端可用；队列满时失败 |
| `receive_control()` | 接收控制 | 树莓派 `observe_only` 会在传输层拒绝控制 |
| `send_reliable_observation()` | 提交会话/采集等关键状态 | 队列满时失败，不静默覆盖 |
| `send_observation()` | 提交视频、地图、车辆最新值 | 队列满时丢旧保新 |
| `receive_observation()` | 接收观察消息 | 仓库参考客户端的观察队列同样丢旧保新 |
| `DebugMotionCommand` | 调试运动 JSON schema | 死手、有效期、车体速度和可选目标朝向 |
| `DebugCaptureCommand` | 调试采集 JSON schema | 开始、停止、抓拍和事件标记 |
| `RemoteSessionStatus` | 会话权限、能力、限值和周期 | TCP 建立后的首条业务消息 |
| `VideoFrameAttributes` | JPEG 帧 header attributes | 严格尺寸、坐标系、时间和标定身份 |
| `VehicleStateObservation` | 车辆观察 JSON schema | UART、轮速、朝向、安全状态和应用命令 ID |
| `CaptureStatusObservation` | 采集观察 JSON schema | 当前记录状态及最近请求结果 |

## 1. 从运行配置装配 UART

设备名和全部传输参数只写入不提交的 `configs/runtime.yaml`。运行代码从同一
配置装配，不直接创建 PySerial：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
uart_channel = config.uart.build_channel()

if uart_channel is None:
    raise RuntimeError("必须在 runtime.yaml 中启用 uart")
```

`build_channel()` 只创建对象，不导入 PySerial，也不打开设备。下文 UART
片段都建立在这个 `uart_channel` 上。

## 2. 打开和关闭 UART

优先使用上下文管理器，确保正常和异常路径都关闭串口：

```python
with uart_channel:
    uart_channel.check_health()
    # 在这里执行下文的 UART 发送和接收片段。
```

需要把生命周期接入更大的应用对象时，也可以显式调用：

```python
uart_channel.start()
try:
    uart_channel.check_health()
    # 应用主循环。
finally:
    uart_channel.stop()
```

`stop()` 可以重复调用。关闭 UART 不等于小车已经停车；需要停车时必须先由
`motion` 在通道仍可写时发送制动命令。

## 3. 发送 UART 数据

行协议通常使用 `send_line()`。它统一添加 `\r\n`，调用方不得自己附加：

```python
with uart_channel:
    # 实际写出的 bytes 是 b"v\r\n"。
    uart_channel.send_line(b"v")
```

确实需要发送不带行结束符的二进制数据时才使用 `send()`：

```python
with uart_channel:
    uart_channel.send(b"\x01\x02\x03")
```

`communication` 不校验 `b"v"` 是否为合法电控命令。Rescue Car 命令应通过
`MotionController` 发送，不要在业务模块中散落手写协议字符串。

## 4. 接收 UART 行

`receive_line()` 返回原始 bytes、线路序号和树莓派接收时间：

```python
with uart_channel:
    received = uart_channel.receive_line(timeout=0.5)
    print(
        received.sequence,
        received.received_timestamp_ns,
        received.payload,
    )
```

超时表示当前没有完整行，不表示通道已经故障：

```python
with uart_channel:
    try:
        received = uart_channel.receive_line(timeout=0.1)
    except TimeoutError:
        uart_channel.check_health()
    else:
        process_raw_uart_line(received)
```

这里的 `process_raw_uart_line()` 代表调用方提供的分发函数。Rescue Car
调用方应改用 `MotionController.receive_message()`，由唯一协议层解析
`OK`、`ERR`、轮速遥测和未知前缀。

STM32 自身的毫秒计时应作为协议字段另外保存，不能替代树莓派
`received_timestamp_ns` 与相机帧对齐。命令回复和主动遥测可能交错，不能
假定“发送后的下一行就是这条命令的回复”。

## 5. 单独使用 `UartLineFramer`

`UartLineFramer` 用于没有 `UartLineChannel` 的字节流测试或其他传输适配，
不会访问硬件：

```python
from rescue_vision.communication import UartLineFramer

framer = UartLineFramer(max_line_bytes=64)

assert framer.feed(b"OK m=0.2") == ()
assert framer.feed(b",0.2\r\nt1,0,0\n") == (
    b"OK m=0.2,0.2",
    b"t1,0,0",
)
```

生产 UART 已在 `UartLineChannel` 内部使用同一个 framer，不应在调用方再次
分帧。

## 6. 从运行配置装配树莓派 TCP 服务端

树莓派通常设置 `remote.role: server`。正式比赛必须使用 `observe_only`；
`debug_control` 只允许赛外受监督调试：

```python
config = load_runtime_config("configs/runtime.yaml")
remote_server = config.remote.build_server()

if remote_server is None:
    raise RuntimeError("必须在 runtime.yaml 中启用 remote")
```

`build_server()` 尚不绑定端口。实际监听、接受一个客户端以及连接线程的
生命周期如下：

```python
with remote_server:
    remote_connection = remote_server.accept(timeout=10.0)
    with remote_connection:
        # TCP 建立后的首条业务消息必须是 session/status。
        ...
```

`RemoteTcpServer` 当前一次只接受一个连接。应用退出、远程断线或处理异常时，
两个上下文管理器会依次关闭连接和监听端点。

## 7. 构造并发送首条会话状态

下面构造一个真实可用的“只观察视频”会话状态。它不宣称尚未装配的运动、
采集、地图或车辆状态能力：

```python
import time
import uuid

from rescue_vision.communication import (
    RemoteSessionStatus,
    RemoteTopic,
)

session_status = RemoteSessionStatus(
    session_id=f"session-{uuid.uuid4()}",
    server_instance_id=f"server-{uuid.uuid4()}",
    timestamp_ns=time.monotonic_ns(),
    access_mode=config.remote.access_mode,
    motion_control_available=False,
    capture_control_available=False,
    video_stream_available=True,
    map_snapshot_available=False,
    vehicle_state_available=False,
    capture_status_available=False,
    target_heading_control_available=False,
    session_status_period_ms=1_000,
    vehicle_state_period_ms=None,
    map_snapshot_period_ms=None,
    capture_status_period_ms=None,
    video_nominal_fps=15.0,
    max_linear_velocity_m_s=None,
    max_angular_velocity_rad_s=None,
    max_motion_command_valid_for_ms=500,
)
```

在前文 `with remote_connection` 内，把它作为首条可靠观察发送：

```python
remote_connection.send_reliable_observation(
    RemoteTopic.SESSION_STATUS.value,
    session_status.to_payload(),
    content_type="application/json",
)
```

应用还必须按 `session_status_period_ms` 周期发送新状态。除
`timestamp_ns` 外，同一连接后续状态必须与首条完全一致。可用
`dataclasses.replace()` 只更新时间：

```python
from dataclasses import replace

session_status = replace(
    session_status,
    timestamp_ns=max(
        time.monotonic_ns(),
        session_status.timestamp_ns + 1,
    ),
)
remote_connection.send_reliable_observation(
    RemoteTopic.SESSION_STATUS.value,
    session_status.to_payload(),
)
```

能力字段必须描述当前会话真实运行的发布器，不能因为 schema 已定义就写
`true`。运动控制尤其要求视频和车辆状态同时可用。

## 8. 发送视频等最新值观察

以下片段建立在前文已经发送首条 `session_status` 的连接上。`encoded_jpeg`
来自相机编码旁路；attributes 中的时间是该相机帧采集时间：

```python
from rescue_vision.communication import (
    ImageCoordinateSystem,
    RemoteTopic,
    VideoFrameAttributes,
)

video_attributes = VideoFrameAttributes(
    frame_sequence=frame.sequence,
    timestamp_ns=frame.timestamp_ns,
    width=frame.image_bgr.shape[1],
    height=frame.image_bgr.shape[0],
    coordinate_system=ImageCoordinateSystem.RAW_PIXEL,
    intrinsics_fingerprint_sha256=None,
)

remote_connection.send_observation(
    RemoteTopic.VIDEO_FRAME.value,
    encoded_jpeg,
    content_type="image/jpeg",
    attributes=video_attributes.to_attributes(),
)
```

这里的 `frame` 和 `encoded_jpeg` 由尚待实现的正式车端发布器提供。
`send_observation()` 非阻塞提交；观察队列满时丢弃旧值，避免网络反压相机
实时路径。地图和车辆最新状态使用相同入口。

会话状态、采集状态等不可被覆盖的关键消息应使用
`send_reliable_observation()`：

```python
remote_connection.send_reliable_observation(
    RemoteTopic.CAPTURE_STATUS.value,
    capture_status.to_payload(),
    content_type="application/json",
)
```

这里的 `capture_status` 由采集领域适配器提供。可靠队列满时会抛出
`RemoteQueueOverflowError`，调用方必须进入显式故障处理。

## 9. 在树莓派接收和解析调试控制

只有服务端配置为 `debug_control` 时，连接才允许收到 control。接收后先按
topic 分派，再用相应 schema 解析：

```python
from rescue_vision.communication import (
    DebugCaptureCommand,
    DebugMotionCommand,
    RemoteTopic,
)

received = remote_connection.receive_control(timeout=0.1)

if received.topic == RemoteTopic.DEBUG_MOTION.value:
    motion_command = DebugMotionCommand.from_payload(received.payload)
    handle_motion_command(received, motion_command)
elif received.topic == RemoteTopic.DEBUG_CAPTURE.value:
    capture_command = DebugCaptureCommand.from_payload(received.payload)
    handle_capture_command(received, capture_command)
else:
    raise ValueError(f"unsupported control topic: {received.topic!r}")
```

这里的两个 `handle_*` 是应用领域适配器。运动指令应直接交给
`RemoteMotionExecutor.execute(received)`，不要在 communication 层换算轮速。
录制指令也不能携带电脑端路径；车端输出根目录必须来自自己的运行配置。

`received.sender_timestamp_ns` 是电脑端单调时间，只用于电脑侧追踪；
`received.received_timestamp_ns` 才是树莓派本机接收时间。两台机器的单调
时钟零点不可直接比较。

## 10. 仓库内参考客户端

本仓库参考客户端只用于自动测试、双机互操作和人工检查。使用另一份
`runtime.yaml`，将 `remote.role` 配为 `client`，并填写树莓派地址：

```python
client_config = load_runtime_config("configs/runtime.client.yaml")
remote_client = client_config.remote.connect_client()

if remote_client is None:
    raise RuntimeError("必须在客户端配置中启用 remote")
```

把首条状态接收封装为一个只依赖“已启动连接”的函数：

```python
from rescue_vision.communication import (
    RemoteMessageConnection,
    RemoteSessionStatus,
    RemoteTopic,
)


def receive_first_status(
    connection: RemoteMessageConnection,
) -> RemoteSessionStatus:
    received = connection.receive_observation(timeout=2.0)
    if received.topic != RemoteTopic.SESSION_STATUS.value:
        raise RuntimeError("首条远程消息不是 session/status")

    status = RemoteSessionStatus.from_payload(received.payload)
    print(status.access_mode, status.video_stream_available)
    return status
```

当本地配置与服务端状态都允许调试运动时，参考客户端可以这样构造和发送一条
twist：

```python
import time

from rescue_vision.communication import (
    DebugMotionCommand,
    MotionControlMode,
    RemoteTopic,
)


def send_debug_twist(
    connection: RemoteMessageConnection,
    status: RemoteSessionStatus,
) -> None:
    command = DebugMotionCommand(
        command_id="manual-drive-001",
        issued_timestamp_ns=time.monotonic_ns(),
        valid_for_ms=200,
        deadman_enabled=True,
        control_mode=MotionControlMode.TWIST,
        linear_velocity_m_s=0.10,
        angular_velocity_rad_s=0.30,
    )

    if not status.motion_control_available:
        raise RuntimeError("服务端没有声明 motion_control capability")
    assert status.max_linear_velocity_m_s is not None
    assert status.max_angular_velocity_rad_s is not None
    if abs(command.linear_velocity_m_s) > status.max_linear_velocity_m_s:
        raise ValueError("线速度超过服务端声明上限")
    if abs(command.angular_velocity_rad_s) > status.max_angular_velocity_rad_s:
        raise ValueError("角速度超过服务端声明上限")
    if command.valid_for_ms > status.max_motion_command_valid_for_ms:
        raise ValueError("命令有效期超过服务端声明上限")

    connection.send_control(
        RemoteTopic.DEBUG_MOTION.value,
        command.to_payload(),
    )
```

最后用一个资源代码段组合前文装配和两个操作函数：

```python
with remote_client:
    status = receive_first_status(remote_client)
    send_debug_twist(remote_client, status)
```

本地 `runtime.client.yaml` 也必须设置 `access_mode: debug_control`，否则
`send_control()` 会在客户端传输层拒绝发送。独立电脑端不得导入本仓库
Python 包；其唯一跨项目依据是
[电脑端通信协议](../../../docs/电脑端通信协议.md)。

## Topic 与 payload 对照

| topic | 方向 | 载荷 |
| --- | --- | --- |
| `observation/session/status` | 树莓派 → 电脑 | `RemoteSessionStatus` JSON |
| `control/debug/motion` | 电脑 → 树莓派 | `DebugMotionCommand` JSON |
| `control/debug/capture` | 电脑 → 树莓派 | `DebugCaptureCommand` JSON |
| `observation/video/frame` | 树莓派 → 电脑 | JPEG 与 `VideoFrameAttributes` |
| `observation/map/snapshot` | 树莓派 → 电脑 | PNG 与 `MapSnapshotAttributes` |
| `observation/vehicle/state` | 树莓派 → 电脑 | `VehicleStateObservation` JSON |
| `observation/capture/status` | 树莓派 → 电脑 | `CaptureStatusObservation` JSON |

`DebugMotionCommand.linear_velocity_m_s` 正负表示前后，
`angular_velocity_rad_s` 逆时针为正。`TARGET_HEADING` 必须声明 `field` 或
`session_start` 参考系；当前没有 IMU/定位适配器，车端只能执行 `TWIST`。

## 队列、故障和降级语义

- UART 行超过 `max_line_bytes`、接收队列溢出、设备断开或后台读取异常都会
  使通道进入故障，不会静默丢弃命令回复或遥测。
- `send_control()` 和 `send_reliable_observation()` 共用可靠优先队列，满时
  抛出 `RemoteQueueOverflowError`。
- `send_observation()` 的发送队列以及参考客户端的观察接收队列均丢旧保新。
  独立客户端应按 topic 分开保存关键状态和大流量图像。
- `observe_only` 在电脑侧禁止发送控制，在树莓派侧拒绝入站控制。
- TCP 帧提供长度边界、严格 header 和线路序号，但不提供认证、加密或防篡改。
  协议 v2 有意删除 PSK、HMAC、握手和密钥文件。
- 远程断开、UART 故障和应用退出后的车辆停车由应用、`motion` 与 STM32
  看门狗共同保证。当前固件资料没有失联看门狗，不能把关闭连接当作停车证据。

## 人工检查入口

仅检查树莓派监听、直接连接和首条最小会话状态：

```bash
python manual_tests/remote_link.py \
  --config configs/runtime.yaml \
  --timeout-seconds 10
```

真实相机图传和远程采集分别使用：

```bash
python manual_tests/remote_video.py \
  --config configs/runtime.yaml \
  --timeout-seconds 30

python manual_tests/remote_capture.py \
  --config configs/runtime.yaml \
  --output-root /data/rescue-targets/remote_test \
  --timeout-seconds 30
```

这些脚本是单链路真机人工验收入口，不是正式发布器或应用入口，也不提供运动
控制或自动重连。

## 当前边界

当前没有正式应用使用的会话/车辆/采集状态发布器、心跳调度、STM32 看门狗、
急停恢复、电脑操控界面或运动脚本。地图渲染入口也尚未实现。现有观察
dataclass 只冻结协议 schema，人工测试脚本不能视为正式远程驾驶或图传应用。
