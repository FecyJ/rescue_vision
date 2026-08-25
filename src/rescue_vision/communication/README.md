# `communication`：UART 与直接 TCP 远程消息通道

本包提供两种协议无关传输基础：

- UART 字节收发、`0x00` 定界、COBS 解码、接收时间戳和有界队列；
- 电脑与树莓派之间免密直连、带长度分帧和有界控制/观察队列的 TCP 消息
  通道。

本包不解释电机、舵机、编码器、IMU 或任务规则。STM32 协议适配器和
远程调试意图到底盘动作的转换位于相邻
[`motion`](../motion/README.md) 包。

## 常用类和函数

| 入口 | 用途 | 关键语义 |
| --- | --- | --- |
| `UartFrameChannel` | 真实 8N1 UART 后台读取与同步发送 | 必须管理生命周期；真实设备独占打开 |
| `send()` | 写出任意非空 bytes | 仅供底层使用，不添加分帧 |
| `send_frame()` | 写出一帧解码态 payload | 自动 COBS 编码并添加 `0x00` |
| `receive_frame()` | 等待一帧已解码数据 | 返回 `ReceivedUartFrame`；超时抛出 `TimeoutError` |
| `check_health()` | 检查后台读取和设备状态 | 通道未启动或已经故障时抛出异常 |
| `ReceivedUartFrame` | 一帧 COBS 解码结果 | 带 UART 序号和树莓派接收完成单调时间 ns |
| `UartFrameFramer` | 对任意字节分块执行定界和 COBS 解码 | 损坏/超长帧丢弃并在下一定界符恢复 |
| `cobs_encode()` / `cobs_decode()` | 无硬件 COBS 编解码 | 不添加或消费末尾 `0x00` |
| `RemoteTcpServer` | 树莓派 TCP 服务端 | 单监听端点；`accept()` 返回一个消息连接 |
| `connect_remote_client()` | 仓库内参考客户端 | 只用于互操作和人工检查 |
| `RemoteMessageConnection` | 一个 TCP 会话的双向消息通道 | 启动两个后台线程，严格验证线路序号 |
| `send_control()` | 提交可靠控制或图像模式请求 | 运动/夹爪/采集仅 `debug_control` 可用；图像模式请求不执行动作；队列满时失败 |
| `receive_control()` | 接收控制或图像模式请求 | `observe_only` 仍拒绝执行控制，但允许 `control/video/mode` |
| `send_reliable_observation()` | 提交会话/采集等关键状态 | 队列满时失败，不静默覆盖 |
| `send_observation()` | 提交视频、地图、车辆最新值 | 按 topic 合并，队列满时丢旧保新 |
| `receive_observation()` | 接收观察消息 | 仓库参考客户端也按 topic 保留最新值 |
| `DebugMotionCommand` | 调试运动 JSON schema | 死手、有效期、车体速度和可选目标朝向 |
| `DebugGripperCommand` | 调试夹爪 JSON | 有效期和张开/闭合扳机按压布尔状态 |
| `DebugCaptureCommand` | 调试采集 JSON schema | 开始、停止、抓拍和事件标记 |
| `VideoModeCommand` | 图像模式请求 JSON schema | 客户端选择 `raw`、`perception` 或 `bev`，不改变车辆动作 |
| `VideoFrameMode` | 图像内容枚举 | 原图、推理叠加图或机器人局部地面 BEV |
| `RemoteSessionStatus` | 会话权限、能力、限值和周期 | TCP 建立后的首条业务消息 |
| `VideoFrameAttributes` | JPEG 帧 header attributes | 严格尺寸、坐标系、时间和标定身份 |
| `MapStateObservation` | 轻量动态地图 JSON | FieldPoint 机器人位姿、已确认目标、时间、置信度和不确定度 |
| `MapTargetState` | 动态地图目标项 | 唯一轨迹 ID、类别、FieldPoint 位置和质量 |
| `VehicleStateObservation` | 车辆观察 JSON | UART、轮速、夹爪角度、显式安全模式和命令 ID |
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

真实 PySerial 设备使用 `exclusive=True`。驾驶进程已打开 UART 时，另一个
监测器或测试进程会以 `Resource temporarily unavailable` 失败，而不是同时
读取并把同一 COBS 字节流拆散。监测前必须先退出驾驶程序。

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

## 3. 发送 UART 帧

业务代码应通过 `MotionController` 生成 CRC 和固定消息。确需单独使用传输层时，
`send_frame()` 接受 COBS 编码前的完整 payload，并自动完成分帧：

```python
with uart_channel:
    uart_channel.send_frame(decoded_protocol_frame)
```

`decoded_protocol_frame` 由 `motion.protocol` 产生，已经包含消息类型和 CRC。
调用方不得自行 COBS 编码、添加 `0x00`，也不得用底层 `send()` 绕过统一分帧。

## 4. 接收 UART 帧

`receive_frame()` 返回 COBS 解码后的 bytes、线路序号和树莓派接收完成时间：

```python
with uart_channel:
    received = uart_channel.receive_frame(timeout=0.5)
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
        received = uart_channel.receive_frame(timeout=0.1)
    except TimeoutError:
        uart_channel.check_health()
    else:
        process_decoded_uart_frame(received)
```

这里的 `process_decoded_uart_frame()` 代表调用方提供的分发函数。底盘调用方
应改用 `MotionController.receive_message()`，由唯一协议层校验 CRC、固定长度、
消息方向、枚举和值域，再返回命令回复、编码器/IMU 或系统状态。

STM32 自身的微秒计时作为协议字段另外保存，不能替代树莓派
`received_timestamp_ns` 与相机帧对齐。命令回复和主动遥测可能交错，不能
假定“发送后的下一帧就是这条命令的回复”。

## 5. 单独使用 `UartFrameFramer`

`UartFrameFramer` 用于没有 `UartFrameChannel` 的字节流测试或其他传输适配，
不会访问硬件：

```python
from rescue_vision.communication import UartFrameFramer, cobs_encode

framer = UartFrameFramer(max_frame_bytes=64)
wire = cobs_encode(b"\x80\x01\x00") + b"\x00"

assert framer.feed(wire[:2]) == ()
assert framer.feed(wire[2:]) == (b"\x80\x01\x00",)
```

生产 UART 已在 `UartFrameChannel` 内部使用同一个 framer，不应在调用方再次
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
    VideoFrameMode,
)

session_status = RemoteSessionStatus(
    session_id=f"session-{uuid.uuid4()}",
    server_instance_id=f"server-{uuid.uuid4()}",
    timestamp_ns=time.monotonic_ns(),
    access_mode=config.remote.access_mode,
    motion_control_available=False,
    gripper_control_available=False,
    capture_control_available=False,
    video_stream_available=True,
    video_modes=(VideoFrameMode.RAW,),
    map_state_available=False,
    vehicle_state_available=False,
    capture_status_available=False,
    target_heading_control_available=False,
    session_status_period_ms=1_000,
    vehicle_state_period_ms=None,
    map_state_period_ms=None,
    capture_status_period_ms=None,
    video_nominal_fps=15.0,
    max_linear_velocity_m_s=None,
    max_angular_velocity_rad_s=None,
    max_control_command_valid_for_ms=500,
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
`true`。运动和夹爪控制尤其要求视频和车辆状态同时可用。

## 8. 发送视频等最新值观察

以下片段建立在前文已经发送首条 `session_status` 的连接上。`encoded_jpeg`
来自相机编码旁路；attributes 中的时间是该相机帧采集时间：

```python
from rescue_vision.communication import (
    ImageCoordinateSystem,
    RemoteTopic,
    VideoFrameMode,
    VideoFrameAttributes,
)

video_attributes = VideoFrameAttributes(
    frame_sequence=frame.sequence,
    timestamp_ns=frame.timestamp_ns,
    width=frame.image_bgr.shape[1],
    height=frame.image_bgr.shape[0],
    coordinate_system=ImageCoordinateSystem.RAW_PIXEL,
    calibration_id=None,
    mode=VideoFrameMode.RAW,
)

remote_connection.send_observation(
    RemoteTopic.VIDEO_FRAME.value,
    encoded_jpeg,
    content_type="image/jpeg",
    attributes=video_attributes.to_attributes(),
)
```

这里的 `frame` 和 `encoded_jpeg` 由车端相机发布器提供；若选择
`VideoFrameMode.PERCEPTION`，发布器应把 perception 叠加结果编码；选择
`VideoFrameMode.BEV` 时必须发送 `bev_pixel` 坐标系和完整地面范围/
`mm_per_pixel` attributes。两者都要在 attributes 回显实际 `mode`。
`send_observation()` 非阻塞提交；同 topic 新值覆盖旧值，容量不足时再丢弃
最早等待的其他 topic，避免网络反压相机
实时路径。地图和车辆最新状态使用相同入口。
运行应用必须为所有需要同时保留的最新值 topic 预留槽位；当前手动采集同时
发布视频、车辆状态和地图，因此 `observation_queue_capacity` 至少为 `3`。

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
`RemoteQueueOverflowError`，调用方必须进入显式故障处理。可靠队列和
最新值队列共享发送唤醒，并优先取可靠消息，保证先提交的首条会话状态不会
被紧随其后的车辆状态或视频抢先发送。

## 9. 在树莓派接收和解析调试控制

运动、夹爪和采集 control 只有服务端配置为 `debug_control` 时才允许。图像模式
请求是唯一例外：它不产生车辆动作，在 `observe_only` 会话中也可收发。接收后
先按 topic 分派，再用相应 schema 解析：

```python
from rescue_vision.communication import (
    DebugCaptureCommand,
    DebugGripperCommand,
    DebugMotionCommand,
    RemoteTopic,
    VideoModeCommand,
)

received = remote_connection.receive_control(timeout=0.1)

if received.topic == RemoteTopic.VIDEO_MODE.value:
    mode_command = VideoModeCommand.from_payload(received.payload)
    if mode_command.mode not in session_status.video_modes:
        raise ValueError("服务端未声明该图像模式")
    handle_video_mode(mode_command)
elif received.topic == RemoteTopic.DEBUG_MOTION.value:
    motion_command = DebugMotionCommand.from_payload(received.payload)
    handle_motion_command(received, motion_command)
elif received.topic == RemoteTopic.DEBUG_GRIPPER.value:
    gripper_command = DebugGripperCommand.from_payload(received.payload)
    handle_gripper_command(received, gripper_command)
elif received.topic == RemoteTopic.DEBUG_CAPTURE.value:
    capture_command = DebugCaptureCommand.from_payload(received.payload)
    handle_capture_command(received, capture_command)
else:
    raise ValueError(f"unsupported control topic: {received.topic!r}")
```

这里的四个 `handle_*` 是应用领域适配器。运动指令应直接交给
`RemoteMotionExecutor.execute(received)`，不要在 communication 层换算轮速。
夹爪指令应交给 `RemoteGripperExecutor.execute(received)`，不要在此层拼接
Rescue Car 的 `g` 字符串。
录制指令也不能携带电脑端路径；车端输出根目录必须来自自己的运行配置。

`received.sender_timestamp_ns` 是电脑端单调时间，只用于电脑侧追踪；
`received.received_timestamp_ns` 才是树莓派本机接收时间。两台机器的单调
时钟零点不可直接比较。

## 9.1 选择图像传输模式

电脑端先解析首条 `RemoteSessionStatus` 的 `video_modes`，只从已声明的模式中
选择。请求不改变运动、夹爪或采集权限，因此 `observe_only` 也可发送；车端
下一帧会在 `VideoFrameAttributes.mode` 回显实际内容：

```python
import time

from rescue_vision.communication import (
    RemoteTopic,
    VideoFrameMode,
    VideoModeCommand,
)

mode = VideoFrameMode.PERCEPTION
if mode not in session_status.video_modes:
    raise RuntimeError("车端未声明 perception 图传")
request = VideoModeCommand(
    request_id=f"video-{time.monotonic_ns()}",
    issued_timestamp_ns=time.monotonic_ns(),
    mode=mode,
)
remote_connection.send_control(
    RemoteTopic.VIDEO_MODE.value,
    request.to_payload(),
)
```

`raw` 是当前相机帧（已按车端配置决定是否去畸变）；`perception` 是同一坐标系
图像上叠加目标框、颜色掩码、K0、置信度和质量信息的结果。车端只保留最新待
推理帧，推理旁路故障会终止当前会话，不会把未经声明的原图伪装成
`perception`。`bev` 仅在可用地面映射包含 BEV 配置时声明；它在独立最新帧
后台旁路生成，坐标为机器人局部 `BevPixel`，不是场地地图或定位结果。车端
启用场地定位旁路时，BEV JPEG 可以叠加同帧中心十字、安全区和终端关联；客户端
不得从叠加颜色反推机器状态，结构化全局位姿只读取 `observation/map/state`。

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
    if command.valid_for_ms > status.max_control_command_valid_for_ms:
        raise ValueError("命令有效期超过服务端声明上限")

    connection.send_control(
        RemoteTopic.DEBUG_MOTION.value,
        command.to_payload(),
    )
```

同一参考客户端通过持续发送夹爪扳机状态控制开合。此片段承接前文
`remote_client` 和 `status`；车端运行配置持有机械端点和固定速度，客户端
只发送经过本地按下阈值判断后的布尔状态：

```python
from rescue_vision.communication import DebugGripperCommand


def send_debug_gripper(
    connection: RemoteMessageConnection,
    status: RemoteSessionStatus,
    *,
    open_pressed: bool,
    close_pressed: bool,
) -> None:
    command = DebugGripperCommand(
        command_id=f"manual-grip-{time.monotonic_ns()}",
        issued_timestamp_ns=time.monotonic_ns(),
        valid_for_ms=200,
        open_pressed=open_pressed,
        close_pressed=close_pressed,
    )
    if not status.gripper_control_available:
        raise RuntimeError("服务端没有声明 gripper_control capability")
    if command.valid_for_ms > status.max_control_command_valid_for_ms:
        raise ValueError("命令有效期超过服务端声明上限")
    connection.send_control(
        RemoteTopic.DEBUG_GRIPPER.value,
        command.to_payload(),
    )
```

非零状态要在按住期间以不超过 `valid_for_ms / 3` 的周期持续刷新。例如
`valid_for_ms=200` 时可每 50 ms 发送一次；左扳机按下时传
`open_pressed=true`，右扳机按下时传 `close_pressed=true`。松开或手柄断开
时立即发送一次两个字段均为 `false` 的命令，随后清除本地待发状态。两个
扳机同时按下时车端停止，避免冲突方向运动。

最后用一个资源代码段组合前文装配和三个操作函数：

```python
with remote_client:
    status = receive_first_status(remote_client)
    send_debug_twist(remote_client, status)
    send_debug_gripper(
        remote_client,
        status,
        open_pressed=False,
        close_pressed=True,
    )
    # 松开扳机时停止车端继续改变舵机目标。
    send_debug_gripper(
        remote_client,
        status,
        open_pressed=False,
        close_pressed=False,
    )
```

本地 `runtime.client.yaml` 只有发送运动、夹爪或采集命令时才必须设置
`access_mode: debug_control`；图像模式请求在 `observe_only` 也允许发送。
独立电脑端不得导入本仓库
Python 包；其唯一跨项目依据是
[电脑端通信协议](../../../docs/电脑端通信协议.md)。

## Topic 与 payload 对照

| topic | 方向 | 载荷 |
| --- | --- | --- |
| `observation/session/status` | 树莓派 → 电脑 | `RemoteSessionStatus` JSON |
| `control/debug/motion` | 电脑 → 树莓派 | `DebugMotionCommand` JSON |
| `control/debug/gripper` | 电脑 → 树莓派 | `DebugGripperCommand` JSON |
| `control/debug/capture` | 电脑 → 树莓派 | `DebugCaptureCommand` JSON |
| `control/video/mode` | 电脑 → 树莓派 | `VideoModeCommand` JSON；不执行车辆动作 |
| `observation/video/frame` | 树莓派 → 电脑 | JPEG 与含实际 `mode` 的 `VideoFrameAttributes` |
| `observation/map/state` | 树莓派 → 电脑 | `MapStateObservation` JSON |
| `observation/vehicle/state` | 树莓派 → 电脑 | `VehicleStateObservation` JSON |
| `observation/capture/status` | 树莓派 → 电脑 | `CaptureStatusObservation` JSON |

电脑端持有静态场地图和 `FieldPoint → UI 像素` 映射；车端不再绘制或发送
PNG。`robot.localized=true` 时全部定位字段必须存在，为 false 时全部为 null；
客户端必须清除旧机器人位置。`targets` 只包含已经确认且具有 FieldPoint 的
轨迹，当前无全局目标提供者时为空数组，不能用机器人局部坐标填充。

`DebugMotionCommand.linear_velocity_m_s` 正负表示前后，
`angular_velocity_rad_s` 逆时针为正。`TARGET_HEADING` 必须声明 `field` 或
`session_start` 参考系；`field` 的零角是场地 `+x` 向右，正角朝红色安全区
方向（`+y`）旋转。当前没有 IMU/定位适配器，车端只能执行 `TWIST`。
`DebugGripperCommand` v2 是短有效期的持续扳机状态；超时、断线、全部松开
或两个方向同时按下会停止继续改变舵机目标，但不会自动跳到开/闭端点。

## 队列、故障和降级语义

- COBS 损坏、空帧和超长帧会被丢弃并在下一 `0x00` 恢复；接收队列溢出、
  设备断开或后台读取异常会使通道进入故障，不会静默丢弃已验证消息。
- 同一设备的第二次打开会被独占锁拒绝；不得移除该锁来同时运行驾驶和监测。
- 读线程故障仍允许退出路径在关闭串口前尽力写出最后一帧停车命令；健康检查
  和接收仍立即报告原始故障，应用不会在失去遥测后继续驾驶。
- `send_control()` 和 `send_reliable_observation()` 共用可靠优先队列，满时
  抛出 `RemoteQueueOverflowError`。
- `send_observation()` 的发送队列以及参考客户端的观察接收队列均按 topic
  合并并丢旧保新；高频视频不会覆盖仍有槽位的最新车辆状态。
  独立客户端应按 topic 分开保存关键状态和大流量图像。
- `observe_only` 在电脑侧和树莓派侧仍禁止运动、夹爪、采集 control；仅允许
  `control/video/mode` 这一条不执行动作的图像模式请求。
- TCP 帧提供长度边界、严格 header 和线路序号，但不提供认证、加密或防篡改。
  协议 v2 有意删除 PSK、HMAC、握手和密钥文件。
- TCP EOF、连接复位和发送断管统一报告为 `RemoteDisconnectedError`。服务端
  应先停车并清理当前会话，再回到等待新连接；正常客户端退出不得终止整个
  服务进程。
- 远程断开、UART 故障和应用退出后的车辆停车由应用、`motion` 与 STM32
  看门狗共同保证。树莓派协议已经实现，STM32 看门狗仍需实现和真机验收；
  不能把关闭连接当作停车证据。

## 人工检查入口

仅检查树莓派监听、直接连接和首条最小会话状态：

```bash
python manual_tests/remote_link.py \
  --config configs/runtime.yaml \
  --timeout-seconds 10
```

真实相机图传检查和受监督手动采集分别使用：

```bash
python manual_tests/remote_video.py \
  --config configs/runtime.yaml \
  --timeout-seconds 30

rescue-vision-manual-capture \
  --config configs/runtime.yaml \
  --output-root /data/rescue-targets/remote_test \
  --supervised-physical-stop-ready \
  --accept-timeout-seconds 30
```

`remote_video.py` 只是单链路人工检查。`rescue-vision-manual-capture` 是
赛外单客户端手动驾驶与采集入口；断线停车后可接受客户端新建的连接，但车端
不主动重连或恢复使能，也不是比赛应用。

## 当前边界

当前手动采集入口已有会话/车辆/采集状态发布和周期调度；仍没有 STM32
看门狗、急停恢复、电脑操控界面、运动脚本、地图渲染或比赛发布器。该赛外
入口不能视为正式比赛远程驾驶应用。
