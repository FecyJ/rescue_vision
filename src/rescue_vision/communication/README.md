# `communication`：UART 与直接 TCP 远程消息通道

本包提供两种协议无关传输基础：

- UART 字节收发、CRLF/LF 行分帧、接收时间戳和有界队列；
- 电脑与树莓派之间免密直连、带长度分帧和有界控制/观察队列的 TCP 消息
  通道。

本包不解释电机、舵机、轮速、IMU 或任务规则。当前 Rescue Car 协议适配器
和把远程调试意图转换为底盘动作的应用层尚未实现。

## 常用类

| 入口 | 用途 | 关键语义 |
| --- | --- | --- |
| `UartLineChannel` | 真实 8N1 UART 的后台读取与同步发送 | 单一读线程；必须 `start/stop` 或使用上下文管理 |
| `ReceivedUartLine` | 一条完整原始行 | `received_timestamp_ns` 为树莓派接收完成时的单调时钟 |
| `UartLineFramer` | 对任意字节分块执行 CRLF/LF 分帧 | 不解码、不按报文前缀分类 |
| `UartReceiveOverflowError` | 有界接收队列溢出 | 通道进入显式故障，不静默丢弃旧遥测或命令回复 |
| `RemoteTcpServer` / `connect_remote_client()` | 树莓派服务端与仓库内参考客户端 | 参考客户端只用于互操作测试；独立电脑端按线级协议实现 |
| `RemoteMessageConnection` | 双向控制/观察消息 | 控制和关键状态可靠；视频/地图/车辆快照丢旧保新 |
| `ReceivedRemoteMessage` | 校验后的 topic、属性和二进制载荷 | 分别保留发送时间与本机接收单调时间 |
| `DebugMotionCommand` | 调试运动意图 | 切换式死手当前状态、有效期、速度/转速及可选显式参考系目标朝向 |
| `DebugCaptureCommand` | 调试录制意图 | 开始、停止、抓拍、事件标记；不能指定车端路径 |
| `RemoteSessionStatus` | 会话权限、真实能力、限值和发布周期 | TCP 连接后的首条业务消息；周期发布 |
| `VideoFrameAttributes` / `MapSnapshotAttributes` | JPEG/PNG 二进制观察元数据 | 严格尺寸、坐标、标定和场地映射 |
| `VehicleStateObservation` | UART、轮速、朝向和安全状态 | 显式控制就绪、看门狗、急停和命令应用 ID |
| `CaptureStatusObservation` | 录制状态与最近请求结果 | 区分 accepted/completed/rejected/failed |

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

远程发送中，`send_control()` 和 `send_reliable_observation()` 共用可靠
优先队列，满时显式抛出 `RemoteQueueOverflowError`；后者用于首条/周期会话
状态和采集状态。`send_observation()` 用于视频、地图和车辆最新值，满时丢旧
保新。独立客户端必须按 topic 分开保存关键状态与大流量图像，不能照搬本
仓库参考客户端的单一观察接收队列。

## 远程运行配置

树莓派通常使用：

```yaml
remote:
  enabled: true
  role: server
  host: 0.0.0.0
  port: 8765
  access_mode: observe_only
  connect_timeout_ms: 2000.0
  io_timeout_ms: 100.0
  control_queue_capacity: 32
  observation_queue_capacity: 2
  max_header_bytes: 4096
  max_payload_bytes: 2097152
```

本仓库双机人工检查使用另一份本机 `runtime.yaml`，把 `role` 改为
`client`、`host` 改成树莓派地址。正式电脑端是独立项目，不读取此配置或
导入本包，只需在自己的配置中填写树莓派地址和端口。协议 v2 有意删除 PSK、
认证握手、HMAC 和密钥文件，赛场连接不需要分发或核对任何密钥。

比赛运行配置必须使用 `observe_only`。此模式在电脑侧禁止提交控制，在
树莓派侧也会拒绝误配置或恶意客户端发来的控制帧。`debug_control` 只用于
赛外数据采集和明确允许的调试，不能用于正式比赛遥控。

树莓派服务端装配：

```python
from collections.abc import Iterable

from rescue_vision.communication import (
    RemoteSessionStatus,
    RemoteTopic,
    VideoFrameAttributes,
)
from rescue_vision.config import load_runtime_config

def serve_observations(
    observations: Iterable[
        RemoteSessionStatus | tuple[bytes, VideoFrameAttributes]
    ],
) -> None:
    """调用方调度首条/周期会话状态和有界 JPEG 编码旁路。"""

    config = load_runtime_config("configs/runtime.yaml")
    server = config.remote.build_server()
    if server is None:
        raise RuntimeError("当前功能需要启用 remote")

    with server:
        connection = server.accept(timeout=10.0)
        with connection:
            first = True
            for observation in observations:
                if isinstance(observation, RemoteSessionStatus):
                    connection.send_reliable_observation(
                        RemoteTopic.SESSION_STATUS.value,
                        observation.to_payload(),
                        content_type="application/json",
                    )
                    first = False
                    continue
                if first:
                    raise RuntimeError(
                        "TCP 连接后的首条业务消息必须是会话状态"
                    )
                encoded_jpeg, attributes = observation
                # 观察队列满时丢旧保新，不反压相机主循环。
                connection.send_observation(
                    RemoteTopic.VIDEO_FRAME.value,
                    encoded_jpeg,
                    content_type="image/jpeg",
                    attributes=attributes.to_attributes(),
                )
```

这里的 `observations` 由尚待实现的车端应用调度；它必须先产出会话状态并按
协议周期重复，再混入相机编码旁路的最新帧。网络模块不会再次创建相机、改变
坐标系或生成新采集时间。

## 调试控制与观察 topic

| topic | 方向 | 载荷 |
| --- | --- | --- |
| `observation/session/status` | 树莓派 → 电脑 | 权限、能力、限值和发布周期 JSON |
| `control/debug/motion` | 电脑 → 树莓派 | `DebugMotionCommand` JSON |
| `control/debug/capture` | 电脑 → 树莓派 | `DebugCaptureCommand` JSON |
| `observation/video/frame` | 树莓派 → 电脑 | JPEG 与 `VideoFrameAttributes` |
| `observation/map/snapshot` | 树莓派 → 电脑 | PNG 与 `MapSnapshotAttributes` |
| `observation/vehicle/state` | 树莓派 → 电脑 | `VehicleStateObservation` JSON |
| `observation/capture/status` | 树莓派 → 电脑 | `CaptureStatusObservation` JSON |

`DebugMotionCommand` 的 `linear_velocity_m_s` 正负表示前后方向，
`angular_velocity_rad_s` 逆时针为正。`TARGET_HEADING` 模式必须同时声明：

- `heading_reference=field`：相对 `FieldPoint` 场地系 `+x`；
- `heading_reference=session_start`：相对本次调试会话启动朝向。

两者角度单位均为 rad、逆时针为正。当前没有 IMU/定位适配器，应用层不得
假装已经能执行目标朝向；在相应证据源完成前只能接受 `TWIST`。
电脑和树莓派的单调时钟不可直接比较；`issued_timestamp_ns` 只用于电脑侧
追踪，车端从 `ReceivedRemoteMessage.received_timestamp_ns` 开始计算
`valid_for_ms`。命令过期、序号重复或死手切到 OFF 时必须进入停车路径。
`deadman_enabled` 传输的是客户端切换后的当前状态，而不是手柄按键是否正被
按住；切换边沿识别、故障强制复位和禁止自动恢复由独立客户端实现。

录制控制不携带本地输出路径。车端必须从运行配置决定记录根目录，并继续
使用现有 `FrameRecorder` 的不覆盖和完整性规则。远程输入的 tags 仍要经过
数据采集领域校验。

电脑端实现的唯一跨项目依据是
[电脑端通信协议](../../../docs/电脑端通信协议.md)，不能从本 README 的
Python 示例反推线级兼容规则。协议支持任意扩展 topic、二进制 payload 和
有限标量 attributes。TCP 帧只提供长度边界、严格 header 和线路序号，不
提供认证、加密或防篡改；这是为降低现场部署复杂度而有意采用的边界。

## 双机人工验收

两端配置好地址和端口后，先只验证观察链路，不发送运动命令：

```bash
python manual_tests/remote_link.py \
  --config configs/runtime.yaml \
  --timeout-seconds 10
```

先启动树莓派 `server`，再启动电脑 `client`。工具会往返一条观察消息，
用于检查地址、端口和基本延迟；它不能作为图传吞吐、控制失联停车或
比赛网络合规的验收证据。

## 当前边界

当前没有小车命令编码、会话/车辆/采集状态发布器、心跳调度、看门狗、急停
恢复、电脑操控界面、远程意图到底盘的适配或运动脚本。实时视频编码和地图
渲染入口也尚未实现；现有观察 dataclass 只冻结协议 schema。它们属于 P1
后续阶段，不得把本模块描述为已完成远程驾驶、图传应用或整车控制。独立
电脑端不得导入本包；线级帧格式、消息 schema 和跨项目验收见
[电脑端通信协议交接](../../../docs/电脑端通信协议.md)。
