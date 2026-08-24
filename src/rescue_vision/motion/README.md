# `motion`：小车运动控制与远程调试执行

本包当前把车体速度意图转换为 Rescue Car v2.0 双轮速度命令，控制双舵机夹爪，
并执行电脑端传来的 `control/debug/motion` 和
`control/debug/gripper`。它依赖调用方注入 UART 行通道和远程连接，不创建
串口、TCP 服务、定位器或规划器。

电控协议依据本次提供的
`docs/Rescue Car v2.0 — 电控代码使用说明书.pdf`：115200 8N1、CRLF
行结束，`m<L>,<R>` 设置轮速、`g<L°>,<R°>` 设置夹爪双舵机、
`b0,0` 柔和停车、`e` 急停，STM32 回传 `OK`、`ERR` 和 10 Hz `t...`
遥测。

树莓派与 STM32 后续共同切换的目标接口以
[`docs/树莓派与单片机通信协议.md`](../../../docs/树莓派与单片机通信协议.md)
为唯一权威。目标协议使用 COBS、CRC16 和固定长度二进制消息，统一运动、夹爪、
固件看门狗及编码器/IMU 遥测；当前代码尚未实现，实施时直接替换本文以下旧
文本协议入口，不增加兼容层。

## 常用类和函数

| 入口 | 输入 | 输出或语义 |
| --- | --- | --- |
| `MotionLimits` | 轮距、车体/车轮速度与单轮加速度上限、远程有效期上限 | 创建时严格校验 |
| `MotionController` | UART 行通道、`MotionLimits` | Rescue Car 运动控制器 |
| `MotionController.drive()` | 前进速度 m/s、逆时针角速度 rad/s | 差速换算后设置左右轮目标 |
| `drive_wheel_limited()` | 分别合法的车体线速度和角速度 | 必要时同比缩放并返回实际 twist，使单轮不超限 |
| `set_wheel_speeds()` | 左右轮速度 m/s | 绕过车体 twist 换算，仍执行轮速限幅校验 |
| `update()` | 可选本机单调时间 ns | 按单轮最大加速度推进并下发目标；返回是否发送 |
| `forward()` / `backward()` | 非负速度 m/s | 直行前进/后退 |
| `turn_left()` / `turn_right()` | 非负角速度 rad/s | 原地左转/右转 |
| `set_gripper_angles()` | 左右舵机角度 degree | 同时下发严格 `[0, 180]` 角度 |
| `gripper_target_angles_deg` | 无 | 启动遥测或本进程最近下发的左右舵机目标；尚无时为 `None` |
| `soft_brake()` / `emergency_stop()` | 无 | 固件斜坡制动/紧急停止 |
| `query_state()` | 无 | 请求固件立即返回状态 |
| `receive_message()` | 可选等待秒数 | 忽略空行，返回轮速、安全状态、命令回复或未知回传 |
| `CarSafetyStatus` | `s1` 状态行 | 固件单调时间、看门狗、锁存急停、命令年龄和停车原因 |
| `drain_messages()` | 无 | 非阻塞排空当前 UART 回传 |
| `RemoteMotionExecutor.execute()` | `ReceivedRemoteMessage` | 校验远程运动消息后执行 |
| `RemoteMotionExecutor.check_timeout()` | 可选本机单调时间 ns | 到期时停车，返回是否触发 |
| `GripperCalibration` | 左右开/闭端点与固定速度全行程时间 | 严格验证安全范围及每组端点相加为 194° |
| `RemoteGripperExecutor.execute()` | `ReceivedRemoteMessage` | 接受一帧持续夹爪扳机状态 |
| `RemoteGripperExecutor.update()` | 可选本机单调时间 ns | 按按压方向以配置速度渐进下发舵机目标 |
| `RemoteGripperExecutor.check_timeout()` / `stop()` | 可选本机单调时间 ns | 超时或退出时停止推进，保留当前角度 |
| `run_remote_motion()` | 远程接收器、执行器、退出回调 | 持续收命令、排空回传、分派其他 control，并在退出时停车 |
| `ManualMotionLogWriter` | recording 内的 `motion.jsonl` | 顺序写入运动/夹爪命令、遥测、UART 扩展和停车事件 |
| `ManualMotionLogWriter.record_gripper()` | `ExecutedRemoteGripper` | 记录扳机状态及 applied/stopped/expired 结果 |
| `record_gripper_timeout()` | 命令 ID、单调时间 ns | 记录持续夹爪命令到期停止 |
| `inspect_manual_motion_log()` | `motion.jsonl` 路径 | 严格校验 schema、事件序号和时间范围摘要 |

机器人坐标系原点为两驱动轮接地点连线的中点，沿用项目约定：`x` 向前、`y`
向左、`z` 向上。左右轮速度正值均表示前进；车体角速度逆时针为正：

```text
left  = linear - angular × wheel_track / 2
right = linear + angular × wheel_track / 2
```

底层 `drive()` 的超限命令会被拒绝，不会静默截断。手动遥控执行器使用
`drive_wheel_limited()`：当线速度和角速度分别合法、但二者合成使外侧轮超限
时，会同比缩放两个分量以保持曲率，并在执行结果和运动日志中记录实际 twist。
合法轮速目标则由 `max_wheel_acceleration_m_s2` 限制每个轮子的速度变化率；
这同时限制直线加速和转向跳变。`target_heading` 在定位或 IMU 尚未提供其显式
参考系前也会被拒绝并停车。

## 1. 从运行配置装配

先在 `configs/runtime.yaml` 填入实测轮距和调试限速，并启用 `uart` 与
`motion`。需要远程夹爪时，还要实测左右开/闭安全端点和固定速度全行程时间，
写入 `motion.gripper` 后单独启用。所有路径、设备名、机械参数和上限只从
这份配置取得：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")

# build_channel() 和 build_controller() 只创建对象，尚未打开串口。
channel = config.uart.build_channel()
controller = config.motion.build_controller(channel)

if channel is None or controller is None:
    raise RuntimeError("必须在 runtime.yaml 中启用 uart 和 motion")
```

下文的本地运动片段都建立在以上 `channel` 和 `controller` 上。实际应用必须
用上下文管理器打开 UART；离开上下文前应进入停车路径：

```python
with channel:
    try:
        # 在这里执行下文的 drive、转向、回传处理等片段。
        ...
    finally:
        controller.soft_brake()
```

`with channel` 负责打开和关闭串口，但关闭串口本身不是停车命令，所以
`soft_brake()` 必须在 UART 仍然可写时调用。

## 2. 使用 `drive()` 控制车体速度

`drive()` 适合上层规划器、手柄或调试逻辑输出车体 twist。它只更新目标，
实时循环必须持续调用 `update()` 才会按配置斜率渐进下发。以下片段应放进
前文 `with channel` 的 `try` 内：

```python
# 以 0.20 m/s 前进，同时以 0.60 rad/s 向左转弯。
controller.drive(
    linear_velocity_m_s=0.20,
    angular_velocity_rad_s=0.60,
)
controller.update()

# 右转使用负角速度。
controller.drive(
    linear_velocity_m_s=0.15,
    angular_velocity_rad_s=-0.40,
)
controller.update()
```

轮距只参与 twist 到左右轮速的换算。线速度、角速度或换算后的任一轮速度
超过 `MotionLimits` 时，调用会抛出 `ValueError`，不会把请求悄悄截断。
每次 `update()` 使用树莓派本机单调时间计算允许的最大轮速增量；时钟倒退会
被拒绝。`run_remote_motion()` 已在每轮循环自动调用它，手动采集应用不需要
另建定时器。其他直接调用方应以不超过 100 ms 的有界周期调用 `update()`，
否则目标只会停留在最近一次实际下发值。
远程手柄在死手仍开启时回中，执行器会调用 `drive(0, 0)` 设置零目标，再由
同一 `update()` 循环按 `max_wheel_acceleration_m_s2` 逐级降低左右轮命令；
不会直接发送 `m0,0` 或固件 `b0,0`。死手关闭、命令过期、非法输入及退出
仍使用独立的柔和停车安全路径。

## 3. 直接设置左右轮速度

已经拥有左右轮目标的底层算法可以直接调用 `set_wheel_speeds()`：

```python
# 左轮 0.10 m/s、右轮 0.25 m/s，小车向左走弧线。
controller.set_wheel_speeds(
    left_m_s=0.10,
    right_m_s=0.25,
)
controller.update()
```

该方法不使用 `wheel_track_m` 做换算，但仍检查
`max_wheel_velocity_m_s`，并和 `drive()` 共用
`max_wheel_acceleration_m_s2` 与 `update()`。上层一般应优先使用
`drive()`，避免多个模块各自实现差速公式。

## 4. 前进、后退和原地转向

便捷方法的参数都要求非负；方向由方法名决定：

```python
controller.forward(speed_m_s=0.15)
controller.backward(speed_m_s=0.10)

controller.turn_left(angular_velocity_rad_s=0.80)
controller.turn_right(angular_velocity_rad_s=0.80)
```

这些调用只设置新的目标，不等待动作完成，也不自行休眠。动作时长和
`update()` 控制周期由应用主循环决定。

## 5. 控制夹爪双舵机

夹爪使用 Rescue Car 的 `g<左角度>,<右角度>` 指令，左右字段不能交换。
以下片段应放在前文已打开的 `with channel` 内：

```python
calibration = config.motion.gripper.build_calibration()
if calibration is None:
    raise RuntimeError("当前运行配置未启用夹爪机械标定")

# 直接角度 API 也只消费同一份车端标定，不另写开/闭常量。
controller.set_gripper_angles(
    left_angle_deg=calibration.open_left_angle_deg,
    right_angle_deg=calibration.open_right_angle_deg,
)
```

两个角度都必须是 `[0, 180]` 内有限 degree；越界、NaN 和 Infinity 会在
写 UART 前被拒绝。该 API 只设置显式双舵机角度，不提供硬编码的
`open_gripper()` / `close_gripper()`：机械连杆、舵机方向和装配零点会改变
开合标定，协议说明书示例不能当作所有车辆的通用闭合值。

夹爪位置由 STM32 锁存，和底盘运动目标具有不同安全语义。
`soft_brake()`、运动命令超时或断线停车不会自动改变夹爪角度，以免运输时
意外丢落；调用方退出前若确实需要释放，必须根据现场安全条件显式发送另一组
已标定角度。10 Hz 遥测中的 `servo_left_deg` / `servo_right_deg` 是固件
`CarState` 报告的目标角度，不是舵机物理位置反馈。
低层 `set_gripper_angles()` 仍忠实编码调用方给出的两个角度；194° 和约束
属于远程夹爪执行器及其 `GripperCalibration`，不会暗中改变其他直接调用方。

## 6. 柔和停车和紧急停止

正常结束动作、远程死手关闭或命令超时时使用柔和停车：

```python
controller.soft_brake()
```

检测到必须立即制动的整车安全事件时发送固件急停：

```python
controller.emergency_stop()
```

`soft_brake()` 发送 `b0,0`，由固件按配置减速度降到零；`emergency_stop()`
发送 `e`。当前固件的急停是否锁存、如何恢复仍需真机冻结，应用不得假设发送
下一条速度命令就能安全解除急停。

## 7. 查询并处理电控回传

发送状态查询后，命令回复和主动 10 Hz 遥测可能交错，因此不能假定下一行
一定是查询回复：

```python
from rescue_vision.motion import (
    CarCommandReply,
    CarSafetyStatus,
    CarTelemetry,
    UnknownCarMessage,
)

controller.query_state()
message = controller.receive_message(timeout=0.5)

if isinstance(message, CarTelemetry):
    print(
        message.received_timestamp_ns,
        message.controller_timestamp_ms,
        message.actual_left_m_s,
        message.actual_right_m_s,
    )
elif isinstance(message, CarSafetyStatus):
    # 只有新鲜且 watchdog_armed=True 的 s1 状态才能作为固件保护证据。
    print(
        message.watchdog_timeout_ms,
        message.watchdog_armed,
        message.emergency_stop_latched,
        message.stop_reason.value,
    )
elif isinstance(message, CarCommandReply):
    print("OK" if message.succeeded else "ERR", message.detail)
elif isinstance(message, UnknownCarMessage):
    # 未来 IMU 等新前缀在显式支持前会保留为原始 bytes。
    print("unknown car message:", message.payload)
```

实时循环应持续消费回传。只发送而不接收会使 UART 有界队列最终溢出：

```python
for message in controller.drain_messages():
    # 调用方可在这里分发轮速遥测、日志或未来 IMU 消息。
    print(message)
```

## 8. 装配远程调试执行器

远程执行复用同一个 `controller` 和同一命令有效期上限，不创建第二套规则：

```python
executor = config.motion.build_remote_executor(controller)
gripper_executor = config.motion.build_remote_gripper_executor(controller)
if executor is None:
    raise RuntimeError("必须在 runtime.yaml 中启用 motion")
if config.motion.gripper.enabled and gripper_executor is None:
    raise RuntimeError("夹爪配置已启用但执行器未完成装配")
```

`gripper_executor is None` 是正常的禁用语义；应用此时必须发布
`gripper_control=false`。不能在调用处补一组默认开闭角度。

应用已经取得一个通过 `RemoteMessageConnection.receive_control()` 接收的消息
时，可以执行单条远程运动指令。UART 必须保持打开；示例结束前显式停车：

```python
with channel:
    try:
        received = remote_connection.receive_control(timeout=0.1)
        outcome = executor.execute(received)
        print(
            outcome.command_id,
            outcome.result.value,
            outcome.deadline_timestamp_ns,
        )
    finally:
        executor.stop()
```

`execute()` 只接受 `control/debug/motion`、`application/json`、空 attributes
和 `TWIST`。线速度或角速度自身超限、非法或不支持的命令会先尝试柔和停车，
再抛出 `RemoteMotionError`。只有二者分别合法但差速合成超出单轮上限时，
执行器才会保持曲率同比缩放；`outcome.linear_velocity_m_s` 和
`outcome.angular_velocity_rad_s` 是实际采用值。

夹爪 topic 由独立执行器处理；以下片段继续假设 UART 已打开，且
`received_gripper` 来自同一远程连接：

```python
gripper_outcome = gripper_executor.execute(received_gripper)
print(
    gripper_outcome.command_id,
    gripper_outcome.result.value,
    gripper_outcome.open_pressed,
    gripper_outcome.close_pressed,
)
```

它只接受 `control/debug/gripper`、空 attributes 和 JSON v2。两个字段都
必须是 boolean：仅闭合为 `true` 时闭合，仅张开为 `true` 时张开；两者相同
（都松开或同时按下）时停止。单方向按下命令返回 `applied` 表示车端已接受
持续控制状态；停止命令返回
`stopped`，过期命令返回 `expired`。这些结果都不代表舵机有物理位置反馈。

接受命令后，应用循环必须以有界周期推进。以下片段承接前文
`gripper_executor`；手动采集入口使用 20 ms 轮询：

```python
timed_out_command_id = gripper_executor.check_timeout()
if timed_out_command_id is None:
    gripper_executor.update()
else:
    record_gripper_timeout(timed_out_command_id)
```

`update()` 从 `MotionController.gripper_target_angles_deg` 取得最近下发值或
STM32 遥测目标；若旧目标不满足新约束，会先投影到
`left + right = 194°` 的有效线段，随后只推进左角并以
`right = 194° - left` 重建每条远程 UART 指令。尚未收到
任何目标时安全等待，不猜测启动角度。非法 envelope、
按压状态或有效期会清除当前持续状态并抛出 `RemoteGripperError`。该执行器不会
自行改变底盘速度；应用把异常传播到 `run_remote_motion()` 时，外层仍会先走
统一停车路径。

跨模块应用可通过 `on_other_control` 把同一连接中的夹爪和采集命令分别交给
上述执行器与采集状态机，
通过 `on_cycle` 执行短时、非阻塞的相机旁路工作；`on_motion_executed` 和
`on_motion_timeout` 用于发布车辆状态。这些钩子任一抛出异常都会进入同一
停车路径，不得在回调中执行无界等待。

## 9. 持续执行一个远程调试会话

应用层应使用 `run_remote_motion()` 持续收命令、检查有效期并排空 UART
回传。以下函数建立在前文的 `channel`、`controller` 和 `executor` 上：

```python
from collections.abc import Callable

from rescue_vision.communication import RemoteMessageConnection
from rescue_vision.motion import ParsedCarMessage, run_remote_motion


def run_debug_motion_session(
    remote_connection: RemoteMessageConnection,
    stop_requested: Callable[[], bool],
) -> None:
    def publish_or_record(message: ParsedCarMessage) -> None:
        # 这里接入车辆状态发布器或运动日志；不要阻塞 UART 排空。
        print(message)

    with channel:
        run_remote_motion(
            remote_connection,
            executor,
            stop_requested=stop_requested,
            on_car_message=publish_or_record,
        )
```

这里的 `remote_connection` 必须已经启动，而且应用已经按电脑端协议发送首条
`observation/session/status`，并真实提供该状态声明的 video/vehicle
observation。`motion` 不重复实现这些发布器。完整 TCP 生命周期和消息发送
方式见 [`communication` README](../communication/README.md)。

该循环只应在 `remote.access_mode: debug_control` 的赛外受监督配置中运行。
比赛配置必须保持 `observe_only`；传输层会拒绝入站控制。

## 时间、故障与降级

手动采集应用会在 recording 有效期间创建 `ManualMotionLogWriter`，并把
运动/夹爪执行结果、运动/夹爪超时回调和 UART 回传写入同一单调时间轴。
调用方不应另建第二份运动日志；完整目录由
`rescue-vision-check-recording` 一并检查。直接使用 writer 时必须严格分开
启动、写入和关闭：

```python
from time import monotonic_ns

from rescue_vision.motion import ManualMotionLogWriter

motion_log = ManualMotionLogWriter(recording_directory / "motion.jsonl")
motion_log.start(timestamp_ns=monotonic_ns())
try:
    # outcome 来自前文 RemoteMotionExecutor.execute()。
    motion_log.record_motion(outcome)
    # gripper_outcome 来自前文 RemoteGripperExecutor.execute()。
    motion_log.record_gripper(gripper_outcome)
finally:
    motion_log.stop(timestamp_ns=monotonic_ns())
```

这里的 `recording_directory` 由上游 recording 会话产生；正式手动采集优先
直接运行 `rescue-vision-manual-capture`，由应用保证它和 session v4
`auxiliary_streams` 声明一致。脚本动作段和计划身份不由本 writer 猜测。

- `valid_for_ms` 从树莓派完成接收该消息的单调时间开始计算，不比较两台机器
  互不共享零点的 `issued_timestamp_ns`。
- 远程夹爪不受运动死手控制，但任一按下状态必须持续刷新；客户端松开时发送
  两个状态均为 `false` 的命令。到期、断线或调用 `stop()` 会停止后续角度推进并
  保留最近目标，不自动开爪或闭爪。
- 远程 twist 的 payload 和电脑端协议不变；最大加速度以及夹爪机械端点/
  全行程时间来自车端运行配置
  运行配置，不由客户端逐条指定，避免绕过统一安全上限。
- 死手保持开启且 twist 回到零时，零目标和其他有效目标一样经过单轮加速度
  限制；真机已发现固件 `b0,0` 会使实测轮速直接归零，因此普通回中不再
  依赖该固件斜坡。
- 死手关闭、命令过期、非法 payload、未知控制模式和循环正常退出均进入柔和
  停车；协议或通信异常也会尝试停车并继续抛出原始异常。
- `drain_messages()` / `run_remote_motion()` 会排空 10 Hz 回传。未知前缀保留
  为 `UnknownCarMessage`，非 ASCII 原始行也按相同方式保留；只有 CR/LF 的
  空 UART 行没有业务内容，会被忽略。以上情况不会导致手动采集循环退出或被
  误判为命令成功；录制期间非空原始 bytes 以十六进制写入 `unknown_uart`。
  冻结的 `s1` 行解析为 `CarSafetyStatus` 并写入运动日志
  当前记录格式；当前固件尚不产生该状态。`parse_car_line()` 对损坏的已知
  `t...` / `s1...` 报文仍严格抛出 `ValueError`；实时
  `MotionController.receive_message()` 会把单条损坏报文隔离为
  `UnknownCarMessage`，防止固件遥测和 `OK` 输出交错时终止采集。后续正常
  报文仍会继续解析。
- 当前固件资料没有失联看门狗。进程被强杀、树莓派掉电或 UART 物理断开时，
  Python 无法保证停车；只能在架空轮或有物理急停、人员全程监督的环境验证，
  不能把本模块的超时当作固件级失控保护。

真机还需核对实际左右轮正方向、实测轮距、轮速上限、急停锁存/恢复、断 UART
和杀进程行为。上述项目目前均为“未验证”。
