# `motion`：小车运动控制与远程调试执行

本包把车体速度意图转换为 Rescue Car v2.0 双轮速度命令，并执行电脑端传来的
`control/debug/motion`。它依赖调用方提供的 UART 行通道和远程连接，不创建
串口、TCP 服务、定位器或规划器。

电控协议依据本次提供的
`docs/Rescue Car v2.0 — 电控代码使用说明书.pdf`：115200 8N1、CRLF
行结束，`m<L>,<R>` 设置轮速、`b0,0` 柔和停车、`e` 急停，STM32 回传
`OK`、`ERR` 和 10 Hz `t...` 遥测。

## 公共入口

| 入口 | 输入 | 输出或语义 |
| --- | --- | --- |
| `MotionLimits` | 轮距、车体/车轮速度上限、远程有效期上限 | 所有值在入口严格校验 |
| `MotionController.drive()` | 前进速度 m/s、逆时针角速度 rad/s | 按双轮差速换算并发送左右轮目标 |
| `forward()` / `backward()` | 非负速度 m/s | 直行便捷函数 |
| `turn_left()` / `turn_right()` | 非负角速度 rad/s | 原地转向便捷函数 |
| `soft_brake()` / `emergency_stop()` | 无 | 分别发送固件柔和制动和急停 |
| `receive_message()` | 可选等待秒数 | `CarTelemetry`、`CarCommandReply` 或 `UnknownCarMessage` |
| `RemoteMotionExecutor.execute()` | `ReceivedRemoteMessage` | 验证 topic、JSON、死手、有效期和限速后执行 |
| `run_remote_motion()` | 远程接收器、执行器、退出回调 | 持续收命令、检查超时、排空 UART 回传；退出时停车 |

机器人坐标系沿用项目约定：`x` 向前、`y` 向左、`z` 向上。左右轮速度正值
均表示前进；车体角速度逆时针为正。因此差速换算为：

```text
left  = linear - angular × wheel_track / 2
right = linear + angular × wheel_track / 2
```

超限命令会被拒绝，不会静默截断。`target_heading` 在定位或 IMU 尚未提供其
显式参考系前也会被拒绝并停车。

## 按运行配置装配

先在 `configs/runtime.yaml` 填入实测轮距和调试限速，再显式启用 UART 与
motion。配置对象只装配资源，不会提前打开串口：

```python
from rescue_vision.config import load_runtime_config
from rescue_vision.motion import run_remote_motion

config = load_runtime_config("configs/runtime.yaml")
channel = config.uart.build_channel()
controller = config.motion.build_controller(channel)
executor = config.motion.build_remote_executor(controller)

if channel is None or controller is None or executor is None:
    raise RuntimeError("UART 和 motion 必须在调试配置中启用")

stop_requested = False

def should_stop() -> bool:
    return stop_requested

def run_debug_motion_session(remote_connection) -> None:
    # remote_connection 由应用层传入：它已启动，且应用已经按电脑端协议发布
    # 首条 session/status，并真实提供其中声明的 video/vehicle observation。
    # 本模块不重复实现这些发布器。调用方拥有并关闭 remote_connection。
    with channel:
        run_remote_motion(
            remote_connection,
            executor,
            stop_requested=should_stop,
            # 回调可将 CarTelemetry 转成车辆状态观察或运动日志。
            on_car_message=lambda message: print(message),
        )
```

该循环只应在 `remote.access_mode: debug_control` 的赛外受监督配置中运行。
比赛配置必须保持 `observe_only`；传输层会拒绝入站控制。

## 生命周期与降级

- `valid_for_ms` 从树莓派完成接收该消息的单调时间开始计算，不比较两台机器
  互不共享零点的 `issued_timestamp_ns`。
- 死手关闭、命令过期、非法 payload、未知控制模式和循环正常退出均进入柔和
  停车；协议或通信异常也会尝试停车并继续抛出原始异常。
- `drain_messages()` / `run_remote_motion()` 会排空 10 Hz 回传，避免 UART
  有界队列因调用方只发送不接收而溢出。未知前缀保持为 `UnknownCarMessage`，
  不会被误判为命令成功。
- 当前固件资料没有失联看门狗。进程被强杀、树莓派掉电或 UART 物理断开时，
  Python 无法保证停车；只能在架空轮或有物理急停、人员全程监督的环境验证，
  不能把本模块的超时当作固件级失控保护。

真机还需核对实际左右轮正方向、实测轮距、轮速上限、急停锁存/恢复、断 UART
和杀进程行为。上述项目目前均为“未验证”。
