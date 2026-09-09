# `rescue_vision` Python 包

本包提供相机、标定、坐标几何、感知、定位、跟踪、世界模型、任务规则、运动和正式应用
装配。硬件对象由配置层创建并注入领域模块；算法模块不直接创建相机、Hailo、串口或
TCP 连接。

## 正式流程

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.match.yaml")
flow = config.build_match_sequence()
```

`flow` 是 `MatchSequence`，只消费结构化 `PerceptionSnapshot`、陀螺仪航向、编码器
累计距离和安全信号，返回线速度、角速度、夹爪姿态和状态原因。正式入口为：

```bash
rescue-vision-match \
  --config configs/runtime.match.yaml \
  --start-area 2 \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

`--start-area` 只接受 `2` 或 `3`，默认是 `2`（地图右上角、红方）。选择 `3` 时
使用地图左下角、蓝方，并将正式流程所用的初始场地位姿、运输终点和场地坐标刹车
过冲相对中心十字 `(0,0)` 做中心对称；固定红蓝物理地图保持原样。

流程的 d1 视觉纠偏在车辆停稳后收集安全区 K0/K1/K2，使用
`SafeZoneCornerLocalizer` 结合静态红蓝安全区地标拟合场地位姿，并覆盖受限航位；
d1→d2 和 d2→末段随后沿校正位姿用编码器和陀螺仪积分。正式流程不使用跨帧目标记忆、
采集时刻位姿对齐或旧的 20 分编排。

CC 独立流程使用 `configs/runtime.cc.yaml`：

```bash
rescue-vision-match-cc \
  --config configs/runtime.cc.yaml \
  --start-area 2 \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

其启动、命令行、日志和安全区运输复用正式入口；解团采用原 match 的5帧空间均值、
单次角度对准、固定前距接近和固定前推/后退动作，但成团严格要求每块都有两个
100 mm 内邻居，并在缺团 1000 ms 后重启搜索。单块通道搜索和绿/黑/橙抓取由
`MatchCCSequence` 独立实现。

## 末端张爪推送—运输联调

```python
flow = config.build_grab_transport_sequence()
```

该入口按正式流程夹取并运输绿色物块；只有安全区末端推进阶段不再重复闭合夹爪，
而是保持张开把物块推入安全区，随后保持张开退出。

或运行：

```bash
rescue-vision-grab-transport \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

该入口从 `FieldPoint(0,0)`、`+90°` 直接搜索通过单物块门禁的绿色物资，禁用解团，
夹取、视觉纠偏、运输和退出状态复用 `MatchSequence`，仅覆盖末端闭爪推进为张爪推进。

## 1.5 m 定距动作 TUI

```bash
rescue-vision-motion-sequence \
  --config configs/runtime.match.yaml \
  --distance-m 1.5 \
  --supervised-physical-stop-ready \
  --local-preview
```

TUI 输入 `a1`、`a2`；通过 `--distance-m` 配置目标路程（默认 1.5 m）。程序自动计算峰值
速度、加速时间和减速时间，并控制车辆按三角速度曲线前进至控制器下发零速。该入口只使用 UART 和
底盘控制器；`--local-preview` 会额外启动相机和 Hailo Pose 本地识别窗口。不采用 `motion`
的速度、加速度和最小非零速度上限，仅受 STM32 协议可编码范围约束，`q`/`Esc` 会
软刹车退出。它仅用于架空轮或有物理急停、全程监督的赛外动作验证。

## 主要模块

| 模块 | 责任 |
| --- | --- |
| `app/match.py` | 正式流程纯逻辑状态机 |
| `app/match_cc.py` | CC 独立解团、分级搜索和单块抓取状态机 |
| `app/match_runtime.py` | 正式流程硬件装配、停车清理和控制循环 |
| `app/match_observers.py` | 本地预览和 observe-only 观察发布 |
| `app/session_log.py` | 有界日志旁路和标准流恢复 |
| `app/grab_transport.py` | 不解团的末端张爪推送—运输联调 |
| `app/cluster_breakup.py` | 赛外固定解团试验 |
| `app/gripper_width.py` | 独立受监督绿黑多目标/单橙色收拢入口与显示旁路 |
| `app/near_field_grasp.py` | 正式/独立入口共用的近场目标包络、策略选组和走廊规划 |
| `app/gripper_width_sequence.py` | 共用的同组复核、对准、张爪、定距收拢、合爪状态机及准备 worker |
| `app/motion_sequence.py` | 1.5 m 定距速度规划、执行器和 TUI |
| `app/scan_target_memory.py` | 可复用目标短时记忆，尚未接入正式流程 |
| `app/field_target_cluster.py` | 可复用带身份场地聚类，尚未接入正式流程 |
| `config/runtime.py` | 严格 YAML schema、校验和对象装配 |
| `geometry/` | 显式坐标类型、去畸变和地面投影 |
| `localization/` | 安全区/中心十字视觉锚点和编码器/IMU 融合 |
| `mission/` | 比赛规则和交付/安全证据状态机 |
| `motion/` | 差速、夹爪、STM32 协议和 D2 遥测 |
| `perception/gripper_width.py` | 颜色掩码地面投影、左右 `y` 极值、前端 `x` 和目标开口宽度 |

坐标调用链固定为 `RawPixel → UndistortedPixel → GroundPoint ↔ BevPixel`；完整物理
外参用于离地 `RobotPoint3D` 和安全区视觉纠偏。`FieldPoint` 只表示场地全局坐标，
不能由局部 `GroundPoint` 直接替代。

## 资源生命周期

纯逻辑测试不打开硬件。车端控制循环只消费最新帧；推理、显示、录像、日志、图传和
地图发布通过有界旁路运行。正常结束、信号、异常和旁路线程失败都会先执行软刹车，再
停止相机、Hailo、UART、TCP、预览和日志资源。

## 验证

```bash
python -m compileall -q src tests manual_tests
python -m pytest
```

真机、相机、Hailo、固件看门狗、现场地标和性能结论必须在 `manual_tests/` 与目标设备
完成；合成 pytest 不能替代这些验收。
