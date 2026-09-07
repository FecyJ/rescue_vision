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
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

流程的 d1 视觉纠偏在车辆停稳后收集安全区 K0/K1/K2，使用
`SafeZoneCornerLocalizer` 结合静态红蓝安全区地标拟合场地位姿，并覆盖受限航位；
d1→d2 和 d2→末段随后沿校正位姿用编码器和陀螺仪积分。正式流程不使用跨帧目标记忆、
采集时刻位姿对齐或旧的 20 分编排。

## 夹取—运输联调

```python
flow = config.build_grab_transport_sequence()
```

或运行：

```bash
rescue-vision-grab-transport \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

该入口从 `FieldPoint(0,0)`、`+90°` 直接搜索通过单物块门禁的绿色物资，禁用解团，
其余夹取、视觉纠偏、运输和退出状态复用 `MatchSequence`。

## 主要模块

| 模块 | 责任 |
| --- | --- |
| `app/match.py` | 正式流程纯逻辑状态机 |
| `app/match_runtime.py` | 正式流程硬件装配、停车清理和控制循环 |
| `app/match_observers.py` | 本地预览和 observe-only 观察发布 |
| `app/session_log.py` | 有界日志旁路和标准流恢复 |
| `app/grab_transport.py` | 不解团的夹取—运输联调 |
| `app/cluster_breakup.py` | 赛外固定解团试验 |
| `app/gripper_width.py` | 独立每秒按地面物块宽度控制双舵机夹爪的测试入口 |
| `app/scan_target_memory.py` | 可复用目标短时记忆，尚未接入正式流程 |
| `app/field_target_cluster.py` | 可复用带身份场地聚类，尚未接入正式流程 |
| `config/runtime.py` | 严格 YAML schema、校验和对象装配 |
| `geometry/` | 显式坐标类型、去畸变和地面投影 |
| `localization/` | 安全区/中心十字视觉锚点和编码器/IMU 融合 |
| `mission/` | 比赛规则和交付/安全证据状态机 |
| `motion/` | 差速、夹爪、STM32 协议和 D2 遥测 |
| `perception/gripper_width.py` | 颜色掩码地面投影、左右 `y` 极值和目标开口宽度 |

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
