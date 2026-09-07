# `app`：可运行应用装配

正式比赛流程由 `MatchSequence` 提供纯逻辑状态机，由 `match_runtime.py` 负责相机、
Hailo、UART、夹爪、观察发布和资源生命周期。纯逻辑对象不创建硬件资源；硬件旁路只
提交最新观测，不阻塞运动控制循环。

## 常用入口

| 入口 | 配置和用途 |
| --- | --- |
| `rescue-vision-match` | 使用 `configs/runtime.match.yaml` 运行正式流程；支持本地预览、observe-only perception 图传和流程日志 |
| `rescue-vision-grab-transport` | 使用同一配置，从 `(0,0,+90°)` 搜索绿色单物块，不执行解团；用于夹爪和安全区运输联调 |
| `rescue-vision-cluster-breakup` | 赛外固定解团试验，不是正式比赛入口 |
| `rescue-vision-manual-capture` | 受监督手动驾驶、图传和数据采集 |
| `rescue-vision-green-grab` | 无地面标定时的绿色像素居中抓取试验 |

两个正式流程入口都要求显式的物理急停和全程监督确认，直到固件看门狗和急停闭环完成
真车验收。正常退出、信号、相机、UART、网络或线程异常都进入软刹车、停止旁路和资源
释放路径。

## `rescue-vision-match` 使用方法

先确认 `configs/runtime.match.yaml` 已按当前车辆填写，并满足以下装配条件：

- `match.enabled: true`；
- `motion`、`motion.odometry` 和 `motion.gripper` 已启用并完成机械标定；
- `geometry.ground_mapping_enabled: true` 且地面标定与当前相机分辨率匹配；
- `hailo.enabled: true`，模型路径和六类 v3 输出顺序有效；
- `remote.enabled: true` 时使用 `role: server` 与 `access_mode: observe_only`。

在仓库根目录安装后运行：

```bash
python -m pip install -e '.[dev]'

rescue-vision-match \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

参数含义如下：

| 参数 | 作用 |
| --- | --- |
| `--config PATH` | 必填；正式流程 YAML，通常为 `configs/runtime.match.yaml` |
| `--supervised-physical-stop-ready` | 确认物理急停已就绪且操作者全程监督；固件看门狗未完成验收前必须提供 |
| `--local-preview` | 打开本地 OpenCV 预览，显示最新感知帧、状态和动作原因；按 `Q`/`Esc` 退出 |
| `--jpeg-quality N` | 观察图传 JPEG 质量，范围 `1..100`，默认 `80` |
| `--observer-image-interval-seconds S` | observe-only 图传最小发布间隔，必须为正数，默认 `1.0` |
| `--log-dir PATH` | 按时间写入流程日志和 D2 遥测，默认 `logs/`；传入 `--log-dir /dev/null` 不适用，需使用有效目录 |

流程启动后会先执行预检、启动转向/直行和目标搜索。终端状态行包含当前状态、动作原因、
编码器距离、航向、目标轮速、绿色走廊诊断、安全区阶段和 D2 遥测丢弃计数。观察图传、
地图状态和本地预览只读取最新数据，不参与运动决策。

正常结束、`Ctrl+C`、`SIGTERM`、相机/Hailo/UART/网络异常或旁路线程失败都会进入统一
软刹车清理路径。重新运行前应确认车辆已停稳、急停状态已复位，并重新提供显式监督确认。

如果只需联调绿色夹取和安全区运输，使用 `rescue-vision-grab-transport`；该入口复用
同一配置，从 `(0,0,+90°)` 直接搜索绿色单物块，不执行目标团解团。

## `rescue-vision-grab-transport` 使用方法

该入口用于在目标已经摆散时单独联调绿色物资夹取、d1 视觉纠偏、安全区运输和退出。
它仍要求与正式流程相同的 Hailo、地面映射、UART、里程计和夹爪标定；配置检查仍使用
`match.enabled` 以及同一组运动安全限制。

```bash
rescue-vision-grab-transport \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

入口会覆盖本次联调的受限初始位姿：场地 `FieldPoint(0, 0)`、航向 `+90°`，跳过正式
流程的固定启动转向/直行，预检通过后立即进入绿色目标搜索。只有通过单物块净空、绿色
前进走廊、安全区和场地边界门禁的目标才会进入夹取；搜索不到合格目标时保持原地搜索，
不会转入目标团解团。

命令行参数与 `rescue-vision-match` 相同：`--config` 指定正式流程配置，
`--supervised-physical-stop-ready` 确认物理急停和全程监督，`--local-preview` 打开本地
预览，`--jpeg-quality` 和 `--observer-image-interval-seconds` 控制观察图传，
`--log-dir` 指定流程日志和 D2 遥测目录。正常退出、`Ctrl+C`、信号或硬件/旁路异常均
使用同一软刹车和资源清理路径。

## 正式流程行为

`MatchSequence` 以 `GroundPoint`、陀螺仪航向和编码器累计距离完成：

- 启动转向/直行，随后搜索目标团；配置允许时，可靠的单个绿色普通物资可直接抢占。
- 完整目标团按绿色成员和成员数排序；候选的场界、双方安全区和机器人包络门禁失败时
  继续尝试下一完整团。
- 目标停车后采集多帧并取均值；远目标先接近到二次对准距离，再按抓取偏移定距接近。
- 到达 `green_grab_offset_mm` 后保持局部张爪进入停车复核窗口；以车体原点为中心检查
  范围内新鲜且可达的绿块，不要求目标在车前，给 tracker 留出确认时间，发现后重新对准并
  循环纳入，直到 `green_preclose_max_carried_blocks` 上限或没有候选才闭爪；动作
  切换由零速保持和陀螺仪航向保持保护。
- 夹取后沿 d1 直线前进。d1 停稳后以 K0/K1/K2 和静态红蓝安全区角点拟合 `FieldPose2D`
  并覆盖当前航位；d2 与末段继续沿校正位姿积分，完成停稳、释放和退出。

视觉纠偏只使用 d1 停稳期间的当前安全区观测。正式流程当前不接入跨帧视野外目标记忆、
采集时刻位姿对齐、记忆目标接管或旧的 20 分编排。

## 纯逻辑装配

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.match.yaml")
flow = config.build_match_sequence()
```

联调入口的类型和装配方法为 `GrabTransportSequence` 与
`config.build_grab_transport_sequence()`。它覆盖初始场地位置和航向，仍复用正式流程
配置、跟踪器、夹爪标定和安全区视觉纠偏。

## 可复用但未接入的模块

`scan_target_memory.py` 保存带场地点和老化时间的可靠目标，可在未来正式流程明确采集
时刻位姿策略后接入；当前不参与任何运动决策。`field_target_cluster.py` 提供带身份的
`FieldPoint` 连通聚类，也只保留为可复用逻辑。两者均有独立 pytest，不能被解释为已接入
正式比赛能力。

## 观察与日志

`match_observers.py` 提供不修改原帧的本地叠加和 observe-only 远程发布；
`session_log.py` 把标准流 tee 到按时间命名的日志文件。D2 到安全区末段的
`D2TelemetryLogger` 逐帧记录编码器/IMU 原始数据、时间差、阶段和丢弃计数，旁路队列
拥塞时不阻塞运动循环。

正式流程设计、状态顺序和视觉纠偏验收见 [`docs/正式流程设计.md`](../../../docs/正式流程设计.md)。
