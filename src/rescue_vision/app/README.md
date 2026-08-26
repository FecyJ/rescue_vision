# `app`：可运行应用装配

本包只放跨相机、通信和运动模块的实际运行入口。当前实现
`rescue-vision-manual-capture`，用于赛外、人员全程监督的低速车载运动采集，
也支持单片机未上电时的仅相机远程调试；另有
`rescue-vision-cluster-breakup` 用于固定出发姿态下的受监督解团试验，以及
`rescue-vision-simulation-20-point` 用于四个绿色普通物资的受限 20 分模拟赛初版。
后者已实现软件行为编排和真实旁路装配，但 STM32 看门狗、真车参数、真实接触/交付
证据和现场验收仍未完成，不能当作正式比赛程序。

## 常用入口

| 入口 | 输入与生命周期 | 输出或语义 |
| --- | --- | --- |
| `rescue-vision-manual-capture` | `runtime.yaml`、车端输出根目录和显式监督确认 | 完整装配 TCP/UART/相机/运动/夹爪/记录；推荐生产入口 |
| `rescue-vision-cluster-breakup` | `runtime.yaml` 和显式监督确认 | 定距越障、目标团搜索/居中、闭爪定距冲散、原地全开、张爪退出、停车合爪、闭爪退离并扫描绿色；找到绿色后停车 |
| `rescue-vision-simulation-20-point` | `runtime.simulation-20min.yaml` 和显式监督确认 | 复用现有解团，跟踪/世界模型/规则门禁、受限预推导航、单绿色推送、接近时 transport 局部打开、接触确认后闭合、交付验证、退离、四次完成停车，并向 observe_only 观察端发布最新 `map/state` 位姿 |
| `ClusterBreakupSequence(...).step()` | `ClusterBreakupRuntimeConfig`、`motion.gripper.full_travel_time_s`、同一单调时间轴、编码器累计路程和最新 `PerceptionSnapshot` | 纯逻辑流程决策；不创建相机、Hailo、UART 或电机；冲散后全开、张爪退出、停车合爪，再进入闭爪退离 |
| `EncoderTravelTracker.submit()` | 带有效双编码器的 `OdometryImu`；连续 `sample_overrun` 预算来自 `motion.odometry.max_consecutive_overrun_samples`，`null` 关闭中止仅保留计数 | 使用运行配置机械标定产生机器人中心累计有符号路程；连续异常超出配置预算时抛错 |
| `OdometryImuFusion.submit_odometry()` | 同一 UART 消费链路中的 `OdometryImu` | 解团临时配置只输出编码器+IMU 连续 `FieldPose2D`；不调用 `submit_visual()` |
| `OdometryFusionPump` | `OdometryImu` 有界队列和 `OdometryImuFusion` | 顺序处理融合，不阻塞运动刷新线程；队列满或 worker 异常进入停车路径 |
| `CameraPerceptionPump` | `FrameSource`、去畸变函数和目标感知旁路 | 在独立线程完成取帧/去畸变/推理提交，不占用运动刷新线程 |
| `RemotePerceptionPublisher` / `RemoteLocalizationPublisher` / `RemotePerceptionTransport` | `observe_only` TCP 连接、已渲染 perception 帧和编码器+IMU估计 | 异步接受一个观察端；独立旁路发送 perception JPEG 与 `map/state` 位姿 JSON，不发送原图/BEV |
| `send_video_frame()` | 已准备好的 `CameraFrame`、`CameraPipeline` 和已启动远程连接 | 在调用线程编码并提交带坐标/标定属性的 JPEG；实时入口应放在独立旁路调用 |
| `build_camera_pipeline()` | 已加载的 `AppConfig` | 创建但不启动配置选择的真机源、可选去畸变和地面映射 |
| `build_session_status()` | 同一 `AppConfig`、服务实例 ID、视频 FPS 和可用模式 | 创建声明运动、夹爪、视频模式、车辆和采集能力的会话状态 |
| `run_manual_capture_session()` | 已启动连接、可选运动/夹爪执行器、采集会话、相机管线和可选 perception/BEV 旁路 | 运行单个完整车辆或仅相机 TCP 会话；断线/故障时清理当前记录 |
| `BevFrameRenderer` | 已加载且包含 BEV 配置的 `GroundProjector`；显式 `start/stop` | 有界丢旧保新的后台 BEV 生成旁路；输出保留源帧号和采集时间 |
| `LatestCenterCrossLocalization` | 场地检测器、中心十字定位器、有效像素掩码和可选 `OdometryImuFusion`；显式 `start/stop` | 有界视觉旁路；完整车辆输出连续融合位姿，仅相机模式保留新鲜单帧视觉语义 |
| `Simulation20PointSequence` | `Simulation20PointRuntimeConfig`、跟踪器、世界模型、mission 和现有 `ClusterBreakupSequence` | 只读取完成的最新快照并返回轻量线速度/角速度/夹爪意图；任何旁路等待由调用方负责 |

`run_manual_capture_session()` 不创建或打开硬件资源。调用方传入
`RemoteMotionExecutor`，并在 `motion.gripper.enabled=true` 时传入共享同一个
`MotionController` 的 `RemoteGripperExecutor`；关闭夹爪能力时传入 `None`。
仅相机会话把两个执行器都传为 `None`，并使用
`build_session_status(camera_only=True)`；两者不得与会话 capability 矛盾。
正式装配由下文 CLI 从运行配置完成，普通调用方不应另写一套设备、限值、
机械端点或协议参数。

## 20 分模拟赛初版

先从专用配置装配流程；这一步只创建纯逻辑对象，不打开相机、UART、Hailo 或
TCP。配置中的 `simulation_20_point` 是动作和安全裕量的唯一权威：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.simulation-20min.yaml")
flow = config.build_simulation_20_point_sequence()
```

启动前由真实车端同步、相机旁路和观察端装配出门禁证据。所有字段都必须来自实际
状态，不得用配置默认值冒充硬件状态；下面的布尔值仅表示调用方已经完成对应
检查：

```python
from time import monotonic_ns
from rescue_vision.app import Simulation20PointState, SimulationPreflight

ready = flow.preflight(
    monotonic_ns(),
    SimulationPreflight(
        telemetry_fresh=True,
        watchdog_armed=True,
        emergency_stop_clear=True,
        zero_speed_command_accepted=True,
        camera_observation_fresh=True,
        observe_only_remote=True,
    ),
)
if ready.state is Simulation20PointState.TERMINAL_STOP:
    raise RuntimeError(ready.reason)
flow.start(monotonic_ns())
```

`flow.start()` 记录一轮唯一启动时刻并进入现有解团流程。后续每个控制周期把相机
推理线程已经完成的 `PerceptionSnapshot`、编码器/IMU 融合器的
`FusedPoseEstimate`、UART 消费回调维护的累计中心路程（单位 m）和旁路健康快照
传入 `step()`：

```python
decision = flow.step(
    monotonic_ns(),
    perception=latest_perception_snapshot,
    pose=latest_fused_pose_estimate,
    cumulative_distance_m=latest_encoder_distance_m,
    health=latest_side_path_health,
)
```

返回的 `SimulationDecision.linear_velocity_m_s`、
`angular_velocity_rad_s` 是机器人轮轴中点参考的差速 twist，角速度左转为正；
夹爪姿态为 `open/transport/closed`。20 分流程在 `APPROACH_GREEN` 和尚未确认的
`ENGAGE_GREEN` 使用 `transport` 局部打开姿态，成功确认接触后切换为 `closed`。
调用方把它交给同一 `MotionController`，不能在 `TERMINAL_STOP` 或 `FINISH_STOP`
覆盖为非零速度。真实控制循环只在姿态变化时下发角度：

```python
from rescue_vision.app import GripperPosture

gripper = config.motion.gripper.build_calibration()
if gripper is None or gripper.transport_angles_deg is None:
    raise RuntimeError("20-point simulation requires transport gripper angles.")

# `last_posture` 由控制循环跨周期保存；`decision` 来自前一个 step() 示例。
if decision.gripper_posture is not last_posture:
    if decision.gripper_posture is GripperPosture.OPEN:
        angles = (gripper.open_left_angle_deg, gripper.open_right_angle_deg)
    elif decision.gripper_posture is GripperPosture.TRANSPORT:
        angles = gripper.transport_angles_deg
    else:
        angles = (gripper.closed_left_angle_deg, gripper.closed_right_angle_deg)
    motion_controller.set_gripper_angles(*angles)
    last_posture = decision.gripper_posture
```

随后按状态发送底盘意图；不能在 `TERMINAL_STOP` 或 `FINISH_STOP` 覆盖为非零速度：

```python
if decision.state in {
    Simulation20PointState.TERMINAL_STOP,
    Simulation20PointState.FINISH_STOP,
}:
    motion_controller.soft_brake()
else:
    motion_controller.drive_wheel_limited(
        decision.linear_velocity_m_s,
        decision.angular_velocity_rad_s,
    )
```

流程的顺序是：现有解团 → 轨迹重置/稳定确认 → 最小航向扫描 → 易搬运绿色筛选 →
预推点和保守走廊 → 对准/几何单目标接触 → 单目标推送 → 己方物资区内缩区域的
连续完全进入证据 → 退离确认。完成四个不同的有效 `delivery_id` 后进入
`FINISH_STOP`，`valid_green_deliveries=4`、`score_points=20`。目标丢失、第二目标
进入接触走廊、定位/视觉过期、证据不足和旁路故障分别进入安全保持或终止停车；
接近中的目标若变为危险会取消当前方案，远场危险目标和解团时的短暂接触不触发安全保持。
安全保持不会自动恢复，必须由外部新鲜证据调用
`resume_after_safety_hold()`。

蓝色或 `unknown` 不会成为绿色候选。常规导航只在它们实际阻挡预推、接近或推动
走廊时取消当前绿色方案并重新评估；已经建立单目标持续推动后若走廊被阻挡则保持
零速，不带着物块自动绕行。解团阶段沿用固定直线动作，不为蓝色物块规划替代方向。

真实车入口负责资源生命周期：UART 通道打开后，相机/Hailo 预热在独立启动线程中
与运动同步并行；预热等待期间主线程持续排空 UART，感知旁路就绪后才进入相机门禁。
融合旁路和观察服务也在主循环外运行。退出顺序为软制动、停止相机/感知、停止观察
服务、停止融合旁路、关闭 UART。相机等待、Hailo 推理、JPEG、TCP 和写盘不在
`step()` 内执行。

20 分模拟赛入口会把同一控制周期取得的编码器/IMU `FusedPoseEstimate` 提交给远程
`map/state` 发布器；发布器仍按 200 ms 节流，只发布最新定位，不复制或另建定位源。

命令行入口：

```bash
rescue-vision-simulation-20-point \
  --config configs/runtime.simulation-20min.yaml \
  --supervised-physical-stop-ready
```

未完成 STM32 看门狗真车闭环前，命令仍要求物理急停和全程监督；观察端只能是
`observe_only`，不能下发运动、夹爪或状态跳转命令。

## 手动采集入口

先从 `configs/runtime.yaml` 加载唯一运行配置。必须启用相机、UART、motion 和
remote server，并将 `remote.access_mode` 明确设为 `debug_control`：

```bash
rescue-vision-manual-capture \
  --config configs/runtime.yaml \
  --output-root data/rescue-targets/manual \
  --supervised-physical-stop-ready \
  --enable-localization \
  --video-fps 2
```

入口创建但不复制相机参数、运动限值或协议规则。连接后发送会话、视频、动态地图状态、车辆
和采集状态，并同时接收 `control/debug/motion`、
`control/debug/gripper`、`control/debug/capture` 与不执行动作的
`control/video/mode`。运动命令继续由
`RemoteMotionExecutor` 校验死手、有效期和限速；夹爪命令由
`RemoteGripperExecutor` 校验有效期和布尔扳机状态，再按配置机械端点与
全行程时间渐进下发；采集命令可独立
开始/停止统一 recording 会话。电脑端通过 `video_modes` 选择 `raw`、
`perception` 或 `bev`；perception 由 `PerceptionFrameRenderer` 在最新帧后台
旁路运行 `TargetPoseDetector`，BEV 只在当前地面映射包含 BEV 配置时由独立
最新帧旁路生成。两者只发布带实际模式和坐标元数据的 JPEG，不阻塞运动安全循环。
启用 `--enable-localization` 时，远程 BEV 优先使用同一定位旁路完成的最新帧：
绿色轴和 `CROSS` 标出中心十字，半透明 `SAFE red/blue` 标出安全区，青色线段
标出已关联终端；定位结果尚未完成时临时回退为纯 BEV，过期检测则显式显示
`FIELD STALE`。该叠加只用于人眼诊断，全局位姿仍以 `observation/map/state`
为机器可读权威。
相机去畸变、录像提交、JPEG 和动态状态发布之间都会再次检查远程命令期限并刷新
轮速；多个旁路耗时不会再累计成一次超过 100 ms 的轮速跃迁。任一单项操作
若独占控制线程超过 100 ms，仍先停车并退出，而不是增大阈值掩盖实时性故障。
未录制时只在下一图传期限到达后处理最新相机帧，不再按相机 20 FPS 逐帧
去畸变；默认图传为 2 FPS，不需要实时画面时可显式传 `--video-fps 1`。开始
录像后恢复逐相机帧处理，保证记录完整性。
手动驾驶默认不启动 CPU 较重的场地特征/中心十字定位旁路；需要地图定位时
显式添加 `--enable-localization`，或启用 `localization.fusion.enabled`。
只要 `world.static_map.regions` 非空，会话就声明并以 200 ms 周期发布轻量
`observation/map/state` JSON；静态底图、区域和 FieldPoint 到显示像素的映射
由电脑端固化。若同时启用可用地面映射、`field_features` 和 `localization`，
状态携带新鲜机器人全局位姿。完整车辆配置启用 `localization.fusion` 时，唯一 UART 消费回调把
100 Hz `OdometryImu` 同时送给车辆状态、运动日志和融合器；中心十字后台按相机
采集时刻取先验、提交延迟纠偏，离开十字可见区后继续发布新鲜连续位姿。遥测
超期或连续性中断时地图立即清除机器人位置。仅相机模式不创建融合器，也不会用
配置起点冒充连续定位。
手动会话的 `motion.jsonl` 逐条
保存运动/夹爪执行结果、编码器/IMU、系统状态、命令回复和停车原因；
这些事件与图像帧统一使用树莓派应用单调时间。
等待首个客户端及断线重连期间，入口仍以有界周期排空 STM32 主动遥测，避免
未连接时填满 UART 接收队列；等待达到 `--accept-timeout-seconds` 后会先
停车并正常退出，不创建空 recording。

### 仅相机远程调试

只给树莓派和相机上电、STM32/UART/电机均不可用时，使用：

```bash
rescue-vision-manual-capture \
  --config configs/runtime.yaml \
  --output-root /data/rescue-targets/camera-only \
  --camera-only \
  --video-fps 10
```

此模式不打开 UART，也不要求 `uart.enabled`、`motion.enabled` 或
`--supervised-physical-stop-ready`。会话仍提供 raw/perception/BEV 图传和采集
控制，但明确声明 `motion_control=false`、`gripper_control=false`。车端继续按
实际循环通常每 100 ms 发布车辆状态，并把协议最大发布周期声明为 500 ms，
使客户端陈旧阈值为 1500 ms。状态固定报告 `control_ready=false`、
`uart_connected=false`、`safety_mode=unavailable` 和 `stop_reason=uart_fault`，
避免现有客户端因车辆状态陈旧而反复重连，同时不会伪装单片机在线。
若旧客户端没有按 capability 停止周期运动/夹爪心跳，车端会严格解析并安全
丢弃这些合法消息，不会因此关闭仅相机会话；消息不会写 UART 或更新“已应用”
命令 ID。畸形 payload 和未知 control 仍按协议错误处理。

视频、车辆状态和地图是三个独立的最新值 topic，因此
`remote.observation_queue_capacity` 必须至少为 `3`；入口会在启动时拒绝更小
容量，避免图像流把车辆心跳挤出队列。

该模式产生的录像使用已有 `recording_kind=camera`，不创建 `motion.jsonl`；切回
完整车辆调试时不要带 `--camera-only`，并继续满足物理急停和全程监督要求。

录像写盘队列满、写盘失败、相机异常、UART 异常、远程断线、非法控制或应用
退出都会离开统一运动循环，并在 UART 尚可写时先发送柔和制动。断电、
`SIGKILL` 和 UART 物理断开仍只能由固件看门狗停车。
完整车辆模式中的 UART 异常同样会退出；仅相机模式根本不打开 UART，因此不受
未上电单片机影响。

## 目标团解团试验

该入口只验证模拟赛的前置动作，不是完整 20 分比赛程序。先在
`configs/runtime.yaml` 启用并实测 `motion`、`motion.odometry`、
`motion.gripper`、`motion.cluster_breakup`、地面映射和 Hailo 模型，然后运行：

```bash
rescue-vision-cluster-breakup \
  --config configs/runtime.simulation-20min.yaml \
  --supervised-physical-stop-ready
```

流程按配置执行：等待有效编码器 → 固定距离直行越过减速带 → 按
`search_angular_velocity_rad_s` 的符号持续转向搜索至少
`cluster_min_detections` 个模型观测 → 用观测框联合中心做比例居中 → 低速接近。
最近有效 K0 地面点进入 `gripper_open_distance_mm` 后，夹爪保持闭合，车辆以
`breakup_speed_m_s` 推进 `breakup_distance_m`；到达后原地停车并切到已标定张开端点，
保持 `motion.gripper.full_travel_time_s` 完全打开，再以
`retreat_speed_m_s` 张爪倒退 `gripper_open_retreat_distance_m`。到达后停车切回闭合
端点，保持同一全行程时间，最后以 `retreat_speed_m_s` 闭爪倒退
`retreat_distance_m`。最后进入 `SCAN_GREEN`，按
`scan_green_angular_velocity_rad_s` 的符号原地扫描，连续看到配置帧数的
`green_supply` 后停车退出。当前不会继续接近或交付绿色目标；
搜索/扫描角速度均为带符号值（左转为正、右转为负），方向由符号唯一决定，
不在入口中固定为左转。解团阶段不额外插入 `BREAKUP_SAFETY_CHECK` 或危险/未知类别门禁；
在非转运阶段若视野只有 `unknown`/`blue_danger`，流程继续原地搜索，不直接锁定零速。

`configs/runtime.simulation-20min.yaml` 是从当前车端 `runtime.yaml` 复制的临时配置：
保留目标 perception 所需的 Hailo 和地面映射，关闭 `localization.enabled` 及
`perception.field_features.enabled`，但开启 `localization.fusion.enabled`。
解团入口把每条有效 `OdometryImu` 同时送入固定距离里程计和
`OdometryImuFusion`，当前只消费编码器+IMU预测；单次双编码器有效的
`sample_overrun` 会等待下一帧并做短时 IMU 插值，连续异常仍触发保守降级；初始场地位姿来自配置的
`localization.fusion.initial_pose`，连续估计会在 `progress` 中打印。尚未接入
中心十字、安全区或其他视觉位姿纠偏；后续只需在独立视觉旁路中调用
`fusion.submit_visual()`。

取帧和全尺寸去畸变在独立输入旁路运行，Hailo 推理再使用单槽最新帧后台旁路；
运动循环只刷新轮速、排空 UART 并消费最新结构化观测。观测过期、
目标团丢失、编码器无效/跳变、连续里程计异常超出
`motion.odometry.max_consecutive_overrun_samples` 预算、阶段超时、
相机/Hailo/UART 异常以及退出都会走停车路径。所有距离、速度、角度、确认帧数和超时均来自
`motion.cluster_breakup`，不能在入口中另写一套参数。
入口每秒输出机器人中心/左右轮累计路程、原始编码器计数及目标/已下发轮速。
同一行还输出 STM32 的 `motor_output`、`watchdog`、`estop`、`stop_reason` 和
最近运动命令年龄；任何命令回复不是 `accepted` 或急停已经锁存都会立即停车。
请求运动 750 ms 后新鲜状态仍报告电机输出关闭，也会直接报出电控未使能，
不再等到定距阶段超时。

当 `remote.enabled=true` 时，入口要求 `remote.role: server` 和
`remote.access_mode: observe_only`，并在独立线程监听一个观察客户端；没有客户端
不会等待或阻塞模拟赛流程。客户端接入后先收到会话状态，随后只在 perception
后台产生新识别结果时收到带框、颜色掩码、K0、置信度和质量信息的
`observation/video/frame` JPEG，默认由 `--observer-image-interval-seconds 1`
限频；同时按会话声明的 `map_state` 能力收到约 200 ms 一次的
`observation/map/state` 编码器+IMU位姿 JSON。JPEG/JSON 编码、队列提交和 TCP
写入均不在运动循环执行；观察连接/发布旁路发生故障时主循环会尽快进入已有停车路径。
定距出发时若左右轮程明显反向，会立即停车并报告前进编码器符号不符合冻结协议；
不得在树莓派端取绝对值或增加符号补偿掩盖固件方向错误。
20 分入口的终端日志只在状态变化或约每秒输出一次，并附带编码器累计路程、左右轮
诊断、目标/已下发轮速和 STM32 电机输出/停止原因；请求非零轮速后约 750 ms 仍
报告电机输出关闭时会立即进入停车异常，不再等待出发阶段超时。

## 安全边界

当前固件协议没有可验证的看门狗和急停锁存状态。命令因此默认拒绝启动；只有
物理急停已就绪且操作员会全程监督时，才可显式传入
`--supervised-physical-stop-ready`。此时车辆状态如实发布
`safety_mode=supervised_physical_stop`、`watchdog_armed=false` 和
`control_ready=true`，使合规客户端能在持续警告下发送低速命令，而不会把
临时监督条件伪装成固件保护。在固件闭环和真机验收前不得用于比赛运行。
重新连接不会复用旧 TCP 会话或旧死手使能，操作者必须重新发送显式使能命令。
夹爪不随运动死手切换，但任一按下状态必须在有效期内持续刷新。松开扳机要
显式发送两个状态均为 `false` 的 v2 命令；命令超时、断线和停车会停止继续改变
角度，不会自动开爪或闭爪，也不会在重连后重放旧命令。只有
`motion.gripper` 的机械开合端点和全行程时间已由当前车辆实测并启用时，会话
才声明 `gripper_control=true`。车辆状态中的舵机角度来自 STM32
`CarState`，不代表有物理位置传感器或已经完成抓取。

解团试验仍不是比赛模式：当前固件看门狗未经真车验收，所以必须保留物理急停和
全程监督。当前流程在冲散距离结束后原地全开，以张爪退出
`gripper_open_retreat_distance_m`，停车合爪并等待全行程，再执行闭爪
`retreat_distance_m`；仍需现场确认机械结构不会形成抓取或承载。张爪退出和闭爪
等待均由状态机控制，不能在退出段中提前合爪。
