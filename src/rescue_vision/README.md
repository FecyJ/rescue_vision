# `rescue_vision` Python 包

本目录包含可导入的运行代码。公共对象从所属子包导入，不在顶层 `rescue_vision` 重导出；调用处应能直接看出相机、几何、感知或数据依赖来自哪个领域。

## 1. 加载运行配置和共享资产

仓库现在包含受限四绿色物资 20 分模拟赛的初版应用入口；赛外手动采集和固定
解团试验仍是独立入口。配置只在启动时加载一次，具体相机、标定、模型、阈值和
模拟赛动作参数都来自本机 YAML。正式比赛能力、真车看门狗和真实接触/交付证据
仍未验收。

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
backend = config.hailo.build_backend()

if geometry is None:
    raise RuntimeError("任务感知必须在 runtime.yaml 中启用内参")
if backend is None:
    raise RuntimeError("任务感知必须在 runtime.yaml 中启用 Hailo")
```

## 2. 装配 20 分模拟赛初版流程

生产车端使用专用临时配置；它明确要求 `simulation_20_point.enabled=true`、
编码器/IMU 融合、目标团解团、Hailo、地面映射和 `observe_only` 观察端。装配只
创建纯逻辑编排对象，不打开相机、UART 或网络：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.simulation-20min.yaml")
flow = config.build_simulation_20_point_sequence()
```

资源由车端 CLI 统一创建和关闭；普通调用方不要复制运动限值、解团距离、场地区域
或交付阈值。启动前把实际 STM32 状态、首个相机观测和 `observe_only` 能力组成
`SimulationPreflight`，确认失败会锁存零运动终止：

```python
from rescue_vision.app import SimulationPreflight
from time import monotonic_ns

preflight = flow.preflight(
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
if preflight.state.value == "terminal_stop":
    raise RuntimeError(preflight.reason)
flow.start(monotonic_ns())
```

每个运动周期只传入旁路已经完成的最新 `PerceptionSnapshot`、
`FusedPoseEstimate` 和编码器累计路程；三者分别由相机/Hailo、编码器/IMU 融合器
和 UART 消费回调产生。返回值是轻量控制意图，必须由 `MotionController` 提交，
`TERMINAL_STOP`/`FINISH_STOP` 只允许零速：

```python
decision = flow.step(
    monotonic_ns(),
    perception=latest_perception_snapshot,
    pose=latest_fused_pose_estimate,
    cumulative_distance_m=latest_encoder_distance_m,
    health=latest_side_path_health,
)
if decision.state.value in {"terminal_stop", "finish_stop"}:
    motion_controller.soft_brake()
else:
    motion_controller.drive_wheel_limited(
        decision.linear_velocity_m_s,
        decision.angular_velocity_rad_s,
    )
```

流程会先调用现有 `ClusterBreakupSequence`，再重置目标轨迹，按最小航向覆盖筛选
单个绿色普通物资，执行旋转—直行—旋转、近场对准、几何单目标接触、保守推送、
完全进入己方物资区验证和退离；第四个唯一 `delivery_id` 被规则状态机接受后
进入 `FINISH_STOP`，展示值为 `4 × 5 = 20`。几何接触证据不能替代真实接触开关，
因此该入口仍受真车门禁和现场验收限制。

车端实际命令：

```bash
rescue-vision-simulation-20-point \
  --config configs/runtime.simulation-20min.yaml \
  --supervised-physical-stop-ready
```

该命令会在后台启动最新帧相机/推理、编码器融合和可选观察图传；旁路异常会传播
到统一停车路径。未完成 STM32 看门狗闭环前不得把 `--supervised-physical-stop-ready`
视为比赛安全证明。

## 3. 创建配置选择的相机源

以下片段承接前文 `config`，只创建对象，尚未打开相机：

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource

# 领域算法依赖统一帧源；只有这里根据配置选择硬件实现。
source_class = (
    Picamera2Source
    if config.camera.backend == "picamera2"
    else RpicamSource
)
source = source_class(
    image_size=config.camera.image_size,
    fps=config.camera.fps,
    lens_position=config.camera.lens_position,
)
```

## 4. 创建感知和纯逻辑对象

以下片段继续使用前文的 `config`、`geometry` 和 `backend`：

```python
from rescue_vision.perception import TargetPoseDetector

detector = TargetPoseDetector(
    backend,
    detection_threshold=config.perception.detection_threshold,
    k0_threshold=config.perception.k0_threshold,
    color_classifier=config.perception.color_classifier,
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=geometry.ground_projector,
    center_cross_refinement=config.perception.center_cross_refinement,
    safe_zone_color=config.perception.safe_zone_color,
)
tracker = config.tracking.build_tracker()
world_model = config.world.build_model()
mission = config.mission.build_state_machine()
```

`start_timestamp_ns` 应来自应用统一单调时钟。开始一轮时单独调用：

```python
from time import monotonic_ns

mission.start(monotonic_ns())
```

## 5. 运行一帧感知到世界模型的主链路

以下片段复用前文全部对象。相机与 Hailo 后端由上下文管理器释放：

```python
from time import monotonic_ns

with source, detector:
    while True:
        raw_frame = source.read(timeout=1.0)
        undistorted_bgr = geometry.camera_model.undistort_image(
            raw_frame.image_bgr
        )
        detection_result = detector.detect_realtime(
            raw_frame,
            undistorted_bgr,
        )
        tracks = tracker.update(
            raw_frame.timestamp_ns,
            detection_result.observations,
        )
        snapshot = world_model.update(
            timestamp_ns=monotonic_ns(),
            visual_timestamp_ns=raw_frame.timestamp_ns,
            tracks=tracks,
            robot_field_point=None,
            # v3 场地结果由同一感知快照产生；定位门禁满足时可提供 FieldPoint。
            # 中心十字没有唯一解时传 None，世界模型保留明确不确定性。
            # v3 目标 ground_point 是机器人系底面中心；不在这里伪造全局坐标。
        )

        # 20 分初版由 Simulation20PointSequence 统一生成
        # TransportStatus/DeliveryEvidence，并把 mission 的停车决定传给运动入口；
        # 其它正式比赛动作仍不能在这里自行扩展规则。
```

`ground_mapping_enabled: false` 时仍可运行图像检测，但 `observation.ground_point` 为 `None`。相机、Hailo 和窗口都必须通过上下文管理或 `try/finally` 释放。

## 子包与主要入口

| 子包 | 常用入口 | 对接责任 |
| --- | --- | --- |
| [`app`](app/README.md) | `run_manual_capture_session()`、`ClusterBreakupSequence`、`Simulation20PointSequence`、`RemotePerceptionTransport`、`RemoteLocalizationPublisher`、`OdometryImuFusion`、三个 `rescue-vision-*` 入口 | 赛外受监督手动驾驶/采集、固定流程解团试验、受限四绿色物资 20 分流程、编码器+IMU航位推算和异步 observe_only perception/位姿图传 |
| [`config`](config/README.md) | `load_runtime_config()`、`GripperRuntimeConfig`、`AppConfig.build_geometry()`、`HailoConfig.build_backend()` | 启动时严格加载、机械标定和装配 |
| [`camera`](camera/README.md) | `FrameSource`、`CameraFrame`、`Picamera2Source`、`RecordingSource` | 产生带时间和序号的最新帧 |
| [`communication`](communication/README.md) | `UartFrameChannel`、`RemoteMessageConnection`、`VideoModeCommand`、`VideoFrameAttributes`、`MapStateObservation`、`RemoteSessionStatus` | COBS UART、直接 TCP 远程消息、可选择 raw/perception/BEV 图传（可声明 perception-only）及轻量 FieldPoint 动态状态 schema |
| [`motion`](motion/README.md) | `MotionController`、`MotionLimits`、`GripperCalibration`、`RemoteMotionExecutor`、`RemoteGripperExecutor`、`run_remote_motion` | 差速运动、持续夹爪双舵机、Rescue Car 协议和远程调试执行 |
| [`geometry`](geometry/README.md) | `CameraModel`、`GroundProjector`、显式坐标类型（含 `MapPixel`） | 去畸变及像素/地面/BEV 转换 |
| [`perception`](perception/README.md) | `TargetPoseDetector`、`PerceptionFrameRenderer`、`PerceptionSnapshot`、`FieldFeatureDetectionResult` | v3 六类同帧分流、任务目标和中心十字/安全区结构化与可视化旁路 |
| [`localization`](localization/README.md) | `CenterCrossLocalizer`、`StaticFieldLandmarkTracker`、`SafeZoneCornerLocalizer`、`ImuFrameCalibration`、`OdometryImuFusion` | 中心十字/安全区角点绝对位姿与地图软门控的纯几何消费层，以及编码器/IMU 连续融合 |
| [`tracking`](tracking/README.md) | `MultiTargetTracker`、`TrackedTarget`、`TrackStatus` | 时间关联、遮挡和轨迹生命周期 |
| [`world`](world/README.md) | `StaticFieldMap`、`WorldModel`、`WorldSnapshot`、`HazardState` | 固定物理地图、任务区域派生、动态目标、对手占据和不确定性 |
| [`mission`](mission/README.md) | `MissionStateMachine`、`replay_mission()`、`MissionDecision` | 规则、安全降级和抽象动作 |
| [`calibration`](calibration/README.md) | 五个 `python -m` 标定命令 | 采集棋盘/ChArUco、生成内参与多图地面映射产物 |
| [`data`](data/README.md) | `inspect_recording()`、`build_dataset_records()`、`split_records()` | 采集验收、清单和防泄漏划分 |
| [`evaluation`](evaluation/README.md) | `observations_to_evaluation_records()`、`evaluate_records()` | 目标匹配适配和离线指标 |


## 对接不变量

- `CameraFrame.image_bgr` 是相机原始帧；任务模型消费 `CameraModel` 产生的全尺寸去畸变图。
- 原始像素使用 `RawPixel`，去畸变像素使用 `UndistortedPixel`，只有后者能交给 `GroundProjector`。
- 感知算法只产生 `TargetObservation`、`TargetGroundGeometry` 或 `FieldFeatureDetectionResult`，不持有跟踪、定位、世界模型或规则状态。
- 实时循环只处理最新帧；录制、显示、日志和通信使用有界旁路。
- 当前正式任务目标模型、场地特征模型推理、完整区域/对手感知、真实接触与交付证据、规划和正式比赛应用尚未完成。传统 OpenCV 场地特征检测与局部场界估计已删除，只保留模型无关的观测契约和定位消费层；旧实拍回归结果不再作为当前策略的证据。`app` 的远程驾驶/图传只用于赛外受监督采集，不能替代固件失联看门狗。
