# `rescue_vision` Python 包

本目录包含可导入的运行代码。公共对象从所属子包导入，不在顶层 `rescue_vision` 重导出；调用处应能直接看出相机、几何、感知或数据依赖来自哪个领域。

## 1. 加载运行配置和共享资产

仓库尚未实现比赛应用入口；赛外手动采集已有独立入口。下面是后续比赛主循环
应采用的分段装配方式。配置只在
启动时加载一次，具体相机、标定、模型和阈值都来自本机
`configs/runtime.yaml`：

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

## 2. 创建配置选择的相机源

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

## 3. 创建感知和纯逻辑对象

以下片段继续使用前文的 `config`、`geometry` 和 `backend`：

```python
from rescue_vision.perception import TargetPoseDetector

detector = TargetPoseDetector(
    backend,
    class_mapping=config.hailo.model_class_mapping(),
    detection_threshold=config.perception.detection_threshold,
    k0_threshold=config.perception.k0_threshold,
    color_classifier=config.perception.color_classifier,
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=geometry.ground_projector,
)
ground_geometry_estimator = (
    config.perception.build_target_ground_geometry_estimator(
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=geometry.ground_projector,
    )
)
field_detector = config.perception.build_field_feature_detector(
    static_map=config.world.static_map,
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=geometry.ground_projector,
)
field_boundary_estimator = config.perception.build_field_boundary_estimator(
    static_map=config.world.static_map,
    ground_projector=geometry.ground_projector,
)
center_cross_localizer = config.build_center_cross_localizer(
    ground_projector=geometry.ground_projector,
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

## 4. 运行一帧感知到世界模型的主链路

以下片段复用前文全部对象。相机与 Hailo 后端由上下文管理器释放：

```python
from time import monotonic_ns

with source, detector:
    while True:
        raw_frame = source.read(timeout=1.0)
        undistorted_bgr = geometry.camera_model.undistort_image(
            raw_frame.image_bgr
        )
        field_realtime_result = (
            field_detector.detect_realtime(
                raw_frame,
                undistorted_bgr,
                valid_mask=geometry.camera_model.valid_mask,
            )
            if field_detector is not None
            else None
        )
        field_result = (
            field_realtime_result.result
            if field_realtime_result is not None
            else None
        )
        field_mask = (
            field_boundary_estimator.update(
                field_result,
                valid_mask=geometry.camera_model.valid_mask,
            )
            if field_boundary_estimator is not None and field_result is not None
            else None
        )
        detection_result = detector.detect_realtime(
            raw_frame,
            undistorted_bgr,
            field_mask=field_mask,
        )
        ground_geometry_result = (
            ground_geometry_estimator.estimate_realtime(
                detection_result.observations
            )
            if ground_geometry_estimator is not None
            else None
        )
        center_pose_observation = (
            center_cross_localizer.localize(field_result)
            if center_cross_localizer is not None and field_result is not None
            else None
        )
        robot_field_point = (
            center_pose_observation.selected_pose.position
            if center_pose_observation is not None
            and center_pose_observation.selected_pose is not None
            else None
        )
        tracks = tracker.update(
            raw_frame.timestamp_ns,
            detection_result.observations,
        )
        snapshot = world_model.update(
            timestamp_ns=monotonic_ns(),
            visual_timestamp_ns=raw_frame.timestamp_ns,
            tracks=tracks,
            robot_field_point=robot_field_point,
            # 中心十字没有唯一解时传 None，世界模型保留明确不确定性。
            # 完整车辆入口会从 OdometryImuFusion 查询该采集时刻的先验。
            # ground_geometry_result 当前提供机器人系中心/足迹；后续规划
            # 接口接入前不在这里伪造全局目标坐标。
        )

        # 真实接触、交付和电控安全适配器尚未实现；应用入口完成后，
        # 在这里把它们组成 TransportStatus/SafetySignals/DeliveryEvidence，
        # 再调用 mission.step(snapshot, ...) 输出不可被规划器覆盖的抽象动作。
```

`ground_mapping_enabled: false` 时仍可运行图像检测，但 `observation.ground_point` 为 `None`。相机、Hailo 和窗口都必须通过上下文管理或 `try/finally` 释放。

## 子包与主要入口

| 子包 | 常用入口 | 对接责任 |
| --- | --- | --- |
| [`app`](app/README.md) | `run_manual_capture_session()`、`LatestCenterCrossLocalization`、`rescue-vision-manual-capture` | 赛外受监督手动驾驶、车载采集和连续融合动态地图状态发布装配 |
| [`config`](config/README.md) | `load_runtime_config()`、`GripperRuntimeConfig`、`AppConfig.build_geometry()`、`HailoConfig.build_backend()` | 启动时严格加载、机械标定和装配 |
| [`camera`](camera/README.md) | `FrameSource`、`CameraFrame`、`Picamera2Source`、`RecordingSource` | 产生带时间和序号的最新帧 |
| [`communication`](communication/README.md) | `UartFrameChannel`、`RemoteMessageConnection`、`VideoModeCommand`、`VideoFrameAttributes`、`MapStateObservation`、`RemoteSessionStatus` | COBS UART、直接 TCP 远程消息、可选择 raw/perception/BEV 图传及轻量 FieldPoint 动态状态 schema |
| [`motion`](motion/README.md) | `MotionController`、`MotionLimits`、`GripperCalibration`、`RemoteMotionExecutor`、`RemoteGripperExecutor`、`run_remote_motion` | 差速运动、持续夹爪双舵机、Rescue Car 协议和远程调试执行 |
| [`geometry`](geometry/README.md) | `CameraModel`、`GroundProjector`、显式坐标类型（含 `MapPixel`） | 去畸变及像素/地面/BEV 转换 |
| [`perception`](perception/README.md) | `TargetPoseDetector`、`PerceptionFrameRenderer`、`TargetGroundGeometryEstimator`、`FieldFeatureDetector`、`FieldBoundaryEstimator` | 任务目标、最新帧可视化旁路、地面几何、静态场地特征和局部场界三态掩膜 |
| [`localization`](localization/README.md) | `CenterCrossLocalizer`、`OdometryImuFusion`、`FusedPoseEstimate` | 中心十字绝对位姿与编码器/IMU 连续融合、延迟视觉纠偏 |
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
- 当前正式任务目标模型、连续定位融合、完整区域/对手感知、真实接触与交付证据、规划、正式任务动作到运动控制的适配和比赛应用入口尚未完成。传统视觉场地特征和中心十字定位目前只有合成测试基线；远程场地图只发布新鲜唯一的低频绝对位姿，不能把过期位置当作连续定位。`app` 的远程驾驶/图传只用于赛外受监督采集，不能替代固件失联看门狗。
