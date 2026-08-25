# `localization`：编码器/IMU 连续定位与中心十字纠偏

本包包含两个明确分层：`CenterCrossLocalizer` 把同帧中心十字、安全区和低精度
场界转换为低频绝对位姿候选；`OdometryImuFusion` 消费应用从唯一 UART 读取链路
分发的 `OdometryImu`，维护连续二维场地位姿并用视觉纠偏。两者都不创建相机、
串口或修改世界模型。

场地原点是中心十字交点。轴方向和四条射线的终端语义来自
`config.world.static_map.center_cross`；默认地图的 `+x` 沿水平基准线向右，
`+y` 指向红色安全区。`FieldPose2D.heading_rad` 是从场地 `+x` 到机器人前向的
逆时针角，范围为 `[-π, π]`。

## 常用类和返回语义

| 入口 | 用途 |
| --- | --- |
| `CenterCrossLocalizer` | 从一帧 `FieldFeatureDetectionResult` 生成绝对位姿候选 |
| `CenterCrossLocalizerConfig` | 射线关联、锚点置信度、先验创新和不确定度下限 |
| `FieldPose2D` | 机器人场地位置与全局航向 |
| `CenterCrossPoseObservation` | 四候选、可选唯一位姿、终端语义、时间和降级原因 |
| `CenterLineTerminalKind` | `red_safe_zone`、`blue_safe_zone`、`plain_boundary` 或 `unknown` |
| `CenterCrossLocalizationQuality` | 缺失/部分十字、无地面坐标、过期、歧义或先验冲突 |
| `OdometryImuFusion` | 累计编码器、陀螺仪和延迟视觉观测的线程安全误差状态 EKF |
| `FusionConfig` / `OdometryCalibration` | 融合噪声/门限与本车计数、轮径、静态零偏 |
| `FusedPoseEstimate` | 可空全局位姿、估计时间、不确定度、绝对锚点来源和质量状态 |

完整双轴十字产生四个相差 90° 的候选。定位器把观察到的终端类别与静态地图
中同类终端所在射线匹配；默认红色对应 `+y`、蓝色对应 `-y`。同一类别只出现
在一条地图射线时才可唯一锚定，默认普通场界位于 `±x`，因此仍保留 180°
歧义。没有可靠锚点时可以由调用方注入时间对齐的先验位姿；
先验或锚点创新超限时 `selected_pose` 保持 `None`。

## 1. 加载配置和装配几何

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
if geometry is None or geometry.ground_projector is None:
    raise RuntimeError("中心十字定位需要启用可用的地面映射")
```

这里的 `geometry` 绑定当前相机、分辨率、安装位姿和标定；静态地图已经由同一
`config` 加载。不能为定位另建单应矩阵、BEV 参数或中心十字方向表。

## 2. 创建场地检测器和定位器

以下片段承接前文的 `config` 和 `geometry`；两个对象都只创建一次：

```python
field_detector = config.perception.build_field_feature_detector(
    static_map=config.world.static_map,
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=geometry.ground_projector,
)
localizer = config.build_center_cross_localizer(
    ground_projector=geometry.ground_projector,
)
if field_detector is None or localizer is None:
    raise RuntimeError("需要同时启用 field_features、localization 和地面映射")
```

创建对象不打开相机，也不启动后台线程。

## 3. 生成一帧视觉位姿观测

以下片段使用上游相机产生的 `raw_frame`，以及同帧经当前 `CameraModel` 产生的
`undistorted_bgr`。`prior_pose` 必须由后续连续推算层对齐到该相机帧时间；没有
可靠先验时传 `None`：

```python
field_result = field_detector.detect(
    raw_frame,
    undistorted_bgr,
    valid_mask=geometry.camera_model.valid_mask,
)
pose_observation = localizer.localize(
    field_result,
    prior_pose=prior_pose,  # FieldPose2D | None
)
```

读取结果时必须检查可选值：

```python
if pose_observation.selected_pose is None:
    # candidates 可能包含四个几何解；质量标记说明为何没有唯一解。
    record_localization_diagnostic(pose_observation)
else:
    visual_pose = pose_observation.selected_pose
    submit_visual_correction(
        visual_pose,
        capture_timestamp_ns=pose_observation.capture_timestamp_ns,
    )
```

完整车辆应用使用 `config.build_odometry_imu_fusion()` 创建融合器。UART 仍由
`MotionController.drain_messages()` 单点读取，调用方只把其中的 `OdometryImu`
送入融合器：

```python
from rescue_vision.motion import OdometryImu

fusion = config.build_odometry_imu_fusion()
if fusion is None:
    raise RuntimeError("当前配置未启用编码器/IMU 融合")

for message in motion_controller.drain_messages():
    record_or_publish(message)
    if isinstance(message, OdometryImu):
        fusion.submit_odometry(message)
```

视觉处理承接前文的 `pose_observation`。先验必须查询相机采集时刻；得到唯一解后
再提交，融合器会在最近历史状态纠偏并重放后续预测：

```python
prior = fusion.pose_at(field_result.capture_timestamp_ns).pose
pose_observation = localizer.localize(field_result, prior_pose=prior)
fusion.submit_visual(pose_observation)

estimate = fusion.latest_estimate(current_timestamp_ns)
if estimate.pose is None:
    clear_global_robot_pose()
else:
    publish_global_robot_pose(estimate)
```

示例中的记录和发布函数属于应用。正式手动采集入口已按相同顺序装配，并在远程
客户端断线时继续排空、融合 UART 遥测。

## 同帧、时效与降级约束

- 中心十字、安全区和场界必须来自同一个 `FieldFeatureDetectionResult`；首版
  不缓存旧锚点。
- 完整十字必须有两条轴及其交点的 `GroundPoint`。单轴或纯像素观测只返回
  显式质量标记，不产生候选。
- 未检测到安全区不等于普通场界；无证据始终是 `unknown`。
- 围栏候选本来就是低精度观测，不能覆盖红/蓝安全区锚点。
- 当前本机地面标定只验收到前方 500 mm，无法证明能从中心看到约 1.5 m 外的
  安全区或场界终端。扩大 BEV 范围前必须重新采集并验收相应距离的地面标定。
- 位置和航向不确定度当前只有配置化保守下限；取得真实场地录像后必须按距离、
  视角、光照和模糊分层校准。

## 连续性与降级

- 状态为场地 `x/y/heading` 与 `gyro_z` 零偏；编码器负责平移和差速转角，陀螺仪
  负责短时航向。加速度只标记静止、倾斜、撞击或饱和，不二次积分平面位置。
- 配置起点仅在进程启动后的首个有效编码器基线使用一次。控制器时间倒退、序号
  反向、超期、计数跳变或编码器失效会清除连续位姿；之后只能由可靠绝对视觉重建。
- IMU 暂时无效可按配置退化为纯轮式预测并放大协方差；编码器无效时不会退化为
  加速度积分。遥测超过 `max_telemetry_age_ms` 时返回显式不可用状态。
- `sample_timestamp_us` 只用于控制器内部 `dt`。融合器以受限最小接收偏移映射到
  树莓派单调时钟供相机对齐，不直接相减两台设备的原始时钟。
- 延迟视觉只在有限历史和时间对齐门限内更新，并经过马氏距离门控；可靠红/蓝或
  静态终端锚点可在连续状态丢失后重新初始化。

离线图片/视频人工检查使用 `manual_tests/cross_localization.py`；命令、JSONL、
双坐标叠加图和固定先验限制见 [`manual_tests` README](../../../manual_tests/README.md)。
