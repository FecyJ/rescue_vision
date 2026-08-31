# `localization`：编码器/IMU 连续定位与中心十字消费层

本包包含两个明确分层：`CenterCrossLocalizer` 把同帧中心十字、安全区和低精度
场界观测转换为低频绝对位姿候选；`OdometryImuFusion` 消费应用从唯一 UART 读取
链路分发的 `OdometryImu`，维护连续二维场地位姿并提供延迟视觉纠偏接口。两者
都不创建相机、串口或修改世界模型。传统 OpenCV 场地检测已删除，当前没有
场地特征模型推理生产者，因此 `CenterCrossLocalizer` 暂无运行时输入。

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
| `StaticFieldLandmarkTracker` | 用静态地图和时间对齐位姿产生可恢复搜索提示，并维护短时地标证据 |
| `SafeZoneCornerLocalizer` | 用安全区入口角点对或三/四角拟合唯一场地位姿 |
| `SafeZoneCornerPoseObservation` | 安全区颜色、所用角点、拟合残差、位姿和不确定度 |
| `FusionConfig` / `ImuFrameCalibration` / `OdometryCalibration` | 融合门限、完整三轴 IMU 校准与本车编码器机械量 |
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

## 2. 创建定位器

以下片段承接前文的 `config` 和 `geometry`；定位器只创建一次：

```python
localizer = config.build_center_cross_localizer(
    ground_projector=geometry.ground_projector,
)
if localizer is None:
    raise RuntimeError("需要同时启用 localization 和地面映射")
```

创建对象不打开相机，也不启动后台线程。传统 OpenCV 场地特征检测器已删除；
`localizer` 只消费上游模型推理后端未来产出的同帧
`FieldFeatureDetectionResult`，当前仓库没有该生产者，因此没有可运行的从图像
到定位的完整装配。

## 3. 消费一帧场地特征观测

以下片段承接前文的 `localizer`。`field_result` 必须由未来实现的场地特征
模型后端针对同一采集帧产出；`prior_pose` 必须由连续推算层对齐到该相机帧
时间，没有可靠先验时传 `None`：

```python
pose_observation = localizer.localize(
    field_result,  # 模型场地特征后端产出的 FieldFeatureDetectionResult
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

`ODOMETRY_IMU` 的三轴陀螺和加速度字段处于 STM32 IMU 自身坐标系。融合器在消费
每个有效帧前使用 `config.localization.fusion.imu_calibration`，分别执行：

```text
bias_at_temperature = bias + temperature_coefficient * (temperature - reference_temperature)
sensor_corrected    = cross_axis_scale @ (sensor_raw - bias_at_temperature)
robot_value         = sensor_to_robot_rotation @ sensor_corrected
```

陀螺单位为 rad/s，加速度单位为 mm/s²，温度单位为 °C。bias/温度系数为三向量，
陀螺和加速度各自拥有可逆的 3×3 比例/交叉轴矩阵；安装旋转必须正交且行列式为
`+1`。之后仅对机器人系 `gyro_z` 应用一次 `motion.odometry.gyro_z_sign`。调用方
不得再次减零偏、旋转或翻转；模板零值和单位阵不能作为真车精度证据。

这一步可以单独作为“编码器+IMU 航位推算”阶段运行，不要求启用
`localization.enabled` 或调用相机视觉定位。估计的
绝对坐标起点来自 `localization.fusion.initial_pose`，因此它是带初始位姿假设和
随时间累积漂移的连续 `FieldPose2D`，不是现场绝对定位证据。配置
`allow_wheel_only` 还可以决定 IMU 无效时是否允许短时双编码器退化。

视觉处理承接前文的 `pose_observation` 和未来模型后端产生的 `field_result`。
先验必须查询相机采集时刻；得到唯一解后
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

正式视觉旁路还会把同一 `pose_at()` 结果交给
`StaticFieldLandmarkTracker.search_hint()`：先验新鲜且不确定度足够小时，生成
携带预测中心地面点、先验半径、两条无向地图轴方向和位置/航向不确定度的
`FieldFeatureSearchHint`，只在预测中心十字和红蓝安全区附近接受候选；融合
方差增大时先验半径和轴走廊随之放宽。先验超界或缺失时返回 `None`，检测器在
整幅有效 BEV 上运行同一套片段、拟合和评分流程。无先验的完整十字候选由
`StaticFieldLandmarkTracker.confirm_center_cross()` 确认：关联容差由实际帧
间隔和运行配置的最大线速度/角速度推导，连续两帧地面交点与轴方向一致后才
升级为 `temporal_confirmed`；先验引导观测直接通过，超时、帧序倒退或十字消失
都会重置待确认状态。追踪保存的旧坐标只用于预测与关联，绝不重新提交为当前
视觉观测。

安全区紫框入口侧两个角点优先与静态地图匹配；三/四角可见时共同拟合并检查
残差。至少两个具有足够基线的当前帧角点才能产生
`SafeZoneCornerPoseObservation`。同帧同时产生安全区角点与中心十字绝对位姿
时，`select_same_frame_pose_observation()` 只保留先验创新更小（无先验时置信度
更高）的一项提交融合器，避免相关视觉证据被二次使用。
`fusion.submit_visual()` 对中心十字和安全区角点使用相同的采集时间对齐、创新
门限和历史重放；可靠角点观测属于绝对地标，可在连续性丢失后重新锚定。

示例中的记录和发布函数属于应用。正式手动采集入口已按相同顺序装配，并在远程
客户端断线时继续排空、融合 UART 遥测。

## 同帧、时效与降级约束

- 中心十字、安全区和场界必须来自同一个 `FieldFeatureDetectionResult`；短时
  追踪只缓存身份和搜索证据，不缓存可重复融合的旧观测。
- 完整十字必须有两条轴及其交点的 `GroundPoint`。单轴或纯像素观测只返回
  显式质量标记，不产生候选。
- 未检测到安全区不等于普通场界；无证据始终是 `unknown`。
- 围栏候选本来就是低精度观测，不能覆盖红/蓝安全区锚点。
- 当前本机地面标定只验收到前方 500 mm，无法证明能从中心看到约 1.5 m 外的
  安全区或场界终端。扩大 BEV 范围前必须重新采集并验收相应距离的地面标定。
- 位置和航向不确定度当前只有配置化保守下限；取得真实场地录像后必须按距离、
  视角、光照和模糊分层校准。

## 连续性与降级

- 状态为场地 `x/y/heading` 与校准后机器人 `gyro_z` 的残余零偏；编码器负责平移和差速转角，陀螺仪
  负责短时航向。加速度只标记静止、倾斜、撞击或饱和，不二次积分平面位置。
- `imu_calibration` 在输入边界依次执行温度零偏、比例/交叉轴和安装旋转；随后
  `motion.odometry.gyro_z_sign` 把机器人系 `gyro_z` 转换为内部左转为正。融合状态、
  `FieldPose2D.heading_rad` 和下游接口不随硬件极性改变。
- 配置起点仅在进程启动后的首个有效编码器基线使用一次。控制器时间倒退、序号
  反向、超期、计数跳变或编码器失效会清除连续位姿；之后只能由可靠绝对视觉重建。
  `continuity_loss_reason` 保留最近一次清除的精确分支原因，成功使用配置起点或绝对视觉
  重新锚定后恢复为 `None`；它只是诊断，不会自动重用起点或放宽连续性门限。
- IMU 暂时无效或陀螺饱和时可按配置退化为纯轮式预测并放大协方差；无效 IMU
  数值不会参与倾斜或冲击计算，饱和原因仍保留为质量状态。单次且双编码器有效的
  `sample_overrun` 会暂存，下一帧正常遥测到达后用前后有效陀螺数据线性插值，并在
  结果中保留 `interpolated_imu` 质量；若连续第二帧仍 overrun，但双编码器、序号、
  时间和轮速仍合法，且 `allow_wheel_only=true`，则对这两段用累计编码器做纯轮式
  降级并同时放大丢帧与纯轮式协方差。连续第三帧 overrun、编码器无效或时间/计数
  不连续仍会中断连续积分。单帧插值是否允许由 `max_interpolated_overrun_samples`
  控制，当前只能是 0 或 1。
  编码器无效时不会退化为加速度积分。遥测超过 `max_telemetry_age_ms` 时返回显式不可用状态。
- `sample_timestamp_us` 只用于控制器内部 `dt`。融合器以受限最小接收偏移映射到
  树莓派单调时钟供相机对齐，不直接相减两台设备的原始时钟。overrun 帧按协议保留
  最近一次有效 IMU 提交时间，UART 又可能批量交付，因此该帧的编码器速度检查不使用
  重复的 `sample_timestamp_us` 或树莓派接收间隔；融合器用协议固定 10 ms 释放周期维护
  编码器有效时间轴，恢复正常 IMU 帧后再回到 raw 提交时间。
- 延迟视觉只在有限历史和时间对齐门限内更新，并经过马氏距离门控；可靠红/蓝或
  静态终端锚点可在连续状态丢失后重新初始化。

传统 OpenCV 场地检测器和 `manual_tests/cross_localization.py` 离线检查工具
已随该策略一起删除；场地特征模型推理后端实现后，需要补充新的真机/回放人工
检查工具，并覆盖命令、JSONL、双坐标叠加图和固定先验限制。
