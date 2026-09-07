# `localization`：v3 地标与编码器/IMU 连续定位

本包把 YOLO Pose v3 同帧中心十字/安全区观测转换为低频视觉纠偏，并由
`OdometryImuFusion` 与编码器、IMU 连续预测融合。它不创建相机、Hailo、串口
或世界模型；应用只读取已完成的最新感知快照。

## 公共入口

| 入口 | 用途 |
| --- | --- |
| `CenterCrossLocalizer` | 双轴十字产生四向候选；单 K0 在新鲜航向先验下只产生位置观测 |
| `SafeZoneCornerLocalizer` | 枚举红/蓝身份和 K1/K2 世界角点对应，用 K0/K1/K2 连线估计航向并做长度门控 |
| `StaticFieldLandmarkTracker` | 有界确认、地图搜索提示和旧帧去重辅助 |
| `VisualLocalizationPipeline` | 同帧选择并保证每个帧序最多提交一次视觉更新 |
| `FieldPositionObservation` | 不含航向测量的中心十字位置证据 |
| `OdometryImuFusion` | 编码器/IMU 预测、延迟全位姿或位置视觉纠偏和历史重放；支持扰动协方差注入与锚健康观测量 |
| `FusedPoseEstimate` | 可空 `FieldPose2D`、不确定度、锚点来源和质量 |
| `VisualAnchorHealth` | 视觉锚最近接受时间、连续拒绝计数与拒绝原因 |

场地坐标原点是中心十字交点，`+x` 向场地图右侧，`+y` 指向红色安全区。
`heading_rad` 是场地 `+x` 到机器人前向的逆时针角。

## 配置和装配

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
fusion = config.build_odometry_imu_fusion()
if geometry is None or geometry.ground_projector is None or fusion is None:
    raise RuntimeError("连续视觉定位要求地面映射和 fusion")

visual = config.build_visual_localization_pipeline(
    ground_projector=geometry.ground_projector,
    fusion=fusion,
)
```

`visual is None` 表示 `localization.enabled`、地面映射或融合门禁未满足。安全区
坐标来自 `world.static_map.safe_zone_landmarks`；只有 `measured: true` 且
`usable: true` 的地标才进入绝对位姿假设。示例配置已换入红方现场记录及其
关于中心十字的蓝方对称坐标。

## 连续融合调用顺序

UART 的唯一消费者先提交里程计：

```python
from rescue_vision.motion import OdometryImu

for message in motion_controller.drain_messages():
    if isinstance(message, OdometryImu):
        fusion.submit_odometry(message)
```

感知后台完成同帧快照后提交场地结果：

```python
snapshot = perception_renderer.latest_snapshot()
if visual is not None and snapshot is not None and snapshot.field_features is not None:
    visual.submit(snapshot.field_features)
```

`VisualLocalizationPipeline` 会查询采集时刻的 `pose_at()`，完成中心十字确认、
安全区四种假设枚举和同帧选择。重复或倒退帧序直接忽略：

- 双轴十字可产生四向全位姿候选，由安全区身份或先验消歧。
- 只有十字 K0 时，以先验航向把 `(0,0)` 地标换算成机器人场地位置，并调用
  `submit_position_landmark()`；融合测量矩阵只更新 x/y，不测量航向。
- 安全区定位先按 K0/K1/K2 的可用组合计算点对连线方向，使用连线方向估计
  `heading_rad`。位置至少需要 K0 与 K1 或 K2 之一；对应的 K0-K1 或 K0-K2
  观测距离必须与 `world.static_map.safe_zone_landmarks` 中的实测距离相差不超过
  `localization.safe_zone_corners.max_k0_corner_distance_error_mm`，否则不提交
  安全区位置纠偏。两点通过时用两点刚体变换，三点通过时用三点残差复核。
- 安全区颜色未知时枚举 red/blue；对每种身份再枚举 K1/K2 两种世界对应。
  无先验时对称解保持歧义，不提交纠偏。
- 同帧同时得到十字和安全区全位姿时只提交一项，避免相关证据重复压缩方差。

解团推撞等编码器把滑移误记为行进的事件结束后，先注入一次扰动协方差再继续
常驻纠偏；否则模型不确定度仍偏小，随后到达的真锚会被创新门限当作离群值
拒绝。注入只膨胀最新条目的协方差（对角平方相加），不动状态均值、锚点来源
与历史结构，会随预测传播到后续条目：

```python
fusion.inject_disturbance(
    position_uncertainty_mm=150.0,
    heading_uncertainty_rad=0.08,
)
```

量级由正式流程或调用方的融合配置注入（配置校验强制分别小于应用位姿门限），历史为空时
静默忽略。读取锚健康量判断定位是否明显偏离：

```python
health = fusion.visual_anchor_health()
if health is not None and health.last_accepted_ns is not None:
    anchor_age_s = (current_timestamp_ns - health.last_accepted_ns) / 1e9
```

`last_accepted_ns` 使用被匹配历史条目的 `timestamp_ns`，与
`FusedPoseEstimate.estimate_timestamp_ns` 同域可比；`None` 表示从未接受过
视觉锚或连续性已丢失。`consecutive_rejections` 只累计对齐超限与创新门限两类
拒绝；`last_rejection_reason` 为 `"alignment_error"` 或 `"innovation_gate"`。

读取连续结果：

```python
estimate = fusion.latest_estimate(current_timestamp_ns)
if estimate.pose is None:
    clear_global_robot_pose()
else:
    publish_global_robot_pose(estimate)
```

## 时效和安全降级

- 视觉按相机采集时间与有限融合历史对齐，超过门限或创新过大则拒绝；这两类
  拒绝累计到 `VisualAnchorHealth.consecutive_rejections` 并记录拒绝原因，
  `(False, None, None)` 的“无观测/无历史”不计入也不清零。接受任意视觉锚即清零
  计数并记录 `last_accepted_ns`；连续性丢失同时清零三者（之后只有绝对视觉可
  重新初始化并再次记录接受）。
- 旧视觉坐标只用于关联和搜索提示，绝不重复提交。
- 单点中心十字不能在连续性丢失后初始化完整位姿；可靠安全区三点或已消歧
  双轴十字可以按现有绝对视觉规则重新锚定。
- 模型/旁路异常由应用传播到统一停车路径；普通漏检、颜色未知或几何歧义只
  停止视觉纠偏，融合按里程计连续性和应用自身绝对位姿门禁保守降级。
- 当前安全区实测坐标、中心至约 1.5 m 的地面标定、真实定位误差和 Hailo
  P50/P95 观测年龄均未验证。
