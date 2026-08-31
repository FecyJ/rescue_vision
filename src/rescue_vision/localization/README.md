# `localization`：v3 地标与编码器/IMU 连续定位

本包把 YOLO Pose v3 同帧中心十字/安全区观测转换为低频视觉纠偏，并由
`OdometryImuFusion` 与编码器、IMU 连续预测融合。它不创建相机、Hailo、串口
或世界模型；应用只读取已完成的最新感知快照。

## 公共入口

| 入口 | 用途 |
| --- | --- |
| `CenterCrossLocalizer` | 双轴十字产生四向候选；单 K0 在新鲜航向先验下只产生位置观测 |
| `SafeZoneCornerLocalizer` | 枚举红/蓝身份和 K1/K2 两种世界角点对应 |
| `StaticFieldLandmarkTracker` | 有界确认、地图搜索提示和旧帧去重辅助 |
| `VisualLocalizationPipeline` | 同帧选择并保证每个帧序最多提交一次视觉更新 |
| `FieldPositionObservation` | 不含航向测量的中心十字位置证据 |
| `OdometryImuFusion` | 编码器/IMU 预测、延迟全位姿或位置视觉纠偏和历史重放 |
| `FusedPoseEstimate` | 可空 `FieldPose2D`、不确定度、锚点来源和质量 |

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
`usable: true` 的地标才进入绝对位姿假设。示例配置保持两者为 `false`。

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
- 安全区颜色未知时枚举 red/blue；对每种身份再枚举 K1/K2 两种世界对应。
  无先验时对称解保持歧义，不提交纠偏。
- 同帧同时得到十字和安全区全位姿时只提交一项，避免相关证据重复压缩方差。

读取连续结果：

```python
estimate = fusion.latest_estimate(current_timestamp_ns)
if estimate.pose is None:
    clear_global_robot_pose()
else:
    publish_global_robot_pose(estimate)
```

## 时效和安全降级

- 视觉按相机采集时间与有限融合历史对齐，超过门限或创新过大则拒绝。
- 旧视觉坐标只用于关联和搜索提示，绝不重复提交。
- 单点中心十字不能在连续性丢失后初始化完整位姿；可靠安全区三点或已消歧
  双轴十字可以按现有绝对视觉规则重新锚定。
- 模型/旁路异常由应用传播到统一停车路径；普通漏检、颜色未知或几何歧义只
  停止视觉纠偏，融合按里程计连续性和应用自身绝对位姿门禁保守降级。
- 当前安全区实测坐标、中心至约 1.5 m 的地面标定、真实定位误差和 Hailo
  P50/P95 观测年龄均未验证。
