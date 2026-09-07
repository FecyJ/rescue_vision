# `perception`：YOLO Pose v3 统一感知

本包消费全尺寸去畸变帧，一次 Hailo 推理同时输出四类任务目标、中心十字和
安全区。模型接口固定为 6 类、`kpt_shape: [3,3]`；旧四类 `[1,3]` 资产会被
拒绝。详细标注和部署约定见 [`docs/Pose视觉模型约定.md`](../../../docs/Pose视觉模型约定.md)。

## 公共入口

| 入口 | 作用 |
| --- | --- |
| `HailoYolo26PoseBackend` | HEF 主干、ONNX 后处理、三关键点反 letterbox |
| `PoseKeypoint` / `ModelDetection` | 后端固定三槽位输出 |
| `TargetPoseDetector` | 六类分流、HSV 任务分类、地面投影和场地特征构造 |
| `PoseDetectionResult` | 同帧 `observations` 与 `field_features` |
| `TargetObservation` | 四类目标框、底面中心 K0、可选 `GroundPoint` 和质量 |
| `FieldFeatureDetectionResult` | 同帧中心十字和安全区集合 |
| `CenterCrossObservation` | bbox、K0、可选局部双轴精修 |
| `SafeZoneObservation` | bbox、K0、图像左 K1、图像右 K2及身份状态 |
| `PerceptionFrameRenderer` / `PerceptionSnapshot` | 丢旧保新的后台推理、先发布结构化快照，再由独立旁路生成叠加图 |

`TargetGroundGeometryEstimator` 是旧接触锚点实验链路，不属于 v3 主链路；本次
迁移未修改它。跟踪和任务流程直接消费 `TargetObservation.ground_point`，其
语义现为目标接触底面的几何中心。

## 配置与装配

先加载配置并创建唯一几何对象：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
if geometry is None:
    raise RuntimeError("v3 感知要求启用固定内参")
```

再创建检测器。检测器接管 Hailo 后端生命周期：

```python
detector = config.build_target_pose_detector(
    ground_projector=geometry.ground_projector,
)
if detector is None:
    raise RuntimeError("Hailo 未启用")
```

`frame` 必须已由上游使用同一 `CameraModel` 生成全尺寸去畸变图：

```python
try:
    result = detector.detect_realtime(frame, frame.image_bgr)
    for target in result.observations:
        consume_target(target)  # ground_point 是机器人系底面中心，mm
    if result.field_features is not None:
        consume_field_features(result.field_features)
finally:
    detector.close()
```

生产应用使用 `PerceptionFrameRenderer`，不要在运动循环同步推理：

```python
renderer = PerceptionFrameRenderer(
    lambda: config.build_target_pose_detector(
        ground_projector=geometry.ground_projector,
    )
)
renderer.start()
try:
    renderer.submit(frame)       # 单槽，自动丢弃旧待处理帧
    snapshot = renderer.latest_snapshot()  # 推理完成即返回，不等待叠加渲染
finally:
    renderer.stop()              # 同时关闭检测器和 Hailo 资源
```

## 三关键点与降级

- 四类目标：只消费 K0；K1/K2 必须无效。K0 低于阈值时 `ground_point=None`，
  不使用 bbox 中心或底边兜底。
- 中心十字：K0 投影到机器人地面系；模型 bbox 内可用 Canny/Hough/直线拟合
  精修两条轴。精修失败保留单点和显式质量，不伪造航向。当前目标与场地
  关键点的机器人地面前向坐标统一应用实测 `+225 mm` 修正，像素坐标不变。
- 安全区：标注语义仍定义为图像左 K1/右 K2；推理时若模型把两个同时有效的
  角点交换，后处理会按去畸变图 `u` 自动交换整组点和置信度。只有一点有效时
  不强行重排；身份颜色证据关闭、不足或遮挡时为 `unknown`，由定位层枚举红蓝
  与角点排列。
- `GroundProjector` 缺失时仍可输出像素观测，但所有地面点为 `None`，不能定位。
- 结果超过 `processing.max_observation_age_ms` 时实时接口只返回过期丢弃原因。

局部十字线精修不是旧全图 OpenCV 场地检测：它只在模型已找到的 bbox 内运行，
失败不会产生新的场地候选。出发区和场界不在 v3 模型或本包公共结果中。

## 部署门禁

`hailo.raw_classes` 必须严格为：

```yaml
- green_supply
- black_core
- orange_injured
- blue_danger
- center_cross
- safe_zone
```

后处理映射必须声明 `num_classes: 6`、`kpt_shape: [3, 3]`。示例 HSV、中心十字
精修阈值和安全区颜色范围都不是现场验收值；正式模型、危险类指标、树莓派 +
Hailo P50/P95 观测年龄及远场精度目前均未验证。
