# 工创赛 YOLO Pose v3 模型约定

本文规定训练、导出和运行时必须共同遵守的接口。详细人工标注定义以
[《工创赛 YOLO Pose 新版标注规则 v3》](工创赛_YOLO_Pose新版标注规则_v3_20260828.md)
为唯一权威；本页只说明仓库对接要求。

## 固定模型接口

```yaml
names:
  0: green_supply
  1: black_core
  2: orange_injured
  3: blue_danger
  4: center_cross
  5: safe_zone
kpt_shape: [3, 3]
flip_idx: [0, 2, 1]
```

三个槽位固定为 `ground_anchor`、`image_left_landmark`、
`image_right_landmark`。四类任务目标和中心十字只使用 K0；安全区使用
K0/K1/K2。未使用或不可可靠确定的槽位必须为 `0 0 0`。

- 四类任务目标 K0 是真实接触底面的几何中心投影，运行时投影结果直接作为
  `TargetObservation.ground_point`，不再表示可见最低接触点。
- 当前相机/模型组合的 `GroundPoint.x` 存在实测 `225 mm` 前向偏差；目标和场地
  关键点统一在 `TargetPoseDetector` 的地面坐标边界应用该修正，原始去畸变像素
  坐标不加修正。该值属于当前部署条件，改变相机、焦点或模型后必须重新验证。
- 中心十字 K0 对应固定 `FieldPoint(0, 0)`；K1/K2 不使用。完整航向仍需模型
  bbox 内的局部双轴精修或其他新鲜先验，单点不能产生航向证据。
- 安全区 K0 是中心线与近场围栏地面基准线交点；标注时 K1/K2 是当前去畸变图像中
  `u` 较小/较大的两个近场地面角点，不是固定世界身份。推理后端允许模型交换
  两个同时有效的点，并按 `u` 自动规范化输出槽位；只有一个点有效时保留其原槽位。

## 图像、训练与导出

- 标注、训练、验证和部署统一使用固定 `CameraModel` 的全尺寸去畸变画布；
  无效边缘填充 BGR `(114, 114, 114)`，不裁剪。
- 初版采用 v3 文档中的保守增强；如启用水平翻转必须交换 K1/K2。
- HEF、ONNX 后处理和张量映射必须共同声明 6 类、`kpt_shape: [3,3]` 和相同
  输入尺寸。运行时严格拒绝旧 `[1,3]`、类别缺失、重排或额外类别。
- 四类目标最终运行时类别仍以 ROI HSV 证据为主；当颜色证据不足或歧义且模型检测
  置信度严格高于 `perception.unknown_override_confidence_threshold`（默认 `0.80`）
  时，回退到模型类别，不输出 `unknown`；场地两类不进入目标 HSV 分类。

## 部署检查清单

1. 数据集 YAML 的类别顺序、关键点形状和 `flip_idx` 与本页完全一致。
2. 数据只包含同一 `calibration_id`、尺寸、方向和 `new_K` 的去畸变图。
3. 导出映射中 `num_classes=6`、`kpt_shape=[3,3]`，HEF 与 ONNX 张量匹配。
4. `configs/runtime.yaml` 指向同一批部署资产；旧资产不能复用。
5. 用 `manual_tests/hailo_pose.py` 检查六类框、三个槽位、坐标和观测年龄。
6. 安全区地标现场实测并设置 `measured: true`、`usable: true` 后才可用于定位。

## 迁移影响

仓库不兼容旧四类 `[1,3]` 部署包，也不保留旧 K0“可见最低接触点”语义。
已有标签必须按 v3 复核或重标并重新训练、导出。旧
`TargetGroundGeometryEstimator` 不属于 v3 主链路，本次不修改也不装配；
任务跟踪和比赛流程直接消费 K0 投影得到的底面中心。
