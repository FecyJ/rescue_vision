# `perception`：任务目标与静态场地特征感知

本包把全尺寸去畸变图上的 Pose 模型结果转换为稳定的
`TargetObservation`：模型负责目标框和 K0，框内 HSV 证据负责最终任务
类别并提供局部颜色分割掩码。`TargetGroundGeometryEstimator` 再利用该掩码、
K0、完整相机外参和可配置三维形状估计目标地面中心、朝向与足迹。独立的
`FieldFeatureDetector` 从同一去畸变帧检测安全区、出发区、中心十字和低精度
边界候选；独立的 `FieldBoundaryEstimator` 在机器人地面系中做共线拟合、矩形
软约束和连续帧确认，输出三态 BEV/图像掩膜；
`localization.CenterCrossLocalizer` 再消费这些同帧观测，不回读
图像或掩码。这些链路都不创建相机、不复制标定矩阵，也不修改跟踪、定位、
世界模型或任务状态。

## 常用类和函数

| 入口 | 对接作用 |
| --- | --- |
| `InferenceBackend` | 推理后端协议：`infer()` 和 `close()` |
| `HailoYolo26PoseBackend` | HEF 推理与 ONNX 后处理的真实后端 |
| `TargetPoseDetector` | 模型框/K0、ROI HSV 分类分割、观测年龄和可选地面投影 |
| `TargetGroundGeometryEstimator` | 四类目标的传统视觉三维模板拟合和地面中心估计 |
| `TargetGroundGeometry` | 接触锚点、可选中心/足迹/朝向、误差和降级原因 |
| `TargetGroundGeometryConfig` | 四类形状尺寸、搜索步长、拟合权重和接受门限 |
| `BoxTargetGeometry` / `RegularTetrahedronTargetGeometry` | 盒体和正四面体尺寸 |
| `FieldFeatureDetector` | 颜色、线段、角点和可选 BEV 的静态场地特征检测 |
| `FieldFeatureDetectionResult` | 同帧安全区、出发区、中心十字和边界候选集合 |
| `RealtimeFieldFeatureResult` | 实时场地结果或明确的过期丢弃原因 |
| `SafeZoneObservation` | 红/蓝物理颜色、入口、隔板和入口视角左右半区 |
| `StartZoneObservation` | 无编号洋红出发区轮廓及可选地面角点 |
| `CenterCrossObservation` | 无序的两条中心轴或单轴部分观测 |
| `BoundaryFeatureObservation` | 显式低置信度的围栏基线或场地角点候选 |
| `FieldFeatureConfig` | 场地颜色、形态学、公差、线段和角点阈值 |
| `FieldBoundaryEstimator` | 把逐帧围栏候选合并为时序确认的机器人系开放场界 |
| `FieldBoundaryMask` / `FieldMaskState` | 只读 `inside` / `uncertain` / `outside` 图像与 BEV 掩膜 |
| `FieldBoundaryConfig` | 共线/RANSAC、矩形公差、时序、边界带和遮挡参数 |
| `RealtimeDetectionResult` | 实时检测结果，并明确记录是否丢弃了过期帧 |
| `render_target_observations()` | 在同坐标系图像副本上叠加框、颜色掩码、K0、置信度和质量 |
| `PerceptionFrameRenderer` | 单槽最新帧后台推理与可视化旁路；不阻塞相机/运动循环；`clear_latest()` 用于切换模式时丢弃旧结果 |
| `StaleObservationError` | 严格 `detect()` 在结果超过允许年龄时抛出的异常 |
| `ModelDetection` | 后端输出；框和 K0 已反映射到去畸变原尺寸 |
| `TargetObservation` | 下游跟踪、定位和评测消费的统一观测 |
| `RoiColorSegmentation` | 整数 ROI、HSV 候选/状态、只读掩码、覆盖率和 dominance |
| `HsvColorClassifierConfig` / `HsvRange` | HSV 闭区间及分类、去噪门槛 |
| `TargetClass` | 四类任务目标及运行时 `unknown` |
| `ClassProbabilities` | 四类目标证据和 `unknown` 剩余概率的完整分布 |
| `ObservationQuality` | 保留颜色不足/歧义、Pose 冲突、K0 不可用等原因 |
| `UndistortedBoundingBox` | 去畸变图中的 `(left, top, right, bottom)` |
| `TargetAnnotation` | 评测适配器使用的人工目标真值 |
| `observations_to_evaluation_records()` | 观测与人工真值匹配后生成评测记录 |

`FakeInferenceBackend` 只用于无硬件自动测试和上层算法注入，不是部署示例。

## 1. 从运行配置加载几何和模型

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
backend = config.hailo.build_backend()

if geometry is None:
    raise RuntimeError("任务感知需要启用内参")
if backend is None:
    raise RuntimeError("任务感知需要启用 Hailo")
```

`geometry` 和 `backend` 都绑定当前运行配置。下文继续复用它们，不在逐帧
循环中重新加载标定或打开 Hailo。

## 2. 按相机配置创建帧源

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource

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

这里只创建 `source`，尚未打开相机。

## 3. 创建检测器

以下片段承接前文的 `config`、`geometry` 和 `backend`。检测器从构造成功
开始接管 backend 生命周期：

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
```

## 4. 读取一帧并执行实时检测

以下片段复用前文的 `source`、`detector` 和 `geometry`；两个上下文负责
关闭相机和 Hailo 后端：

```python
with source, detector:
    raw_frame = source.read(timeout=1.0)
    undistorted_bgr = geometry.camera_model.undistort_image(
        raw_frame.image_bgr
    )
    result = detector.detect_realtime(raw_frame, undistorted_bgr)
    if result.stale_dropped:
        # 记录性能异常即可；过期结果已经被模块清空，不会传给下游。
        print("dropped stale frame:", result.dropped_stale_age_ms)
    observations = result.observations
```

`detect()` 的两个输入必须属于同一采集帧：第一个参数提供原始 `sequence/timestamp_ns`，第二个参数是该帧经当前 `CameraModel` 产生的去畸变图。不能把缓存旧图、裁剪图或另一相机的图像配给当前帧。

可选的 `field_mask` 必须由同一帧或仍在配置时效内的场界估计产生。掩膜过期
时检测器自动使用原图；明确场外的普通目标按 K0 丢弃，危险类和 `unknown`
只添加 `outside_field_suspected`，不会被单帧场界静默删除。

实时调用统一使用 `detect_realtime()`：它只兜底
`StaleObservationError`，返回空观测并通过 `stale_dropped` /
`dropped_stale_age_ms` 记录原因。输入尺寸、类别映射、模型或硬件异常仍会抛出。
离线评测和需要严格失败语义的工具可以直接调用 `detect()`。

## 4.1 为远程图传创建 perception 可视化旁路

远程图传使用当前 `runtime.yaml` 的 Hailo、类别和 HSV 配置，不在通信或应用
模块另写模型路径、阈值或映射。以下片段承接第 1、2 节的 `config`、`geometry`
和 `source`；它使用配置装配方法，因此不要同时复用第 3 节已经手工创建的
`backend`：

```python
from rescue_vision.perception import PerceptionFrameRenderer

renderer = PerceptionFrameRenderer(config.build_target_pose_detector)
renderer.start()
```

创建并启动后，远程发布循环只提交最新的去畸变 `CameraFrame`，不等待推理：
下例中的 `undistorted_frame` 由前文 `source` 读取并经过同一 `CameraModel` 准备。

```python
renderer.submit(undistorted_frame)
visualized_frame = renderer.latest()
if visualized_frame is not None:
    # visualized_frame.image_bgr 与 undistorted_frame 使用同一
    # (width, height) 和 UndistortedPixel 图像坐标；其 sequence/timestamp_ns
    # 保留原相机帧身份，再交给 JPEG 图传发布器。
    publish_jpeg(visualized_frame)
```

退出或断线时，调用方仍必须停止旁路；切回只传原图时可以保留已启动的旁路，
但不再提交帧：

```python
renderer.stop()
```

旁路只保留一个待处理帧和一个最新结果；未产生第一张可视化图时返回 `None`，
而不是把原图伪装成 perception 结果。推理过期时返回带红色 `STALE dropped`
标记的图；Hailo、输入或渲染异常通过 `submit()`/`latest()` 抛出，调用方应关闭
当前远程会话并按应用安全路径清理资源。

若 `ground_mapping_enabled: false`，检测仍正常运行，但所有 `ground_point` 为 `None`。后续定位模块不能把 `None` 当作 `(0, 0)`。

## 5. 读取统一观测

```python
from rescue_vision.perception import ObservationQuality, TargetClass

for observation in observations:
    # HSV 覆盖不足或多色歧义不会被强行四选一。
    if observation.target_class is TargetClass.UNKNOWN:
        print(
            "uncertain target:",
            observation.color_segmentation.status,
            observation.box,
        )
        continue

    if ObservationQuality.K0_UNAVAILABLE in observation.quality:
        # 可以保留图像框用于跟踪，但不能用于地面定位。
        print("image-only observation:", observation.box)
        continue

    if observation.ground_point is not None:
        # 这是后续跟踪/定位接口所需的核心信息。
        local_measurement = {
            "target_class": observation.target_class,
            "point_mm": observation.ground_point,
            "capture_timestamp_ns": observation.capture_timestamp_ns,
        }
```

实际下游应重点使用：

| 字段 | 含义 |
| --- | --- |
| `frame_sequence` | 采集帧序号，用于同帧关联 |
| `capture_timestamp_ns` | 相机帧进入应用的单调时间 |
| `result_timestamp_ns` | 推理、HSV 分类和掩码后处理全部完成的时间 |
| `model_target_class` | Pose 模型原始类别映射，仅用于诊断 |
| `target_class` / `class_probabilities` | HSV 主分类结果及颜色像素分布 |
| `detection_confidence` | 后端原始检测分，不冒充颜色置信度 |
| `box` | 全尺寸去畸变图中的目标框 |
| `color_segmentation` | ROI 局部掩码、颜色候选/状态、覆盖率和 dominance |
| `k0` / `k0_confidence` | 目标与地面的接触锚点及置信度 |
| `ground_point` | 可选机器人地面系毫米坐标 |
| `quality` | 不应被静默丢弃的降级原因 |

## 6. 类别与 K0

| `TargetClass` | 任务语义 |
| --- | --- |
| `GREEN_SUPPLY` / `green_supply` | 普通物资 |
| `BLACK_CORE` / `black_core` | 核心物资 |
| `ORANGE_INJURED` / `orange_injured` | 伤员 |
| `BLUE_DANGER` / `blue_danger` | 危险目标 |
| `UNKNOWN` / `unknown` | 运行时保守降级，不是第五个训练标签 |

公共契约只包含 `K0 bottom_contact_anchor`。后端可以读取旧三关键点部署包，但只消费 K0；新部署包使用 `kpt_shape: [1, 3]`。完整标注和导出约定见 [`docs/Pose视觉模型约定.md`](../../../docs/Pose视觉模型约定.md)。

最终任务类别遵循以下固定顺序：

1. 后端检测分低于 `detection_threshold` 时不形成观测。
2. 对保留框的 ROI 按配置生成四类 HSV 掩码，执行开闭运算和小连通域清理。
3. 顶部颜色覆盖不足时标记 `color_evidence_insufficient`；顶部与第二颜色
   过近时标记 `color_evidence_ambiguous`；两者都输出 `unknown`。
4. HSV 证据充分时直接采用 HSV 类别。即使 Pose 类别不同也不回退，只添加
   `pose_color_conflict` 供诊断；这用于直接纠正已知的蓝绿混淆。

`RoiColorSegmentation.mask` 是与 `roi_box` 同尺寸的只读 `uint8` 数组，前景
为 255、背景为 0。它只覆盖一个检测框，不是整帧掩码；即使最终类别因颜色
不足或歧义成为 `unknown`，仍保留顶部颜色候选掩码。完全无颜色像素时候选为
`unknown` 且掩码全零。

## 7. 传统视觉目标几何与场地特征

### 7.1 估计任务目标地面中心

以下片段承接第 1 节的 `config`、`geometry`，以及第 4 节刚产生的
`detection_result.observations`。配置关闭时不构造估计器；启用时必须加载带
完整物理外参的地面标定：

```python
ground_geometry_estimator = (
    config.perception.build_target_ground_geometry_estimator(
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=geometry.ground_projector,
    )
)
```

在实时帧循环中，紧接目标检测调用：

```python
if ground_geometry_estimator is not None:
    geometry_result = ground_geometry_estimator.estimate_realtime(
        detection_result.observations
    )
    if geometry_result.stale_dropped:
        target_geometries = ()
    else:
        target_geometries = geometry_result.estimates
else:
    target_geometries = ()
```

结果顺序与输入 `TargetObservation` 一致，帧号和采集时间保持不变。现有
`TargetObservation.ground_point` 仍只表示 K0 接触锚点；不能把它当作中心。
中心只读取 `TargetGroundGeometry.center_ground`，拟合不充分时该字段为
`None`：

```python
for estimate in target_geometries:
    if estimate.center_ground is None:
        print(estimate.target_class, estimate.quality)
        continue
    print(
        estimate.center_ground,
        estimate.footprint_ground,
        estimate.center_uncertainty_mm,
    )
```

当前实现对四类使用两种三维模型：

- 绿色普通物资、浅蓝危险目标：可配置长、宽、高的盒体；
- 橘色伤员：可配置长、宽、高的长方体；
- 黑色核心物资：以正三角形面接地、棱长可配置的正四面体。

估计器先按粗朝向枚举 K0 可能对应的底面顶点和边中点，并用少量偏移容纳锚点
误差；随后交替细化中心两轴与朝向，最后只检查局部联合邻域。每个候选仍把
三维顶点投影回去畸变 ROI，以颜色掩码 IoU、轮廓距离和 K0 到接触边的距离
评分。这样避免逐点扫描整个足迹外接圆。正方形盒体朝向按 90° 对称，长方体
按 180°，正四面体按 120°；对称朝向不会伪装成中心歧义。

颜色为 `unknown`、掩码不可用、拟合分数不足、K0 与模型接触边矛盾或候选中心
分散超过门限时，保留失败分数和质量标记但返回 `center_ground=None`。
K0 不可用时可从检测框底部建立搜索种子，但会显式附加
`k0_unavailable`。黑色阴影也可能进入 HSV 掩码，因此实物验收必须单列阴影、
低对比地面和遮挡失败样例。

形状尺寸和全部拟合门限只在运行配置的
`perception.target_ground_geometry` 中维护。当前合成投影测试证明坐标、形状
和降级契约可运行；真实单调时钟回归会检查默认 150 ms 时效预算。该回归不
证明现场中心误差或树莓派端到端性能，目标硬件仍需按 P50/P95 观测年龄验收。

### 7.2 静态场地特征

以下片段承接第 1 节的 `config` 和 `geometry`。传统视觉不需要 Hailo；
配置关闭时构造函数返回 `None`，不会静默运行另一套默认阈值：

```python
field_detector = config.perception.build_field_feature_detector(
    static_map=config.world.static_map,
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=geometry.ground_projector,
)
if field_detector is None:
    raise RuntimeError("需要启用 perception.field_features")
```

对已经由当前 `CameraModel` 去畸变的同一帧执行：

```python
realtime_field_result = field_detector.detect_realtime(
    raw_frame,
    undistorted_bgr,
    valid_mask=geometry.camera_model.valid_mask,
    # 绝对定位不使用普通场界消歧，省去两次昂贵 Hough；需要区域/场界输出的
    # 通用感知调用方保持默认 True。
    include_boundary_features=False,
)
if realtime_field_result.stale_dropped:
    # 不向定位层传递过期地标。
    field_result = None
else:
    field_result = realtime_field_result.result

for safe_zone in field_result.safe_zones if field_result is not None else ():
    print(safe_zone.physical_color, safe_zone.quality)
    for half in safe_zone.halves:
        # side 是从场内面向入口时的 approach_left / approach_right。
        print(half.side, half.polygon_ground)
```

`valid_mask` 必须与去畸变图同尺寸，直接使用当前 `CameraModel.valid_mask`。
检测器会在颜色、线段和 BEV 路径中排除无效填充区，防止去畸变填充边界被
误报为围栏或场角。

`detect()` / `detect_realtime()` 的 `include_boundary_features` 默认为 `True`，
因此普通调用方仍得到边界候选。只消费中心十字和红蓝方向锚点的实时定位旁路可
显式传 `False`；此时 `boundary_features=()`，不能同时把该帧当作场界观测。

启用带 BEV 配置的 `GroundProjector` 时，检测器每帧只生成一次 BEV 并在颜色、
线段和角点步骤中复用。输出仍保留 `UndistortedPixel`；只有地面上的区域角点、
线段端点才附带机器人系 `GroundPoint`。围栏顶部等非地面特征不会被地面
单应性强行投影。

安全区首先输出 `red` / `blue` 物理颜色，不在检测器中解释己方/对方或
物资/伤员用途。左右以“从场内面向入口”为准；只有入口紫色证据、黑色隔板和
地面方向均成立时才生成两个 `halves`。证据不足时保留整区并标记
`entrance_unresolved`、`divider_unresolved` 或 `side_unresolved`。

完整彩色区域优先按静态地图尺寸和矩形度确认。紫色围栏有高度、遮挡导致完整
尺寸不再可见时，检测器不会单纯放宽长宽比：同色可见区域必须落入同一个紫色
围栏外接矩形、总颜色面积和可见矩形度仍达标，并满足双颜色分区、黑色中隔或
BEV 围栏物理尺寸三项中的至少一项，才输出带
`partial`、`entrance_unresolved`、`divider_unresolved` 和
`side_unresolved` 的保守安全区。散落的同色任务物块没有共同紫色围栏，不会
触发该降级路径。部分安全区只提供可见颜色范围的多边形，不补画被遮挡区域；
定位层仍需另外通过中心十字射线、距离、横向偏差、置信度和时效门限后才能把它
作为方向锚点。

中心十字 Hough 候选去重优先保留局部线支撑和点划证据更强的轴，长度只作为
次级排序，避免远场围栏或场边长线把较短但语义更明确的真实十字轴挤出候选集。
安全区确认后会把其保守多边形加入十字排除掩膜，紫色围栏及内部隔板不会再次
参与中心十字拟合。

检测到可靠安全区围栏时，还会检查十字候选的四条轴向射线：至少一条严格对齐
围栏质心并实际穿过围栏多边形边界，且交点距离超过最小轴长，才获得安全区锚点
优先级。该证据只选择更可信的图像十字候选，不直接解释红蓝全局方向；最终消歧
仍由定位层的射线角度、500–1800 mm 距离、横向误差、置信度和时效门限完成。

出发区只输出洋红轮廓和角点，不做数字 OCR，也不包含 1–4 编号。中心十字先
合并两类证据：HSV 深色点划线，以及比局部地面更暗、低饱和的连续细线；后者
用于真实 BEV 中因材料、曝光和缩放而呈灰色实线的中心标记。Hough 重复线会按
方向和偏移折叠，再要求两轴近似垂直、交点远离 BEV 无效边缘，并且交点位于
每条轴的内部而非端点。只有一条有效轴时标记 `partial`；有效区边缘形成的 L 形
交点不会作为中心十字。两条轴在地图匹配前仍保持无序。围栏基线和角点始终带
`low_confidence_boundary`，单帧结果不能直接当作闭合场界。

`center_cross.local_window_fraction` 控制局部背景尺度，
`local_contrast_threshold` 与 `max_saturation` 控制灰色细线响应；
`min_floor_value` 与 `min_white_surround_fraction` 要求候选轴两侧主要是低饱和
高亮度白地板，允许有色物块造成局部遮挡，但排除货架、紫色围栏和场界外暗背景；
`min_line_support_fraction`、`min_axis_balance_fraction` 和
`min_intersection_margin_fraction` 分别限制沿轴支撑、交点两侧覆盖和距无效边缘
的距离。完整十字置信度由垂直度、两侧平衡和局部线支撑共同计算，不再使用固定
常数。上述值仍是基于当前真实 BEV 样例的启动参数，必须通过不同距离、曝光、
遮挡和场地材料的录像分层校准。

中心十字的四向对称、红蓝安全区终端关联、先验门控和 `FieldPose2D` 约定由
[`localization` README](../localization/README.md) 统一定义；感知层不把
`GroundPoint` 伪装成 `FieldPoint`。

完整算法阈值位于 `configs/runtime.example.yaml` 的
`perception.field_features`；安全区和出发区物理尺寸只从
`world.static_map.regions` 推导。红、蓝、洋红和紫色初值来自命题示意图，
不是官方色卡；尺寸公差也只是启动值。获得现场材料、固定曝光和实际地面映射后
必须重新标定。

### 7.3 时序局部场界和三态掩膜

以下片段承接 7.2 的 `field_detector`、`geometry` 和同一帧变量。场界估计要求
地面标定包含 BEV；它不需要 `FieldPose2D`，也不会产生或伪造全局坐标：

```python
field_boundary_estimator = config.perception.build_field_boundary_estimator(
    static_map=config.world.static_map,
    ground_projector=geometry.ground_projector,
)
```

在每帧中先更新场界，再把结果交给目标检测器：

```python
field_mask = None
if field_result is not None and field_boundary_estimator is not None:
    field_mask = field_boundary_estimator.update(
        field_result,
        valid_mask=geometry.camera_model.valid_mask,
    )

detection_result = detector.detect_realtime(
    raw_frame,
    undistorted_bgr,
    field_mask=field_mask,
)
```

逐帧边界候选以长底边为主，并用与底边近似垂直的重复支撑线及线段两侧亮度
变化提高置信度；这些证据不足时仍保留低置信度候选，不强判场界。
`FieldBoundaryEstimator` 仅消费 `FENCE_BASE_SEGMENT` 地面线段，以确定性 RANSAC
合并共线端点，使用场地图尺寸约束不合理平行边，并由机器人原点及同帧安全区、
出发区、中心十字地面点确定场内法向。边界必须达到连续帧确认次数才参与分类；
有限线段只在配置的端部扩展范围内生效，单边或双边不会被补成闭合矩形。

掩膜编码固定为 `outside=0`、`uncertain=127`、`inside=255`。只有 `outside`
会在推理副本中以配置的中性 BGR 渐变填充；原始录像、显示图和 HSV 复核仍使用
未遮挡图像。`hard_mask_enabled` 默认关闭，必须用真实场边录像验收边界误差和
危险类召回后才能开启。无候选、未确认、过期或低置信度时
`filter_ready=false`、`hard_mask_ready=false`，检测前后都自动回退为不使用
场界过滤。

离线图片或视频检查：

```bash
python manual_tests/field_features.py recordings/field_sample.mp4 \
  --config configs/runtime.yaml \
  --already-undistorted \
  --output-jsonl output/field_features.jsonl \
  --overlay-dir output/field_feature_overlays
```

若输入是原始相机图像，去掉 `--already-undistorted`，脚本会使用配置中的
`CameraModel` 去畸变。该工具不访问相机和 Hailo；合成图测试只证明逻辑和
契约可运行，不能作为现场精度或树莓派性能证据。

## 8. Hailo 部署包和配置

部署包包含：

```text
model_bundle/
├── model.hef
├── postprocess.onnx
└── onnx_split_config.json
```

资产路径、原始类别顺序、类别映射及后端粗筛来自
`hailo`；检测、K0 和 HSV 参数来自 `perception`：

```yaml
hailo:
  enabled: true
  hef_path: ../models/target_pose/model.hef
  postprocess_onnx_path: ../models/target_pose/postprocess.onnx
  output_mapping_path: ../models/target_pose/onnx_split_config.json
  raw_classes: [green_supply, black_core, orange_injured, blue_danger]
  class_mapping:
    green_supply: green_supply
    black_core: black_core
    orange_injured: orange_injured
    blue_danger: blue_danger
  backend_score_threshold: 0.01
  max_detections: 100

perception:
  detection_threshold: 0.25
  k0_threshold: 0.50
  color_classifier:
    ranges:
      green_supply:
        - lower: [35, 70, 71]
          upper: [84, 255, 255]
      black_core:
        - lower: [0, 0, 0]
          upper: [179, 255, 70]
      orange_injured:
        - lower: [0, 90, 80]
          upper: [20, 255, 255]
        - lower: [170, 90, 80]
          upper: [179, 255, 255]
      blue_danger:
        - lower: [85, 50, 71]
          upper: [110, 255, 255]
    min_color_fraction: 0.15
    min_color_dominance: 0.70
    min_dominance_margin: 0.20
    morphology_kernel_size: 3
    open_iterations: 1
    close_iterations: 1
    min_component_area_fraction: 0.002
```

`backend_score_threshold` 是 Hailo/后处理的低成本粗筛，必须不高于
`perception.detection_threshold`；后者决定检测是否进入统一观测。
`perception.k0_threshold` 只决定接触锚点及地面坐标是否可用。

OpenCV HSV 使用 H=`0..179`、S/V=`0..255`。上述范围按《规则讲解》的目标
示意图取样后放宽，只是尚未经过现场实物验证的启动值；命题文件明确存在色差。
固定相机曝光/白平衡和实物后，必须基于分层实拍重新校准范围、覆盖率、
dominance 与去噪参数，危险类单独核对。

`raw_classes` 的位置就是模型 class ID。`config.hailo.class_mapping` 是内部的顺序元组；创建检测器时必须调用 `config.hailo.model_class_mapping()` 得到整数 ID 映射。`HailoConfig.build_backend()` 会延迟导入 HailoRT 并校验部署资产；`TargetPoseDetector` 从构造开始接管后端，构造校验失败也会关闭它，因此优先使用检测器上下文管理。

真机单图验收命令：

```bash
python manual_tests/hailo_pose.py \
  --config configs/runtime.yaml \
  --undistorted-image recordings/session_001/frames/00000001_时间戳.jpg
```

输入必须已经是与配置内参一致的去畸变图。该命令用于部署连通性检查，不替代正式数据集指标、持续运行时延和温度验收。

实时相机、去畸变、推理和叠加预览：

```bash
python manual_tests/camera_undistort_perception.py
```

窗口中的绿色框为目标框、红点为 K0，ROI 内半透明区域为当前顶部颜色候选
掩码；标签显示最终类别、HSV 候选/状态、覆盖率、dominance，启用地面映射后
还会附加机器人地面 `(x, y) mm`。

## 9. 转换为评测输入

人工标注与观测准备好后，通过适配器完成类别无关 IoU 一对一匹配。以下片段位于已取得当前 `raw_frame`、`annotations` 和 `observations` 的评测循环中：

```python
from time import monotonic_ns

from rescue_vision.perception.evaluation_adapter import (
    TargetAnnotation,
    observations_to_evaluation_records,
)

# 非空时使用检测器记录的统一结果时间；空结果时在 detect() 返回后
# 记录当前时间，使漏检样例仍有端到端时延。
result_timestamp_ns = (
    observations[0].result_timestamp_ns
    if observations
    else monotonic_ns()
)
records = observations_to_evaluation_records(
    sample_id="session_001/frame_00000001",
    annotations=annotations,  # list[TargetAnnotation]
    observations=observations,
    capture_timestamp_ns=raw_frame.timestamp_ns,
    result_timestamp_ns=result_timestamp_ns,
    iou_threshold=0.5,
    tags={"lighting": "indoor_bright"},
)
```

生产评测流水线最好在检测编排层显式保存每帧统一的结果时间，避免不同调用方各自推断。
