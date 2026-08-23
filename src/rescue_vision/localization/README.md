# `localization`：中心十字视觉位姿观测

本包把同一帧的中心十字、安全区和低精度场界观测转换为场地坐标中的机器人
位姿候选。它只负责低频几何视觉观测，不读取相机、串口、编码器或 IMU，不
维护跨帧状态，也不直接修改世界模型。

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

示例中的 `record_localization_diagnostic()` 和 `submit_visual_correction()` 由未来
比赛应用或融合层提供，不属于本包。当前仓库尚未实现 IMU/编码器融合。

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

后续短时跨帧关联应放在融合层：由它按时间对齐历史锚点和连续推算状态，再复用
本包的终端语义及 `FieldPose2D`，不能把状态缓存塞回 `FieldFeatureDetector`。
