# `config`：运行配置与对象装配

本包是运行参数的唯一入口。`load_runtime_config()` 读取 YAML，拒绝未知字段、
错误类型和不一致资产；相对路径以 YAML 所在目录为基准。`camera` 必填，
其余段可以省略并采用安全默认值。

本机运行统一读取不提交的 `configs/runtime.yaml`。`configs/runtime.example.yaml` 只用于创建新配置：

```bash
cp configs/runtime.example.yaml configs/runtime.yaml
```

当前格式不再包含 `mission.danger_avoid_distance_mm`；旧本机配置需删除该字段，
否则严格加载会报告未知键。危险目标的实际路径检查由 20 分应用走廊规划负责。

## 常用类和函数

| 入口 | 用途 | 重要返回语义 |
| --- | --- | --- |
| `load_runtime_config(path)` | 严格读取一次 YAML | 返回不可变 `AppConfig` |
| `AppConfig.build_camera_model()` | 按内参开关创建 `CameraModel` | 内参关闭时返回 `None` |
| `AppConfig.build_geometry()` | 创建相机模型和可选地面映射 | 内参关闭时返回 `None`；只有内参时 projector 为 `None` |
| `AppConfig.build_target_pose_detector()` | 按当前 Hailo、类别和 HSV 配置创建任务目标检测器 | Hailo 关闭时返回 `None`；返回对象接管 backend 生命周期 |
| `AppConfig.build_center_cross_localizer()` | 创建中心十字绝对位姿观测器 | 定位关闭或地面映射不可用时返回 `None`；当前仓库没有场地特征模型生产者 |
| `AppConfig.build_static_field_landmark_tracker()` | 创建静态地图搜索提示和短时地标追踪器 | 不打开相机或传感器 |
| `AppConfig.build_safe_zone_corner_localizer()` | 创建安全区角点绝对位姿拟合器 | 使用同一 `world.static_map` |
| `AppConfig.build_odometry_imu_fusion()` | 创建连续二维编码器/IMU航位推算器 | 融合关闭时返回 `None`；启用时要求 motion、UART 和里程计机械标定，不强制启用视觉场地特征 |
| `AppConfig.build_simulation_20_point_sequence()` | 按 `simulation_20_point` 装配受限四绿色物资流程 | 未显式开启时拒绝；只创建纯逻辑对象，不打开硬件 |
| `UartConfig.build_channel()` | 创建协议无关 UART 行通道 | UART 关闭时返回 `None`；创建时尚不打开设备 |
| `RemoteConfig.build_server()` | 创建树莓派直接 TCP 服务端 | 远程关闭时返回 `None`；创建时尚不监听；解团 `observe_only` 入口在独立线程异步接受观察端 |
| `RemoteConfig.connect_client()` | 仓库内参考客户端建立直接 TCP 连接 | 只用于互操作/人工检查；独立电脑端不得依赖 |
| `MotionRuntimeConfig.build_controller()` | 用已有 UART 通道创建运动控制器 | motion 关闭时返回 `None`；不打开 UART |
| `MotionRuntimeConfig.build_remote_executor()` | 创建远程调试运动执行器 | 复用同一个运动控制器和限速 |
| `MotionRuntimeConfig.build_remote_gripper_executor()` | 创建远程夹爪执行器 | 夹爪禁用时返回 `None`；否则复用控制器、有效期与机械标定 |
| `HailoConfig.build_backend()` | 校验模型资产并创建 Hailo 后端 | Hailo 关闭时返回 `None` |
| `HailoConfig.model_class_mapping()` | 把模型 class ID 映射为 `TargetClass` | 直接传给 `TargetPoseDetector` |
| `PerceptionConfig.build_target_ground_geometry_estimator()` | 创建目标地面中心估计器 | 禁用时返回 `None`；启用时要求完整地面外参 |
| `TrackingConfig.build_tracker()` | 创建一轮使用的多目标跟踪器 | 初始无轨迹 |
| `WorldRuntimeConfig.build_model()` | 按 `team_color` 把物理静态地图派生为任务区域 | 队伍颜色未知时只派生场界 |
| `MissionConfig.build_state_machine()` | 创建一轮使用的规则状态机 | 初始为 `WAIT_START` |

主要配置 dataclass：

| 类 | 常用字段 |
| --- | --- |
| `CameraConfig` | `backend`、`image_size`、`fps`、`lens_position` |
| `GeometryConfig` | 内参与地面映射各自的开关和路径 |
| `RecordingConfig` | `queue_capacity`、`image_format` |
| `ProcessingConfig` | `max_observation_age_ms` |
| `UartConfig` | 设备名、波特率、读写超时、有界接收容量和最大行长度 |
| `RemoteConfig` | 服务端/客户端、观察/调试权限、连接/IO 超时和有界队列 |
| `MotionRuntimeConfig` | 实测轮距、车体/车轮速度上限、单轮加速度上限、远程命令有效期、`odometry`、`gripper` 和解团试验参数 |
| `OdometryRuntimeConfig` | 每圈计数、左右有效轮径和校准后机器人 `gyro_z` 极性 |
| `GripperRuntimeConfig` | 能力开关、左右开/闭安全角度、角度和与固定速度全行程时间 |
| `ClusterBreakupRuntimeConfig` | 定距出发、左右搜索方向、居中、接近、张爪冲散、张爪退出、停车合爪、闭爪退离和绿色扫描参数 |
| `Simulation20PointRuntimeConfig` | 预推/走廊、扫描、接触、交付确认、退离和重复解团上限；目标交付数固定为 4 |
| `TrackingConfig` | 关联、确认、滑行、衰减和删除阈值 |
| `WorldRuntimeConfig` | 世界阈值、`TeamColor`、`StaticFieldMap` 及任务区域派生 |
| `MissionConfig` | 比赛计时、安全超时和目标优先级；路径避障由应用规划器按实际走廊负责 |
| `PerceptionConfig` | 目标检测/K0、ROI HSV 和目标地面几何参数 |
| `HsvColorClassifierConfig` | 四类 HSV 闭区间、颜色证据门槛和掩码去噪参数 |
| `TargetGroundGeometryConfig` | 四类三维形状尺寸、搜索步长、评分权重和接受门限 |
| `CenterCrossLocalizerConfig` | 终端射线关联、锚点置信度、先验创新和不确定度下限 |
| `LocalizationRuntimeConfig` / `FusionConfig` | 中心十字配置，以及起点、三轴 IMU 温度/矩阵校准、噪声、时效、物理跳变、视觉门控和历史长度 |
| `StaticLandmarkTrackingConfig` / `SafeZoneCornerLocalizerConfig` | 地图搜索范围、三帧重捕获、角点基线/残差和不确定度下限 |
| `HailoConfig` | 模型资产、身份、类别映射和后端粗筛阈值 |
| `RuntimeGeometry` | `camera_model`、可选 `ground_projector` |

## 1. 加载运行配置

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
```

`config` 是不可变 `AppConfig`。进程启动时读取一次，后续各装配片段复用它；
不要在逐帧循环中反复解析 YAML。

## 2. 装配通信对象

以下片段承接前文 `config`，用于树莓派服务端进程（`remote.role: server`）。
创建对象不等于打开设备或监听端口：

```python
# 尚未打开 UART。
uart_channel = config.uart.build_channel()

# 尚未绑定 TCP 监听端口。
remote_server = config.remote.build_server()

# motion 复用同一个 UART 通道；禁用时返回 None。
motion_controller = config.motion.build_controller(uart_channel)
motion_executor = config.motion.build_remote_executor(motion_controller)
gripper_executor = config.motion.build_remote_gripper_executor(
    motion_controller
)

# 只创建纯逻辑融合状态；不会再打开一个串口读取器。
localization_fusion = config.build_odometry_imu_fusion()
```

UART/TCP 生命周期和 motion 的停止语义分别见相邻模块 README，不要把
`build_*()` 的成功返回误认为硬件已经连通。
`gripper_executor is None` 表示当前配置未完成机械标定或明确关闭夹爪能力；
应用必须在会话状态中声明 `gripper_control=false`，不能自行补默认角度。
`motion.gripper` 字段语义如下：

| 字段 | 语义 |
| --- | --- |
| `enabled` | 是否允许装配并宣告远程夹爪能力 |
| `open_left_angle_deg` / `open_right_angle_deg` | 当前车辆完全张开时的左右安全目标角度 |
| `closed_left_angle_deg` / `closed_right_angle_deg` | 当前车辆完全闭合时的左右安全目标角度 |
| `transport_left_angle_deg` / `transport_right_angle_deg` | 单个物块收入后的局部打开运输姿态；必须严格位于开/闭端点之间 |
| `full_travel_time_s` | 单方向持续按下时，从一个端点匀速到另一端点的时间 |
| `angle_sum_deg` | 舵机联动约束的左右角度和；开、闭端点及运行中目标都必须满足该值 |

四个角度单位都是 degree，范围 `[0, 180]`。每组端点必须满足
`left + right = motion.gripper.angle_sum_deg`。加载器会拒绝缺少角度和、
不满足该和约束或开闭端点相同的配置。执行器以左角为单一自由度并始终用
`right = motion.gripper.angle_sum_deg - left` 构造远程下发；
客户端只发送按下/松开 boolean，不读取这些机械值。
20 分模拟赛入口在 `APPROACH_GREEN` 及尚未通过接触确认的 `ENGAGE_GREEN` 下发
运输局部打开姿态，接触和 mission 门禁通过后下发闭合端点；现有解团和远程扳机流程
不会自动切换到运输姿态。使用前需要在真车上确认单个物块不会滑落、夹持或形成违规抓取。

`motion.odometry` 是编码器机械量和 `gyro_z` 极性的唯一配置；轮距继续复用
`motion.wheel_track_m`，定位配置不得复制。IMU 三轴温度零偏、比例/交叉轴矩阵和
传感器到机器人系的安装旋转统一配置在 `localization.fusion.imu_calibration`，
融合入口按 v3 协议规定的顺序应用一次。
`localization.fusion.initial_pose` 使用固定物理 `FieldPoint` 和全局航向，只在进程首个
有效遥测基线使用一次。`gyro_z_sign` 只能为 `1` 或 `-1`，用于把完成三轴校准和安装
旋转后的机器人 `gyro_z` 转换为定位内部“左转为正、右转为负”；本车若实测左转为负
应设为 `-1`。bias 与温度系数必须是有限三向量，比例/交叉轴矩阵必须有限且可逆，
安装旋转必须是有限正交 3×3 右手旋转。模板的零值和单位阵不是实车标定结果。
`localization.fusion.max_interpolated_overrun_samples` 控制是否允许单帧插值恢复；
当前只允许 `0` 或 `1`，默认 `1` 表示单次双编码器有效异常会等待下一帧并插值 IMU。
若下一帧仍为 overrun，`allow_wheel_only=true` 且双编码器/时间/序号/轮速均合法时，
两段改用纯轮式并放大协方差；连续第三帧仍清除连续位姿。
轮速合法性使用协议固定 10 ms 遥测释放周期；overrun 帧的 IMU 时间戳会重复，UART
接收也可能成批到达，两者都不能作为该帧编码器轮速的分母。
`motion.odometry.max_consecutive_overrun_samples` 只约束解团/20 分流程中
`EncoderTravelTracker` 的定距积分：允许的连续采样超期（`SAMPLE_OVERRUN`）帧数，
超出即抛错停车；`null` 关闭该中止，仅保留计数诊断。默认 `1` 保持原行为；
它不改变融合的单帧插值、两帧纯轮式和三帧中止策略。

## 3. 装配几何对象

```python
# build_geometry() 会同时检查运行分辨率、标定可用性、
# 相机模型和地面映射中的 calibration_id。
geometry = config.build_geometry()
if geometry is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用内参")

camera_model = geometry.camera_model
ground_projector = geometry.ground_projector  # 尚无地面映射时为 None

target_ground_geometry_estimator = (
    config.perception.build_target_ground_geometry_estimator(
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=ground_projector,
    )
)
```

`target_ground_geometry.enabled=false` 时最后一项为 `None`；启用时缺少地面映射
或完整相机外参会立即报错，不会退化成检测框中心。

## 4. 装配 Hailo 后端

只有真正需要推理时才创建 Hailo 设备。以下片段仍承接同一个 `config`：

```python
# 只有真正需要推理时才创建 Hailo 设备。
backend = config.hailo.build_backend()
if backend is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用 Hailo")

class_mapping = config.hailo.model_class_mapping()
color_classifier = config.perception.color_classifier
```

`backend` 必须由调用方关闭；正常生产路径通常立即交给
`TargetPoseDetector`，再由检测器上下文统一释放。

需要把 detector 交给远程 perception 图传旁路时，优先使用同一 `AppConfig` 的
装配方法；它不会在 Hailo 关闭时偷偷创建测试后端：

```python
detector = config.build_target_pose_detector()
if detector is None:
    raise RuntimeError("当前配置未启用 Hailo，不能提供 perception 图传")

try:
    # `frame` 是上游 FrameSource 已准备好的 CameraFrame；
    # PerceptionFrameRenderer 或其他调用方在这里消费 detector。
    observations = detector.detect_realtime(frame, frame.image_bgr)
finally:
    detector.close()
```

实时远程发布应把 detector 放入有界最新帧旁路，不要在运动安全循环中同步调用
推理；`perception.PerceptionFrameRenderer` 已提供该生命周期和降级语义。

## 5. 装配纯逻辑对象

```python
# 纯逻辑对象同样只装配一次；每轮结束后显式 reset，
# 或为下一轮创建新对象，不能在逐帧循环中反复构造。
tracker = config.tracking.build_tracker()
world_model = config.world.build_model()
mission = config.mission.build_state_machine()
```

配置对象和已构建对象应在进程生命周期内复用，不能在逐帧循环中重新读取
YAML、重新生成去畸变映射或重复创建 Hailo 设备。

中心十字绝对位姿的纯几何消费方仍可单独装配；当前仓库没有场地特征模型
推理生产者，因此运行时不能形成从图像到该定位器的完整链路：

```python
center_cross_localizer = config.build_center_cross_localizer(
    ground_projector=ground_projector,
)
static_landmark_tracker = config.build_static_field_landmark_tracker()
safe_zone_corner_localizer = config.build_safe_zone_corner_localizer()
```

`build_center_cross_localizer()` 要求 `localization.enabled=true` 和可用地面
映射，任一条件缺失时返回 `None`。定位器从 `config.world.static_map` 读取
中心十字交点和终端方向，不创建另一套地图、标定或 BEV。其输入
`FieldFeatureDetectionResult` 必须由未来的场地特征模型后端产生；在该后端
实现前，不要把合成观测当作现场检测结果。

## 6. 几何开关组合

```yaml
geometry:
  intrinsics_enabled: true
  intrinsics_path: ../src/rescue_vision/calibration/output/内参目录/selected_calibration.json
  ground_mapping_enabled: false
  ground_mapping_path: null
```

| 内参 | 地面映射 | 结果 |
| --- | --- | --- |
| 关 | 关 | 原图采集或非几何诊断；`build_geometry()` 返回 `None` |
| 开 | 关 | 可去畸变、录制目标数据和运行图像检测 |
| 开 | 开 | 可进一步把 K0 投影到机器人地面毫米坐标 |
| 关 | 开 | 非法配置，加载阶段直接拒绝 |

正式四类目标采集和感知必须启用内参。地面映射尚未完成时保持关闭，不应
伪造路径或绕过 `calibration_id` 配对检查。

## 7. Hailo 装配语义

`hailo.enabled: false` 时，导入配置和运行无硬件测试不会导入 HailoRT。启用后，`build_backend()` 才会：

1. 检查 HEF、ONNX 后处理和张量映射文件；
2. 使用 `raw_classes` 数量和推理阈值创建后端；
3. 打开 Hailo 设备资源。

后端必须由调用方 `close()`，通常交给 `TargetPoseDetector` 的上下文管理统一释放。

`hailo.backend_score_threshold` 是后端粗筛，必须小于等于
`perception.detection_threshold`；后者决定模型框是否进入统一观测。
`perception.k0_threshold` 独立控制接触点是否可用。最终任务类别不再由
模型分类分数决定，而由 `perception.color_classifier` 对检测框 ROI 进行
HSV 分类；颜色覆盖不足或多色歧义时输出 `unknown`。

## 8. ROI HSV 配置

OpenCV HSV 的 H 范围为 `0..179`，S/V 为 `0..255`。每个任务类别可配置一个
或多个闭区间；橘色跨 Hue 边界，因此示例使用两个区间。不同类别的区间必须
互不重叠，避免一个像素被解释为多个任务类别。

```yaml
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

这些 HSV 范围由官方命题示意图取样后放宽，只是尚未经过现场验证的启动值。
固定相机、曝光/白平衡、光照和实物后，应通过实拍数据同时校准 HSV 范围、
覆盖率、dominance 和去噪参数；危险类必须单独报告失败样例。

## 9. 目标地面几何配置

`perception.target_ground_geometry.objects` 必须完整列出四类。盒体使用
`length_mm / width_mm / height_mm`，正四面体使用 `edge_mm`；尺寸改变只改
配置，不改估计代码：

```yaml
perception:
  target_ground_geometry:
    enabled: false
    objects:
      green_supply:
        shape: box
        length_mm: 40.0
        width_mm: 40.0
        height_mm: 40.0
      black_core:
        shape: regular_tetrahedron
        edge_mm: 40.0
      orange_injured:
        shape: box
        length_mm: 80.0
        width_mm: 40.0
        height_mm: 40.0
      blue_danger:
        shape: box
        length_mm: 40.0
        width_mm: 40.0
        height_mm: 40.0
```

完整粗搜索/细化步长、拟合权重、接触残差、中心歧义和接受门限见
`configs/runtime.example.yaml`。三项评分权重之和必须为 1；现场修改尺寸后
需要重新验证各距离和角度下的中心误差。

## 10. UART 装配语义

```yaml
uart:
  enabled: true
  device: /dev/serial0
  read_timeout_ms: 100.0
  write_timeout_ms: 100.0
  receive_queue_capacity: 256
```

`build_channel()` 返回协议无关的 `UartFrameChannel`；进入上下文时才导入
PySerial 并打开设备。230400 8N1 和 64-byte COBS 解码边界由冻结协议直接
确定，不在 YAML 重复配置；消息类型、CRC 和字段布局同样不能塞进运行配置。
完整生命周期和故障语义见
[`communication` README](../communication/README.md)。

## 11. 远程通信装配语义

`remote.role: server` 用于树莓派监听，`client` 只用于本仓库参考客户端和
双机人工检查。独立电脑端维护自己的配置和协议实现，不读取本仓库 YAML 或
导入 Python 包。
远程协议有意采用 IP 与端口直接连接，不使用 PSK、认证握手、HMAC 或密钥
文件。`access_mode` 只有：

- `observe_only`：禁止电脑提交控制，适用于正式比赛图传/地图观察；
- `debug_control`：预留运动和录制控制，只用于赛外采集和调试。

完整配置、消息 topic、资源生命周期和双机检查见
[`communication` README](../communication/README.md) 和
[电脑端通信协议](../../../docs/电脑端通信协议.md)。`build_server()` 不打开
网络；`connect_client()` 会立即建立 TCP 连接，因此只能在本仓库参考工具
的装配层调用。

## 默认值与路径

- `geometry`、`uart`、`remote`、`motion`、`motion.gripper` 和 `hailo`
  缺省时全部关闭；远程访问缺省为 `observe_only`。
- `perception` 可以省略；此时目标三维拟合安全关闭，ROI HSV
  分类及检测门限采用 `runtime.example.yaml` 展示的初始值。现场颜色、尺寸和
  接受门限仍应在 YAML 中明确覆盖并重新标定。
- 录制队列、通信超时、运动上限、跟踪和任务阈值缺省为
  `runtime.example.yaml` 展示的值。现场只需写需要覆盖的字段。
- 子系统一旦启用，设备路径、轮距、夹爪机械端点、标定路径和模型资产仍然
  必须完整有效，不会猜测这些安全关键参数。
- `detection_threshold` 和 `k0_threshold` 位于 `perception`；模型类别只保留
  为诊断证据，最终类别由 ROI HSV 决定。
- 位于 `configs/` 的 YAML 指向仓库根目录资产时通常以 `../` 开头。
- 路径、类别和阈值只在配置中维护，不在业务模块再次硬编码。

新增配置字段时必须同步校验、默认值、`configs/runtime.example.yaml`、
无硬件测试和受影响模块 README。
