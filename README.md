# 工创赛智能救援上位机视觉

2027 工创赛“智能救援”赛项的上位机视觉工程，目标平台为 Raspberry Pi 5、Hailo-8L 和 Camera Module 3 NoIR Wide。

当前已完成相机、标定与地面几何、严格配置、录制回放、数据集工具、离线评测、协议无关 UART、直接 TCP 远程消息通道、Rescue Car v2.0 差速与双舵机夹爪控制、受监督手动驾驶采集入口、统一目标观测、Hailo YOLO Pose 后端、四类目标传统视觉地面几何估计基线、传统视觉场地特征观测基线，以及可用合成事件运行的目标跟踪、最小世界模型和规则状态机；尚未完成正式四类目标模型、定位、完整区域/对手感知、规划、脚本运动采集、固件失联看门狗闭环、真实接触/交付证据适配和比赛应用入口。这不是可直接参赛的完整程序。

## 快速上手

### 1. 安装

目标环境为 Raspberry Pi OS Trixie、Python 3.13；代码支持 Python 3.10 及以上。

```bash
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-opencv \
  python3-picamera2

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

HailoRT 与 ONNX Runtime 仅在创建真实 Hailo 后端时需要；PySerial 仅在创建真实 UART 通道时访问设备。开发机测试、标定和数据工具不依赖相应硬件。

### 2. 验证开发环境

```bash
python -m compileall -q src tests manual_tests
python -m pytest
```

`tests/` 不访问相机或 Hailo。需要相机、显示器、Hailo 或本地标定资产的检查在 [`manual_tests/`](manual_tests/README.md)。

### 3. 选择当前工作

| 目的 | 从这里开始 |
| --- | --- |
| 相机已经接好，准备采集数据 | [数据采集手册](docs/数据采集工具使用.md) |
| 采集棋盘、求内参或地面映射 | [标定说明](src/rescue_vision/calibration/README.md) |
| 标注或部署四类目标模型 | [Pose 模型约定](docs/Pose视觉模型约定.md) |
| 对接独立 PyQt/手柄采集客户端 | [电脑端通信协议交接](docs/电脑端通信协议.md) |
| 开发新模块 | [项目结构](docs/项目结构.md) → [后续优先级](docs/后续优先级.md) |
| 理解比赛类别和安全规则 | [赛题约束与视觉需求](docs/赛题约束与视觉需求.md) |
| 查数据或评测 JSONL 格式 | [数据集与评测 schema](docs/数据集与评测.md) |
| 修改仓库 | [AGENTS.md](AGENTS.md) |

官方资料位于 `docs/命题文件/`。项目文档用于工程实现；冲突时以最新正式文件和现场通知为准。

## 当前模块

| 模块 | 状态 | 责任 |
| --- | --- | --- |
| [`camera`](src/rescue_vision/camera/README.md) | 已实现 | 真机最新帧、离线回放和有界异步记录 |
| [`calibration`](src/rescue_vision/calibration/README.md) | 已实现 | 棋盘/ChArUco 采集、三模型内参比较和固定机器人多位置地面映射 |
| [`geometry`](src/rescue_vision/geometry/README.md) | 已实现 | 去畸变、显式坐标类型、地面/三维点投影与 BEV 转换 |
| [`config`](src/rescue_vision/config/README.md) | 已实现 | 安全默认配置、UART/远程/motion/夹爪机械标定/几何/模型和感知算法装配 |
| [`communication`](src/rescue_vision/communication/README.md) | 已实现基础设施 | UART、直接 TCP 远程消息、raw/perception 图传模式选择、运动/夹爪/采集严格 schema 和有界队列；比赛发布器待接入 |
| [`motion`](src/rescue_vision/motion/README.md) | 已实现基础设施 | 差速运动、持续扳机双舵机夹爪、单轮加速度限制、Rescue Car 协议解析和远程超时保护 |
| [`app`](src/rescue_vision/app/README.md) | 已实现手动采集入口 | 赛外受监督驾驶、持续夹爪控制、可选择 raw/perception 图传、采集控制与状态装配；比赛入口待实现 |
| [`data`](src/rescue_vision/data/README.md) | 已实现 | 记录检查、清单生成和按会话防泄漏划分 |
| [`evaluation`](src/rescue_vision/evaluation/README.md) | 已实现 | 分类、地面误差、时延和失败样例报告 |
| [`perception`](src/rescue_vision/perception/README.md) | 已实现基础设施 | Pose 框/K0、ROI HSV 分类分割、四类可配置三维模板地面中心估计，以及安全区、无编号出发区、中心十字和低精度边界候选；正式模型、实物精度与树莓派性能待验证 |
| [`tracking`](src/rescue_vision/tracking/README.md) | 已实现纯逻辑 | 时间关联、轨迹确认、短时遮挡、衰减和删除 |
| [`world`](src/rescue_vision/world/README.md) | 已实现纯逻辑 | 静态区域、动态目标、危险状态、对手占据和不确定性 |
| [`mission`](src/rescue_vision/mission/README.md) | 已实现纯逻辑 | 首次/容量/伤员/危险规则、安全降级和抽象动作 |
| 定位至比赛应用主链路 | 未实现 | 定位、真实区域/接触证据、规划、正式动作到运动控制的适配和比赛入口待完成 |

各包常用 API、命令和实际对接示例见 [`src/rescue_vision/README.md`](src/rescue_vision/README.md)。

## 核心数据流

```text
FrameSource → CameraFrame → CameraModel → 去畸变帧
                                      ├─> TargetPoseDetector
                                      │          └─> TargetObservation
                                      │                     ├─> TargetGroundGeometryEstimator
                                      │                     │          └─> TargetGroundGeometry
                                      │                     ↓
                                      │          MultiTargetTracker
                                      │                     ↓
                                      │          WorldModel / MissionStateMachine
                                      └─> FieldFeatureDetector
                                                 └─> FieldFeatureDetectionResult
                                                              ↓
                                                     定位（P4，未实现）

RemoteMessageConnection → DebugMotionCommand → RemoteMotionExecutor
                                                ↓
                                      MotionController → UART

RemoteMessageConnection → DebugGripperCommand → RemoteGripperExecutor
                                                 ↓
                                      MotionController → UART
```

- 像素必须区分 `RawPixel` 与 `UndistortedPixel`；地面点使用 `GroundPoint`，单位 mm。
- `CameraModel` 是去畸变唯一权威；`GroundProjector` 是去畸变像素与机器人地面/三维投影的唯一权威。
- 实时路径只处理最新帧；录像、显示和日志使用有界旁路。
- 危险目标允许 `unknown`/疑似危险，不得用总体指标掩盖危险类漏检。
- 规则状态机只消费显式世界、接触、交付和安全证据；当前真实证据提供者尚未完成。
- 场地特征检测只输出去畸变像素和可选机器人地面观测，不在定位完成前伪造 `FieldPoint` 或直接修改世界模型。
- 目标地面几何估计保留 K0 接触锚点和中心的语义区别；拟合不充分时中心为 `None`，不会用检测框中心兜底。
- 原始录像、批量图片、标定临时输出、正式数据集和模型权重不提交 Git。

## 坐标系约定

以下定义是当前仓库的统一约定。图像尺寸一律写作 `(width, height)`；NumPy
数组形状一律为 `(height, width, channels)`。任何跨模块传递的点都应使用带
坐标语义的类型，不能把没有说明坐标系的 `(u, v)` 或 `(x, y)` 当作通用点。

### 公共坐标类型

| 类型/坐标系 | 原点与轴方向 | 单位 | 主要用途 |
| --- | --- | --- | --- |
| `RawPixel(u, v)` 原始像素系 | 原始畸变图左上角为原点；`u` 向右增大，`v` 向下增大 | 像素 | 相机原始帧、地面标定采点输入；不能直接用于地面投影 |
| `UndistortedPixel(u, v)` 去畸变像素系 | 全尺寸 `new_K` 去畸变图左上角为原点；`u` 向右增大，`v` 向下增大 | 像素 | 检测框、K0、场地特征和 `GroundProjector` 的图像侧输入；不裁剪、不改变尺寸 |
| `RobotPoint3D(x, y, z)` 机器人三维系 | 原点为两驱动轮接地点连线的中点；`x` 向前，`y` 向左，`z` 向上 | mm | 完整外参下的离地目标、相机射线与已知高度平面求交 |
| `GroundPoint(x, y)` 机器人地面系 | `RobotPoint3D` 的 `z = 0` 平面，原点仍为两驱动轮接地点中点；`x` 向前，`y` 向左 | mm | K0 接触点、目标地面几何、地面特征和局部跟踪；这是机器人相对坐标，不是场地全局坐标 |
| `BevPixel(u, v)` 鸟瞰图像素系 | BEV 图左上角为原点；`u` 向右，`v` 向下；图像上方是机器人前方，左侧是机器人左方 | 像素 | 按 `BevConfig` 从机器人地面系生成的局部鸟瞰图 |
| `FieldPoint(x, y)` 场地全局系 | 原点为场地中心十字点划线交点；`x` 沿水平点划线向右，`y` 沿竖直点划线指向红色安全区 | mm | 定位后的机器人/目标位置、静态区域和对手占据多边形 |
| `MapPixel(u, v)` 场地图像素系 | 场地图 PNG 左上角为原点；`u` 向右，`v` 向下；不是相机像素或 `BevPixel` | 像素 | `MapSnapshotAttributes` 与 `FieldPoint` 之间的显示映射 |

标定内部还使用 OpenCV 相机三维系：原点在相机光心，`x` 向图像右方、`y`
向图像下方、`z` 沿光轴向前，单位 mm。它没有单独的公共点类型，只出现在
地面标定的物理外参与诊断中，变换约定为
`p_camera = R_robot_to_camera @ p_robot + t_robot_to_camera`。
`CameraModel` 的 `K`/`new_K` 是内参矩阵，不代表又增加了一套像素坐标轴。

### 映射关系

```text
RawPixel
    │ CameraModel.undistort
    ▼
UndistortedPixel ── GroundProjector（z=0）──↔ GroundPoint ──↔ BevPixel
    │
    ├─ 已知 z + 完整物理外参 ──↔ RobotPoint3D
    │
    └─ Hailo letterbox（内部临时）↔ 模型输入像素

GroundPoint ── 定位（当前未实现）──> FieldPoint ──↔ MapPixel
```

- `CameraModel` 是 `RawPixel → UndistortedPixel` 的唯一实现；`GroundProjector`
  是去畸变像素与机器人地面/三维投影以及地面与 BEV 转换的唯一实现。
- `pixel_to_ground()` 只表示与机器人地面 `z=0` 的交点。目标顶部、围栏顶部等
  离地点必须在已知高度时使用完整外参求 `RobotPoint3D`，不能强行使用地面单应性。
- BEV 的范围和分辨率来自 `BevConfig`。对 `GroundPoint(x, y)`，当前实现的
  像素映射为 `u_bev = (y_max - y) / mm_per_pixel`、
  `v_bev = (x_max - x) / mm_per_pixel`；因此 BEV 左上角对应
  `(x_max, y_max)`，不是机器人坐标原点。
- `FieldPoint` 的零点和方向固定对应官方《规则讲解》场地图（第 37 页）：中心
  十字点划线交点为原点，`+x` 沿水平点划线向右，`+y` 沿竖直点划线指向红色
  安全区；红蓝方抽签不改变这个物理坐标方向。
- `FieldPoint` 与 `GroundPoint` 不能直接互换。定位尚未实现时，世界模型可以
  保留缺失的 `FieldPoint`，但不能把当前机器人局部地面点伪装成场地全局点。

### 内部和显示侧的局部像素

- Hailo Pose 先把去畸变全尺寸图等比例缩放并居中填充为模型输入尺寸。模型
  输出的框和 K0 会由 `LetterboxTransform` 反变换回
  `UndistortedPixel`；模型输入像素只在推理后端内部存在，不能作为观测输出。
- `UndistortedBoundingBox.x_min/y_min/x_max/y_max` 仍是全尺寸
  `UndistortedPixel` 的水平矩形边界，其中 `x` 对应 `u`、`y` 对应 `v`。
  `RoiColorSegmentation.mask` 则是该框左上角为原点的局部数组，访问顺序为
  `mask[v_roi, u_roi]`，前景为 `255`、背景为 `0`；映射回整图时使用
  `u = x_min + u_roi`、`v = y_min + v_roi`。
- 通信中的 raw/perception JPEG 使用 `raw_pixel` 或 `undistorted_pixel` 标记，
  后者必须携带匹配的 `calibration_id`；两种图像都遵循左上原点、`u` 右、`v`
  下。场地图 PNG 使用 `MapPixel`，通过场地范围映射到 `FieldPoint`，它不是
  相机像素，也不是 `BevPixel`：

  ```text
  u_map = (x - field_min_x_mm) / (field_max_x_mm - field_min_x_mm) * (width - 1)
  v_map = (field_max_y_mm - y) / (field_max_y_mm - field_min_y_mm) * (height - 1)
  ```

完整类定义和投影 API 见 [`geometry` README](src/rescue_vision/geometry/README.md)；
标定、Pose 标注和场地图通信的专项约束分别见
[`calibration` README](src/rescue_vision/calibration/README.md)、
[`Pose视觉模型约定`](docs/Pose视觉模型约定.md) 和
[`电脑端通信协议`](docs/电脑端通信协议.md)。

## 命令行工具

| 命令 | 用途 |
| --- | --- |
| `rescue-vision-record` | 录制可回放相机会话 |
| `rescue-vision-check-recording` | 检查单次采集的完整性、帧率、丢帧和元数据 |
| `rescue-vision-manifest` | 记录目录转严格数据清单 |
| `rescue-vision-manual-capture` | 受监督手动驾驶与车载运动采集 |
| `rescue-vision-split` | 按 `recording_id` 整组划分 |
| `rescue-vision-evaluate` | 生成离线评测报告 |

采集命令与现场步骤统一见[数据采集手册](docs/数据采集工具使用.md)，格式定义见[数据集与评测 schema](docs/数据集与评测.md)。`examples/` 仅是合成格式夹具，不能作为实拍或性能证据。
