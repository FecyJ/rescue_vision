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
| [`calibration`](src/rescue_vision/calibration/README.md) | 已实现 | 棋盘采集、三模型内参比较和地面映射 |
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
