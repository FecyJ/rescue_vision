# 工创赛智能救援上位机视觉

2027 工创赛“智能救援”赛项的上位机视觉工程，目标平台为 Raspberry Pi 5、Hailo-8L 和 Camera Module 3 NoIR Wide。

当前已完成相机、标定与地面几何、严格配置、录制回放、数据集工具、离线评测、统一目标观测和 Hailo YOLO Pose 后端；尚未完成正式四类目标模型、跟踪、定位、世界模型、规则状态机、规划、通信和应用入口。这不是可直接参赛的完整程序。

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

HailoRT 与 ONNX Runtime 仅在创建真实 Hailo 后端时需要；开发机测试、标定和数据工具不依赖它们。

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
| [`geometry`](src/rescue_vision/geometry/README.md) | 已实现 | 去畸变、显式坐标类型、地面与 BEV 转换 |
| [`config`](src/rescue_vision/config/README.md) | 已实现 | schema v2 配置、标定一致性和模型身份校验 |
| [`data`](src/rescue_vision/data/README.md) | 已实现 | 记录检查、清单生成和按会话防泄漏划分 |
| [`evaluation`](src/rescue_vision/evaluation/README.md) | 已实现 | 分类、地面误差、时延和失败样例报告 |
| [`perception`](src/rescue_vision/perception/README.md) | 已实现基础设施 | 四类目标契约、K0 投影、假后端和 Hailo 后端；正式模型待训练 |
| 跟踪至通信主链路 | 未实现 | 只在真实实现和测试落地时创建模块 |

各包最简 Python 示例见 [`src/rescue_vision/README.md`](src/rescue_vision/README.md)。

## 核心数据流

```text
FrameSource → CameraFrame → CameraModel → TargetPoseDetector
                                      ↓             ↓
                              GroundProjector → TargetObservation
                                                    ↓
                               跟踪/定位/世界模型/策略（未实现）
```

- 像素必须区分 `RawPixel` 与 `UndistortedPixel`；地面点使用 `GroundPoint`，单位 mm。
- `CameraModel` 是去畸变唯一权威；`GroundProjector` 是去畸变像素与机器人地面的唯一权威。
- 实时路径只处理最新帧；录像、显示和日志使用有界旁路。
- 危险目标允许 `unknown`/疑似危险，不得用总体指标掩盖危险类漏检。
- 原始录像、批量图片、标定临时输出、正式数据集和模型权重不提交 Git。

## 命令行工具

| 命令 | 用途 |
| --- | --- |
| `rescue-vision-record` | 录制可回放相机会话 |
| `rescue-vision-check-recording` | 检查单次采集的完整性、帧率、丢帧和元数据 |
| `rescue-vision-manifest` | 记录目录转严格数据清单 |
| `rescue-vision-split` | 按 `recording_id` 整组划分 |
| `rescue-vision-evaluate` | 生成离线评测报告 |

采集命令与现场步骤统一见[数据采集手册](docs/数据采集工具使用.md)，格式定义见[数据集与评测 schema](docs/数据集与评测.md)。`examples/` 仅是合成格式夹具，不能作为实拍或性能证据。
