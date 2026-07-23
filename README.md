# 工创赛智能救援上位机视觉

2027 工创赛“智能救援”赛项的上位机视觉工程。目标平台是 Raspberry Pi 5、Hailo-8L 和 Camera Module 3 NoIR Wide。

当前完成的是可复现的视觉基础设施和任务目标感知契约，不是完整比赛程序：相机、标定、地面投影、配置、录制回放、数据集工具、离线评测、统一目标观测与 Hailo YOLO Pose 后端已经实现；训练完成的单 K0 正式模型、跟踪、定位、世界模型、规则状态机、规划、通信和应用入口尚未实现。P0 还缺四类规则目标的现场实拍数据。

## 从这里开始

| 要做的事 | 首先查看 |
| --- | --- |
| 了解代码放在哪里 | [项目结构](docs/项目结构.md) |
| 选择下一项工作 | [后续优先级](docs/后续优先级.md) |
| 理解比赛语义和安全约束 | [赛题约束与视觉需求](docs/赛题约束与视觉需求.md) |
| 标定相机或地面 | [标定说明](src/rescue_vision/calibration/README.md) |
| 录制、划分或评测数据 | [数据集与评测](docs/数据集与评测.md) |
| 执行相机采集和单会话验收 | [数据采集工具使用手册](docs/数据采集工具使用.md) |
| 到现场采集四类目标 | [目标数据采集清单](docs/目标数据采集清单.md) |
| 标注或部署任务目标模型 | [Pose 视觉模型约定](docs/Pose视觉模型约定.md) |
| 运行真机或 GUI 检查 | [人工验收脚本](manual_tests/README.md) |
| 修改仓库 | [AGENTS.md](AGENTS.md) |

官方资料位于 `docs/命题文件/`。仓库内的解释用于工程设计；发生冲突时以最新正式文件和现场通知为准。

## 当前能力

| 目录 | 状态 | 责任 |
| --- | --- | --- |
| [`camera/`](src/rescue_vision/camera/README.md) | 已实现 | 两种真机后端、统一帧、离线回放和异步记录 |
| [`calibration/`](src/rescue_vision/calibration/README.md) | 已实现 | 棋盘采集、三模型内参比较、地面映射 |
| [`geometry/`](src/rescue_vision/geometry/README.md) | 已实现 | 去畸变、坐标类型、地面与 BEV 转换 |
| [`config/`](src/rescue_vision/config/README.md) | 已实现 | schema v2 严格配置、标定一致性和模型校验和校验 |
| [`data/`](src/rescue_vision/data/README.md) | 已实现 | 记录转清单、哈希验证、按录像整组划分 |
| [`evaluation/`](src/rescue_vision/evaluation/README.md) | 已实现 | 分类、地面误差、时延和失败样例报告 |
| [`perception/`](src/rescue_vision/perception/README.md) | 已实现 | 四类目标契约、K0 投影、假后端、评测适配和 Hailo YOLO Pose 后端 |
| `manual_tests/` | 人工验收 | 相机、GUI、实际地面映射 |
| 跟踪到通信主链路 | 未实现 | 只在功能落地时创建对应目录 |

各包的最简 Python 示例和典型用法汇总见 [`src/rescue_vision/README.md`](src/rescue_vision/README.md)。

核心数据流：

```text
FrameSource → CameraFrame → CameraModel → TargetPoseDetector
                                      ↓             ↓
                              GroundProjector → TargetObservation
                                                    ↓
                               跟踪/定位/世界模型/策略（未实现）
```

坐标必须显式区分 `RawPixel`、`UndistortedPixel`、`GroundPoint`、`FieldPoint` 和 `BevPixel`。机器人地面坐标为 `x` 向前、`y` 向左、单位 mm；像素为 `u` 向右、`v` 向下。

## 安装与检查

目标环境为 Raspberry Pi OS Trixie、Python 3.13；包支持 Python 3.10 及以上。当前实现使用系统提供的 OpenCV、NumPy、Picamera2 和 libcamera：

```bash
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-opencv \
  python3-picamera2

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

HailoRT 与 ONNX Runtime 只在创建实际 Hailo 后端时需要，不是开发机测试、标定或数据工具的前置条件。串口依赖等对应模块落地后再安装。

开发机基础验证：

```bash
python -m compileall -q src tests manual_tests
python -m pytest
```

`tests/` 不依赖相机或 Hailo；需要硬件、显示器或本地标定资产的检查只放在 `manual_tests/`。

## 工具入口

| 命令 | 用途 |
| --- | --- |
| `rescue-vision-record` | 录制可回放相机会话 |
| `rescue-vision-check-recording` | 完整回放并检查单次采集健康状态 |
| `rescue-vision-manifest` | 记录目录转严格数据清单 |
| `rescue-vision-split` | 按 `recording_id` 防泄漏划分 |
| `rescue-vision-evaluate` | 生成离线评测报告 |

参数、示例和 schema 统一见[数据集与评测](docs/数据集与评测.md)。`examples/` 只是格式夹具，不能充当训练或现场验证数据。

## 关键约束

- `CameraModel` 是去畸变唯一权威；`GroundProjector` 是去畸变像素与机器人地面的唯一权威。
- 相机、分辨率、裁剪、焦点、安装姿态或 `new_K` 改变后必须重新验证相应标定。
- 实时路径只处理最新帧；录像、显示和日志使用有界旁路。
- 危险目标允许“未知/疑似危险”，不得为了总体指标降低危险类安全要求。
- 原始录像、批量图片、标定临时输出和模型权重不提交 Git。
