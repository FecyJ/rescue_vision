# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

2027 工创赛"智能救援"赛项的上位机视觉工程。目标平台为 Raspberry Pi 5 + Hailo-8L + Camera Module 3 NoIR Wide,目标 Python 3.13(代码支持 3.10+)。

**[AGENTS.md](AGENTS.md) 是本仓库的权威协作规范,修改代码前必须遵守。** 它包含按任务加载文档的对照表、规则安全约束、提交规范和禁止事项;本文件只提供命令和架构入口,不重复其细节。

## 常用命令

```bash
# 安装(Raspberry Pi OS 上先 apt 安装 python3-numpy/opencv/picamera2,再建 venv)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'

# 交付前最低验证(AGENTS.md 要求)
python -m compileall -q src tests manual_tests
python -m pytest

# 运行单个测试文件 / 单个用例
python -m pytest tests/test_tracking.py
python -m pytest tests/test_mission.py -k first_delivery
```

- `tests/` 全部可在无相机、无 Hailo 的开发机运行;pytest 配置在 `pyproject.toml`(`pythonpath = ["src"]`,`testpaths = ["tests"]`)。
- 需要真机、显示器或本地标定资产的人工验收在 `manual_tests/`(不参与 pytest 收集,脚本不得在收集时导入 `picamera2` 或打开窗口)。典型联调:`python manual_tests/camera_undistort_perception.py --config configs/runtime.yaml`。
- CLI 入口(定义在 `pyproject.toml [project.scripts]`):`rescue-vision-record`、`rescue-vision-check-recording`、`rescue-vision-manifest`、`rescue-vision-split`、`rescue-vision-evaluate`。现场采集流程见 `docs/数据采集工具使用.md`。

## 架构大图

核心数据流(逐帧、单向):

```text
FrameSource → CameraFrame → CameraModel(去畸变) → TargetPoseDetector
                                        ↓                ↓
                                GroundProjector → TargetObservation
                                                       ↓
                                            MultiTargetTracker
                                                       ↓
                                          WorldModel / WorldSnapshot
                                                       ↓
                                     MissionStateMachine / AbstractAction
                                                       ↓
                                       定位/规划/通信(未实现)
```

各责任的权威文件定位表在 `docs/项目结构.md`;各包公共 API 和生产装配示例在 `src/rescue_vision/README.md` 及各包相邻 `README.md`。

### 关键不变量(跨模块阅读才能发现,务必维持)

- **坐标必须带语义类型**:`RawPixel → UndistortedPixel → GroundPoint ↔ BevPixel`(定义在 `geometry/types.py`),禁止无语义 `(u, v)` 跨模块传递。机器人地面系 `x` 向前、`y` 向左、`z` 向上,单位 mm;像素 `u` 向右、`v` 向下。图像尺寸统一 `(width, height)`,NumPy shape 为 `(height, width, channels)`。
- **单一权威**:`CameraModel`(`geometry/camera_model.py`)是去畸变唯一权威,`GroundProjector` 是去畸变像素↔机器人地面唯一权威;其他模块不得复制标定矩阵。规则集中在 `mission/state_machine.py`,不散落在检测器。
- **装配只在配置层发生**:`config/runtime.py` 的 `load_runtime_config()` 加载严格校验的 `configs/runtime.yaml`(schema v4,本机文件不提交;模板为 `configs/runtime.example.yaml`),由 `AppConfig.build_geometry()` / `hailo.build_backend()` 装配几何与推理后端。算法模块依赖 `FrameSource`、推理后端等抽象,不直接创建 Picamera2、Hailo 或串口对象。
- **观测是唯一输出契约**:检测器只产出带帧号、时间、坐标系、置信度和质量标记的 `TargetObservation`,不修改定位或全局状态。跟踪器消费同帧观测;世界模型消费轨迹和外部证据;状态机只消费世界快照和显式接触/交付/安全事件。
- **实时路径只处理最新帧**;录像、显示、日志和通信走有界旁路(如 `recording.queue_capacity` 满则丢弃),不积累无界队列。
- **实现状态必须如实**:定位、正式四类目标模型、区域/对手感知、规划、通信和应用入口**均未实现**,不要假设存在,也不得在文档中描述为已完成。当前缺四类规则目标实拍数据;`examples/` 仅是合成 schema 夹具。

### 规则安全(改动前必读 AGENTS.md「规则安全」节)

危险目标漏检会直接结束比赛回合:相关改动必须单独评估漏检并保留 `unknown`/疑似危险状态;不为让测试变绿而删断言、降安全阈值或跳过危险类用例。

## 代码风格要点

- `from __future__ import annotations`;公共数据结构优先 `dataclass(slots=True)`;时间字段带单位(如 `timestamp_ns`)。
- 项目文档保持中文;代码标识符与对外 schema 用英文。
- 文件名直接表达实现(如 `rpicam_source.py`),不建 `utils.py`/`common.py`;共享抽象只在两个以上真实实现需要时提取。
- 入口处校验矩阵形状、有限值、尺寸和枚举,错误信息包含实际值;默认拒绝 `quality.usable=false` 或元数据不匹配的标定。

## 提交与文档同步

- 提交用 `feat:`/`fix:`/`docs:`/`refactor:` 前缀,按可独立验证的功能边界组织;只暂存本任务文件,不顺带提交用户已有修改。禁止自主 `push`、`--amend`、rebase、reset。
- 实现状态变化 → 更新 README 状态表;模块边界变化 → `docs/项目结构.md`;公共 API/CLI/配置变化 → 同步相邻包 README 的入口表和使用示例(示例必须走 `load_runtime_config()`,不得另写一套参数)。
