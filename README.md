# 工创赛智能救援上位机视觉

2027 工创赛“智能救援”赛项的上位机视觉工程，目标平台为 Raspberry Pi 5、Hailo-8L 和 Camera Module 3 NoIR Wide。

当前已完成相机、标定与地面几何、严格配置、录制回放、数据集工具、离线评测、COBS/CRC16 UART 帧通道及 STM32 二进制协议树莓派端、直接 TCP 远程消息通道、差速与双舵机夹爪控制、受监督手动驾驶采集入口、YOLO Pose v3 六类三关键点解析与同帧分流、中心十字/安全区视觉定位消费、编码器/IMU 二维连续融合、连续场地图远程发布、目标跟踪、最小世界模型、规则状态机和正式流程。尚未完成正式 v3 模型资产、实测安全区地标、远场地面标定、完整区域/对手感知、现场接触/交付证据、固件失联看门狗闭环和整车比赛验收。这不是可直接参赛的完整程序。

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
| 对接 STM32、编码器或 IMU | [树莓派与单片机通信协议 v3](docs/树莓派与单片机通信协议v3.md) |
| 开发新模块 | [项目结构](docs/项目结构.md) → [后续优先级](docs/后续优先级.md) |
| 理解比赛类别和安全规则 | [赛题约束与视觉需求](docs/赛题约束与视觉需求.md) |
| 实现和联调正式流程 | [正式流程设计](docs/正式流程设计.md) |
| 查看可复用但尚未接入的目标记忆模块 | [项目结构](docs/项目结构.md) |
| 查数据或评测 JSONL 格式 | [数据集与评测 schema](docs/数据集与评测.md) |
| 修改仓库 | [AGENTS.md](AGENTS.md) |

官方资料位于 `docs/命题文件/`。项目文档用于工程实现；冲突时以最新正式文件和现场通知为准。

## 当前模块

| 模块 | 状态 | 责任 |
| --- | --- | --- |
| [`camera`](src/rescue_vision/camera/README.md) | 已实现 | 真机最新帧、离线回放和有界异步记录 |
| [`calibration`](src/rescue_vision/calibration/README.md) | 已实现 | 棋盘/ChArUco 采集、三模型内参比较和固定机器人多位置地面映射 |
| [`geometry`](src/rescue_vision/geometry/README.md) | 已实现 | 去畸变、显式坐标类型、地面/三维点投影与 BEV 转换 |
| [`config`](src/rescue_vision/config/README.md) | 已实现 | 安全默认配置、静态场地、UART/远程/motion/夹爪机械标定/几何/模型和感知算法装配 |
| [`communication`](src/rescue_vision/communication/README.md) | 已实现基础设施 | COBS UART 帧、直接 TCP 远程消息、raw/perception/BEV 图传、轻量动态地图 JSON、运动/夹爪/采集严格 schema 和有界队列；正式流程观察发布器待真车验收 |
| [`motion`](src/rescue_vision/motion/README.md) | 已实现基础设施 | STM32 v3 固定二进制协议、序号安全同步、差速运动、持续扳机双舵机夹爪、独立线/角加减速度限制、NB 反馈式定距/定角动作和远程超时保护；真车制动/IMU标定待验收 |
| [`app`](src/rescue_vision/app/README.md) | 已实现受限初版 | 正式流程（含近场宽度抓取）、match 简化开场变体、末端张爪推送—运输联调、受监督驾驶/采集、固定解团试验、像素居中抓取试验、独立绿黑多目标/单橙色近场收拢试验、编码器+IMU航位推算和不阻塞 observe_only 图传/`map/state` 位姿发布；真车门禁与正式比赛能力待验收 |
| [`data`](src/rescue_vision/data/README.md) | 已实现 | 记录检查、清单生成和按会话防泄漏划分 |
| [`evaluation`](src/rescue_vision/evaluation/README.md) | 已实现 | 分类、地面误差、时延和失败样例报告 |
| [`perception`](src/rescue_vision/perception/README.md) | 已实现基础设施 | v3 六类/[3,3] Hailo 解析、任务目标与场地特征同帧分流、ROI HSV、中心十字局部轴线精修、独立夹爪宽度估计和统一可视化；正式模型资产与现场性能待验证 |
| [`localization`](src/rescue_vision/localization/README.md) | 已实现融合基础 | 中心十字四向/位置观测、安全区身份与角点排列枚举、单帧去重、编码器/IMU EKF 与延迟纠偏；实测安全区地标和远场精度待验证 |
| [`tracking`](src/rescue_vision/tracking/README.md) | 已实现纯逻辑 | 时间关联、轨迹确认、短时遮挡、衰减和删除 |
| [`world`](src/rescue_vision/world/README.md) | 已实现纯逻辑 | 固定物理地图、红蓝任务区域派生、动态目标、危险状态、对手占据和不确定性 |
| [`mission`](src/rescue_vision/mission/README.md) | 已实现纯逻辑 | 首次/容量/伤员/危险规则、安全降级和抽象动作 |
| 定位至比赛应用主链路 | 受限初版 | 正式流程已接入跟踪、规则状态机、相对运动意图和 d1 安全区视觉纠偏；真实区域/接触证据、真车参数与整车验收待完成 |

各包常用 API、命令和实际对接示例见 [`src/rescue_vision/README.md`](src/rescue_vision/README.md)。

正式抓取链由 `MatchSequence.grasp_task` 保持跨搜索、接近、停稳、解团和再抓取的物理目标及总预算。
稳定场景在后台一次完成抓取选组与必要恢复规划，首次合法帧直接计入确认，不再等待消费者锁组后追要第二张图。
启动结束后进入正常抓取搜索，首趟有合适单绿即可直接夹取；首趟现已按邻块 K0 与配置实体尺寸检查扫掠净空，避免仅中心在走廊外就放行，
证据及验证边界见[开局单绿净空诊断](docs/match开局单绿净空诊断.md)。必要解团先标记团内得分物块，
对准后全程闭爪前进 0.5 m、后退 0.3 m，持续按采集位姿更新标记位置；退离后用新观测
接回标记目标的对准和夹取检查。补夹保留本趟容量、原扫描预算和物理失败记录，带载不开放解团。
1650 日志暴露的解团终点回退重启与动态诊断触发重复软刹车已修复；终点锁存、停稳等待有界，
制动期间持续零轮速心跳，实际制动距离和下位机响应仍待真机复测。
任务、稳定证据、延迟结果与失败退出的完整契约见[正式流程设计](docs/正式流程设计.md)。
1818 实机暴露的转向完成时间漂移、启动无航向纠偏和远场误交接已补软件修复；证据见[诊断记录](docs/match1818诊断与修复.md)，修复后真机效果待验证。
绿/黑近场抓取（含贪心补夹）已按配置底面尺寸和闭爪末端补足行程，避免仅中心到位即合爪；
贪心补夹可用 `near_field_grasp.greedy_target_final_x_mm` 单独调节终点；真实抓获改善待真机复测。
2315软件回归覆盖快控制/慢感知及恢复后实际抓取推进；Pi/Hailo控制周期、真实抓获、危险类指标和整车验收仍未验证。
正式运输已接入默认开启的实验性门前清障模块：按己方两类安全区门前 200 mm 的异类/蓝色 K0 证据，
暂存原载荷、沿 S1—S2 横扫、移至中场，再用普通近场链完整回取原载荷；真机效果尚未验证。
正式流程已补本趟夹取累计、夹爪内颜色误夹恢复（含黑线/彩色暗面区分及蓝色模型框触及识别区判定）、完整bbox门禁、按画面位置选对侧两点、停稳两帧校准（首帧等待与确认分开计时），
退区停稳后直接转向搜索；参数、日志诊断与真机未验证项见[正式流程设计](docs/正式流程设计.md)。
正式流程已补启动异常链落盘、失败阶段及 UART 底层原因诊断；偶发 UART 启动失败的
现场根因和修复后的启动成功率仍待验证，排查入口见 [app 说明](src/rescue_vision/app/README.md)。

另有独立的末端张爪推送—运输联调入口：车辆从 `FieldPoint(0,0)`、`+90°` 出发，直接搜索固定 `TRANSPORT` 走廊内的单个绿色 K0；它不会进入解团状态，按正式流程夹取并运输，只有末端推进时保持夹爪张开推入安全区再退出：

```bash
rescue-vision-grab-transport \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

## 核心数据流

```text
FrameSource → CameraFrame → CameraModel → 去畸变帧
                                      ├─> TargetPoseDetector
                                      │          └─> TargetObservation
                                      │                     ↓
                                      │          MultiTargetTracker
                                      │                     ↓
                                      │          WorldModel / MissionStateMachine
                                      └─> 同一 YOLO Pose v3 的中心十字/安全区结果
                                                 └─> FieldFeatureDetectionResult
                                                              ↓
                                                    CenterCrossLocalizer
                                                              ↓
                                               CenterCrossPoseObservation
                                                              ↓
                                               OdometryImuFusion
                                                       ↓
                                             连续 FieldPose2D / 地图

RemoteMessageConnection → DebugMotionCommand → RemoteMotionExecutor
                                                ↓
                                      MotionController → UART

RemoteMessageConnection → DebugGripperCommand → RemoteGripperExecutor
                                                 ↓
                                      MotionController → UART

PerceptionFrameRenderer → RemotePerceptionPublisher → RemoteMessageConnection
                         （observe_only perception JPEG；独立低频旁路）

OdometryImu → OdometryImuFusion → FieldPose2D
             （树莓派先做温度零偏、交叉轴/比例和安装旋转校正；v3 地标新鲜且门禁满足时提交视觉纠偏）

FieldPose2D → RemoteLocalizationPublisher → observation/map/state
              （独立低频 JSON 旁路）

PerceptionSnapshot + capture-time pose history ──────→ MatchSequence
                                                       → 当前目标几何/动作意图 → MotionController

PerceptionSnapshot + near-field session request ──────→ bounded GraspPreparationWorker
                                                       → dynamic grasp plan/angles → MatchSequence
```

- 像素必须区分 `RawPixel` 与 `UndistortedPixel`；地面点使用 `GroundPoint`，单位 mm。
- `CameraModel` 是去畸变唯一权威；`GroundProjector` 是去畸变像素与机器人地面/三维投影的唯一权威。
- 实时路径只处理最新帧；录像、显示和日志使用有界旁路。
- 危险目标允许“证据不足/疑似危险”区分，不得用总体指标掩盖危险类漏检。
- 规则状态机只消费显式世界、接触、交付和安全证据；正式流程的几何门禁和 d1 视觉纠偏仍需真实接触、地标和现场验收。
- `scan_target_memory.py` 与 `field_target_cluster.py` 是可复用纯逻辑模块，当前不接入正式流程；正式 match 自己保留有界的采集时刻编码器/IMU 位姿历史用于延迟几何对齐，不做记忆目标接管。
- 场地特征观测契约只输出去畸变像素和可选机器人地面观测；检测实现不得在定位
  完成前伪造 `FieldPoint` 或直接修改世界模型。传统全图 OpenCV 场地检测与局部
  场界已删除；仅允许在模型 bbox 内进行中心十字轴线精修。
- v3 任务目标优先使用模型 K0 作为底面几何中心投影；K0 缺失或低置信时使用检测框底边中点作为地面锚点，保证四类目标仍可进入路径与抓取判断。旧 `TargetGroundGeometryEstimator` 不在 v3 主链路中。
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
| `MapPixel(u, v)` 电脑端场地图像素系 | 电脑端固化底图左上角为原点；`u` 向右，`v` 向下 | 像素 | 电脑端绘制 `MapStateObservation` 的 FieldPoint 动态覆盖 |

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

GroundPoint ── 中心十字绝对观测 + 编码器/IMU 连续融合 ──> FieldPoint ──↔ MapPixel
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
- `FieldPoint` 与 `GroundPoint` 不能直接互换。中心十字没有唯一位姿且连续
  融合没有有效绝对锚点时，世界模型保留缺失的 `FieldPoint`，不能把机器人局部地面点
  伪装成场地全局点。

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
  下。BEV JPEG 使用 `bev_pixel` 并携带机器人地面范围和 `mm_per_pixel`；
  它不是定位后的场地图。电脑端静态底图使用 `MapPixel` 并在本地映射
  `FieldPoint`；车端不再发送场地图 PNG：

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
| `rescue-vision-cluster-breakup` | 固定出发姿态的受监督解团与绿色扫描试验 |
| `rescue-vision-green-grab` | 识别绿色物资、开夹爪像素居中接近并合爪的简化试验入口 |
| `rescue-vision-gripper-width` | 自动选择 1～3 个绿黑物资或单个橙色目标，对准、按几何收拢并合爪保持；不掉头 |
| `rescue-vision-motion-sequence` | TUI 输入 `a1 a2`，通过 `--distance-m` 配置距离并自动计算速度；默认前进 1.5 m |
| `rescue-vision-teach-replay` | 记录手推过程的编码器/IMU JSONL，并立即或以后受监督回放轮轨迹 |
| `rescue-vision-match` | 正式流程入口；硬件和首帧预检完成后等待 Enter 瞬时放行，支持 `--start-area 2/3`、本地图像预览、observe-only perception JPEG、D2 遥测和按时间命名的流程日志 |
| `rescue-vision-match-cc` | CC 独立流程入口；使用 `configs/runtime.cc.yaml`，执行稳健解团、单块分级搜索和正式安全区运输 |
| `rescue-vision-match-nb` | 当前正式流程的开场变体；从起点按 `match.nb_opening_actions` 配置的转角/直行序列执行，随后复用正式流程 |
| `python -m rescue_vision.app.match_strategy` | 蓝色优先策略变体入口（尚无 console script）；使用 `configs/runtime.strategy.yaml`，启动两段冲刺后只搜蓝色危险物块，能直接夹取则单块转运、否则对全蓝团解团，两趟分别放到对面安全区左右 D2 点，随后翻转进正式流程复用绿块搜索 |
| `rescue-vision-grab-transport` | 末端张爪推送—运输联调入口；从场地 `(0,0,+90°)` 直接搜索绿色物资，按正式流程夹取运输，末端保持张开推入并退出，不执行解团 |
| `rescue-vision-split` | 按 `recording_id` 整组划分 |
| `rescue-vision-evaluate` | 生成离线评测报告 |

采集命令与现场步骤统一见[数据采集手册](docs/数据采集工具使用.md)，格式定义见[数据集与评测 schema](docs/数据集与评测.md)。`examples/` 仅是合成格式夹具，不能作为实拍或性能证据。
