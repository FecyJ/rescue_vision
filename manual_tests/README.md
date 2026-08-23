# 人工与硬件验收脚本

本目录中的脚本需要 Raspberry Pi 相机、桌面显示、标定图片或本地输出，不参与 pytest 自动测试。

这些脚本只用于人工验收，不是可被运行代码导入的公共 API。默认相机脚本使用 `camera/rpicam_source.py`；逐帧传感器元数据后端由自动测试和录制命令覆盖。

- `camera_capture.py`：抓取单帧。
- `camera_stream.py`：实时预览和 FPS。
- `camera_undistort.py --intrinsics PATH`：加载指定内参实时预览去畸变结果。
- `picamera_minimal.py`：直接使用 Picamera2 的最小检查。
- `geometry_projection.py`：使用本地测试图片人工检查 BEV 和点投影。
- `hailo_pose.py`：从实际 `runtime.yaml` 加载 YOLO Pose 部署包，检查单张去畸变图像的框、K0、HSV 类别和 ROI 分割摘要；启用 `target_ground_geometry` 时同时输出中心、足迹、朝向、拟合分数和降级原因。
- `target_ground_geometry.py`：从实际相机逐帧运行 YOLO Pose、HSV 分割和 `TargetGroundGeometryEstimator`，在去畸变画面叠加地面中心、足迹和质量信息，并按 JSONL 周期输出机器人地面系毫米坐标；按 `Q/Esc` 退出。
- `dataset_perception.py`：按数据清单批量运行 Hailo，覆盖式写出观测 JSONL，保留 Pose 类别、HSV 候选/覆盖率、UNKNOWN 和质量信息。
- `field_features.py`：从图片或视频离线检测安全区、无编号出发区、中心十字和低精度边界候选，写出 JSONL 及可选叠加图；不访问相机或 Hailo。
- `camera_undistort_perception.py`：按实际 `runtime.yaml` 连续执行相机、去畸变、Hailo Pose、ROI HSV 掩码和 K0/地面点叠加预览，按 `Q/Esc` 退出；偶发过期帧会标红并丢弃，不会终止预览。
- `remote_link.py --config PATH`：在树莓派侧以 `remote.role: server` 监听电脑端客户端，连接后发送协议要求的最小会话状态，并持续打印收到的 control；该状态有意声明所有业务能力不可用，所以正式客户端应保持控制禁用。仅验证连接可使用 `observe_only`；用自制底层客户端检查 control 帧时使用 `debug_control`。
- `remote_video.py --config PATH`：发送真实相机的最新 JPEG 帧和周期会话状态，接收电脑端的原图/perception 图像模式请求，但不接收或执行运动、夹爪和采集控制。
- `remote_capture.py`：兼容旧人工命令的薄包装；正式入口为 `rescue-vision-manual-capture`。

运行前先执行 `python -m pip install -e .`，并确保系统包和显示环境可用。

数据采集不另建重复的硬件脚本：用
`rescue-vision-record --frames 200 --display` 执行真机短录，再用
`rescue-vision-check-recording RECORDING --display` 可视化回放并完成图片
解码、帧率、丢帧和元数据验收。完整步骤见
[`docs/数据采集工具使用.md`](../docs/数据采集工具使用.md)。

## 传统视觉场地特征离线检查

输入已经按当前配置去畸变时：

```bash
python manual_tests/field_features.py recordings/field_sample.mp4 \
  --config configs/runtime.yaml \
  --already-undistorted \
  --output-jsonl output/field_features.jsonl \
  --overlay-dir output/field_feature_overlays
```

原始图片或视频应去掉 `--already-undistorted`，并启用有效内参。启用地面映射
后才会进行 BEV 尺寸筛选并输出 `GroundPoint`；当前开发机检查不等于树莓派
实时性能、围栏泛化或现场颜色精度验收。

中心十字绝对位姿的远场验收当前为“未验证”。现有本机地面映射只覆盖前方
约 500 mm，不能用它证明能从中心看到约 1.5 m 外的安全区或场界终端。后续
取得全范围可用标定和真实场地录像后，应在同帧输出中核对：十字交点位置误差、
航向误差、红/蓝锚点成功率、普通场界导致的 180° 歧义保留，以及遮挡、模糊和
颜色偏差失败样例；不得用合成 BEV 关闭这些验收项。

## 相机、去畸变和 Hailo 联调

持续预览：

```bash
python manual_tests/camera_undistort_perception.py \
  --config configs/runtime.yaml
```

检查目标地面中心和足迹：

```bash
python manual_tests/target_ground_geometry.py \
  --config configs/runtime.yaml
```

运行前必须在实际配置中启用内参、带完整相机外参的地面映射、Hailo 和
`perception.target_ground_geometry.enabled`。画面使用全尺寸去畸变图；中心和
足迹叠加通过地面投影重新映射到该图像，坐标输出为机器人地面系毫米，`x`
向前、`y` 向左。终端默认每 0.5 秒输出一次 JSONL；可用
`--print-interval 0.1` 提高输出频率，或用 `--frames 20` 处理有限帧数。
拟合失败会保留 `quality`、分数和 `center_ground_mm: null`，不会用检测框中心
代替拟合结果。

无人值守地检查 20 帧后退出：

```bash
QT_QPA_PLATFORM=offscreen \
python manual_tests/camera_undistort_perception.py \
  --config configs/runtime.yaml \
  --frames 20
```

画面左上角显示从相机时间戳到 Pose 推理、HSV 分类和掩码后处理全部完成的
`age`，以及全尺寸去畸变耗时。超过
`processing.max_observation_age_ms` 的结果会显示为红色 `STALE dropped`
并被丢弃；这属于实时安全降级，不应通过盲目增大阈值消除。

### `Schema error: ... already registered`

当前 Raspberry Pi OS/Debian 的系统 `python3-onnxruntime 1.21.0` 在创建 ONNX 后处理会话时可能一次性输出大量重复 schema 注册信息。本机已在“不导入 Hailo、只创建 ONNX Runtime 会话”的条件下复现，且会话仍能成功创建，因此这串信息本身不表示相机或 Hailo 断链。

判断链路时应继续看末尾日志：

- 出现相机 `configuring streams` 且画面/帧计数继续，说明相机已打开；
- 能显示每帧 `age` 和检测结果，说明 Hailo 推理及 ONNX 后处理已运行；
- 若程序退出，以最后一段 Python traceback 或 Hailo/libcamera 明确错误为准。

可独立复现当前系统包日志：

```bash
python - <<'PY'
import onnxruntime as ort
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
session = ort.InferenceSession(str(config.hailo.postprocess_onnx_path))
print(session.get_providers())
PY
```

输出包含 `['CPUExecutionProvider']` 表示 ONNX 后处理会话已创建。该噪声来自系统依赖层；不要在项目代码中全局重定向 `stderr`，否则会同时吞掉真正的相机和推理错误。

## 最小图传检查

树莓派侧可保持 `remote.access_mode: observe_only`：

```bash
python manual_tests/remote_video.py \
  --config configs/runtime.yaml \
  --timeout-seconds 30 \
  --video-fps 10 \
  --jpeg-quality 80
```

脚本先启动配置中的真实相机，客户端连接后首条发送会话状态，其中
`video_modes` 声明当前是否配置 Hailo perception；电脑端通过
`control/video/mode` 选择 `raw` 或 `perception`，再按
`observation/video/frame` 发送带实际 `mode` 的最新 JPEG。若单帧超过
`remote.max_payload_bytes` 会明确失败；应降低 JPEG quality、相机分辨率或
合理提高双方一致的 payload 上限，不能静默截断。

## 受监督手动驾驶采集检查

仅在赛外调试配置中设置 `remote.access_mode: debug_control`：

```bash
rescue-vision-manual-capture \
  --config configs/runtime.yaml \
  --output-root /data/rescue-targets/remote_test \
  --supervised-physical-stop-ready \
  --accept-timeout-seconds 30
```

入口始终声明 `motion_control`、`video_stream`、`vehicle_state`、
`capture_control` 和 `capture_status`；只有运行配置完成并启用机械标定时才
声明 `gripper_control`。它接收手动运动、持续夹爪扳机状态与采集命令。采集操作：

- `start`：在 `<output-root>/recordings/` 创建新的标准 recording
  会话，持续记录经过配置去畸变的帧及 `motion.jsonl`；
- `stop`：冲洗队列并完整关闭当前记录；
- `snapshot`：在 `<output-root>/snapshots/` 保存 JPEG 和同名 JSON 元数据；
- `mark_event`：录制期间向当前会话的 `events.jsonl` 追加事件。

每个请求先发布 `accepted`，执行完成后发布
`completed` / `rejected` / `failed`。相同 `request_id` 只重发原终态，不会
重复创建文件。`--output-root` 是树莓派本地测试参数，不在线上传输；电脑端
仍不得提交路径。入口退出时先停车，再关闭尚未结束的记录、相机和通信资源。
它是同时只服务一个客户端的赛外入口，不是正式比赛应用。断线会先停车并关闭
当前记录，随后可接受一个新连接；车端不会主动重连，也不会恢复旧死手使能。
夹爪真机检查前必须先标定左右舵机的安全开合范围和固定速度全行程时间，写入
`motion.gripper` 后再启用。按住左/右扳机时持续发送张开/闭合按压状态，松开
时发送两个状态均为 `false`；释放、命令超时或断线只停止角度继续变化，不会
自动开合，也不能直接套用协议说明书中的示例角度。逐次核对 UART/车辆状态中
的左右目标角之和始终为 194°，包括从不满足该和约束的旧目标首次开始运动时。
