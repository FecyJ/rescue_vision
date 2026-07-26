# 人工与硬件验收脚本

本目录中的脚本需要 Raspberry Pi 相机、桌面显示、标定图片或本地输出，不参与 pytest 自动测试。

这些脚本只用于人工验收，不是可被运行代码导入的公共 API。默认相机脚本使用 `camera/rpicam_source.py`；逐帧传感器元数据后端由自动测试和录制命令覆盖。

- `camera_capture.py`：抓取单帧。
- `camera_stream.py`：实时预览和 FPS。
- `camera_undistort.py --intrinsics PATH`：加载指定内参实时预览去畸变结果。
- `picamera_minimal.py`：直接使用 Picamera2 的最小检查。
- `geometry_projection.py`：使用本地测试图片人工检查 BEV 和点投影。
- `hailo_pose.py`：从实际 `runtime.yaml` 加载 YOLO Pose 部署包，检查单张去畸变图像的 K0 观测。
- `dataset_perception.py`：按 schema v2 数据清单批量运行 Hailo，覆盖式写出保留 UNKNOWN、质量信息和模型身份的观测 JSONL。
- `camera_undistort_perception.py`：按实际 `runtime.yaml` 连续执行相机、去畸变、Hailo Pose、K0/地面点叠加预览，按 `Q/Esc` 退出；偶发过期帧会标红并丢弃，不会终止预览。

运行前先执行 `python -m pip install -e .`，并确保系统包和显示环境可用。

数据采集不另建重复的硬件脚本：用 `rescue-vision-record --frames 200 --display` 执行真机短录，再用 `rescue-vision-check-recording RECORDING --display` 可视化回放并完成哈希、帧率、丢帧和元数据验收。完整步骤见 [`docs/数据采集工具使用.md`](../docs/数据采集工具使用.md)。

## 相机、去畸变和 Hailo 联调

持续预览：

```bash
python manual_tests/camera_undistort_perception.py \
  --config configs/runtime.yaml
```

无人值守地检查 20 帧后退出：

```bash
QT_QPA_PLATFORM=offscreen \
python manual_tests/camera_undistort_perception.py \
  --config configs/runtime.yaml \
  --frames 20
```

画面左上角显示从相机时间戳到推理完成的 `age`，以及全尺寸去畸变耗时。超过 `processing.max_observation_age_ms` 的结果会显示为红色 `STALE dropped` 并被丢弃；这属于实时安全降级，不应通过盲目增大阈值消除。

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
