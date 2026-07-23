# 人工与硬件验收脚本

本目录中的脚本需要 Raspberry Pi 相机、桌面显示、标定图片或本地输出，不参与 pytest 自动测试。

这些脚本只用于人工验收，不是可被运行代码导入的公共 API。默认相机脚本使用 `camera/rpicam_source.py`；逐帧传感器元数据后端由自动测试和录制命令覆盖。

- `camera_capture.py`：抓取单帧。
- `camera_stream.py`：实时预览和 FPS。
- `camera_undistort.py --intrinsics PATH`：加载指定内参实时预览去畸变结果。
- `picamera_minimal.py`：直接使用 Picamera2 的最小检查。
- `geometry_projection.py`：使用本地测试图片人工检查 BEV 和点投影。
- `hailo_pose.py`：从实际 `runtime.yaml` 加载 YOLO Pose 部署包，检查单张去畸变图像的 K0 观测。

运行前先执行 `python -m pip install -e .`，并确保系统包和显示环境可用。

数据采集不另建重复的硬件脚本：用 `rescue-vision-record --frames 200 --display` 执行真机短录，再用 `rescue-vision-check-recording RECORDING --display` 可视化回放并完成哈希、帧率、丢帧和元数据验收。完整步骤见 [`docs/数据采集工具使用.md`](../docs/数据采集工具使用.md)。
