# 人工与硬件验收脚本

本目录中的脚本需要 Raspberry Pi 相机、桌面显示、标定图片或本地输出，不参与 pytest 自动测试。

这些脚本只用于人工验收，不是可被运行代码导入的公共 API。默认相机脚本使用 `camera/rpicam_source.py`；逐帧传感器元数据后端由自动测试和录制命令覆盖。

- `camera_capture.py`：抓取单帧。
- `camera_stream.py`：实时预览和 FPS。
- `camera_undistort.py --intrinsics PATH`：加载指定内参实时预览去畸变结果。
- `picamera_minimal.py`：直接使用 Picamera2 的最小检查。
- `geometry_projection.py`：使用本地测试图片人工检查 BEV 和点投影。
- `hailo_pose.py`：从实际 `runtime.yaml` 加载 YOLO Pose v3 部署包，检查单张去畸变图像的六类框、K0/K1/K2、目标 HSV 类别、场地特征和 ROI 分割摘要。
- `target_ground_geometry.py`：旧接触锚点实验工具，不兼容 v3 K0 底面中心语义；仅用于复现旧结果，不作为新版模型验收入口。
- `dataset_perception.py`：按数据清单批量运行 Hailo，覆盖式写出观测 JSONL，保留 Pose 类别、HSV 候选/覆盖率、UNKNOWN 和质量信息。
- `camera_undistort_perception.py`：按实际 `runtime.yaml` 连续执行相机、去畸变、Hailo Pose、ROI HSV 掩码和 K0/地面点叠加预览；终端打印中心十字 K0、安全区 K0/K1/K2 的去畸变像素与机器人地面坐标，预览窗口也叠加对应地面坐标，按 `Q/Esc` 退出；偶发过期帧会标红并丢弃，不会终止预览。
- `remote_link.py --config PATH`：在树莓派侧以 `remote.role: server` 监听电脑端客户端，连接后发送协议要求的最小会话状态，并持续打印收到的 control；该状态有意声明所有业务能力不可用，所以正式客户端应保持控制禁用。仅验证连接可使用 `observe_only`；用自制底层客户端检查 control 帧时使用 `debug_control`。
- `remote_video.py --config PATH`：发送真实相机的最新 JPEG 帧和周期会话状态，接收电脑端的原图/perception 图像模式请求，但不接收或执行运动、夹爪和采集控制。
- `remote_capture.py`：兼容旧人工命令的薄包装；正式入口为 `rescue-vision-manual-capture`。
- `motion_minimal.py`：按配置以低速直行一小段，打开 UART 后先等待 v2
  `SOFT_BRAKE accepted` 安全同步，周期刷新轮速并在退出时柔和停车。
- `keyboard_drive.py`：终端键盘操控前进/后退/转弯，相机与 Hailo Pose 模型
  结果经独立旁路线程回传（终端打印类别/置信度/地面点，可选 `--display`
  窗口），`Esc`/`q` 退出，`空格`停车；`X` 切换夹爪运输姿态。
- `imu_rotation_monitor.py`：按配置低速原地旋转，周期打印 STM32 原始传感器系
  `gyro_z`、编码器和状态位；只发送轮速心跳，`Ctrl+C` 或异常时柔和停车。
- `imu_static_calibration.py`：车辆静止时被动收集有效 IMU 样本，输出传感器坐标系
  陀螺零偏、噪声、温度和可复制的配置片段；不发送任何运动命令。
- `safe_zone_straight_diagnostic.py`：只执行配置中的 `LEAVE_START` 越障直行段，
  实时打印轮速目标/实际值、ODOMETRY_IMU 序号与双时钟间隔、编码器、IMU、UART
  队列和 STM32 状态；发现连续 overrun 时停车。
- `stm32_monitor.py`：默认只读监测配置中的 STM32 COBS/CRC16 UART，周期显示
  编码器/IMU、系统状态、实际频率、序号丢帧和协议错误；可选只发送一次
  `QUERY_STATUS`，不发送运动或夹爪命令。

手动驾驶默认关闭编码器/IMU 融合；需要地图定位时在命令中添加
`--enable-localization`。这不会改变静态场地图发布，只控制融合与 `map/state`
位姿发布；v3 场地结果会在实测地标和定位门禁满足时提交视觉纠偏。

运行前先执行 `python -m pip install -e .`，并确保系统包和显示环境可用。

## STM32 串口监测

持续被动监测，按 `Ctrl+C` 退出：

```bash
python manual_tests/stm32_monitor.py \
  --config configs/runtime.yaml
  --verbose
```

默认每秒打印汇总和最新一帧定位/状态。只运行 10 秒：

```bash
python manual_tests/stm32_monitor.py \
  --config configs/runtime.yaml \
  --duration-seconds 10
```

打开后发送一次不刷新运动看门狗、也不控制执行器的状态查询：

```bash
python manual_tests/stm32_monitor.py \
  --config configs/runtime.yaml \
  --query-status
```

需要逐帧查看时添加 `--verbose`；正常链路约产生 110 行/秒，不适合作为默认
显示。`SUMMARY` 中的 `cobs_drop` 表示 COBS 定界/解码丢弃，
`protocol_error` 表示 CRC、长度、类型、方向、枚举或状态位错误，
`*_missing/duplicate/regression` 分别表示遥测序号缺失、重复或倒退。串口可能
按批到达，因此定位时间必须看 `ODOM sample_us`，不能用终端打印间隔积分。
监测器显示的是 STM32 线路原始传感器坐标系 `gyro_z`；正式融合会先按
`localization.fusion.imu_calibration` 做温度零偏、交叉轴/比例和安装旋转，再应用
`motion.odometry.gyro_z_sign`。人工左右转一次确认变换后的极性：若原始数据左转为负、
右转为正，配置 `-1`；若左转为正、右转为负，配置 `1`。矩阵必须先用静止重力方向和
已知姿态标定，不能用单次搬运姿态直接填写。

数据采集不另建重复的硬件脚本：用
`rescue-vision-record --frames 200 --display` 执行真机短录，再用
`rescue-vision-check-recording RECORDING --display` 可视化回放并完成图片
解码、帧率、丢帧和元数据验收。完整步骤见
[`docs/数据采集工具使用.md`](../docs/数据采集工具使用.md)。

## 最小运动检查

在架空轮，或已确认物理急停可立即触发且操作员全程监督的条件下运行：

```bash
python manual_tests/motion_minimal.py \
  --config configs/runtime.yaml \
  --supervised-physical-stop-ready
```

默认以 `0.05 m/s` 前进 `1 s`。可用 `--speed-m-s` 和
`--duration-seconds` 调整测试值，但仍受 `motion` 配置硬限制约束。脚本会在
运动前完成运动序号安全同步，运动期间持续调用 `MotionController.update()`，
同时非阻塞排空 STM32 回传，
因此不能用一次 `drive()` 后长时间 `sleep` 替代；正常结束、Ctrl+C 或异常都会
先发送 `SOFT_BRAKE`，这不能替代物理急停或固件看门狗。

## 键盘操控驾驶检查

在架空轮，或已确认物理急停可立即触发且操作员全程监督的条件下运行。脚本用
终端按键操控车辆（通过 SSH 也有效），相机与 Hailo Pose 模型的识别结果在独立
旁路线程处理并回传到终端，主线程只做轮速刷新、UART 排空和最新快照读取，
因此慢取帧、慢去畸变或慢推理不会拖慢运动安全循环：

```bash
python manual_tests/keyboard_drive.py \
  --config configs/runtime.yaml \
  --supervised-physical-stop-ready
```

按键：`W`/`↑` 前进，`S`/`↓` 后退，`A`/`←` 原地左转，`D`/`→` 原地右转；
按住前进与转向可组合成弧线；`Z` 夹爪打开、`X` 夹爪运输姿态、`C` 夹爪关闭，
`空格` 松手停车，`Esc`/`q` 退出。默认速度取
`min(0.15, motion.max_linear_velocity_m_s)` m/s、转向取
`min(0.60, motion.max_angular_velocity_rad_s)` rad/s，可用 `--speed-m-s` 和
`--turn-rad-s` 覆盖，但不得超出配置硬上限。终端每 200 ms 打印一行命令名、
目标 twist、左右轮目标、夹爪目标角度、目标类别/置信度/地面点、中心十字与
安全区数量，以及 STM32 电机/看门狗/急停/停车原因状态。

需要在图像中显示场地绝对坐标和航角时，先在 `runtime.yaml` 中完成并启用
`localization.fusion`，再添加 `--enable-localization`。窗口会显示融合得到的
`field_xy=(x,y)mm` 和 `heading=...deg`；它们是场地全局坐标，不是机器人局部
地面坐标。未启用或尚未收到有效编码器/IMU数据时显示
`pose=unavailable`，不会伪造位姿。

本地有显示器时可加 `--display` 弹出叠加识别框的 OpenCV 窗口（窗口聚焦时也可
用 `q`/`Esc` 关闭）。运行前必须在 `runtime.yaml` 中启用 `uart`、`motion` 和
`hailo`；夹爪键需要 `motion.gripper` 已标定并启用，其中 `X` 的运输姿态还要
配置 `motion.gripper.transport_*_angle_deg`。未配置地面标定时目标无
`g=(x,y)mm`，仍可正常识别与显示。正常退出、`Ctrl+C`、`SIGTERM` 或异常都会
先 `SOFT_BRAKE` 停车，再关闭相机、Hailo 和串口；这不能替代物理急停或固件
看门狗。

## IMU 原地旋转监视

在架空轮，或已确认物理急停可立即触发且操作员全程监督的条件下运行。脚本默认
以 `0.15 rad/s` 左转，并持续到 `Ctrl+C`；`gyro_z` 是 STM32 线路原始传感器坐标系
值，单位为 `rad/s`，不会在显示层重复应用树莓派的 `imu_calibration`：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/imu_rotation_monitor.py \
  --config configs/runtime.yaml \
  --angular-velocity-rad-s 0.15 \
  --direction left \
  --supervised-physical-stop-ready
```

右转把 `--direction` 改为 `right`。`--print-interval-seconds` 只控制终端输出频率，
不改变轮速刷新周期。脚本会在打开 UART 后先完成 `SOFT_BRAKE accepted` 安全同步，
运动期间持续调用 `MotionController.update()` 和排空 UART；`Ctrl+C`、命令拒绝、急停
锁存或其他异常均进入软停车。观察机器人系校准后的值应使用当前融合入口的
`localization.fusion.imu_calibration`，不要把原始 `gyro_z` 直接与机器人系航向比较。

## 轮子启动 `sample_overrun` 检查


该脚本只打开 UART，不打开相机、Hailo 或远程控制；必须在架空轮、物理急停就绪
且人员全程监督时运行。它先完成 `SOFT_BRAKE` 序号同步，再以低速启动轮子，
持续排空 `ODOMETRY_IMU`，首次收到 `sample_overrun` 就停车并打印遥测序号、
STM32 采样时间和编码器计数：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/motion_sample_overrun.py \
  --config configs/runtime.simulation-20min.yaml \
  --speed-m-s 0.05 \
  --duration-seconds 5 \
  --supervised-physical-stop-ready
```

检测到异常时默认返回 0 但输出 `RESULT=sample_overrun_detected`，方便现场收集
日志；需要让脚本以失败状态退出时增加 `--fail-on-overrun`。该脚本不会忽略或
降级 `sample_overrun`，不能用来证明固件采样链路已经合格。若状态中的
`reply_queue_full` 或 `tx_degraded` 为真，脚本会立即停车并退出；
`rx_degraded` 仍会打印并记录，但不再作为树莓派后续有效运动控制的单独门禁，
因为它是本次 STM32 启动期间的历史接收告警。`SOFT_BRAKE` 序号重同步不会清除
这些 STM32 粘滞告警。
排除串口接线、波特率、非法帧或多个进程同时读取串口后，必须复位或重新上电
STM32，再重新运行检查。

## IMU 静止零偏标定

车辆必须断开驱动动作并保持完全静止；工具只被动读取 `ODOMETRY_IMU`，不会发送
轮速、软刹车或其他控制命令。默认预热 5 秒、采样 30 秒，输出原始传感器坐标系
的陀螺零偏和噪声报告：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/imu_static_calibration.py \
  --config configs/runtime.yaml \
  --output /tmp/imu_static_calibration.json \
  --stationary-vehicle-confirmed
```

将报告中的 `recommended_config.localization.fusion.imu_calibration` 人工复核后
复制到运行配置；它只标定静止零偏和噪声，不会推导安装旋转、动态比例/交叉轴或
`gyro_z_sign`。这些仍需结合已知姿态和低速正反原地旋转复核。温漂补偿按需求保持
关闭（温度系数为零）；静态零偏和正反转动态复核通过后，当前
`localization.fusion.enabled` 已开启。

## 越障安全区直行诊断

该脚本不启动相机、Hailo、目标搜索或解团，只执行
`motion.cluster_breakup.departure_speed_m_s` 和
`motion.cluster_breakup.departure_distance_m` 定义的 `LEAVE_START` 直行段。
它会按 `motion.odometry.max_consecutive_overrun_samples` 允许配置的连续 overrun
数量，打印每个 overrun 的遥测序号、`sample_timestamp_us` 间隔、树莓派接收间隔、
编码器差分、三轴 IMU、状态位及当时轮速；超过该预算即停止（`null` 时不因超期
停止，只打印诊断）。运行条件必须是架空轮，或
物理急停可立即触发且操作员全程监督：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/safe_zone_straight_diagnostic.py \
  --config configs/runtime.simulation-20min.yaml \
  --supervised-physical-stop-ready
```

`--speed-m-s`、`--distance-m` 可覆盖本次诊断值，但默认优先使用配置中的越障直行
参数。`RESULT=departure_distance_reached` 表示达到编码器定距；退出码为 `2` 表示
本次出现过 sample overrun 或连续 overrun 导致停止。`sample_dt_ms` 是 STM32 采样
时间间隔，`host_dt_ms` 是树莓派接收时间间隔，不能用终端输出间隔代替。

## 编码器/IMU 连续融合真车验收

先用 `stm32_monitor.py` 确认 `ODOMETRY_IMU` 稳定达到 100 Hz、传感器原始轴方向和
单位正确、静止加速度模长约为 `9807 mm/s²`，再用
`localization.fusion.imu_calibration` 完整转换后检查机器人系约为
`(0, 0, +9807) mm/s²`，并实测填写 `motion.wheel_track_m`、`motion.odometry` 和
`localization.fusion.initial_pose`。未完成标定前不得启用
`localization.fusion.enabled`。

启用后运行 `rescue-vision-manual-capture`，通过远程场地图记录每段开始/结束位置、
航向、融合估计时间和主机接收时间。依次验收：

1. 静止至少 30 秒，记录零偏收敛和停车位置/航向漂移。
2. 定距直线、原地正反各一整圈、正反弧线，分别重复至少 5 次。
3. 在物理急停和全程监督下制造短时 IMU invalid、正向遥测丢帧和 STM32 重启；
   核对轮式降级、协方差增长和地图清除。视觉重锚定必须使用正式 v3 模型、
   实测安全区地标和远场标定，不能用局部精修或合成结果替代。
4. 真车往返后测量编码器+IMU 推算的闭环位置与航向误差，并记录 P50/P95
   定位年龄，并单列中心十字与安全区视觉纠偏误差。

每项必须保存真实 `motion.jsonl`、相机 recording、测量基准和失败样例。当前只有
合成轨迹 pytest，真车误差、长期温漂、UART 延迟分布和树莓派性能均为“未验证”。

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

输出包含 `['CPUExecutionProvider']` 表示 ONNX 后处理会话已创建。正式 Hailo 后端只在
`InferenceSession` 构造的窄作用域内过滤完全匹配的重复注册行，并把其他 stderr
原样转发；本段直接调用 ONNX Runtime，仍可能看到这类系统依赖噪声。

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
  --supervised-physical-stop-ready
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
的左右目标角之和始终为 `motion.gripper.angle_sum_deg`，包括从不满足该和约束的旧目标首次开始运动时。

## 固定流程解团检查

只在空旷、已划出中心目标区、物理急停可立即触发且人员全程监督时运行：

```bash
rescue-vision-cluster-breakup \
  --config configs/runtime.yaml \
  --supervised-physical-stop-ready
```

若 `runtime.yaml` 启用了 `remote`，该流程必须使用 `role: server` 和
`access_mode: observe_only`。它不会等待电脑端连接；连接成功后只向观察端发送
`observation/video/frame` 中的 perception 识别可视化 JPEG，默认每秒最多一帧，
并通过 `observation/map/state` 发送编码器+IMU的 `FieldPoint` 位姿 JSON；两条
观察旁路都不参与运动决策。

首次上车把 `departure_distance_m`、`breakup_speed_m_s`、
`breakup_distance_m`、`gripper_open_retreat_distance_m` 和
`retreat_distance_m` 调到保守小值，并依次验收：

1. 架空轮确认左右编码器有效、前进累计路程为正、配置的左右搜索方向正确；
2. 不放目标，只验证定距越障后左转，超时和 Ctrl-C 均停车；
3. 放置静止目标团但禁用电机，核对联合框中心和 K0 前向距离触发位置；
4. 低速短行程张爪推送，确认张爪退出配置距离后停车合爪，再执行闭爪退离；
5. 解团后连续两帧识别绿色，终端打印 `green_found` 并停车。

记录实际路程误差、目标团居中误差、张爪触发距离、每类物块最大位移及是否出现
夹持、钻车底或接近场界/安全区。该检查没有通过前不得提高冲散速度或用于比赛。
若报 `departure_timeout`，查看每秒 `progress`：两轮距离接近零表示电机未动或
编码器未累计；左右距离异号表示固件方向违反“前进均为正”的协议；两轮同号但
中心距离不足则核对减速带卡阻、轮半径、每圈计数和阶段超时，不要直接扩大超时。
若 `motor_output=false`，继续查看 `stop_reason`、`watchdog` 和 `estop`；先修复
固件使能、急停复位或命令拒绝原因，不得通过伪造里程或跳过定距门禁继续流程。
