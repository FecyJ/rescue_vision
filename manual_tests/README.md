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
- `rescue-vision-motion-sequence`：受监督 TUI 定距动作试验；用 `--distance-m` 配置目标距离
  （默认 1.5 m），输入 `a1`、`a2`，程序自动计算峰值速度、加速时间和减速时间并控制车辆前进。默认只打开 UART
  和底盘，增加 `--local-preview` 可同时启动相机/Hailo 本地识别窗口；不采用 `motion` 的速度/加速度上限，
  但仍受协议可编码范围约束，运行中按 `q`/`Esc` 软刹车。
- `imu_rotation_monitor.py`：按配置低速原地旋转，周期打印 STM32 原始传感器系
  `gyro_z`、编码器和状态位；只发送轮速心跳，`Ctrl+C` 或异常时柔和停车。
- `imu_static_calibration.py`：收集静止有效 IMU 样本，输出传感器坐标系陀螺零偏、
  噪声、温度和 `sensor_to_robot_rotation` 矩阵；可在物理急停和全程监督下直接
  控制正/反原地旋转，用编码器角度复核 `gyro_z_sign`，不需要先另存运动 JSONL。
- `safe_zone_straight_diagnostic.py`：只执行配置中的 `LEAVE_START` 越障直行段，
  实时打印轮速目标/实际值、ODOMETRY_IMU 序号与双时钟间隔、编码器、IMU、UART
  队列和 STM32 状态；发现连续 overrun 时停车。
- `stm32_monitor.py`：默认只读监测配置中的 STM32 COBS/CRC16 UART，周期显示
  编码器/IMU、系统状态、实际频率、序号丢帧和协议错误；可选只发送一次
  `QUERY_STATUS`，不发送运动或夹爪命令。
- `pid_tune.py`：通过 UART 文本协议（115200）标定左右轮速度比例系数，使
  「相同轮速指令走直线」；按多个速度各直行 1.5 m、积分实测轮速更新 scale、
  原地转 90° 进入下一轮。

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

按键：`W`/`↑` 锁存前进，`S`/`↓` 锁存后退；先按 W/S，再按住
`A`/`←` 或 `D`/`→` 即可组合左/右弧线，松开转向键后自动回正。终端没有
key-up 事件，因此纵向运动会保持到按下相反纵向键或 `空格`，不能依靠松开 W/S
停车。`Z` 夹爪打开、`X` 夹爪运输姿态、`C` 夹爪关闭，`Esc`/`q` 退出。默认速度取
`min(0.15, motion.max_linear_velocity_m_s)` m/s、转向取
`min(0.60, motion.max_angular_velocity_rad_s)` rad/s，可用 `--speed-m-s` 和
`--turn-rad-s` 覆盖，但不得超出配置硬上限。终端每 200 ms 打印一行命令名、
目标 twist、左右轮目标、夹爪目标角度、目标类别/置信度/地面点、中心十字与
安全区数量，以及 STM32 电机/看门狗/急停/停车原因状态。

每次运行都会先打印日志路径，并把底盘指令变化实时输出为 `control_log=...`，同时
通过有界旁路线程写入 `--log-dir` 下按时间命名的 `keyboard_drive_*.jsonl`（默认
`logs/`）。日志保存相对单调时间、控制器实际采用的 twist、目标左右轮速，以及
100 Hz 累计编码器和 Z 轴陀螺仪遥测。退出时先写零速并软刹车，然后继续记录至编码器
稳定或 1.5 s 有界截止，使制动尾程也进入参考轨迹。夹爪键不属于车辆运动轨迹，不写入
该日志。

将车辆放回同一起点、确认相同运行配置和清空运动区域后，可回放底盘轨迹：

```bash
python manual_tests/keyboard_drive.py \
  --config configs/runtime.yaml \
  --replay logs/keyboard_drive_YYYYmmdd_HHMMSS_ffffff.jsonl \
  --supervised-physical-stop-ready
```

脚本会在打开 UART 前校验日志 schema、设备时间单调性、编码器有效位、首尾零速、
轮速上限，以及轮距、轮径、每转计数、陀螺仪方向、速度/加速度限制和左右轮权重；
随后要求人工输入 `REPLAY`。当前 v2 日志是唯一支持格式，旧的纯指令 v1 日志会被拒绝。
回放不启动相机或 Hailo，而是以记录轮速为前馈、以左右轮累计行程误差为反馈，并用记录
和现场 IMU 相对航向误差作有界差速修正。参考时间结束后继续闭环收敛到左右轮各 5 mm、
航向 3° 的终点容差，最长额外等待 3 s；现场编码器/IMU 超过 200 ms 未更新、控制链路
异常或终点无法收敛均会软刹车并报错。过程中按 `空格`、`q`、`Esc`、`Ctrl+C` 或
SIGTERM 也会软刹车。

这种方式能抵消调度抖动、轮速建立差异及一部分左右轮误差，但编码器无法观测轮胎相对
地面的纵向/横向打滑，IMU 相对航向也不是场地绝对位置。地面、负载和轮胎条件应尽量与
录制时一致，车辆必须回到相同起点，并由操作员全程监督；需要场地绝对轨迹重复精度时，
还需接入已完成真机验收的视觉绝对定位闭环。

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
  --config configs/runtime.match.yaml \
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

## IMU 零偏、安装矩阵与自转复核

静止阶段车辆必须断开驱动动作并保持完全静止；脚本会先发送安全同步的软刹车，
默认预热 5 秒、采样 30 秒，输出原始传感器坐标系的陀螺零偏、噪声和由静止重力
方向求得的安装旋转矩阵。矩阵打印为 `sensor frame → robot frame`，并写入报告：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/imu_static_calibration.py \
  --config configs/runtime.yaml \
  --output /tmp/imu_static_calibration.json \
  --stationary-vehicle-confirmed
```

如需脚本直接控制正、反各一整圈并用编码器角度复核极性，必须架空轮或确认可立即
触发物理急停、人员全程监督，然后显式增加：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/imu_static_calibration.py \
  --config configs/runtime.yaml \
  --output /tmp/imu_calibration.json \
  --rotation-angle-deg 360 \
  --rotation-direction both \
  --rotation-angular-velocity-rad-s 0.20 \
  --stationary-vehicle-confirmed
```

脚本会在每个方向达到配置编码器角度后停车，并打印每一方向的编码器角度、陀螺
积分角度、比例诊断和合并后的 `gyro_z_sign`。如果比例偏离 1，只作为动态量程/比例
复核结果，不会擅自把不完整的单轴结果写入三轴 `gyro_cross_axis_scale`。

静止重力只能确定安装矩阵的横滚/俯仰，绕竖直轴的偏航在单一姿态中不可观测；脚本
会明确打印这一状态，并采用最小旋转。若安装同时改变了平面偏航，需要再提供已知的
机器人轴向基准后补充矩阵，不能只凭这次静止采样覆盖配置。温漂补偿按需求保持关闭
（温度系数为零）。将报告中的 `recommended_config` 人工复核后复制到实际运行配置，
再用独立低速正反转确认。

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
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready
```

`--speed-m-s`、`--distance-m` 可覆盖本次诊断值，但默认优先使用配置中的越障直行
参数。`RESULT=departure_distance_reached` 表示达到编码器定距；退出码为 `2` 表示
本次出现过 sample overrun 或连续 overrun 导致停止。`sample_dt_ms` 是 STM32 采样
时间间隔，`host_dt_ms` 是树莓派接收时间间隔，不能用终端输出间隔代替。

## 左右轮比例系数标定（走直线）

该脚本面向 UART 文本协议（115200、`\r\n` 结尾）的底盘固件：`m<L>,<R>` 设
双轮速度、`b0,0` 柔和刹车、`p<Kp>,<Ki>,<Kd>` 设共享速度环 PID、`v` 查询，
并周期回传 `t<ms>,<实测左速>,<实测右速>,...`。它不复用仓库 v3 二进制
`MotionController`，直接按上述文本协议收发。左右轮速度正值均表示前进；若
实测速度方向相反，先核对固件方向，不要在脚本里加符号翻转。

目标「给左右轮发送相同速度即可走直线」无法靠一组共享 PID 补偿左右轮机械
不对称，因此脚本标定的是**每轮一个、与速度无关的比例系数**（scale），放在
树莓派侧：下发时套用 `m<v*scaleL>,<v*scaleR>`。每轮先直行 `--distance-m`，
持续积分实测左右轮路程，结束后按「实测 / (指令 × scale)」更新两个 scale，
再原地转 `--turn-angle-deg`、等待 `--settle-seconds` 进入下一速度轮次，验证
同一组 scale 在 0.05–0.20 m/s 各速度下均走直线。

在架空轮，或已确认物理急停可立即触发且操作员全程监督的条件下运行：

```bash
PYTHONPATH=src .venv/bin/python \
  manual_tests/pid_tune.py \
  --device /dev/ttyAMA10 \
  --wheel-track-m 0.19 \
  --speeds-m-s 0.05,0.10,0.15,0.20 \
  --supervised-physical-stop-ready
```

- `--wheel-track-m` 必须填入实测轮距，仅用于航向偏差换算和转 90° 时长。
- `--speeds-m-s` 每个值必须落在 `[0.05, 0.20] m/s`；`--passes` 可对整组速度
  重复多轮以收敛。
- 可选 `--set-pid 3.5,0.25,0` 在标定前发一次 `p<Kp>,<Ki>,<Kd>`（范围
  Kp/Ki ∈ [0,20]、Kd ∈ [0,5]），仅设置共享速度环，不参与左右平衡。
- 每轮打印 `ROUND ... heading_dev_deg=... scales=(旧)->(新)`；`heading_dev_deg`
  为正值表示左轮走得多、车辆向左偏。最终 `PID_TUNE_DONE` 给出
  `left_scale/right_scale` 和 `right_over_left`，把这两个比例系数写入后续
  直行代码即可。
- scale 只保存在树莓派侧运行内存；断电不影响本脚本，但也不写入固件。正常
  结束、Ctrl+C、超时或异常都先发 `b0,0`，这不替代物理急停。

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

正式 `rescue-vision-match` 的动态解团延迟回归见下节；本节的固定试验入口不使用动态策略。

### 正式流程解团闭环验收（真机未验证）

运行 `rescue-vision-match --config configs/runtime.match.yaml --supervised-physical-stop-ready
--local-preview --log-dir logs`，确认解团严格按以下顺序运行：

1. 车辆先真实停稳；停稳必须由连续有效轮编码器/IMU证据确认，零速命令不能替代它。
2. 停稳后的当前帧找到一个带可靠 K0 的接触核心，并一次完成危险目标、场界、安全区和
   机器人包络检查；不等待近场抓取准备器。
3. 当前核心通过 `match.breakup_confirmation_frames` 个不同有效帧后出现
   `breakup_plan_frozen`，随后按闭爪前推、停稳张爪、后退、停稳闭爪执行；不经过单独
   对准、接近或接触后第二轮复核。
4. 注入真实轮计数变化、IMU旋转、遥测断档/无效、重复帧、外围目标闪烁和危险侵入，确认
   旧几何不会冻结；确认超时后退出当前区域并扫描其它候选。
5. 控制轮询、感知延迟和帧间隔仍按 AGENTS 的时序场景测量，分别记录非零动作发生时间、
   观测年龄、确认帧数和前推/后退实际里程。蓝色危险类单列记录误纳入、最大推移与越界/入区风险。

静止证据、日志字段和状态机约束以 [AGENTS.md](../AGENTS.md) 及
[正式流程设计](../docs/正式流程设计.md) 为准。自动回归不能证明真实碰撞效果或 Pi/Hailo 性能。

### 固定入口步骤

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


## 无解团开场航点验收（未验证）

开场由固定启动转向+直行改为绝对场地坐标直线航点的变体入口。航点、速度、夹爪角度和
对准参数以 [app README](../src/rescue_vision/app/README.md) 为权威，这里只列真机核对项。
先在物理急停与全程监督下运行：

```bash
rescue-vision-match-nb --config configs/runtime.match_nb.yaml \
  --supervised-physical-stop-ready --log-dir logs
```

逐项确认：

1. 每段直线先原地旋转对准航向，再开始平移。若看到车辆一边转向一边前进、轨迹明显是
   弧线，说明对准门没有生效，必须停车检查——起始航向误差可达 60° 以上，边转边走会让
   终点横向偏出数百 mm。
2. 每次阶段变化都会打印 `nb_opening=phase=... target=... position=... error=...`。
   核对 `error` 的绝对值不超过 `nb_opening_align_tolerance_mm`（加上一个控制周期的
   行程余量）；`error` 明显偏大时，先区分是车辆没到位还是航位推算本身漂移。
3. 每个航点到位后有 `nb_opening_*_arrival_settle` 的零速停顿，时长约
   `nb_opening_settle_time_s`；停稳结束才从新位姿起算下一段。测量车轮在停稳期间确实不动。
4. 倒车段线速度为负，车辆沿原路退回而不是继续向前。
5. 人为制造航向不可信（例如临时把 `motion.odometry.gyro_z_sign` 取反）后运行，确认连续
   未对准超过 `nb_opening_align_timeout_s` 时打印
   `nb_opening_align_timeout_stop` 并进入 `terminal_stop`，而不是持续旋转。

## 蓝色优先策略变体验收（未验证）

只搜蓝色危险物块、两趟投放后翻转入正式流程的变体。流程、继承边界和诊断以
[app README](../src/rescue_vision/app/README.md)、[正式流程设计](../docs/正式流程设计.md)
为权威，这里只列真机核对项。先在物理急停与全程监督下运行：

```bash
python -m rescue_vision.app.match_strategy --config configs/runtime.strategy.yaml \
  --supervised-physical-stop-ready --log-dir logs
```

**这是 2026-09-12 同步 2047 语义后的首次复跑，策略阶段的解团后退行程、旋转预算、
安全区排除与确认时序都已改变，不能沿用旧运行结论。**

逐项确认：

1. 开场两段冲刺方向相反（先右后左）、各自转够 `startup_turn_angle_rad`；第二段冲刺
   距离按 `cluster_relocate_distance_m`。两段都结束后才进入搜索，中途不得提前搜索。
2. 策略阶段只选蓝色危险物块。场上同时有绿/黑/橙时，车辆不得改抓这些类别；蓝色前进
   走廊被别的目标占住时应继续搜索，而不是把候选换成非蓝色。
3. 单个蓝块直取：确认 `transport_align_green` 阶段的对准行为与 2026-09-11 运行一致
   （变体在蓝色运输期间固定走历史对准路径 `_align_green_legacy`，不走
   `_step_formal_green_align`）。若对准表现出有界转角预算、路径阻挡改走重选等新行为，
   说明相位分派失效。
4. 全蓝目标团解团：核对实际后退距离以“退出实际穿入深度”为准，不再整段抵消接近空程；
   记录每次解团的穿入深度、后退距离与实际位移，确认没有退得过远。
5. 两趟分别在对面安全区左、右 D2 点释放（`safe_zone_fallback_target_field` 为 x<0，
   `safe_zone_injured_target_field` 为 x>0），终点不再向安全区深处二次推进。
6. **阶段翻转**：第二趟返回结束时打印
   `strategy_blue_tasks_complete_start_match_green_search`。`logs/strategy/` 的五次
   2026-09-11 运行从未到达这一步，因此翻转后的正式绿块流程（含首轮单绿机会、旋转预算
   与解团失败记忆复位）在真机上完全未验证，必须单独作为一次完整验收对待。
7. 中途人为制造蓝色走廊阻挡与解团失败，确认失败记忆不会因原地转向或新会话被解除，
   且在有界时间内换目标或换位。
6. 把 `nb_opening_align_angular_velocity_rad_s` 调到接近
   `2*motion.min_wheel_velocity_m_s/wheel_track_m` 时，确认单轮最低速度抬升导致的
   旋转加速仍在可接受范围。

## 正式流程近场接管验收（未验证）

运行 `rescue-vision-match` 时，确认搜索态先按固定
`motion.gripper.transport_*` 开口检查前向扫掠走廊。目标从远场进入 `near_field_grasp.max_range_mm` 后状态变为
`transport_near_field_grasp`，且交接阶段保持 `TRANSPORT`，不先发全开口。首趟只摆放一个
绿色物资，随后分别摆放绿黑混合组、单橙色伤员和蓝色危险阻挡物，核对首趟单绿、后续绿黑最多 3 个、
橙色只能单独抓取。橙色抓取后核对路线终点为 `match.safe_zone_injured_target_field_mm`，
绿黑仍使用 `match.safe_zone_fallback_target_field_mm`；确认伤员进入己方伤员区而不是物资区。同时核对
动态开爪角度、编码器定距、静态安全区/场界门禁、软刹车和会话 ID 日志。开爪后不应因
单帧漏检或 `rx_degraded` 自行停车；里程计缺失、明确路径阻挡或急停必须停车并按日志原因
恢复/重选。抓取完成仍记录 `capture_confirmed=False`，真实接触与交付需另行人工验收。

`rescue-vision-grab-transport` 仍是单绿联调入口，不用它验证橙色正式流程。

## 多目标近场收拢验收（未验证）

在物理急停与全程监督条件下运行 `rescue-vision-gripper-width --config
configs/runtime.match.yaml --supervised-physical-stop-ready --local-preview --once
--log-dir logs/near_field`。
参数及动作契约以 [app README](../src/rescue_vision/app/README.md) 为权威。

1. 架空或固定底盘，测量左右安全开闭端点、实际最大开口、夹臂厚度/轮廓、夹爪前端
   参考线和目标结束位置，复核 `target_final_x_mm`、`corridor_start_x_mm` 与横向余量；
   检查合爪方向。先单绿、单黑、单橙，再并排绿绿、黑黑、绿黑及三物资，记录计划跨度、开口、
   实际开口、最远 K0、橙色纵向包络最近端/跨度、行程和最终留存数量。
2. 在矩形前进走廊内外、开口两侧及目标组之间摆放蓝色危险块，危险类单列报告；
   再检查橙色与绿黑同组、橙色多目标、未知/颜色冲突和缺失 K0。以监督停止避免实际
   危险转运，确认橙色只形成单目标计划，记录误纳入和漏检，不能只报告总体成功率。
3. 测试走廊内额外绿黑被纳入后总数不超过 3；测试 4 个、最大开口边界、纵向间距较大、
   掩码投影受顶部影响、强阴影和短暂遮挡。确认危险记忆未因一帧消失而清空。
4. 验证对准允许范围、滞回、唯一确认窗口的 `confirmation_frames` 和总预算
   `alignment_timeout_ms`（超时应带原因回到搜索）。选不出方案时应只等
   `no_plan_wait_ms` 就带原因换候选/解团，不得占用整个提交预算空等；实测成功提交
   链路（停稳→当前帧→异步结果→开爪）耗时，确认它仍落在 `alignment_timeout_ms` 内。
   同一帧重复提交不得增加确认进度；快控制/慢感知、
   短暂漏帧和一次规划旁路异常不得把已取得的确认永久清零。若出现新的明确危险或不合法组合，
   旧确认必须失效。在橙色目标周围 50 mm 内摆放有可靠 K0 的其它物块，确认橙色计划被硬拒绝；另测缺少地面位置的目标，确认它不会被无依据当作禁区阻挡。再验证静止规划阶段的危险/未知目标拒绝、编码器
   中断/停滞、相机/规划旁路失效和 GUI 关闭。张爪后运动模糊或新 `unknown` 不应单独
   重检并打断已冻结的前进计划；单独注入 `rx_degraded` 不应停止动作，明确急停应停止
   底盘。检查退出条件、恢复所需新帧及夹爪最后角度。
5. 确认 `--once` 完成退出、普通模式合爪后保持而不掉头/重新抓取，退出释放 UART、
   相机与线程。记录 `capture_confirmed=False`，通过独立人工观察记录真实收拢结果。
6. 在目标 Raspberry Pi/Hailo 上让 perception 持续运行，记录日志中的
   `perception_timing`：采集→提交、队列等待、推理、后处理、采集→结果和渲染的
   P50/P95，以及 `submitted/processed/replaced/stale_dropped`。近场每秒状态日志还应核对
   `session`、`confirmation`、`plan_age_ms` 和 `preparation_age_ms`，确认未把两种年龄混写。
   同时记录采集→计划→
   运动命令端到端年龄、设备温度、内存和控制循环间隔；分别核验有/无预览、日志
   阻塞及最密集目标布局。这些现场指标仍为**未验证**，不得仅凭
   `processing.max_observation_age_ms` 配置值宣称已满足延迟要求。

当前 4 mm 总开口余量、`target_final_x_mm`、`corridor_start_x_mm`、10 mm 横向走廊余量
和推进行程均为试验初值；上述项目及整车真实净空、抓取效果、危险类指标、目标硬件性能
均未验证。合成 pytest 仅证明逻辑，不可替代这些记录。

### 延迟几何开爪回归（真机未验证）

使用 `configs/runtime.match.yaml`，对孤立单绿完成对准和停车。记录真实
`plan_age_ms`、`preparation_age_ms`、`stationary_since_ms`、`motion_age_ms` 和
`gyro_z_rad_s`：在300～450 ms视觉延迟、有效静止遥测持续更新时，应完成确认并出现
`open_group_width`，随后定距前进、合爪并进入原d1运输。分别开/关预览和详细日志测量
端到端P50/P95，不把合成延迟回归当作实车性能结果。
拍摄后移动底盘、断开遥测或重放数秒旧计划，均不得依据旧几何开爪；重新停稳并得到
新计划后应恢复。危险进入走廊和伤员混运仍应拒绝，危险类单列。

### 1930日志的选目标/循环对准回归（真机未验证）

首次交付后放置近场单橙、其侧后方绿块、远方黑块及黑块路径中的蓝块。橙块实际
夹爪扫掠与绿块包络分离时，应选择近橙、完成必要对准并进入确认；不得重复
`near_field_group_preview` → `green_collecting_10_point_reference` → 路径拒绝的循环。
将邻居移入橙块扫掠、移到橙块正后方贴邻或替换为蓝色/未知，应拒绝橙块，危险类单列。
测量搜索至开爪时间、无动作时间、重复选中同一不可执行目标次数及端到端观测年龄。
日志K0布局的合成回归不能证明现场像素包络或真实夹获率。
