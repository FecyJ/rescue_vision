# `app`：可运行应用装配

本包只放跨相机、通信和运动模块的实际运行入口。当前实现
`rescue-vision-manual-capture`，用于赛外、人员全程监督的低速车载运动采集，
也支持单片机未上电时的仅相机远程调试；
比赛自主应用仍未实现。

## 常用入口

| 入口 | 输入与生命周期 | 输出或语义 |
| --- | --- | --- |
| `rescue-vision-manual-capture` | `runtime.yaml`、车端输出根目录和显式监督确认 | 完整装配 TCP/UART/相机/运动/夹爪/记录；推荐生产入口 |
| `build_camera_pipeline()` | 已加载的 `AppConfig` | 创建但不启动配置选择的真机源、可选去畸变和地面映射 |
| `build_session_status()` | 同一 `AppConfig`、服务实例 ID、视频 FPS 和可用模式 | 创建声明运动、夹爪、视频模式、车辆和采集能力的会话状态 |
| `run_manual_capture_session()` | 已启动连接、可选运动/夹爪执行器、采集会话、相机管线和可选 perception/BEV 旁路 | 运行单个完整车辆或仅相机 TCP 会话；断线/故障时清理当前记录 |
| `BevFrameRenderer` | 已加载且包含 BEV 配置的 `GroundProjector`；显式 `start/stop` | 有界丢旧保新的后台 BEV 生成旁路；输出保留源帧号和采集时间 |
| `FieldMapSnapshotRenderer` | `world.static_map` 和 `team_color` | 把唯一静态地图、可选新鲜机器人位姿及未来确认目标覆盖编码为场地图 PNG |
| `LatestCenterCrossLocalization` | 已装配的场地特征检测器、中心十字定位器和有效像素掩码；显式 `start/stop` | 有界最新帧定位旁路；只返回未超过配置时效的唯一 `FieldPose2D` |

`run_manual_capture_session()` 不创建或打开硬件资源。调用方传入
`RemoteMotionExecutor`，并在 `motion.gripper.enabled=true` 时传入共享同一个
`MotionController` 的 `RemoteGripperExecutor`；关闭夹爪能力时传入 `None`。
仅相机会话把两个执行器都传为 `None`，并使用
`build_session_status(camera_only=True)`；两者不得与会话 capability 矛盾。
正式装配由下文 CLI 从运行配置完成，普通调用方不应另写一套设备、限值、
机械端点或协议参数。

## 手动采集入口

先从 `configs/runtime.yaml` 加载唯一运行配置。必须启用相机、UART、motion 和
remote server，并将 `remote.access_mode` 明确设为 `debug_control`：

```bash
rescue-vision-manual-capture \
  --config configs/runtime.yaml \
  --output-root /data/rescue-targets/manual \
  --supervised-physical-stop-ready \
  --video-fps 10
```

入口创建但不复制相机参数、运动限值或协议规则。连接后发送会话、视频、场地图、车辆
和采集状态，并同时接收 `control/debug/motion`、
`control/debug/gripper`、`control/debug/capture` 与不执行动作的
`control/video/mode`。运动命令继续由
`RemoteMotionExecutor` 校验死手、有效期和限速；夹爪命令由
`RemoteGripperExecutor` 校验有效期和布尔扳机状态，再按配置机械端点与
全行程时间渐进下发；采集命令可独立
开始/停止统一 recording 会话。电脑端通过 `video_modes` 选择 `raw`、
`perception` 或 `bev`；perception 由 `PerceptionFrameRenderer` 在最新帧后台
旁路运行 `TargetPoseDetector`，BEV 只在当前地面映射包含 BEV 配置时由独立
最新帧旁路生成。两者只发布带实际模式和坐标元数据的 JPEG，不阻塞运动安全循环。
只要 `world.static_map.regions` 非空，会话就声明并以 500 ms 周期发布
`observation/map/snapshot`；若同时启用可用地面映射、`field_features` 和
`localization`，独立最新帧旁路把中心十字唯一位姿绘制为地图箭头并写入严格
attributes。观测超时或没有唯一解时地图仍发布，但明确清除机器人位置。当前不
包含 IMU/编码器连续推算，离开中心十字可观测区域后不会沿用旧位姿。
手动会话的 `motion.jsonl` 逐条
保存运动/夹爪执行结果、编码器/IMU、系统状态、命令回复和停车原因；
这些事件与图像帧统一使用树莓派应用单调时间。
等待首个客户端及断线重连期间，入口仍以有界周期排空 STM32 主动遥测，避免
未连接时填满 UART 接收队列；等待达到 `--accept-timeout-seconds` 后会先
停车并正常退出，不创建空 recording。

### 仅相机远程调试

只给树莓派和相机上电、STM32/UART/电机均不可用时，使用：

```bash
rescue-vision-manual-capture \
  --config configs/runtime.yaml \
  --output-root /data/rescue-targets/camera-only \
  --camera-only \
  --video-fps 10
```

此模式不打开 UART，也不要求 `uart.enabled`、`motion.enabled` 或
`--supervised-physical-stop-ready`。会话仍提供 raw/perception/BEV 图传和采集
控制，但明确声明 `motion_control=false`、`gripper_control=false`。车端继续按
实际循环通常每 100 ms 发布车辆状态，并把协议最大发布周期声明为 500 ms，
使客户端陈旧阈值为 1500 ms。状态固定报告 `control_ready=false`、
`uart_connected=false`、`safety_mode=unavailable` 和 `stop_reason=uart_fault`，
避免现有客户端因车辆状态陈旧而反复重连，同时不会伪装单片机在线。
若旧客户端没有按 capability 停止周期运动/夹爪心跳，车端会严格解析并安全
丢弃这些合法消息，不会因此关闭仅相机会话；消息不会写 UART 或更新“已应用”
命令 ID。畸形 payload 和未知 control 仍按协议错误处理。

视频、车辆状态和地图是三个独立的最新值 topic，因此
`remote.observation_queue_capacity` 必须至少为 `3`；入口会在启动时拒绝更小
容量，避免图像流把车辆心跳挤出队列。

该模式产生的录像使用已有 `recording_kind=camera`，不创建 `motion.jsonl`；切回
完整车辆调试时不要带 `--camera-only`，并继续满足物理急停和全程监督要求。

录像写盘队列满、写盘失败、相机异常、UART 异常、远程断线、非法控制或应用
退出都会离开统一运动循环，并在 UART 尚可写时先发送柔和制动。断电、
`SIGKILL` 和 UART 物理断开仍只能由固件看门狗停车。
完整车辆模式中的 UART 异常同样会退出；仅相机模式根本不打开 UART，因此不受
未上电单片机影响。

## 安全边界

当前固件协议没有可验证的看门狗和急停锁存状态。命令因此默认拒绝启动；只有
物理急停已就绪且操作员会全程监督时，才可显式传入
`--supervised-physical-stop-ready`。此时车辆状态如实发布
`safety_mode=supervised_physical_stop`、`watchdog_armed=false` 和
`control_ready=true`，使合规客户端能在持续警告下发送低速命令，而不会把
临时监督条件伪装成固件保护。在固件闭环和真机验收前不得用于比赛运行。
重新连接不会复用旧 TCP 会话或旧死手使能，操作者必须重新发送显式使能命令。
夹爪不随运动死手切换，但任一按下状态必须在有效期内持续刷新。松开扳机要
显式发送两个状态均为 `false` 的 v2 命令；命令超时、断线和停车会停止继续改变
角度，不会自动开爪或闭爪，也不会在重连后重放旧命令。只有
`motion.gripper` 的机械开合端点和全行程时间已由当前车辆实测并启用时，会话
才声明 `gripper_control=true`。车辆状态中的舵机角度来自 STM32
`CarState`，不代表有物理位置传感器或已经完成抓取。

脚本运动模式不属于本入口，已后调到手动采集和固件安全闭环之后。
