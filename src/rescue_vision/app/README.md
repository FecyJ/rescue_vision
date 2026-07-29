# `app`：可运行应用装配

本包只放跨相机、通信和运动模块的实际运行入口。当前实现
`rescue-vision-manual-capture`，用于赛外、人员全程监督的低速车载运动采集；
比赛自主应用仍未实现。

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

入口创建但不复制相机参数、运动限值或协议规则。连接后发送会话、视频、车辆
和采集状态，并同时接收 `control/debug/motion` 与
`control/debug/capture`。运动命令继续由 `RemoteMotionExecutor` 校验死手、
有效期和限速；采集命令可独立开始/停止标准 recording schema v4 会话。
手动会话的 `motion.jsonl` 逐条保存运动执行结果、轮速遥测、未知 UART
扩展报文和停车原因；这些事件与图像帧统一使用树莓派应用单调时间。
等待首个客户端及断线重连期间，入口仍以有界周期排空 STM32 主动遥测，避免
未连接时填满 UART 接收队列；等待达到 `--accept-timeout-seconds` 后会先
停车并正常退出，不创建空 recording。

录像写盘队列满、写盘失败、相机异常、UART 异常、远程断线、非法控制或应用
退出都会离开统一运动循环，并在 UART 尚可写时先发送柔和制动。断电、
`SIGKILL` 和 UART 物理断开仍只能由固件看门狗停车。

## 安全边界

当前固件协议没有可验证的看门狗和急停锁存状态。命令因此默认拒绝启动；只有
物理急停已就绪且操作员会全程监督时，才可显式传入
`--supervised-physical-stop-ready`。此时车辆状态如实发布
`safety_mode=supervised_physical_stop`、`watchdog_armed=false` 和
`control_ready=true`，使合规客户端能在持续警告下发送低速命令，而不会把
临时监督条件伪装成固件保护。在固件闭环和真机验收前不得用于比赛运行。
重新连接不会复用旧 TCP 会话或旧死手使能，操作者必须重新发送显式使能命令。

脚本运动模式不属于本入口，已后调到手动采集和固件安全闭环之后。
