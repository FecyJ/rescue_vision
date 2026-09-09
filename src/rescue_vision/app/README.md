# `app`：可运行应用装配

正式比赛流程由 `MatchSequence` 提供纯逻辑状态机，由 `match_runtime.py` 负责相机、
Hailo、UART、夹爪、观察发布和资源生命周期。纯逻辑对象不创建硬件资源；硬件旁路只
提交最新观测，不阻塞运动控制循环。

## 常用入口

| 入口 | 配置和用途 |
| --- | --- |
| `rescue-vision-match` | 使用 `configs/runtime.match.yaml` 运行正式流程；支持 `--start-area 2/3`、本地预览、observe-only perception 图传和流程日志 |
| `rescue-vision-match-cc` | 使用 `configs/runtime.cc.yaml` 运行 CC 流程；启动、CLI、日志和安全区运输与 match 相同，其余搜索/解团/夹取策略独立 |
| `rescue-vision-grab-transport` | 使用同一配置，从 `(0,0,+90°)` 搜索绿色单物块，并在安全区末端保持张爪推送，不执行解团 |
| `rescue-vision-cluster-breakup` | 赛外固定解团试验，不是正式比赛入口 |
| `rescue-vision-manual-capture` | 受监督手动驾驶、图传和数据采集 |
| `rescue-vision-green-grab` | 无地面标定时的绿色像素居中抓取试验 |
| `rescue-vision-gripper-width` | 受监督的绿黑多目标或单橙色近场收拢；合爪后保持，不掉头 |
| `rescue-vision-motion-sequence` | 受监督 TUI 定距动作：输入 `a1/a2`，自动规划并前进 1.5 m |

## `rescue-vision-match-cc` 流程

CC 流程要求团内每个物块在 K0 地面坐标中都至少有两个不超过 100 mm 的团内邻居。
首次发现团后停车，并按原 match 解团方式逐帧重新成团：收集5帧团中心和最近正向
K0 的平均值，不绑定首次 tracker ID；找不到符合条件的团最多等待 1000 ms，超时重新
旋转搜索。随后按锁定平均中心只对准一次，接近到固定前距，再执行固定距离前推、张爪、
固定距离退出和合爪；舵机与底盘运动指令之间仍至少间隔 20 ms。之后只选择原点至目标
K0 线段净空的绿色单块，并在
`y=0±5 mm` 内锁定；450 mm 外持续更新坐标接近，进入近场后按 `x-150 mm`
定距、使用 `TRANSPORT` 夹爪姿态并合爪。

首次运输退出后每轮依次整圈搜索单橙、单黑/绿和含橙/黑/绿的团。橙色通道宽度来自
与 `gripper_width` 相同的 ROI 颜色掩码地面投影，近场抓取复用同一非对称夹爪反解；
单黑/绿阶段优先黑色、再按距离排序，团阶段按橙、黑、绿优先级选择。橙色安全区
运输终点的 FieldPoint 绝对 `x` 始终为正，其余运输和退出动作沿用 match。

```bash
rescue-vision-match-cc \
  --config configs/runtime.cc.yaml \
  --start-area 2 \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

## `rescue-vision-motion-sequence` 使用方法

该入口默认只打开配置中的 UART 和底盘控制器；增加 `--local-preview` 后，会启动相机和
Hailo Pose，并在 OpenCV 窗口显示最新 perception 识别叠加帧，不启动远程服务。启动前必须
确认车辆处于架空/安全测试环境，物理急停可用，并由操作员全程监督：

```bash
rescue-vision-motion-sequence \
  --config configs/runtime.match.yaml \
  --distance-m 1.5 \
  --supervised-physical-stop-ready \
  --local-preview
```

TUI 输入 `a1`、`a2`：单位均为 m/s²，`a2` 输入正的减速度幅值。`--distance-m` 配置
目标路程，默认 1.5 m。程序按目标路程自动计算三角速度曲线的峰值速度、加速时间和减速
时间，并逐周期下发目标速度。
该入口不采用 `motion` 配置中的最大速度、最大加速度和最小非零速度限制，但仍拒绝超过
STM32 线路 int16 可编码范围的轮速；运行中按 `q` 或 `Esc` 会软刹车退出。该入口是赛外受监督
动作试验，不是正式比赛流程；真车动作前仍需验证轮距、轮向、加速度、制动距离和急停。

## `rescue-vision-gripper-width` 使用方法

该入口在近场自动选择 1～3 个绿色/黑色物资，或单个橙色目标，执行「对准目标 → 静止复核
→ 张爪 → 编码器定距前进 → 合爪 → 完成」。正式 `match` 在 450 mm 近场交接后复用
同一状态机，并按运输趟次把首轮策略限制为最近的单个绿色物资；最近绿色的前进走廊被
蓝/橙/黑/未知目标阻挡时直接进入解团，不改选更远目标。正式流程远场对准和接近保持闭爪，
进入近场后才按计划的实际舵机映射张开；本独立入口不处理运输或交付。
完成仅表示动作指令与计时完成，`capture_confirmed=False`。

```bash
rescue-vision-gripper-width \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs/near_field
```

合爪后底盘保持静止，夹爪保持最后角度，不自动掉头、不搜索下一组。`--once` 在完成后
退出；普通模式等待 `q`、`Esc` 或 `Ctrl+C`。确认帧数只从 YAML 的
`near_field_grasp.confirmation_frames` 读取。旧 CLI 的 `--center-y-half-range-mm`、`--clearance-mm`、
`--min-mask-pixels` 已迁入 `near_field_grasp.center_tolerance_mm`、`clearance_mm`、
`min_mask_pixels`；不保留第二套参数来源。`--log-dir` 指定按分钟命名的
`gripper_width_YYYYmmdd_HHMM.log` 目录，默认是当前目录下的 `logs`；日志同时保留在终端。

准备线程在候选几何状态变化时立即记录 `grasp_candidate`，稳定状态最多每秒汇总一次，包含 `track_id`、类别、帧号、
采集时间、是否可选、`x0_mm`、`x1_mm`、`depth_mm`、`k0_x_mm`、`opening_mm`、
`target_final_x_mm`、`corridor_start_x_mm`、`corridor_end_x_mm`、
`corridor_half_width_mm`、`forward_distance_mm`、对准角和拒绝原因。`x0_mm`、`x1_mm`、
`depth_mm` 仅用于诊断颜色掩码的纵向投影，不再表示目标自身的硬容纳门禁；不可用地面
包络的目标也会记录，但几何字段为 `none`。

### 选组与试验几何

配置权威为 `near_field_grasp`，代码默认近场半径和最大行程均为 450 mm；正式 match 配置
可按现场标定调整（当前配置均为 450 mm），最多 3 个物资。
为限制后台枚举量，最多从最近 12 个可选物资建立组合（可配置为不超过 20 个）；
其余所有检测仍参加障碍检查或顺带纳入，不会因候选截断而被忽略。

单目标测量通过唯一 `GroundProjector` 投影 ROI 颜色掩码，并包含 K0。整组宽度取
最左至最右边界的跨度，包含空隙；对准依据横向边界中点和成员纵向中心的中点。
对准后只用横向包络决定开口，不要求成员初始纵向包络整体落入夹爪深度，也不启用
`depth_spread_exceeded`。默认总开口余量 4 mm；开口宽度为
`y1 - y0 + clearance_mm`，其中左夹爪目标位置为 `y1 + clearance_mm/2`，右夹爪
目标位置为 `y0 - clearance_mm/2`，两侧分别反解舵机角度，不再用对称开角。目标组进入
对准允许范围由 `near_field_grasp.center_tolerance_mm` 配置（当前 match 为 20 mm，边界包含）。
进入范围后由唯一确认窗口按 `confirmation_frames` 个不同有效帧确认；确认计时和必要对准共同受
`alignment_timeout_ms` 限时。提交开爪前按下文的实际静止证据、观测失联上限和准备结果年龄检查几何有效性。已确认的组使用 `alignment_hysteresis_mm` 的滞回范围，锁定后固定旋转方向；
数据短缺到期回到带原因的搜索/重选，只有明确几何阻挡才进入解团；恢复链最多远距重接近一次。
预测旋转后的走廊会提前检查已知危险目标。任一侧超过安全舵机行程时淘汰，
不截断。精对准使用 `match.green_alignment_min_wheel_velocity_m_s` 的单次轮速下限；
进入 `fine_alignment_zone_rad`（当前 0.08 rad）后改用
`fine_alignment_min_wheel_velocity_m_s`，当前 match 配置为 0 m/s，允许误差收敛时自然减速。
启动、搜索、直行和运输等普通动作仍使用 `motion.min_wheel_velocity_m_s`。

选择器依次把不同物资作为最远 X 锚点，分别生成不同长度的扫掠走廊；橙色目标不参与
多成员组合，只生成单目标计划。每个走廊会把实际进入其中的合法物资闭包纳入。通过
危险、容量、开口和边界硬门禁后，先最大化成员数量，再按配置权重比较净空、行程和对准角。
绿/黑前进行程使用对准后的最远成员 K0：
`forward_distance = max(0, max(member.K0.x) - target_final_x_mm)`；单个橙色目标使用
旋转后颜色包络的最近端 `x0` 和最远端 `x1`：`d = x1 - x0`，
`forward_distance = max(0, x0 + d - orange_target_final_x_mm)`。这允许目标在揽入
过程中相互滑动/转动；`max_forward_distance_mm` 仍限制底盘动作时长。危险检查使用单个
矩形走廊，纵向范围为
`[corridor_start_x_mm, corridor_start_x_mm + forward_distance]`，横向范围为
`[right_tip_y - corridor_lateral_margin_mm, left_tip_y + corridor_lateral_margin_mm]`。
走廊只覆盖夹爪张开宽度将要扫过的区域，不再覆盖车体或夹臂的历史全扫掠包络。

绿/黑候选只对蓝色侧邻执行额外门禁：在预测抓取方向的 `x′/y′` 轴上，只有蓝色与候选
成员的纵向中心差不超过 `side_neighbor_longitudinal_margin_mm`、横向中心差不超过
`side_neighbor_lateral_margin_mm`，且横向确实存在分离时，才判为侧邻；明显前后错开的
目标不触发。受蓝色侧邻影响的候选直接淘汰；若没有其它安全方案，正式路由进入解团。
橙色侧邻不影响绿/黑计划；橙色独立性门禁只约束单橙计划。纯绿黑目标即使分布密集，
只要没有蓝色侧邻影响，仍允许最多三个成员的多目标计划。

蓝色、未知、类别冲突及不确定目标的 K0 落入走廊时直接拒绝，不能用评分抵消；橙色只能
作为单目标计划，不能作为额外成员加入绿黑组合。橙色目标地面中心周围
`orange_isolation_radius_mm` 内有当前可定位的其它目标时拒绝；其它
目标未观测或没有当前地面点时不无条件推定其位于禁区。额外绿黑的 K0 落入走廊时加入组合并
重新检查数量和横向开口；蓝色、未知、类别
冲突及不确定目标只在静止规划阶段影响选组。最终确认通过并提交计划后不再用运动中的检测重检走廊。规划
停车后的近场走廊门禁只使用当前帧仍有可靠 K0 地面点的目标；当前帧缺少地面点或已经
失观的历史轨迹不参与该走廊阻挡，避免侧方静态目标的历史膨胀包络制造幽灵障碍。全局
tracker 仍保留短时历史，明确危险证据也继续在轨迹存续期内保留，但二者不替代近场停车
窗口所需的当前空间证据。

橙色从候选生成阶段起只形成单目标方案；绿/黑只彼此组合。橙色落入绿黑走廊、或绿黑落入
橙色走廊时都只作为阻挡物，不能顺带加入另一类方案。远场 handoff prior 只唯一匹配关联门限内
K0 最近的局部目标；首轮单绿把该目标作为排他候选。一般阶段先按规则分和单橙优先级排序，
再在同一规则层内优先选择包含该目标的合法组合，最后比较净空、行程和对准角：绿色 5 分、黑色 10 分、橙色 15 分；规则总分较高者优先。同为 15 分时，
单橙优先权重必须大于其余次级权重之和，因此单橙严格优先于三绿或一绿一黑。只有规则总分和
单橙层级相同时，才继续比较成员数量、净空、行程和对准角。

| 次级特征 | 默认权重 | 特征值 |
| --- | ---: | --- |
| 单橙优先 | 0.55 | 单橙为 1，其余为 0 |
| 数量 | 0.10 | 数量 / 3 |
| 净空 | 0.15 | 扣除余量后的最小净空 / 100 mm，上限 1；无邻近障碍取 1 |
| 距离 | 0.12 | `1 − 行程 / 最大行程`，截断至 `[0,1]` |
| 对准 | 0.08 | `1 − |转角| / (π/2)`，截断至 `[0,1]` |

### 可复用接口、时间与生命周期

`measure_target_envelope()` 输出单目标 `TargetGroundEnvelope`，不做居中或规则选择。
`NearFieldGraspSelector.select()` 输入带 tracker ID 的目标和可选的
`NearFieldGraspPolicy`，输出 `GraspSelection`，包含
可选 `NearFieldGraspPlan` 和拒绝原因。计划包含成员、帧号、采集时间、对准点、组合边界、
开口/最大开口、舵机角度、行程、评分分项和动作区域。所有地面点为机器人系 `GroundPoint`，
单位 mm；时间为相机与应用共用的单调时钟 ns。`bounds` 是计划对准后的机器人系，
`regions` 是该计划采集帧的机器人系，不能当作场地坐标。

`GraspPreparationSession` 串行消费新帧，复用 `GraspTargetTracker` 和现有 tracker。
正式 match 在刹车 settle 完成后打开唯一确认窗口：同一目标/组合 ID 的不同有效帧按
`confirmation_frames` 累计，同一帧不会重复计数；控制循环更快时保留已有进度。首轮
的 handoff prior 只把远场已选定的单绿交给近场，邻近绿块不能接管；一般阶段继续由
选择器按规则分值选择绿/黑组合或单橙。确认期间同一 ID 的最新有效几何更新开口和行程，
出现明确危险或不合法组合才使确认失效，短暂漏检只等待当前 ID 的新证据。
一般阶段交接目标旁有当前可选但尚未完成局部轨迹确认的绿/黑物资时，先在原有有界
窗口内等待相邻物资确认，再选组锁定；首轮单绿不使用此等待。无关候选的拒绝诊断
不清空合法计划的确认。静态场界/安全区路径失败返回搜索重选，不触发解团；入口
路径检查复用近场静态边界门禁，失败日志附估计位置、航向、行程、边界和余量。
开爪提交时冻结执行计划；提交门禁分别记录计划采集年龄和准备结果年龄，过期计划不能
冒充新结果。
生产入口在每个真实 UART `OdometryImu` 到达时调用
`MatchSequence.observe_grasp_motion(message)`；独立入口调用
`GripperWidthPickupSequence.observe_motion(message)`，随后再调用 `step()`。
静止证据要求两轮原始计数不变、IMU有效且 `abs(gyro_z)` 不超过
`near_field_grasp.stationary_max_gyro_rad_s`（默认 0.03 rad/s）；无效、饱和、重复/倒退
样本及超过 `grasp_commit_max_observation_age_ms` 的遥测间断均打断静止区间。
当前计划必须在已证实静止之后采集，且直到当前遥测一直静止；其真实采集年龄使用
`processing.max_observation_age_ms` 作为失联上限（match 配置为800 ms），不再强制低于150 ms。
计划必须来自当前准备快照且成员实际被观测，不能给缓存旧计划换发布时间续期。
准备完成后的排队年龄和最近遥测年龄仍受150 ms配置限制；确认总预算不因新帧重置。
未注入遥测的纯逻辑调用只能使用原有短龄计划路径，不能使用静止有效期。
日志增加 `stationary_since_ms`、`motion_age_ms`、`gyro_z_rad_s`，用于区分正常处理延迟、
底盘仍在移动和遥测断档。零速命令不是静止证据。

NearFieldGraspConfig 默认值与该档位一致。确认只使用同一成员 ID 的当前有效几何，
不跨目标或跨身份混合。
近场锁定后固定旋转方向；成员身份在本次尝试内保持不变，提交前使用同一 ID 的最新有效几何
更新开口和行程。偏置目标只按组几何生成对准预览，不使用预测走廊提前拒绝；旋转完成后
重新计算横向组宽、夹爪开口和前进走廊，再做完整障碍检查。
开爪前失效会解锁重选，限定时间仍无有效组则回到带原因的正式搜索，
不会持续停在 `ABORTED`。单帧未知/普通质量异常进入可恢复疑似态，连续
`supply_recovery_frames` 个干净可抓目标观测后恢复；明确危险证据保持到轨迹消失。
确认成功后短暂漏检或旁路异常不单独取消动作。
正式流程为每趟准备结果附加会话 ID；首趟通过 `NearFieldGraspPolicy` 限制为单个绿色，
后续趟次允许绿色/黑色 1～3 个或单独 1 个橙色伤员。`MatchDecision` 可携带精确双舵机角度、
单次轮速下限和 `soft_brake` 意图，运行时据此保持动态开口并避免重复刷写命令。

生产装配见 `gripper_width._run()` 和 `match_runtime._run_hardware()`：从
`configs/runtime.match.yaml` 构造唯一几何对象、
`GraspTargetTracker(config.tracking.build_tracker(), projector, config.near_field_grasp)`、
`NearFieldGraspSelector` 和 `GraspPreparationSession`。后两者不打开硬件。准备器放在
单个后台线程串行调用 `update(snapshot, locked_ids=..., handoff_prior=...)`，
队列只保留最新一帧。全候选诊断独立线程使用一个最新结果槽，最多每秒计算一次；
退出上下文时唤醒并回收准备/诊断线程。正式流程还会校验准备结果的近场会话 ID，迟到结果不会跨运输轮次
驱动动作。运动层在自己的循环调用
`GripperWidthPickupSequence.step(now_ns, preparation, cumulative_distance_m=...)`，将返回的
速度/角度意图交给已装配的控制器。`GripperWidthPickupResult` 返回成员 ID/类别、完成时间、
最终舵机命令及未确认收拢标志；旧单目标 `GripperWidthPickupPlan` 已替换为多目标计划。

开爪提交时冻结唯一确认窗口形成的执行计划，不再把运动中的目标检测送入走廊或成员包络复核，
因此运动模糊、`unknown` 或目标遮挡不会单独打断已提交动作。编码器仍按动作计划定距，
并保留关键里程计不可用、编码器方向错误和持续无前进进度时的退出保护；明确急停、硬件
健康故障和进程异常仍走软刹车路径。开爪阶段底盘保持静止，张爪计时结束后直接按冻结
计划进入 `forward_open_loop`。动作退出后必须用退出之后采集的新证据重新选组，不能直接
重放旧计划。
合爪指令提交后只等待计时并保持底盘静止，遮挡不会自行重新抓取；不把视觉消失当成成功
证据。`rx_degraded` 只保留在日志诊断中。

`GraspPreparation` 的 `confirmation_progress`、计划的 `capture_timestamp_ns` 和
`prepared_timestamp_ns` 分别用于确认进度、计划年龄和准备结果年龄，
`result_timestamp_ns` 保留推理结果的真实完成时刻；
两种年龄均来自单调时钟，不能由控制循环改写。

相机/推理、规划、预览和有界终端日志均在旁路。GUI 或日志失败被报告并隔离；规划或感知
不可用时由动作层根据剩余有效证据决定退出。退出先软刹车，再结束准备线程、相机、
UART 并恢复信号处理器，夹爪保持最后一次显式命令。预览标记成员 ID、禁止卷入目标、
动作包络、开口、最大开口和评分；不代表已经验证无碰撞。

运行前必须检查相机标定、Hailo 模型、UART、双舵机安全端点、里程计和轮距。
4 mm 总开口余量、`target_final_x_mm`、`corridor_start_x_mm` 和 10 mm 横向走廊余量都是
试验初值。掩码含离地表面、真实夹臂厚度、车头形状、目标滑动和有效前进行程尚未实测
验证，不能用这些近似几何证明真实抓取安全。实物收拢效果、危险类表现和目标设备延迟
均为**未验证**；验收步骤见 `manual_tests/README.md` 的近场收拢章节。

两个正式流程入口都要求显式的物理急停和全程监督确认，直到固件看门狗和急停闭环完成
真车验收。正常退出、信号、相机、UART、网络或线程异常都进入软刹车、停止旁路和资源
释放路径。

## `rescue-vision-match` 使用方法

先确认 `configs/runtime.match.yaml` 已按当前车辆填写，并满足以下装配条件：

- `match.enabled: true`；
- `motion`、`motion.odometry` 和 `motion.gripper` 已启用并完成机械标定；
- `geometry.ground_mapping_enabled: true` 且地面标定与当前相机分辨率匹配；
- `hailo.enabled: true`，模型路径和六类 v3 输出顺序有效；
- `remote.enabled: true` 时使用 `role: server` 与 `access_mode: observe_only`。

在仓库根目录安装后运行：

```bash
python -m pip install -e '.[dev]'

rescue-vision-match \
  --config configs/runtime.match.yaml \
  --start-area 2 \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

参数含义如下：

| 参数 | 作用 |
| --- | --- |
| `--config PATH` | 必填；正式流程 YAML，通常为 `configs/runtime.match.yaml` |
| `--start-area {2,3}` | 启动区域；2 为地图右上角红方，3 为地图左下角蓝方，默认 2；3 会把场地 `FieldPoint` 和初始位姿绕 `(0,0)` 中心对称 |
| `--supervised-physical-stop-ready` | 确认物理急停已就绪且操作者全程监督；固件看门狗未完成验收前必须提供 |
| `--local-preview` | 打开本地 OpenCV 预览，显示最新感知帧、状态和动作原因；按 `Q`/`Esc` 退出 |
| `--jpeg-quality N` | 观察图传 JPEG 质量，范围 `1..100`，默认 `80` |
| `--observer-image-interval-seconds S` | observe-only 图传最小发布间隔，必须为正数，默认 `1.0` |
| `--log-dir PATH` | 按时间写入流程日志和 D2 遥测，默认 `logs/`；传入 `--log-dir /dev/null` 不适用，需使用有效目录 |

流程启动后会先执行预检、启动转向/直行和目标搜索。搜索态的机会检查使用固定
`TRANSPORT` 夹爪包络的前向走廊：首轮只选择最近的单个绿色 K0；若其走廊被蓝/橙/黑/未知目标阻挡，直接进入解团，不改选更远绿块；一般阶段支持绿/黑物资组或单个橙色伤员快速入口。固定走廊外的合法目标先生成
近场组预览并旋转对准，只有没有合法预览方案时才继续目标团解团。远场和近场等待阶段保持 `CLOSED`，不会先发固定开口；近场
计划 ready 后才发送动态开度。规划期间的同帧画面叠加半透明走廊：黄色为对准预览、
青色为复核、绿色为 ready、红色为阻挡；match 中任何已从多个候选中选出的远场目标、解团目标团或近场成员框，
均以橙色粗线和 `SEL#track_id` 标记，
组中心显示十字并沿对准方向画箭头。本地预览和观察图传使用同一叠加帧。终端状态行包含当前状态、动作原因、
编码器距离、航向、目标轮速、绿色走廊诊断、安全区阶段和 D2 遥测丢弃计数。观察图传、
地图状态和本地预览只读取最新数据，不参与运动决策。

安全区末段的绿/黑物资使用 `match.safe_zone_d2_to_final_speed_m_s` 和
`match.safe_zone_d2_to_final_braking_overrun_mm`；单个橙色伤员使用独立的
`match.safe_zone_orange_d2_to_final_speed_m_s` 和
`match.safe_zone_orange_d2_to_final_braking_overrun_mm`。解团阶段的临时最大轮加速度由
`match.breakup_max_wheel_acceleration_m_s2` 配置，`null` 表示继承 `motion` 全局值。

正常结束、`Ctrl+C`、`SIGTERM`、相机/Hailo/UART/网络异常或旁路线程失败都会进入统一
软刹车清理路径。重新运行前应确认车辆已停稳、急停状态已复位，并重新提供显式监督确认。

如果只需联调绿色物块夹取、末端张爪推送和安全区运输，使用 `rescue-vision-grab-transport`；该入口复用
同一配置，从 `(0,0,+90°)` 直接搜索固定 `TRANSPORT` 走廊内的单个绿色 K0，不执行目标团解团。

## `rescue-vision-grab-transport` 使用方法

该入口用于在目标已经摆散时单独联调绿色物资夹取、d1 视觉纠偏、安全区末端张爪推送和退出。
车辆按正式流程先夹取并运输；到达安全区末端后不再重复闭合夹爪，而是保持张开把物块推入安全区，
再保持张开退出。它仍要求与正式流程相同的
Hailo、地面映射、UART、里程计和夹爪标定；配置检查仍使用
`match.enabled` 以及同一组运动安全限制。
末端推进不再重复闭合夹爪；前段夹取、d2 张爪、车辆停稳、视觉校准和路径安全门禁仍按原流程保留。

```bash
rescue-vision-grab-transport \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

入口会覆盖本次联调的受限初始位姿：场地 `FieldPoint(0, 0)`、航向 `+90°`，跳过正式
流程的固定启动转向/直行，预检通过后立即进入绿色目标搜索。只接受固定 `TRANSPORT`
开口前向走廊内的单个绿色 K0，并继续检查安全区和场地边界；搜索不到合格目标时保持原地搜索，
不会转入目标团解团。

命令行参数与 `rescue-vision-match` 的通用参数相同；`--start-area` 仅用于正式比赛入口。
`--config` 指定正式流程配置，
`--supervised-physical-stop-ready` 确认物理急停和全程监督，`--local-preview` 打开本地
预览，`--jpeg-quality` 和 `--observer-image-interval-seconds` 控制观察图传，
`--log-dir` 指定流程日志和 D2 遥测目录。正常退出、`Ctrl+C`、信号或硬件/旁路异常均
使用同一软刹车和资源清理路径。

## 正式流程行为

`MatchSequence` 以 `GroundPoint`、陀螺仪航向和编码器累计距离完成：

- 启动转向/直行，随后搜索目标团；首轮只抢占最近的可靠单绿，走廊不安全时直接解团，一般阶段优先尝试可抓取物资组。
- 完整目标团按绿色成员和成员数排序；候选的场界、双方安全区和机器人包络门禁失败时
  继续尝试下一完整团。
- 刹车 settle 完成后进入唯一确认窗口；同一目标 ID 的不同有效帧完成确认，远目标先接近到
  `near_field_grasp.max_range_mm`，再交给近场宽度抓取状态机完成选组、必要对准、动态开爪、定距前进和闭爪。
  数据短缺或结果延迟在总预算内重新观察，只有真实几何阻挡才进入解团。
- 首次运输限制为单个绿色普通物资；首次交付后允许绿色/黑色物资 1～3 个，或单独转运 1 个橙色伤员。蓝色、
  未知和不确定目标只有在当前帧有可靠地面点且实际进入近场走廊时才作为障碍；橙色隔离半径内的邻居需要按实际包络复核；危险、未知或扫掠侵入仍拒绝橙色计划，缺少 K0 的无关目标不参与判定。
- 抓取结果为橙色时路线使用 `match.safe_zone_injured_target_field_mm`；绿/黑使用 `match.safe_zone_fallback_target_field_mm`。
  选择 `--start-area 3` 时这些终点和 d1/d2 直线会相对中心十字对称到负 y 侧，动作切换由零速保持和陀螺仪航向保持保护。
- 夹取后若 y 已沿己方方向越过 d1，则原地开始纠偏，否则沿 d1 直线前进。先确保安全区
  bbox 完整在画面内，再以 K0/K1/K2 和静态红蓝安全区角点拟合 `FieldPose2D` 并覆盖当前
  航位；d2 与末段继续沿校正位姿积分。释放并直线退出后再次完成同样纠偏，直接进入搜索。

视觉纠偏只使用 d1 停稳期间的当前安全区观测。正式流程当前不接入跨帧视野外目标记忆、
采集时刻位姿对齐、记忆目标接管或旧的 20 分编排。

## 纯逻辑装配

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.match.yaml")
flow = config.build_match_sequence()
```

联调入口的类型和装配方法为 `GrabTransportSequence` 与
`config.build_grab_transport_sequence()`。它覆盖初始场地位置和航向，仍复用正式流程
配置、跟踪器、夹爪标定和安全区视觉纠偏。

## 可复用但未接入的模块

`scan_target_memory.py` 保存带场地点和老化时间的可靠目标，可在未来正式流程明确采集
时刻位姿策略后接入；当前不参与任何运动决策。`field_target_cluster.py` 提供带身份的
`FieldPoint` 连通聚类，也只保留为可复用逻辑。两者均有独立 pytest，不能被解释为已接入
正式比赛能力。

## 观察与日志

`match_observers.py` 提供不修改原帧的本地叠加和 observe-only 远程发布；
`session_log.py` 把标准流 tee 到按时间命名的日志文件。D2 到安全区末段的
`D2TelemetryLogger` 逐帧记录编码器/IMU 原始数据、时间差、阶段和丢弃计数，旁路队列
拥塞时不阻塞运动循环。

正式流程设计、状态顺序和视觉纠偏验收见 [`docs/正式流程设计.md`](../../../docs/正式流程设计.md)。

### 一般阶段选目标与重复等待修复

正式 `match` 首次有效交付后统一比较固定走廊内外的接近种子：近场范围内优先，
同层按配置的类别分值降序、距离升序。进入近场后仍由同一选择器按组合总分确定
绿黑1～3个或单橙；首轮单绿策略不变。种子、重接近和对准复核复用目标方向走廊，
仅物资组可忽略可共同收拢的绿/黑成员，伤员路径不能忽略其它目标。
正式入口的已确认跟踪点直接用于必要对准，不重复等待十帧；最后仍执行唯一近场确认。

`orange_isolation_radius_mm` 数值未降低。入口只把可信侧后方绿/黑邻居交给精细复核，
不据此提交开爪。`NearFieldGraspSelector` 要求邻居完整当前包络全部位于目标方向的
橙色中心后方，横向完全位于夹爪扫掠宽度及配置余量之外，且两物体包络距离大于
`clearance_mm`。蓝色、未知、质量异常、缺包络或侵入者不使用例外。
这修正的是圆形邻近区把侧后方物资误判成混运的工程策略，不改变伤员单独运输规则。
