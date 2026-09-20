# `app`：可运行应用装配

正式比赛流程由 `MatchSequence` 提供纯逻辑状态机，由 `match_runtime.py` 负责相机、
Hailo、UART、夹爪、观察发布和资源生命周期。纯逻辑对象不创建硬件资源；硬件旁路只
提交最新观测，不阻塞运动控制循环。

运动下发由 `match_runtime._apply_match_motion()` 消费 `MatchDecision`，返回连续制动标志；
一次制动只发送一次 `SOFT_BRAKE`，日志内容与状态切换不触发重复同步。调用方仍每周期
处理 UART 并调用 `controller.update()`，以确认应答及刷新零轮速心跳，真实停稳使用编码器/IMU。
解团终点已锁存，里程回落不会重新驱动；停稳失败有明确截止时间和 `breakup_stop_unconfirmed` 原因。

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
| `rescue-vision-motion-sequence` | 独立的受监督 TUI 定距动作试验（不负责 `match_nb` 开场） |
| `rescue-vision-teach-replay` | 手推示教时记录编码器/IMU JSONL，并按左右轮轨迹受监督回放 |

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

## `rescue-vision-teach-replay` 使用方法

该入口打开 UART 后保持软刹车，只读取 100 Hz 编码器/IMU。手动缓慢推动车辆，按 Enter
或 `Ctrl+C` 结束录制；程序把每个原始样本、设备/本机时间戳和质量位写入独立 JSONL，
先打印日志路径，再等待输入 `REPLAY`。确认后按约 50 ms 窗口计算左右轮平均速度并回放；
回放中按 Enter 可随时软刹车：

```bash
rescue-vision-teach-replay \
  --config configs/runtime.match.yaml \
  --log-dir logs \
  --supervised-physical-stop-ready
```

以后可直接复用日志；`--speed-scale 0.5` 会把速度减半、时间加倍，轮行程保持不变：

```bash
rescue-vision-teach-replay \
  --config configs/runtime.match.yaml \
  --replay logs/teach_replay_YYYYmmdd_HHMMSS_ffffff.jsonl \
  --speed-scale 0.5 \
  --use-gyro \
  --supervised-physical-stop-ready
```

日志会绑定编码器每转计数、左右轮半径和陀螺仪方向；当前配置不匹配、编码器质量无效、
采样溢出、设备时间倒退或派生轮速超过配置上限时，程序在电机动作前拒绝回放。
`--no-use-gyro` 是默认行为，只按编码器差分开环回放；`--use-gyro` 会积分日志与回放现场的
Z 轴角速度，用相对航向误差对左右轮作有界差速修正，修正仍受运行配置的角速度和单轮速度
上限约束。日志中的零散无效 IMU 样本会被明确计数并跳过；回放现场短暂无效时暂停
航向修正并重新建立积分基线，连续不可用超过 200 ms 或设备时间不递增才软刹车。地面打滑、
零偏变化、人工推动速度尖峰、电机死区和控制器加速度限制仍会
造成路径误差，该入口不能代替完整定位或无人监督运行。

## `rescue-vision-gripper-width` 使用方法

该入口在近场自动选择 1～3 个绿色/黑色物资，或单个橙色目标，执行「选择当前可达目标 → 静止复核
→ 张爪 → 编码器定距前进 → 合爪 → 完成」。正式 `match` 在 450 mm 近场交接后复用
同一状态机，并按运输趟次把首轮策略限制为单个绿色物资；最近绿色的前进走廊被
蓝色危险/橙色/黑色目标阻挡时先尝试其它安全绿色候选，没有合法抓取方案才进入局部解团。正式流程远场对准和接近保持闭爪，
进入近场后才按计划的实际舵机映射张开；本独立入口不处理运输或交付。
完成仅表示动作指令与计时完成，`capture_confirmed=False`。
近场编码器定距的最后 50 mm 按
`match.pickup_terminal_speed_gain_s_inv × 剩余距离` 收速；默认增益 `1.0 s^-1`，
最低目标速度为 `0.005 m/s`。该字段也用于正式流程进入近场前的精细接近。

```bash
rescue-vision-gripper-width \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs/near_field
```

合爪后底盘保持静止，夹爪保持最后角度，不自动掉头、不搜索下一组。`--once` 在完成后
退出；普通模式等待 `q`、`Esc` 或 `Ctrl+C`。停稳后复核帧数只从 YAML 的
`near_field_grasp.confirmation_frames` 读取，正式配置为 1。旧 CLI 的 `--clearance-mm`、
`--min-mask-pixels` 已迁入 `near_field_grasp.clearance_mm`、
`min_mask_pixels`；不保留第二套参数来源。`--log-dir` 指定按分钟命名的
`gripper_width_YYYYmmdd_HHMM.log` 目录，默认是当前目录下的 `logs`；日志同时保留在终端。

准备线程在候选几何状态变化时立即记录 `grasp_candidate`，稳定状态最多每秒汇总一次，包含 `track_id`、类别、帧号、
采集时间、是否可选、`x0_mm`、`x1_mm`、`depth_mm`、`k0_x_mm`、`opening_mm`、
`target_final_x_mm`（贪心补夹时使用 `greedy_target_final_x_mm` 的实际生效值）、`corridor_start_x_mm`、`corridor_end_x_mm`、
`corridor_half_width_mm`、`forward_distance_mm`、对准角和拒绝原因。`x0_mm`、`x1_mm`、
`depth_mm` 仅用于诊断优先颜色掩码或 bbox 底边形成的纵向投影，不再表示目标自身的
硬容纳门禁；缺少 `GroundProjector` 时几何字段为 `none`。

### 选组与试验几何

配置权威为 `near_field_grasp`，近场半径是单次实际收拢的接管条件，不是补夹的全场发现半径；
正式 match 可按现场标定调整，最多 3 个物资。远场补夹目标先沿正常入口接近，进入该半径后
再由近场收拢接管。
为限制后台枚举量，最多从最近 12 个可选物资建立组合（可配置为不超过 20 个）；
其余所有检测仍参加障碍检查或顺带纳入，不会因候选截断而被忽略。

单目标测量优先通过唯一 `GroundProjector` 投影 ROI 颜色掩码；颜色不足或歧义时使用
bbox 底边横向边界，并包含模型 K0 或 bbox 底边中点兜底。整组宽度取
候选包络最左至最右边界的跨度，包含空隙。近场先在当前朝向计算左右边界、独立开度、真实
前进行程和扫掠走廊；若当前朝向已物理可达，包络中心不论是否在 ±5 mm 内都直接进入停稳
提交。仅当前朝向不可达时才选择最小必要转角，也不要求成员
纵向包络整体落入夹爪深度。默认总开口余量 4 mm；开口宽度为
`y1 - y0 + clearance_mm`，其中左夹爪目标位置为 `y1 + clearance_mm/2`，右夹爪
目标位置为 `y0 - clearance_mm/2`，两侧分别反解舵机角度，不再用对称开角。任一侧边界
超过安全舵机行程时直接淘汰，不截断，也不通过超出真实边界的开度伪造可抓候选。必要转角
由 IMU 闭环执行一次有界粗转，进入可夹范围即制动，最多再做一次有依据的修正；没有新计划
时不会持续按上一帧视觉角旋转，`alignment_continue_max_age_ms` 只作为无运动历史时的短
动作上限。
候选通过后立即制动；
真实停稳后的一个当前新帧重新执行原始包络、危险、走廊和静态路径门禁，通过后直接
开爪。`alignment_timeout_ms` 只限制停稳和当前异步结果的总提交时间，选不出方案的空等由
`no_plan_wait_ms` 单独限制；正式配置
`confirmation_frames=1`，不再执行额外中心确认。同一停车视野中新出现且与已有目标 bbox
的 IoU 不超过 0.20 的物块会在当前会话首帧冻结为已有目标，不等待 tracker 二次确认。
颜色不足或歧义不影响类别、候选资格和确认进度；模型类别始终是类别权威，明确蓝色危险
仍按危险路径处理。

### 正式流程抓取与恢复

`MatchSequence` 持有跨搜索、接近、稳定规划、解团和再抓取的 `GraspTask`。
解团是当前物理抓取任务的恢复动作，退离后接回该任务；执行次序、规则与时间预算统一见
[正式流程设计](../../../docs/正式流程设计.md#抓取任务与动作顺序)。
`breakup_forward_distance_m` / `breakup_backward_distance_m` 在正式流程中是完整动作距离，
当前配置为闭爪前进 0.5 m、闭爪后退 0.3 m。`grasp_task.marked_targets` 持有团内得分物块
的类别、最后实测 `FieldPoint`、采集时间（ns）与可选主 tracker ID；采集位姿补偿与唯一关联
在解团期间持续更新，退离后以新观测交接同一物块。预览只高亮同帧关联框，日志保留标记及年龄。
CC 继续使用独立固定动作，单绿运输联调与带载补夹不开放解团。

选择器依次把不同物资作为最远 X 锚点，分别生成不同长度的扫掠走廊；橙色目标不参与
多成员组合，只生成单目标计划。每个走廊会把实际进入其中的合法物资闭包纳入。通过
危险、容量、开口和边界硬门禁后，先最大化成员数量，再按配置权重比较净空、行程和对准角。
前进行程一律使用对准后最前成员 K0 的 `x`：
`forward_distance = max(0, max(member.K0.x) - endpoint)`，其中绿/黑组的 `endpoint`
是普通近场的 `target_final_x_mm`，贪心补夹使用 `greedy_target_final_x_mm`，单个橙色目标是
`orange_target_final_x_mm`。绿/黑还逐成员要求
`distance >= member.K0.x + 底面外接半径 - 闭爪末端x`，取所需行程的最大值，
避免只送入中心却把物块前半部留在爪尖外。半径复用配置中的实体尺寸；没有可靠底面 yaw 时
使用外接半径，不用颜色纵向包络。诊断终点输出实际生效值，延长后的完整行程继续参与
危险、容量、场界及最大行程检查。橙色不再使用
上表面颜色投影的最近端加跨度（`x0 + d`）：颜色掩码的纵向拉长不是真实前进深度，
可靠底面中心 K0 才是权威。这允许目标在揽入过程中相互滑动/转动；
`max_forward_distance_mm` 仍限制底盘动作时长。危险检查使用单个矩形走廊，纵向范围为
`[corridor_start_x_mm, max(left_tip.x, right_tip.x, corridor_start_x_mm) + forward_distance]`，
即前端包含按实际开口反解出的夹爪末端前伸距离；横向范围为
`[right_tip_y - corridor_lateral_margin_mm, left_tip_y + corridor_lateral_margin_mm]`。
`grasp_candidate` 诊断的 `corridor_end_x_mm` 与执行走廊共用同一公式。走廊只覆盖夹爪
张开宽度将要扫过的区域，不再覆盖车体或夹臂的历史全扫掠包络。

绿/黑候选不再因蓝色侧邻被直接淘汰：蓝色只按实际扫掠走廊判定——仅检查 K0
是否进入夹爪开口将要扫过的矩形；没有其它安全方案时正式路由进入解团。
`_side_neighbor_metrics()` 的纵向/横向中心差门禁只用于单橙计划的侧邻检查，明显前后
错开的目标不触发。橙色侧邻不影响绿/黑计划；橙色独立性门禁只约束单橙计划。纯绿黑目标
即使分布密集，也不会因为彼此接近被拆散，仍允许最多三个成员的多目标计划。

蓝色危险目标 K0 进入走廊时直接拒绝，不能用评分抵消；不再按实体半径或完整物体轮廓扩大
扫掠阻挡范围。四类目标缺失模型 K0 时统一使用 bbox 底边中点，因此走廊判断
仍有明确地面锚点；橙色只能
作为单目标计划，不能作为额外成员加入绿黑组合。橙色目标地面中心周围
`orange_isolation_radius_mm` 内有当前可定位的其它目标时拒绝；其它
目标未观测或没有当前地面点时不无条件推定其位于禁区。扫掠门禁只判断目标 K0，完整物体
外轮廓不扩大阻挡范围；额外绿黑的 K0 落入走廊时加入组合并
重新检查数量和横向开口；蓝色危险目标只在静止规划阶段影响选组。最终确认通过并提交计划后
不再用运动中的检测重检走廊。规划停车后的近场走廊门禁只使用当前帧 K0 或 bbox 底边
中点地面点；已经
失观的历史轨迹不参与该走廊阻挡，避免侧方静态目标的历史膨胀包络制造幽灵障碍。全局
tracker 仍保留短时历史，明确危险证据也继续在轨迹存续期内保留，但二者不替代近场停车
窗口所需的当前空间证据。

橙色从候选生成阶段起只形成单目标方案；绿/黑只彼此组合。橙色落入绿黑走廊、或绿黑落入
橙色走廊时都只作为阻挡物，不能顺带加入另一类方案。远场 handoff prior 只唯一匹配关联门限内
K0 最近的局部目标；首轮单绿优先合法交接目标，也可选择其它合法单绿。一般阶段在通过
容量、机械开度和 K0 扫掠检查的组合中，依次优先成员数量、黑色数量、规则总分、单橙优先级、
交接匹配及净空/行程/对准代价。绿黑并列可共同收拢时优先多抓，橙色仍严格单独转运。
补夹阶段的 handoff prior 则是锁定入口目标：全场绿/黑候选按机器人地面距离优先，目标确认
后不因新出现的更近目标改选；锁定目标失观或本次动作明确失败后才接管下一个候选，并重新走同一对准、接近
和近场确认链。

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

`NearFieldGraspPolicy.obstacle_extent_required` 默认关闭，由正式 match 首趟单绿启用；
`select()` 与 `recheck()` 共用当前 K0 和配置实体尺寸的净空检查，日志输出同名字段及阻挡 ID/类别。
不新增 YAML 配置；后续转运、补夹及独立 CC/单绿运输入口保持原策略。

`GraspPreparationSession` 串行处理固定场景；首次合法选组自动持有核心并计入当前帧，
消费者无须回传锁组后追要另一帧。`locked_ids` 可约束已经冻结的核心，省略时沿用准备器
持有的核心；更换任务或恢复后的新场景由 worker 的 `begin(session_id)` 隔离旧结果；普通抓取转角后的采集仍由连续静止证据校验。
`GraspPreparation.action` 返回 `grasp`、`motion`、`recovery`、`observe` 或 `exit`。
正式受阻场景可传 `recovery_context=sequence.grasp_recovery_context(snapshot)`，
在同一后台请求中调用 `breakup_planner.py`；独立近场入口不提供恢复能力。

采集时间不改写。硬件循环先用真实编码器/IMU更新 `StationaryMotionEvidence`，
再检查 `near_field_observation_window_open(now_ns)` 和 `grasp_scene_capture_valid(snapshot, now_ns)`。
使用 `latest_snapshot()` 保留迟到结果，不用通用流的采集年龄提前过滤静止计划。
连续静止场景可容纳超过800 ms的处理延迟，但采集年龄仍受提交预算限制、结果发布后的
失联年龄受通用观测上限约束，真实运动、遥测无效或更新的危险/核心缺失仍阻止提交。
`stationary_reason` 区分编码器运动、旋转、无效传感器、重复设备样本和遥测断档。
完整时间与失败语义见[正式流程设计](../../../docs/正式流程设计.md#延迟预算和旁路)。

生产装配见 `gripper_width._run()` 和 `match_runtime._run_hardware()`：从
`configs/runtime.match.yaml` 构造唯一几何对象、
`GraspTargetTracker(config.tracking.build_tracker(), projector, config.near_field_grasp)`、
`NearFieldGraspSelector` 和 `GraspPreparationSession`。后两者不打开硬件。准备器放在
单个后台线程串行调用 `update(snapshot, locked_ids=..., handoff_prior=...,
require_handoff=..., excluded_observation_indices=..., recovery_context=...)`；索引只改变候选资格，不从 `targets` 删除观测，
因此已携带/已交付物资和蓝色危险仍参加扫掠障碍检查。索引在同帧去重和锁定 ID
空间映射后仍绑定对应物理观测。
队列只保留最新一帧。全候选诊断独立线程使用一个最新结果槽，最多每秒计算一次；
退出上下文时唤醒并回收准备/诊断线程。正式流程还会校验准备结果的近场会话 ID，迟到结果不会跨运输轮次
驱动动作。运动层在自己的循环调用
`GripperWidthPickupSequence.step(now_ns, preparation, cumulative_distance_m=..., heading_rad=...)`，将返回的
速度/角度意图交给已装配的控制器。`GripperWidthPickupResult` 返回成员 ID/类别、完成时间、
最终舵机命令及未确认收拢标志；旧单目标 `GripperWidthPickupPlan` 已替换为多目标计划。

开爪提交时冻结唯一确认窗口形成的执行计划，不再把运动中的目标检测送入走廊或成员包络复核，
因此运动模糊、颜色证据不足或目标遮挡不会单独打断已提交动作。编码器仍按动作计划定距，
并保留关键里程计不可用、编码器方向错误和持续无前进进度时的退出保护；明确急停、硬件
健康故障和进程异常仍走软刹车路径。开爪阶段底盘保持静止，张爪计时结束后直接按冻结
计划进入 `forward_encoder_heading_hold`。动作退出后必须用退出之后采集的新证据重新选组，不能直接
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
4 mm 总开口余量、普通/贪心的 `target_final_x_mm` 与 `greedy_target_final_x_mm`、
`corridor_start_x_mm` 和 10 mm 横向走廊余量都是
试验初值。掩码含离地表面、真实夹臂厚度、车头形状、目标滑动和有效前进行程尚未实测
验证，不能用这些近似几何证明真实抓取安全。实物收拢效果、真机控制响应时间、夹持成功率、
解团效果、危险类表现和目标设备延迟均为**未验证**；验收步骤见 `manual_tests/README.md`
的近场收拢章节。

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

流程启动后先完成配置装配、UART 零速同步、相机/Hailo 首帧、遥测和预检，并保持
零速心跳等待。终端显示 `start_gate=ready` 后，操作者直接按 Enter 才启动比赛计时和
状态机；等待期间仍持续检查 UART、急停、相机、预览和观察发布，失效时拒绝发车。
Enter 放行后立即执行启动转向/直行和目标搜索，输入其它文字不会误触发。
搜索态的机会检查使用固定
`TRANSPORT` 夹爪包络的前向走廊：首轮按距离尝试单个绿色 K0；若候选走廊被蓝/橙/黑目标阻挡，先尝试其它安全绿色，只有没有合法抓取方案才进入局部解团；一般阶段支持绿/黑物资组或单个橙色伤员快速入口。固定走廊外的合法目标先生成
近场直接可达性预览；当前方向没有合法方案时才继续目标团解团，不再增加一次近场旋转。远场和近场等待阶段保持 `CLOSED`，不会先发固定开口；近场
计划 ready 后才发送动态开度。规划期间的同帧画面叠加半透明走廊：黄色为对准预览、
青色为复核、绿色为 ready、红色为阻挡；match 中任何已从多个候选中选出的远场目标、解团目标团或近场成员框，
均以橙色粗线和 `SEL#track_id` 标记，
组中心显示十字并沿对准方向画箭头。本地预览和观察图传使用同一叠加帧。终端状态行包含当前状态、动作原因、
编码器距离、航向、目标轮速、绿色走廊诊断、安全区阶段和 D2 遥测丢弃计数。观察图传、
地图状态和本地预览只读取最新数据，不参与运动决策。

安全区纠偏按 bbox 图像位置选对侧两点（偏左 K0/K2、偏右 K0/K1）；完整 bbox 四边保留余量且两点可靠可见才停稳取两帧，
不再强制居中。单侧裁剪以固定角速度持续扫描，完整 bbox 入镜后立即制动；两侧/上下裁剪才一次有界倒车，调整完成后用编码器/IMU
确认停稳再取样。首个有效样本启动一次独立确认窗口，截止当轮先消费有效帧；短暂漏点保留仍属连续静止区间的
有效样本。观察/确认预算耗尽仍显式退出视觉纠偏并以现有航位继续规划 D2。
新调整参数、删除的追框字段、采集/发布时间和降级限制统一见[正式流程设计](../../../docs/正式流程设计.md)。

安全区末段的绿/黑物资使用 `match.safe_zone_d2_to_final_speed_m_s` 和
`match.safe_zone_d2_to_final_braking_overrun_mm`；单个橙色伤员使用独立的
`match.safe_zone_orange_d2_to_final_speed_m_s` 和
`match.safe_zone_orange_d2_to_final_braking_overrun_mm`。解团与 D2→末段分别通过
`match.breakup_max_{linear,angular}_{acceleration,deceleration}_*` 和
`match.safe_zone_d2_to_final_max_{linear,angular}_{acceleration,deceleration}_*`
逐项覆盖四项车体限制，`null` 表示继承 `motion` 对应全局值。

正常结束、`Ctrl+C`、`SIGTERM`、相机/Hailo/UART/网络异常或旁路线程失败都会进入统一
软刹车清理路径。重新运行前应确认车辆已停稳、急停状态已复位，并重新提供显式监督确认。

启用 `--log-dir` 时，启动/运行失败的完整异常链会在日志关闭前写入当前
`match*.log`。`runtime_phase` 标明打开 UART、序号同步、等待相机、首帧、预检或
控制阶段；`uart_sync_attempt` 记录同步失败原因、次数和单调时间截止点（ns）。
制动或关闭 UART 再次失败时作为原异常的附注保留，仍尝试关闭通道。
排查偶发启动失败应查看该次日志末尾的底层异常，而不是仅凭 `UART reader failed`
推断原因。打开失败、设备断开和队列溢出不做无条件重试；本次修复不代表真机启动
成功率已验证。

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

搜索、接近、静止规划、抓取、补夹和解团恢复的权威说明见
[正式流程设计](../../../docs/正式流程设计.md#抓取任务与动作顺序)。
`MatchSequence.grasp_task` 提供当前物理任务上下文，`grasp_task_diagnostic(now_ns)`
输出身份、入口、核心、恢复进度与总截止时间；不要把 `near_field_session_id` 当作新物理目标。
`approach_seed_diagnostic(now_ns)` 逐个列出新鲜目标被远场入口淘汰的第一个门禁，与
`_find_approach_seed` 共用同一判据，用于区分「没看见」和「看见了但被否决」；它随每次
状态决策写入日志的 `approach_seed=` 字段。
NB 仅替换开场；策略变体的正式阶段显式委托同一基类，删除了与基类相同的方法副本。
CC 与单绿运输联调保持原有动作能力限制。

## 纯逻辑装配

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.match.yaml")
flow = config.build_match_sequence()
```

联调入口的类型和装配方法为 `GrabTransportSequence` 与
`config.build_grab_transport_sequence()`。它覆盖初始场地位置和航向，仍复用正式流程
配置、跟踪器、夹爪标定和安全区视觉纠偏。

`MatchNBSequence` 与 `config.build_match_nb_sequence()` 对应
`rescue-vision-match-nb`。它直接继承当前 `MatchSequence`，只替换区域 2 开场：从当前起点
按配置的 `turn` / `straight` 动作顺序执行，完成指定动作后张爪，随后进入正式目标团搜索。
该入口不提供区域 3 中心对称。

转向使用 IMU 的相对航向进度和角速度，直行使用编码器累计路程并以动作开始时冻结的路线
航向做保持。两类动作都由 `motion.relative_action.RelativeActionController` 按
“高速运行 → 提前减速 → 低速闭环收尾”输出目标；制动区包含实际遥测年龄、底层 40 ms
轮速命令刷新周期和可标定执行响应。几何误差达标后仍需实测轮速/角速度足够小，以及
`StationaryMotionEvidence` 观察到连续的新遥测停稳，才会切换或张爪。超调修正有幅度和
时间上限，超时进入停车，不把失败算作完成。动作序列放在
`configs/runtime.match_nb.yaml` 的 `match` 节中：

```yaml
nb_opening_actions:
  - type: turn
    angle_rad: -1.0
    angular_velocity_rad_s: 0.5
  - type: straight
    distance_m: 1.2
    speed_m_s: 0.4
  - type: turn
    angle_rad: 1.0
    angular_velocity_rad_s: 0.4
  - type: straight
    distance_m: 1.0
    speed_m_s: 0.3
nb_opening_gripper_after_action: 2
```

配置字段（`match` 节，动作列表中每项均可独立配置）：

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| `nb_opening_actions[].type: turn` | — | 原地转向；`angle_rad` 左正右负，`angular_velocity_rad_s` 为正速度幅值 |
| `nb_opening_actions[].type: straight` | — | 直行；`distance_m` 前进正倒车负，`speed_m_s` 为正速度幅值 |
| `nb_opening_gripper_after_action` | 1-based 序号 | 完成该动作并停稳后张爪；`null` 表示不自动张爪 |
| `nb_opening_gripper_left_deg` / `nb_opening_gripper_right_deg` | deg | 张爪左右角度 |
| `nb_opening_turn_tolerance_rad` | rad | 转向完成误差 |
| `nb_opening_distance_tolerance_m` | m | 直行完成误差 |
| `nb_opening_settle_time_s` | s | 连续新遥测的停稳确认时长，不是固定 sleep |
| `nb_opening_turn_timeout_s` | s | 单次 NB 动作截止时间，超时后停车 |
| `nb_opening_execution_response_s` | s | 下发到车体响应的待标定延迟 |
| `nb_opening_effective_*_deceleration_m_s2` | m/s²、rad/s² | 真车测得的有效减速度；`null` 使用 NB 控制器保守初值，填值会被 motion 软件限幅截断；当前 match_nb 直线初值为 2.0 m/s²，仍需真车标定 |
| `nb_opening_max_telemetry_age_ms` | ms | 反馈可用于停稳确认的最大年龄 |
| `nb_opening_fine_*` / `nb_opening_stop_*` | m/s、rad/s | 低速收尾和实测停稳门限 |
| `nb_opening_heading_*` | rad、rad/s | 直线航向保持比例项和限幅 |
| `nb_opening_correction_*` | m、rad、s | 小幅超调修正的幅度/时间上限 |

车端诊断：`MatchNBSequence.nb_opening_route_phase` 返回当前动作，`nb_opening_diagnostic`
返回一行「动作 / 配置量 / 实际进度 / 航向 / 制动距离 / 延迟 / 停稳证据」。`match_runtime`
在动作变化时打印一次，例如：

```
nb_opening=action=1/5 angle=-1.156rad progress=0.000/1.156rad \
  speed=0.500rad/s heading=-90.0deg cumulative_distance_m=0.0
```

开场结束后两个属性都返回 `None`，不会在解团/运输阶段产生误导读数。

## 蓝色优先策略变体

`MatchStrategySequence`（`app/match_strategy.py`）是 `MatchSequence` 的**子类**，
入口为 `python -m rescue_vision.app.match_strategy --config configs/runtime.strategy.yaml`
（当前没有 console script）。它分两个阶段：

- **策略阶段**（`_strategy_formal_phase is False`）：开场两段反向冲刺（右转短冲 →
  左转长冲，转角与速度来自 `startup_turn_*` / `cluster_relocate_*`），随后**只搜蓝色
  危险物块**；能直接夹取则单块转运，否则对全蓝目标团执行动态解团。两趟分别放到对面
  安全区左右 D2 点（`safe_zone_fallback_target_field` / `safe_zone_injured_target_field`），
  在 D2 释放而不推进到安全区深处。
- **正式阶段**：第二趟返回结束（共享实现给出 `FINISH_STOP`）时由 `_step_return_backup`
  拦截并调用 `_begin_formal_match_after_strategy`，恢复己方 y 方向的 D2 端点、重新打开
  首轮单绿机会、复位旋转预算与解团失败记忆，然后进入正式绿块搜索。

继承与委托的边界：正式阶段通过 `_SharedMatchSequence.<method>(self, ...)` 显式委托回基类，
策略阶段使用本地实现；`__init__` / `start` 只 `super()` 委托后追加 `_strategy_*` 字段，
不再复制基类字段清单。策略阶段 `near_field_enabled` 返回 `False` 并临时把
`_near_field_pickup` 置空（真身存在 `_strategy_saved_near_field_pickup`），蓝色近场计划
由 `_strategy_blue_grasp_preparation` 在控制线程内直接生成；翻转时恢复。

`_step_align_green` 是唯一按相位显式分派的对准入口：共享实现在 `_near_field_pickup`
不为 `None` 时会改走 `_step_formal_green_align`，而蓝色运输恰好会重新暴露该序列，
因此变体在 `_strategy_blue_transport` 期间固定走 `_align_green_legacy`（真机验证过的
历史对准路径），其余阶段才走共享实现。

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

### 2315 现场问题的软件回归

`tests/test_match2315_grasp_task.py` 使用实际准备器覆盖单稳定帧、300–1500 ms准备延迟、
5/10 ms轮询、250/400 ms帧间隔、同任务解团后再抓取并交接运输、危险侵入与入口失败归属。
这些合成回归不能证明真实推移效果、危险类指标或 Pi/Hailo 控制周期；真机验收见
[manual_tests](../../../manual_tests/README.md#2315-抓取任务整链验收未验证)。

1818 日志修复：正式入口在缺准备结果时继续传入 IMU，转向完成时间只记录一次；抓取开爪时冻结航向并用于前进纠偏。远场交接继续已检查的接近，局部退出优先接管可见合法替代目标。证据与迁移见 [1818 诊断](../../../docs/match1818诊断与修复.md)。

正式流程按完成的夹取会话累计 `MatchSequence.carried_target_count`，本趟2个最多补1个、3个直接运输。
夹爪颜色门禁只消费同帧独立ROI证据，冲突由 `MISGRASP_OPEN → MISGRASP_BACKUP → MISGRASP_SETTLE`
执行开爪、编码器后退250 mm并重选；进入 d2→安全区末段推进、末端释放和退出安全区后不再检测误夹，
由安全区路线状态机接管；不在感知线程下发动作。颜色/线宽参数见
[config README](../config/README.md#夹爪内颜色门禁配置)。合爪后的同一新鲜帧若有至少两个模型橙色目标
bbox 与夹爪内侧多边形 ROI 存在正面积重叠、每个框内的橙色 HSV 掩码占比均达到配置门限，
且 bbox IoU 和 K0 像素距离均达到配置的明显分离门限，也复用该误夹退出链；单个合格框、
橙色占比不足、疑似重复框、缺 K0、ROI 外框和仅边界接触不触发。
蓝色模型检测框与夹爪内侧识别区多边形相交或仅边/角接触，即判为误夹并复用开爪恢复；
不要求蓝色 HSV 面积占比或 K0，沿用上述采集时间、结果时效和阶段门禁。
误夹颜色证据只用于任务层恢复和日志诊断，不进入本地或远程预览渲染；预览始终继续消费
普通最新感知帧。
启动仍使用 `configs/runtime.match.yaml` 和本页命令，
资源由 `match_runtime` 的现有上下文与 `finally` 管理。退区停稳后立即转向搜索，取消退区安全区校准；
两点校准、累计数量、时间语义与现场限制统一见[正式流程设计](../../../docs/正式流程设计.md)。
