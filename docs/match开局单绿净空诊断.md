# match 开局单绿净空诊断（2026-09-16）

## 日志证据

用户截图显示 `process_t=35572.1ms`、`transport_near_field_grasp`、
`alignment_recent_checked_plan ids=(6,)`。最吻合的是
`logs/match_20260916_2159.log` 的首趟任务 6；日志没有截图瞬间的逐帧完整几何，
因此以下定位到同一时间段和锁定目标，不能声称已精确重放该图片。

| 日志行 | 进程时间 | 观测 |
| --- | --- | --- |
| 180 | 候选帧 648 | 单绿 6，K0 x=108.06 mm，颜色包络 x=220.67～309.25 mm，对准角 0.89 rad；顶部包络不能当成底面中心 |
| 186 | 34376.4 ms | `alignment_recent_checked_plan`，锁定 `(6,)`，`rejections=()`，观测年龄 653.2 ms，准备年龄 1609.8 ms |
| 189 | 35387.4 ms | `waiting_locked_target_observation`，仍锁定 `(6,)`，观测年龄 774.0 ms，准备年龄 2620.8 ms |
| 204 | 36420.8 ms | 再次 `alignment_recent_checked_plan`，拒绝为空，准备年龄 299.9 ms，任务恢复次数仍为 0 |
| 208～210 | 随后的退出 | `confirmation_timeout`，退出重选，没有在该次任务进入解团 |
| 362～363 | 70595.8 ms 附近 | 后来才进入 `breakup_forward_dynamic` |

这些记录证明密集团里的单绿进入了直接抓取对准，不能证明它最终成功合爪或发生碰撞。
原始日志保留本地，不复制批量日志或图片进仓库。

## 根因和修复

`NearFieldGraspSelector._obstacle_distance()` 原先只计算其他目标 K0 到扫掠区域的距离。
物块实体伸进路径但中心在外时，选组仍可通过；恢复准备器只在明确阻挡时启动解团，
因此它收到合法单绿计划后不会主动解团。单独放宽解团阈值无法解决这一入口问题。

正式 match 在首次交付前启用 `NearFieldGraspPolicy.obstacle_extent_required`。
选组（包括对准后的预测路径）和锁定动作剩余路径复核，使用当前 K0 与现有配置物理尺寸的
外接半径检查其他物块；不使用上表面颜色投影作为底面实体位置，不另加经验距离配置。
绿块邻居同样参与，避免第一趟夹带第二块。

阻挡继续沿用 `blocked_target:<id>:<class>`，交给原准备器生成和验证解团方案；
安全孤立单绿仍直接执行。锁定后新侵入记录 `new_sweep_obstacle`。
运行日志增加 `obstacle_extent_required`，配合现有成员、拒绝原因、帧号和年龄定位策略。
完整运行语义以[正式流程设计](正式流程设计.md)为准。

首次交付后策略关闭；没有修改解团规划器、推退距离、确认帧数、停稳及超时预算。
CC 和独立单绿运输的自有 policy 不启用此项；NB 及策略变体回到正式首趟阶段时复用该保护。
无需 YAML 或 CLI 迁移。新增 policy 字段默认 false，其他调用方保留原行为。

## 验证和边界

新增 `tests/test_match_opening_clearance.py`：7 项通过。
覆盖四类邻块中心在外、实体侵入的拒绝，锁定后危险侵入复核，孤立单绿放行，后续和补夹关闭策略；
5/10 ms 控制轮询、250/400 ms 帧间隔、300/600 ms 延迟下，真实准备器与任务状态机完成
解团前推、退离、再抓取和运输交接，任务身份与截止时间不重启。

两组扩展验证合计 395 项通过、9 项失败（共 404 项，无重复统计）：

- 317 项：开局净空、near_field_grasp、match_near_field、gripper_width_sequence、match_breakup、
  match_misgrasp_breakup、match_nb、match_strategy、grab_transport；315 通过。
- 87 项：match2315_grasp_task、match_breakup_latency、match_marked_breakup、match_far_breakup；80 通过。

9 项失败在 `/tmp` 源码副本关闭本次新增策略后同样复现：
邻组失败归属 2 项、远场延迟旋转 2 项、单块扩组/橙色诊断/场界固定行程各 1 项，
以及 NB 开场配置与测试期望不一致 2 项。未改变这些已有行为或降低断言；本次未提交 Git。
`git diff --check` 通过。系统 Python 缺少 pytest，实际测试使用 `.venv/bin/python`。

目标硬件控制周期、端到端延迟、截图场景真实解团成功率与危险类碰撞指标：**未验证**。
外接半径不依赖物块朝向，但窄缝中可能比真实定向轮廓更保守；仅限制首趟，现场应复测
“蓝块贴近单绿”和“确实孤立单绿”两类场景，确认不会把可直接夹取的孤立物资变成无效解团。
