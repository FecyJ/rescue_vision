# `tracking`：任务目标多目标跟踪

本包把同一单调时间轴上的 `TargetObservation` 关联为带稳定 `track_id` 的 `TrackedTarget`。它负责短时遮挡、置信衰减和过期删除，不创建相机、不运行模型，也不解释交付规则。

## 公共入口

| 入口 | 输入 | 输出与生命周期 |
| --- | --- | --- |
| `TrackingConfig` | 关联门限、确认次数、滑行时间和衰减参数 | 从 `runtime.yaml` 取得；`build_tracker()` 创建一轮使用的跟踪器 |
| `MultiTargetTracker.update()` | 一个采集时间戳和该帧全部观测 | 按 `track_id` 排序的 `tuple[TrackedTarget, ...]` |
| `MultiTargetTracker.reset()` | 无 | 新一轮比赛前清空 ID、历史和时间轴 |
| `TrackedTarget` | 只读跟踪状态 | 供世界模型消费，不应由下游修改 |
| `TrackStatus` | `TENTATIVE`、`CONFIRMED`、`COASTING` | 区分未确认、稳定和短时遮挡 |

## 实时对接

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
tracker = config.tracking.build_tracker()

# detection_result 来自 TargetPoseDetector.detect_realtime()。
# 即使本帧没有目标，也必须用本帧 capture timestamp 更新一次，
# 这样遮挡计时和置信衰减才沿真实采集时间推进。
tracks = tracker.update(
    raw_frame.timestamp_ns,
    detection_result.observations,
)

for track in tracks:
    if track.status.value != "confirmed":
        # tentative/coasting 仍进入世界模型保留不确定性，
        # 但不能直接成为高收益动作目标。
        continue
    print(track.track_id, track.target_class, track.ground_point)
```

一批非空观测必须来自同一帧，且每个
`observation.capture_timestamp_ns == update(timestamp_ns)`；混帧或时间倒退会抛出 `ValueError`。空批次仍是一次有效视觉更新。

## 关联与坐标语义

- 两侧都有 `GroundPoint` 时，优先按机器人地面系距离关联，单位 mm；
- 缺少地面映射时退化为全尺寸去畸变图上的框 IoU；
- 两个明确且不同的规则类别不会关联；`UNKNOWN` 可以与已有类别关联，以便保留历史证据；
- 当前实现针对相邻最新帧的短时关联。机器人快速自运动时，局部地面坐标会变化；P4 定位接入前不能把长期局部轨迹当作场地静态真值。

`ground_point` 属于该目标最近一次实际观测；若当前匹配观测缺少 K0，它会变为 `None`，不会静默沿用旧地面点。

## 遮挡、衰减和删除

未匹配目标转为 `COASTING`，置信度按
`exp(-confidence_decay_per_second * elapsed_s)` 衰减。满足任一条件就删除：

- 距离最后一次实际观测超过 `max_coast_ms`；
- 置信度低于 `min_confidence`。

重新匹配会保持原 `track_id`、增加 `hit_count` 并清零 `missed_count`。达到
`confirmation_hits` 才进入 `CONFIRMED`。这些参数来自 `runtime.yaml`，现场调参不修改算法代码。

`ever_confirmed` 与当前 `status` 分开保存：已确认轨迹在遮挡时会变为
`COASTING`，但其危险历史不会因此被世界模型降级。

## 错误与限制

- 时间倒退、混合帧时间戳、非法配置或非 `TargetObservation` 输入会立即报错；
- 当前使用确定性候选排序和一对一贪心匹配，不声称达到正式 MOT 指标；
- 没有实拍标注轨迹前，只能证明逻辑、遮挡和时间行为，不能报告 ID 切换率；
- 危险判定不在本包完成，完整概率历史交给 `world` 统一做保守解释。
