# `world`：最小世界模型

本包把 `TrackedTarget`、静态场地区域、可选场地坐标和对手占据整理成不可变 `WorldSnapshot`。它是跟踪与规则状态机之间的唯一世界语义层，不运行检测、不维护比赛得分。

## 公共入口

| 入口 | 作用 |
| --- | --- |
| `WorldRuntimeConfig.build_model()` | 从 `runtime.yaml` 的世界参数和静态区域创建 `WorldModel` |
| `WorldModel.update()` | 在单调时间轴上产生一个完整快照 |
| `WorldSnapshot` | 状态机消费的动态目标、区域、对手和不确定性 |
| `WorldSnapshot.target_region_kinds()` | 查询目标所在静态区域；缺少场地坐标时返回 `None` |
| `WorldSnapshot.target_in_opponent_occupancy()` | 查询目标是否落入对手占据；缺少场地坐标时返回 `None` |
| `StaticRegion` / `RegionKind` | `FieldPoint` 多边形及场地语义 |
| `OpponentOccupancy` | 带置信度和时间戳的对手场地占据多边形 |
| `WorldTarget` | 跟踪目标的类别概率、局部/全局位置和危险状态 |
| `HazardState` | `CLEAR`、`SUSPECTED`、`CONFIRMED` |
| `WorldUncertainty` | 视觉过期、缺坐标、未确认或对手信息过期等原因 |

## 实时对接

```python
from time import monotonic_ns

from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
world_model = config.world.build_model()

# tracks 来自同一帧 tracker.update()。
# 当前没有定位时，robot_field_point 和 target_field_points 保持缺省；
# 世界模型会明确标记缺失，而不是伪造 FieldPoint。
snapshot = world_model.update(
    timestamp_ns=monotonic_ns(),
    visual_timestamp_ns=raw_frame.timestamp_ns,
    tracks=tracks,
    opponent_occupancies=latest_opponent_occupancies,
)

for target in snapshot.targets:
    print(
        target.track_id,
        target.hazard_state,
        target.ground_point,  # 当前机器人地面系，mm
        target.field_point,   # 定位接入前通常为 None
    )
```

`timestamp_ns` 是本次世界更新时刻，`visual_timestamp_ns` 是最近有效视觉帧时刻。两者必须来自同一单调时钟。没有检测目标的有效新帧仍应更新 `visual_timestamp_ns`，否则会被误判为视觉中断。

## 静态区域

区域从 `runtime.yaml` 加载，顶点使用 `FieldPoint` 的 `[x_mm, y_mm]`：

```yaml
world:
  # 其余阈值省略
  regions:
    - region_id: own-material
      kind: own_material
      polygon_field_mm:
        - [-1400.0, -1400.0]
        - [-900.0, -1400.0]
        - [-900.0, -1000.0]
        - [-1400.0, -1000.0]
```

允许的 `kind` 为 `field`、`own_material`、`own_injured` 和
`opponent_safe`。多边形至少三个点、面积非零，边界点视为进入区域。示例坐标仅说明格式，必须在 P3 根据现场红蓝方变换和实测场地替换，不能直接作为比赛地图。

## 危险与不确定性

世界模型集中融合跟踪状态和完整类别概率：

- 曾完成跨帧确认且危险概率达到 `danger_confirm_threshold` 时是
  `CONFIRMED`；短时 `COASTING` 不会抹掉危险历史；
- 危险概率达到较低疑似阈值、未知概率过高、轨迹未确认或处于
  `COASTING`、类别为 `UNKNOWN` 时是 `SUSPECTED`；
- 其余才是 `CLEAR`。

状态机只允许选择 `CLEAR` 目标。`SUSPECTED` 不会被强行改成普通物资。

`WorldSnapshot.uncertainties` 是快照级诊断集合。缺少机器人
`FieldPoint` 不妨碍局部目标跟踪，但不能据此断言机器人“不在对方安全区”；需要区域判断的调用方必须等待定位或其它明确证据。

目标区域查询同样保留三态语义：`True`/非空集合表示已有明确场地证据，
`False`/空集合表示明确不在其中，`None` 表示目标没有 `FieldPoint`、无法判断。
未知 `track_id` 会抛出 `KeyError`。状态机只排除已经明确位于安全区或对手占据
内的候选，不把缺失定位伪装成区域结论。

## 对手占据生命周期

同一 `opponent_id` 的新占据会替换旧值；没有新观测时保留到
`opponent_max_age_ms`，随后删除并标记 `STALE_OPPONENT`。占据多边形属于场地坐标，不能把检测框或 `GroundPoint` 直接传入。

## 错误与降级

- 世界时间倒退、未来视觉/对手时间戳、重复 track ID、未知场地映射 ID 会报错；
- 视觉年龄超过 `max_visual_age_ms` 时保留快照但增加 `STALE_VISION`，由状态机输出保守停止；
- 缺 K0 的目标仍保留图像与类别历史，同时标记
  `TARGET_WITHOUT_GROUND_POINT`，不能交给地面规划；
- 当前没有区域感知、定位或对手检测实现；这些输入可用合成事件开发，但不能描述成已完成现场能力。
