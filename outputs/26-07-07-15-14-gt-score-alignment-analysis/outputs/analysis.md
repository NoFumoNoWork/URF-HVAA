可以看懂，而且这些图其实已经把结论推得很清楚了：**现在主要瓶颈不再是“怎么把 interval 合并/切分得更漂亮”，而是 GT 标注和 VLM/score 证据的时间粒度不一致**。报告里的总量是：1394 个 GT interval、640 个视频、640 个 score JSON，没有跳过文件；其中 strong support 809 个，占 58.03%，weak support 145 个，占 10.40%，score unsupported 只有 55 个，占 3.95%，但 sparsely sampled 有 326 个，占 23.39%。

下面我按图解释。

------

## 1. GT duration vs max_score

这张图看的是：

> 每个人工异常区间里，机器/VLM 给出的最高异常分是多少。

横轴是 GT 持续帧数，log scale；纵轴是这个 GT 内部的 `max_score`。

你能看到大量点卡在：

```text
max_score = 0.8, 0.9, 1.0
```

这说明很多人工异常段里，**至少有某个时刻机器也认为很异常**。

但是它不代表整个 GT 都被机器认为异常。因为 `max_score` 只要有一个点高就会高。所以这张图的含义是：

**多数 GT 至少存在局部 score 支持，但这不能说明机器覆盖了完整人工标注区间。**

这对 peak-aware 是有利的：如果 GT 内有高峰，那么后处理理论上还有救。

------

## 2. GT duration vs mean_score

这张比上一张更重要。它看的是：

> 每个人工异常区间里，机器/VLM 平均认为有多异常。

你会发现 `mean_score` 分布很散，尤其很多 GT 的 max_score 很高，但 mean_score 并不高。这说明一个现象：

**人工标注常常是事件级大区间，而 score 响应更像局部关键瞬间。**

例如车祸、爆炸、枪击这类事件，人工可能标一整段过程，但 VLM 只在撞击、爆炸火光、枪击明显帧附近给高分；事件前后、停顿、烟雾之后、人物逃跑等片段分数可能下降。

所以这张图支持的结论是：

> VLM score 更像局部视觉证据，而不是完整事件边界标注。

------

## 3. GT support classification by dataset

这张图把每个 GT interval 分成几类：

| 类别                        | 意思                         |
| --------------------------- | ---------------------------- |
| strongly_score_supported    | GT 内有强 score 证据         |
| weakly_score_supported      | 有一定 score 证据，但不强    |
| ambiguous_mid_score         | 中间状态                     |
| score_unsupported           | GT 内 score 很低             |
| sparsely_sampled            | GT 内 score 点太少，判断不稳 |
| barely_sampled              | 几乎没采到点                 |
| unobserved_or_missing_score | 没有 score                   |

最关键的是：**strongly_score_supported 是最大类**。报告里 strong 是 809/1394，也就是 58.03%；weak 是 145/1394，也就是 10.40%。合起来大约 68.43% 的 GT 是有较明确 score 支持的。

但是还有一个很大的类：**sparsely_sampled = 326，占 23.39%**。这说明很多 GT 不是 score 低，而是 score 点太少，尤其短 GT 会被 16F 或类似 stride 的采样稀释。

所以这张图的结论是：

**大部分 GT 不是完全 unsupported；真正 score_unsupported 只有 3.95%。但采样稀疏是一个很大的不确定来源。**

------

## 4. Top labels: support ratios

这张图目前**不太好读**，因为 y 轴叫 `ratio sum across datasets`，导致有些 label 堆叠高度接近 2。这不是“比例超过 100%”，而是 UCF-Crime 和 XD-Violence 两个 dataset 的比例被加在一起了。

所以这张图不能按“每个 label 内 strong/weak/unsupported 加起来等于 1”来读。

但它仍然能看出几个趋势：

- `Fighting`、`Shooting`、`Explosion` 的 strong 支持比较高；
- `Shoplifting`、`RoadAccidents`、`Abuse` 的 unsupported 或 sparse 比例更明显；
- `Car_accident` 里面 sparse 比较重，说明可能存在标注段短、score 采样不足、或事故过程与关键视觉异常不完全同步的问题。

报告里也列了 unsupported ratio 较高的 label：UCF-Crime / Abuse 为 50%，Shoplifting 为 48%，Robbery 为 20%，Vandalism 为 12.5%，Explosion 为 9.09%。不过 Abuse n=2，样本太少，不能过度解释。

这张图建议后面重画成：**每个 label 单独归一化到 1.0，不要跨 dataset 直接相加**。

------

## 5. Outside-GT high-score evidence

这张图看的是：

> GT 外面有多少高分证据。

它很重要，因为它对应你之前说的：

**机器认为异常，但人工没有标异常。**

图里 XD-Violence 的 outside-GT high-score ratio 明显高于 UCF-Crime，大概接近 0.37；UCF-Crime 大概 0.12。报告里说 outside-GT high-score intervals 一共有 5578 个。

这说明 XD-Violence 中大量高分片段不在人工 GT 内。可能原因有三种：

1. VLM/score 误报；
2. 人工标注边界较窄或漏标；
3. VLM 把上下文危险、混乱、冲突前后过程也视为异常，但 GT 只标核心事件。

所以这张图不能直接说“模型错很多”，更稳的说法是：

> XD-Violence shows stronger score-positive / GT-negative disagreement, indicating larger mismatch between score-level anomaly evidence and event-level annotation boundaries.

------

## 6. Post-processing recoverable upper bound

这是最重要的图之一。它回答：

> 如果只改后处理，理论上最多能救回多少 GT？

结果是：

- UCF-Crime：recoverable 73.72%，uncertain 12.82%，unrecoverable 13.46%
- XD-Violence：recoverable 80.45%，uncertain 14.14%，unrecoverable 5.41%

这说明：**后处理理论上还有空间，但不可能解决全部问题。**

尤其是 UCF-Crime 的 unrecoverable 更高，说明有一部分 GT 内 score 本身没响应。对于这些区间，peak-aware、merge、split 都没法凭空恢复。

这张图可以作为你报告里的核心结论图。

------

## 7. Threshold sensitivity

这张图看不同 score 阈值下，几个比例怎么变。

你可以这样读：

- `recoverable_ratio` 基本不变，说明 recoverable 的定义主要来自 GT interval support 分类，不太受 score_positive_threshold 影响；
- `gt_pos_score_pos_ratio` 随 threshold 从 0.4 到 0.8 下降，说明阈值越高，GT 内被判为 score-positive 的窗口越少；
- `gt_neg_score_pos_ratio` 也下降，说明阈值越高，GT 外高分误报/分歧减少；
- `outside_gt_high_score_ratio` 也下降。

这个图说明存在一个 trade-off：

```text
降低阈值：GT 覆盖更好，但 GT 外高分更多
提高阈值：误报/分歧减少，但 GT 内覆盖下降
```

这正好回应你之前担心的“面向结果调参”：现在可以说不是只调一个阈值追结果，而是在分析阈值下的 coverage–false-positive trade-off。

------

## 8. Window-level GT/score consistency

这张图有点难读，因为每个柱子堆起来到 2，而不是 1。原因应该是正类和负类比例分别归一化后又堆在一起了。所以它不是常规意义的 stacked percentage bar。

但趋势是清楚的：

- 窗口从 30F 到 300F，`GT+ Score+` 比例上升；
- 同时 `GT+ Score-` 比例下降；
- XD-Violence 在 300F 下 GT+Score+ 最高，报告里写最高是 XD-Violence window=300，达到 92.82%。

这说明：

**窗口越大，GT 和 score 越一致。**

这非常重要。它说明 VLM score 可能不适合非常精确的帧级边界，但在较粗的事件窗口下，它和人工 GT 的一致性明显更好。

所以你可以把它解释为：

> VLM score is more reliable as coarse event-level evidence than as precise temporal boundary annotation.

------

# 总体结论

这些图合起来说明四件事：

第一，**大多数人工 GT 有 score 支持**。strong + weak 大约 68.43%，再加上 uncertain/sparse，说明不是 VLM 完全不认 GT。

第二，**大量 GT 的 score 支持是局部的，而不是全段持续的**。max_score 高、mean_score 分散，这说明 VLM 对关键瞬间敏感，但不一定覆盖完整人工事件区间。

第三，**XD-Violence 的 GT 外高分更多**。这说明它的 score-positive / GT-negative 分歧比 UCF-Crime 更严重，可能来自标注边界、事件定义或 VLM 对上下文异常的敏感性。

第四，**后处理有上限**。UCF-Crime 约 73.72%、XD-Violence 约 80.45% 是 recoverable；剩下 unrecoverable 和 uncertain 不能靠 peak/merge/split 全部解决。

------

## 我建议你后面改两张图

最需要改的是：

1. **Top labels support ratios**
   现在跨 dataset ratio 相加，导致 y 轴到 2，不直观。建议改成每个 label 内归一化到 100%。
2. **Window-level GT/score consistency**
   现在四类比例堆到 2，不直观。建议拆成两个图：
   - GT-positive windows 内：Score+ vs Score−
   - GT-negative windows 内：Score+ vs Score−

这样读者一眼能看懂。

一句话概括：**这些统计支持你的判断：当前主要矛盾不是 interval 后处理单独导致的，而是 score-level anomaly evidence 与 human event-level GT 在时间粒度和异常定义上不完全一致。**