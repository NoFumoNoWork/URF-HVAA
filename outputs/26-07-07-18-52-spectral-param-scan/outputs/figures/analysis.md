这些图确实不太直观，因为它把 **5 个指标同时画在一起**，而且有些指标是“越高越好”，有些指标是“越低越好”。先抓住这个就行：

| 指标                        | 含义                                    | 越高/越低                  |
| --------------------------- | --------------------------------------- | -------------------------- |
| `GT_coverage`               | 人工标注 GT 有多少被预测区间覆盖到      | 越高越好                   |
| `predicted_GT_fraction`     | 预测出来的区间里，有多少真的落在 GT 里  | 越高越好，类似 purity      |
| `supportable_gt_coverage`   | 有 score 证据支持的 GT 被覆盖多少       | 越高越好                   |
| `unsupportable_gt_coverage` | 没有 score 证据支持的 GT 也被覆盖了多少 | 通常越低越好               |
| `predicted_duration_ratio`  | 预测区间占视频总时长多少                | 不能太高，太高说明区间太宽 |

所以读图时不要找“所有柱子都高”的方法。真正好的方法应该是：

**蓝色高、绿色高、橙色高；红色低；灰色不要太高。**

------

## 1. 第一张：Ablation metrics 看什么？

这张图是在问：

**SG、airPLS、trend 这些模块分别有没有用？**

最重要的结论是：

### Full Spectral-Fusion-Refined 确实比 Peak-Aware-Refined 覆盖更强

右边的 `Full Spectral-Fusion-Refined` 蓝色和橙色都比最左边 `Peak-Aware-Refined` 高：

- GT coverage 更高；
- supportable GT coverage 更高；
- unsupportable coverage 明显比 Peak-Aware 低一些；
- 但灰色 duration ratio 也略高。

也就是说：

**完整 spectral fusion 确实多找回了一些 GT，尤其是有 score 支持的 GT；代价是预测区间稍微更宽。**

### SG only / trend only 不适合作为主方法

`Spectral-Fusion SG only` 和 `trend only` 的蓝色柱很低，说明它们单独用时覆盖不到多少 GT。

这说明：

**SG 或 trend 单独不是 detector，它们更像辅助证据。**

### without airPLS 很有意思

`without airPLS` 的绿色很高、红色很低、灰色也低，但蓝色/橙色比 Full 低。

这表示：

**去掉 airPLS 后，预测更保守、更干净，但少覆盖了一些 GT。**

所以 airPLS/residual 的作用大概是：帮你扩召回，但也可能引入一些宽化风险。

------

## 2. 第二张：Default vs best operating points 看什么？

这张最重要，建议你优先看这一张。

它比较的是几个代表性操作点：

1. `Peak-Aware-Refined`：原 baseline；
2. `Full Spectral-Fusion-Refined`：默认 spectral fusion；
3. `trend_threshold_0.5`：当前扫描出来的最佳平衡点；
4. `Spectral-Fusion SG only`：高 purity / 低覆盖的保守点；
5. `combo_fusion_threshold0.45...`：duration-controlled 的折中点。

这里可以这样理解：

### `trend_threshold_0.5` 是召回最强点

它的蓝色和橙色最高：

- GT coverage 最高；
- supportable GT coverage 最高。

但它的灰色也最高，说明预测区间更宽。红色也不算最低。

所以它适合被称为：

**recall-oriented / supportable-coverage-oriented operating point**

不是“绝对最好”，而是“最能找回 GT”。

### `combo_fusion_threshold0.45...` 是更保守的折中点

它的蓝色/橙色略低于 `trend_threshold_0.5`，但是红色明显低、灰色也更低。

也就是说：

**它牺牲一点 GT coverage，换来更少的 unsupported coverage 和更短的预测区间。**

这个方法可能更适合作为论文里的“稳健主方法”，因为它不那么激进。

### `SG only` 不是主方法

它绿色很高，红色和灰色很低，但蓝色/橙色太低。

意思是：

**它预测得很少，所以预测出来的部分比较纯，但漏掉太多 GT。**

这不能作为最终方法，只能作为“保守下界”或“purity-oriented baseline”。

------

## 3. airPLS lambda 图：基本不用太在意

这张图里几条线几乎是平的。

意思是：

**airPLS lambda 在当前范围内不是关键参数。**

虽然 lambda 从很小扫到 100000，但 GT coverage、supportable coverage、purity 都变化很小。红色有一点变化，但整体不是主要矛盾。

所以你后面不用重点讲 airPLS lambda。可以一句话带过：

> airPLS baseline stiffness had limited influence on the final interval-level metrics, suggesting that the fusion stage was not highly sensitive to this preprocessing hyperparameter.

------

## 4. fusion_threshold 图：这是最重要的参数图之一

这张图非常关键。

> unsupportable指代人工 GT 标注为异常，但在 VAD/anomaly score 曲线上没有足够分数证据支持的那部分 GT
>
> 阈值是系统对interval计算的综合分数，报告里说 Spectral-Fusion-Refined 会融合多种证据，包括 normalized raw、SG、residual、trend、peak-count、length、low-residual evidence。
>
> `fusion_threshold` 越高，要求越严格：
>
> - 证据不够强的候选区间会被删掉；
>
> - 预测区间会变少、变短；
>
> - coverage 会下降；
>
> - purity 会上升；
>
> - unsupportable coverage 会下降。
>
> 低 fusion_threshold 会保留更多边缘候选区间，于是覆盖更多 GT，但也会覆盖很多 score 不支持的人工 GT。

横轴 `fusion_threshold` 越大，表示保留 interval 的要求越严格。

你看趋势：

- 蓝色 GT coverage 下降；
- 橙色 supportable coverage 下降；
- 绿色 predicted_GT_fraction 上升；
- 红色 unsupportable coverage 大幅下降；         
- 灰色 predicted duration ratio 下降。

这就是非常标准的 trade-off：

**阈值越高，预测越保守，覆盖率下降，但纯度上升，区间变短，unsupported 覆盖减少。**

所以 fusion threshold 是真正控制 operating point 的参数。它决定你要：

- 更激进地多覆盖 GT；
- 还是更保守地减少误报和宽区间。

这张图可以作为你报告里最重要的“参数解释图”。

------

## 5. length_penalty_weight 图：影响不大，但方向正确

`length_penalty_weight` 越大，越惩罚长区间。

图里能看到：

- 灰色 duration ratio 下降；
- 红色 unsupportable coverage 下降；
- 蓝色/橙色略微下降；
- 绿色略微上升。

这说明 length penalty 的行为是合理的：

**它确实在压缩区间、减少过宽预测，但效果不剧烈。**

所以它不是主控参数，而是一个“修边参数”。

你可以理解成：

> fusion_threshold 是方向盘，length_penalty_weight 是刹车。

------

## 6. peak_mad_k 图：影响很小

`peak_mad_k` 是局部峰检测里的阈值参数。

图里整体几乎不变，只有红色在 k=2.5 后下降一些。

这说明：

**最终结果对 peak 检测阈值不太敏感。**

这其实也验证了一个结论：

你的最终效果不是由“尖峰检测参数”决定的，而是由更高层的 fusion / trend / interval 机制决定的。

这对你之前工作不是坏事，而是说明：

**尖峰识别是辅助证据，不是主导变量。**

------

## 7. trend_window 图：小窗口略好，但整体影响有限

`trend_window` 从 30 到 300。

图里 30F 的蓝色/橙色最高，灰色也最高一点；到 50F 之后基本平稳。

这说明：

**较短 trend window 能捕捉更多 GT，但 window 再变大后收益不明显。**

也就是说，trend 证据确实有用，但你不用把 30、50、100、150、300 每个都当成很重要的差异。真正结论是：

> trend evidence helps, but the exact trend window is less important once it exceeds a moderate scale.

------

## 8. Coverage-purity Pareto field 怎么看？

这张散点图横轴是 `predicted_GT_fraction`，纵轴是 `GT_coverage`。

理想点在右上角：

- 越右：预测越纯；
- 越上：覆盖越高。

但是图里大多数点形成一条横向带：

- GT coverage 大多在 0.70–0.75；
- predicted_GT_fraction 大多在 0.50–0.52。

这说明：

**大多数参数组合其实差别不大，都在同一个 coverage-purity 区域里。**

右边有一些点 purity 高，但 coverage 很低，例如 SG only 那类。这种点不是好主方法，因为它只是预测得少。

所以这张图真正说明的是：

**当前参数扫描没有找到一个“又高覆盖又高纯度”的神奇点；大多数可用方法都在同一条 trade-off 前沿附近。**

这也支持“不急着贝叶斯优化”的判断。

------

## 9. Supportable vs unsupportable coverage 怎么看？

横轴是 supportable coverage，纵轴是 unsupportable coverage。

理想点在：

**右下角。**

也就是：

- supportable coverage 高；
- unsupportable coverage 低。

图里大多数点聚在右侧，supportable coverage 大概 0.72–0.78，但 unsupportable coverage 在 0.11–0.30 之间波动。

这说明：

**不同参数主要不是决定能不能覆盖 supportable GT，而是在决定会不会顺手覆盖很多 unsupportable GT。**

最上面那个点红色很高，可能就是 Peak-Aware-Refined 或某个过宽/过松方法。它虽然 supportable coverage 不差，但 unsupportable coverage 太高，说明它覆盖了很多没有 score 支持的 GT 区域。

这张图对论文很有用，因为它可以说明：

> 我们不只追求 GT coverage，而是区分 score-supported 和 score-unsupported GT，从而避免把不可由 score 解释的覆盖也算成成功。

------

## 10. Top 10 configurations 图怎么读？

这张图是按 `stricter_balanced_score` 排名前 10 的配置。

最左边是 `trend_threshold_0.5`，它排第一。

但你会发现前 10 个配置的形状很像：

- GT coverage 大多 0.70–0.77；
- supportable coverage 大多 0.72–0.78；
- predicted_GT_fraction 大多 0.50–0.54；
- duration ratio 大多 0.50–0.59；
- unsupportable coverage 有明显差别。

这说明：

**真正区分 top configurations 的，不是 GT coverage 差很多，而是 unsupportable coverage 和 duration ratio 的控制。**

所以你挑主方法时不能只看排名第一。可以这样选择：

- 想强调召回：选 `trend_threshold_0.5`；
- 想强调稳健和区间控制：选 `combo_fusion_threshold0.45_trend_window50_trend_weight0.35_length_penalty_weight0.3`；
- 想展示参数扫描不是乱调：报告 top 10，并说明它们集中在相似的 Pareto 区域。

------

## 最简单的总读法

你可以把这些图压缩成一句话理解：

**这个系统的主要矛盾不是“哪个预处理算法最好”，而是“要用多宽的预测区间换多少 GT coverage”。fusion threshold 和 trend threshold 是主控参数；SG、airPLS、peak_mad_k、lambda 这些更像辅助或低敏感参数。**

更具体一点：

1. **Full Spectral-Fusion 比 Peak-Aware 更能覆盖 GT，尤其是 supportable GT。**
2. **trend_threshold_0.5 是最强召回点，但区间更宽。**
3. **fusion_threshold 越高，越保守，purity 上升，coverage 下降。**
4. **length penalty 能温和压缩区间。**
5. **SG only / peak detection only 不适合作为主检测器。**
6. **airPLS lambda、peak_mad_k、trend_window 的细调不是主要收益来源。**
7. **后续应该围绕 operating point 和 error taxonomy 讲，而不是继续无限调参。**



更稳健的折中配置

fusion_threshold = 0.45
trend_window = 50
trend_weight = 0.35
length_penalty_weight = 0.3