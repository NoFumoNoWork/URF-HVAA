### fig_duration_vs_max_score.png & fig_duration_vs_mean_score.png

![fig_duration_vs_max_score](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_duration_vs_max_score.png)

- 很多人工异常段里，**至少有某个时刻机器也认为很异常**



---

### fig

![fig_duration_vs_mean_score](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_duration_vs_mean_score.png)

-  `mean_score` 分布很散，尤其很多 GT 的 max_score 很高，但 mean_score 并不高
- 人工标注常常是事件级大区间，而 score 响应更像局部关键瞬间



---

### fig_gt_support_by_label_topk.png

![fig_gt_support_by_label_topk](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_gt_support_by_label_topk.png)

- 最大值为2是因为两个数据集的比例被加在一起了



| 类型                          | 判断标准                                                     |
  | ----------------------------- | ------------------------------------------------------------ |
  | `strongly_score_supported`    | score 点数足够，并且 `max_score >= 0.8` **或** `mean_score >= 0.6` |
  | `weakly_score_supported`      | score 点数足够，且不满足 strong，但 `max_score >= 0.5`       |
  | `score_unsupported`           | score 点数足够，且 `max_score < 0.4`                         |
  | `ambiguous_mid_score`         | 不属于以上几类的中间分数情况                                 |
  | `sparsely_sampled`            | `2 <= score_point_count < 5`                                 |
  | `barely_sampled`              | `0 < score_point_count < 2`，也就是通常只有 1 个 score 点    |
  | `unobserved_or_missing_score` | `score_point_count == 0`，或 mean/max score 缺失，或 score 文件缺失 |



---

### fig_recoverable_upper_bound.png

![fig_recoverable_upper_bound](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_recoverable_upper_bound.png)

- 如果只改后处理，理论上最多能救回 GT 的比例
- 后处理理论上还有空间，但不可能解决全部问题
- 对于 score 本身没有响应的区间，peak-aware、merge、split 都没法凭空恢复



---

### **fig_method_gt_coverage_vs_purity.png**

![fig_method_gt_coverage_vs_purity](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_method_gt_coverage_vs_purity.png)

- 点越靠右，预测越干净；点越靠上，GT 覆盖越多
- 理想方法在右上角。



---

### fig_param_sensitivity_fusion_threshold.png

![image-20260707210931299](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_param_sensitivity_fusion_threshold.png)

unsupportable指代人工 GT 标注为异常，但在 VAD/anomaly score 曲线上没有足够分数证据支持的那部分 GT

阈值是系统对interval计算的综合分数，报告里说 Spectral-Fusion-Refined 会融合多种证据，包括 normalized raw、SG、residual、trend、peak-count、length、low-residual evidence。

`fusion_threshold` 越高，要求越严格：

- 证据不够强的候选区间会被删掉；

- 预测区间会变少、变短；

- coverage 会下降；

- purity 会上升；

- unsupportable coverage 会下降。

低 fusion_threshold 会保留更多边缘候选区间，于是覆盖更多 GT，但也会覆盖很多 score 不支持的人工 GT。

**阈值越高，预测越保守，覆盖率下降，但纯度上升，区间变短，unsupported 覆盖减少。**



---

### fig_recall_vs_strict_tradeoff.png

![fig_recall_vs_strict_tradeoff](T:\Bigwork\SMILES.URF-HVAA\key_figs\fig_recall_vs_strict_tradeoff.png)

- 横轴：stricter balanced score；
- 纵轴：GT coverage；
- 颜色：unsupported coverage。

理想上，越右上越好，颜色越浅越好。

recall trend=0.5, SG=0, residual=0.25的效果最好，但unsupported rate不是最低

