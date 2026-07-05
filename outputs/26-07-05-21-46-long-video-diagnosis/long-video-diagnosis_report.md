# Long Video Multi-Anomaly Diagnosis

## 1. Background

URF-HVAA 当前的 suspicious window 机制由 `src/score_filter.py` 中的 `find_extreme_intervals(scores)` 实现。它对每个视频的异常分数序列使用窗口 `window_size = max(max_frame // 10, 300)`，从每个已打分帧开始计算窗口均分，最后只输出一个 `highest_interval` 和一个 `lowest_interval`。

该机制适合为单一显著异常生成 prior，但在长视频、多段异常或多峰 score curve 中，会把多个不连续异常压缩成一个 Wmax。

## 2. Data Availability

本次诊断不重新运行 VideoLLaMA3 / Llama3.1 / Qwen 等大模型，不假设 `.mp4` 存在，只使用本地 annotation 与 score JSON。

- XD-Violence: annotation 800 个视频；`refined_scores/videollama3` 有 800 个 JSON；无 `scores/videollama3`；无 raw videos；captions 有 800 个 JSON。
- UCF-Crime: annotation 290 个视频；`scores/videollama3` 有 290 个 JSON；`refined_scores/videollama3` 有 290 个 JSON；无 raw videos；captions 有 290 个 JSON。

因此当前可以做时间坐标、窗口覆盖、分数曲线与 event proposal 诊断；不能做原始视频视觉复核或依赖 `.mp4` 的再抽帧实验。

## 3. Existing Issue

已有报告 `reports/multi_anomaly_miss_test_report.md` 显示，在只看多异常视频时，single `highest_interval` 对 GT 异常片段覆盖不足：

- XD-Violence 多异常样本：998 个异常片段中遗漏 726 个，遗漏率 72.75%。
- UCF-Crime 多异常样本：32 个异常片段中遗漏 23 个，遗漏率 71.88%。

这不是完整 VAD 检测漏检率，而是 single-window coverage miss rate：单个 Wmax 对 GT 异常片段的覆盖遗漏率。

## 4. Dataset Diagnostics

统一事件索引输出：`outputs/anomaly_event_index.json` 和 `outputs/anomaly_event_index.csv`。

- 有异常视频数：640
- 多异常视频数：276
- 异常事件数：1394
- XD-Violence 事件数：1238
- UCF-Crime 事件数：156

分布摘要：

- 视频长度：min 96，median 1664，p75 2768，max 21696，mean 2421.975 frames
- 每视频异常数：min 1，median 1，p75 2，max 22，mean 2.178
- 异常持续时间：min 7，median 144，p75 378，max 15000，mean 446.566 frames
- 异常覆盖率：median 0.297，mean 0.385

这说明数据中既有单异常视频，也有明显多事件长尾；最大单视频可有 22 个异常片段。

## 5. Timeline Visualization

典型时间轴图已生成：

- `outputs/timeline_plots/XD-Violence/v_38GQ9L2meyE___1_label_B6-0-0.png`
- `outputs/timeline_plots/XD-Violence/v_uQY15O3LKI0___1_label_B6-0-0.png`
- `outputs/timeline_plots/UCF-Crime/Assault010_x264.png`

图中包含三层：GT anomaly intervals、original highest_interval、score curve。典型现象是：GT 异常分散在多个时间段，而 Wmax 只能覆盖一个局部高分区域。

## 6. Score-Level Diagnosis

遗漏片段分数诊断输出：`outputs/missed_interval_score_analysis.json`、`outputs/missed_interval_score_analysis.csv` 和 `reports/missed_interval_score_analysis.md`。

结果：

- 有 score 的 GT intervals：1394
- 被 single highest interval 遗漏：864
- 遗漏但 GT max score 位于全视频 score 的 80th percentile 以上：664
- 遗漏且分数较低/中等：199

按数据集：

- UCF-Crime: total 156，missed 86，missed_high_score 77
- XD-Violence: total 1238，missed 778，missed_high_score 587

解释：

- 大量遗漏片段本身已有较高异常分数，说明 score curve 已经看到了这些事件，主要瓶颈是 single-window 输出结构。
- 仍有一部分遗漏片段分数不高，说明还存在 caption/scoring 识别不足、窗口尺度不合适或分数噪声问题。

## 7. Top-K Coverage

Top-K interval 输出：

- `outputs/topk_intervals/topk_k1.json`
- `outputs/topk_intervals/topk_k2.json`
- `outputs/topk_intervals/topk_k3.json`
- `outputs/topk_intervals/topk_k5.json`
- `outputs/topk_intervals/topk_k10.json`

覆盖率评估输出：`outputs/topk_coverage_results.json`、`reports/topk_coverage_report.md`、`outputs/topk_coverage_curve.png`。

| K | Missed segments | Segment miss rate | Video any miss rate | Mean coverage |
|---:|---:|---:|---:|---:|
| 1 | 864 | 61.98% | 59.22% | 0.255 |
| 2 | 615 | 44.12% | 40.47% | 0.420 |
| 3 | 485 | 34.79% | 32.97% | 0.529 |
| 5 | 312 | 22.38% | 23.28% | 0.687 |
| 10 | 79 | 5.67% | 7.81% | 0.902 |

K 增大后遗漏率显著下降，尤其 K=10 将片段遗漏率降至 5.67%。这强烈支持：single Wmax 是主要瓶颈。

## 8. Multi-Scale Coverage

多尺度输出：`outputs/multiscale_intervals.json`、`reports/multiscale_coverage_report.md`、`outputs/multiscale_coverage_curve.png`。

当前朴素多尺度配置：

- fixed scales: 300, 600, 1200
- adaptive scales: `max_frame // 20`, `max_frame // 10`
- 每尺度先取候选，再合并 NMS，最终保留 10 个窗口

结果：

- Multiscale final K=10 segment miss rate: 22.88%
- Same-scale Top-K K=10 segment miss rate: 5.67%
- Original-like Top-K K=1 segment miss rate: 61.98%

当前多尺度版本明显优于 single Wmax，但弱于同尺度 K=10。可能原因是：多尺度候选先按尺度截断后再合并，部分有效同尺度候选被过早丢弃；NMS 对不同尺度窗口的抑制过强；最终排序仍只看 mean score，偏向短高分窗口或特定尺度。

## 9. Conclusion

结论：single Wmax / single `highest_interval` 是长视频多异常场景中的主要结构性瓶颈。

证据链：

- 多异常视频中，single-window 对 GT 片段遗漏率约 72%。
- 全部有异常事件中，K=1 遗漏率为 61.98%，K=10 下降到 5.67%。
- 被 single-window 遗漏的 864 个片段中，有 664 个片段本身在 score curve 上有高分信号。

因此，问题不只是模型没打分；大量情况下是已有高分异常没有被 single output schema 保留下来。

## 10. Next Method Proposal

建议将 prior 从 single-window prior 改为 multi-event proposal prior：

1. 用 Top-K same-scale intervals 替代单个 `highest_interval`，默认 K=5 或 K=10。
2. 加入 NMS/overlap 去重，避免候选全部挤在同一事件附近。
3. 对候选窗口构建 event timeline，每个 event 保留 start/end、mean score、max score、duration、caption summary 和 anomaly tags。
4. 在 VAU / VAL prior 中从 single suspicious part 改为 multi-event timeline 或 event graph reasoning。
5. 继续调试多尺度版本：先生成全量候选，再统一 NMS；排序时结合 mean score、max score、覆盖宽度和局部峰值，而不是仅依赖均分。

当前最稳妥的下一步不是直接替换主流程，而是把 `suspicious_intervals` 作为并行字段写出，与原 `highest_interval` 兼容，并在 downstream prompt 中显式传入多个候选事件。
