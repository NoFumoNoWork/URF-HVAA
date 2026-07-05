# 长视频多异常遗漏测试报告

## 1. 测试目的

本测试针对当前仓库中的异常窗口筛选策略进行数据分析：当一个长视频中存在多个异常片段时，`src/score_filter.py` 只输出单个最高均分窗口 `highest_interval`，可能只覆盖最大或最显著异常，而遗漏其他异常片段。

本报告使用本地已有标注和分数文件，不重新跑大模型推理。统计脚本为 `scripts/analyze_multi_anomaly_misses.py`，输出为 `outputs/multi_anomaly_miss_analysis.json`。

## 2. 测试视频来源及长度

数据来源来自仓库 README 中说明的预处理异常检测数据包，以及本地已存在的数据目录：

- XD-Violence: `data/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt`
- XD-Violence 分数: `data/xd_violence/refined_scores/videollama3/`
- UCF-Crime: `data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`
- UCF-Crime 分数: `data/ucf_crime/scores/videollama3/`

本次只统计“同一视频含两个及以上标注异常区间，且本地存在分数 JSON”的样本。

| 数据集 | 多异常视频数 | 视频长度范围/帧 | 平均长度/帧 | 约合时长范围/秒，按 30 FPS 估算 |
|---|---:|---:|---:|---:|
| XD-Violence | 260 | 304 - 19600 | 2557.2 | 10.1 - 653.3 |
| UCF-Crime | 16 | 1024 - 16192 | 4199.0 | 34.1 - 539.7 |

说明：`run_eval.ps1` 中评估参数使用 `--video_fps 30`，因此报告中的秒数按 30 FPS 估算。真实原始视频 FPS 如果不同，秒级时长会有偏差，但帧级覆盖统计不受影响。

## 3. 数据标注方法

XD-Violence 标注文件每行格式为：

```text
video_id label start_1 end_1 start_2 end_2 ... -1 -1 ...
```

例如同一视频可包含多组异常区间，异常类型可能是 `Car_accident`、`Fighting`、`Riot`、`Shooting`、`Explosion` 等。`Normal` 行通常以 `-1` 表示无异常区间。

UCF-Crime 标注文件每行格式为：

```text
video_id class start_1 end_1 start_2 end_2
```

本次统计把每一对合法的 `start/end` 作为一个独立异常片段；`-1` 后续内容视为无效填充。

## 4. 窗口切分与当前算法

当前窗口筛选函数位于 `src/score_filter.py` 的 `find_extreme_intervals(scores)`：

- 输入：每个视频一个 JSON，键为帧号，值为异常分数。
- 分数采样：本地分数通常以 16 帧为间隔，如 `0, 16, 32, ...`。
- 窗口长度：`window_size = max(max_frame // 10, 300)`。
- 候选窗口：从每个已打分帧 `s` 开始，窗口为 `[s, s + window_size)`。
- 评分方式：计算窗口内已有分数的平均值。
- 输出：只返回一个最高均分窗口 `highest_interval` 和一个最低均分窗口 `lowest_interval`。

这意味着，即使一个视频有多个空间上分离的异常片段，后续流程也只拿到一个最高窗口；没有 top-k、非极大值抑制、阈值连通区间或多峰保留机制。

## 5. 遗漏判定方法

对每个标注异常区间，计算其与单个 `highest_interval` 的重叠比例：

```text
coverage = overlap(annotation_interval, highest_interval) / annotation_interval_length
```

若覆盖比例低于 10%，判定该标注异常片段被遗漏。10% 是宽松阈值：只要最高窗口碰到标注片段的一小部分，就算覆盖。因此统计结果偏保守，真实业务可见遗漏可能更严重。

## 6. 多异常遗漏结果

| 数据集 | 标注异常片段总数 | 被遗漏片段数 | 片段遗漏率 | 有任意遗漏的视频数 | 视频遗漏率 |
|---|---:|---:|---:|---:|---:|
| XD-Violence | 998 | 726 | 72.75% | 248/260 | 95.38% |
| UCF-Crime | 32 | 23 | 71.88% | 16/16 | 100.00% |

主要结论：

- 单最高窗口策略在多异常视频上系统性遗漏，不是个别边界情况。
- XD-Violence 的多异常样本更能暴露问题：260 个样本中 248 个至少漏掉一个异常片段。
- UCF-Crime 的双异常样本全部出现遗漏，其中多例两个异常段都未被最高窗口覆盖。

## 7. 典型遗漏案例

### XD-Violence: `v=38GQ9L2meyE__#1_label_B6-0-0`

- 类别：`Car_accident`
- 视频长度：4656 帧，约 155.2 秒
- 窗口长度：464 帧
- 当前选中最高窗口：`[1600, 2064]`
- 标注异常片段数：22
- 遗漏片段数：18

该视频异常片段分布在全视频多个位置，单最高窗口只覆盖中段附近少量片段，前段和后段大量事故片段没有进入输出。

### XD-Violence: `v=uQY15O3LKI0__#1_label_B6-0-0`

- 类别：`Car_accident`
- 视频长度：3984 帧，约 132.8 秒
- 当前选中最高窗口：`[16, 412]`
- 标注异常片段数：17
- 遗漏片段数：16

最高窗口偏向最前面的局部高分区域，后续密集出现的异常段几乎全部遗漏。

### UCF-Crime: `Assault010_x264`

- 类别：`Assault`
- 视频长度：16192 帧，约 539.7 秒
- 窗口长度：1617 帧
- 当前选中最高窗口：`[13568, 15185]`
- 标注异常片段：`[11330, 11680]`、`[12260, 12930]`
- 遗漏片段数：2/2

该案例说明单最高窗口不一定落在标注异常处；当分数后段偏高或噪声较强时，两个真实异常段都可能被漏掉。

## 8. 原因分析

1. 输出结构限制：`highest_interval` 是单值字段，天然无法表达多个不连续异常。
2. 平均分窗口偏置：长窗口会奖励持续高分区域，短而尖锐的异常段可能被平均稀释。
3. 长视频窗口过宽：`max_frame // 10` 会随视频长度增长，长视频中一个窗口可能跨越较长时间，但仍只能代表一个局部区域。
4. 缺少多峰保留：没有对高分曲线做阈值连通、局部峰值聚类、top-k 窗口或 NMS 去重。
5. 末尾窗口可越界：例如窗口 `[7408, 8148]` 可超出视频末尾；这不一定导致统计错误，但会让窗口语义不够干净。

## 9. 建议

- 将 `highest_interval` 扩展为 `suspicious_intervals`，保留 top-k 高分窗口。
- 对候选窗口做 NMS，避免 top-k 全部挤在同一异常附近。
- 增加阈值连通区间策略：对帧级/片段级分数平滑后，提取所有超过阈值的连续异常段。
- 报告每个异常窗口的均分、峰值、长度、覆盖帧数和排序，方便后续解释与人工复核。
- 对长视频单独设置更小步长和多尺度窗口，例如 300、600、1200 帧并行，再合并区间。

## 10. 复现命令

```powershell
python scripts\analyze_multi_anomaly_misses.py
```

生成文件：

- `outputs/multi_anomaly_miss_analysis.json`
- `reports/multi_anomaly_miss_test_report.md`

## 11. 限制

- 本报告使用本地已有分数文件，没有重新跑 VideoLLaMA3 或 Llama3.1 推理。
- XD-Violence 使用的是本地 `refined_scores/videollama3`，而不是缺失的 `scores/videollama3` 路径。
- 秒级长度按 30 FPS 估算；帧级统计为主结论。
- 覆盖阈值设置为 10%，属于保守口径。
