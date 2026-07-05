# Research Gap Assessment

Date: 2026-07-05

This note assesses whether the three proposed research directions correspond to real limitations in the current URF-HVAA codebase. The assessment is based on local repository inspection and one synthetic probe of `src.score_filter.find_extreme_intervals`. It does not yet include real-dataset experiments because the official data, precomputed scores, raw videos, and checkpoints are not currently available locally.

## 1. Multiple suspicious segments in long videos

Status: confirmed as a real code/pipeline limitation.

Evidence:

- `src/score_filter.py` computes a single fixed window size per video:
  - `window_size = max(max_frame // 10, 300)`
- It returns only one highest-average window and one lowest-average window:
  - `(best_s, best_e, best_avg, worst_s, worst_e, worst_avg)`
- It writes only:
  - `highest_interval`
  - `highest_avg_score`
  - `lowest_interval`
  - `lowest_avg_score`
  - summary statistics
- Downstream code consumes the single highest interval:
  - `src/summarize_window.py` reads `interval_info["highest_interval"]`.
  - `src/draw_bboxes.py` samples frames from `info["highest_interval"]`.
  - `src/vau_priors.py` loads `data["highest_interval"]` and `data["highest_avg_score"]`.
  - `src/refine_with_tag.py` gates videos using `highest_avg_score`.
- A synthetic probe with two score peaks still produced only one highest window and one lowest window:
  - output: `(0, 300, 0.31842105263157894, 256, 556, 0.05)`

Missing pieces:

- A real-dataset measurement of how many videos have multiple separated ground-truth anomaly intervals.
- Candidate-recall metrics such as whether Top-K candidate windows cover all labeled intervals.
- A comparison between:
  - current single-window selection;
  - thresholded candidates;
  - multi-scale windows;
  - Top-K;
  - temporal non-maximum suppression or interval deduplication.

Minimum validation experiment:

- Use existing score JSONs or precomputed scores.
- Parse temporal annotation start/end pairs.
- Generate candidate windows from score curves using several scales and thresholds.
- Report:
  - candidate recall at IoU/time-overlap thresholds;
  - number of GT anomaly intervals covered per video;
  - missed secondary intervals;
  - extra candidate count per video.

## 2. Sustained scene-rule violations become normalized by VLMs

Status: plausible and important, but not yet proven by local evidence.

What the code supports:

- `src/video_pre_caption.py` captions local windows independently around frame indices.
- `src/llm_anomaly_scorer.py` scores each caption independently with a prompt asking whether the described scene is suspicious.
- There is no explicit scene-rule memory, normal-context model, or comparison between early and late parts of a sustained condition.
- The current pipeline does not appear to track whether a visually stable abnormal state loses anomaly score over time.

What is not yet proven:

- The local code inspection does not prove VideoLLaMA3 or Llama3.1 actually normalizes sustained abnormalities.
- This requires real or curated videos where a persistent violation remains visible but becomes visually repetitive.

Missing pieces:

- A dataset subset with sustained state anomalies, for example:
  - abandoned object remaining in a restricted area;
  - person lying motionless for a long period;
  - vehicle stopped in an abnormal zone;
  - crowd/traffic rule violation persisting across many clips.
- Clip-level score curves showing whether anomaly score decays during the same sustained abnormal condition.
- Captions over time showing whether the abnormal object/state disappears from the description or becomes background.
- A scene-rule baseline or prompt that distinguishes persistent rule violation from visual novelty.

Minimum validation experiment:

- Select sustained-anomaly videos and manually mark the start/end of the sustained state.
- Run or use precomputed captions and scores.
- Plot score versus time for the sustained interval.
- Measure:
  - early-state score;
  - middle-state score;
  - late-state score;
  - caption mention rate of the abnormal condition.
- Confirm the issue only if scores or captions systematically fade while the abnormal condition remains visible.

## 3. Small anomalies are ignored when a larger anomaly is present

Status: plausible, partially implied by the single-candidate design, but not yet empirically proven.

What the code supports:

- The current suspicious-window path stores only the maximum-average window. If one major anomaly dominates the score curve, weaker events are not stored as independent candidates.
- Downstream tag extraction, bounding-box drawing, refinement priors, and VAU priors use the single highest-window artifact.
- Global frame-level ROC/PR in `src/eval.py` can look acceptable even if a secondary event in the same video is not separately surfaced as a candidate.

What is not yet proven:

- The current local repository does not include a real evaluation showing small anomalies are missed because of large anomalies.
- It is also possible that frame-level scores still rise on small events, but the current candidate-selection layer discards them.

Missing pieces:

- Videos with multiple anomaly severities in the same long clip.
- Instance/segment-level labels distinguishing major and minor anomalies.
- Per-event recall, not only global frame-level AUC/AP.
- A severity-stratified analysis of score peaks.

Minimum validation experiment:

- Create a multi-event evaluation subset.
- Label each event with approximate severity or salience.
- Compare current single-window output against Top-K/multi-scale candidates.
- Report recall for:
  - strongest event per video;
  - non-strongest events;
  - low-duration or low-score events;
  - events far away from the strongest event.

## Practical next step

The first direction is the safest starting point because it is already supported by concrete code evidence and can be tested using only score JSONs plus temporal annotations. Directions 2 and 3 should be framed as hypotheses until real videos/precomputed captions/scores are available.
