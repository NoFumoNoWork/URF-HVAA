import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    coverage_by_intervals,
    estimate_video_length,
    load_scores,
    mean,
    median,
    parse_temporal_annotations,
    percentile_rank,
    repo_root,
    score_metadata,
    score_path_for_video,
    score_values_in_interval,
    write_csv,
    write_json,
)
from src.score_filter import find_extreme_intervals  # noqa: E402


def analyze() -> list[dict]:
    root = repo_root()
    rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            best_s, best_e, best_avg, *_ = find_extreme_intervals({str(k): v for k, v in scores.items()})
            selected = [{"start": best_s, "end": best_e}]
            selected_vals = score_values_in_interval(scores, best_s, best_e)
            normal_vals = [
                value
                for frame, value in scores.items()
                if not any(i["start"] <= frame < i["end"] for i in meta["intervals"])
            ]
            all_vals = list(scores.values())
            smeta = score_metadata(scores)
            video_len = estimate_video_length(smeta, meta["intervals"])
            for idx, interval in enumerate(meta["intervals"], start=1):
                vals = score_values_in_interval(scores, interval["start"], interval["end"])
                gt_mean = mean(vals)
                gt_max = max(vals) if vals else None
                coverage = coverage_by_intervals(interval, selected)
                rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "label_or_class": meta["label"],
                        "video_length_frame_est": video_len,
                        "anomaly_id": idx,
                        "anomaly_start": interval["start"],
                        "anomaly_end": interval["end"],
                        "gt_interval_coverage_by_highest": round(coverage, 6),
                        "missed_by_highest": coverage < 0.1,
                        "gt_mean_score": round(gt_mean, 6) if gt_mean is not None else None,
                        "gt_max_score": round(gt_max, 6) if gt_max is not None else None,
                        "gt_median_score": round(median(vals), 6) if vals else None,
                        "selected_window_mean_score": round(mean(selected_vals), 6) if selected_vals else None,
                        "selected_window_max_score": round(max(selected_vals), 6) if selected_vals else None,
                        "normal_region_mean_score": round(mean(normal_vals), 6) if normal_vals else None,
                        "score_rank_percentile_of_gt_max": round(percentile_rank(all_vals, gt_max), 6) if gt_max is not None else None,
                        "score_rank_percentile_of_gt_mean": round(percentile_rank(all_vals, gt_mean), 6) if gt_mean is not None else None,
                    }
                )
    return rows


def write_report(path: Path, rows: list[dict]) -> None:
    missed = [r for r in rows if r["missed_by_highest"]]
    high_score_missed = [
        r for r in missed
        if r["score_rank_percentile_of_gt_max"] is not None and r["score_rank_percentile_of_gt_max"] >= 0.8
    ]
    low_score_missed = [
        r for r in missed
        if r["score_rank_percentile_of_gt_max"] is not None and r["score_rank_percentile_of_gt_max"] < 0.8
    ]
    by_dataset = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        ds = [r for r in rows if r["dataset"] == dataset]
        ds_missed = [r for r in ds if r["missed_by_highest"]]
        ds_high = [r for r in ds_missed if r["score_rank_percentile_of_gt_max"] is not None and r["score_rank_percentile_of_gt_max"] >= 0.8]
        by_dataset[dataset] = (len(ds), len(ds_missed), len(ds_high))

    lines = [
        "# Missed Interval Score Analysis",
        "",
        f"- Total GT intervals with scores: {len(rows)}",
        f"- Missed by single highest interval: {len(missed)}",
        f"- Missed but high score percentile (gt_max >= 80th percentile): {len(high_score_missed)}",
        f"- Missed and low/moderate score percentile: {len(low_score_missed)}",
        "",
        "## By Dataset",
        "",
    ]
    for dataset, (total, miss, high) in by_dataset.items():
        lines.append(f"- {dataset}: total={total}, missed={miss}, missed_high_score={high}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Missed intervals with high score percentiles indicate the output structure is a bottleneck: the score curve saw the event, but one Wmax could not keep multiple regions.",
            "- Missed intervals with low score percentiles indicate additional score/caption recognition weakness or unsuitable temporal scale.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    rows = analyze()
    write_json(root / "outputs/missed_interval_score_analysis.json", rows)
    write_csv(root / "outputs/missed_interval_score_analysis.csv", rows)
    write_report(root / "reports/missed_interval_score_analysis.md", rows)
    print(json.dumps({"intervals": len(rows), "output": "outputs/missed_interval_score_analysis.json"}, indent=2))


if __name__ == "__main__":
    main()
