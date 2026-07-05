import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    coverage_by_intervals,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    score_metadata,
    score_path_for_video,
    write_json,
)
from src.baseline_interval_filters import peak_expanded_interval, random_same_length_interval  # noqa: E402
from src.score_filter import find_extreme_intervals  # noqa: E402


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_interval(meta: dict, interval: dict) -> list[float]:
    return [coverage_by_intervals(gt, [interval]) for gt in meta["intervals"]]


def summarize_rows(rows: list[dict], method: str) -> dict:
    subset = [row for row in rows if row["method"] == method]
    if not subset:
        return {}
    total = len(subset)
    missed = sum(row["missed"] for row in subset)
    videos = defaultdict(list)
    for row in subset:
        videos[(row["dataset"], row["video_id"])].append(row)
    any_miss = sum(any(item["missed"] for item in items) for items in videos.values())
    return {
        "method": method,
        "segments": total,
        "covered_segments": total - missed,
        "missed_segments": missed,
        "segment_miss_rate": round(missed / total, 6),
        "video_count": len(videos),
        "videos_with_any_miss": any_miss,
        "video_any_miss_rate": round(any_miss / len(videos), 6) if videos else None,
        "mean_coverage": round(sum(row["coverage"] for row in subset) / total, 6),
    }


def summarize_random_runs(random_run_summaries: list[dict]) -> dict:
    if not random_run_summaries:
        return {}

    def mean(values):
        return sum(values) / len(values)

    def std(values):
        avg = mean(values)
        return (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5

    keys = ["segment_miss_rate", "video_any_miss_rate", "mean_coverage"]
    result = {
        "method": "random_same_length",
        "runs": len(random_run_summaries),
    }
    for key in keys:
        values = [item[key] for item in random_run_summaries if item.get(key) is not None]
        result[f"{key}_mean"] = round(mean(values), 6)
        result[f"{key}_std"] = round(std(values), 6)
        result[f"{key}_min"] = round(min(values), 6)
        result[f"{key}_max"] = round(max(values), 6)
    return result


def build_experiment(runs: int, seed: int) -> dict:
    root = repo_root()
    deterministic_rows = []
    random_run_rows = []
    intervals = {"wmax": {}, "peak_expand": {}, "random_same_length": {}}

    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        intervals["wmax"][dataset] = {}
        intervals["peak_expand"][dataset] = {}
        intervals["random_same_length"][dataset] = {}
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            smeta = score_metadata(scores)
            video_length = estimate_video_length(smeta, meta["intervals"])
            wmax_s, wmax_e, wmax_avg, *_ = find_extreme_intervals({str(k): v for k, v in scores.items()})
            window_size = wmax_e - wmax_s
            wmax = {
                "start": int(wmax_s),
                "end": int(wmax_e),
                "mean_score": round(wmax_avg, 6),
                "window_size": int(window_size),
            }
            peak = peak_expanded_interval(scores, video_length=video_length, window_size=window_size)

            intervals["wmax"][dataset][video_id] = wmax
            intervals["peak_expand"][dataset][video_id] = peak

            for method, pred in [("wmax", wmax), ("peak_expand", peak)]:
                coverages = evaluate_interval(meta, pred)
                for idx, cov in enumerate(coverages, start=1):
                    deterministic_rows.append(
                        {
                            "method": method,
                            "run": "",
                            "dataset": dataset,
                            "video_id": video_id,
                            "anomaly_id": idx,
                            "coverage": cov,
                            "missed": cov < 0.1,
                            "interval_start": pred["start"],
                            "interval_end": pred["end"],
                            "interval_mean_score": pred.get("mean_score"),
                            "window_size": window_size,
                            "video_length_frame_est": video_length,
                        }
                    )

            for run_idx in range(runs):
                rng = random.Random(seed + run_idx * 1000003 + stable_int(f"{dataset}::{video_id}") % 100000)
                random_interval = random_same_length_interval(scores, video_length, rng, window_size=window_size)
                intervals["random_same_length"].setdefault(dataset, {}).setdefault(video_id, []).append(random_interval)
                coverages = evaluate_interval(meta, random_interval)
                for idx, cov in enumerate(coverages, start=1):
                    random_run_rows.append(
                        {
                            "method": "random_same_length",
                            "run": run_idx,
                            "dataset": dataset,
                            "video_id": video_id,
                            "anomaly_id": idx,
                            "coverage": cov,
                            "missed": cov < 0.1,
                            "interval_start": random_interval["start"],
                            "interval_end": random_interval["end"],
                            "interval_mean_score": random_interval.get("mean_score"),
                            "window_size": window_size,
                            "video_length_frame_est": video_length,
                        }
                    )

    deterministic_summary = [
        summarize_rows(deterministic_rows, "wmax"),
        summarize_rows(deterministic_rows, "peak_expand"),
    ]
    random_run_summaries = []
    for run_idx in range(runs):
        run_rows = [row for row in random_run_rows if row["run"] == run_idx]
        random_run_summaries.append(summarize_rows(run_rows, "random_same_length"))

    return {
        "runs": runs,
        "seed": seed,
        "deterministic_rows": deterministic_rows,
        "random_rows": random_run_rows,
        "summary": {
            "deterministic": deterministic_summary,
            "random_same_length": summarize_random_runs(random_run_summaries),
            "random_runs": random_run_summaries,
        },
        "intervals": intervals,
    }


def write_report(path: Path, summary: dict) -> None:
    deterministic = summary["deterministic"]
    random_summary = summary["random_same_length"]
    lines = [
        "# Wmax Replacement Baseline Experiment",
        "",
        "This experiment compares three single-interval strategies with the same window length as the original Wmax:",
        "",
        "- `wmax`: original highest-average sliding window.",
        "- `random_same_length`: random interval with the same length, repeated multiple times.",
        "- `peak_expand`: expand a same-length interval around the maximum score peak; if multiple max peaks exist, choose the expanded interval with the highest mean anomaly score.",
        "",
        "## Results",
        "",
        "| Method | Segment miss rate | Video any miss rate | Mean coverage |",
        "|---|---:|---:|---:|",
    ]
    for item in deterministic:
        lines.append(
            f"| {item['method']} | {item['segment_miss_rate']:.2%} | "
            f"{item['video_any_miss_rate']:.2%} | {item['mean_coverage']:.3f} |"
        )
    lines.append(
        f"| random_same_length mean ({random_summary['runs']} runs) | "
        f"{random_summary['segment_miss_rate_mean']:.2%} ± {random_summary['segment_miss_rate_std']:.2%} | "
        f"{random_summary['video_any_miss_rate_mean']:.2%} ± {random_summary['video_any_miss_rate_std']:.2%} | "
        f"{random_summary['mean_coverage_mean']:.3f} ± {random_summary['mean_coverage_std']:.3f} |"
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Random same-length intervals estimate how much coverage comes from window length and anomaly density alone.",
            "- Peak-expanded intervals test whether anchoring on the strongest score point is enough, independent of averaging all candidate windows.",
            "- If peak expansion approaches Wmax, the original averaging window mainly selects around score peaks. If it underperforms Wmax, the surrounding score distribution matters.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path: Path, summary: dict) -> None:
    methods = [item["method"] for item in summary["deterministic"]] + ["random_mean"]
    miss_rates = [item["segment_miss_rate"] for item in summary["deterministic"]]
    miss_rates.append(summary["random_same_length"]["segment_miss_rate_mean"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, miss_rates, color=["#4c78a8", "#59a14f", "#f28e2b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("segment miss rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    root = repo_root()
    result = build_experiment(args.runs, args.seed)
    write_json(root / "outputs/wmax_replacement_baselines/intervals.json", result["intervals"])
    write_json(root / "outputs/wmax_replacement_baselines/summary.json", result["summary"])
    write_csv(root / "outputs/wmax_replacement_baselines/deterministic_interval_coverage.csv", result["deterministic_rows"])
    write_csv(root / "outputs/wmax_replacement_baselines/random_interval_coverage.csv", result["random_rows"])
    write_report(root / "reports/wmax_replacement_baseline_experiment.md", result["summary"])
    write_plot(root / "outputs/wmax_replacement_baselines/segment_miss_rate.png", result["summary"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
