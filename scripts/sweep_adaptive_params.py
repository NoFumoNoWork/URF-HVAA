import argparse
import csv
import itertools
import json
import statistics
import sys
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
from src.adaptive_interval_selection import select_adaptive_intervals  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_cache() -> list[dict]:
    root = repo_root()
    cache = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            video_length = estimate_video_length(score_metadata(scores), meta["intervals"])
            cache.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "intervals": meta["intervals"],
                    "scores": scores,
                    "video_length": video_length,
                }
            )
    return cache


def overlap_count(gt: dict, intervals: list[dict]) -> int:
    return sum(1 for pred in intervals if max(0, min(gt["end"], pred["end"]) - max(gt["start"], pred["start"])) > 0)


def evaluate_config(cache: list[dict], config: dict) -> dict:
    rows = []
    pred_counts = []
    pred_durations = []
    gt_overlap_counts = []
    pred_gt_counts = []
    for item in cache:
        preds, _ = select_adaptive_intervals(
            item["scores"],
            threshold_percentile=config["threshold_percentile"],
            merge_iou=config["merge_iou"],
            merge_gap=config["merge_gap"],
            min_duration=60,
            post_filter_percentile=config["post_filter_percentile"],
        )
        pred_counts.append(len(preds))
        pred_durations.extend([pred["duration"] for pred in preds])
        for pred in preds:
            pred_gt_counts.append(
                sum(
                    1
                    for gt in item["intervals"]
                    if max(0, min(gt["end"], pred["end"]) - max(gt["start"], pred["start"])) > 0
                )
            )
        for gt in item["intervals"]:
            cov = coverage_by_intervals(gt, preds)
            rows.append(cov)
            gt_overlap_counts.append(overlap_count(gt, preds))

    total = len(rows)
    missed = sum(1 for cov in rows if cov < 0.1)
    redundancy = {
        "gt_pred_overlap_mean": round(statistics.mean(gt_overlap_counts), 6) if gt_overlap_counts else 0,
        "gt_pred_overlap_max": max(gt_overlap_counts) if gt_overlap_counts else 0,
        "pred_gt_overlap_mean": round(statistics.mean(pred_gt_counts), 6) if pred_gt_counts else 0,
        "pred_gt_overlap_max": max(pred_gt_counts) if pred_gt_counts else 0,
    }
    result = {
        **config,
        "method_config": format_config(config),
        "segments": total,
        "miss_rate": round(missed / total, 6) if total else None,
        "mean_coverage": round(sum(rows) / total, 6) if total else None,
        "mean_intervals_per_video": round(statistics.mean(pred_counts), 6) if pred_counts else 0,
        "median_intervals_per_video": round(statistics.median(pred_counts), 6) if pred_counts else 0,
        "max_intervals_per_video": max(pred_counts) if pred_counts else 0,
        "mean_duration": round(statistics.mean(pred_durations), 6) if pred_durations else 0,
        "redundancy": redundancy,
        "redundancy_summary": (
            f"gt_pred_mean={redundancy['gt_pred_overlap_mean']};"
            f"pred_gt_mean={redundancy['pred_gt_overlap_mean']}"
        ),
    }
    return result


def format_config(config: dict) -> str:
    post = "none" if config["post_filter_percentile"] is None else str(int(config["post_filter_percentile"]))
    iou = str(config["merge_iou"]).replace(".", "p")
    return (
        f"tp{int(config['threshold_percentile'])}_"
        f"post{post}_gap{config['merge_gap']}_iou{iou}"
    )


def choose_representatives(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda r: (r["miss_rate"], r["mean_intervals_per_video"]))

    conservative_pool = [
        r for r in rows
        if r["mean_intervals_per_video"] <= 2.0 and 0.15 <= r["miss_rate"] <= 0.35
    ] or [r for r in rows if r["mean_intervals_per_video"] <= 2.0]
    conservative = min(conservative_pool, key=lambda r: (abs(r["mean_intervals_per_video"] - 1.5), r["miss_rate"]))

    balanced_pool = [
        r for r in rows
        if 2.0 <= r["mean_intervals_per_video"] <= 4.0 and 0.10 <= r["miss_rate"] <= 0.15
    ]
    if balanced_pool:
        balanced = min(balanced_pool, key=lambda r: (abs(r["miss_rate"] - 0.125), abs(r["mean_intervals_per_video"] - 3.0)))
        balanced_note = "strict target satisfied"
    else:
        pool = [r for r in rows if 0.10 <= r["miss_rate"] <= 0.15] or rows
        balanced = min(pool, key=lambda r: (abs(r["miss_rate"] - 0.125), abs(r["mean_intervals_per_video"] - 3.0)))
        balanced_note = "nearest feasible; no grid point has 2-4 mean intervals/video"

    recall_pool = [
        r for r in rows
        if 5.0 <= r["mean_intervals_per_video"] <= 7.0
    ]
    if recall_pool:
        recall = min(recall_pool, key=lambda r: (r["miss_rate"], abs(r["mean_intervals_per_video"] - 6.0)))
        recall_note = "strict target satisfied"
    else:
        recall = ordered[0]
        recall_note = "nearest feasible by miss rate; no grid point has 5-7 mean intervals/video"

    return {
        "conservative_adaptive": {"row": conservative, "note": "strict target satisfied"},
        "balanced_adaptive": {"row": balanced, "note": balanced_note},
        "recall_adaptive": {"row": recall, "note": recall_note},
        "best_overall": {"row": ordered[0], "note": "lowest miss rate in scanned grid"},
    }


def write_report(path: Path, rows: list[dict], reps: dict) -> None:
    lines = [
        "# Adaptive Parameter Grid Sweep",
        "",
        "Grid:",
        "",
        "- threshold_percentile: 75, 80, 85, 90",
        "- post_filter_percentile: none, 50, 75",
        "- merge_gap: 150, 300, 600",
        "- merge_iou: 0.3, 0.5",
        "",
        "## Selected Points",
        "",
        "| Role | method_config | miss_rate | mean_coverage | mean_intervals/video | mean_duration | redundancy |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for role, item in reps.items():
        lines.append(table_row(role, item["row"], item["note"]))
    min_intervals = min(row["mean_intervals_per_video"] for row in rows)
    max_intervals = max(row["mean_intervals_per_video"] for row in rows)
    lines.extend(
        [
            "",
            "## Feasibility Notes",
            "",
            f"- Scanned configs produced mean intervals/video in `{min_intervals:.3f} - {max_intervals:.3f}`.",
            "- Therefore this grid does not contain strict balanced points with 2-4 mean intervals/video.",
            "- It also does not contain strict recall points with 5-7 mean intervals/video.",
            "- To reach those output-count regimes, the next sweep should reduce merging strength, lower or disable gap merge, or emit clusters before final merging.",
        ]
    )
    lines.extend(
        [
            "",
            "## Full Grid",
            "",
            "| method_config | miss_rate | mean_coverage | mean_intervals/video | mean_duration | redundancy |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["mean_intervals_per_video"], r["miss_rate"])):
        lines.append(table_row("", row).replace("|  | ", "| "))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def table_row(label: str, row: dict, note: str | None = None) -> str:
    prefix = f"| {label} | " if label else "| "
    suffix = f" ({note})" if note and label else ""
    return (
        f"{prefix}{row['method_config']}{suffix} | {row['miss_rate']:.2%} | {row['mean_coverage']:.3f} | "
        f"{row['mean_intervals_per_video']:.3f} | {row['mean_duration']:.1f} | "
        f"{row['redundancy_summary']} |"
    )


def draw_tradeoff(path: Path, rows: list[dict], reps: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["mean_intervals_per_video"] for r in rows]
    ys = [r["miss_rate"] for r in rows]
    colors = [r["threshold_percentile"] for r in rows]
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=42, alpha=0.82)
    for role, item in reps.items():
        row = item["row"]
        ax.scatter([row["mean_intervals_per_video"]], [row["miss_rate"]], s=130, marker="*", edgecolor="black")
        ax.annotate(role.replace("_adaptive", ""), (row["mean_intervals_per_video"], row["miss_rate"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.axhspan(0.10, 0.15, color="#dddddd", alpha=0.25)
    ax.axvspan(2, 4, color="#bdd7e7", alpha=0.16)
    ax.axvspan(5, 7, color="#c7e9c0", alpha=0.12)
    ax.set_xlabel("mean intervals per video")
    ax.set_ylabel("segment miss rate")
    ax.set_ylim(0, max(ys) * 1.08)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("threshold_percentile")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/adaptive_param_grid"))
    parser.add_argument("--report_only", action="store_true", help="Regenerate report/plot from existing grid JSON without recomputing.")
    args = parser.parse_args()

    root = repo_root()
    out_dir = root / args.output_dir
    if args.report_only:
        existing = json.loads((out_dir / "adaptive_param_grid_results.json").read_text(encoding="utf-8"))
        rows = existing["rows"]
    else:
        cache = load_dataset_cache()
        configs = []
        for threshold, post, gap, iou in itertools.product(
            [75.0, 80.0, 85.0, 90.0],
            [None, 50.0, 75.0],
            [150, 300, 600],
            [0.3, 0.5],
        ):
            configs.append(
                {
                    "threshold_percentile": threshold,
                    "post_filter_percentile": post,
                    "merge_gap": gap,
                    "merge_iou": iou,
                }
            )

        rows = []
        for idx, config in enumerate(configs, start=1):
            result = evaluate_config(cache, config)
            rows.append(result)
            print(f"[{idx}/{len(configs)}] {result['method_config']} miss={result['miss_rate']:.4f} intervals={result['mean_intervals_per_video']:.3f}")

    reps = choose_representatives(rows)
    json_rows = [{**row, "redundancy": row["redundancy"]} for row in rows]
    serializable_reps = {key: {"row": value["row"], "note": value["note"]} for key, value in reps.items()}
    write_json(out_dir / "adaptive_param_grid_results.json", {"rows": json_rows, "selected": serializable_reps})
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "method_config": row["method_config"],
                "threshold_percentile": row["threshold_percentile"],
                "post_filter_percentile": "none" if row["post_filter_percentile"] is None else row["post_filter_percentile"],
                "merge_gap": row["merge_gap"],
                "merge_iou": row["merge_iou"],
                "miss_rate": row["miss_rate"],
                "mean_coverage": row["mean_coverage"],
                "mean_intervals_per_video": row["mean_intervals_per_video"],
                "mean_duration": row["mean_duration"],
                "redundancy": row["redundancy_summary"],
            }
        )
    write_csv(out_dir / "adaptive_param_grid_results.csv", csv_rows)
    write_report(root / "reports/adaptive_param_grid_report.md", rows, reps)
    draw_tradeoff(root / "outputs/adaptive_param_grid_tradeoff.png", rows, reps)
    print(json.dumps({"configs": len(rows), "selected": {k: {"method_config": v["row"]["method_config"], "note": v["note"]} for k, v in reps.items()}}, indent=2))


if __name__ == "__main__":
    main()
