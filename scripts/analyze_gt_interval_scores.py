import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    safe_name,
    score_metadata,
    score_path_for_video,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/26-07-07-14-43-gt-score-window-curves")
WINDOWS = [300, 100, 30]


def rounded(value: float | None) -> float | None:
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 6)


def interval_values(scores: dict[int, float], start: int, end: int) -> list[tuple[int, float]]:
    return [(frame, value) for frame, value in scores.items() if start <= frame < end]


def score_stats(values: list[float]) -> dict:
    if not values:
        return {
            "score_point_count": 0,
            "mean_score": None,
            "variance_score": None,
            "max_score": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "score_point_count": int(len(arr)),
        "mean_score": rounded(float(np.mean(arr))),
        "variance_score": rounded(float(np.var(arr))),
        "max_score": rounded(float(np.max(arr))),
    }


def centered_sliding_average(frames: list[int], values: list[float], window_frames: int) -> list[float]:
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if not frames:
        return []
    frame_arr = np.asarray(frames, dtype=int)
    value_arr = np.asarray(values, dtype=float)
    prefix = np.concatenate([[0.0], np.cumsum(value_arr)])
    half = window_frames / 2.0
    result = []
    for frame in frame_arr:
        left = int(math.ceil(frame - half))
        right = int(math.floor(frame + half))
        lo = int(np.searchsorted(frame_arr, left, side="left"))
        hi = int(np.searchsorted(frame_arr, right, side="right"))
        if hi <= lo:
            result.append(float(value_arr[int(np.searchsorted(frame_arr, frame))]))
        else:
            result.append(float((prefix[hi] - prefix[lo]) / (hi - lo)))
    return result


def collect_stats(output_root: Path) -> tuple[list[dict], list[dict], dict]:
    root = repo_root()
    rows = []
    video_rows = []
    summary = {
        "datasets": {},
        "videos_with_gt": 0,
        "videos_with_scores": 0,
        "gt_intervals": 0,
        "gt_intervals_with_scores": 0,
        "gt_intervals_without_scores": 0,
    }
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        dataset_summary = {
            "videos_with_gt": 0,
            "videos_with_scores": 0,
            "gt_intervals": 0,
            "gt_intervals_with_scores": 0,
            "gt_intervals_without_scores": 0,
        }
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            dataset_summary["videos_with_gt"] += 1
            summary["videos_with_gt"] += 1
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            has_scores = bool(scores)
            if has_scores:
                dataset_summary["videos_with_scores"] += 1
                summary["videos_with_scores"] += 1
            smeta = score_metadata(scores)
            video_len = estimate_video_length(smeta, meta["intervals"])
            video_rows.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "label": meta["label"],
                    "score_json_path": str(score_path.relative_to(root)) if score_path else "",
                    "gt_interval_count": len(meta["intervals"]),
                    "has_scores": has_scores,
                    "score_point_count": smeta["score_point_count"],
                    "score_stride_est": smeta["score_stride_est"],
                    "video_length_frame_est": video_len,
                }
            )
            for anomaly_id, interval in enumerate(meta["intervals"], start=1):
                dataset_summary["gt_intervals"] += 1
                summary["gt_intervals"] += 1
                pairs = interval_values(scores, interval["start"], interval["end"]) if scores else []
                values = [value for _, value in pairs]
                stats = score_stats(values)
                if values:
                    dataset_summary["gt_intervals_with_scores"] += 1
                    summary["gt_intervals_with_scores"] += 1
                else:
                    dataset_summary["gt_intervals_without_scores"] += 1
                    summary["gt_intervals_without_scores"] += 1
                rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "label": meta["label"],
                        "anomaly_id": anomaly_id,
                        "gt_start": interval["start"],
                        "gt_end": interval["end"],
                        "gt_duration": interval["end"] - interval["start"],
                        "score_json_path": str(score_path.relative_to(root)) if score_path else "",
                        "first_score_frame_in_gt": pairs[0][0] if pairs else "",
                        "last_score_frame_in_gt": pairs[-1][0] if pairs else "",
                        **stats,
                    }
                )
        summary["datasets"][dataset] = dataset_summary
    write_csv(output_root / "outputs/gt_interval_score_stats.csv", rows)
    write_csv(output_root / "outputs/video_score_curve_inventory.csv", video_rows)
    write_json(output_root / "outputs/gt_interval_score_stats.json", {"summary": summary, "rows": rows, "videos": video_rows})
    return rows, video_rows, summary


def plot_video(dataset: str, video_id: str, meta: dict, scores: dict[int, float], score_path: Path | None, output_root: Path) -> dict:
    frames = list(scores.keys())
    values = list(scores.values())
    smeta = score_metadata(scores)
    video_len = estimate_video_length(smeta, meta["intervals"])
    curves = {
        "raw": values,
        "avg_300F": centered_sliding_average(frames, values, 300),
        "avg_100F": centered_sliding_average(frames, values, 100),
        "avg_30F": centered_sliding_average(frames, values, 30),
    }
    fig, ax = plt.subplots(figsize=(18, 7))
    for interval in meta["intervals"]:
        ax.axvspan(interval["start"], interval["end"], color="#d95f02", alpha=0.12)
    ax.plot(frames, curves["avg_300F"], color="#1b9e77", linewidth=2.0, alpha=0.92, label="300F sliding avg", zorder=2)
    ax.plot(frames, curves["avg_100F"], color="#e7298a", linewidth=1.7, alpha=0.9, label="100F sliding avg", zorder=3)
    ax.plot(frames, curves["avg_30F"], color="#6a3d9a", linewidth=1.25, alpha=0.78, linestyle="--", label="30F sliding avg", zorder=4)
    ax.plot(frames, curves["raw"], color="#8ecae6", linewidth=1.05, alpha=0.95, label="raw", zorder=5)
    ax.set_title(f"{dataset} | {video_id} | {meta['label']} | GT intervals={len(meta['intervals'])} | len={video_len} frames", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("anomaly score")
    ax.set_xlim(0, max(video_len, max(frames)))
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = output_root / "outputs/score_curve_plots" / dataset / f"{safe_name(video_id)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    multi_out = ""
    if len(meta["intervals"]) >= 2:
        multi_path = output_root / "outputs/multi_gt_score_curve_plots" / dataset / f"{safe_name(video_id)}.png"
        multi_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(multi_path, dpi=160)
        multi_out = str(multi_path.relative_to(repo_root()))
    plt.close(fig)
    return {
        "dataset": dataset,
        "video_id": video_id,
        "label": meta["label"],
        "status": "ok",
        "gt_interval_count": len(meta["intervals"]),
        "score_json_path": str(score_path.relative_to(repo_root())) if score_path else "",
        "output": str(out.relative_to(repo_root())),
        "multi_gt_output": multi_out,
    }


def plot_all(output_root: Path, max_plots: int | None = None) -> list[dict]:
    root = repo_root()
    results = []
    made = 0
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            if max_plots is not None and made >= max_plots:
                break
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                results.append({"dataset": dataset, "video_id": video_id, "status": "missing score"})
                continue
            results.append(plot_video(dataset, video_id, meta, scores, score_path, output_root))
            made += 1
    write_json(output_root / "outputs/score_curve_plot_manifest.json", results)
    return results


def write_report(output_root: Path, summary: dict, plot_results: list[dict], args: argparse.Namespace) -> None:
    ok_plots = sum(1 for item in plot_results if item["status"] == "ok")
    missing_plots = sum(1 for item in plot_results if item["status"] != "ok")
    single_gt_plots = sum(1 for item in plot_results if item["status"] == "ok" and int(item.get("gt_interval_count", 0)) == 1)
    multi_gt_plots = sum(1 for item in plot_results if item["status"] == "ok" and int(item.get("gt_interval_count", 0)) >= 2)
    lines = [
        "# GT Interval Score Statistics And Window Curves",
        "",
        "## Method",
        "",
        "- For each manually annotated abnormal interval, score points satisfying `gt_start <= frame < gt_end` are collected.",
        "- Statistics use the raw anomaly score values inside each GT interval.",
        "- Variance is population variance, computed with `numpy.var`.",
        "- Sliding-average curves are centered frame-window means over 300F, 100F, and 30F windows, plotted together with the raw curve.",
        "- Curves are drawn in this order: 300F, 100F, 30F, then raw. The raw curve is light blue and placed on top so it remains visible even when close to the 30F curve.",
        "- GT intervals are shown as light orange background spans in each plot.",
        "",
        "## Outputs",
        "",
        "- `outputs/gt_interval_score_stats.csv`",
        "- `outputs/gt_interval_score_stats.json`",
        "- `outputs/video_score_curve_inventory.csv`",
        "- `outputs/score_curve_plot_manifest.json`",
        "- `outputs/score_curve_plots/`",
        "- `outputs/multi_gt_score_curve_plots/`",
        "",
        "## Summary",
        "",
        f"- videos with GT abnormal intervals: {summary['videos_with_gt']}",
        f"- videos with both GT and scores: {summary['videos_with_scores']}",
        f"- GT abnormal intervals: {summary['gt_intervals']}",
        f"- GT intervals with score points: {summary['gt_intervals_with_scores']}",
        f"- GT intervals without score points: {summary['gt_intervals_without_scores']}",
        f"- plots generated: {ok_plots}",
        f"- single-GT plots: {single_gt_plots}",
        f"- multi-GT plots: {multi_gt_plots}",
        f"- plot misses: {missing_plots}",
        f"- max_plots argument: {args.max_plots if args.max_plots is not None else 'all'}",
        "",
        "## Dataset Breakdown",
        "",
    ]
    for dataset, item in summary["datasets"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- videos with GT: {item['videos_with_gt']}",
                f"- videos with GT and scores: {item['videos_with_scores']}",
                f"- GT intervals: {item['gt_intervals']}",
                f"- GT intervals with scores: {item['gt_intervals_with_scores']}",
                f"- GT intervals without scores: {item['gt_intervals_without_scores']}",
                "",
            ]
        )
    (output_root / "gt-score-window-curves_report.md").write_text("\n".join(lines), encoding="utf-8")


def copy_program(output_root: Path) -> None:
    root = repo_root()
    dst = output_root / "programs/scripts/analyze_gt_interval_scores.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "scripts/analyze_gt_interval_scores.py", dst)
    manifest = [
        "# gt-score-window-curves",
        "",
        f"- archive_folder: `{output_root.relative_to(root)}`",
        "- primary_report: `gt-score-window-curves_report.md`",
        "",
        "## Contents",
        "",
        "- `programs/`: copied scripts needed to reproduce or inspect this experiment.",
        "- `outputs/`: CSV/JSON score statistics and score-curve plots.",
        "- `gt-score-window-curves_report.md`: experiment summary.",
    ]
    (output_root / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_plots", type=int, default=None, help="Limit plot count for quick checks. Default draws all scored GT videos.")
    args = parser.parse_args()

    root = repo_root()
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows, _, summary = collect_stats(output_root)
    plot_results = plot_all(output_root, max_plots=args.max_plots)
    write_report(output_root, summary, plot_results, args)
    copy_program(output_root)
    print(
        json.dumps(
            {
                "gt_interval_rows": len(rows),
                "plots_ok": sum(1 for item in plot_results if item["status"] == "ok"),
                "report": str((output_root / "gt-score-window-curves_report.md").relative_to(root)),
                "stats_csv": str((output_root / "outputs/gt_interval_score_stats.csv").relative_to(root)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
