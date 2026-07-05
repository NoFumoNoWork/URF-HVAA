import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import DATASET_CONFIGS, load_scores, parse_temporal_annotations, repo_root, score_path_for_video, write_json  # noqa: E402
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


def output_name(threshold_percentile: float, merge_gap: int, merge_iou: float) -> str:
    pct = str(int(threshold_percentile)) if threshold_percentile == int(threshold_percentile) else str(threshold_percentile).replace(".", "p")
    iou = str(merge_iou).replace(".", "")
    return f"adaptive_intervals_p{pct}_gap{merge_gap}_iou{iou}.json"


def generate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    root = repo_root()
    all_results = {}
    summary_rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        all_results[dataset] = {}
        for video_id in sorted(annotations):
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            intervals, metadata = select_adaptive_intervals(
                scores,
                threshold_percentile=args.threshold_percentile,
                merge_iou=args.merge_iou,
                merge_gap=args.merge_gap,
                min_duration=args.min_duration,
                post_filter_percentile=None if args.disable_post_filter else args.post_filter_percentile,
            )
            all_results[dataset][video_id] = {
                "intervals": intervals,
                "metadata": metadata,
            }
            durations = [item["duration"] for item in intervals]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "score_json_path": str(score_path.relative_to(root)) if score_path else "",
                    "candidate_count": metadata["candidate_count"],
                    "retained_candidate_count": metadata["retained_candidate_count"],
                    "merged_interval_count": metadata["merged_interval_count"],
                    "final_interval_count": metadata["final_interval_count"],
                    "window_sizes": "|".join(str(x) for x in metadata["window_sizes"]),
                    "percentile_threshold": metadata["percentile_threshold"],
                    "post_filter_threshold": metadata["post_filter_threshold"],
                    "mean_interval_duration": round(sum(durations) / len(durations), 3) if durations else 0,
                    "max_interval_duration": max(durations) if durations else 0,
                }
            )
    return all_results, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold_percentile", type=float, default=85.0)
    parser.add_argument("--merge_iou", type=float, default=0.3)
    parser.add_argument("--merge_gap", type=int, default=150)
    parser.add_argument("--min_duration", type=int, default=60)
    parser.add_argument("--post_filter_percentile", type=float, default=75.0)
    parser.add_argument("--disable_post_filter", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    results, rows = generate(args)
    out_dir = root / "outputs/adaptive_intervals"
    json_path = out_dir / output_name(args.threshold_percentile, args.merge_gap, args.merge_iou)
    write_json(json_path, results)
    write_csv(out_dir / "adaptive_intervals_summary.csv", rows)
    print(json.dumps({"videos": sum(len(v) for v in results.values()), "output": str(json_path.relative_to(root))}, indent=2))


if __name__ == "__main__":
    main()
