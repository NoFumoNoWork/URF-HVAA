import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    bin_value,
    coverage_by_intervals,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    score_metadata,
    score_path_for_video,
    write_json,
)


KS = [1, 2, 3, 5, 10]
COUNT_BINS = [(0, 2, "1"), (2, 4, "2-3"), (4, 8, "4-7"), (8, 10**9, "8+")]
LENGTH_BINS = [(0, 1000, "<1k"), (1000, 3000, "1k-3k"), (3000, 6000, "3k-6k"), (6000, 10**9, "6k+")]


def load_topk(root: Path, k: int) -> dict:
    path = root / "outputs/topk_intervals" / f"topk_k{k}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def evaluate_k(k: int) -> dict:
    root = repo_root()
    preds = load_topk(root, k)
    rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            intervals = preds.get(dataset, {}).get(video_id, [])
            if not intervals:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            video_len = estimate_video_length(score_metadata(load_scores(score_path)), meta["intervals"])
            for interval in meta["intervals"]:
                cov = coverage_by_intervals(interval, intervals)
                rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "coverage": cov,
                        "missed": cov < 0.1,
                        "anomaly_count": len(meta["intervals"]),
                        "video_length": video_len,
                    }
                )
    total = len(rows)
    missed = sum(r["missed"] for r in rows)
    videos = defaultdict(list)
    for row in rows:
        videos[(row["dataset"], row["video_id"])].append(row)
    video_any_miss = sum(any(item["missed"] for item in items) for items in videos.values())

    def group_stats(key_fn):
        groups = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        return {
            key: {
                "segments": len(items),
                "segment_miss_rate": round(sum(i["missed"] for i in items) / len(items), 6),
                "mean_coverage": round(sum(i["coverage"] for i in items) / len(items), 6),
            }
            for key, items in sorted(groups.items())
        }

    return {
        "k": k,
        "segments": total,
        "covered_segments": total - missed,
        "missed_segments": missed,
        "segment_miss_rate": round(missed / total, 6) if total else None,
        "video_count": len(videos),
        "videos_with_any_miss": video_any_miss,
        "video_any_miss_rate": round(video_any_miss / len(videos), 6) if videos else None,
        "mean_coverage": round(sum(r["coverage"] for r in rows) / total, 6) if total else None,
        "coverage_by_dataset": group_stats(lambda r: r["dataset"]),
        "coverage_by_anomaly_count_bin": group_stats(lambda r: bin_value(r["anomaly_count"], COUNT_BINS)),
        "coverage_by_video_length_bin": group_stats(lambda r: bin_value(r["video_length"], LENGTH_BINS)),
    }


def write_report(path: Path, results: list[dict]) -> None:
    lines = ["# Top-K Coverage Report", ""]
    lines.append("| K | Segments | Missed | Segment miss rate | Video any miss rate | Mean coverage |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['k']} | {r['segments']} | {r['missed_segments']} | "
            f"{r['segment_miss_rate']:.2%} | {r['video_any_miss_rate']:.2%} | {r['mean_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If miss rate falls sharply as K grows, the single-window output structure is the main bottleneck.",
            "- If miss rate stays high, the score curve, window scale, or event proposal method also needs improvement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_curve(path: Path, results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["k"] for r in results], [r["segment_miss_rate"] for r in results], marker="o", label="segment miss rate")
    ax.plot([r["k"] for r in results], [r["video_any_miss_rate"] for r in results], marker="s", label="video any miss rate")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = repo_root()
    results = [evaluate_k(k) for k in KS]
    write_json(root / "outputs/topk_coverage_results.json", results)
    write_report(root / "reports/topk_coverage_report.md", results)
    write_curve(root / "outputs/topk_coverage_curve.png", results)
    print(json.dumps({"k": KS, "output": "outputs/topk_coverage_results.json"}, indent=2))


if __name__ == "__main__":
    main()
