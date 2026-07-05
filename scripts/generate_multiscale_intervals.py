import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    coverage_by_intervals,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    score_path_for_video,
    write_json,
)
from src.topk_score_filter import find_multiscale_intervals  # noqa: E402


def generate() -> dict:
    root = repo_root()
    output = {}
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        output[dataset] = {}
        for video_id in annotations:
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            output[dataset][video_id] = find_multiscale_intervals(scores, final_k=10)
    return output


def evaluate(preds: dict) -> dict:
    root = repo_root()
    rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            intervals = preds.get(dataset, {}).get(video_id, [])
            if not intervals:
                continue
            for interval in meta["intervals"]:
                cov = coverage_by_intervals(interval, intervals)
                rows.append({"dataset": dataset, "video_id": video_id, "coverage": cov, "missed": cov < 0.1})
    videos = defaultdict(list)
    for row in rows:
        videos[(row["dataset"], row["video_id"])].append(row)
    total = len(rows)
    missed = sum(r["missed"] for r in rows)
    return {
        "segments": total,
        "covered_segments": total - missed,
        "missed_segments": missed,
        "segment_miss_rate": round(missed / total, 6) if total else None,
        "mean_coverage": round(sum(r["coverage"] for r in rows) / total, 6) if total else None,
        "video_count": len(videos),
        "videos_with_any_miss": sum(any(item["missed"] for item in items) for items in videos.values()),
    }


def write_report(path: Path, result: dict, topk_results: list[dict] | None) -> None:
    lines = ["# Multiscale Coverage Report", ""]
    lines.extend(
        [
            "## Multiscale Top-K",
            "",
            f"- Segments: {result['segments']}",
            f"- Missed segments: {result['missed_segments']}",
            f"- Segment miss rate: {result['segment_miss_rate']:.2%}",
            f"- Mean coverage: {result['mean_coverage']:.3f}",
            "",
        ]
    )
    if topk_results:
        k1 = next((r for r in topk_results if r["k"] == 1), None)
        k10 = next((r for r in topk_results if r["k"] == 10), None)
        if k1 and k10:
            lines.extend(
                [
                    "## Comparison",
                    "",
                    f"- Original-like Top-K K=1 miss rate: {k1['segment_miss_rate']:.2%}",
                    f"- Same-scale Top-K K=10 miss rate: {k10['segment_miss_rate']:.2%}",
                    f"- Multiscale final K=10 miss rate: {result['segment_miss_rate']:.2%}",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_curve(path: Path, result: dict, topk_results: list[dict] | None) -> None:
    labels = []
    values = []
    if topk_results:
        for k in [1, 3, 5, 10]:
            item = next((r for r in topk_results if r["k"] == k), None)
            if item:
                labels.append(f"topk{k}")
                values.append(item["segment_miss_rate"])
    labels.append("multiscale")
    values.append(result["segment_miss_rate"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#8da0cb"] * (len(labels) - 1) + ["#66c2a5"])
    ax.set_ylabel("segment miss rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = repo_root()
    preds = generate()
    write_json(root / "outputs/multiscale_intervals.json", preds)
    result = evaluate(preds)
    topk_path = root / "outputs/topk_coverage_results.json"
    topk_results = json.loads(topk_path.read_text(encoding="utf-8")) if topk_path.exists() else None
    write_report(root / "reports/multiscale_coverage_report.md", result, topk_results)
    write_curve(root / "outputs/multiscale_coverage_curve.png", result, topk_results)
    print(json.dumps({"output": "outputs/multiscale_intervals.json", "coverage": result}, indent=2))


if __name__ == "__main__":
    main()
