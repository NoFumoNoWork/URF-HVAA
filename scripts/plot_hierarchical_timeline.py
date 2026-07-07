import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
    write_json,
)


DEFAULT_CASES = [
    ("XD-Violence", "v=38GQ9L2meyE__#1_label_B6-0-0"),
    ("XD-Violence", "v=uQY15O3LKI0__#1_label_B6-0-0"),
    ("UCF-Crime", "Assault010_x264"),
]


def load_predictions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_prediction(preds: dict, dataset: str, video_id: str) -> dict:
    if video_id in preds and preds[video_id].get("dataset") == dataset:
        return preds[video_id]
    key = f"{dataset}::{video_id}"
    if key in preds:
        return preds[key]
    for value in preds.values():
        if isinstance(value, dict) and value.get("dataset") == dataset and value.get("video_id") == video_id:
            return value
    return {"events": []}


def merged_intervals(pred: dict) -> list[dict]:
    return [
        {"start": event["merged_start"], "end": event["merged_end"], "event_id": event["event_id"]}
        for event in pred.get("events", [])
    ]


def micro_intervals(pred: dict) -> list[dict]:
    items = []
    for event in pred.get("events", []):
        for micro in event.get("micro_intervals", []):
            item = dict(micro)
            item["event_id"] = event["event_id"]
            items.append(item)
    return items


def gap_intervals(pred: dict) -> list[dict]:
    items = []
    for event in pred.get("events", []):
        for gap in event.get("gaps", []):
            item = dict(gap)
            item["event_id"] = event["event_id"]
            items.append(item)
    return items


def draw_bars(ax, intervals: list[dict], start_key: str, end_key: str, color: str, alpha: float = 0.85) -> None:
    for item in intervals:
        start = item[start_key]
        end = item[end_key]
        if end > start:
            ax.broken_barh([(start, end - start)], (0, 1), facecolors=color, alpha=alpha)


def plot_case(dataset: str, video_id: str, preds: dict, output_dir: Path, show_gaps: bool) -> dict:
    root = repo_root()
    cfg = DATASET_CONFIGS[dataset]
    annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
    meta = annotations.get(video_id)
    if not meta:
        return {"dataset": dataset, "video_id": video_id, "status": "missing annotation"}
    score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
    scores = load_scores(score_path)
    if not scores:
        return {"dataset": dataset, "video_id": video_id, "status": "missing score"}

    pred = video_prediction(preds, dataset, video_id)
    merged = merged_intervals(pred)
    micros = micro_intervals(pred)
    gaps = gap_intervals(pred)
    video_len = estimate_video_length(score_metadata(scores), meta["intervals"])

    row_count = 5 if show_gaps else 4
    ratios = [1, 1, 1, 1, 2] if show_gaps else [1, 1, 1, 2]
    fig, axes = plt.subplots(row_count, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": ratios})
    tracks = [
        ("GT", meta["intervals"], "start", "end", "#d95f02", 0.9),
        ("Merged", merged, "start", "end", "#1b9e77", 0.78),
        ("Micro", micros, "start", "end", "#7570b3", 0.6),
    ]
    if show_gaps:
        tracks.append(("Gaps", gaps, "gap_start", "gap_end", "#e7298a", 0.42))
    for ax, (label, intervals, start_key, end_key, color, alpha) in zip(axes[:-1], tracks):
        ax.set_ylabel(label)
        ax.set_yticks([])
        draw_bars(ax, intervals, start_key, end_key, color, alpha)

    axes[-1].plot(list(scores.keys()), list(scores.values()), color="#386cb0", linewidth=1.2)
    axes[-1].set_ylabel("score")
    axes[-1].set_xlabel("frame")
    axes[-1].grid(alpha=0.25)
    axes[-1].set_xlim(0, max(video_len, max(scores)))
    fig.suptitle(
        f"{dataset} | {video_id} | GT={len(meta['intervals'])} | events={len(merged)} | micro={len(micros)}",
        fontsize=10,
    )
    fig.tight_layout()
    out = output_dir / dataset / f"{safe_name(video_id)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"dataset": dataset, "video_id": video_id, "status": "ok", "output": str(out.relative_to(root))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/hierarchical_intervals/hierarchical_intervals.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/hierarchical_intervals/timeline_plots"))
    parser.add_argument("--case", action="append", help="dataset::video_id")
    parser.add_argument("--hide_gaps", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    preds = load_predictions(root / args.input)
    if args.case:
        cases = [tuple(item.split("::", 1)) for item in args.case]
    else:
        cases = DEFAULT_CASES
    results = [plot_case(dataset, video_id, preds, root / args.output_dir, not args.hide_gaps) for dataset, video_id in cases]
    write_json(root / "outputs/hierarchical_intervals/hierarchical_timeline_plots.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
