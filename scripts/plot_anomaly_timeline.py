import argparse
import json
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
    safe_name,
    score_metadata,
    score_path_for_video,
)
from src.score_filter import find_extreme_intervals  # noqa: E402


DEFAULT_CASES = [
    ("XD-Violence", "v=38GQ9L2meyE__#1_label_B6-0-0"),
    ("XD-Violence", "v=uQY15O3LKI0__#1_label_B6-0-0"),
    ("UCF-Crime", "Assault010_x264"),
]


def plot_case(dataset: str, video_id: str) -> dict:
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

    str_scores = {str(k): v for k, v in scores.items()}
    best_s, best_e, best_avg, *_ = find_extreme_intervals(str_scores)
    selected = [{"start": best_s, "end": best_e}]
    miss_count = sum(coverage_by_intervals(i, selected) < 0.1 for i in meta["intervals"])
    smeta = score_metadata(scores)
    video_len = estimate_video_length(smeta, meta["intervals"])

    frames = list(scores.keys())
    vals = list(scores.values())
    out = root / "outputs/timeline_plots" / dataset / f"{safe_name(video_id)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2]})
    axes[0].set_ylabel("GT")
    axes[0].set_yticks([])
    for interval in meta["intervals"]:
        axes[0].broken_barh([(interval["start"], interval["end"] - interval["start"])], (0, 1), facecolors="#d95f02")

    axes[1].set_ylabel("Wmax")
    axes[1].set_yticks([])
    axes[1].broken_barh([(best_s, best_e - best_s)], (0, 1), facecolors="#1b9e77")

    axes[2].plot(frames, vals, color="#386cb0", linewidth=1.4)
    axes[2].set_ylabel("score")
    axes[2].set_xlabel("frame")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlim(0, max(video_len, max(frames)))
    fig.suptitle(
        f"{dataset} | {video_id} | {meta['label']} | len={video_len} frames | "
        f"events={len(meta['intervals'])} | misses={miss_count} | Wmax_avg={best_avg:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"dataset": dataset, "video_id": video_id, "status": "ok", "output": str(out.relative_to(root))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", help="dataset::video_id")
    args = parser.parse_args()
    cases = []
    if args.case:
        for item in args.case:
            dataset, video_id = item.split("::", 1)
            cases.append((dataset, video_id))
    else:
        cases = DEFAULT_CASES
    results = [plot_case(dataset, video_id) for dataset, video_id in cases]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
