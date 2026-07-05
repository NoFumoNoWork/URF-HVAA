import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import DATASET_CONFIGS, load_scores, parse_temporal_annotations, repo_root, score_path_for_video, write_json  # noqa: E402
from src.topk_score_filter import find_topk_intervals  # noqa: E402


def generate(k: int, nms_iou: float) -> dict:
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
            output[dataset][video_id] = find_topk_intervals(scores, k=k, nms_iou=nms_iou)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    parser.add_argument("--nms_iou", type=float, default=0.5)
    args = parser.parse_args()
    root = repo_root()
    summary = {}
    for k in args.ks:
        data = generate(k, args.nms_iou)
        path = root / "outputs/topk_intervals" / f"topk_k{k}.json"
        write_json(path, data)
        summary[f"k{k}"] = sum(len(videos) for videos in data.values())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
