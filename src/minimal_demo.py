import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.score_filter import find_extreme_intervals, group_stats


def make_frame(path: Path, frame_idx: int, abnormal_start: int, abnormal_end: int) -> None:
    is_abnormal = abnormal_start <= frame_idx <= abnormal_end
    bg = (35, 42, 52) if not is_abnormal else (95, 24, 32)
    accent = (80, 185, 120) if not is_abnormal else (235, 78, 70)
    img = Image.new("RGB", (320, 180), bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 60, 300, 125), outline=accent, width=4)
    draw.text((28, 24), f"frame {frame_idx:03d}", fill=(235, 235, 235))
    label = "normal motion" if not is_abnormal else "suspicious event"
    draw.text((28, 138), label, fill=(235, 235, 235))
    img.save(path, quality=90)


def build_synthetic_inputs(output_dir: Path, num_frames: int, frame_interval: int) -> tuple[Path, Path, Path]:
    demo_root = output_dir / "synthetic_dataset"
    frames_dir = demo_root / "frames" / "synthetic_demo"
    captions_dir = demo_root / "captions" / "mock_videollama3"
    scores_dir = demo_root / "scores" / "mock_videollama3"
    annotations_dir = demo_root / "annotations"

    for directory in (frames_dir, captions_dir, scores_dir, annotations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    abnormal_start, abnormal_end = 48, 95
    for frame_idx in range(num_frames):
        make_frame(frames_dir / f"{frame_idx + 1:06d}.jpg", frame_idx, abnormal_start, abnormal_end)

    captions = {}
    scores = {}
    for frame_idx in range(0, num_frames, frame_interval):
        if abnormal_start <= frame_idx <= abnormal_end:
            captions[str(frame_idx)] = "A person moves abruptly near a restricted area and the scene looks suspicious"
            score = 0.85
        elif abs(frame_idx - abnormal_start) <= frame_interval or abs(frame_idx - abnormal_end) <= frame_interval:
            captions[str(frame_idx)] = "People are moving normally but the scene is close to a suspicious moment"
            score = 0.45
        else:
            captions[str(frame_idx)] = "A quiet ordinary scene with no visible abnormal activity"
            score = 0.08
        scores[str(frame_idx)] = score

    (captions_dir / "synthetic_demo.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    (scores_dir / "synthetic_demo.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    (annotations_dir / "test.txt").write_text(f"synthetic_demo 0 {num_frames - 1} 1\n", encoding="utf-8")
    (annotations_dir / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text(
        f"synthetic_demo.mp4 1 {abnormal_start} {abnormal_end} -1 -1\n",
        encoding="utf-8",
    )
    return demo_root, captions_dir, scores_dir


def labels_for_demo(num_frames: int, abnormal_start: int, abnormal_end: int) -> np.ndarray:
    labels = np.zeros(num_frames, dtype=bool)
    labels[abnormal_start : abnormal_end + 1] = True
    return labels


def evaluate_scores(scores: dict[str, float], num_frames: int, frame_interval: int) -> dict[str, float]:
    ordered = [scores[key] for key in sorted(scores, key=lambda x: int(x))]
    smoothed = gaussian_filter1d(np.array(ordered, dtype=np.float32), sigma=1)
    frame_scores = np.repeat(smoothed, frame_interval)[:num_frames]
    labels = labels_for_demo(num_frames, 48, 95)

    fpr, tpr, thresholds = roc_curve(labels, frame_scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, frame_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "best_f1": float(np.max(f1_scores)),
        "optimal_threshold_roc": float(thresholds[np.argmax(tpr - fpr)]),
        "optimal_threshold_pr": float(pr_thresholds[np.argmax(f1_scores[:-1])]),
    }


def run(output_dir: Path, num_frames: int, frame_interval: int) -> Path:
    demo_root, captions_dir, scores_dir = build_synthetic_inputs(output_dir, num_frames, frame_interval)
    scores_path = scores_dir / "synthetic_demo.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))

    best_s, best_e, best_avg, worst_s, worst_e, worst_avg = find_extreme_intervals(scores)
    std, avg_high, avg_low, gap = group_stats(scores, best_avg, worst_avg)
    interval_summary = {
        "synthetic_demo": {
            "highest_interval": [best_s, best_e],
            "highest_avg_score": round(best_avg, 3),
            "lowest_interval": [worst_s, worst_e],
            "lowest_avg_score": round(worst_avg, 3),
            "std": round(std, 5),
            "avg_high_group": round(avg_high, 3),
            "avg_low_group": round(avg_low, 3),
            "gap_high_low": round(gap, 3),
        }
    }
    (scores_dir / "highest_lowest_intervals.json").write_text(
        json.dumps(interval_summary, indent=2),
        encoding="utf-8",
    )

    metrics = evaluate_scores(scores, num_frames, frame_interval)
    metrics_dir = scores_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for key, value in metrics.items():
        (metrics_dir / f"{key}.txt").write_text(f"{value}\n", encoding="utf-8")

    result = {
        "mode": "synthetic smoke demo, not paper reproduction",
        "dataset_root": str(demo_root),
        "captions_path": str(captions_dir / "synthetic_demo.json"),
        "scores_path": str(scores_path),
        "intervals_path": str(scores_dir / "highest_lowest_intervals.json"),
        "metrics": metrics,
        "sample_explanation": (
            "The demo assigns higher anomaly scores to frames 48-95, then uses the repository's "
            "score filtering logic to recover the suspicious interval."
        ),
    }
    result_path = output_dir / "minimal_demo_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal URF-HVAA smoke demo without model weights.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/minimal_demo"))
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--frame_interval", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = run(args.output_dir, args.num_frames, args.frame_interval)
    print(f"Wrote demo result to {path}")
