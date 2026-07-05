import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.score_filter import find_extreme_intervals


def parse_xd_annotations(path: Path):
    rows = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        name, label = parts[0], parts[1]
        nums = [int(x) for x in parts[2:] if re.fullmatch(r"-?\d+", x)]
        intervals = []
        for s, e in zip(nums[0::2], nums[1::2]):
            if s == -1 or e == -1:
                break
            intervals.append((s, e))
        rows[name] = {"label": label, "intervals": intervals}
    return rows


def parse_ucf_annotations(path: Path):
    rows = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        name, label = parts[0], parts[1]
        nums = [int(x) for x in parts[2:] if re.fullmatch(r"-?\d+", x)]
        intervals = []
        for s, e in zip(nums[0::2], nums[1::2]):
            if s == -1 or e == -1:
                break
            intervals.append((s, e))
        rows[name] = {"label": label, "intervals": intervals}
    return rows


def coverage_fraction(interval, selected):
    s, e = interval
    ss, se = selected
    overlap = max(0, min(e, se) - max(s, ss))
    denom = max(1, e - s)
    return overlap / denom


def analyze_dataset(name: str, annotations: dict, scores_dir: Path, min_cover: float):
    cases = []
    missing_scores = 0
    for video, meta in annotations.items():
        intervals = meta["intervals"]
        if len(intervals) < 2:
            continue
        score_path = scores_dir / f"{video}.json"
        if not score_path.exists():
            missing_scores += 1
            continue
        scores = json.loads(score_path.read_text(encoding="utf-8"))
        if not scores:
            continue
        best_s, best_e, best_avg, worst_s, worst_e, worst_avg = find_extreme_intervals(scores)
        selected = (best_s, best_e)
        covered = [coverage_fraction(interval, selected) >= min_cover for interval in intervals]
        cases.append(
            {
                "dataset": name,
                "video": video,
                "label": meta["label"],
                "length_frames": max(int(k) for k in scores.keys()) + 16,
                "window_size_frames": max(max(int(k) for k in scores.keys()) // 10, 300),
                "selected_window": [best_s, best_e],
                "selected_avg": round(best_avg, 4),
                "intervals": [list(x) for x in intervals],
                "covered": covered,
                "missed_count": covered.count(False),
                "total_intervals": len(intervals),
            }
        )

    case_count = len(cases)
    interval_total = sum(c["total_intervals"] for c in cases)
    missed_total = sum(c["missed_count"] for c in cases)
    any_missed = sum(1 for c in cases if c["missed_count"] > 0)
    all_covered = sum(1 for c in cases if c["missed_count"] == 0)
    label_counter = Counter(c["label"] for c in cases if c["missed_count"] > 0)
    lengths = [c["length_frames"] for c in cases]
    windows = [c["window_size_frames"] for c in cases]
    return {
        "dataset": name,
        "score_dir": str(scores_dir),
        "multi_anomaly_cases_with_scores": case_count,
        "multi_anomaly_cases_missing_scores": missing_scores,
        "length_frames_min": min(lengths) if lengths else 0,
        "length_frames_max": max(lengths) if lengths else 0,
        "length_frames_avg": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "window_frames_min": min(windows) if windows else 0,
        "window_frames_max": max(windows) if windows else 0,
        "window_frames_avg": round(sum(windows) / len(windows), 1) if windows else 0,
        "total_annotated_intervals": interval_total,
        "missed_intervals": missed_total,
        "missed_interval_rate": round(missed_total / interval_total, 4) if interval_total else 0,
        "videos_with_any_miss": any_missed,
        "video_any_miss_rate": round(any_missed / case_count, 4) if case_count else 0,
        "videos_all_intervals_covered": all_covered,
        "missed_by_label": dict(label_counter.most_common()),
        "top_examples": sorted(cases, key=lambda c: (-c["missed_count"], -c["length_frames"]))[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/multi_anomaly_miss_analysis.json"))
    parser.add_argument("--min_cover", type=float, default=0.1)
    args = parser.parse_args()

    repo = Path.cwd()
    results = []

    xd_ann = parse_xd_annotations(
        repo / "data/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt"
    )
    results.append(
        analyze_dataset(
            "XD-Violence",
            xd_ann,
            repo / "data/xd_violence/refined_scores/videollama3",
            args.min_cover,
        )
    )

    ucf_ann = parse_ucf_annotations(
        repo / "data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    )
    results.append(
        analyze_dataset(
            "UCF-Crime",
            ucf_ann,
            repo / "data/ucf_crime/scores/videollama3",
            args.min_cover,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
