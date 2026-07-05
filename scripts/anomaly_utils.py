import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


FPS_EST = 30.0


DATASET_CONFIGS = {
    "XD-Violence": {
        "root": Path("data/xd_violence"),
        "annotation": Path("data/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt"),
        "score_dirs": [
            Path("data/xd_violence/refined_scores/videollama3"),
            Path("data/xd_violence/scores/videollama3"),
        ],
    },
    "UCF-Crime": {
        "root": Path("data/ucf_crime"),
        "annotation": Path("data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
        "score_dirs": [
            Path("data/ucf_crime/scores/videollama3"),
            Path("data/ucf_crime/refined_scores/videollama3"),
        ],
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_temporal_annotations(path: Path, dataset: str) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        video_id = parts[0]
        label = parts[1]
        nums = [int(x) for x in parts[2:] if re.fullmatch(r"-?\d+", x)]
        intervals = []
        for start, end in zip(nums[0::2], nums[1::2]):
            if start == -1 or end == -1:
                break
            if end > start:
                intervals.append({"start": start, "end": end})
        rows[video_id] = {
            "dataset": dataset,
            "video_id": video_id,
            "label": label,
            "intervals": intervals,
        }
    return rows


def score_jsons(score_dir: Path) -> dict[str, Path]:
    if not score_dir.exists():
        return {}
    excluded = {
        "highest_lowest_intervals.json",
        "suspicious_part_phrases.json",
        "context_prompt.txt",
        "format_prompt.txt",
    }
    return {
        path.stem: path
        for path in sorted(score_dir.glob("*.json"))
        if path.name not in excluded
    }


def first_existing_score_dir(score_dirs: list[Path]) -> Path | None:
    for score_dir in score_dirs:
        if score_dir.exists():
            return score_dir
    return None


def score_path_for_video(video_id: str, score_dirs: list[Path]) -> Path | None:
    for score_dir in score_dirs:
        candidate = score_dir / f"{video_id}.json"
        if candidate.exists():
            return candidate
    return None


def load_scores(path: Path | None) -> dict[int, float]:
    if path is None or not path.exists():
        return {}
    raw = read_json(path)
    scores = {}
    for key, value in raw.items():
        try:
            scores[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(scores.items()))


def score_metadata(scores: dict[int, float]) -> dict:
    if not scores:
        return {
            "score_point_count": 0,
            "score_frame_min": None,
            "score_frame_max": None,
            "score_stride_est": None,
            "video_length_frame_est": None,
        }
    frames = sorted(scores)
    diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
    stride = int(Counter(diffs).most_common(1)[0][0]) if diffs else 16
    return {
        "score_point_count": len(frames),
        "score_frame_min": frames[0],
        "score_frame_max": frames[-1],
        "score_stride_est": stride,
        "video_length_frame_est": frames[-1] + stride,
    }


def estimate_video_length(meta: dict, intervals: list[dict]) -> int:
    from_scores = meta.get("video_length_frame_est")
    from_ann = max((i["end"] for i in intervals), default=0)
    return int(max(from_scores or 0, from_ann))


def overlap_length(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def coverage_by_intervals(interval: dict, predicted: list[dict]) -> float:
    length = max(1, interval["end"] - interval["start"])
    overlaps = []
    for pred in predicted:
        overlaps.append((max(interval["start"], pred["start"]), min(interval["end"], pred["end"])))
    merged = merge_intervals([{"start": s, "end": e} for s, e in overlaps if e > s])
    covered = sum(i["end"] - i["start"] for i in merged)
    return covered / length


def merge_intervals(intervals: list[dict]) -> list[dict]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x["start"], x["end"]))
    merged = [dict(ordered[0])]
    for item in ordered[1:]:
        last = merged[-1]
        if item["start"] <= last["end"]:
            last["end"] = max(last["end"], item["end"])
        else:
            merged.append(dict(item))
    return merged


def score_values_in_interval(scores: dict[int, float], start: int, end: int) -> list[float]:
    return [value for frame, value in scores.items() if start <= frame < end]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile_rank(values: list[float], target: float | None) -> float | None:
    if target is None or not values:
        return None
    return sum(1 for v in values if v <= target) / len(values)


def distribution(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(float(v) for v in values)
    def pct(p):
        idx = min(len(ordered) - 1, max(0, math.ceil((p / 100) * len(ordered)) - 1))
        return ordered[idx]
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "p25": round(pct(25), 3),
        "median": round(pct(50), 3),
        "p75": round(pct(75), 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def all_dataset_annotations() -> dict[str, dict[str, dict]]:
    root = repo_root()
    result = {}
    for dataset, cfg in DATASET_CONFIGS.items():
        result[dataset] = parse_temporal_annotations(root / cfg["annotation"], dataset)
    return result


def bin_value(value: float, bins: list[tuple[float, float, str]]) -> str:
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return bins[-1][2] if bins else "unknown"
