import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    distribution,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    score_metadata,
    score_path_for_video,
    write_csv,
    write_json,
)


def build_rows() -> list[dict]:
    root = repo_root()
    rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            smeta = score_metadata(scores)
            video_len = estimate_video_length(smeta, meta["intervals"])
            anomaly_count = len(meta["intervals"])
            for idx, interval in enumerate(meta["intervals"], start=1):
                duration = interval["end"] - interval["start"]
                rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "label_or_class": meta["label"],
                        "video_length_frame_est": video_len,
                        "video_length_sec_est": round(video_len / 30.0, 3) if video_len else None,
                        "anomaly_id": idx,
                        "anomaly_start": interval["start"],
                        "anomaly_end": interval["end"],
                        "anomaly_duration": duration,
                        "anomaly_duration_sec_est": round(duration / 30.0, 3),
                        "anomaly_count_in_video": anomaly_count,
                        "score_json_path": str(score_path.relative_to(root)) if score_path else "",
                        "score_point_count": smeta["score_point_count"],
                        "score_frame_min": smeta["score_frame_min"],
                        "score_frame_max": smeta["score_frame_max"],
                        "score_stride_est": smeta["score_stride_est"],
                    }
                )
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    by_video = {}
    duration_by_video = defaultdict(int)
    for row in rows:
        key = (row["dataset"], row["video_id"])
        by_video[key] = row
        duration_by_video[key] += row["anomaly_duration"]

    anomaly_counts = [row["anomaly_count_in_video"] for row in by_video.values()]
    video_lengths = [row["video_length_frame_est"] for row in by_video.values() if row["video_length_frame_est"]]
    durations = [row["anomaly_duration"] for row in rows]
    coverage = [
        duration_by_video[key] / row["video_length_frame_est"]
        for key, row in by_video.items()
        if row["video_length_frame_est"]
    ]
    dataset_counts = Counter(row["dataset"] for row in rows)
    multi_videos = sum(1 for row in by_video.values() if row["anomaly_count_in_video"] >= 2)

    lines = [
        "# Anomaly Event Index Summary",
        "",
        f"- Videos with anomalies: {len(by_video)}",
        f"- Multi-anomaly videos: {multi_videos}",
        f"- Anomaly events: {len(rows)}",
        f"- Events by dataset: {dict(dataset_counts)}",
        "",
        "## Distributions",
        "",
        f"- Video length frames: `{distribution(video_lengths)}`",
        f"- Anomaly count per video: `{distribution(anomaly_counts)}`",
        f"- Anomaly duration frames: `{distribution(durations)}`",
        f"- Anomaly coverage ratio: `{distribution(coverage)}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    rows = build_rows()
    write_json(root / "outputs/anomaly_event_index.json", rows)
    write_csv(root / "outputs/anomaly_event_index.csv", rows)
    write_summary(root / "reports/anomaly_event_index_summary.md", rows)
    print(json.dumps({"events": len(rows), "output": "outputs/anomaly_event_index.json"}, indent=2))


if __name__ == "__main__":
    main()
