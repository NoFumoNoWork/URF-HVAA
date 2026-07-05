import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    parse_temporal_annotations,
    repo_root,
    score_jsons,
    write_json,
)


def dataset_inventory(dataset: str, cfg: dict, root: Path) -> dict:
    dataset_root = root / cfg["root"]
    annotation_path = root / cfg["annotation"]
    annotations = parse_temporal_annotations(annotation_path, dataset)
    annotation_videos = set(annotations)
    multi_videos = {k for k, v in annotations.items() if len(v["intervals"]) >= 2}

    score_dirs = []
    all_score_videos = set()
    for score_dir_rel in cfg["score_dirs"]:
        score_dir = root / score_dir_rel
        scores = score_jsons(score_dir)
        all_score_videos.update(scores)
        score_dirs.append(
            {
                "path": str(score_dir_rel),
                "exists": score_dir.exists(),
                "score_json_count": len(scores),
                "matched_annotation_videos": len(annotation_videos & set(scores)),
            }
        )

    missing_scores = sorted(annotation_videos - all_score_videos)
    videos_dir = dataset_root / "videos"
    frames_dir = dataset_root / "frames"
    captions_dir = dataset_root / "captions"

    can_do = []
    cannot_do = []
    if annotation_path.exists() and all_score_videos:
        can_do.extend(["temporal coverage analysis", "score curve analysis", "top-k interval evaluation"])
    if not videos_dir.exists() or not any(videos_dir.iterdir() if videos_dir.exists() else []):
        cannot_do.append("raw mp4 visual inspection")
    if not frames_dir.exists() or not any(frames_dir.iterdir() if frames_dir.exists() else []):
        cannot_do.append("frame-level visual verification for all videos")

    return {
        "dataset": dataset,
        "dataset_root": str(cfg["root"]),
        "annotation_path": str(cfg["annotation"]),
        "annotation_exists": annotation_path.exists(),
        "annotation_video_count": len(annotation_videos),
        "multi_anomaly_video_count": len(multi_videos),
        "score_dirs": score_dirs,
        "score_videos_any_dir": len(all_score_videos),
        "matched_annotation_videos_any_score": len(annotation_videos & all_score_videos),
        "missing_score_video_count": len(missing_scores),
        "missing_score_video_examples": missing_scores[:10],
        "videos_dir_exists": videos_dir.exists(),
        "frames_dir_exists": frames_dir.exists(),
        "captions_dir_exists": captions_dir.exists(),
        "videos_file_count": len(list(videos_dir.glob("*"))) if videos_dir.exists() else 0,
        "frames_video_dir_count": len([p for p in frames_dir.iterdir() if p.is_dir()]) if frames_dir.exists() else 0,
        "caption_file_count": len(list(captions_dir.rglob("*.json"))) if captions_dir.exists() else 0,
        "can_do": can_do,
        "cannot_do": cannot_do,
    }


def write_report(path: Path, rows: list[dict]) -> None:
    lines = ["# Data Inventory", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['dataset']}",
                "",
                f"- Annotation exists: {row['annotation_exists']} (`{row['annotation_path']}`)",
                f"- Annotation videos: {row['annotation_video_count']}",
                f"- Multi-anomaly videos: {row['multi_anomaly_video_count']}",
                f"- Videos dir exists: {row['videos_dir_exists']} ({row['videos_file_count']} entries)",
                f"- Frames dir exists: {row['frames_dir_exists']} ({row['frames_video_dir_count']} video frame dirs)",
                f"- Captions dir exists: {row['captions_dir_exists']} ({row['caption_file_count']} json files)",
                f"- Matched annotation videos with any score: {row['matched_annotation_videos_any_score']}",
                f"- Missing score videos: {row['missing_score_video_count']}",
                "",
                "Score dirs:",
            ]
        )
        for score_dir in row["score_dirs"]:
            lines.append(
                f"- `{score_dir['path']}`: exists={score_dir['exists']}, "
                f"json={score_dir['score_json_count']}, matched={score_dir['matched_annotation_videos']}"
            )
        lines.extend(["", "Can do:"])
        lines.extend([f"- {item}" for item in row["can_do"]] or ["- None confirmed"])
        lines.extend(["", "Cannot do:"])
        lines.extend([f"- {item}" for item in row["cannot_do"]] or ["- None confirmed"])
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    rows = [dataset_inventory(name, cfg, root) for name, cfg in DATASET_CONFIGS.items()]
    write_json(root / "outputs/data_inventory.json", rows)
    write_report(root / "reports/data_inventory.md", rows)
    print(json.dumps({"datasets": len(rows), "output": "outputs/data_inventory.json"}, indent=2))


if __name__ == "__main__":
    main()
