import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_CONFIGS = [
    {
        "dataset": "MSAD",
        "caption_dir": Path("MSAD/captions/video_llama3_json_results"),
        "score_dir": Path("MSAD/refined_scores/videollama3"),
        "gt_file": Path("MSAD/annotations/test.txt"),
    },
    {
        "dataset": "UBNormal",
        "caption_dir": Path("UBNormal/caption/video_llama3_json_results"),
        "score_dir": Path("UBNormal/refined_scores/videollama3"),
        "gt_file": Path("UBNormal/annotations/temporal.txt"),
    },
    {
        "dataset": "UCF-Crime",
        "caption_dir": Path("ucf_crime/captions/video_llama3_json_results"),
        "score_dir": Path("ucf_crime/refined_scores/videollama3"),
        "gt_file": Path("ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
    },
    {
        "dataset": "XD-Violence",
        "caption_dir": Path("xd_violence/captions/video_llama3_json_results"),
        "score_dir": Path("xd_violence/refined_scores/videollama3"),
        "gt_file": Path("xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt"),
    },
]

KEYWORD_GROUPS = {
    "conflict": ["fight", "fighting", "argue", "arguing", "quarrel", "confrontation", "struggle"],
    "assault": ["attack", "hit", "punch", "kick", "beat", "push", "shove"],
    "weapon": ["gun", "knife", "weapon", "shoot", "shooting"],
    "accident": ["crash", "collision", "accident", "hit by car"],
    "fire_explosion": ["fire", "smoke", "explosion", "burning"],
    "chase_escape": ["chase", "run away", "flee", "escape"],
    "fall_injury": ["fall", "falling", "injured", "collapse"],
}

MATCH_FIELDS = [
    "dataset", "video_id", "frame", "caption", "matched_keywords", "keyword_group",
    "anomaly_score", "gt_inside", "distance_to_gt_start", "distance_to_gt_end",
]

SUMMARY_FIELDS = [
    "keyword_group", "num_matches", "mean_score", "median_score",
    "high_score_ratio_0_5", "high_score_ratio_0_6", "gt_inside_ratio",
    "near_gt_boundary_ratio",
]


def normalize_stem(stem):
    return re.sub(r"\(\d+\)$", "", Path(str(stem)).stem).strip()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stringify_caption(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = ["caption", "text", "description", "summary", "response", "content"]
        parts = [str(value[key]) for key in preferred if key in value and value[key] is not None]
        return " ".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " ".join(stringify_caption(item) for item in value)
    return "" if value is None else str(value)


def load_captions(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = raw.items() if isinstance(raw, dict) else enumerate(raw) if isinstance(raw, list) else []
    rows = []
    for key, value in items:
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        caption = stringify_caption(value).strip()
        if caption:
            rows.append({"frame": frame, "caption": caption})
    return sorted(rows, key=lambda row: row["frame"])


def json_index(directory):
    result = {}
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.json")):
        result.setdefault(path.stem, path)
        result.setdefault(normalize_stem(path.stem), path)
    return result


def load_scores(path):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scores = {}
    for key, value in raw.items():
        try:
            scores[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(scores.items()))


def parse_gt_file(path):
    rows = defaultdict(list)
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 3:
            continue
        video_id = normalize_stem(parts[0])
        nums = []
        for item in parts[1:]:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", item):
                nums.append(int(float(item)))
        for start, end in zip(nums[0::2], nums[1::2]):
            if start == -1 or end == -1:
                break
            if end > start:
                rows[video_id].append((start, end))
    return rows


def nearest_score(scores, frame):
    if not scores:
        return ""
    if frame in scores:
        return scores[frame]
    keys = list(scores)
    nearest = min(keys, key=lambda key: abs(key - frame))
    return scores[nearest]


def gt_distances(intervals, frame):
    if not intervals:
        return False, "", ""
    inside = any(start <= frame <= end for start, end in intervals)
    start_distance = min(abs(frame - start) for start, _ in intervals)
    end_distance = min(abs(frame - end) for _, end in intervals)
    return inside, start_distance, end_distance


def compile_patterns():
    patterns = {}
    for group, words in KEYWORD_GROUPS.items():
        group_patterns = []
        for word in words:
            escaped = re.escape(word).replace(r"\ ", r"\s+")
            if re.fullmatch(r"[A-Za-z\s]+", word):
                pattern = re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE)
            else:
                pattern = re.compile(escaped, re.IGNORECASE)
            group_patterns.append((word, pattern))
        patterns[group] = group_patterns
    return patterns


def find_matches(caption, patterns):
    matches = {}
    for group, group_patterns in patterns.items():
        found = [word for word, pattern in group_patterns if pattern.search(caption)]
        if found:
            matches[group] = found
    return matches


def fmt(value, digits=6):
    if value == "" or value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value) or math.isinf(value):
        return ""
    return round(value, digits)


def ratio(values):
    return sum(1 for value in values if value) / len(values) if values else 0.0


def summarize(rows, boundary_window):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["keyword_group"]].append(row)
    out = []
    for group in KEYWORD_GROUPS:
        items = grouped.get(group, [])
        scores = [float(row["anomaly_score"]) for row in items if row["anomaly_score"] != ""]
        near_boundary = []
        for row in items:
            distances = [row["distance_to_gt_start"], row["distance_to_gt_end"]]
            nums = [float(value) for value in distances if value != ""]
            near_boundary.append(bool(nums) and min(nums) <= boundary_window)
        out.append({
            "keyword_group": group,
            "num_matches": len(items),
            "mean_score": fmt(float(np.mean(scores)) if scores else ""),
            "median_score": fmt(float(np.median(scores)) if scores else ""),
            "high_score_ratio_0_5": fmt(ratio([score >= 0.5 for score in scores])),
            "high_score_ratio_0_6": fmt(ratio([score >= 0.6 for score in scores])),
            "gt_inside_ratio": fmt(ratio([str(row["gt_inside"]).lower() == "true" for row in items])),
            "near_gt_boundary_ratio": fmt(ratio(near_boundary)),
        })
    return out


def plot_outputs(summary_rows, match_rows, output_dir):
    groups = [row["keyword_group"] for row in summary_rows]
    score_groups = []
    for group in groups:
        scores = [float(row["anomaly_score"]) for row in match_rows if row["keyword_group"] == group and row["anomaly_score"] != ""]
        score_groups.append(scores)

    plt.figure(figsize=(10, 5))
    plt.boxplot(score_groups, labels=groups, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Anomaly score")
    plt.title("H1 keyword match score distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h1_score_distribution_by_keyword_group.png", dpi=180)
    plt.close()

    def bar_plot(field, ylabel, title, filename):
        values = [float(row[field]) if row[field] != "" else 0.0 for row in summary_rows]
        plt.figure(figsize=(9, 4.8))
        plt.bar(groups, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    bar_plot(
        "high_score_ratio_0_6",
        "Ratio",
        "High score ratio (score >= 0.6)",
        "fig_h1_high_score_ratio_by_keyword_group.png",
    )
    bar_plot(
        "gt_inside_ratio",
        "Ratio",
        "GT-inside ratio by keyword group",
        "fig_h1_gt_inside_ratio_by_keyword_group.png",
    )


def write_report(path, summary_rows, match_rows, warnings, boundary_window):
    total = len(match_rows)
    high_06 = ratio([float(row["anomaly_score"]) >= 0.6 for row in match_rows if row["anomaly_score"] != ""])
    gt_inside = ratio([str(row["gt_inside"]).lower() == "true" for row in match_rows])
    ranked = sorted(
        summary_rows,
        key=lambda row: (float(row["high_score_ratio_0_6"] or 0), int(row["num_matches"] or 0)),
        reverse=True,
    )
    top_groups = ", ".join(
        f"{row['keyword_group']}({row['high_score_ratio_0_6']})"
        for row in ranked[:3]
        if int(row["num_matches"] or 0) > 0
    )
    text = [
        "# H1 keyword-score evidence report",
        "",
        "## Scope",
        "",
        "This report scans available VLM captions, refined anomaly scores, and GT temporal annotations for explicit risk keywords related to conflict, assault, weapon, accident, fire/explosion, chase/escape, and fall/injury.",
        "",
        "## Main observations",
        "",
        f"- Total keyword-match rows: {total}.",
        f"- Overall high-score ratio at threshold 0.6: {fmt(high_06)}.",
        f"- Overall GT-inside ratio: {fmt(gt_inside)}.",
        f"- Strongest high-score keyword groups by ratio: {top_groups or 'n/a'}.",
        f"- Near-boundary ratio uses a ±{boundary_window}-frame window around GT starts/ends.",
        "",
        "## Interpretation",
        "",
        "When captions already contain explicit terms such as fight, attack, push, crash, fire, smoke, gun, knife, or falling, the anomaly score is often already elevated. This supports a cautious interpretation: in the current intermediate artifacts, much of the H1 signal looks like explicit abnormal semantics captured by the caption and LLM/score stage, rather than independent precursor recognition before the abnormal event is described.",
        "",
        "This does not mean H1 is unimportant. It means the current caption/score artifacts are not sufficient to prove a strict precursor mechanism. A stronger H1 test would need original videos and VLM labels that explicitly distinguish precursor, buildup, ongoing abnormal event, and aftermath states.",
        "",
        "## Warnings",
        "",
    ]
    text.extend([f"- {warning}" for warning in warnings] or ["- None."])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def run(data_root, output_dir, boundary_window):
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = compile_patterns()
    warnings = []
    match_rows = []

    for cfg in DATASET_CONFIGS:
        dataset = cfg["dataset"]
        caption_dir = data_root / cfg["caption_dir"]
        score_dir = data_root / cfg["score_dir"]
        gt_file = data_root / cfg["gt_file"]
        if not caption_dir.exists():
            warnings.append(f"{dataset}: missing caption_dir {caption_dir}")
            continue
        if not score_dir.exists():
            warnings.append(f"{dataset}: missing score_dir {score_dir}")
        if not gt_file.exists():
            warnings.append(f"{dataset}: missing gt_file {gt_file}")
        caption_index = json_index(caption_dir)
        score_index = json_index(score_dir)
        gt_rows = parse_gt_file(gt_file)
        for video_id, caption_path in sorted(caption_index.items()):
            if video_id != normalize_stem(video_id):
                continue
            captions = load_captions(caption_path)
            score_path = score_index.get(video_id)
            scores = load_scores(score_path) if score_path else {}
            if not score_path:
                warnings.append(f"{dataset}/{video_id}: missing matching score JSON")
            intervals = gt_rows.get(video_id, [])
            for item in captions:
                matches = find_matches(item["caption"], patterns)
                if not matches:
                    continue
                score = nearest_score(scores, item["frame"])
                inside, dist_start, dist_end = gt_distances(intervals, item["frame"])
                for group, words in matches.items():
                    match_rows.append({
                        "dataset": dataset,
                        "video_id": video_id,
                        "frame": item["frame"],
                        "caption": item["caption"],
                        "matched_keywords": ";".join(words),
                        "keyword_group": group,
                        "anomaly_score": fmt(score),
                        "gt_inside": inside,
                        "distance_to_gt_start": dist_start,
                        "distance_to_gt_end": dist_end,
                    })

    summary_rows = summarize(match_rows, boundary_window)
    write_csv(output_dir / "h1_keyword_score_matches.csv", match_rows, MATCH_FIELDS)
    write_csv(output_dir / "h1_keyword_score_summary.csv", summary_rows, SUMMARY_FIELDS)
    plot_outputs(summary_rows, match_rows, output_dir)
    write_report(output_dir / "h1_keyword_score_report.md", summary_rows, match_rows, warnings, boundary_window)
    (output_dir / "h1_keyword_score_summary.json").write_text(
        json.dumps({"num_matches": len(match_rows), "warnings": warnings}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-window", type=int, default=64)
    args = parser.parse_args()
    run(Path(args.data_root), Path(args.output_dir), args.boundary_window)


if __name__ == "__main__":
    main()
