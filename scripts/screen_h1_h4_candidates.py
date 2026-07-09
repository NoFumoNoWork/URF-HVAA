import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import load_scores, repo_root, safe_name, score_metadata  # noqa: E402


TASK_NAME = "h1-h4-candidate-screen"

PRECURSOR_KEYWORDS = [
    "arguing", "argument", "yelling", "screaming", "shouting",
    "running", "chasing", "panic", "distress", "threat",
    "confrontation", "aggressive", "struggle", "fleeing",
    "escape", "trapped", "anxious", "tense", "fear",
]

STRONG_ANOMALY_KEYWORDS = [
    "gun", "shooting", "fire", "explosion", "crash", "accident",
    "fight", "fighting", "attack", "blood", "knife", "weapon",
    "flames", "smoke", "burning", "injured", "dead",
]

SCENE_QUALITY_OR_UNCERTAINTY_KEYWORDS = [
    "dark", "dimly lit", "unclear", "blurry", "occluded",
    "hard to see", "obscured", "low light", "smoke",
]

TRANSITION_KEYWORDS = [
    "scene cuts", "scene cut", "cuts to", "cut to",
    "scene transitions", "scene transition", "transitions to",
    "scene shifts", "scene shift", "shifts to",
    "camera angle changes", "angle changes",
    "camera pans", "camera zooms", "camera focuses",
    "the final frames", "video ends",
]


def rounded(value, digits=6):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return round(value, digits)


def mean(values):
    return float(np.mean(values)) if values else None


def std(values):
    return float(np.std(values)) if values else None


def normalize_stem(stem):
    return re.sub(r"\(\d+\)$", "", stem).strip()


def json_index(directory):
    result = {}
    if not directory or not directory.exists():
        return result
    for path in sorted(directory.glob("*.json")):
        result.setdefault(path.stem, path)
        result.setdefault(normalize_stem(path.stem), path)
    return result


def stringify_caption(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = ["caption", "text", "description", "summary", "response", "content"]
        parts = [str(value[k]) for k in preferred if k in value and value[k] is not None]
        if parts:
            return " ".join(parts)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " ".join(stringify_caption(item) for item in value)
    return "" if value is None else str(value)


def load_captions(path):
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    captions = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        return {}
    for key, value in items:
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        captions[frame] = stringify_caption(value)
    return dict(sorted(captions.items()))


def parse_gt_csv(path):
    rows = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            return rows
        fields = {name.lower().strip(): name for name in reader.fieldnames}
        video_col = fields.get("video_id") or fields.get("video") or fields.get("name")
        start_col = fields.get("gt_start") or fields.get("start")
        end_col = fields.get("gt_end") or fields.get("end")
        label_col = fields.get("label") or fields.get("class") or fields.get("category")
        if not video_col or not start_col or not end_col:
            return rows
        for row in reader:
            try:
                start = int(float(row[start_col]))
                end = int(float(row[end_col]))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            video_id = normalize_stem(Path(str(row[video_col]).strip()).stem)
            rows[video_id].append({"start": start, "end": end, "label": row.get(label_col, "") if label_col else ""})
    return rows


def parse_gt_lines(path):
    rows = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 3:
            continue
        video_id = normalize_stem(Path(parts[0]).stem)
        numeric = []
        for item in parts[1:]:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", item):
                numeric.append(int(float(item)))
        intervals = []
        for start, end in zip(numeric[0::2], numeric[1::2]):
            if start == -1 or end == -1:
                break
            if end > start:
                intervals.append({"start": start, "end": end, "label": ""})
        if intervals:
            rows[video_id].extend(intervals)
    return rows


def load_gt(path):
    if not path or not path.exists():
        return defaultdict(list)
    csv_rows = parse_gt_csv(path) if path.suffix.lower() == ".csv" else defaultdict(list)
    return csv_rows if csv_rows else parse_gt_lines(path)


def keyword_matches(text, keywords):
    low = (text or "").lower()
    return [kw for kw in keywords if kw in low]


def frames_in_range(mapping, start, end):
    return [(frame, mapping[frame]) for frame in sorted(mapping) if start <= frame < end]


def score_pairs_in_range(scores, start, end):
    return [(frame, value) for frame, value in sorted(scores.items()) if start <= frame < end]


def count_keywords(captions, keywords):
    matches = []
    for text in captions:
        matches.extend(keyword_matches(text, keywords))
    return len(matches), sorted(set(matches))


def relevant_captions(caption_pairs, scores, keywords, limit=3):
    ranked = []
    for frame, caption in caption_pairs:
        matches = keyword_matches(caption, keywords)
        score = scores.get(frame, -1.0)
        ranked.append((bool(matches), score, frame, caption))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return " || ".join(f"{frame}: {caption}" for _, _, frame, caption in ranked[:limit])


def consecutive_mid_run(score_pairs, caption_pairs_by_frame, args):
    run = 0
    best = 0
    saw_keyword = False
    for frame, value in score_pairs:
        caption = caption_pairs_by_frame.get(frame, "")
        has_keyword = bool(keyword_matches(caption, PRECURSOR_KEYWORDS))
        if args.mid_low <= value <= args.mid_high:
            run += 1
            saw_keyword = saw_keyword or has_keyword
            best = max(best, run)
        else:
            run = 0
    return best >= args.min_mid_run and saw_keyword


def confidence_from_types(types):
    if len(types) >= 2:
        return "high"
    if types:
        return "medium"
    return "low"


def screen_h1(video_id, gt_intervals, captions, scores, args):
    rows = []
    caption_by_frame = dict(captions)
    for interval in gt_intervals:
        gt_start = int(interval["start"])
        gt_end = int(interval["end"])
        early_start = max(0, gt_start - args.early_window_frames)
        early_end = gt_start
        early_scores = score_pairs_in_range(scores, early_start, early_end)
        inside_scores = score_pairs_in_range(scores, gt_start, gt_end)
        first_inside = inside_scores[: max(args.post_k, 1)]
        early_values = [value for _, value in early_scores]
        inside_values = [value for _, value in first_inside]
        early_mean = mean(early_values)
        early_max = max(early_values) if early_values else None
        early_min = min(early_values) if early_values else None
        early_std = std(early_values)
        inside_mean = mean(inside_values)
        score_jump = inside_mean - early_mean if inside_mean is not None and early_mean is not None else None
        high_frame = next((frame for frame, value in sorted(scores.items()) if frame >= gt_start and value >= args.high_score_threshold), None)
        time_to_high = high_frame - gt_start if high_frame is not None else ""
        early_caption_pairs = frames_in_range(captions, early_start, early_end)
        early_caption_texts = [text for _, text in early_caption_pairs]
        risk_count, risk_matches = count_keywords(early_caption_texts, PRECURSOR_KEYWORDS)
        strong_count, strong_matches = count_keywords(early_caption_texts, STRONG_ANOMALY_KEYWORDS)
        transition_count, transition_matches = count_keywords(early_caption_texts, TRANSITION_KEYWORDS)
        types = []
        if risk_count > 0 and early_max is not None and early_max <= args.low_score_threshold:
            types.append("H1_low_score_precursor")
        if (
            risk_count > 0
            and early_mean is not None
            and args.mid_low <= early_mean <= args.mid_high
            and score_jump is not None
            and score_jump >= 0.2
        ):
            types.append("H1_mid_score_precursor")
        if early_mean is not None and inside_mean is not None and early_mean <= args.low_score_threshold and inside_mean >= args.high_score_threshold:
            types.append("H1_sudden_jump")
        if consecutive_mid_run(early_scores, caption_by_frame, args) or (
            risk_count > 0 and score_jump is not None and score_jump >= 0.2 and len(early_scores) >= args.min_mid_run
        ):
            types.append("H1_temporal_accumulation")
        if not types:
            continue
        matched = sorted(set(risk_matches + strong_matches + transition_matches))
        rows.append(
            {
                "video_id": video_id,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "early_start": early_start,
                "early_end": early_end,
                "h1_type": ";".join(types),
                "early_mean_score": rounded(early_mean),
                "early_max_score": rounded(early_max),
                "early_min_score": rounded(early_min),
                "early_score_std": rounded(early_std),
                "mean_score_inside_gt_start": rounded(inside_mean),
                "score_jump_after_gt_start": rounded(score_jump),
                "time_to_high_score": time_to_high,
                "early_risk_keyword_count": risk_count,
                "early_strong_keyword_count": strong_count,
                "early_transition_keyword_count": transition_count,
                "num_windows": len(early_scores),
                "matched_keywords": ";".join(matched),
                "representative_captions": relevant_captions(early_caption_pairs, scores, PRECURSOR_KEYWORDS + STRONG_ANOMALY_KEYWORDS),
                "confidence": confidence_from_types(types),
            }
        )
    return rows


def near_gt(frame, intervals, margin):
    for item in intervals:
        if item["start"] <= frame <= item["end"]:
            return True
        if abs(frame - item["start"]) <= margin or abs(frame - item["end"]) <= margin:
            return True
    return False


def screen_h4(video_id, gt_intervals, captions, scores, args):
    rows = []
    frames = sorted(captions)
    for idx, frame in enumerate(frames):
        caption = captions[frame]
        transition_matches = keyword_matches(caption, TRANSITION_KEYWORDS)
        if not transition_matches:
            continue
        pre_frames = frames[max(0, idx - args.pre_k):idx]
        post_frames = frames[idx + 1:idx + 1 + args.post_k]
        pre_scores = [scores[f] for f in pre_frames if f in scores]
        post_scores = [scores[f] for f in post_frames if f in scores]
        pre_mean = mean(pre_scores)
        post_mean = mean(post_scores)
        score_drop = pre_mean - post_mean if pre_mean is not None and post_mean is not None else None
        pre_risk_count, _ = count_keywords([captions[f] for f in pre_frames], PRECURSOR_KEYWORDS + STRONG_ANOMALY_KEYWORDS)
        post_risk_count, _ = count_keywords([captions[f] for f in post_frames], PRECURSOR_KEYWORDS + STRONG_ANOMALY_KEYWORDS)
        is_near_gt = near_gt(frame, gt_intervals, args.near_gt_margin)
        types = []
        if (
            pre_mean is not None
            and post_mean is not None
            and score_drop is not None
            and pre_mean >= args.h4_high_score_threshold
            and post_mean <= args.h4_low_score_threshold
            and score_drop >= args.h4_drop_threshold
        ):
            types.append("H4_score_drop")
        if pre_risk_count > 0 and post_risk_count == 0 and score_drop is not None and score_drop >= 0.2:
            types.append("H4_risk_description_drop")
        if is_near_gt and score_drop is not None and score_drop >= args.h4_drop_threshold:
            types.append("H4_near_gt_drop")
        if not types:
            continue
        rows.append(
            {
                "video_id": video_id,
                "transition_frame": frame,
                "transition_time_sec": rounded(frame / args.fps, 3),
                "h4_type": ";".join(types),
                "pre_mean_score": rounded(pre_mean),
                "post_mean_score": rounded(post_mean),
                "score_drop": rounded(score_drop),
                "pre_risk_keyword_count": pre_risk_count,
                "post_risk_keyword_count": post_risk_count,
                "matched_transition_keywords": ";".join(sorted(set(transition_matches))),
                "pre_captions": " || ".join(f"{f}: {captions[f]}" for f in pre_frames),
                "transition_caption": caption,
                "post_captions": " || ".join(f"{f}: {captions[f]}" for f in post_frames),
                "whether_near_gt": is_near_gt,
                "confidence": confidence_from_types(types),
            }
        )
    return rows


def rolling_mean(values, width=5):
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    width = max(1, min(width, len(arr)))
    kernel = np.ones(width) / width
    return np.convolve(arr, kernel, mode="same").tolist()


def plot_video(video_id, scores, gt_intervals, h1_rows, h4_rows, output_path):
    if not scores:
        return
    frames = sorted(scores)
    values = [scores[f] for f in frames]
    fig, ax = plt.subplots(figsize=(16, 6))
    for interval in gt_intervals:
        ax.axvspan(interval["start"], interval["end"], color="#d95f02", alpha=0.16, label="GT interval")
    ax.plot(frames, values, color="#377eb8", linewidth=1.1, alpha=0.78, label="raw score")
    ax.plot(frames, rolling_mean(values), color="#1b9e77", linewidth=2.0, label="rolling mean")
    if h1_rows:
        h1_x = [int(row["gt_start"]) for row in h1_rows]
        h1_y = [scores.get(x, max(values) if values else 0.8) for x in h1_x]
        ax.scatter(h1_x, h1_y, marker="^", s=70, color="#e7298a", edgecolor="black", linewidth=0.4, label="H1 candidate", zorder=5)
        for row in h1_rows:
            ax.axvspan(int(row["early_start"]), int(row["early_end"]), color="#e7298a", alpha=0.07)
    if h4_rows:
        h4_x = [int(row["transition_frame"]) for row in h4_rows]
        h4_y = [scores.get(x, 0.05) for x in h4_x]
        ax.scatter(h4_x, h4_y, marker="v", s=70, color="#984ea3", edgecolor="black", linewidth=0.4, label="H4 candidate", zorder=5)
    ax.set_title(f"{video_id} | H1={len(h1_rows)} H4={len(h4_rows)}", fontsize=11)
    ax.set_xlabel("frame")
    ax.set_ylabel("anomaly score")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv_sig(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir, args, summary):
    report = output_dir / f"{TASK_NAME}_report.md"
    lines = [
        "# H1/H4 Candidate Screening Report",
        "",
        "## Inputs",
        "",
        f"- caption_dir: `{args.caption_dir}`",
        f"- score_dir: `{args.score_dir}`",
        f"- gt_file: `{args.gt_file}`",
        "",
        "## Outputs",
        "",
        "- `data/h1_candidates.csv`: GT-interval-level early-stage underestimation candidates.",
        "- `data/h4_candidates.csv`: transition-window-level context forgetting candidates.",
        "- `data/video_summary.csv`: per-video counts and score/caption summaries.",
        "- `plots/*.png`: score timeline diagnostics with GT intervals and candidate markers.",
        "- `figures/*.png`: mirrored plot files for the standard archive image directory.",
        "",
        "## Summary",
        "",
        f"- processed videos: {summary['processed_videos']}",
        f"- skipped videos: {summary['skipped_videos']}",
        f"- total H1 candidates: {summary['total_h1']}",
        f"- total H4 candidates: {summary['total_h4']}",
        f"- warnings: {len(summary['warnings'])}",
    ]
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in summary["warnings"][:50])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_video_ids(caption_files, score_files, gt_rows):
    return sorted(set(caption_files) | set(score_files) | set(gt_rows))


def process(args):
    caption_dir = Path(args.caption_dir)
    score_dir = Path(args.score_dir)
    gt_file = Path(args.gt_file)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{datetime.now():%y-%m-%d-%H-%M}-{TASK_NAME}"
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
    figure_dir = output_dir / "figures"
    script_dir = output_dir / "scripts"
    caption_files = json_index(caption_dir)
    score_files = json_index(score_dir)
    gt_rows = load_gt(gt_file)
    warnings = []
    h1_all = []
    h4_all = []
    video_rows = []
    processed = 0
    skipped = 0
    for video_id in build_video_ids(caption_files, score_files, gt_rows):
        caption_path = caption_files.get(video_id)
        score_path = score_files.get(video_id)
        if not caption_path or not score_path:
            skipped += 1
            warnings.append(f"{video_id}: missing {'caption' if not caption_path else 'score'} JSON")
            continue
        captions = load_captions(caption_path)
        scores = load_scores(score_path)
        if not captions or not scores:
            skipped += 1
            warnings.append(f"{video_id}: empty or unreadable caption/score JSON")
            continue
        intervals = gt_rows.get(video_id, [])
        h1_rows = screen_h1(video_id, intervals, captions, scores, args)
        h4_rows = screen_h4(video_id, intervals, captions, scores, args)
        h1_all.extend(h1_rows)
        h4_all.extend(h4_rows)
        score_values = list(scores.values())
        caption_texts = list(captions.values())
        risk_total, _ = count_keywords(caption_texts, PRECURSOR_KEYWORDS)
        strong_total, _ = count_keywords(caption_texts, STRONG_ANOMALY_KEYWORDS)
        transition_mentions, _ = count_keywords(caption_texts, TRANSITION_KEYWORDS)
        smeta = score_metadata(scores)
        video_rows.append(
            {
                "video_id": video_id,
                "num_gt_intervals": len(intervals),
                "num_caption_windows": len(captions),
                "score_mean": rounded(mean(score_values)),
                "score_max": rounded(max(score_values) if score_values else None),
                "score_min": rounded(min(score_values) if score_values else None),
                "score_stride_est": smeta["score_stride_est"],
                "num_h1_candidates": len(h1_rows),
                "num_h4_candidates": len(h4_rows),
                "num_transition_mentions": transition_mentions,
                "risk_keyword_total": risk_total,
                "strong_keyword_total": strong_total,
                "h1_candidate_types": ";".join(sorted({t for row in h1_rows for t in row["h1_type"].split(";")})),
                "h4_candidate_types": ";".join(sorted({t for row in h4_rows for t in row["h4_type"].split(";")})),
            }
        )
        if args.plot:
            plot_path = plot_dir / f"{safe_name(video_id)}.png"
            plot_video(video_id, scores, intervals, h1_rows, h4_rows, plot_path)
            figure_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plot_path, figure_dir / plot_path.name)
        processed += 1
    h1_fields = [
        "video_id", "gt_start", "gt_end", "early_start", "early_end", "h1_type",
        "early_mean_score", "early_max_score", "early_min_score", "early_score_std",
        "mean_score_inside_gt_start", "score_jump_after_gt_start", "time_to_high_score",
        "early_risk_keyword_count", "early_strong_keyword_count", "early_transition_keyword_count",
        "num_windows", "matched_keywords", "representative_captions", "confidence",
    ]
    h4_fields = [
        "video_id", "transition_frame", "transition_time_sec", "h4_type", "pre_mean_score",
        "post_mean_score", "score_drop", "pre_risk_keyword_count", "post_risk_keyword_count",
        "matched_transition_keywords", "pre_captions", "transition_caption", "post_captions",
        "whether_near_gt", "confidence",
    ]
    summary_fields = [
        "video_id", "num_gt_intervals", "num_caption_windows", "score_mean", "score_max",
        "score_min", "score_stride_est", "num_h1_candidates", "num_h4_candidates",
        "num_transition_mentions", "risk_keyword_total", "strong_keyword_total",
        "h1_candidate_types", "h4_candidate_types",
    ]
    write_csv_sig(data_dir / "h1_candidates.csv", h1_all, h1_fields)
    write_csv_sig(data_dir / "h4_candidates.csv", h4_all, h4_fields)
    write_csv_sig(data_dir / "video_summary.csv", video_rows, summary_fields)
    write_csv_sig(data_dir / "warnings.csv", [{"warning": item} for item in warnings], ["warning"])
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    summary = {
        "processed_videos": processed,
        "skipped_videos": skipped,
        "total_h1": len(h1_all),
        "total_h4": len(h4_all),
        "warnings": warnings,
        "output_dir": str(output_dir),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir, args, summary)
    print(f"processed videos: {processed}")
    print(f"skipped videos: {skipped}")
    print(f"total H1 candidates: {len(h1_all)}")
    print(f"total H4 candidates: {len(h4_all)}")
    print(f"output dir: {output_dir}")
    print(f"h1 candidates: {data_dir / 'h1_candidates.csv'}")
    print(f"h4 candidates: {data_dir / 'h4_candidates.csv'}")
    print(f"video summary: {data_dir / 'video_summary.csv'}")
    if args.plot:
        print(f"plots: {plot_dir}")
        print(f"figures: {figure_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Screen H1/H4 VAD diagnostic candidates from caption, score, and GT files.")
    parser.add_argument("--caption_dir", required=True, help="Directory containing caption JSON files keyed by frame index.")
    parser.add_argument("--score_dir", required=True, help="Directory containing anomaly score JSON files keyed by frame index.")
    parser.add_argument("--gt_file", required=True, help="GT annotation txt/csv file.")
    parser.add_argument("--output_dir", default="", help="Output archive directory. Default uses outputs/yy-mm-dd-hh-mm-h1-h4-candidate-screen.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--early_window_frames", type=int, default=96)
    parser.add_argument("--low_score_threshold", type=float, default=0.2)
    parser.add_argument("--high_score_threshold", type=float, default=0.7)
    parser.add_argument("--mid_low", type=float, default=0.3)
    parser.add_argument("--mid_high", type=float, default=0.6)
    parser.add_argument("--min_mid_run", type=int, default=3)
    parser.add_argument("--pre_k", type=int, default=3)
    parser.add_argument("--post_k", type=int, default=3)
    parser.add_argument("--h4_high_score_threshold", type=float, default=0.6)
    parser.add_argument("--h4_low_score_threshold", type=float, default=0.3)
    parser.add_argument("--h4_drop_threshold", type=float, default=0.3)
    parser.add_argument("--near_gt_margin", type=int, default=48)
    parser.add_argument("--plot", action="store_true", help="Write one diagnostic PNG per processed video.")
    parser.add_argument("--video_dir", default="", help="Reserved for future visual shot-boundary detection.")
    parser.add_argument("--use_visual_shot_boundary", action="store_true", help="Reserved. TODO: replace caption transition detection with visual shot boundaries.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_visual_shot_boundary:
        print("warning: --use_visual_shot_boundary is reserved; using caption-based transition detection for now.")
    process(args)


if __name__ == "__main__":
    main()
