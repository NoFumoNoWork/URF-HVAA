import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FULL_FIELDS = [
    "dataset", "video_id", "frame", "score", "confidence", "types",
    "lexical_drop", "semantic_drop_adjacent", "semantic_drop_block",
    "score_before", "score_after", "score_drop", "gt_inside",
    "gt_distance_to_nearest_boundary", "caption_before", "caption_current",
    "caption_after", "boundary_reason",
]

DATASET_FIELDS = [
    "dataset", "num_videos", "num_candidates", "num_strong", "num_medium",
    "num_weak", "num_explicit_transition_boundary", "num_lexical_topic_boundary",
    "num_composite_context_boundary", "num_multi_scene_compression_boundary",
    "num_possible_context_forgetting", "num_event_onset_not_h4", "mean_score_drop",
    "positive_score_drop_ratio", "negative_score_drop_ratio",
]

VIDEO_FIELDS = [
    "dataset", "video_id", "num_candidates", "num_strong", "num_medium",
    "num_weak", "dominant_type", "num_possible_context_forgetting",
    "mean_score_drop", "positive_score_drop_ratio", "max_boundary_score",
    "top_candidate_frames",
]

H4_FIELDS = [
    "dataset", "video_id", "frame", "score", "confidence", "types",
    "score_drop", "lexical_drop", "semantic_drop_block", "caption_before",
    "caption_current", "caption_after", "gt_inside",
    "gt_distance_to_nearest_boundary", "preliminary_h4_label",
]

TYPE_FIELDS = [
    "type", "count", "mean_score_drop", "median_score_drop",
    "positive_score_drop_ratio", "negative_score_drop_ratio",
    "zero_or_near_zero_score_drop_ratio", "mean_lexical_drop",
    "mean_semantic_drop_block",
]

METHOD_FIELDS = [
    "method", "total_candidates", "num_strong", "num_medium", "num_weak",
    "num_possible_context_forgetting", "num_multi_scene_compression_boundary",
    "mean_score_drop", "positive_score_drop_ratio", "top_video_concentration_ratio",
]


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, digits)
    return value


def ratio(count, total):
    return count / total if total else ""


def type_list(row):
    return [item for item in (row.get("types") or row.get("candidate_types") or "").split(";") if item]


def run_command(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def ensure_rule_only(args, output_dir):
    rule_dir = output_dir / "rule_only"
    candidate_path = rule_dir / "caption_boundary_candidates.csv"
    debug_path = rule_dir / "caption_boundary_debug.csv"
    if candidate_path.exists() and debug_path.exists() and not args.rerun_screen:
        return rule_dir, {"cmd": "reuse existing rule_only output", "returncode": 0, "stdout": "", "stderr": ""}
    cmd = [
        sys.executable,
        "scripts/caption_boundary_screen.py",
        "--data-root",
        args.data_root,
        "--output-dir",
        str(rule_dir),
        "--plot-top-n",
        str(args.plot_top_n),
    ]
    result = run_command(cmd, ROOT)
    if result["returncode"] != 0:
        raise RuntimeError(f"rule-only screen failed: {result['stderr']}")
    return rule_dir, result


def sbert_available():
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, ""


def maybe_run_sbert(args, output_dir):
    available, reason = sbert_available()
    if not available:
        return None, {
            "cmd": "skip SBERT full run",
            "returncode": 0,
            "stdout": "",
            "stderr": f"sentence_transformers unavailable: {reason}",
            "available": False,
            "reason": reason,
        }
    sbert_dir = output_dir / "sbert"
    candidate_path = sbert_dir / "caption_boundary_candidates.csv"
    if candidate_path.exists() and not args.rerun_screen:
        return sbert_dir, {"cmd": "reuse existing SBERT output", "returncode": 0, "stdout": "", "stderr": "", "available": True}
    cmd = [
        sys.executable,
        "scripts/caption_boundary_screen.py",
        "--data-root",
        args.data_root,
        "--output-dir",
        str(sbert_dir),
        "--plot-top-n",
        str(args.plot_top_n),
        "--use-sbert",
    ]
    result = run_command(cmd, ROOT)
    result["available"] = True
    if result["returncode"] != 0:
        result["reason"] = result["stderr"]
        return None, result
    return sbert_dir, result


def debug_index(debug_rows):
    return {(r.get("dataset"), r.get("video_id"), r.get("frame")): r for r in debug_rows}


def boundary_reason(row):
    parts = []
    types = type_list(row)
    if types:
        parts.append("types=" + ";".join(types))
    if row.get("matched_transition_keywords"):
        parts.append("transition_keywords=" + row["matched_transition_keywords"])
    if row.get("notes"):
        parts.append("notes=" + row["notes"])
    return " | ".join(parts)


def build_full_candidates(candidate_rows, debug_rows):
    index = debug_index(debug_rows)
    result = []
    missing_distance = 0
    for row in candidate_rows:
        dbg = index.get((row.get("dataset"), row.get("video_id"), row.get("frame")), {})
        if not dbg:
            missing_distance += 1
        result.append(
            {
                "dataset": row.get("dataset", ""),
                "video_id": row.get("video_id", ""),
                "frame": row.get("frame", ""),
                "score": row.get("boundary_score", ""),
                "confidence": row.get("confidence", ""),
                "types": row.get("candidate_types", ""),
                "lexical_drop": row.get("lexical_drop", ""),
                "semantic_drop_adjacent": row.get("semantic_drop_adjacent", ""),
                "semantic_drop_block": row.get("semantic_drop_block", ""),
                "score_before": row.get("pre_score_mean", ""),
                "score_after": row.get("post_score_mean", ""),
                "score_drop": row.get("score_drop", ""),
                "gt_inside": row.get("inside_gt", ""),
                "gt_distance_to_nearest_boundary": dbg.get("distance_to_gt_boundary", ""),
                "caption_before": row.get("caption_prev", ""),
                "caption_current": row.get("caption_current", ""),
                "caption_after": row.get("caption_next", ""),
                "boundary_reason": boundary_reason(row),
            }
        )
    return result, {"missing_debug_join_rows": missing_distance}


def score_drop_values(rows):
    return [v for v in (as_float(r.get("score_drop")) for r in rows) if v is not None]


def summarize_group(rows):
    drops = score_drop_values(rows)
    positives = sum(1 for v in drops if v > 1e-9)
    negatives = sum(1 for v in drops if v < -1e-9)
    return {
        "mean_score_drop": fmt(statistics.fmean(drops) if drops else None),
        "positive_score_drop_ratio": fmt(ratio(positives, len(drops))),
        "negative_score_drop_ratio": fmt(ratio(negatives, len(drops))),
    }


def dataset_summary(rows, video_summaries):
    grouped = defaultdict(list)
    videos = defaultdict(set)
    for row in rows:
        grouped[row["dataset"]].append(row)
        videos[row["dataset"]].add(row["video_id"])
    for row in video_summaries:
        videos[row["dataset"]].add(row["video_id"])
    output = []
    for dataset in sorted(videos):
        items = grouped.get(dataset, [])
        conf = Counter(row["confidence"] for row in items)
        types = Counter(t for row in items for t in type_list(row))
        base = summarize_group(items)
        output.append(
            {
                "dataset": dataset,
                "num_videos": len(videos[dataset]),
                "num_candidates": len(items),
                "num_strong": conf["strong"],
                "num_medium": conf["medium"],
                "num_weak": conf["weak"],
                "num_explicit_transition_boundary": types["explicit_transition_boundary"],
                "num_lexical_topic_boundary": types["lexical_topic_boundary"],
                "num_composite_context_boundary": types["composite_context_boundary"],
                "num_multi_scene_compression_boundary": types["multi_scene_compression_boundary"],
                "num_possible_context_forgetting": types["possible_context_forgetting"],
                "num_event_onset_not_h4": types["event_onset_not_h4"],
                **base,
            }
        )
    return output


def video_summary(rows, video_summaries):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["video_id"])].append(row)
    output = []
    keys = set(grouped)
    keys.update((row["dataset"], row["video_id"]) for row in video_summaries)
    for (dataset, video_id) in sorted(keys):
        items = grouped.get((dataset, video_id), [])
        conf = Counter(row["confidence"] for row in items)
        types = Counter(t for row in items for t in type_list(row))
        dominant = types.most_common(1)[0][0] if types else ""
        drops = summarize_group(items)
        max_score = max((as_float(row["score"]) or 0 for row in items), default=0)
        top = sorted(items, key=lambda row: (as_float(row.get("score")) or 0), reverse=True)[:5]
        output.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "num_candidates": len(items),
                "num_strong": conf["strong"],
                "num_medium": conf["medium"],
                "num_weak": conf["weak"],
                "dominant_type": dominant,
                "num_possible_context_forgetting": types["possible_context_forgetting"],
                "mean_score_drop": drops["mean_score_drop"],
                "positive_score_drop_ratio": drops["positive_score_drop_ratio"],
                "max_boundary_score": fmt(max_score),
                "top_candidate_frames": ";".join(row["frame"] for row in top),
            }
        )
    return output


def preliminary_h4_label(row):
    types = set(type_list(row))
    drop = as_float(row.get("score_drop"))
    if "event_onset_not_h4" in types:
        return "likely_event_onset_not_h4"
    if "possible_context_forgetting" in types:
        return "likely_h4_candidate"
    if row.get("confidence") == "strong" and drop is not None and drop > 0 and (
        "multi_scene_compression_boundary" in types or "explicit_transition_boundary" in types
    ):
        return "needs_visual_check"
    return "unclear"


def h4_strong_candidates(rows):
    output = []
    for row in rows:
        types = set(type_list(row))
        drop = as_float(row.get("score_drop"))
        keep = "possible_context_forgetting" in types
        keep = keep or (row.get("confidence") == "strong" and "multi_scene_compression_boundary" in types and drop is not None and drop > 0)
        keep = keep or (row.get("confidence") == "strong" and "explicit_transition_boundary" in types and drop is not None and drop > 0)
        if not keep:
            continue
        output.append(
            {
                "dataset": row["dataset"],
                "video_id": row["video_id"],
                "frame": row["frame"],
                "score": row["score"],
                "confidence": row["confidence"],
                "types": row["types"],
                "score_drop": row["score_drop"],
                "lexical_drop": row["lexical_drop"],
                "semantic_drop_block": row["semantic_drop_block"],
                "caption_before": row["caption_before"],
                "caption_current": row["caption_current"],
                "caption_after": row["caption_after"],
                "gt_inside": row["gt_inside"],
                "gt_distance_to_nearest_boundary": row["gt_distance_to_nearest_boundary"],
                "preliminary_h4_label": preliminary_h4_label(row),
            }
        )
    return output


def type_score_drop_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        for item in type_list(row):
            grouped[item].append(row)
    output = []
    for item_type, items in sorted(grouped.items()):
        drops = score_drop_values(items)
        positives = sum(1 for v in drops if v > 1e-9)
        negatives = sum(1 for v in drops if v < -1e-9)
        near_zero = sum(1 for v in drops if abs(v) <= 1e-9)
        lexical = [v for v in (as_float(r.get("lexical_drop")) for r in items) if v is not None]
        semantic = [v for v in (as_float(r.get("semantic_drop_block")) for r in items) if v is not None]
        output.append(
            {
                "type": item_type,
                "count": len(items),
                "mean_score_drop": fmt(statistics.fmean(drops) if drops else None),
                "median_score_drop": fmt(statistics.median(drops) if drops else None),
                "positive_score_drop_ratio": fmt(ratio(positives, len(drops))),
                "negative_score_drop_ratio": fmt(ratio(negatives, len(drops))),
                "zero_or_near_zero_score_drop_ratio": fmt(ratio(near_zero, len(drops))),
                "mean_lexical_drop": fmt(statistics.fmean(lexical) if lexical else None),
                "mean_semantic_drop_block": fmt(statistics.fmean(semantic) if semantic else None),
            }
        )
    return output


def method_row(method, rows):
    conf = Counter(row["confidence"] for row in rows)
    types = Counter(t for row in rows for t in type_list(row))
    by_video = Counter((row["dataset"], row["video_id"]) for row in rows)
    top_count = by_video.most_common(1)[0][1] if by_video else 0
    group = summarize_group(rows)
    return {
        "method": method,
        "total_candidates": len(rows),
        "num_strong": conf["strong"],
        "num_medium": conf["medium"],
        "num_weak": conf["weak"],
        "num_possible_context_forgetting": types["possible_context_forgetting"],
        "num_multi_scene_compression_boundary": types["multi_scene_compression_boundary"],
        "mean_score_drop": group["mean_score_drop"],
        "positive_score_drop_ratio": group["positive_score_drop_ratio"],
        "top_video_concentration_ratio": fmt(ratio(top_count, len(rows))),
    }


def load_method_candidates(method_dir):
    candidates = read_csv(method_dir / "caption_boundary_candidates.csv")
    debug = read_csv(method_dir / "caption_boundary_debug.csv")
    full, _ = build_full_candidates(candidates, debug)
    return full


def write_report(path, args, run_info, full_rows, ds_rows, video_rows, type_rows, h4_rows, method_rows):
    conf = Counter(row["confidence"] for row in full_rows)
    types = Counter(t for row in full_rows for t in type_list(row))
    beautiful = [
        row for row in video_rows
        if row["dataset"] == "XD-Violence" and row["video_id"] == "A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A"
    ]
    top_videos = sorted(video_rows, key=lambda row: int(row["num_candidates"]), reverse=True)[:10]
    h4_top = sorted(h4_rows, key=lambda row: (as_float(row.get("score")) or 0, as_float(row.get("score_drop")) or 0), reverse=True)[:20]
    lines = [
        "# H4 Caption Boundary Experiment Report",
        "",
        "## 1. Experiment Goal",
        "",
        "This experiment checks whether caption-level context boundaries, explicit transitions, multi-scene compression, or possible context forgetting align with anomaly-score discontinuities. The goal is to produce H4 verification candidates, not to claim causal proof.",
        "",
        "## 2. Inputs and Settings",
        "",
        f"- data_root: `{args.data_root}`",
        "- rule-only command: `python scripts/caption_boundary_screen.py --data-root data --output-dir outputs/caption_boundary_screen/rule_only --plot-top-n 20`",
        "- SBERT command requested: `python scripts/caption_boundary_screen.py --data-root data --output-dir outputs/caption_boundary_screen/sbert --use-sbert`",
        f"- SBERT status: {run_info['sbert_status']}",
        f"- processed videos: {run_info['rule_summary'].get('processed_videos', '')}",
        f"- datasets: {', '.join(run_info['rule_summary'].get('datasets', []))}",
        "",
        "## 3. Overall Candidate Summary",
        "",
        f"- total candidates: {len(full_rows)}",
        f"- confidence counts: strong={conf['strong']}, medium={conf['medium']}, weak={conf['weak']}",
        "",
        "Candidate type counts:",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(types.items()))
    lines.extend(["", "## 4. Dataset-level Distribution", ""])
    lines.append("| dataset | videos | candidates | strong | possible H4 | mean score_drop | positive drop ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in ds_rows:
        lines.append(
            f"| {row['dataset']} | {row['num_videos']} | {row['num_candidates']} | {row['num_strong']} | "
            f"{row['num_possible_context_forgetting']} | {row['mean_score_drop']} | {row['positive_score_drop_ratio']} |"
        )
    lines.extend(["", "## 5. Video-level Concentration", ""])
    lines.append("| dataset | video_id | candidates | strong | dominant_type | top frames |")
    lines.append("|---|---|---:|---:|---|---|")
    for row in top_videos:
        lines.append(
            f"| {row['dataset']} | {row['video_id']} | {row['num_candidates']} | {row['num_strong']} | "
            f"{row['dominant_type']} | {row['top_candidate_frames']} |"
        )
    if beautiful:
        row = beautiful[0]
        lines.extend(
            [
                "",
                f"XD-Violence `A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A` contributes {row['num_candidates']} candidates, including {row['num_strong']} strong candidates. This should be watched as a possible concentration case rather than treated as representative by itself.",
            ]
        )
    lines.extend(["", "## 6. Type vs Score-drop Analysis", ""])
    lines.append("| type | count | mean drop | median drop | positive ratio | negative ratio | near-zero ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in type_rows:
        lines.append(
            f"| {row['type']} | {row['count']} | {row['mean_score_drop']} | {row['median_score_drop']} | "
            f"{row['positive_score_drop_ratio']} | {row['negative_score_drop_ratio']} | {row['zero_or_near_zero_score_drop_ratio']} |"
        )
    lines.extend(["", "## 7. H4 Strong Candidates", ""])
    lines.append("These rows are rule-prioritized candidates for manual inspection, not confirmed H4 events.")
    lines.append("")
    lines.append("| dataset | video_id | frame | label | score_drop | caption_current |")
    lines.append("|---|---|---:|---|---:|---|")
    for row in h4_top:
        caption = (row["caption_current"] or "").replace("|", "/")[:180]
        lines.append(
            f"| {row['dataset']} | {row['video_id']} | {row['frame']} | {row['preliminary_h4_label']} | "
            f"{row['score_drop']} | {caption} |"
        )
    lines.extend(["", "## 8. Rule-only vs SBERT/Hybrid Comparison", ""])
    lines.append("| method | candidates | strong | possible H4 | mean drop | positive ratio | top video concentration |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['total_candidates']} | {row['num_strong']} | "
            f"{row['num_possible_context_forgetting']} | {row['mean_score_drop']} | "
            f"{row['positive_score_drop_ratio']} | {row['top_video_concentration_ratio']} |"
        )
    if run_info["sbert_status"] != "completed":
        lines.append("")
        lines.append(f"SBERT/hybrid was not run as a full comparison because: {run_info['sbert_status']}.")
    lines.extend(
        [
            "",
            "## 9. Interpretation",
            "",
            "- The caption streams contain many automatically detectable context boundaries across datasets.",
            "- A smaller subset combines high boundary scores with positive anomaly-score drops; these are useful H4 manual-check candidates.",
            "- These statistics do not prove that context forgetting is widespread, and they do not prove that boundaries cause score drops.",
            "- `event_onset_not_h4` rows are retained as an explicit exclusion/control group because genuine event onset can look like a semantic boundary.",
            "",
            "## 10. Next Steps",
            "",
            "- Inspect raw video around `h4_strong_candidates.csv` rows for visual cuts, shot changes, and multi-scene compression.",
            "- Separate visual boundary cases from event-onset cases using the before/current/after captions.",
            "- Check score/GT alignment around each strong candidate, especially whether score drops occur inside GT or near GT boundaries.",
            "- Compare false-positive and false-negative interval behavior around boundary frames before making H4 claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run H4 caption-boundary validation exports and report.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/caption_boundary_screen")
    parser.add_argument("--plot-top-n", type=int, default=20)
    parser.add_argument("--rerun-screen", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rule_dir, rule_cmd = ensure_rule_only(args, output_dir)
    sbert_dir, sbert_cmd = maybe_run_sbert(args, output_dir)

    rule_candidates = read_csv(rule_dir / "caption_boundary_candidates.csv")
    rule_debug = read_csv(rule_dir / "caption_boundary_debug.csv")
    rule_video_summaries = read_csv(rule_dir / "caption_boundary_summary.csv")
    full_rows, join_info = build_full_candidates(rule_candidates, rule_debug)
    ds_rows = dataset_summary(full_rows, rule_video_summaries)
    vid_rows = video_summary(full_rows, rule_video_summaries)
    h4_rows = h4_strong_candidates(full_rows)
    type_rows = type_score_drop_summary(full_rows)

    method_rows = [method_row("rule_only", full_rows)]
    sbert_status = "not_available"
    if sbert_dir:
        sbert_summary = json.loads((sbert_dir / "caption_boundary_summary.json").read_text(encoding="utf-8"))
        if sbert_summary.get("sbert_used"):
            method_rows.append(method_row("sbert_hybrid", load_method_candidates(sbert_dir)))
            sbert_status = "completed"
        else:
            sbert_status = "requested_but_not_used"
    else:
        sbert_status = sbert_cmd.get("stderr") or sbert_cmd.get("reason") or "not_available"

    write_csv(output_dir / "candidates_full.csv", full_rows, FULL_FIELDS)
    write_csv(output_dir / "dataset_summary.csv", ds_rows, DATASET_FIELDS)
    write_csv(output_dir / "video_summary.csv", vid_rows, VIDEO_FIELDS)
    write_csv(output_dir / "h4_strong_candidates.csv", h4_rows, H4_FIELDS)
    write_csv(output_dir / "type_score_drop_summary.csv", type_rows, TYPE_FIELDS)
    write_csv(output_dir / "method_comparison.csv", method_rows, METHOD_FIELDS)
    run_info = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rule_command": rule_cmd,
        "sbert_command": sbert_cmd,
        "sbert_status": sbert_status,
        "join_info": join_info,
        "rule_summary": json.loads((rule_dir / "caption_boundary_summary.json").read_text(encoding="utf-8")),
        "outputs": {
            "candidates_full": str(output_dir / "candidates_full.csv"),
            "dataset_summary": str(output_dir / "dataset_summary.csv"),
            "video_summary": str(output_dir / "video_summary.csv"),
            "h4_strong_candidates": str(output_dir / "h4_strong_candidates.csv"),
            "type_score_drop_summary": str(output_dir / "type_score_drop_summary.csv"),
            "method_comparison": str(output_dir / "method_comparison.csv"),
            "report": str(output_dir / "h4_boundary_experiment_report.md"),
        },
    }
    write_json(output_dir / "h4_boundary_experiment_summary.json", run_info)
    write_report(output_dir / "h4_boundary_experiment_report.md", args, run_info, full_rows, ds_rows, vid_rows, type_rows, h4_rows, method_rows)
    print(f"rule-only candidates: {len(full_rows)}")
    print(f"h4 strong candidates: {len(h4_rows)}")
    print(f"SBERT status: {sbert_status}")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
