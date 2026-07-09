import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


DATASET_CONFIGS = [
    {
        "dataset": "MSAD",
        "caption_dir": "data/MSAD/captions/video_llama3_json_results",
        "score_dir": "data/MSAD/refined_scores/videollama3",
        "gt_file": "data/MSAD/annotations/test.txt",
    },
    {
        "dataset": "UBNormal",
        "caption_dir": "data/UBNormal/caption/video_llama3_json_results",
        "score_dir": "data/UBNormal/refined_scores/videollama3",
        "gt_file": "data/UBNormal/annotations/temporal.txt",
    },
    {
        "dataset": "UCF-Crime",
        "caption_dir": "data/ucf_crime/captions/video_llama3_json_results",
        "score_dir": "data/ucf_crime/refined_scores/videollama3",
        "gt_file": "data/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    },
    {
        "dataset": "XD-Violence",
        "caption_dir": "data/xd_violence/captions/video_llama3_json_results",
        "score_dir": "data/xd_violence/refined_scores/videollama3",
        "gt_file": "data/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt",
    },
]

SOURCE_PROXY = {
    "XD-Violence": "edited/movie/web-video proxy",
    "UCF-Crime": "surveillance/crime-video proxy",
    "MSAD": "surveillance-like proxy",
    "UBNormal": "surveillance-like proxy",
}

CANONICAL_FIELDS = [
    "dataset", "video_id", "unit_type", "item_id", "start", "end", "center",
    "caption", "anomaly_score", "is_gt", "gt_interval_id", "h4_nearby",
    "h4_count_nearby", "nearest_h4_distance", "nearest_h4_type",
    "nearest_h4_score",
]

H4_DIAG_FIELDS = [
    "dataset", "video_id", "h4_id", "h4_position", "h4_type", "h4_score",
    "score_drop", "caption_before", "caption_after", "inside_gt",
    "distance_to_nearest_gt_start", "distance_to_nearest_gt_end",
    "distance_to_nearest_prediction_gap", "inside_prediction_gap",
    "near_prediction_gap", "nearest_gap_id", "gap_oracle_label",
    "source_dataset_or_proxy",
]


def normalize_stem(value):
    return re.sub(r"\(\d+\)$", "", Path(str(value)).stem).strip()


def read_csv(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
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


def as_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    value = as_float(value, default)
    if value is None:
        return default
    return int(value)


def fmt(value, digits=6):
    if value in ("", None):
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value) or math.isinf(value):
        return ""
    return round(value, digits)


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


def load_json_map(path):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def load_captions(path):
    raw = load_json_map(path)
    rows = []
    for key, value in raw.items():
        frame = as_int(key)
        if frame is None:
            continue
        rows.append({"frame": frame, "caption": stringify_caption(value).strip()})
    return sorted(rows, key=lambda row: row["frame"])


def load_scores(path):
    raw = load_json_map(path)
    out = {}
    for key, value in raw.items():
        frame = as_int(key)
        score = as_float(value)
        if frame is not None and score is not None:
            out[frame] = score
    return dict(sorted(out.items()))


def json_index(directory):
    result = {}
    path = Path(directory)
    if not path.exists():
        return result
    for item in sorted(path.glob("*.json")):
        result.setdefault(item.stem, item)
        result.setdefault(normalize_stem(item.stem), item)
    return result


def parse_gt_file(path):
    rows = defaultdict(list)
    path = Path(path)
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
        label = ""
        numeric = []
        for item in parts[1:]:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", item):
                numeric.append(int(float(item)))
            elif not label:
                label = item
        for idx, (start, end) in enumerate(zip(numeric[0::2], numeric[1::2])):
            if start == -1 or end == -1:
                break
            if end > start:
                rows[video_id].append({
                    "gt_interval_id": f"{video_id}_gt_{idx}",
                    "start": start,
                    "end": end,
                    "label": label,
                })
    return rows


def interval_overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def nearest_value(values, center):
    if not values:
        return None, None
    nearest = min(values, key=lambda item: abs(item - center))
    return nearest, abs(nearest - center)


def infer_item_end(rows, idx):
    start = rows[idx]["frame"]
    if idx + 1 < len(rows):
        return rows[idx + 1]["frame"]
    if idx > 0:
        return start + max(1, start - rows[idx - 1]["frame"])
    return start + 1


def field_preview(path, kind):
    path = Path(path)
    info = {
        "path": str(path).replace("\\", "/"),
        "kind_guess": kind,
        "exists": path.exists(),
        "rows_or_items": "",
        "columns_or_keys": [],
        "preview": [],
        "field_mapping_clear": False,
    }
    if not path.exists():
        return info
    if path.suffix.lower() == ".csv":
        rows = read_csv(path)
        info["rows_or_items"] = len(rows)
        info["columns_or_keys"] = list(rows[0].keys()) if rows else []
        info["preview"] = rows[:5]
        info["field_mapping_clear"] = bool(rows)
    elif path.suffix.lower() == ".json":
        raw = load_json_map(path)
        info["rows_or_items"] = len(raw)
        info["columns_or_keys"] = list(raw.keys())[:10]
        info["preview"] = [{key: raw[key]} for key in list(raw.keys())[:5]]
        info["field_mapping_clear"] = kind in {"caption", "score"} and bool(raw)
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        info["rows_or_items"] = len(lines)
        info["columns_or_keys"] = ["whitespace-delimited"]
        info["preview"] = lines[:5]
        info["field_mapping_clear"] = bool(lines)
    return info


def build_schema_guess(output_dir, h4_candidates, prediction_intervals):
    resources = {
        "caption": [],
        "score": [],
        "h4": [{
            "path": str(h4_candidates).replace("\\", "/"),
            "mapping": {
                "dataset": "dataset",
                "video_id": "video_id",
                "position": "frame",
                "h4_score": "score",
                "h4_type": "types",
                "score_drop": "score_drop",
                "caption_before": "caption_before",
                "caption_after": "caption_after",
            },
        }],
        "gt": [],
        "prediction": [],
    }
    for cfg in DATASET_CONFIGS:
        resources["caption"].append({
            "dataset": cfg["dataset"],
            "path": cfg["caption_dir"],
            "mapping": {"json_key": "frame", "json_value": "caption"},
        })
        resources["score"].append({
            "dataset": cfg["dataset"],
            "path": cfg["score_dir"],
            "mapping": {"json_key": "frame", "json_value": "anomaly_score"},
        })
        resources["gt"].append({
            "dataset": cfg["dataset"],
            "path": cfg["gt_file"],
            "mapping": {"first_token": "video_id", "numeric_pairs": "gt_start/gt_end"},
        })
    if prediction_intervals and Path(prediction_intervals).exists():
        resources["prediction"].append({
            "path": str(prediction_intervals).replace("\\", "/"),
            "mapping": {
                "dataset": "dataset",
                "video_id": "video_id",
                "start": "start_frame",
                "end": "end_frame",
                "score": "mean_score/max_score",
            },
        })
    else:
        resources["prediction"].append({
            "path": "",
            "mapping": "not found; generate from score threshold",
        })
    obj = {
        "created_for": "H4 caption-level resource preparation",
        "unit_assumption": "frame for current verified caption/score/H4/GT files; schema keeps unit_type for future second/clip inputs",
        "resources": resources,
        "canonical_timeline_output": str((output_dir / "canonical_timeline.csv")).replace("\\", "/"),
    }
    write_json(output_dir / "input_schema_guess.json", obj)
    return obj


def write_inventory(output_dir, h4_candidates, prediction_intervals):
    entries = []
    for cfg in DATASET_CONFIGS:
        cap_dir = Path(cfg["caption_dir"])
        score_dir = Path(cfg["score_dir"])
        cap_files = sorted(cap_dir.glob("*.json")) if cap_dir.exists() else []
        score_files = sorted(score_dir.glob("*.json")) if score_dir.exists() else []
        if cap_files:
            item = field_preview(cap_files[0], "caption")
            item["path"] = str(cap_dir).replace("\\", "/")
            item["rows_or_items"] = f"{len(cap_files)} JSON files; preview uses {cap_files[0].name}"
            entries.append(item)
        if score_files:
            usable = [p for p in score_files if p.name not in {"context_prompt.txt", "format_prompt.txt"}]
            sample = usable[0] if usable else score_files[0]
            item = field_preview(sample, "score")
            item["path"] = str(score_dir).replace("\\", "/")
            item["rows_or_items"] = f"{len(score_files)} JSON files; preview uses {sample.name}"
            entries.append(item)
        entries.append(field_preview(cfg["gt_file"], "gt"))
    entries.append(field_preview(h4_candidates, "h4"))
    if prediction_intervals:
        entries.append(field_preview(prediction_intervals, "prediction"))
    else:
        entries.append({
            "path": "",
            "kind_guess": "prediction",
            "exists": False,
            "rows_or_items": "",
            "columns_or_keys": [],
            "preview": ["No required prediction interval input supplied; can be generated from anomaly score threshold."],
            "field_mapping_clear": False,
        })

    lines = ["# Input inventory", ""]
    for entry in entries:
        lines.extend([
            f"## {entry['kind_guess']}: {entry['path'] or 'not supplied'}",
            "",
            f"- exists: {entry['exists']}",
            f"- rows/items: {entry['rows_or_items']}",
            f"- columns/keys: `{entry['columns_or_keys']}`",
            f"- field mapping clear: {entry['field_mapping_clear']}",
            "- preview:",
            "",
            "```text",
            json.dumps(entry["preview"], ensure_ascii=False, indent=2)[:3000],
            "```",
            "",
        ])
    (output_dir / "input_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "input_inventory.json", entries)
    return entries


def load_all_gt():
    gt = {}
    for cfg in DATASET_CONFIGS:
        for video_id, intervals in parse_gt_file(cfg["gt_file"]).items():
            gt[(cfg["dataset"], video_id)] = intervals
    return gt


def load_h4(path):
    rows = []
    for idx, row in enumerate(read_csv(path)):
        dataset = row.get("dataset", "")
        video_id = normalize_stem(row.get("video_id", ""))
        position = as_int(row.get("frame"))
        if not dataset or not video_id or position is None:
            continue
        rows.append({
            "dataset": dataset,
            "video_id": video_id,
            "h4_id": f"h4_{idx}",
            "position": position,
            "h4_type": row.get("types", ""),
            "h4_score": row.get("score", ""),
            "score_drop": row.get("score_drop", ""),
            "caption_before": row.get("caption_before", ""),
            "caption_after": row.get("caption_after", ""),
        })
    return rows


def build_canonical_timeline(output_dir, h4_candidates, h4_window):
    h4_rows = load_h4(h4_candidates)
    h4_by_key = defaultdict(list)
    for row in h4_rows:
        h4_by_key[(row["dataset"], row["video_id"])].append(row)
    for key in h4_by_key:
        h4_by_key[key].sort(key=lambda row: row["position"])
    gt_by_key = load_all_gt()
    canonical = []
    warnings = []
    for cfg in DATASET_CONFIGS:
        caption_index = json_index(cfg["caption_dir"])
        score_index = json_index(cfg["score_dir"])
        for video_id, caption_path in sorted(caption_index.items()):
            if video_id != normalize_stem(video_id):
                continue
            captions = load_captions(caption_path)
            scores = load_scores(score_index.get(video_id, ""))
            if not scores:
                warnings.append(f"{cfg['dataset']}/{video_id}: missing score JSON or empty scores")
            h4s = h4_by_key.get((cfg["dataset"], video_id), [])
            h4_positions = [row["position"] for row in h4s]
            gt_intervals = gt_by_key.get((cfg["dataset"], video_id), [])
            for idx, item in enumerate(captions):
                start = item["frame"]
                end = infer_item_end(captions, idx)
                center = (start + end) / 2.0
                score_frame, _ = nearest_value(list(scores.keys()), center)
                score = scores.get(score_frame, "")
                gt_hits = [
                    gt["gt_interval_id"]
                    for gt in gt_intervals
                    if interval_overlap(start, end, gt["start"], gt["end"]) > 0
                ]
                nearest_h4, h4_distance = nearest_value(h4_positions, center)
                near_h4s = [row for row in h4s if abs(row["position"] - center) <= h4_window]
                nearest_row = next((row for row in h4s if row["position"] == nearest_h4), {})
                canonical.append({
                    "dataset": cfg["dataset"],
                    "video_id": video_id,
                    "unit_type": "frame",
                    "item_id": f"{video_id}_{start}",
                    "start": start,
                    "end": end,
                    "center": fmt(center),
                    "caption": item["caption"],
                    "anomaly_score": fmt(score),
                    "is_gt": bool(gt_hits),
                    "gt_interval_id": ";".join(gt_hits),
                    "h4_nearby": bool(near_h4s),
                    "h4_count_nearby": len(near_h4s),
                    "nearest_h4_distance": fmt(h4_distance),
                    "nearest_h4_type": nearest_row.get("h4_type", ""),
                    "nearest_h4_score": nearest_row.get("h4_score", ""),
                })
    write_csv(output_dir / "canonical_timeline.csv", canonical, CANONICAL_FIELDS)
    notes = [
        "# Schema notes",
        "",
        "- Current verified caption, score, H4, GT, and prediction interval files use frame-like integer keys. `unit_type` is still retained for future second/clip/window inputs.",
        "- Caption rows are JSON key/value pairs; key is treated as frame start, value as caption text.",
        "- `end` is inferred from the next caption frame. For the final caption in a video, the previous stride is reused; if unavailable, length is one frame.",
        "- Score alignment is nearest-frame alignment from refined score JSON to caption center.",
        "- GT alignment marks a caption item as GT if its inferred frame span overlaps any GT interval.",
        "- H4 alignment treats H4 as a boundary candidate only. `h4_nearby` means an H4 candidate is within the configured window; it does not mean a true scene switch or a merge decision.",
        "- H4 candidate semantics remain ambiguous: same event, new viewpoint, related consequence, narrative jump, caption phrasing change, and true visual transition are not separated here.",
        "",
        "## Warnings",
        "",
    ]
    notes.extend([f"- {warning}" for warning in warnings] or ["- None."])
    (output_dir / "schema_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return canonical, warnings


def gt_distance(gt_intervals, pos):
    if not gt_intervals:
        return False, "", ""
    inside = any(gt["start"] <= pos <= gt["end"] for gt in gt_intervals)
    start_dist = min(abs(pos - gt["start"]) for gt in gt_intervals)
    end_dist = min(abs(pos - gt["end"]) for gt in gt_intervals)
    return inside, start_dist, end_dist


def load_gaps(path):
    rows = read_csv(path)
    by_key = defaultdict(list)
    for row in rows:
        key = (row.get("dataset", ""), normalize_stem(row.get("video_id", "")))
        by_key[key].append(row)
    return by_key


def build_h4_diagnostic(output_dir, h4_candidates, prediction_gaps, h4_window):
    h4_rows = load_h4(h4_candidates)
    gt_by_key = load_all_gt()
    gaps_by_key = load_gaps(prediction_gaps) if prediction_gaps and Path(prediction_gaps).exists() else {}
    out = []
    for row in h4_rows:
        key = (row["dataset"], row["video_id"])
        pos = row["position"]
        inside_gt, dist_gt_start, dist_gt_end = gt_distance(gt_by_key.get(key, []), pos)
        gaps = gaps_by_key.get(key, [])
        nearest_gap = None
        nearest_dist = None
        inside_gap = False
        for gap in gaps:
            start = as_int(gap.get("gap_start"))
            end = as_int(gap.get("gap_end"))
            if start is None or end is None:
                continue
            if start <= pos <= end:
                dist = 0
                inside_gap = True
            else:
                dist = min(abs(pos - start), abs(pos - end))
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_gap = gap
        out.append({
            "dataset": row["dataset"],
            "video_id": row["video_id"],
            "h4_id": row["h4_id"],
            "h4_position": pos,
            "h4_type": row["h4_type"],
            "h4_score": row["h4_score"],
            "score_drop": row["score_drop"],
            "caption_before": row["caption_before"],
            "caption_after": row["caption_after"],
            "inside_gt": inside_gt,
            "distance_to_nearest_gt_start": dist_gt_start,
            "distance_to_nearest_gt_end": dist_gt_end,
            "distance_to_nearest_prediction_gap": fmt(nearest_dist),
            "inside_prediction_gap": inside_gap,
            "near_prediction_gap": nearest_dist is not None and nearest_dist <= h4_window,
            "nearest_gap_id": nearest_gap.get("gap_id", "") if nearest_gap else "",
            "gap_oracle_label": nearest_gap.get("merge_oracle_label", "") if nearest_gap else "",
            "source_dataset_or_proxy": SOURCE_PROXY.get(row["dataset"], ""),
        })
    write_csv(output_dir / "h4_diagnostic_table.csv", out, H4_DIAG_FIELDS)
    write_h4_summary(output_dir / "h4_diagnostic_summary.md", out)
    return out


def write_h4_summary(path, rows):
    type_counts = Counter()
    gt_inside = 0
    gt_boundary = 0
    near_gap = 0
    relation = Counter()
    for row in rows:
        for item in str(row["h4_type"]).split(";"):
            if item:
                type_counts[item] += 1
        if str(row["inside_gt"]).lower() == "true":
            gt_inside += 1
        dists = [
            as_float(row["distance_to_nearest_gt_start"]),
            as_float(row["distance_to_nearest_gt_end"]),
        ]
        dists = [v for v in dists if v is not None]
        if dists and min(dists) <= 64:
            gt_boundary += 1
        if str(row["near_prediction_gap"]).lower() == "true":
            near_gap += 1
        label = row.get("gap_oracle_label", "") or "no_near_gap"
        for item in str(row["h4_type"]).split(";"):
            if item:
                relation[(item, label)] += 1
    lines = [
        "# H4 diagnostic summary",
        "",
        f"- H4 total rows: {len(rows)}",
        f"- H4 inside GT rows: {gt_inside}",
        f"- H4 near GT boundary rows within 64 frames: {gt_boundary}",
        f"- H4 near prediction gap rows: {near_gap}",
        "",
        "## H4 type counts",
        "",
    ]
    lines.extend([f"- {name}: {count}" for name, count in type_counts.most_common()])
    lines.extend(["", "## H4 type vs nearest gap oracle label", ""])
    lines.extend([f"- {name} / {label}: {count}" for (name, label), count in relation.most_common(30)])
    lines.extend([
        "",
        "## Field uncertainty",
        "",
        "- H4 rows are caption-level boundary candidates, not true camera transition labels.",
        "- `near_prediction_gap` depends on threshold-generated or supplied prediction gaps and should be interpreted as a diagnostic trigger only.",
        "- `gap_oracle_label` uses GT and is for upper-bound analysis; it is not an available deployment signal.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_inputs(output_dir, prediction_intervals):
    lines = [
        "# Missing inputs and manual confirmation needs",
        "",
        "## Missing or uncertain inputs",
        "",
        "- Original videos are not present in the resource-prep pipeline, so true camera transition / shot-boundary verification is not possible.",
        "- Per-video source-type labels are not present; any source category must remain a dataset-level proxy.",
        "- Explicit VLM prompt, VAL prompt, VAU prompt, and full video-to-score inference pipeline are not present.",
        "- Some future inputs may use seconds or clip/window units; current verified files use frame-like integer keys.",
    ]
    if not prediction_intervals:
        lines.append("- No external prediction interval file was supplied for this stage; prediction intervals are generated from anomaly score thresholds.")
    lines.extend([
        "",
        "## Recommended manual confirmations",
        "",
        "- Confirm whether frame keys in caption JSON and refined score JSON share the same sampling stride for every dataset.",
        "- Confirm whether XD-Violence long IDs should be matched exactly or by prefix in all future scripts.",
        "- Confirm acceptable H4 window sizes for gap-level enrichment analysis.",
    ])
    (output_dir / "missing_inputs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h4-candidates", default="outputs/26-07-09-00-35-caption_boundary_screen/h4_strong_candidates.csv")
    parser.add_argument("--prediction-intervals", default="")
    parser.add_argument("--prediction-gaps", default="")
    parser.add_argument("--h4-window", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h4_candidates = Path(args.h4_candidates)
    prediction_intervals = Path(args.prediction_intervals) if args.prediction_intervals else None
    prediction_gaps = Path(args.prediction_gaps) if args.prediction_gaps else None

    write_inventory(output_dir, h4_candidates, prediction_intervals)
    build_schema_guess(output_dir, h4_candidates, prediction_intervals)
    build_canonical_timeline(output_dir, h4_candidates, args.h4_window)
    build_h4_diagnostic(output_dir, h4_candidates, prediction_gaps, args.h4_window)
    write_missing_inputs(output_dir, prediction_intervals)


if __name__ == "__main__":
    main()
