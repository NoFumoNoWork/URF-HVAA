import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import load_scores, safe_name, score_metadata  # noqa: E402


TASK_NAME = "caption-boundary-screen"

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

TRANSITION_KEYWORDS = [
    "scene cuts", "scene cut", "cuts to", "cut to",
    "scene transitions", "scene transition", "transitions to", "transition to",
    "scene shifts", "scene shift", "shifts to",
    "scene changes", "scene changes to", "changes to",
    "camera angle changes", "angle changes",
    "camera pans", "camera zooms", "camera focuses", "focus shifts",
    "the next scene", "next scene",
    "subsequent scenes", "final frames", "video ends",
    "cuts back", "returns to", "back to",
]

RETURN_CUT_KEYWORDS = ["cuts back", "returns to", "back to"]

MULTI_SCENE_MARKERS = [
    "followed by", "then", "next", "subsequent", "finally",
    "final frames", "video begins", "video opens", "video ends",
    "the scene", "camera then",
]

LOCATION_KEYWORDS = [
    "room", "street", "market", "alley", "stairs", "staircase", "tower",
    "helicopter", "car", "bus", "van", "road", "bridge", "building",
    "restaurant", "bar", "office", "control room", "pool", "balcony",
    "field", "hallway", "parking lot",
]

OBJECT_KEYWORDS = [
    "gun", "knife", "weapon", "chain", "ball", "car", "bus", "van",
    "helicopter", "screen", "map", "box", "bottle", "fire", "smoke",
]

RISK_KEYWORDS = [
    "fight", "fighting", "attack", "chase", "chasing", "running",
    "gun", "shooting", "explosion", "crash", "fire", "smoke",
    "injured", "blood", "distress", "screaming", "yelling", "panic",
    "threat", "weapon", "stab", "choke", "falling",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "her", "his", "in", "into", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "them", "there", "they", "this",
    "to", "with", "while", "who", "where", "which", "was", "were", "will",
}

CAPTION_BOILERPLATE = {
    "video", "shows", "showing", "scene", "frames", "frame", "camera",
    "person", "people", "man", "woman", "appears", "seen", "visible",
}


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


def normalize_stem(stem):
    return re.sub(r"\(\d+\)$", "", stem).strip()


def json_index(directory):
    result = {}
    if not directory:
        return result
    directory = Path(directory)
    if not directory.exists():
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
        parts = [str(value[key]) for key in preferred if key in value and value[key] is not None]
        return " ".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " ".join(stringify_caption(item) for item in value)
    return "" if value is None else str(value)


def load_captions(path):
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        return []
    rows = []
    for key, value in items:
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        caption = stringify_caption(value).strip()
        rows.append({"frame": frame, "caption": caption})
    return sorted(rows, key=lambda row: row["frame"])


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
        if not video_col or not start_col or not end_col:
            return rows
        for row in reader:
            try:
                start = int(float(row[start_col]))
                end = int(float(row[end_col]))
            except (TypeError, ValueError):
                continue
            if end > start:
                video_id = normalize_stem(Path(str(row[video_col]).strip()).stem)
                rows[video_id].append({"start": start, "end": end})
    return rows


def parse_gt_lines(path):
    rows = defaultdict(list)
    if not path or not path.exists():
        return rows
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
        for start, end in zip(numeric[0::2], numeric[1::2]):
            if start == -1 or end == -1:
                break
            if end > start:
                rows[video_id].append({"start": start, "end": end})
    return rows


def load_gt(path):
    if not path:
        return defaultdict(list)
    path = Path(path)
    if not path.exists():
        return defaultdict(list)
    parsed = parse_gt_csv(path) if path.suffix.lower() == ".csv" else defaultdict(list)
    return parsed if parsed else parse_gt_lines(path)


def tokens(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and w not in CAPTION_BOILERPLATE and len(w) > 1]


def cosine_from_counters(left, right):
    if not left or not right:
        return None
    shared = set(left) & set(right)
    dot = sum(left[k] * right[k] for k in shared)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


def jaccard(left_tokens, right_tokens):
    left = set(left_tokens)
    right = set(right_tokens)
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def keyword_matches(text, keywords):
    low = (text or "").lower()
    return sorted({kw for kw in keywords if kw in low})


def keyword_set(texts, keywords):
    joined = " ".join(texts)
    return set(keyword_matches(joined, keywords))


def mean(values):
    usable = [float(v) for v in values if v is not None]
    return float(np.mean(usable)) if usable else None


def mad_threshold(values, k, fallback):
    usable = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    if len(usable) == 0:
        return fallback
    med = float(np.median(usable))
    mad = float(np.median(np.abs(usable - med)))
    if mad == 0:
        std = float(np.std(usable))
        return med + k * std
    return med + k * mad


def global_or_adaptive_flags(rows, value_key, flag_key, mode, global_threshold, k):
    threshold = global_threshold if mode == "global" else mad_threshold([r.get(value_key) for r in rows], k, global_threshold)
    for row in rows:
        value = row.get(value_key)
        row[flag_key] = bool(value is not None and value >= threshold)
        row[f"{flag_key}_threshold"] = threshold


def score_context(scores, frame, ordered_frames, idx, window_size):
    pre_frames = ordered_frames[max(0, idx - window_size):idx]
    post_frames = ordered_frames[idx + 1:idx + 1 + window_size]
    pre_scores = [scores[f] for f in pre_frames if f in scores]
    post_scores = [scores[f] for f in post_frames if f in scores]
    pre_mean = mean(pre_scores)
    post_mean = mean(post_scores)
    return pre_mean, post_mean, pre_mean - post_mean if pre_mean is not None and post_mean is not None else None


def gt_context(frame, intervals, margin):
    inside = False
    nearest_start = ""
    nearest_end = ""
    best_dist = None
    near_boundary = False
    for item in intervals:
        if item["start"] <= frame <= item["end"]:
            inside = True
        for boundary in (item["start"], item["end"]):
            dist = abs(frame - boundary)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                nearest_start = item["start"]
                nearest_end = item["end"]
            if dist <= margin:
                near_boundary = True
    return inside, near_boundary, nearest_start, nearest_end, best_dist if best_dist is not None else ""


def sentence_and_clause_features(caption):
    sentence_count = len([part for part in re.split(r"[.!?]+", caption or "") if part.strip()])
    clause_parts = re.split(r",|;|\bthen\b|\bfollowed by\b", caption or "", flags=re.IGNORECASE)
    clause_count = len([part for part in clause_parts if part.strip()])
    return sentence_count, clause_count


def len_spike(rows, idx, window_size):
    lo = max(0, idx - window_size)
    hi = min(len(rows), idx + window_size + 1)
    local = [rows[i]["caption_len_tokens"] for i in range(lo, hi) if i != idx]
    local_mean = mean(local)
    if not local_mean:
        return "", False
    ratio = rows[idx]["caption_len_tokens"] / local_mean if local_mean else None
    return ratio, bool(ratio is not None and ratio >= 1.5)


def load_sbert(args, warnings):
    if not args.use_sbert:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        warnings.append(f"sentence_transformers unavailable; semantic features disabled: {exc}")
        return None
    try:
        device = None if args.device == "auto" else args.device
        return SentenceTransformer(args.sbert_model, device=device)
    except Exception as exc:
        warnings.append(f"failed to load SBERT model `{args.sbert_model}`; semantic features disabled: {exc}")
        return None


def embedding_cosine(left, right):
    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return None
    return float(np.dot(left, right) / denom)


def add_semantic_features(rows, model, args):
    if model is None or not rows:
        for row in rows:
            row["semantic_sim_adjacent"] = None
            row["semantic_drop_adjacent"] = None
            row["semantic_sim_block"] = None
            row["semantic_drop_block"] = None
            row["semantic_boundary_flag"] = False
            row["semantic_boundary_flag_threshold"] = ""
        return
    captions = [row["caption"] for row in rows]
    embeddings = model.encode(captions, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    for idx, row in enumerate(rows):
        if idx == 0:
            row["semantic_sim_adjacent"] = None
            row["semantic_drop_adjacent"] = None
        else:
            sim = embedding_cosine(embeddings[idx - 1], embeddings[idx])
            row["semantic_sim_adjacent"] = sim
            row["semantic_drop_adjacent"] = 1 - sim if sim is not None else None
        left = embeddings[max(0, idx - args.window_size):idx]
        right = embeddings[idx:min(len(rows), idx + args.window_size)]
        if len(left) and len(right):
            sim = embedding_cosine(np.mean(left, axis=0), np.mean(right, axis=0))
        else:
            sim = None
        row["semantic_sim_block"] = sim
        row["semantic_drop_block"] = 1 - sim if sim is not None else None
    global_or_adaptive_flags(
        rows,
        "semantic_drop_block",
        "semantic_boundary_flag",
        args.semantic_threshold_mode,
        args.semantic_drop_threshold,
        args.semantic_mad_k,
    )


def candidate_type_and_score(row, args):
    score = 0.0
    score += 2.0 if row["has_explicit_transition"] else 0.0
    score += 1.0 if row["has_multi_scene_compression"] else 0.0
    score += 1.0 if row["lexical_boundary_flag"] else 0.0
    score += 1.0 if row["semantic_boundary_flag"] else 0.0
    score += 1.0 if row["location_shift_flag"] else 0.0
    score += 1.0 if row["object_shift_flag"] else 0.0
    score += 1.0 if row["len_spike_flag"] else 0.0
    score += 1.0 if row["risk_gain_or_drop_flag"] else 0.0
    score += 1.0 if row["has_return_cut"] else 0.0

    types = []
    if row["has_explicit_transition"]:
        types.append("explicit_transition_boundary")
    if row["lexical_boundary_flag"] and not row["has_explicit_transition"]:
        types.append("lexical_topic_boundary")
    if row["semantic_boundary_flag"] and not row["has_explicit_transition"]:
        types.append("semantic_boundary")
    if row["has_multi_scene_compression"] and row["len_spike_flag"]:
        types.append("multi_scene_compression_boundary")

    possible_forgetting = (
        score >= args.boundary_score_threshold
        and row["pre_score_mean"] is not None
        and row["post_score_mean"] is not None
        and row["score_drop"] is not None
        and row["pre_score_mean"] >= args.h4_high_score_threshold
        and row["post_score_mean"] <= args.h4_low_score_threshold
        and row["score_drop"] >= args.h4_drop_threshold
    )
    if possible_forgetting:
        types.append("possible_context_forgetting")

    event_onset = (
        (row["lexical_boundary_flag"] or row["semantic_boundary_flag"])
        and row["location_shift_count"] == 0
        and row["risk_gain_count"] > 0
    )
    if event_onset:
        types.append("event_onset_not_h4")
    if score >= 2 and not types:
        types.append("composite_context_boundary")

    if possible_forgetting or score >= 5:
        confidence = "strong"
    elif score >= 3:
        confidence = "medium"
    elif score >= 2:
        confidence = "weak"
    else:
        confidence = ""
    return score, types, confidence


def build_video_rows(dataset, video_id, captions, scores, intervals, args, sbert_model):
    frames = [row["frame"] for row in captions]
    rows = []
    for idx, item in enumerate(captions):
        caption = item["caption"]
        frame = item["frame"]
        left_items = captions[max(0, idx - args.window_size):idx]
        right_items = captions[idx:min(len(captions), idx + args.window_size)]
        left_texts = [x["caption"] for x in left_items]
        right_texts = [x["caption"] for x in right_items]
        left_tokens = [tok for text in left_texts for tok in tokens(text)]
        right_tokens = [tok for text in right_texts for tok in tokens(text)]
        lex_sim = cosine_from_counters(Counter(left_tokens), Counter(right_tokens))
        jac_sim = jaccard(left_tokens, right_tokens)

        transition_matches = keyword_matches(caption, TRANSITION_KEYWORDS)
        multi_matches = keyword_matches(caption, MULTI_SCENE_MARKERS)
        sentence_count, clause_count = sentence_and_clause_features(caption)
        caption_token_count = len(re.findall(r"\S+", caption or ""))
        caption_char_count = len(caption or "")
        pre_mean, post_mean, drop = score_context(scores, frame, frames, idx, args.window_size) if scores else (None, None, None)
        inside_gt, near_gt, nearest_start, nearest_end, dist = gt_context(frame, intervals, args.near_gt_margin) if intervals else (False, False, "", "", "")

        left_locations = keyword_set(left_texts, LOCATION_KEYWORDS)
        right_locations = keyword_set(right_texts, LOCATION_KEYWORDS)
        left_objects = keyword_set(left_texts, OBJECT_KEYWORDS)
        right_objects = keyword_set(right_texts, OBJECT_KEYWORDS)
        left_risk = keyword_set(left_texts, RISK_KEYWORDS)
        right_risk = keyword_set(right_texts, RISK_KEYWORDS)
        risk_drop = len(left_risk - right_risk)
        risk_gain = len(right_risk - left_risk)
        row = {
            "dataset": dataset,
            "video_id": video_id,
            "frame": frame,
            "time_sec": frame / args.fps,
            "caption": caption,
            "caption_len_tokens": caption_token_count,
            "caption_len_chars": caption_char_count,
            "explicit_transition_count": len(transition_matches),
            "matched_transition_keywords": ";".join(transition_matches),
            "has_explicit_transition": bool(transition_matches),
            "has_return_cut": bool(set(transition_matches) & set(RETURN_CUT_KEYWORDS)),
            "multi_scene_marker_count": len(multi_matches),
            "matched_multi_scene_markers": ";".join(multi_matches),
            "has_multi_scene_compression": bool(multi_matches),
            "number_of_sentences": sentence_count,
            "number_of_clauses_rough": clause_count,
            "compression_score": len(multi_matches) + 0.25 * max(0, sentence_count - 1) + 0.15 * max(0, clause_count - 1),
            "lexical_similarity": lex_sim,
            "lexical_drop": 1 - lex_sim if lex_sim is not None else None,
            "jaccard_similarity": jac_sim,
            "jaccard_drop": 1 - jac_sim if jac_sim is not None else None,
            "location_set_left": ";".join(sorted(left_locations)),
            "location_set_right": ";".join(sorted(right_locations)),
            "location_shift_count": len(left_locations ^ right_locations),
            "location_shift_flag": bool(left_locations ^ right_locations),
            "object_set_left": ";".join(sorted(left_objects)),
            "object_set_right": ";".join(sorted(right_objects)),
            "object_shift_count": len(left_objects ^ right_objects),
            "object_shift_flag": bool(left_objects ^ right_objects),
            "risk_set_left": ";".join(sorted(left_risk)),
            "risk_set_right": ";".join(sorted(right_risk)),
            "risk_drop_count": risk_drop,
            "risk_gain_count": risk_gain,
            "risk_gain_or_drop_flag": bool(risk_drop or risk_gain),
            "pre_score_mean": pre_mean,
            "post_score_mean": post_mean,
            "score_drop": drop,
            "inside_gt": inside_gt,
            "near_gt_boundary": near_gt,
            "nearest_gt_start": nearest_start,
            "nearest_gt_end": nearest_end,
            "distance_to_gt_boundary": dist,
        }
        rows.append(row)

    global_or_adaptive_flags(
        rows,
        "lexical_drop",
        "lexical_boundary_flag",
        args.lexical_threshold_mode,
        args.lexical_drop_threshold,
        args.lexical_mad_k,
    )
    add_semantic_features(rows, sbert_model, args)
    for idx, row in enumerate(rows):
        ratio, flag = len_spike(rows, idx, args.window_size)
        row["local_len_mean"] = "" if ratio == "" else row["caption_len_tokens"] / ratio
        row["len_spike_ratio"] = ratio
        row["len_spike_flag"] = flag
        boundary_score, types, confidence = candidate_type_and_score(row, args)
        row["boundary_score"] = boundary_score
        row["candidate_types"] = ";".join(types)
        row["confidence"] = confidence
    return rows


DEBUG_FIELDS = [
    "dataset", "video_id", "frame", "time_sec", "caption", "caption_len_tokens", "caption_len_chars",
    "explicit_transition_count", "matched_transition_keywords", "has_explicit_transition",
    "multi_scene_marker_count", "matched_multi_scene_markers", "has_multi_scene_compression",
    "number_of_sentences", "number_of_clauses_rough", "compression_score",
    "lexical_similarity", "lexical_drop", "jaccard_similarity", "jaccard_drop",
    "lexical_boundary_flag", "lexical_boundary_flag_threshold",
    "semantic_sim_adjacent", "semantic_drop_adjacent", "semantic_sim_block", "semantic_drop_block",
    "semantic_boundary_flag", "semantic_boundary_flag_threshold",
    "location_set_left", "location_set_right", "location_shift_count", "location_shift_flag",
    "object_set_left", "object_set_right", "object_shift_count", "object_shift_flag",
    "risk_set_left", "risk_set_right", "risk_drop_count", "risk_gain_count", "risk_gain_or_drop_flag",
    "local_len_mean", "len_spike_ratio", "len_spike_flag",
    "pre_score_mean", "post_score_mean", "score_drop",
    "inside_gt", "near_gt_boundary", "nearest_gt_start", "nearest_gt_end", "distance_to_gt_boundary",
    "boundary_score", "candidate_types", "confidence",
]

CANDIDATE_FIELDS = [
    "dataset", "video_id", "frame", "time_sec", "candidate_types", "boundary_score", "confidence",
    "caption_prev", "caption_current", "caption_next",
    "matched_transition_keywords", "explicit_transition_count", "multi_scene_marker_count",
    "lexical_drop", "jaccard_drop", "semantic_drop_adjacent", "semantic_drop_block",
    "location_set_left", "location_set_right", "object_set_left", "object_set_right",
    "risk_set_left", "risk_set_right", "pre_score_mean", "post_score_mean", "score_drop",
    "inside_gt", "near_gt_boundary", "notes",
]

SUMMARY_FIELDS = [
    "dataset", "video_id", "num_windows", "num_candidates", "num_explicit_transition",
    "num_lexical_boundary", "num_semantic_boundary", "num_possible_context_forgetting",
    "num_event_onset_not_h4", "mean_lexical_drop", "max_lexical_drop",
    "mean_semantic_drop", "max_semantic_drop", "mean_boundary_score", "max_boundary_score",
]

DATASET_SUMMARY_FIELDS = [
    "dataset", "num_videos", "num_windows", "num_candidates", "num_explicit_transition",
    "num_lexical_boundary", "num_semantic_boundary", "num_possible_context_forgetting",
    "num_event_onset_not_h4", "mean_candidates_per_video", "mean_lexical_drop",
    "max_lexical_drop", "mean_semantic_drop", "max_semantic_drop", "mean_boundary_score",
    "max_boundary_score",
]


def clean_for_csv(row):
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, (int, str)):
            cleaned[key] = value
        elif isinstance(value, float):
            cleaned[key] = rounded(value)
        elif value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = value
    return cleaned


def write_csv_sig(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_for_csv(row))


def candidate_rows_from_debug(rows):
    candidates = []
    for idx, row in enumerate(rows):
        if row["boundary_score"] < 2 and not row["candidate_types"]:
            continue
        notes = []
        if "event_onset_not_h4" in row["candidate_types"]:
            notes.append("risk/event onset without location shift; verify before treating as H4")
        if "possible_context_forgetting" in row["candidate_types"]:
            notes.append("score-drop verifier triggered")
        candidates.append(
            {
                "video_id": row["video_id"],
                "dataset": row["dataset"],
                "frame": row["frame"],
                "time_sec": row["time_sec"],
                "candidate_types": row["candidate_types"],
                "boundary_score": row["boundary_score"],
                "confidence": row["confidence"],
                "caption_prev": rows[idx - 1]["caption"] if idx > 0 else "",
                "caption_current": row["caption"],
                "caption_next": rows[idx + 1]["caption"] if idx + 1 < len(rows) else "",
                "matched_transition_keywords": row["matched_transition_keywords"],
                "explicit_transition_count": row["explicit_transition_count"],
                "multi_scene_marker_count": row["multi_scene_marker_count"],
                "lexical_drop": row["lexical_drop"],
                "jaccard_drop": row["jaccard_drop"],
                "semantic_drop_adjacent": row["semantic_drop_adjacent"],
                "semantic_drop_block": row["semantic_drop_block"],
                "location_set_left": row["location_set_left"],
                "location_set_right": row["location_set_right"],
                "object_set_left": row["object_set_left"],
                "object_set_right": row["object_set_right"],
                "risk_set_left": row["risk_set_left"],
                "risk_set_right": row["risk_set_right"],
                "pre_score_mean": row["pre_score_mean"],
                "post_score_mean": row["post_score_mean"],
                "score_drop": row["score_drop"],
                "inside_gt": row["inside_gt"],
                "near_gt_boundary": row["near_gt_boundary"],
                "notes": "; ".join(notes),
            }
        )
    return candidates


def summarize_video(dataset, video_id, rows, candidates):
    lexical = [r["lexical_drop"] for r in rows if r["lexical_drop"] is not None]
    semantic = [r["semantic_drop_block"] for r in rows if r["semantic_drop_block"] is not None]
    scores = [r["boundary_score"] for r in rows]
    type_counter = Counter()
    for row in candidates:
        type_counter.update([t for t in row["candidate_types"].split(";") if t])
    return {
        "dataset": dataset,
        "video_id": video_id,
        "num_windows": len(rows),
        "num_candidates": len(candidates),
        "num_explicit_transition": type_counter["explicit_transition_boundary"],
        "num_lexical_boundary": type_counter["lexical_topic_boundary"],
        "num_semantic_boundary": type_counter["semantic_boundary"],
        "num_possible_context_forgetting": type_counter["possible_context_forgetting"],
        "num_event_onset_not_h4": type_counter["event_onset_not_h4"],
        "mean_lexical_drop": mean(lexical),
        "max_lexical_drop": max(lexical) if lexical else "",
        "mean_semantic_drop": mean(semantic),
        "max_semantic_drop": max(semantic) if semantic else "",
        "mean_boundary_score": mean(scores),
        "max_boundary_score": max(scores) if scores else "",
    }


def summarize_datasets(video_summaries, candidates):
    by_dataset = defaultdict(list)
    for row in video_summaries:
        by_dataset[row["dataset"]].append(row)
    type_by_dataset = defaultdict(Counter)
    for row in candidates:
        type_by_dataset[row["dataset"]].update([t for t in row["candidate_types"].split(";") if t])
    result = []
    for dataset, rows in sorted(by_dataset.items()):
        num_videos = len(rows)
        num_candidates = sum(int(row["num_candidates"]) for row in rows)
        lexical_means = [row["mean_lexical_drop"] for row in rows if row["mean_lexical_drop"] != ""]
        semantic_means = [row["mean_semantic_drop"] for row in rows if row["mean_semantic_drop"] != ""]
        boundary_means = [row["mean_boundary_score"] for row in rows if row["mean_boundary_score"] != ""]
        type_counts = type_by_dataset[dataset]
        result.append(
            {
                "dataset": dataset,
                "num_videos": num_videos,
                "num_windows": sum(int(row["num_windows"]) for row in rows),
                "num_candidates": num_candidates,
                "num_explicit_transition": type_counts["explicit_transition_boundary"],
                "num_lexical_boundary": type_counts["lexical_topic_boundary"],
                "num_semantic_boundary": type_counts["semantic_boundary"],
                "num_possible_context_forgetting": type_counts["possible_context_forgetting"],
                "num_event_onset_not_h4": type_counts["event_onset_not_h4"],
                "mean_candidates_per_video": num_candidates / num_videos if num_videos else "",
                "mean_lexical_drop": mean(lexical_means),
                "max_lexical_drop": max((row["max_lexical_drop"] for row in rows if row["max_lexical_drop"] != ""), default=""),
                "mean_semantic_drop": mean(semantic_means),
                "max_semantic_drop": max((row["max_semantic_drop"] for row in rows if row["max_semantic_drop"] != ""), default=""),
                "mean_boundary_score": mean(boundary_means),
                "max_boundary_score": max((row["max_boundary_score"] for row in rows if row["max_boundary_score"] != ""), default=""),
            }
        )
    return result


def plot_video(dataset, video_id, rows, scores, intervals, candidates, output_path):
    if not rows:
        return
    frames = [r["frame"] for r in rows]
    boundary = [r["boundary_score"] for r in rows]
    has_scores = bool(scores)
    fig, ax1 = plt.subplots(figsize=(16, 6))
    if intervals:
        for interval in intervals:
            ax1.axvspan(interval["start"], interval["end"], color="#d95f02", alpha=0.12, label="GT interval")
    if has_scores:
        score_values = [scores.get(frame, np.nan) for frame in frames]
        ax1.plot(frames, score_values, color="#377eb8", linewidth=1.1, label="score")
        ax1.set_ylabel("anomaly score")
        ax1.set_ylim(-0.03, 1.05)
        ax2 = ax1.twinx()
    else:
        ax2 = ax1
        ax1.set_ylabel("boundary score")
    ax2.plot(frames, boundary, color="#1b9e77", linewidth=1.8, label="boundary score")
    ax2.set_ylabel("boundary score")
    ax2.set_ylim(-0.2, max(6.0, max(boundary) + 1 if boundary else 6.0))

    marker_specs = [
        ("explicit_transition_boundary", "^", "#e7298a", "explicit"),
        ("lexical_topic_boundary", "o", "#ff7f00", "lexical"),
        ("semantic_boundary", "s", "#4daf4a", "semantic"),
        ("possible_context_forgetting", "v", "#984ea3", "possible forgetting"),
    ]
    for candidate_type, marker, color, label in marker_specs:
        xs = [int(row["frame"]) for row in candidates if candidate_type in row["candidate_types"]]
        ys = [next((r["boundary_score"] for r in rows if r["frame"] == x), 0) for x in xs]
        if xs:
            ax2.scatter(xs, ys, marker=marker, s=65, color=color, edgecolor="black", linewidth=0.4, label=label, zorder=5)

    ax1.set_title(f"{dataset} | {video_id} | caption boundaries={len(candidates)}", fontsize=11)
    ax1.set_xlabel("frame")
    ax1.grid(alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    dedup = dict(zip(labels1 + labels2, handles1 + handles2))
    ax1.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(output_dir, args, summary, candidates, type_counts, warnings, sbert_used):
    report = output_dir / "caption_boundary_screen_report.md"
    top = sorted(candidates, key=lambda row: (float(row.get("boundary_score") or 0), float(row.get("lexical_drop") or 0)), reverse=True)[:20]
    lines = [
        "# Caption Boundary Screen Report",
        "",
        "## Inputs",
        "",
        f"- data_root: `{args.data_root or ''}`",
        f"- caption_dir: `{args.caption_dir or ''}`",
        f"- score_dir: `{args.score_dir or ''}`",
        f"- gt_file: `{args.gt_file or ''}`",
        f"- use_sbert requested: `{args.use_sbert}`",
        f"- SBERT used: `{sbert_used}`",
        "",
        "## Summary",
        "",
        f"- processed videos: {summary['processed_videos']}",
        f"- skipped videos: {summary['skipped_videos']}",
        f"- datasets: {', '.join(summary.get('datasets', []))}",
        f"- total candidates: {len(candidates)}",
        f"- warnings: {len(warnings)}",
        "",
        "## Candidate Type Counts",
        "",
    ]
    if type_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(type_counts.items()))
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Top 20 Strongest Candidates",
            "",
            "| dataset | video_id | frame | score | confidence | types | lexical_drop | semantic_drop_block | score_drop |",
            "|---|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in top:
        lines.append(
            f"| {row['dataset']} | {row['video_id']} | {row['frame']} | {rounded(row['boundary_score'], 3)} | {row['confidence']} | "
            f"{row['candidate_types']} | {rounded(row['lexical_drop'], 3)} | {rounded(row['semantic_drop_block'], 3)} | {rounded(row['score_drop'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Explicit boundary cues detect captions that directly mention cuts, scene transitions, camera shifts, or returns to prior context.",
            "- Lexical cohesion uses left/right caption blocks and bag-of-words cosine/Jaccard drops as a lightweight TextTiling-style topic-shift signal.",
            "- Optional SBERT computes adjacent-caption and left/right-block semantic drops when `sentence_transformers` is available.",
            "- Location, object, and risk keyword shifts help separate scene/context switches from ordinary event onset.",
            "- Possible context forgetting requires a high combined boundary score plus a high-to-low anomaly score drop.",
            "",
            "## Notes",
            "",
            "- This is caption-level semantic boundary screening, not direct visual shot boundary detection.",
            "- Detected boundary candidates should be manually verified or compared with visual shot-boundary detection when raw videos are available.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:80])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_jobs(args):
    if args.data_root:
        root = Path(args.data_root)
        jobs = []
        for cfg in DATASET_CONFIGS:
            caption_dir = root / cfg["caption_dir"]
            if not caption_dir.exists():
                continue
            jobs.append(
                {
                    "dataset": cfg["dataset"],
                    "caption_dir": caption_dir,
                    "score_dir": root / cfg["score_dir"],
                    "gt_file": root / cfg["gt_file"],
                }
            )
        return jobs
    if not args.caption_dir:
        raise ValueError("Either --data_root or --caption_dir is required.")
    return [
        {
            "dataset": args.dataset_name,
            "caption_dir": Path(args.caption_dir),
            "score_dir": Path(args.score_dir) if args.score_dir else None,
            "gt_file": Path(args.gt_file) if args.gt_file else None,
        }
    ]


def process(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{datetime.now():%y-%m-%d-%H-%M}-{TASK_NAME}"
    plot_dir = output_dir / "plots"
    script_dir = output_dir / "scripts"
    warnings = []
    jobs = discover_jobs(args)
    if not jobs:
        raise ValueError(f"No caption datasets found for data_root={args.data_root!r}")
    sbert_model = load_sbert(args, warnings)
    sbert_used = sbert_model is not None
    all_debug = []
    all_candidates = []
    summaries = []
    plot_cache = {}
    processed = 0
    skipped = 0
    for job in jobs:
        dataset = job["dataset"]
        caption_files = json_index(job["caption_dir"])
        score_files = json_index(job["score_dir"]) if job["score_dir"] else {}
        gt_rows = load_gt(job["gt_file"]) if job["gt_file"] else defaultdict(list)
        video_ids = sorted(caption_files)
        if args.max_videos:
            video_ids = video_ids[: args.max_videos]
        for video_id in video_ids:
            caption_path = caption_files.get(video_id)
            captions = load_captions(caption_path)
            if not captions:
                skipped += 1
                warnings.append(f"{dataset}/{video_id}: missing or unreadable captions")
                continue
            score_path = score_files.get(video_id) if score_files else None
            scores = load_scores(score_path) if score_path else {}
            intervals = gt_rows.get(video_id, [])
            rows = build_video_rows(dataset, video_id, captions, scores, intervals, args, sbert_model)
            candidates = candidate_rows_from_debug(rows)
            all_debug.extend(rows)
            all_candidates.extend(candidates)
            summaries.append(summarize_video(dataset, video_id, rows, candidates))
            if args.plot:
                out = plot_dir / dataset / f"{safe_name(video_id)}_caption_boundary.png"
                plot_video(dataset, video_id, rows, scores, intervals, candidates, out)
            elif args.plot_top_n and candidates:
                max_score = max(float(row["boundary_score"] or 0) for row in candidates)
                plot_cache[(dataset, video_id)] = {
                    "score": max_score,
                    "rows": rows,
                    "scores": scores,
                    "intervals": intervals,
                    "candidates": candidates,
                }
            processed += 1

    plotted = 0
    if args.plot_top_n and not args.plot:
        top_items = sorted(plot_cache.items(), key=lambda item: item[1]["score"], reverse=True)[: args.plot_top_n]
        for (dataset, video_id), item in top_items:
            out = plot_dir / dataset / f"{safe_name(video_id)}_caption_boundary.png"
            plot_video(dataset, video_id, item["rows"], item["scores"], item["intervals"], item["candidates"], out)
            plotted += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_sig(output_dir / "caption_boundary_debug.csv", all_debug, DEBUG_FIELDS)
    write_csv_sig(output_dir / "caption_boundary_candidates.csv", all_candidates, CANDIDATE_FIELDS)
    write_csv_sig(output_dir / "caption_boundary_summary.csv", summaries, SUMMARY_FIELDS)
    dataset_rows = summarize_datasets(summaries, all_candidates)
    write_csv_sig(output_dir / "caption_boundary_dataset_summary.csv", dataset_rows, DATASET_SUMMARY_FIELDS)
    write_csv_sig(output_dir / "warnings.csv", [{"warning": item} for item in warnings], ["warning"])
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    type_counts = Counter()
    for row in all_candidates:
        type_counts.update([t for t in row["candidate_types"].split(";") if t])
    run_summary = {
        "processed_videos": processed,
        "skipped_videos": skipped,
        "datasets": [job["dataset"] for job in jobs],
        "total_candidates": len(all_candidates),
        "candidate_type_counts": dict(type_counts),
        "plot_count": plotted if args.plot_top_n and not args.plot else processed if args.plot else 0,
        "sbert_used": sbert_used,
        "warnings": warnings,
        "output_dir": str(output_dir),
    }
    (output_dir / "caption_boundary_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir, args, run_summary, all_candidates, type_counts, warnings, sbert_used)
    print(f"processed videos: {processed}")
    print(f"skipped videos: {skipped}")
    print(f"total candidates: {len(all_candidates)}")
    print(f"datasets: {', '.join(run_summary['datasets'])}")
    print(f"plots generated: {run_summary['plot_count']}")
    print(f"SBERT used: {sbert_used}")
    print(f"output dir: {output_dir}")
    print(f"candidates: {output_dir / 'caption_boundary_candidates.csv'}")
    print(f"summary: {output_dir / 'caption_boundary_summary.csv'}")
    print(f"debug: {output_dir / 'caption_boundary_debug.csv'}")
    if args.plot or args.plot_top_n:
        print(f"plots: {plot_dir}")
    return run_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Caption-level semantic boundary screening for H4 context shifts.")
    parser.add_argument("--data_root", "--data-root", dest="data_root", default="", help="Run all known datasets under this root, e.g. data.")
    parser.add_argument("--dataset_name", "--dataset-name", dest="dataset_name", default="custom", help="Dataset label for single-directory mode.")
    parser.add_argument("--caption_dir", "--caption-dir", dest="caption_dir", default="")
    parser.add_argument("--score_dir", "--score-dir", dest="score_dir", default="")
    parser.add_argument("--gt_file", "--gt-file", dest="gt_file", default="")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--use_sbert", "--use-sbert", dest="use_sbert", action="store_true")
    parser.add_argument("--sbert_model", "--sbert-model", dest="sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window_size", "--window-size", dest="window_size", type=int, default=3)
    parser.add_argument("--lexical_threshold_mode", "--lexical-threshold-mode", dest="lexical_threshold_mode", choices=["global", "adaptive"], default="adaptive")
    parser.add_argument("--semantic_threshold_mode", "--semantic-threshold-mode", dest="semantic_threshold_mode", choices=["global", "adaptive"], default="adaptive")
    parser.add_argument("--lexical_drop_threshold", "--lexical-drop-threshold", dest="lexical_drop_threshold", type=float, default=0.65)
    parser.add_argument("--semantic_drop_threshold", "--semantic-drop-threshold", dest="semantic_drop_threshold", type=float, default=0.45)
    parser.add_argument("--lexical_mad_k", "--lexical-mad-k", dest="lexical_mad_k", type=float, default=2.0)
    parser.add_argument("--semantic_mad_k", "--semantic-mad-k", dest="semantic_mad_k", type=float, default=2.0)
    parser.add_argument("--boundary_score_threshold", "--boundary-score-threshold", dest="boundary_score_threshold", type=float, default=3.0)
    parser.add_argument("--h4_high_score_threshold", "--h4-high-score-threshold", dest="h4_high_score_threshold", type=float, default=0.6)
    parser.add_argument("--h4_low_score_threshold", "--h4-low-score-threshold", dest="h4_low_score_threshold", type=float, default=0.3)
    parser.add_argument("--h4_drop_threshold", "--h4-drop-threshold", dest="h4_drop_threshold", type=float, default=0.3)
    parser.add_argument("--near_gt_margin", "--near-gt-margin", dest="near_gt_margin", type=int, default=48)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_top_n", "--plot-top-n", dest="plot_top_n", type=int, default=0, help="When --plot is not set, draw only the top-N videos by max boundary score.")
    parser.add_argument("--max_videos", "--max-videos", dest="max_videos", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
