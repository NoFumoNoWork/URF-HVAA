import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.h1_h4_boundary_aware_postprocess import (  # noqa: E402
    DATASET_CONFIGS,
    as_float,
    as_int,
    evaluate_method,
    intersect_duration,
    load_all_inputs,
    load_h4_candidates,
    merge_ranges,
    normalize_stem,
    raw_intervals_from_scores,
)
from scripts.h1_h4_trigger_diagnostics import (  # noqa: E402
    changed_cases,
    relaxed_merge,
)


TASK_NAME = "vlm-semantic-continuity"

CASE_FIELDS = [
    "dataset", "video_id", "h4_frame", "gap_start", "gap_end", "gap_length",
    "distance_to_h4", "h4_types", "h4_score_drop", "fixed_or_worsened",
    "left_interval_start", "left_interval_end", "right_interval_start",
    "right_interval_end", "left_caption_summary", "gap_caption_summary",
    "right_caption_summary", "caption_before_h4", "caption_at_h4",
    "caption_after_h4", "left_risk_terms", "right_risk_terms",
    "shared_risk_terms", "left_object_terms", "right_object_terms",
    "shared_object_terms", "left_action_terms", "right_action_terms",
    "shared_action_terms", "lexical_overlap", "risk_overlap",
    "object_overlap", "action_overlap", "same_event_rule_label",
    "transition_type_rule_label", "merge_recommendation_rule", "rule_reason",
]

SUMMARY_FIELDS = [
    "case_group", "num_cases", "mean_gap_length", "mean_distance_to_h4",
    "mean_h4_score_drop", "same_event_ratio", "different_event_ratio",
    "unclear_ratio", "merge_recommendation_ratio", "do_not_merge_ratio",
    "recheck_ratio", "mean_lexical_overlap", "mean_risk_overlap",
    "mean_object_overlap", "mean_action_overlap", "top_h4_types",
    "top_transition_type_labels",
]

METRIC_FIELDS = [
    "method", "frame_precision", "frame_recall", "frame_f1",
    "segment_precision", "segment_recall", "segment_f1",
    "avg_segments_per_video", "fragmented_gt_ratio",
    "false_positive_duration", "false_negative_duration", "changed_cases",
    "fixed_cases", "worsened_cases",
]

ACCEPT_REJECT_FIELDS = [
    "dataset", "video_id", "h4_frame", "gap_start", "gap_end", "h4_types",
    "fixed_or_worsened", "same_event_rule_label", "transition_type_rule_label",
    "merge_recommendation_rule", "left_caption_summary",
    "gap_caption_summary", "right_caption_summary", "rule_reason",
]

RISK_TERMS = {
    "abuse", "accident", "aggression", "aggressive", "alarm", "altercation",
    "ambush", "arrest", "assault", "attack", "attacking", "bleeding", "blood",
    "bomb", "brawl", "break", "burning", "chase", "chasing", "collision",
    "crash", "crime", "danger", "dangerous", "dead", "distress", "explosion",
    "fall", "falling", "fight", "fighting", "fire", "flames", "gun", "harm",
    "hit", "injured", "injury", "knife", "panic", "punch", "pushing",
    "riot", "robbery", "running", "scream", "screaming", "shoot", "shooting",
    "smoke", "stab", "struggle", "threat", "violence", "violent", "weapon",
}

OBJECT_TERMS = {
    "ambulance", "bag", "bike", "bus", "car", "corridor", "crowd", "door",
    "floor", "gun", "hall", "hallway", "knife", "motorcycle", "person",
    "people", "police", "road", "room", "street", "vehicle", "weapon",
    "window", "woman", "man", "child", "group",
}

ACTION_TERMS = {
    "approach", "argue", "attack", "carry", "chase", "crash", "drive",
    "fall", "fight", "flee", "grab", "help", "hit", "jump", "kick", "lie",
    "move", "push", "restrain", "run", "shoot", "stand", "strike", "walk",
}

TRANSITION_TERMS = {
    "angle", "camera", "close", "closeup", "cut", "cuts", "pans", "pan",
    "perspective", "shot", "view", "zoom", "zooms",
}

NEW_EVENT_PHRASES = [
    "another scene", "different scene", "different location",
    "new scene", "next scene", "unrelated event", "different event",
    "switches to", "transitions to", "cuts to a different",
]

MULTI_SCENE_PHRASES = [
    "multiple scenes", "several scenes", "various scenes",
    "montage", "compilation", "different clips", "several clips",
    "multiple events", "various events",
]

STOPWORDS = {
    "the", "and", "with", "that", "this", "video", "shows", "seen", "scene",
    "there", "they", "then", "from", "into", "onto", "while", "some", "other",
    "others", "appears", "appear", "wearing", "around", "towards", "toward",
    "their", "them", "where", "which", "being", "also", "more", "over",
}


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


def tokens(text):
    return {
        item
        for item in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(item) > 2 and item not in STOPWORDS
    }


def stem_variants(items):
    out = set(items)
    for item in items:
        if item.endswith("ing") and len(item) > 5:
            out.add(item[:-3])
        if item.endswith("ed") and len(item) > 4:
            out.add(item[:-2])
        if item.endswith("s") and len(item) > 4:
            out.add(item[:-1])
    return out


def term_hits(words, vocabulary):
    words = stem_variants(words)
    vocab = stem_variants(vocabulary)
    return sorted(words & vocab)


def jaccard(left, right):
    left = set(left)
    right = set(right)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def join_terms(items):
    return ";".join(sorted(set(items)))


def has_phrase(text, phrases):
    low = (text or "").lower()
    return any(phrase in low for phrase in phrases)


def caption_summary(text, limit=240):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def semantic_rule(caption_before, caption_current, caption_after, h4_types):
    left_text = " ".join(item for item in [caption_before, caption_current] if item)
    right_text = " ".join(item for item in [caption_current, caption_after] if item)
    all_text = " ".join([caption_before or "", caption_current or "", caption_after or ""])
    left_tokens = tokens(left_text)
    right_tokens = tokens(right_text)
    left_risk = term_hits(left_tokens, RISK_TERMS)
    right_risk = term_hits(right_tokens, RISK_TERMS)
    left_object = term_hits(left_tokens, OBJECT_TERMS)
    right_object = term_hits(right_tokens, OBJECT_TERMS)
    left_action = term_hits(left_tokens, ACTION_TERMS)
    right_action = term_hits(right_tokens, ACTION_TERMS)
    shared_risk = sorted(set(left_risk) & set(right_risk))
    shared_object = sorted(set(left_object) & set(right_object))
    shared_action = sorted(set(left_action) & set(right_action))
    lexical = jaccard(left_tokens, right_tokens)
    risk_overlap = jaccard(left_risk, right_risk)
    object_overlap = jaccard(left_object, right_object)
    action_overlap = jaccard(left_action, right_action)
    transition_hits = term_hits(tokens(all_text), TRANSITION_TERMS)
    explicit_new = has_phrase(all_text, NEW_EVENT_PHRASES)
    multi_scene = has_phrase(all_text, MULTI_SCENE_PHRASES) or "multi_scene_compression" in (h4_types or "")

    if multi_scene and not (shared_risk or shared_action):
        return {
            "same_event_rule_label": "unclear",
            "transition_type_rule_label": "multi_scene_compression",
            "merge_recommendation_rule": "recheck",
            "rule_reason": "Caption/H4 type suggests multi-scene compression without enough shared risk/action evidence.",
            "left_risk": left_risk,
            "right_risk": right_risk,
            "shared_risk": shared_risk,
            "left_object": left_object,
            "right_object": right_object,
            "shared_object": shared_object,
            "left_action": left_action,
            "right_action": right_action,
            "shared_action": shared_action,
            "lexical_overlap": lexical,
            "risk_overlap": risk_overlap,
            "object_overlap": object_overlap,
            "action_overlap": action_overlap,
        }
    if explicit_new and not (shared_risk or (shared_object and shared_action)):
        same = "different_event"
        transition = "new_scene_new_event"
        recommendation = "do_not_merge"
        reason = "Caption contains explicit new/different scene or event cues and lacks shared risk/action continuity."
    elif shared_risk or (shared_action and shared_object):
        same = "same_event"
        transition = "same_event_new_view" if transition_hits else "unclear"
        recommendation = "merge"
        reason = "Left/right captions share risk/action/object semantics, indicating the same abnormal event may continue across the H4 point."
    elif transition_hits and (shared_object or lexical >= 0.18):
        same = "same_event"
        transition = "same_event_new_view"
        recommendation = "merge"
        reason = "Caption mainly signals a camera/view transition while preserving major objects or lexical context."
    elif lexical < 0.08 and not shared_object and not shared_action:
        same = "different_event"
        transition = "new_scene_new_event"
        recommendation = "do_not_merge"
        reason = "Very low lexical continuity and no shared object/action terms suggest a different event."
    elif multi_scene:
        same = "unclear"
        transition = "multi_scene_compression"
        recommendation = "recheck"
        reason = "Multi-scene compression cue is present, but some continuity evidence remains; manual or LLM recheck is safer."
    else:
        same = "unclear"
        transition = "pure_camera_transition" if transition_hits else "unclear"
        recommendation = "recheck"
        reason = "Caption evidence is insufficient to decide whether both sides describe the same abnormal event."

    return {
        "same_event_rule_label": same,
        "transition_type_rule_label": transition,
        "merge_recommendation_rule": recommendation,
        "rule_reason": reason,
        "left_risk": left_risk,
        "right_risk": right_risk,
        "shared_risk": shared_risk,
        "left_object": left_object,
        "right_object": right_object,
        "shared_object": shared_object,
        "left_action": left_action,
        "right_action": right_action,
        "shared_action": shared_action,
        "lexical_overlap": lexical,
        "risk_overlap": risk_overlap,
        "object_overlap": object_overlap,
        "action_overlap": action_overlap,
    }


def load_candidate_caption_index(path):
    rows = read_csv(path)
    index = defaultdict(deque)
    frame_index = {}
    for row in rows:
        dataset = row.get("dataset", "")
        video_id = normalize_stem(row.get("video_id", ""))
        frame = as_int(row.get("frame"), None)
        if frame is None:
            continue
        item = dict(row)
        item["_frame"] = frame
        item["_score_drop"] = as_float(row.get("score_drop"), 0.0) or 0.0
        index[(dataset, video_id, frame)].append(item)
        frame_index[(dataset, video_id, frame)] = item
    return index, frame_index


def lookup_candidate(candidate_index, frame_index, dataset, video_id, h4_frame):
    candidates = candidate_index.get((dataset, video_id, h4_frame), deque())
    if candidates:
        return candidates[0]
    exact = frame_index.get((dataset, video_id, h4_frame), {})
    if exact:
        return exact
    matches = []
    for (cand_dataset, cand_video_id, cand_frame), item in frame_index.items():
        if cand_dataset != dataset or cand_frame != h4_frame:
            continue
        if cand_video_id.startswith(video_id + ".") or video_id.startswith(cand_video_id + "."):
            matches.append(item)
    if len(matches) == 1:
        return matches[0]
    return {}


def load_diag_index(path):
    rows = read_csv(path)
    by_exact = defaultdict(deque)
    by_frame = defaultdict(deque)
    for row in rows:
        dataset = row.get("dataset", "")
        video_id = normalize_stem(row.get("video_id", ""))
        h4_frame = as_int(row.get("nearest_h4_frame"), None)
        if h4_frame is None:
            continue
        gap_length = as_int(row.get("gap_length"), None)
        by_exact[(dataset, video_id, h4_frame, gap_length)].append(row)
        by_frame[(dataset, video_id, h4_frame)].append(row)
    return by_exact, by_frame


def build_semantic_cases(changed_path, h4_path, diag_path, warnings):
    changed_rows = read_csv(changed_path)
    candidate_index, frame_index = load_candidate_caption_index(h4_path)
    diag_exact, diag_frame = load_diag_index(diag_path) if diag_path and Path(diag_path).exists() else ({}, {})
    if not changed_rows:
        warnings.append(f"changed cases file has no rows: {changed_path}")
    if not frame_index:
        warnings.append(f"H4 candidates file has no usable caption rows: {h4_path}")
    if not diag_exact:
        warnings.append(f"merge diagnostics unavailable or empty: {diag_path}")

    rows = []
    for changed in changed_rows:
        dataset = changed.get("dataset", "")
        video_id = normalize_stem(changed.get("video_id", ""))
        h4_frame = as_int(changed.get("h4_frame"), None)
        gap_length = as_int(changed.get("gap_length"), None)
        diag = None
        if h4_frame is not None:
            exact_key = (dataset, video_id, h4_frame, gap_length)
            if exact_key in diag_exact and diag_exact[exact_key]:
                diag = diag_exact[exact_key].popleft()
            else:
                frame_key = (dataset, video_id, h4_frame)
                if frame_key in diag_frame and diag_frame[frame_key]:
                    diag = diag_frame[frame_key].popleft()
        if h4_frame is None:
            warnings.append(f"missing h4_frame in changed case: {dataset}/{video_id}")
            continue
        candidate = lookup_candidate(candidate_index, frame_index, dataset, video_id, h4_frame)
        if not candidate:
            warnings.append(f"missing H4 caption row for changed case: {dataset}/{video_id}@{h4_frame}")
        h4_types = changed.get("h4_types") or candidate.get("types", "")
        score_drop = changed.get("h4_score_drop") or candidate.get("score_drop", "")
        caption_before = candidate.get("caption_before", "")
        caption_current = candidate.get("caption_current", "")
        caption_after = candidate.get("caption_after", "")
        rule = semantic_rule(caption_before, caption_current, caption_after, h4_types)
        row = {
            "dataset": dataset,
            "video_id": video_id,
            "h4_frame": h4_frame,
            "gap_start": diag.get("gap_start", "") if diag else "",
            "gap_end": diag.get("gap_end", "") if diag else "",
            "gap_length": changed.get("gap_length") or (diag.get("gap_length", "") if diag else ""),
            "distance_to_h4": diag.get("distance_to_h4", "") if diag else "",
            "h4_types": h4_types,
            "h4_score_drop": score_drop,
            "fixed_or_worsened": changed.get("fixed_or_worsened", ""),
            "left_interval_start": diag.get("left_interval_start", "") if diag else "",
            "left_interval_end": diag.get("left_interval_end", "") if diag else "",
            "right_interval_start": diag.get("right_interval_start", "") if diag else "",
            "right_interval_end": diag.get("right_interval_end", "") if diag else "",
            "left_caption_summary": caption_summary(caption_before),
            "gap_caption_summary": caption_summary(caption_current),
            "right_caption_summary": caption_summary(caption_after),
            "caption_before_h4": caption_before,
            "caption_at_h4": caption_current,
            "caption_after_h4": caption_after,
            "left_risk_terms": join_terms(rule["left_risk"]),
            "right_risk_terms": join_terms(rule["right_risk"]),
            "shared_risk_terms": join_terms(rule["shared_risk"]),
            "left_object_terms": join_terms(rule["left_object"]),
            "right_object_terms": join_terms(rule["right_object"]),
            "shared_object_terms": join_terms(rule["shared_object"]),
            "left_action_terms": join_terms(rule["left_action"]),
            "right_action_terms": join_terms(rule["right_action"]),
            "shared_action_terms": join_terms(rule["shared_action"]),
            "lexical_overlap": fmt(rule["lexical_overlap"]),
            "risk_overlap": fmt(rule["risk_overlap"]),
            "object_overlap": fmt(rule["object_overlap"]),
            "action_overlap": fmt(rule["action_overlap"]),
            "same_event_rule_label": rule["same_event_rule_label"],
            "transition_type_rule_label": rule["transition_type_rule_label"],
            "merge_recommendation_rule": rule["merge_recommendation_rule"],
            "rule_reason": rule["rule_reason"],
        }
        rows.append(row)
    return rows


def mean(rows, key):
    values = []
    for row in rows:
        value = as_float(row.get(key), None)
        if value is not None:
            values.append(value)
    return fmt(sum(values) / len(values) if values else "")


def ratio(rows, key, value):
    if not rows:
        return 0.0
    return fmt(sum(1 for row in rows if row.get(key) == value) / len(rows))


def top_values(rows, key, split=False, limit=5):
    counter = Counter()
    for row in rows:
        value = row.get(key, "")
        if split:
            for item in re.split(r"[;|,]+", value):
                item = item.strip()
                if item:
                    counter[item] += 1
        elif value:
            counter[value] += 1
    return ";".join(f"{k}:{v}" for k, v in counter.most_common(limit))


def semantic_summary(rows):
    out = []
    for group in ["fixed", "worsened", "neutral", "unclear"]:
        subset = [row for row in rows if row.get("fixed_or_worsened") == group]
        if not subset and group in {"neutral", "unclear"}:
            continue
        out.append(
            {
                "case_group": group,
                "num_cases": len(subset),
                "mean_gap_length": mean(subset, "gap_length"),
                "mean_distance_to_h4": mean(subset, "distance_to_h4"),
                "mean_h4_score_drop": mean(subset, "h4_score_drop"),
                "same_event_ratio": ratio(subset, "same_event_rule_label", "same_event"),
                "different_event_ratio": ratio(subset, "same_event_rule_label", "different_event"),
                "unclear_ratio": ratio(subset, "same_event_rule_label", "unclear"),
                "merge_recommendation_ratio": ratio(subset, "merge_recommendation_rule", "merge"),
                "do_not_merge_ratio": ratio(subset, "merge_recommendation_rule", "do_not_merge"),
                "recheck_ratio": ratio(subset, "merge_recommendation_rule", "recheck"),
                "mean_lexical_overlap": mean(subset, "lexical_overlap"),
                "mean_risk_overlap": mean(subset, "risk_overlap"),
                "mean_object_overlap": mean(subset, "object_overlap"),
                "mean_action_overlap": mean(subset, "action_overlap"),
                "top_h4_types": top_values(subset, "h4_types", split=True),
                "top_transition_type_labels": top_values(subset, "transition_type_rule_label"),
            }
        )
    return out


def nearest_h4(candidates, gap_start, gap_end):
    best = None
    best_dist = None
    for cand in candidates:
        frame = cand["_frame"]
        if gap_start <= frame <= gap_end:
            dist = 0
        else:
            dist = min(abs(frame - gap_start), abs(frame - gap_end))
        if best_dist is None or dist < best_dist:
            best = cand
            best_dist = dist
    return best, best_dist


def semantic_allowed(candidate):
    rule = semantic_rule(
        candidate.get("caption_before", ""),
        candidate.get("caption_current", ""),
        candidate.get("caption_after", ""),
        candidate.get("types", ""),
    )
    return rule["merge_recommendation_rule"] == "merge" or rule["same_event_rule_label"] == "same_event"


def semantic_merge(intervals, h4s, args):
    if not intervals:
        return [], []
    out = [intervals[0]]
    events = []
    for current in intervals[1:]:
        prev = out[-1]
        gap_start, gap_end = prev[1], current[0]
        gap_len = gap_end - gap_start
        cand, distance = nearest_h4(h4s, gap_start, gap_end)
        if (
            cand
            and gap_len <= args.h4_gap_max_frames
            and distance <= args.h4_window_size
            and cand["_score_drop"] > 0
            and semantic_allowed(cand)
        ):
            merged = (prev[0], max(prev[1], current[1]))
            out[-1] = merged
            events.append({"left": prev, "right": current, "merged": merged, "h4": cand, "gap_length": gap_len})
        else:
            out.append(current)
    return out, events


def oracle_merge(intervals, h4s, fixed_keys, args):
    if not intervals:
        return [], []
    out = [intervals[0]]
    events = []
    for current in intervals[1:]:
        prev = out[-1]
        gap_start, gap_end = prev[1], current[0]
        gap_len = gap_end - gap_start
        cand, distance = nearest_h4(h4s, gap_start, gap_end)
        event_key = None
        if cand:
            event_key = (cand.get("_dataset", ""), cand.get("_video_id", ""), cand["_frame"], gap_len)
        if (
            cand
            and gap_len <= args.h4_gap_max_frames
            and distance <= args.h4_window_size
            and cand["_score_drop"] > 0
            and event_key in fixed_keys
        ):
            merged = (prev[0], max(prev[1], current[1]))
            out[-1] = merged
            events.append({"left": prev, "right": current, "merged": merged, "h4": cand, "gap_length": gap_len})
        else:
            out.append(current)
    return out, events


def attach_candidate_keys(h4_by_key):
    for key, rows in h4_by_key.items():
        dataset, video_id = key
        for row in rows:
            row["_dataset"] = dataset
            row["_video_id"] = video_id


def fp_duration(preds, gts):
    return max(0, sum(e - s for s, e in preds) - intersect_duration(preds, gts))


def fn_duration(preds, gts):
    return max(0, sum(e - s for s, e in gts) - intersect_duration(preds, gts))


def metric_row(method, metrics, cases):
    labels = Counter(row["fixed_or_worsened"] for row in cases)
    return {
        "method": method,
        "frame_precision": metrics["frame_precision"],
        "frame_recall": metrics["frame_recall"],
        "frame_f1": metrics["frame_f1"],
        "segment_precision": metrics["segment_precision"],
        "segment_recall": metrics["segment_recall"],
        "segment_f1": metrics["segment_f1"],
        "avg_segments_per_video": metrics["average_predicted_segments_per_video"],
        "fragmented_gt_ratio": metrics["fragmented_gt_ratio"],
        "false_positive_duration": metrics["false_positive_duration"],
        "false_negative_duration": metrics["false_negative_duration"],
        "changed_cases": len(cases),
        "fixed_cases": labels["fixed"],
        "worsened_cases": labels["worsened"],
    }


def run_interval_comparison(data_root, h4_candidates, semantic_rows, args):
    scores_by_key, gt_by_key, warnings = load_all_inputs(data_root)
    h4_by_key = load_h4_candidates(h4_candidates)
    attach_candidate_keys(h4_by_key)
    all_keys = sorted(set(scores_by_key) | set(gt_by_key))
    original_by_key = {}
    proximity_by_key = {}
    semantic_by_key = {}
    oracle_by_key = {}
    proximity_events = {}
    semantic_events = {}
    oracle_events = {}
    fixed_keys = {
        (row["dataset"], row["video_id"], as_int(row["h4_frame"]), as_int(row["gap_length"]))
        for row in semantic_rows
        if row.get("fixed_or_worsened") == "fixed"
    }
    for key in all_keys:
        scores = scores_by_key.get(key, {})
        raw = raw_intervals_from_scores(scores, args.score_threshold)
        original = merge_ranges(raw, args.merge_gap_frames)
        proximity, p_events = relaxed_merge(original, h4_by_key.get(key, []), args)
        semantic, s_events = semantic_merge(original, h4_by_key.get(key, []), args)
        oracle, o_events = oracle_merge(original, h4_by_key.get(key, []), fixed_keys, args)
        original_by_key[key] = original
        proximity_by_key[key] = proximity
        semantic_by_key[key] = semantic
        oracle_by_key[key] = oracle
        proximity_events[key] = p_events
        semantic_events[key] = s_events
        oracle_events[key] = o_events
    proximity_cases = changed_cases(original_by_key, proximity_by_key, gt_by_key, proximity_events)
    semantic_cases = changed_cases(original_by_key, semantic_by_key, gt_by_key, semantic_events)
    oracle_cases = changed_cases(original_by_key, oracle_by_key, gt_by_key, oracle_events)
    metrics = [
        metric_row("original", evaluate_method("original", original_by_key, gt_by_key, all_keys), []),
        metric_row("h4_proximity_merge", evaluate_method("h4_proximity_merge", proximity_by_key, gt_by_key, all_keys), proximity_cases),
        metric_row("vlm_semantic_continuity_merge", evaluate_method("vlm_semantic_continuity_merge", semantic_by_key, gt_by_key, all_keys), semantic_cases),
        metric_row("vlm_semantic_recheck_oracle", evaluate_method("vlm_semantic_recheck_oracle", oracle_by_key, gt_by_key, all_keys), oracle_cases),
    ]
    return metrics, {
        "proximity_cases": proximity_cases,
        "semantic_cases": semantic_cases,
        "oracle_cases": oracle_cases,
        "warnings": warnings,
    }


def accepted_rejected_rows(rows, accepted=True):
    if accepted:
        subset = [row for row in rows if row.get("merge_recommendation_rule") == "merge" or row.get("same_event_rule_label") == "same_event"]
    else:
        subset = [row for row in rows if not (row.get("merge_recommendation_rule") == "merge" or row.get("same_event_rule_label") == "same_event")]
    return subset


def case_bullets(rows, limit=5):
    lines = []
    for row in rows[:limit]:
        lines.extend(
            [
                f"- {row['dataset']} / {row['video_id']} @ H4 {row['h4_frame']} ({row['fixed_or_worsened']}): {row['rule_reason']}",
                f"  - left: {row['left_caption_summary']}",
                f"  - gap: {row['gap_caption_summary']}",
                f"  - right: {row['right_caption_summary']}",
            ]
        )
    if not lines:
        lines.append("- None.")
    return lines


def write_report(path, args, semantic_rows, summary_rows, metrics, accepted, rejected, warnings):
    fixed = next((row for row in summary_rows if row["case_group"] == "fixed"), {})
    worsened = next((row for row in summary_rows if row["case_group"] == "worsened"), {})
    metric_by_name = {row["method"]: row for row in metrics}
    original = metric_by_name.get("original", {})
    proximity = metric_by_name.get("h4_proximity_merge", {})
    semantic = metric_by_name.get("vlm_semantic_continuity_merge", {})
    oracle = metric_by_name.get("vlm_semantic_recheck_oracle", {})
    lines = [
        "# VLM Semantic Continuity Recheck Report",
        "",
        "## Goal",
        "",
        "This experiment tests whether judging event continuity from VLM captions is more useful for reconstructing continuous abnormal intervals than directly merging every H4-boundary-adjacent gap.",
        "",
        "## Background",
        "",
        "- H4 proximity merging reduced fragmented GT and improved frame recall in the previous run.",
        "- It also increased false-positive duration and produced more worsened than fixed changed cases.",
        "- The key question is whether the captions before/at/after the H4 point still describe the same risk event.",
        "",
        "## Data and Inputs",
        "",
        f"- changed cases: `{args.changed_cases}`",
        f"- H4 candidates: `{args.h4_candidates}`",
        f"- merge diagnostics: `{args.merge_diagnostics}`",
        f"- data root: `{args.data_root}`",
        "- score/GT inputs are discovered from the dataset configs in `scripts/h1_h4_boundary_aware_postprocess.py`.",
        "",
        "## Semantic Continuity Rule",
        "",
        "The default rule is offline and caption-based. It extracts lexical tokens, risk terms, object terms, action terms, explicit new-scene cues, camera-transition cues, and multi-scene compression cues from the VLM captions. Shared risk/action/object evidence favors `same_event` and `merge`; explicit new-event language or very low continuity favors `different_event` and `do_not_merge`; ambiguous multi-scene compression is marked `recheck`.",
        "",
        "## Fixed vs Worsened Analysis",
        "",
        "| group | cases | same-event | different-event | merge | do-not-merge | recheck | mean lexical | mean risk | top transitions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case_group']} | {row['num_cases']} | {row['same_event_ratio']} | "
            f"{row['different_event_ratio']} | {row['merge_recommendation_ratio']} | "
            f"{row['do_not_merge_ratio']} | {row['recheck_ratio']} | "
            f"{row['mean_lexical_overlap']} | {row['mean_risk_overlap']} | "
            f"{row['top_transition_type_labels']} |"
        )
    lines.extend(
        [
            "",
            f"- Fixed same-event ratio: {fixed.get('same_event_ratio', '')}; worsened same-event ratio: {worsened.get('same_event_ratio', '')}.",
            f"- Fixed do-not-merge ratio: {fixed.get('do_not_merge_ratio', '')}; worsened do-not-merge ratio: {worsened.get('do_not_merge_ratio', '')}.",
            f"- Top fixed H4 types: {fixed.get('top_h4_types', '')}.",
            f"- Top worsened H4 types: {worsened.get('top_h4_types', '')}.",
            "",
            "## Interval Reconstruction Comparison",
            "",
            "| method | frame P | frame R | frame F1 | segment F1 | avg seg/video | fragmented GT | FP duration | FN duration | changed | fixed | worsened |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['frame_precision']} | {row['frame_recall']} | "
            f"{row['frame_f1']} | {row['segment_f1']} | {row['avg_segments_per_video']} | "
            f"{row['fragmented_gt_ratio']} | {row['false_positive_duration']} | "
            f"{row['false_negative_duration']} | {row['changed_cases']} | "
            f"{row['fixed_cases']} | {row['worsened_cases']} |"
        )
    lines.extend(
        [
            "",
            f"- H4 proximity changes recall from {original.get('frame_recall')} to {proximity.get('frame_recall')} and FP duration from {original.get('false_positive_duration')} to {proximity.get('false_positive_duration')}.",
            f"- VLM semantic continuity changes recall to {semantic.get('frame_recall')} and FP duration to {semantic.get('false_positive_duration')}.",
            f"- Oracle fixed-only merging gives recall {oracle.get('frame_recall')} and FP duration {oracle.get('false_positive_duration')}; this is an upper-bound diagnostic, not a usable method.",
            "",
            "## Case Studies",
            "",
            "### Accepted semantic merges",
            "",
        ]
    )
    lines.extend(case_bullets(accepted, 5))
    lines.extend(["", "### Rejected or recheck cases", ""])
    lines.extend(case_bullets(rejected, 5))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- H4 boundary points are useful as triggers for semantic rechecking.",
            "- Naive H4 proximity merging over-merges many cases, so H4 should not be treated as a final merge decision by itself.",
            "- Caption semantic continuity can distinguish some fixed and worsened merges when fixed cases have higher same-event/merge ratios and semantic-gated metrics reduce worsened cases or FP duration.",
            "- If the semantic-gated metrics do not improve, the likely reason is that current captions are not explicit enough about event continuity; this supports moving continuity modeling into the VLM output stage.",
            "- The more promising direction is event-continuity-aware VLM descriptions rather than merging all boundary-adjacent fragments.",
            "",
            "## Next Steps",
            "",
            "- Adjust the VLM prompt so the model directly outputs structured event continuity fields.",
            "- Add small manual labels for same event, new event, and multi-scene compression around H4 gaps.",
            "- Test an optional LLM recheck or visual shot-boundary detector after the rule-based pass.",
            "- Add semantic continuity as a VAU interpreter field instead of only using it as post-processing.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings[:100])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prompt(path):
    path.write_text(
        """# VLM Event Continuity Prompt

For each video clip, describe the abnormal-event continuity explicitly. Return only valid JSON with the following fields:

```json
{
  "scene_description": "Brief visual description of the clip.",
  "main_actors": ["people or groups involved"],
  "objects": ["salient objects, vehicles, weapons, scene objects"],
  "actions": ["observable actions"],
  "risk_actions": ["risk-relevant actions such as fight, chase, fall, crash, fire, weapon use"],
  "abnormal_event_state": "none | precursor | ongoing | aftermath | unclear",
  "transition_from_previous": "none | same_scene | same_event_new_view | new_scene_new_event | unclear",
  "event_continuity_with_previous": "same_event | related_event | different_event | unclear",
  "should_link_with_previous_segment": "yes | no | uncertain",
  "reason": "One concise reason grounded in visible actors, objects, actions, location, and risk state."
}
```

Decision rules:

- Use `same_event` when the same abnormal event continues despite a camera cut, view change, zoom, close-up, or angle change.
- Use `different_event` when the clip shows a different location, actors, objects, or unrelated risk event.
- Use `related_event` when the clip is a precursor or aftermath of the same incident but not the same continuous action.
- Use `unclear` and `uncertain` when the clip is compressed, contains multiple scenes, or lacks enough visual evidence.
- Do not infer continuity only from a camera transition. Continuity requires shared actors, objects, location, action, or risk state.
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based VLM semantic continuity recheck for H4 merge cases.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--changed-cases", default="outputs/26-07-09-01-06-h1-h4-trigger-diagnostics/h4_relaxed_changed_cases.csv")
    parser.add_argument("--h4-candidates", default="outputs/26-07-09-00-35-caption_boundary_screen/h4_strong_candidates.csv")
    parser.add_argument("--merge-diagnostics", default="outputs/26-07-09-01-06-h1-h4-trigger-diagnostics/merge_opportunity_diagnostics.csv")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--caption-window", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--merge-gap-frames", type=int, default=32)
    parser.add_argument("--h4-window-size", type=int, default=48)
    parser.add_argument("--h4-gap-max-frames", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{datetime.now():%y-%m-%d-%H-%M}-{TASK_NAME}"
    warnings = []
    data_root = Path(args.data_root)
    changed_path = Path(args.changed_cases)
    h4_path = Path(args.h4_candidates)
    diag_path = Path(args.merge_diagnostics) if args.merge_diagnostics else None
    if not changed_path.exists():
        raise FileNotFoundError(f"changed cases file not found: {changed_path}")
    if not h4_path.exists():
        raise FileNotFoundError(f"H4 candidates file not found: {h4_path}")
    semantic_rows = build_semantic_cases(changed_path, h4_path, diag_path, warnings)
    summary_rows = semantic_summary(semantic_rows)
    metrics, interval_extra = run_interval_comparison(data_root, h4_path, semantic_rows, args)
    warnings.extend(interval_extra["warnings"])
    accepted = accepted_rejected_rows(semantic_rows, accepted=True)
    rejected = accepted_rejected_rows(semantic_rows, accepted=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "semantic_recheck_cases.csv", semantic_rows, CASE_FIELDS)
    write_csv(output_dir / "fixed_vs_worsened_semantic_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "semantic_merge_metrics.csv", metrics, METRIC_FIELDS)
    write_csv(output_dir / "semantic_merge_accepted_cases.csv", accepted, ACCEPT_REJECT_FIELDS)
    write_csv(output_dir / "semantic_merge_rejected_cases.csv", rejected, ACCEPT_REJECT_FIELDS)
    write_report(output_dir / "vlm_semantic_continuity_report.md", args, semantic_rows, summary_rows, metrics, accepted, rejected, warnings)
    write_prompt(output_dir / "vlm_event_continuity_prompt.md")
    write_json(
        output_dir / "vlm_semantic_continuity_summary.json",
        {
            "args": vars(args),
            "num_semantic_cases": len(semantic_rows),
            "accepted_cases": len(accepted),
            "rejected_or_recheck_cases": len(rejected),
            "warnings": warnings,
            "dataset_configs": [
                {
                    "dataset": cfg["dataset"],
                    "score_dir": str(Path(args.data_root) / cfg["score_dir"]),
                    "gt_file": str(Path(args.data_root) / cfg["gt_file"]),
                }
                for cfg in DATASET_CONFIGS
            ],
        },
    )
    script_dir = output_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    print(f"semantic cases: {len(semantic_rows)}")
    print(f"accepted semantic merges: {len(accepted)}")
    print(f"rejected/recheck cases: {len(rejected)}")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
