import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_FIELDS = ["metric", "value", "source_file", "note"]
METHOD_FIELDS = [
    "method", "frame_precision", "frame_recall", "frame_f1", "segment_f1",
    "avg_segments_per_video", "fragmented_gt_ratio", "false_positive_duration",
    "false_negative_duration", "changed_cases", "fixed_cases", "worsened_cases",
]
FIXED_FIELDS = [
    "group", "cases", "same_event_ratio", "different_event_ratio",
    "merge_ratio", "do_not_merge_ratio", "recheck_ratio", "mean_lexical_overlap",
    "mean_risk_overlap", "top_h4_types", "top_transition_types",
]


def read_csv(path):
    if not path or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=6):
    if value is None or value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value) or math.isinf(value):
        return ""
    return round(value, digits)


def latest_file(root, name):
    matches = sorted(root.glob(f"**/{name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def add_summary(rows, metric, value, source, note=""):
    rows.append({
        "metric": metric,
        "value": fmt(value),
        "source_file": str(source).replace("\\", "/") if source else "",
        "note": note,
    })


def load_sources(outputs_root):
    return {
        "h4_summary_json": latest_file(outputs_root, "h4_boundary_experiment_summary.json"),
        "h4_candidates": latest_file(outputs_root, "h4_strong_candidates.csv"),
        "type_summary": latest_file(outputs_root, "type_score_drop_summary.csv"),
        "caption_method": latest_file(outputs_root, "method_comparison.csv"),
        "merge_metrics": latest_file(outputs_root, "semantic_merge_metrics.csv"),
        "fixed_worsened": latest_file(outputs_root, "fixed_vs_worsened_semantic_summary.csv"),
        "filter_funnel": latest_file(outputs_root, "filter_funnel.csv"),
        "changed_cases": latest_file(outputs_root, "h4_relaxed_changed_cases.csv"),
        "semantic_cases": latest_file(outputs_root, "semantic_recheck_cases.csv"),
    }


def build_evidence_summary(sources):
    rows = []
    warnings = []

    if sources["h4_summary_json"]:
        try:
            summary = json.loads(sources["h4_summary_json"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        rule_summary = summary.get("rule_summary", {})
        add_summary(rows, "processed_videos", rule_summary.get("processed_videos"), sources["h4_summary_json"])

    caption_method = read_csv(sources["caption_method"])
    if caption_method:
        row = caption_method[0]
        add_summary(rows, "total_candidates", row.get("total_candidates"), sources["caption_method"])
        add_summary(rows, "possible_context_forgetting_count", row.get("num_possible_context_forgetting"), sources["caption_method"])
        add_summary(rows, "multi_scene_compression_boundary_count", row.get("num_multi_scene_compression_boundary"), sources["caption_method"])
    else:
        warnings.append("missing method_comparison.csv")

    type_summary = read_csv(sources["type_summary"])
    if type_summary:
        type_map = {row.get("type"): row for row in type_summary}
        add_summary(rows, "explicit_transition_boundary_count", type_map.get("explicit_transition_boundary", {}).get("count"), sources["type_summary"])
        if not any(row.get("metric") == "multi_scene_compression_boundary_count" for row in rows):
            add_summary(rows, "multi_scene_compression_boundary_count", type_map.get("multi_scene_compression_boundary", {}).get("count"), sources["type_summary"])
    else:
        warnings.append("missing type_score_drop_summary.csv")

    merge_metrics = read_csv(sources["merge_metrics"])
    metrics = {row.get("method"): row for row in merge_metrics}
    original = metrics.get("original", {})
    h4 = metrics.get("h4_proximity_merge", {})
    semantic = metrics.get("vlm_semantic_continuity_merge", {})
    for prefix, row in [("original", original), ("h4_proximity", h4), ("semantic_merge", semantic)]:
        if row:
            add_summary(rows, f"{prefix}_frame_recall", row.get("frame_recall"), sources["merge_metrics"])
            add_summary(rows, f"{prefix}_fragmented_gt", row.get("fragmented_gt_ratio"), sources["merge_metrics"])
            add_summary(rows, f"{prefix}_fp_duration", row.get("false_positive_duration"), sources["merge_metrics"])
            add_summary(rows, f"{prefix}_segment_f1", row.get("segment_f1"), sources["merge_metrics"])
    if h4:
        add_summary(rows, "h4_proximity_worsened_cases", h4.get("worsened_cases"), sources["merge_metrics"])
    if semantic:
        add_summary(rows, "semantic_merge_worsened_cases", semantic.get("worsened_cases"), sources["merge_metrics"])
    if not merge_metrics:
        warnings.append("missing semantic_merge_metrics.csv")

    return rows, warnings


def build_method_comparison(sources):
    rows = []
    for row in read_csv(sources["merge_metrics"]):
        rows.append({field: fmt(row.get(field)) for field in METHOD_FIELDS})
    return rows


def build_fixed_worsened(sources):
    rows = []
    for row in read_csv(sources["fixed_worsened"]):
        rows.append({
            "group": row.get("case_group", ""),
            "cases": row.get("num_cases", ""),
            "same_event_ratio": fmt(row.get("same_event_ratio")),
            "different_event_ratio": fmt(row.get("different_event_ratio")),
            "merge_ratio": fmt(row.get("merge_recommendation_ratio")),
            "do_not_merge_ratio": fmt(row.get("do_not_merge_ratio")),
            "recheck_ratio": fmt(row.get("recheck_ratio")),
            "mean_lexical_overlap": fmt(row.get("mean_lexical_overlap")),
            "mean_risk_overlap": fmt(row.get("mean_risk_overlap")),
            "top_h4_types": row.get("top_h4_types", ""),
            "top_transition_types": row.get("top_transition_type_labels", ""),
        })
    return rows


def plot_method_metrics(rows, output_dir):
    if not rows:
        return
    methods = [row["method"] for row in rows if row["method"] in {"original", "h4_proximity_merge", "vlm_semantic_continuity_merge"}]
    selected = [row for row in rows if row["method"] in methods]
    ratio_metrics = ["frame_recall", "frame_f1", "segment_f1", "fragmented_gt_ratio"]
    duration_metrics = ["false_positive_duration"]

    x = range(len(methods))
    width = 0.2
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(ratio_metrics):
        values = [as_float(row.get(metric), 0) for row in selected]
        plt.bar([v + (i - 1.5) * width for v in x], values, width=width, label=metric)
    plt.xticks(list(x), methods, rotation=15, ha="right")
    plt.ylim(0, 0.75)
    plt.ylabel("Ratio")
    plt.title("H4 method ratio metrics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_method_metric_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    values = [as_float(row.get("false_positive_duration"), 0) for row in selected]
    plt.bar(methods, values)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Frames")
    plt.title("False positive duration by method")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_method_fp_duration.png", dpi=180)
    plt.close()


def plot_fixed_worsened(rows, output_dir):
    if not rows:
        return
    methods = [row["method"] for row in rows if row["method"] in {"h4_proximity_merge", "vlm_semantic_continuity_merge"}]
    fixed = [as_float(row.get("fixed_cases"), 0) for row in rows if row["method"] in methods]
    worsened = [as_float(row.get("worsened_cases"), 0) for row in rows if row["method"] in methods]
    x = range(len(methods))
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - 0.18 for i in x], fixed, width=0.36, label="fixed")
    plt.bar([i + 0.18 for i in x], worsened, width=0.36, label="worsened")
    plt.xticks(list(x), methods, rotation=15, ha="right")
    plt.ylabel("Cases")
    plt.title("Fixed vs worsened cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_fixed_worsened_counts.png", dpi=180)
    plt.close()


def plot_funnel(path, output_dir):
    rows = read_csv(path)
    if not rows:
        return
    stages = [row.get("stage", "") for row in rows]
    remaining = [as_float(row.get("remaining_gaps"), 0) for row in rows]
    plt.figure(figsize=(9, 4.8))
    plt.plot(stages, remaining, marker="o")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Remaining gaps")
    plt.title("H4 trigger filter funnel")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_filter_funnel.png", dpi=180)
    plt.close()


def plot_type_counts(sources, output_dir):
    rows = read_csv(sources["type_summary"])
    if rows:
        types = [row.get("type", "") for row in rows]
        counts = [as_float(row.get("count"), 0) for row in rows]
    else:
        candidates = read_csv(sources["h4_candidates"])
        counter = Counter()
        for row in candidates:
            for item in str(row.get("types", "")).split(";"):
                if item:
                    counter[item] += 1
        types = [name for name, _ in counter.most_common()]
        counts = [count for _, count in counter.most_common()]
    if not types:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(types, counts)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("H4 candidate type counts")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_boundary_type_counts.png", dpi=180)
    plt.close()


def write_report(path, summary_rows, method_rows, fixed_rows, warnings):
    values = {row["metric"]: row["value"] for row in summary_rows}
    method_map = {row["method"]: row for row in method_rows}
    original = method_map.get("original", {})
    h4 = method_map.get("h4_proximity_merge", {})
    semantic = method_map.get("vlm_semantic_continuity_merge", {})
    text = [
        "# H4 boundary evidence report",
        "",
        "## Main observations",
        "",
        f"- Caption boundary screening found {values.get('total_candidates', 'n/a')} total candidates, including {values.get('possible_context_forgetting_count', 'n/a')} possible context-forgetting rows and {values.get('multi_scene_compression_boundary_count', 'n/a')} multi-scene compression rows.",
        f"- H4 proximity merge raised frame recall from {original.get('frame_recall', 'n/a')} to {h4.get('frame_recall', 'n/a')} and reduced fragmented GT ratio from {original.get('fragmented_gt_ratio', 'n/a')} to {h4.get('fragmented_gt_ratio', 'n/a')}.",
        f"- The same proximity rule increased false-positive duration from {original.get('false_positive_duration', 'n/a')} to {h4.get('false_positive_duration', 'n/a')} and lowered segment F1 from {original.get('segment_f1', 'n/a')} to {h4.get('segment_f1', 'n/a')}.",
        f"- Semantic continuity gating kept most recall gain ({semantic.get('frame_recall', 'n/a')}) while reducing false-positive duration to {semantic.get('false_positive_duration', 'n/a')} and worsened cases to {semantic.get('worsened_cases', 'n/a')}.",
        "",
        "## Interpretation",
        "",
        "H4 boundary phenomena can be detected at scale from captions, but a boundary point alone is not a reliable merge rule. The interval experiment shows a real recall/fragmentation benefit, yet also a false-positive and segment-quality cost. Caption-level semantic continuity reduces some over-merge cases, which suggests that event continuity is the key variable. However, ordinary captions remain too coarse and sometimes conflate new scenes, ongoing events, and compressed multi-scene summaries.",
        "",
        "The stronger next step is to make the VLM output event-continuity-aware structured descriptions, or to inspect original video around candidate boundaries. Post-hoc inference from generic captions should be treated as diagnostic evidence, not final model-level validation.",
        "",
        "## Fixed vs worsened semantic differences",
        "",
    ]
    for row in fixed_rows:
        text.append(
            f"- {row['group']}: cases={row['cases']}, same_event_ratio={row['same_event_ratio']}, "
            f"recheck_ratio={row['recheck_ratio']}, mean_risk_overlap={row['mean_risk_overlap']}."
        )
    text.extend(["", "## Warnings", ""])
    text.extend([f"- {warning}" for warning in warnings] or ["- None."])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def run(outputs_root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(outputs_root)
    summary_rows, warnings = build_evidence_summary(sources)
    method_rows = build_method_comparison(sources)
    fixed_rows = build_fixed_worsened(sources)

    write_csv(output_dir / "h4_evidence_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "h4_method_comparison.csv", method_rows, METHOD_FIELDS)
    write_csv(output_dir / "h4_fixed_vs_worsened_summary.csv", fixed_rows, FIXED_FIELDS)

    plot_method_metrics(method_rows, output_dir)
    plot_fixed_worsened(method_rows, output_dir)
    plot_funnel(sources["filter_funnel"], output_dir)
    plot_type_counts(sources, output_dir)
    write_report(output_dir / "h4_boundary_evidence_report.md", summary_rows, method_rows, fixed_rows, warnings)

    manifest = {key: str(value).replace("\\", "/") if value else "" for key, value in sources.items()}
    manifest["warnings"] = warnings
    (output_dir / "h4_boundary_evidence_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(Path(args.outputs_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
