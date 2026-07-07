import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    coverage_by_intervals,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    score_metadata,
    score_path_for_video,
    write_json,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_prediction(preds: dict, dataset: str, video_id: str) -> dict:
    if video_id in preds and preds[video_id].get("dataset") == dataset:
        return preds[video_id]
    key = f"{dataset}::{video_id}"
    if key in preds:
        return preds[key]
    for value in preds.values():
        if isinstance(value, dict) and value.get("dataset") == dataset and value.get("video_id") == video_id:
            return value
    return {"events": []}


def event_intervals(video_pred: dict) -> list[dict]:
    return [
        {
            "event_id": event["event_id"],
            "start": event["merged_start"],
            "end": event["merged_end"],
            "duration": event["merged_duration"],
        }
        for event in video_pred.get("events", [])
    ]


def micro_intervals(video_pred: dict) -> list[dict]:
    items = []
    for event in video_pred.get("events", []):
        for micro in event.get("micro_intervals", []):
            item = dict(micro)
            item["event_id"] = event["event_id"]
            items.append(item)
    return items


def overlaps(a: dict, b: dict) -> int:
    a_start = a.get("start", a.get("merged_start"))
    a_end = a.get("end", a.get("merged_end"))
    b_start = b.get("start", b.get("merged_start"))
    b_end = b.get("end", b.get("merged_end"))
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def summarize_coverage(rows: list[dict], prefix: str) -> dict:
    total = len(rows)
    missed = sum(row[f"{prefix}_missed"] for row in rows)
    return {
        f"{prefix}_segments": total,
        f"{prefix}_covered_segments": total - missed,
        f"{prefix}_missed_segments": missed,
        f"{prefix}_segment_miss_rate": round(missed / total, 6) if total else None,
        f"{prefix}_mean_coverage": round(sum(row[f"{prefix}_coverage"] for row in rows) / total, 6) if total else None,
    }


def evaluate(preds: dict) -> tuple[dict, list[dict], list[dict]]:
    root = repo_root()
    gt_rows = []
    event_rows = []
    event_counts = []
    micro_counts_per_event = []
    gap_durations = []
    gt_counts_per_event = []
    micro_counts_per_gt = []
    agent_review_cases = []

    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            video_len = estimate_video_length(score_metadata(scores), meta["intervals"])
            pred = video_prediction(preds, dataset, video_id)
            events = pred.get("events", [])
            merged = event_intervals(pred)
            micros = micro_intervals(pred)
            event_counts.append(len(events))
            for event in events:
                micro_counts_per_event.append(len(event.get("micro_intervals", [])))
                gap_durations.extend(gap["gap_duration"] for gap in event.get("gaps", []))
                covered_gts = sum(1 for gt in meta["intervals"] if overlaps(event, gt) > 0)
                gt_counts_per_event.append(covered_gts)
                mean_gap = statistics.mean([gap["gap_duration"] for gap in event.get("gaps", [])]) if event.get("gaps") else 0
                event_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "event_id": event["event_id"],
                        "merged_start": event["merged_start"],
                        "merged_end": event["merged_end"],
                        "merged_duration": event["merged_duration"],
                        "micro_interval_count": len(event.get("micro_intervals", [])),
                        "gap_count": len(event.get("gaps", [])),
                        "mean_gap_duration": round(mean_gap, 3),
                        "gt_intervals_covered": covered_gts,
                    }
                )
                if covered_gts >= 2 or (len(event.get("gaps", [])) >= 2 and mean_gap >= 60):
                    agent_review_cases.append(
                        {
                            "dataset": dataset,
                            "video_id": video_id,
                            "event_id": event["event_id"],
                            "reason": "merged_event_spans_multiple_gt_or_large_internal_gaps",
                            "gt_intervals_covered": covered_gts,
                            "micro_interval_count": len(event.get("micro_intervals", [])),
                            "mean_gap_duration": round(mean_gap, 3),
                        }
                    )
            for gt_idx, gt in enumerate(meta["intervals"], start=1):
                merged_coverage = coverage_by_intervals(gt, merged)
                micro_coverage = coverage_by_intervals(gt, micros)
                micro_hits = sum(1 for micro in micros if overlaps(gt, micro) > 0)
                event_hits = sum(1 for event in merged if overlaps(gt, event) > 0)
                micro_counts_per_gt.append(micro_hits)
                gt_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "anomaly_id": gt_idx,
                        "gt_start": gt["start"],
                        "gt_end": gt["end"],
                        "gt_duration": gt["end"] - gt["start"],
                        "video_length": video_len,
                        "merged_coverage": round(merged_coverage, 6),
                        "merged_missed": merged_coverage < 0.1,
                        "micro_coverage": round(micro_coverage, 6),
                        "micro_missed": micro_coverage < 0.1,
                        "merged_events_overlapping_gt": event_hits,
                        "micro_intervals_inside_gt": micro_hits,
                    }
                )
                if micro_hits >= 8:
                    agent_review_cases.append(
                        {
                            "dataset": dataset,
                            "video_id": video_id,
                            "anomaly_id": gt_idx,
                            "reason": "gt_interval_has_many_micro_fragments",
                            "micro_intervals_inside_gt": micro_hits,
                            "micro_coverage": round(micro_coverage, 6),
                        }
                    )

    summary = {}
    summary.update(summarize_coverage(gt_rows, "merged"))
    summary.update(summarize_coverage(gt_rows, "micro"))
    summary.update(
        {
            "video_count": len(event_counts),
            "mean_events_per_video": round(statistics.mean(event_counts), 6) if event_counts else 0,
            "median_events_per_video": round(statistics.median(event_counts), 6) if event_counts else 0,
            "max_events_per_video": max(event_counts) if event_counts else 0,
            "mean_micro_intervals_per_event": round(statistics.mean(micro_counts_per_event), 6) if micro_counts_per_event else 0,
            "median_micro_intervals_per_event": round(statistics.median(micro_counts_per_event), 6) if micro_counts_per_event else 0,
            "mean_gap_duration_inside_event": round(statistics.mean(gap_durations), 6) if gap_durations else 0,
            "median_gap_duration_inside_event": round(statistics.median(gap_durations), 6) if gap_durations else 0,
            "mean_gt_intervals_covered_by_each_merged_event": round(statistics.mean(gt_counts_per_event), 6) if gt_counts_per_event else 0,
            "max_gt_intervals_covered_by_each_merged_event": max(gt_counts_per_event) if gt_counts_per_event else 0,
            "mean_micro_intervals_inside_each_gt": round(statistics.mean(micro_counts_per_gt), 6) if micro_counts_per_gt else 0,
            "max_micro_intervals_inside_each_gt": max(micro_counts_per_gt) if micro_counts_per_gt else 0,
            "events_covering_multiple_gt": sum(1 for value in gt_counts_per_event if value >= 2),
            "gt_with_many_micro_fragments": sum(1 for value in micro_counts_per_gt if value >= 8),
            "agent_review_case_count": len(agent_review_cases),
            "agent_review_cases": agent_review_cases[:50],
        }
    )
    return summary, gt_rows, event_rows


def judgement_text(summary: dict) -> list[str]:
    micro_good = summary["micro_segment_miss_rate"] is not None and summary["micro_segment_miss_rate"] <= 0.2
    merged_good = summary["merged_segment_miss_rate"] is not None and summary["merged_segment_miss_rate"] <= 0.2
    overmerge = summary["events_covering_multiple_gt"] > 0
    fragmented = summary["gt_with_many_micro_fragments"] > 0 or summary["mean_micro_intervals_per_event"] >= 10
    return [
        "Yes" if micro_good else "Partially",
        "Yes" if merged_good else "Partially",
        "Yes" if overmerge else "Limited",
        "Yes" if fragmented else "Limited",
    ]


def write_report(path: Path, summary: dict, input_path: Path) -> None:
    spike, gt_block, overmerge, fragmented = judgement_text(summary)
    lines = [
        "# Hierarchical Interval Report",
        "",
        "## Input",
        "",
        f"- prediction_file: `{input_path}`",
        "",
        "## Coverage",
        "",
        f"- merged event segment miss rate: {summary['merged_segment_miss_rate']:.2%}",
        f"- merged event mean coverage: {summary['merged_mean_coverage']:.3f}",
        f"- micro interval segment miss rate: {summary['micro_segment_miss_rate']:.2%}",
        f"- micro interval mean coverage: {summary['micro_mean_coverage']:.3f}",
        "",
        "## Structure",
        "",
        f"- mean events per video: {summary['mean_events_per_video']:.3f}",
        f"- mean micro intervals per event: {summary['mean_micro_intervals_per_event']:.3f}",
        f"- mean gap duration inside event: {summary['mean_gap_duration_inside_event']:.3f}",
        f"- mean GT intervals covered by each merged event: {summary['mean_gt_intervals_covered_by_each_merged_event']:.3f}",
        f"- mean micro intervals inside each GT interval: {summary['mean_micro_intervals_inside_each_gt']:.3f}",
        "",
        "## Questions",
        "",
        f"- micro proposals capture score spikes: {spike}. The direct signal is micro miss rate and micro mean coverage.",
        f"- merged events cover GT blocks: {gt_block}. The direct signal is merged miss rate and merged mean coverage.",
        f"- average micro intervals per merged event: {summary['mean_micro_intervals_per_event']:.3f}.",
        f"- over-merging exists: {overmerge}. Events covering multiple GT intervals: {summary['events_covering_multiple_gt']}.",
        f"- over-fragmentation exists: {fragmented}. GT intervals with at least 8 overlapping micro intervals: {summary['gt_with_many_micro_fragments']}.",
        f"- cases needing agent same-event/separate-event judgement: {summary['agent_review_case_count']}.",
        "",
        "## Agent Review Cases",
        "",
    ]
    for item in summary["agent_review_cases"][:20]:
        label = item.get("event_id", item.get("anomaly_id"))
        lines.append(f"- {item['dataset']} `{item['video_id']}` item={label}: {item['reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/hierarchical_intervals/hierarchical_intervals.json"))
    args = parser.parse_args()

    root = repo_root()
    preds = load_predictions(root / args.input)
    summary, gt_rows, event_rows = evaluate(preds)
    write_json(
        root / "outputs/hierarchical_intervals/hierarchical_interval_eval.json",
        {"summary": summary, "gt_rows": gt_rows, "event_rows": event_rows},
    )
    write_csv(root / "outputs/hierarchical_intervals/hierarchical_gt_coverage.csv", gt_rows)
    write_csv(root / "outputs/hierarchical_intervals/hierarchical_event_structure.csv", event_rows)
    write_report(root / "reports/hierarchical_interval_report.md", summary, args.input)
    print(json.dumps({"summary": summary, "report": "reports/hierarchical_interval_report.md"}, indent=2))


if __name__ == "__main__":
    main()
