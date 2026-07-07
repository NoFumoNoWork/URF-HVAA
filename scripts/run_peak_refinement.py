import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    safe_name,
    score_metadata,
    score_path_for_video,
    write_json,
)
from src.peak_refinement import (  # noqa: E402
    compute_interval_peak_features,
    detect_peaks,
    estimate_baseline,
    expand_peak_intervals,
    merge_intervals,
    rescue_peak_intervals,
    split_merged_intervals_by_peak_gaps,
    summarize_sources,
)


DEFAULT_ARCHIVE = Path("outputs/26-07-06-08-39-hierarchical-intervals")
DEFAULT_CASES = [
    ("XD-Violence", "v=38GQ9L2meyE__#1_label_B6-0-0"),
    ("XD-Violence", "v=uQY15O3LKI0__#1_label_B6-0-0"),
    ("UCF-Crime", "Assault010_x264"),
]


def load_predictions(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing hierarchical interval JSON: {path}")
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
    return {"dataset": dataset, "video_id": video_id, "events": []}


def require_events(video_pred: dict, dataset: str, video_id: str) -> None:
    if "events" not in video_pred:
        raise KeyError(f"hierarchical prediction for {dataset}::{video_id} is missing required field: events")


def frame_end_for_index(frames: list[int], idx: int, stride: int) -> int:
    if idx < len(frames):
        return int(frames[idx])
    return int(frames[-1] + stride) if frames else 0


def frame_interval_to_index(interval: dict, frames: list[int]) -> dict:
    start = int(np.searchsorted(frames, int(interval["start"]), side="left"))
    end = int(np.searchsorted(frames, int(interval["end"]), side="left"))
    if end <= start:
        end = min(len(frames), start + 1)
    result = dict(interval)
    result["start"] = max(0, min(len(frames), start))
    result["end"] = max(0, min(len(frames), end))
    result["duration"] = result["end"] - result["start"]
    return result


def index_interval_to_frame(interval: dict, frames: list[int], stride: int) -> dict:
    start_idx = max(0, min(len(frames), int(interval["start"])))
    end_idx = max(start_idx, min(len(frames), int(interval["end"])))
    result = dict(interval)
    result["start_index"] = start_idx
    result["end_index"] = end_idx
    result["start"] = frame_end_for_index(frames, start_idx, stride)
    result["end"] = frame_end_for_index(frames, end_idx, stride)
    result["duration"] = result["end"] - result["start"]
    if "peak_index" in result:
        peak_idx = max(0, min(len(frames) - 1, int(result["peak_index"]))) if frames else 0
        result["peak_score_index"] = int(result["peak_index"])
        result["peak_index"] = int(frames[peak_idx]) if frames else int(result["peak_index"])
    return result


def peak_to_frame(peak: dict, frames: list[int]) -> dict:
    peak_idx = int(peak["peak_index"])
    result = dict(peak)
    result["score_index"] = peak_idx
    result["peak_index"] = int(frames[peak_idx]) if 0 <= peak_idx < len(frames) else peak_idx
    for key in ["left_base", "right_base", "left_boundary", "right_boundary"]:
        idx = int(result[key])
        result[f"{key}_score_index"] = idx
        if 0 <= idx < len(frames):
            result[key] = int(frames[idx])
    return result


def extract_intervals(video_pred: dict) -> tuple[list[dict], list[dict]]:
    merged = []
    micro = []
    for event in video_pred.get("events", []):
        if not {"merged_start", "merged_end"}.issubset(event):
            raise KeyError("hierarchical event is missing merged_start or merged_end")
        merged.append(
            {
                "start": int(event["merged_start"]),
                "end": int(event["merged_end"]),
                "duration": int(event["merged_end"]) - int(event["merged_start"]),
                "event_id": event.get("event_id"),
                "source": "merged_original",
            }
        )
        for item in event.get("micro_intervals", []):
            if not {"start", "end"}.issubset(item):
                raise KeyError("micro interval is missing start or end")
            micro_item = dict(item)
            micro_item["source"] = "micro_original"
            micro_item["event_id"] = event.get("event_id")
            micro.append(micro_item)
    return micro, merged


def refine_video(dataset: str, video_id: str, video_pred: dict, scores: dict[int, float], args: argparse.Namespace) -> dict:
    require_events(video_pred, dataset, video_id)
    if not scores:
        raise ValueError(f"missing score sequence for {dataset}::{video_id}")
    frames = sorted(scores)
    values = np.asarray([scores[frame] for frame in frames], dtype=float)
    meta = score_metadata(scores)
    stride = int(meta.get("score_stride_est") or 16)

    micro_frame, merged_frame = extract_intervals(video_pred)
    micro_idx = [frame_interval_to_index(item, frames) for item in micro_frame]
    merged_idx = [frame_interval_to_index(item, frames) for item in merged_frame]

    config = {
        "baseline_window": args.baseline_window,
        "baseline_method": args.baseline_method,
        "baseline_quantile": args.baseline_quantile,
        "peak_mad_k": args.peak_mad_k,
        "peak_min_width": args.peak_min_width,
        "peak_min_distance": args.peak_min_distance,
        "peak_stop_ratio": args.peak_stop_ratio,
        "peak_rescue_min_prominence": args.peak_rescue_min_prominence,
        "peak_rescue_min_area": args.peak_rescue_min_area,
        "peak_split_min_gap_len": args.peak_split_min_gap_len,
    }
    baseline = estimate_baseline(values, window=args.baseline_window, method=args.baseline_method, quantile=args.baseline_quantile)
    peaks_idx = detect_peaks(
        values,
        baseline=baseline,
        min_prominence=args.peak_min_prominence,
        min_height=args.peak_min_height,
        min_width=args.peak_min_width,
        min_distance=args.peak_min_distance,
        mad_k=args.peak_mad_k,
    )
    peak_intervals_idx = expand_peak_intervals(
        values,
        baseline,
        peaks_idx,
        stop_ratio=args.peak_stop_ratio,
        min_len=args.peak_min_len,
        max_len=args.peak_max_len,
    )
    rescued_idx = rescue_peak_intervals(
        peak_intervals_idx,
        micro_idx,
        merged_idx,
        min_prominence=args.peak_rescue_min_prominence,
        min_area=args.peak_rescue_min_area,
        overlap_tolerance=args.overlap_tolerance,
    )
    split_idx, split_diagnostics_idx = split_merged_intervals_by_peak_gaps(
        merged_idx,
        micro_idx,
        peaks_idx,
        values,
        baseline,
        min_gap_len=args.peak_split_min_gap_len,
        low_score_quantile=args.peak_split_low_score_quantile,
    )
    refined_idx = merge_intervals(split_idx + rescued_idx)
    for item in micro_idx:
        item["peak_features"] = compute_interval_peak_features(item, peaks_idx, values, baseline)
    for item in merged_idx:
        item["peak_features"] = compute_interval_peak_features(item, peaks_idx, values, baseline)
    for item in refined_idx:
        item["peak_features"] = compute_interval_peak_features(item, peaks_idx, values, baseline)

    peaks_frame = [peak_to_frame(item, frames) for item in peaks_idx]
    peak_intervals_frame = [index_interval_to_frame(item, frames, stride) for item in peak_intervals_idx]
    rescued_frame = [index_interval_to_frame(item, frames, stride) for item in rescued_idx]
    refined_frame = [index_interval_to_frame(item, frames, stride) for item in refined_idx]
    micro_features_frame = [index_interval_to_frame(item, frames, stride) for item in micro_idx]
    merged_features_frame = [index_interval_to_frame(item, frames, stride) for item in merged_idx]
    split_diagnostics_frame = []
    for item in split_diagnostics_idx:
        converted = dict(item)
        converted["original_merged_score_index"] = item["original_merged"]
        converted["split_gap_score_index"] = item["split_gap"]
        converted["original_merged"] = [
            frame_end_for_index(frames, item["original_merged"][0], stride),
            frame_end_for_index(frames, item["original_merged"][1], stride),
        ]
        converted["split_gap"] = [
            frame_end_for_index(frames, item["split_gap"][0], stride),
            frame_end_for_index(frames, item["split_gap"][1], stride),
        ]
        split_diagnostics_frame.append(converted)

    return {
        "video_id": video_id,
        "dataset": dataset,
        "score_length": len(values),
        "score_frame_min": frames[0],
        "score_frame_max": frames[-1],
        "score_stride_est": stride,
        "config": config,
        "original_micro_intervals": [[item["start"], item["end"]] for item in micro_frame],
        "original_merged_intervals": [[item["start"], item["end"]] for item in merged_frame],
        "micro_intervals_with_peak_features": micro_features_frame,
        "merged_intervals_with_peak_features": merged_features_frame,
        "peaks": peaks_frame,
        "peak_expanded_intervals": peak_intervals_frame,
        "rescued_intervals": rescued_frame,
        "split_diagnostics": split_diagnostics_frame,
        "refined_intervals": refined_frame,
        "summary": {
            "num_peaks": len(peaks_frame),
            "num_peak_expanded": len(peak_intervals_frame),
            "num_rescued": len(rescued_frame),
            "num_splits": len(split_diagnostics_frame),
            "num_original_micro": len(micro_frame),
            "num_original_merged": len(merged_frame),
            "num_refined": len(refined_frame),
            "refined_sources": summarize_sources(refined_frame),
        },
    }


def draw_bars(ax, intervals: list[dict], color: str, alpha: float, start_key: str = "start", end_key: str = "end") -> None:
    for item in intervals:
        start = int(item[start_key])
        end = int(item[end_key])
        if end > start:
            ax.broken_barh([(start, end - start)], (0, 1), facecolors=color, alpha=alpha)


def draw_visualization(dataset: str, video_id: str, diagnostic: dict, scores: dict[int, float], annotations: dict, output_dir: Path) -> dict:
    meta = annotations.get(video_id, {"intervals": [], "label": "unknown"})
    frames = list(scores.keys())
    values = list(scores.values())
    video_len = estimate_video_length(score_metadata(scores), meta.get("intervals", []))
    fig, axes = plt.subplots(7, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 2]})
    tracks = [
        ("GT", meta.get("intervals", []), "#d95f02", 0.9),
        ("Micro", [{"start": s, "end": e} for s, e in diagnostic["original_micro_intervals"]], "#7570b3", 0.5),
        ("Merged", [{"start": s, "end": e} for s, e in diagnostic["original_merged_intervals"]], "#1b9e77", 0.72),
        ("Peak-expanded", diagnostic["peak_expanded_intervals"], "#e7298a", 0.48),
        ("Rescued", diagnostic["rescued_intervals"], "#fdae61", 0.75),
        ("Refined", diagnostic["refined_intervals"], "#2c7fb8", 0.78),
    ]
    for ax, (label, intervals, color, alpha) in zip(axes[:-1], tracks):
        ax.set_ylabel(label)
        ax.set_yticks([])
        draw_bars(ax, intervals, color=color, alpha=alpha)
        for peak in diagnostic["peaks"]:
            ax.axvline(peak["peak_index"], color="#444444", alpha=0.15, linewidth=0.8)

    axes[-1].plot(frames, values, color="#386cb0", linewidth=1.2, label="raw score")
    peak_frames = [peak["peak_index"] for peak in diagnostic["peaks"]]
    if peak_frames:
        peak_values = [scores.get(frame, 0) for frame in peak_frames]
        axes[-1].scatter(peak_frames, peak_values, marker="^", color="#e7298a", s=45, label="peaks", zorder=3)
    axes[-1].set_ylabel("score")
    axes[-1].set_xlabel("frame")
    axes[-1].grid(alpha=0.25)
    axes[-1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlim(0, max(video_len, max(frames)))
    label = meta.get("label", "unknown")
    fig.suptitle(
        f"{dataset} | {video_id} | {label} | peaks={diagnostic['summary']['num_peaks']} | "
        f"rescued={diagnostic['summary']['num_rescued']} | splits={diagnostic['summary']['num_splits']}",
        fontsize=10,
    )
    fig.tight_layout()
    out = output_dir / dataset / f"{safe_name(video_id)}_peak_refined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"dataset": dataset, "video_id": video_id, "status": "ok", "output": str(out.relative_to(repo_root()))}


def write_report(path: Path, diagnostics: dict, visual_results: list[dict], args: argparse.Namespace) -> None:
    summaries = [item["summary"] for item in diagnostics.values()]
    total_videos = len(summaries)
    total_peaks = sum(item["num_peaks"] for item in summaries)
    total_rescued = sum(item["num_rescued"] for item in summaries)
    total_splits = sum(item["num_splits"] for item in summaries)
    lines = [
        "# Peak-Aware Refinement Report",
        "",
        "## Method",
        "",
        "This run preserves raw anomaly scores for peak height and local maxima. The baseline is used only as local background; residual is `raw_score - baseline`.",
        "",
        "## Config",
        "",
        f"- baseline: `{args.baseline_method}`, window={args.baseline_window}, quantile={args.baseline_quantile}",
        f"- peak detection: mad_k={args.peak_mad_k}, min_width={args.peak_min_width}, min_distance={args.peak_min_distance}",
        f"- peak expansion: stop_ratio={args.peak_stop_ratio}, min_len={args.peak_min_len}",
        f"- rescue: min_prominence={args.peak_rescue_min_prominence}, min_area={args.peak_rescue_min_area}",
        f"- split: min_gap_len={args.peak_split_min_gap_len}, low_score_quantile={args.peak_split_low_score_quantile}",
        "",
        "## Summary",
        "",
        f"- videos processed: {total_videos}",
        f"- total peaks: {total_peaks}",
        f"- total rescued intervals: {total_rescued}",
        f"- total split gaps: {total_splits}",
        f"- videos with no peaks: {sum(item['num_peaks'] == 0 for item in summaries)}",
        "",
        "## Visualizations",
        "",
    ]
    for item in visual_results:
        lines.append(f"- {item['dataset']} `{item['video_id']}`: {item['status']} {item.get('output', '')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_peak_refine", action="store_true", default=True, help="Dedicated script defaults to enabled.")
    parser.add_argument("--hierarchical_input", type=Path, default=DEFAULT_ARCHIVE / "outputs/hierarchical_intervals/hierarchical_intervals.json")
    parser.add_argument("--output_root", type=Path, default=Path("outputs/26-07-07-01-08-peak-aware-refinement"))
    parser.add_argument("--baseline_window", type=int, default=101)
    parser.add_argument("--baseline_method", choices=["median", "quantile"], default="median")
    parser.add_argument("--baseline_quantile", type=float, default=0.3)
    parser.add_argument("--peak_mad_k", type=float, default=2.5)
    parser.add_argument("--peak_min_prominence", type=float, default=None)
    parser.add_argument("--peak_min_height", type=float, default=None)
    parser.add_argument("--peak_min_width", type=int, default=3)
    parser.add_argument("--peak_min_distance", type=int, default=10)
    parser.add_argument("--peak_stop_ratio", type=float, default=0.2)
    parser.add_argument("--peak_min_len", type=int, default=3)
    parser.add_argument("--peak_max_len", type=int, default=None)
    parser.add_argument("--peak_rescue_min_prominence", type=float, default=0.2)
    parser.add_argument("--peak_rescue_min_area", type=float, default=0.3)
    parser.add_argument("--overlap_tolerance", type=int, default=0)
    parser.add_argument("--peak_split_min_gap_len", type=int, default=50)
    parser.add_argument("--peak_split_low_score_quantile", type=float, default=0.4)
    parser.add_argument("--case", action="append", help="dataset::video_id to visualize")
    args = parser.parse_args()

    if not args.enable_peak_refine:
        raise SystemExit("peak refinement is disabled; pass --enable_peak_refine or use this script defaults")

    root = repo_root()
    preds = load_predictions(root / args.hierarchical_input)
    output_root = root / args.output_root
    output_dir = output_root / "outputs/peak_refinement"
    visual_dir = output_dir / "visualizations"
    diagnostics = {}
    refined = {}
    annotations_by_dataset = {
        dataset: parse_temporal_annotations(root / cfg["annotation"], dataset)
        for dataset, cfg in DATASET_CONFIGS.items()
    }

    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = annotations_by_dataset[dataset]
        for video_id in sorted(annotations):
            video_pred = video_prediction(preds, dataset, video_id)
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            diagnostic = refine_video(dataset, video_id, video_pred, scores, args)
            key = video_id if video_id not in diagnostics else f"{dataset}::{video_id}"
            diagnostics[key] = diagnostic
            refined[key] = {
                "dataset": dataset,
                "video_id": video_id,
                "refined_intervals": diagnostic["refined_intervals"],
                "rescued_intervals": diagnostic["rescued_intervals"],
                "split_diagnostics": diagnostic["split_diagnostics"],
                "summary": diagnostic["summary"],
            }

    write_json(output_dir / "peak_refinement_report.json", diagnostics)
    write_json(output_dir / "refined_intervals.json", refined)

    if args.case:
        cases = [tuple(item.split("::", 1)) for item in args.case]
    else:
        cases = DEFAULT_CASES
    visual_results = []
    for dataset, video_id in cases:
        cfg = DATASET_CONFIGS[dataset]
        scores = load_scores(score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]]))
        diagnostic = diagnostics.get(video_id) or diagnostics.get(f"{dataset}::{video_id}")
        if diagnostic is None or not scores:
            visual_results.append({"dataset": dataset, "video_id": video_id, "status": "missing diagnostic or score"})
            continue
        visual_results.append(draw_visualization(dataset, video_id, diagnostic, scores, annotations_by_dataset[dataset], visual_dir))
    write_json(output_dir / "visualization_peak_refined.json", visual_results)
    write_report(output_root / "peak-aware-refinement_report.md", diagnostics, visual_results, args)

    print(
        json.dumps(
            {
                "videos": len(diagnostics),
                "report": str((output_root / "peak-aware-refinement_report.md").relative_to(root)),
                "diagnostics": str((output_dir / "peak_refinement_report.json").relative_to(root)),
                "visualizations": visual_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
