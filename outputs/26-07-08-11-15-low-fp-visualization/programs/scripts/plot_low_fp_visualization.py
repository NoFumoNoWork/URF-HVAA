import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import write_json  # noqa: E402
from scripts.evaluate_interval_methods import group_gt, load_gt_rows, load_inventory, merge_ranges, write_csv  # noqa: E402
from scripts.run_spectral_final_materials import DEFAULT_CACHE_SOURCE  # noqa: E402
from scripts.run_spectral_param_scan import precompute_curves  # noqa: E402
from scripts.run_spectral_score_decomposition import group_positive_runs, local_find_peaks, safe_name  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-08-11-15-low-fp-visualization")
DEFAULT_SOURCE_ARCHIVE = Path("outputs/26-07-07-22-50-low-fp-ablation-scan")
DEFAULT_GT_STATS = Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv")
DEFAULT_GT_SUPPORT = Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv")
DEFAULT_INVENTORY = Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv")
LOW_MARKED_THRESHOLD = 0.25
HIGH_UNMARKED_THRESHOLD = 0.75
NORMAL_ANCHOR_MARGIN = 8

COLORS = {
    "score": "#333333",
    "supportable": "#2CA02C",
    "uncertain": "#F2B701",
    "unsupportable": "#D62728",
    "pi": "#4E79A7",
    "cut": "#E15759",
    "smooth": "#5B5B5B",
    "high_unmarked": "#C51B7D",
    "low_marked": "#17BECF",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_intervals(rows: list[dict], *, action: str | None = None, use_cut_margin: bool = False) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grouped = defaultdict(list)
    for row in rows:
        if action is not None and row.get("action") != action:
            continue
        key = (row["dataset"], row["video_id"])
        if use_cut_margin:
            start = max(int(float(row.get("pi_start", row["start"]))), int(float(row["start"])) - NORMAL_ANCHOR_MARGIN)
            end = min(int(float(row.get("pi_end", row["end"]))), int(float(row["end"])) + NORMAL_ANCHOR_MARGIN)
        else:
            start = int(float(row["start"]))
            end = int(float(row["end"]))
        if end > start:
            grouped[key].append((start, end))
    return {key: merge_ranges(value) for key, value in grouped.items()}


def ranges_for_gt(rows: list[dict], group: str) -> list[tuple[int, int]]:
    return merge_ranges([(int(row["start"]), int(row["end"])) for row in rows if row.get("support_group") == group])


def in_any_interval(frame: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= frame < end for start, end in intervals)


def video_limit(key, gt_rows, pi_ranges, cut_ranges, data) -> int:
    length = 0
    if data:
        frames = data["frames"]
        stride = int(data.get("stride", 16) or 16)
        if len(frames):
            length = max(length, int(frames[-1]) + stride)
    for row in gt_rows:
        length = max(length, int(row["end"]))
    for start, end in pi_ranges + cut_ranges:
        length = max(length, end)
    return length


def draw_span_row(ax, intervals: list[tuple[int, int]], color: str, alpha: float = 0.82) -> None:
    for start, end in intervals:
        ax.axvspan(start, end, ymin=0.18, ymax=0.82, color=color, alpha=alpha, linewidth=0)


def add_below_legend(ax, handles, ncol: int) -> None:
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.2,
    )


def plot_video(path: Path, key: tuple[str, str], data: dict, gt_rows: list[dict], pi_ranges: list[tuple[int, int]], cut_ranges: list[tuple[int, int]]) -> dict:
    frames = data["frames"]
    curves = data["curves"]
    stride = int(data.get("stride", 16) or 16)
    raw = curves["raw_score"]
    smooth = curves.get("rolling_mean_100", raw)
    supportable = ranges_for_gt(gt_rows, "supportable")
    uncertain = ranges_for_gt(gt_rows, "uncertain")
    unsupportable = ranges_for_gt(gt_rows, "unsupportable")

    high_peak_idx = local_find_peaks(smooth, HIGH_UNMARKED_THRESHOLD, 1e-6)
    high_unmarked = [idx for idx in high_peak_idx if smooth[idx] > HIGH_UNMARKED_THRESHOLD and not in_any_interval(int(frames[idx]), pi_ranges)]
    low_marked_mask = np.asarray([in_any_interval(int(frame), pi_ranges) and value < LOW_MARKED_THRESHOLD for frame, value in zip(frames, smooth)], dtype=bool)
    low_marked_ranges = group_positive_runs(frames, low_marked_mask, stride, max_gap_frames=stride * 2)

    limit = video_limit(key, gt_rows, pi_ranges, cut_ranges, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 0.78, 0.78, 2.0]},
    )
    fig.suptitle(f"{key[0]} / {key[1]} | Low-FP with valley cut final", fontsize=12, y=0.985)

    axes[0].plot(frames, raw, color=COLORS["score"], linewidth=1.2)
    for ranges, color in [(supportable, COLORS["supportable"]), (uncertain, COLORS["uncertain"]), (unsupportable, COLORS["unsupportable"])]:
        for start, end in ranges:
            axes[0].axvspan(start, end, color=color, alpha=0.23, linewidth=0)
    axes[0].set_ylabel("score + GT")
    axes[0].set_ylim(-0.04, max(1.04, float(np.max(raw)) + 0.08 if len(raw) else 1.04))
    add_below_legend(
        axes[0],
        [
            Line2D([0], [0], color=COLORS["score"], lw=1.5, label="anomaly score"),
            Patch(facecolor=COLORS["supportable"], alpha=0.35, label="supportable GT"),
            Patch(facecolor=COLORS["uncertain"], alpha=0.35, label="uncertain GT"),
            Patch(facecolor=COLORS["unsupportable"], alpha=0.35, label="unsupportable GT"),
        ],
        4,
    )

    draw_span_row(axes[1], pi_ranges, COLORS["pi"])
    axes[1].set_ylabel("final PI")
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    add_below_legend(axes[1], [Patch(facecolor=COLORS["pi"], alpha=0.82, label="final low-FP abnormal interval")], 1)

    draw_span_row(axes[2], cut_ranges, COLORS["cut"])
    axes[2].set_ylabel("valley cut")
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    add_below_legend(axes[2], [Patch(facecolor=COLORS["cut"], alpha=0.82, label="interval removed by valley cut")], 1)

    axes[3].plot(frames, smooth, color=COLORS["smooth"], linewidth=1.2)
    axes[3].axhline(HIGH_UNMARKED_THRESHOLD, color=COLORS["high_unmarked"], linewidth=0.8, linestyle="--", alpha=0.55)
    axes[3].axhline(LOW_MARKED_THRESHOLD, color=COLORS["low_marked"], linewidth=0.8, linestyle="--", alpha=0.55)
    if high_unmarked:
        axes[3].scatter(frames[high_unmarked], smooth[high_unmarked], s=26, color=COLORS["high_unmarked"], zorder=4)
    for start, end in low_marked_ranges:
        axes[3].axvspan(start, end, ymin=0.0, ymax=0.25, color=COLORS["low_marked"], alpha=0.40, linewidth=0)
    axes[3].set_ylabel("100F smooth")
    axes[3].set_ylim(-0.04, max(1.04, float(np.max(smooth)) + 0.08 if len(smooth) else 1.04))
    axes[3].set_xlabel("frame")
    add_below_legend(
        axes[3],
        [
            Line2D([0], [0], color=COLORS["smooth"], lw=1.5, label="100F smoothed score"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["high_unmarked"], markersize=6, label="unmarked peak > 0.75"),
            Patch(facecolor=COLORS["low_marked"], alpha=0.40, label="marked segment < 0.25"),
        ],
        3,
    )

    for ax in axes:
        ax.set_xlim(0, max(1, limit))
        ax.grid(axis="x", alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.99, hspace=0.78)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "dataset": key[0],
        "video_id": key[1],
        "plot_path": str(path).replace("\\", "/"),
        "gt_supportable_count": len(supportable),
        "gt_uncertain_count": len(uncertain),
        "gt_unsupportable_count": len(unsupportable),
        "final_pi_count": len(pi_ranges),
        "valley_cut_count": len(cut_ranges),
        "high_unmarked_peak_count": len(high_unmarked),
        "low_marked_segment_count": len(low_marked_ranges),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source_archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--gt_stats_csv", type=Path, default=DEFAULT_GT_STATS)
    parser.add_argument("--gt_support_csv", type=Path, default=DEFAULT_GT_SUPPORT)
    parser.add_argument("--video_inventory_csv", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--max_plots", type=int, default=0, help="0 means all eligible videos")
    parser.add_argument("--reuse_cached_curves", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    reports = args.output_dir / "reports"
    plots = args.output_dir / "outputs" / "low_fp_visualizations"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    gt_by_video = group_gt(gt_rows)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    warnings = []
    pre_args = SimpleNamespace(output_dir=DEFAULT_CACHE_SOURCE.parent.parent if DEFAULT_CACHE_SOURCE.exists() else args.output_dir / "outputs", reuse_cached_curves=args.reuse_cached_curves)
    decomp = precompute_curves(pre_args, inventory, warnings)

    source_reports = args.source_archive / "reports"
    pi_rows = read_csv(source_reports / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv")
    event_rows = read_csv(source_reports / "negative_evidence_events_final.csv")
    pi_by_video = group_intervals(pi_rows)
    cut_by_video = group_intervals(event_rows, action="cut_pi", use_cut_margin=True)
    keys = sorted((set(gt_by_video) | set(pi_by_video)) & set(decomp))
    if args.max_plots and args.max_plots > 0:
        keys = keys[: args.max_plots]

    index_rows = []
    for idx, key in enumerate(keys, start=1):
        filename = f"{idx:04d}_{key[0]}_{safe_name(key[1])}.png"
        row = plot_video(plots / filename, key, decomp[key], gt_by_video.get(key, []), pi_by_video.get(key, []), cut_by_video.get(key, []))
        index_rows.append(row)

    write_csv(
        reports / "low_fp_visualization_index.csv",
        index_rows,
        [
            "dataset",
            "video_id",
            "plot_path",
            "gt_supportable_count",
            "gt_uncertain_count",
            "gt_unsupportable_count",
            "final_pi_count",
            "valley_cut_count",
            "high_unmarked_peak_count",
            "low_marked_segment_count",
        ],
    )
    write_json(
        args.output_dir / "low_fp_visualization_summary.json",
        {
            "plot_count": len(index_rows),
            "output_dir": str(args.output_dir).replace("\\", "/"),
            "plots_dir": str(plots).replace("\\", "/"),
            "source_archive": str(args.source_archive).replace("\\", "/"),
            "warnings": len(warnings),
            "layout": [
                "row1 anomaly score + supportable/uncertain/unsupportable GT",
                "row2 final low-FP abnormal intervals",
                "row3 valley-cut removed intervals",
                "row4 100F smoothed score with unmarked >0.75 peaks and marked <0.25 segments",
            ],
        },
    )
    (args.output_dir / "low-fp-visualization_report.md").write_text(
        "\n".join(
            [
                "# Low-FP Visualization Batch",
                "",
                f"- Source archive: `{str(args.source_archive).replace(chr(92), '/')}`",
                f"- Plot count: {len(index_rows)}",
                f"- Plots: `{str(plots).replace(chr(92), '/')}`",
                "- Row 1: anomaly score with supportable, uncertain, and unsupportable GT colors.",
                "- Row 2: intervals finally marked abnormal by low-FP with valley cut.",
                "- Row 3: intervals removed by valley cut.",
                "- Row 4: 100F smoothed curve with unmarked peaks above 0.75 and marked low-score portions below 0.25.",
                "- Color legends are placed below the corresponding subplot boxes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "MANIFEST.md").write_text(
        "\n".join(
            [
                "# MANIFEST",
                "",
                "- `low-fp-visualization_report.md`: archive report.",
                "- `outputs/low_fp_visualizations/*.png`: per-video low-FP visualization plots.",
                "- `reports/low_fp_visualization_index.csv`: plot index and marker counts.",
                "- `low_fp_visualization_summary.json`: machine-readable summary.",
                "- `programs/scripts/plot_low_fp_visualization.py`: copied generator script.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    program = args.output_dir / "programs" / "scripts" / Path(__file__).name
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program)
    print(json.dumps({"plot_count": len(index_rows), "output_dir": str(args.output_dir).replace("\\", "/")}, indent=2))


if __name__ == "__main__":
    main()
