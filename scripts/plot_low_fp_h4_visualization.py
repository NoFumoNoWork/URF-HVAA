import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-09-16-18-low-fp-h4-visualization")

COLORS = {
    "raw": "#2B2B2B",
    "smooth": "#616161",
    "supportable": "#2CA02C",
    "uncertain": "#F2B701",
    "unsupportable": "#D62728",
    "final": "#4E79A7",
    "cut": "#E15759",
    "h4_gap": "#7B61FF",
    "h4_point": "#3B1FA3",
    "high": "#C51B7D",
    "low": "#17BECF",
}
LOW_MARKED_THRESHOLD = 0.25
HIGH_UNMARKED_THRESHOLD = 0.75


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def legend_below(ax, handles, ncol: int) -> None:
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.1,
    )


def spans(ax, rows: list[dict], color: str, ymin=0.18, ymax=0.82, alpha=0.82, start_key="start", end_key="end") -> None:
    for row in rows:
        start = as_int(row.get(start_key))
        end = as_int(row.get(end_key))
        if end > start:
            ax.axvspan(start, end, ymin=ymin, ymax=ymax, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    curve_rows = read_csv(args.output_dir / "case_curve_data.csv")
    gt_rows = read_csv(args.output_dir / "case_gt_intervals.csv")
    final_rows = read_csv(args.output_dir / "case_final_intervals.csv")
    cut_rows = read_csv(args.output_dir / "case_valley_cut_intervals.csv")
    h4_gap_rows = read_csv(args.output_dir / "case_h4_gaps.csv")
    h4_candidate_rows = read_csv(args.output_dir / "case_h4_candidates.csv")

    frames = [as_int(row["frame"]) for row in curve_rows]
    raw = [as_float(row["raw_score"]) for row in curve_rows]
    smooth = [as_float(row["smooth_100"]) for row in curve_rows]
    high_frames = [as_int(row["frame"]) for row in curve_rows if row.get("high_unmarked_peak") == "1"]
    high_values = [as_float(row["smooth_100"]) for row in curve_rows if row.get("high_unmarked_peak") == "1"]
    low_frames = [as_int(row["frame"]) for row in curve_rows if row.get("low_marked_point") == "1"]
    dataset = curve_rows[0]["dataset"] if curve_rows else "unknown"
    video_id = curve_rows[0]["video_id"] if curve_rows else "unknown"
    x_limit = max(frames + [as_int(row.get("end")) for row in final_rows + cut_rows + h4_gap_rows] + [1])

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 0.78, 0.98, 2.0]},
    )
    fig.suptitle(f"{dataset} / {video_id} | Low-FP intervals with H4 gap-level proxy overlay", fontsize=12, y=0.986)

    axes[0].plot(frames, raw, color=COLORS["raw"], linewidth=1.15)
    for role in ["supportable", "uncertain", "unsupportable"]:
        spans(axes[0], [row for row in gt_rows if row.get("interval_role") == role], COLORS[role], ymin=0, ymax=1, alpha=0.22)
    axes[0].set_ylabel("score + GT")
    axes[0].set_ylim(-0.04, max(1.04, max(raw or [1.0]) + 0.08))
    legend_below(
        axes[0],
        [
            Line2D([0], [0], color=COLORS["raw"], lw=1.5, label="anomaly score"),
            Patch(facecolor=COLORS["supportable"], alpha=0.35, label="supportable GT"),
            Patch(facecolor=COLORS["uncertain"], alpha=0.35, label="uncertain GT"),
            Patch(facecolor=COLORS["unsupportable"], alpha=0.35, label="unsupportable GT"),
        ],
        4,
    )

    spans(axes[1], final_rows, COLORS["final"])
    axes[1].set_ylabel("final PI")
    axes[1].set_yticks([])
    axes[1].set_ylim(0, 1)
    legend_below(axes[1], [Patch(facecolor=COLORS["final"], alpha=0.82, label="final low-FP abnormal interval")], 1)

    spans(axes[2], cut_rows, COLORS["cut"], ymin=0.10, ymax=0.48, alpha=0.78)
    spans(axes[2], h4_gap_rows, COLORS["h4_gap"], ymin=0.54, ymax=0.92, alpha=0.55)
    for row in h4_candidate_rows:
        x = as_int(row.get("h4_position"))
        if x > 0:
            axes[2].axvline(x, ymin=0.52, ymax=0.95, color=COLORS["h4_point"], linewidth=0.8, alpha=0.55)
    axes[2].set_ylabel("cut + H4")
    axes[2].set_yticks([])
    axes[2].set_ylim(0, 1)
    legend_below(
        axes[2],
        [
            Patch(facecolor=COLORS["cut"], alpha=0.78, label="valley-cut removed interval"),
            Patch(facecolor=COLORS["h4_gap"], alpha=0.55, label="H4 bridge/recheck candidate gap"),
            Line2D([0], [0], color=COLORS["h4_point"], lw=1.2, label="H4 candidate position"),
        ],
        3,
    )

    axes[3].plot(frames, smooth, color=COLORS["smooth"], linewidth=1.15)
    axes[3].axhline(HIGH_UNMARKED_THRESHOLD, color=COLORS["high"], linewidth=0.8, linestyle="--", alpha=0.55)
    axes[3].axhline(LOW_MARKED_THRESHOLD, color=COLORS["low"], linewidth=0.8, linestyle="--", alpha=0.55)
    if high_frames:
        axes[3].scatter(high_frames, high_values, color=COLORS["high"], s=28, zorder=4)
    if low_frames:
        axes[3].scatter(low_frames, [LOW_MARKED_THRESHOLD * 0.36] * len(low_frames), color=COLORS["low"], s=8, alpha=0.42, marker="s", zorder=3)
    axes[3].set_ylabel("100F smooth")
    axes[3].set_xlabel("frame")
    axes[3].set_ylim(-0.04, max(1.04, max(smooth or [1.0]) + 0.08))
    legend_below(
        axes[3],
        [
            Line2D([0], [0], color=COLORS["smooth"], lw=1.5, label="100F smoothed score"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["high"], markersize=6, label="unmarked peak > 0.75"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["low"], markersize=6, label="marked point < 0.25"),
        ],
        3,
    )

    for ax in axes:
        ax.set_xlim(0, max(1, x_limit))
        ax.grid(axis="x", alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.99, hspace=0.84)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.output_dir / f"fig_low_fp_case_visualization.{ext}", dpi=160 if ext == "png" else None)
    plt.close(fig)

    (args.output_dir / "figure_notes.md").write_text(
        "\n".join(
            [
                "# Figure Notes",
                "",
                f"- Case: `{dataset}` / `{video_id}`.",
                "- Row 1 shows the raw anomaly score with supportable, uncertain, and unsupportable GT intervals.",
                "- Row 2 shows final abnormal intervals from the selected low-FP configuration.",
                "- Row 3 separates valley-cut removals from H4 bridge/recheck candidate gaps. H4 is an overlay from caption/gap resources, not an enabled frame-level fusion input.",
                "- Row 4 shows the 100-frame smoothed score with high unmarked peaks above `0.75` and marked low-score points below `0.25`.",
                "- The figure supports case-level discussion of why H4 should be treated as an interval merge/recheck signal, while retaining the limitation that raw video and shot-boundary labels are required for visual claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), args.output_dir / "programs" / "scripts" / Path(__file__).name)
    print((args.output_dir / "fig_low_fp_case_visualization.png").as_posix())


if __name__ == "__main__":
    main()
