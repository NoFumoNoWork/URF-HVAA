import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_ARCHIVE = Path("outputs/26-07-07-22-50-low-fp-ablation-scan")
DEFAULT_GT_SUPPORT = Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv")
SCAN_NAMES = [
    "fusion_threshold",
    "merge_gap_frames",
    "post_min_duration",
    "post_min_raw_max",
    "valley_low_score_threshold",
    "valley_min_normal_duration",
]
AB_FIELDS = [
    "Variant",
    "PI Ratio",
    "s-GT Recall",
    "uc-GT Recall",
    "Eval Recall",
    "GT Precision",
    "FP Duration",
    "FP Ratio in PI",
    "Delta Eval Recall vs final_full",
    "Delta GT Precision vs final_full",
    "Delta FP Duration vs final_full",
]
SCAN_FIELDS = [
    "Parameter",
    "Value",
    "PI Ratio",
    "s-GT Recall",
    "uc-GT Recall",
    "Eval Recall",
    "GT Precision",
    "FP Duration",
    "FP Ratio in PI",
    "us-GT Coverage",
    "FP Removed by NI",
    "TP Lost by NI",
    "NI-over-sGT Ratio",
]
NEAR_FAR_FIELDS = [
    "method",
    "margin",
    "PI Duration",
    "FP Duration",
    "near_GT_FP_duration",
    "far_GT_FP_duration",
    "near_GT_FP_ratio_in_FP",
    "far_GT_FP_ratio_in_FP",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def num(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=3) -> str:
    return f"{num(value):.{digits}f}"


def interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def subtract_ranges(interval: tuple[int, int], blockers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces = [interval]
    for blocker in blockers:
        next_pieces = []
        for start, end in pieces:
            if blocker[1] <= start or blocker[0] >= end:
                next_pieces.append((start, end))
                continue
            if start < blocker[0]:
                next_pieces.append((start, max(start, blocker[0])))
            if blocker[1] < end:
                next_pieces.append((min(end, blocker[1]), end))
        pieces = [(start, end) for start, end in next_pieces if end > start]
    return pieces


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted((int(start), int(end)) for start, end in ranges if int(end) > int(start))
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def duration(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def load_gt(gt_support_csv: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    by_video: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row in read_csv(gt_support_csv):
        key = (row["dataset"], row["video_id"])
        by_video.setdefault(key, []).append((int(float(row["gt_start"])), int(float(row["gt_end"]))))
    return {key: merge_ranges(ranges) for key, ranges in by_video.items()}


def compute_near_far(pi_rows: list[dict], gt_by_video: dict[tuple[str, str], list[tuple[int, int]]], margin: int) -> dict:
    near = 0
    far = 0
    fp_total = 0
    pi_total = 0
    for row in pi_rows:
        key = (row["dataset"], row["video_id"])
        interval = (int(float(row["start"])), int(float(row["end"])))
        pi_total += interval[1] - interval[0]
        gt_ranges = gt_by_video.get(key, [])
        non_gt_pieces = subtract_ranges(interval, gt_ranges)
        buffered_gt = merge_ranges([(start - margin, end + margin) for start, end in gt_ranges])
        for piece in non_gt_pieces:
            piece_duration = piece[1] - piece[0]
            near_duration = sum(interval_overlap(piece, buffered) for buffered in buffered_gt)
            near_duration = min(piece_duration, near_duration)
            far_duration = max(0, piece_duration - near_duration)
            near += near_duration
            far += far_duration
            fp_total += piece_duration
    return {
        "method": pi_rows[0]["method"] if pi_rows else "",
        "margin": margin,
        "PI Duration": pi_total,
        "FP Duration": fp_total,
        "near_GT_FP_duration": near,
        "far_GT_FP_duration": far,
        "near_GT_FP_ratio_in_FP": near / fp_total if fp_total else 0,
        "far_GT_FP_ratio_in_FP": far / fp_total if fp_total else 0,
    }


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if field.endswith("ratio_in_FP") or field in {"PI Ratio", "s-GT Recall", "uc-GT Recall", "Eval Recall", "GT Precision", "FP Ratio in PI", "us-GT Coverage", "FP Removed by NI", "TP Lost by NI", "NI-over-sGT Ratio"}:
                values.append(fmt(value))
            elif field.endswith("duration") or field in {"PI Duration", "FP Duration", "Delta FP Duration vs final_full"}:
                values.append(str(int(num(value))))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_ablation_md(reports: Path, rows: list[dict]) -> None:
    text = "# Low-FP Ablation Summary\n\n" + markdown_table(rows, AB_FIELDS) + "\n"
    (reports / "low_fp_ablation_summary.md").write_text(text, encoding="utf-8")


def write_scan_mds_and_summary(reports: Path) -> dict[str, list[dict]]:
    all_rows = []
    scan_rows = {}
    for name in SCAN_NAMES:
        rows = read_csv(reports / f"scan_{name}.csv")
        scan_rows[name] = rows
        all_rows.extend(rows)
        (reports / f"scan_{name}.md").write_text("# Scan " + name + "\n\n" + markdown_table(rows, SCAN_FIELDS) + "\n", encoding="utf-8")
    write_csv(reports / "low_fp_parameter_scan_summary.csv", all_rows, ["Variant"] + SCAN_FIELDS + ["NI-over-ucGT Ratio", "NI-over-usGT Ratio", "config_json"])
    return scan_rows


def write_near_far(reports: Path, rows: list[dict]) -> None:
    write_csv(reports / "near_far_fp_diagnostics.csv", rows, NEAR_FAR_FIELDS)
    main = next(row for row in rows if int(row["margin"]) == 64)
    lines = [
        "# Near/Far FP Diagnostics",
        "",
        "Near/far FP is computed only on false-positive duration, i.e. `PI intersect nonGT`. `near_GT_FP` is the part of FP lying inside `buffer(GT, margin)`, and `far_GT_FP` is the remaining FP outside that buffer.",
        "",
        "Margin 64 frames is the main diagnostic setting; margins 32 and 128 frames are included as sensitivity checks.",
        "",
        markdown_table(rows, NEAR_FAR_FIELDS),
        "",
        "## Main Reading",
        "",
        f"At margin 64, near-GT FP is {int(num(main['near_GT_FP_duration']))} frames ({fmt(main['near_GT_FP_ratio_in_FP'])} of FP), while far-GT FP is {int(num(main['far_GT_FP_duration']))} frames ({fmt(main['far_GT_FP_ratio_in_FP'])} of FP).",
        "Therefore the current high FP ratio is dominated by far-from-GT false positives rather than only boundary spillover around GT intervals.",
        "",
    ]
    (reports / "near_far_fp_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(reports: Path, ablation_rows: list[dict], scan_rows: dict[str, list[dict]], near_far_rows: list[dict]) -> None:
    final_row = next(row for row in ablation_rows if row["Variant"] == "final_full")
    no_valley = next(row for row in ablation_rows if row["Variant"] == "no_valley_cut")
    no_length = next(row for row in ablation_rows if row["Variant"] == "no_length_penalty")
    no_low_residual = next(row for row in ablation_rows if row["Variant"] == "no_low_residual_penalty")
    main_near = next(row for row in near_far_rows if int(row["margin"]) == 64)
    lines = [
        "# Low-FP Ablation And Scan Report",
        "",
        "## 1. Final Configuration",
        "",
        "The fixed final configuration is `low_fp_with_valley_cut_final`: a Low-FP base configuration plus negative-evidence / valley post-processing. It should be interpreted as a precision-first operating point, not as the recall-optimal method.",
        "",
        "Key parameters: `fusion_threshold=0.38`, `merge_gap_frames=32`, `post_min_duration=32`, `post_min_raw_max=0.60`, `length_penalty_weight=0.22`, `low_residual_penalty_weight=0.15`, `residual_weight=0.25`, `trend_weight=0.20`, `min_normal_duration=96`, `protect_sgt_ucgt=True`.",
        "",
        "## 2. Main Result",
        "",
        f"- s-GT Recall: {fmt(final_row['s-GT Recall'])}.",
        f"- uc-GT Recall: {fmt(final_row['uc-GT Recall'])}.",
        f"- Eval Recall: {fmt(final_row['Eval Recall'])}.",
        f"- GT Precision: {fmt(final_row['GT Precision'])}.",
        f"- FP Duration: {int(num(final_row['FP Duration']))}.",
        f"- FP Ratio in PI: {fmt(final_row['FP Ratio in PI'])}.",
        "",
        "This operating point keeps FP lower than the looser settings, but it gives up recall compared with recall-oriented configurations. That is the intended trade-off.",
        "",
        "## 3. Ablation Study",
        "",
        f"Valley cut is a lightweight refinement: compared with `no_valley_cut`, it removes {int(num(no_valley['FP Duration']) - num(final_row['FP Duration']))} FP frames, changes GT Precision by {fmt(num(final_row['GT Precision']) - num(no_valley['GT Precision']))}, and does not change Eval Recall at the displayed precision.",
        "",
        f"The main FP-control modules are the length penalty and low residual penalty. Removing length penalty raises FP Duration from {int(num(final_row['FP Duration']))} to {int(num(no_length['FP Duration']))}; removing low residual penalty raises it to {int(num(no_low_residual['FP Duration']))}.",
        "",
        "Trend and residual components are recall-bearing evidence. The `no_trend_component` and `no_residual_component` rows have lower FP mostly because they predict far fewer intervals; their Eval Recall collapses, so they should not be interpreted as useful FP improvements.",
        "",
        markdown_table(ablation_rows, AB_FIELDS),
        "",
        "## 4. Parameter Scan",
        "",
        "The local scans vary one parameter around the final setting while keeping the rest fixed. They are explanatory local checks, not a global search.",
        "",
        "The `fusion_threshold` scan shows the clearest recall-precision trade-off: increasing the threshold monotonically lowers FP Duration and raises GT Precision, while Eval Recall drops.",
        "",
        "`valley_low_score_threshold` is insensitive in the tested range under the current constraints: all rows produce the same final PI metrics at the displayed precision.",
        "",
        "`min_normal_duration=48` removes more FP than the final value, but it has a higher NI-over-sGT Ratio. The final `min_normal_duration=96` is therefore the more conservative safety choice.",
        "",
    ]
    for name in SCAN_NAMES:
        lines.extend([f"### {name}", "", markdown_table(scan_rows[name], SCAN_FIELDS), ""])
    lines.extend(
        [
            "## 5. Near/Far FP Diagnostic",
            "",
            "Near/far FP splits `PI intersect nonGT` into FP close to any GT interval versus FP outside a buffered GT neighborhood.",
            "",
            markdown_table(near_far_rows, NEAR_FAR_FIELDS),
            "",
            f"At the main margin of 64 frames, near-GT FP is {int(num(main_near['near_GT_FP_duration']))} frames ({fmt(main_near['near_GT_FP_ratio_in_FP'])} of FP), while far-GT FP is {int(num(main_near['far_GT_FP_duration']))} frames ({fmt(main_near['far_GT_FP_ratio_in_FP'])} of FP). The high FP Ratio is therefore dominated by far-GT FP, not just GT boundary spillover.",
            "",
            "## 6. Limitations",
            "",
            "- FP Ratio remains nontrivial even after valley cut.",
            "- GT/VAD evidence is layered and boundary-uncertain.",
            "- Valley detector can touch s-GT; `protect_sgt_ucgt=True` is required.",
            "- Low-FP sacrifices Eval Recall, so it is suited for low-false-positive reporting rather than maximum recall.",
        ]
    )
    (reports / "low_fp_ablation_and_scan_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--gt_support_csv", type=Path, default=DEFAULT_GT_SUPPORT)
    args = parser.parse_args()

    reports = args.archive / "reports"
    ablation_rows = read_csv(reports / "low_fp_ablation_summary.csv")
    scan_rows = write_scan_mds_and_summary(reports)
    write_ablation_md(reports, ablation_rows)

    pi_rows = read_csv(reports / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv")
    gt_by_video = load_gt(args.gt_support_csv)
    near_far_rows = [compute_near_far(pi_rows, gt_by_video, margin) for margin in [32, 64, 128]]
    write_near_far(reports, near_far_rows)
    write_report(reports, ablation_rows, scan_rows, near_far_rows)
    shutil.copy2(reports / "low_fp_ablation_and_scan_report.md", args.archive / "low-fp-ablation-scan_report.md")

    program = args.archive / "programs" / "scripts" / Path(__file__).name
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program)
    print(f"updated {reports}")


if __name__ == "__main__":
    main()
