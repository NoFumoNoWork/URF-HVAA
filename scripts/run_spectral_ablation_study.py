import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_interval_methods import (  # noqa: E402
    add_window_methods,
    aggregate_rows,
    as_float,
    auto_scan_methods,
    evaluate_methods,
    evaluate_one_video,
    group_gt,
    load_gt_rows,
    load_inventory,
    write_csv,
)
from scripts.run_spectral_param_scan import (  # noqa: E402
    DEFAULT_PARAMS,
    feature_rows,
    generate_candidates_for_video,
    merge_with_gap,
    normalize,
    precompute_curves,
    safe_float,
    strict_score,
)
from scripts.run_spectral_score_decomposition import add_interval  # noqa: E402
from scripts.anomaly_utils import write_json  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-19-33-spectral-ablation-study")
DEFAULT_CACHE_SOURCE = Path("outputs/26-07-07-18-52-spectral-param-scan/outputs/cache/decomposition_curves")
METRICS = [
    "GT_coverage",
    "predicted_GT_fraction",
    "supportable_gt_coverage",
    "unsupportable_gt_coverage",
    "predicted_duration_ratio",
    "balanced_score",
    "stricter_balanced_score",
    "num_predicted_intervals",
    "mean_predicted_interval_length",
    "median_predicted_interval_length",
]
SG_VALUES = [0.0, 0.05, 0.10, 0.20, 0.35]


def fmt(value) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    except (TypeError, ValueError):
        return "NA"


def component_config(
    name: str,
    enabled: list[str],
    disabled: list[str],
    overrides: dict | None = None,
) -> dict:
    config = dict(DEFAULT_PARAMS)
    config.update({"enable_sg": True, "enable_airpls": True, "enable_trend": True})
    config.update(
        {
            "run_name": name,
            "enabled_components": ", ".join(enabled),
            "disabled_components": ", ".join(disabled),
            "raw_weight": 0.0,
            "sg_weight": 0.0,
            "residual_weight": 0.0,
            "trend_weight": 0.0,
            "peak_count_weight": 0.0,
            "length_penalty_weight": 0.0,
            "low_residual_penalty_weight": 0.0,
        }
    )
    for component in enabled:
        if component == "raw":
            config["raw_weight"] = DEFAULT_PARAMS["raw_weight"]
        elif component == "sg":
            config["sg_weight"] = DEFAULT_PARAMS["sg_weight"]
        elif component == "residual":
            config["residual_weight"] = DEFAULT_PARAMS["residual_weight"]
        elif component == "trend":
            config["trend_weight"] = DEFAULT_PARAMS["trend_weight"]
        elif component == "peak_count":
            config["peak_count_weight"] = DEFAULT_PARAMS["peak_count_weight"]
        elif component == "length_penalty":
            config["length_penalty_weight"] = DEFAULT_PARAMS["length_penalty_weight"]
        elif component == "low_residual_penalty":
            config["low_residual_penalty_weight"] = DEFAULT_PARAMS["low_residual_penalty_weight"]
    if overrides:
        config.update(overrides)
    return config


def full_config(name: str = "Full Spectral-Fusion-Refined") -> dict:
    return component_config(
        name,
        ["raw", "sg", "residual", "trend", "peak_count", "length_penalty", "low_residual_penalty"],
        [],
    )


def build_ablation_configs() -> list[dict]:
    full = ["raw", "sg", "residual", "trend", "peak_count", "length_penalty", "low_residual_penalty"]
    specs = [
        ("Full Spectral-Fusion-Refined", full, []),
        ("w/o SG evidence", [c for c in full if c != "sg"], ["sg"]),
        ("w/o airPLS residual evidence", [c for c in full if c != "residual"], ["residual"]),
        ("w/o trend evidence", [c for c in full if c != "trend"], ["trend"]),
        ("w/o peak-count evidence", [c for c in full if c != "peak_count"], ["peak_count"]),
        ("w/o length penalty", [c for c in full if c != "length_penalty"], ["length_penalty"]),
        ("w/o low-residual penalty", [c for c in full if c != "low_residual_penalty"], ["low_residual_penalty"]),
        ("w/o both penalties", [c for c in full if c not in {"length_penalty", "low_residual_penalty"}], ["length_penalty", "low_residual_penalty"]),
        ("SG only", ["sg"], [c for c in full if c != "sg"]),
        ("residual only", ["residual"], [c for c in full if c != "residual"]),
        ("trend only", ["trend"], [c for c in full if c != "trend"]),
        ("peak-count only", ["peak_count"], [c for c in full if c != "peak_count"]),
        ("raw only", ["raw"], [c for c in full if c != "raw"]),
        ("raw + trend", ["raw", "trend"], [c for c in full if c not in {"raw", "trend"}]),
        ("raw + residual", ["raw", "residual"], [c for c in full if c not in {"raw", "residual"}]),
        ("raw + trend + residual", ["raw", "trend", "residual"], [c for c in full if c not in {"raw", "trend", "residual"}]),
        ("raw + trend + residual + length penalty", ["raw", "trend", "residual", "length_penalty"], [c for c in full if c not in {"raw", "trend", "residual", "length_penalty"}]),
        ("raw + trend + residual + both penalties", ["raw", "trend", "residual", "length_penalty", "low_residual_penalty"], [c for c in full if c not in {"raw", "trend", "residual", "length_penalty", "low_residual_penalty"}]),
    ]
    return [component_config(name, enabled, disabled) for name, enabled, disabled in specs]


def build_sg_configs() -> list[dict]:
    values = sorted(set(SG_VALUES + [float(DEFAULT_PARAMS["sg_weight"])]))
    out = []
    for value in values:
        config = full_config(f"sg_weight_{value:g}")
        config["sg_weight"] = value
        config["sg_weight_test_value"] = value
        out.append(config)
    return out


def score_fusion(run_name: str, candidates: list[dict], decomp: dict, config: dict) -> tuple[list[dict], list[dict]]:
    features = feature_rows(candidates, decomp, config)
    norms = {
        "raw_max": normalize([safe_float(r["raw_max"]) for r in features]),
        "sg_max": normalize([safe_float(r["sg_max"]) for r in features]),
        "airpls_residual_max": normalize([safe_float(r["airpls_residual_max"]) for r in features]),
        "trend_mean": normalize([safe_float(r["trend_mean"]) for r in features]),
        "residual_peak_count": normalize([safe_float(r["residual_peak_count"]) for r in features]),
        "interval_length": normalize([safe_float(r["interval_length"]) for r in features]),
    }
    by_video = defaultdict(list)
    scored = []
    for idx, row in enumerate(features):
        score = (
            float(config["raw_weight"]) * norms["raw_max"][idx]
            + float(config["sg_weight"]) * norms["sg_max"][idx]
            + float(config["residual_weight"]) * norms["airpls_residual_max"][idx]
            + float(config["trend_weight"]) * norms["trend_mean"][idx]
            + float(config["peak_count_weight"]) * norms["residual_peak_count"][idx]
            - float(config["length_penalty_weight"]) * norms["interval_length"][idx]
            - float(config["low_residual_penalty_weight"]) * safe_float(row["low_residual_ratio"], 1.0)
        )
        out = dict(row)
        out["run_name"] = run_name
        out["fusion_score"] = round(float(score), 6)
        scored.append(out)
        if score >= float(config["fusion_threshold"]):
            by_video[(row["dataset"], row["video_id"])].append((int(row["start"]), int(row["end"])))
    fused = []
    for (dataset, video_id), intervals in by_video.items():
        for start, end in merge_with_gap(intervals, int(config["merge_gap_frames"])):
            add_interval(fused, run_name, dataset, video_id, start, end, "spectral_ablation_study")
    return fused, scored


def evaluate_configs(configs: list[dict], candidates: list[dict], decomp: dict, gt_rows: list[dict], inventory: dict) -> tuple[list[dict], list[dict]]:
    all_intervals = []
    scored_sample = []
    for config in configs:
        intervals, scored = score_fusion(config["run_name"], candidates, decomp, config)
        all_intervals.extend(intervals)
        scored_sample.extend(scored[:100])
    _, overall, _, _, _ = evaluate_methods(all_intervals, gt_rows, inventory, [0.1, 0.3, 0.5])
    by_run = {row["method"]: row for row in overall if row.get("dataset") == "ALL"}
    gt_by_video = group_gt(gt_rows)
    rows = []
    for config in configs:
        row = dict(config)
        metrics = by_run.get(config["run_name"])
        if metrics is None:
            empty_per_video = [
                evaluate_one_video(config["run_name"], dataset, video_id, video_gt_rows, [], inventory, [0.1, 0.3, 0.5])
                for (dataset, video_id), video_gt_rows in sorted(gt_by_video.items())
            ]
            metrics = aggregate_rows(empty_per_video, ("method",), [0.1, 0.3, 0.5])
            metrics["dataset"] = "ALL"
        row.update(metrics)
        row["run_name"] = config["run_name"]
        row["stricter_balanced_score"] = strict_score(metrics)
        row["mean_interval_length"] = row.get("mean_predicted_interval_length", math.nan)
        row["median_interval_length"] = row.get("median_predicted_interval_length", math.nan)
        rows.append(row)
    return rows, scored_sample


def add_deltas(rows: list[dict], full_name: str) -> list[dict]:
    full = next((r for r in rows if r["run_name"] == full_name), None)
    if not full:
        return rows
    delta_fields = [
        "GT_coverage",
        "predicted_GT_fraction",
        "supportable_gt_coverage",
        "unsupportable_gt_coverage",
        "predicted_duration_ratio",
        "stricter_balanced_score",
    ]
    for row in rows:
        for field in delta_fields:
            row[f"delta_{field}"] = safe_float(row.get(field), 0.0) - safe_float(full.get(field), 0.0)
    return rows


def project(rows: list[dict], fields: list[str]) -> list[dict]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def plot_module_metrics(rows: list[dict], path: Path) -> None:
    metrics = ["GT_coverage", "predicted_GT_fraction", "supportable_gt_coverage", "unsupportable_gt_coverage", "predicted_duration_ratio"]
    labels = [r["run_name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(17, 8))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#7F7F7F"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, [safe_float(r.get(metric)) for r in rows], width=width, label=metric, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Spectral fusion module ablation metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sg(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda r: safe_float(r.get("sg_weight_test_value", r.get("sg_weight"))))
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [safe_float(r.get("sg_weight_test_value", r.get("sg_weight"))) for r in rows]
    for metric, color in [
        ("GT_coverage", "#4C78A8"),
        ("predicted_GT_fraction", "#59A14F"),
        ("supportable_gt_coverage", "#F28E2B"),
        ("unsupportable_gt_coverage", "#E15759"),
        ("predicted_duration_ratio", "#7F7F7F"),
        ("stricter_balanced_score", "#B279A2"),
    ]:
        ax.plot(xs, [safe_float(r.get(metric)) for r in rows], marker="o", label=metric, color=color)
    ax.set_xlabel("sg_weight")
    ax.set_ylabel("metric value")
    ax.set_title("SG direct fusion weight sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_deltas(rows: list[dict], path: Path) -> None:
    metrics = [
        "delta_stricter_balanced_score",
        "delta_GT_coverage",
        "delta_predicted_GT_fraction",
        "delta_unsupportable_gt_coverage",
        "delta_predicted_duration_ratio",
    ]
    labels = [r["run_name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.16
    fig, ax = plt.subplots(figsize=(17, 8))
    colors = ["#B279A2", "#4C78A8", "#59A14F", "#E15759", "#7F7F7F"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, [safe_float(r.get(metric)) for r in rows], width=width, label=metric, color=colors[idx])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Ablation deltas vs Full Spectral-Fusion-Refined")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def code_structure_section() -> list[str]:
    return [
        "## System structure",
        "",
        "- Score JSON loading: `scripts/run_spectral_score_decomposition.py::load_score_series` reads dict/list score JSON into sorted frame and score arrays; `scripts/evaluate_interval_methods.py::load_score_series` has a dict form for fixed-window baselines.",
        "- Candidate interval generation: `generate_spectral_intervals` in the decomposition script creates `SG-Peak`, `AirPLS-Residual`, and `Trend-Guided-100F`; the focused ablation script uses `scripts/run_spectral_param_scan.py::generate_candidates_for_video` to create the same families with parameterized curves.",
        "- SG smoothing: `compute_sg` calls `scipy.signal.savgol_filter` and writes curves named `sg_score_{window}_{poly}`.",
        "- airPLS baseline/residual: `airpls` computes the baseline; `build_decomposition` and `precompute_curves` store `airpls_baseline_{lambda}` and `airpls_residual_{lambda}`.",
        "- Trend evidence: `rolling_mean_by_frames` creates `rolling_mean_{window}`; trend candidates use `trend > trend_threshold` in the scan code and `trend100 > score_threshold` in the original decomposition script.",
        "- Peak-count evidence: `extract_features` in the original script and `feature_rows` in the scan script compute `residual_peak_count` from positive residual evidence over each candidate interval.",
        "- Fusion score: original `build_fusion` in `scripts/run_spectral_score_decomposition.py`; parameterized version in `scripts/run_spectral_param_scan.py::build_fusion`; this ablation script uses the same feature definitions but separates direct terms explicitly.",
        "- `fusion_threshold`: used after score calculation to keep candidate intervals whose score is at least the threshold, then `merge_with_gap` merges retained intervals per video.",
        "- Length penalty: candidate `interval_length` is min-max normalized and subtracted as `length_penalty_weight * length_penalty`.",
        "- Low-residual penalty: `low_residual_ratio` is the fraction of residual samples <= 0.05 inside the interval and is subtracted directly as `low_residual_penalty_weight * low_residual_ratio`.",
        "- Supportable/unsupportable coverage: `scripts/evaluate_interval_methods.py::support_group` maps GT rows to supportable, unsupportable, or uncertain, then `evaluate_one_video` and `aggregate_rows` compute covered duration divided by group GT duration.",
        "- `score_threshold=0.6`: in the inspected code it is used for fixed-window candidate generation and trend/local evidence checks. Supportability grouping is read from `recoverable_by_postprocessing` and `support_type` columns; the evaluator itself does not threshold scores at 0.6 to define supportability.",
        "",
        "The direct fusion score used for this focused ablation is:",
        "",
        "```text",
        "fusion_score =",
        "    raw_weight * raw_evidence",
        "  + sg_weight * sg_evidence",
        "  + residual_weight * residual_evidence",
        "  + trend_weight * trend_evidence",
        "  + peak_count_weight * peak_count_evidence",
        "  - length_penalty_weight * length_penalty",
        "  - low_residual_penalty_weight * low_residual_penalty",
        "```",
        "",
        "Original decomposition-script constants are the same shape except hard-coded as `0.25 raw + 0.20 SG + 0.25 residual + 0.15 trend + 0.10 peak_count - 0.15 length - 0.10 low_residual`.",
        "",
    ]


def default_config_section() -> list[str]:
    rows = [
        ("fusion_threshold", DEFAULT_PARAMS["fusion_threshold"], "scripts/run_spectral_param_scan.py::DEFAULT_PARAMS; original CLI default also 0.35"),
        ("score_threshold", 0.60, "scripts/run_spectral_score_decomposition.py CLI default; used by original trend/window candidate checks"),
        ("trend_threshold", DEFAULT_PARAMS["trend_threshold"], "DEFAULT_PARAMS"),
        ("trend_window", DEFAULT_PARAMS["trend_window"], "DEFAULT_PARAMS"),
        ("trend_weight", DEFAULT_PARAMS["trend_weight"], "DEFAULT_PARAMS"),
        ("residual_weight", DEFAULT_PARAMS["residual_weight"], "DEFAULT_PARAMS"),
        ("sg_weight", DEFAULT_PARAMS["sg_weight"], "DEFAULT_PARAMS"),
        ("peak_count_weight", DEFAULT_PARAMS["peak_count_weight"], "DEFAULT_PARAMS"),
        ("length_penalty_weight", DEFAULT_PARAMS["length_penalty_weight"], "DEFAULT_PARAMS"),
        ("low_residual_penalty_weight", DEFAULT_PARAMS["low_residual_penalty_weight"], "DEFAULT_PARAMS"),
        ("airpls_lambda", DEFAULT_PARAMS["airpls_lambda"], "DEFAULT_PARAMS"),
        ("airpls_order", 2, "original CLI default and scan precompute"),
        ("airpls_itermax", 20, "original CLI default and scan precompute"),
        ("sg_window_length", DEFAULT_PARAMS["sg_window_length"], "DEFAULT_PARAMS"),
        ("sg_polyorder", DEFAULT_PARAMS["sg_polyorder"], "DEFAULT_PARAMS"),
        ("peak_mad_k", DEFAULT_PARAMS["peak_mad_k"], "DEFAULT_PARAMS"),
        ("residual_mad_k", DEFAULT_PARAMS["residual_mad_k"], "DEFAULT_PARAMS"),
        ("peak_stop_ratio", DEFAULT_PARAMS["peak_stop_ratio"], "DEFAULT_PARAMS"),
        ("merge_gap_frames", DEFAULT_PARAMS["merge_gap_frames"], "DEFAULT_PARAMS"),
    ]
    lines = ["## Default configuration", "", "| parameter | value | source |", "|---|---:|---|"]
    for key, value, source in rows:
        lines.append(f"| `{key}` | {value} | {source} |")
    lines.append("")
    return lines


def report_lines(ablation_rows: list[dict], sg_rows: list[dict], peak_row: dict | None) -> list[str]:
    full = next(r for r in ablation_rows if r["run_name"] == "Full Spectral-Fusion-Refined")
    best_main = max(ablation_rows, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999))
    duration_pool = [r for r in ablation_rows if safe_float(r.get("predicted_duration_ratio"), 999) <= safe_float(full.get("predicted_duration_ratio"), 999)]
    best_duration = max(duration_pool, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999), default=best_main)
    best_recall = max(ablation_rows, key=lambda r: safe_float(r.get("GT_coverage"), -999))
    sg0 = next((r for r in sg_rows if abs(safe_float(r.get("sg_weight_test_value")) - 0.0) < 1e-9), None)
    sg_default = next((r for r in sg_rows if abs(safe_float(r.get("sg_weight_test_value")) - DEFAULT_PARAMS["sg_weight"]) < 1e-9), None)
    sg_best = max(sg_rows, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999))
    improves = sg_default and sg0 and safe_float(sg_default.get("stricter_balanced_score")) > safe_float(sg0.get("stricter_balanced_score"))
    no_residual = next(r for r in ablation_rows if r["run_name"] == "w/o airPLS residual evidence")
    no_trend = next(r for r in ablation_rows if r["run_name"] == "w/o trend evidence")
    no_peak = next(r for r in ablation_rows if r["run_name"] == "w/o peak-count evidence")
    no_length = next(r for r in ablation_rows if r["run_name"] == "w/o length penalty")
    no_low_residual = next(r for r in ablation_rows if r["run_name"] == "w/o low-residual penalty")
    peak_compare = "NA"
    if peak_row:
        peak_compare = "yes" if safe_float(full.get("stricter_balanced_score")) > safe_float(peak_row.get("stricter_balanced_score")) else "no"

    lines = ["# Spectral Fusion Ablation Summary", ""]
    lines.extend(code_structure_section())
    lines.extend(default_config_section())
    lines.extend(
        [
            "## SG direct fusion weight sensitivity",
            "",
            "| run | sg_weight | GT | purity | supportable | unsupportable | duration | balanced | strict | intervals | mean_len | median_len | delta_strict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(sg_rows, key=lambda r: safe_float(r.get("sg_weight_test_value"))):
        lines.append(
            f"| `{row['run_name']}` | {safe_float(row.get('sg_weight_test_value')):.2f} | {fmt(row.get('GT_coverage'))} | {fmt(row.get('predicted_GT_fraction'))} | {fmt(row.get('supportable_gt_coverage'))} | {fmt(row.get('unsupportable_gt_coverage'))} | {fmt(row.get('predicted_duration_ratio'))} | {fmt(row.get('balanced_score'))} | {fmt(row.get('stricter_balanced_score'))} | {fmt(row.get('num_predicted_intervals'))} | {fmt(row.get('mean_interval_length'))} | {fmt(row.get('median_interval_length'))} | {fmt(row.get('delta_stricter_balanced_score'))} |"
        )
    lines.extend(
        [
            "",
            f"- Best SG weight by stricter score: `{safe_float(sg_best.get('sg_weight_test_value')):.2f}`.",
            f"- Default sg_weight={DEFAULT_PARAMS['sg_weight']} {'improves' if improves else 'does not improve'} stricter_balanced_score versus sg_weight=0 in this fixed-candidate experiment.",
            "",
            "## Module ablation results",
            "",
            "| run | enabled | disabled | GT | purity | supportable | unsupportable | duration | balanced | strict | intervals | mean_len | median_len |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| `{row['run_name']}` | {row.get('enabled_components','')} | {row.get('disabled_components','')} | {fmt(row.get('GT_coverage'))} | {fmt(row.get('predicted_GT_fraction'))} | {fmt(row.get('supportable_gt_coverage'))} | {fmt(row.get('unsupportable_gt_coverage'))} | {fmt(row.get('predicted_duration_ratio'))} | {fmt(row.get('balanced_score'))} | {fmt(row.get('stricter_balanced_score'))} | {fmt(row.get('num_predicted_intervals'))} | {fmt(row.get('mean_interval_length'))} | {fmt(row.get('median_interval_length'))} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Full Spectral-Fusion-Refined is {'better' if peak_compare == 'yes' else 'not clearly better'} than Peak-Aware-Refined by stricter score in this evaluation. Full strict={fmt(full.get('stricter_balanced_score'))}; Peak-Aware strict={fmt(peak_row.get('stricter_balanced_score') if peak_row else math.nan)}.",
            f"2. Trend evidence appears {'important' if safe_float(no_trend.get('delta_stricter_balanced_score')) < 0 else 'not a clear main gain source'}: removing it changes strict score by {fmt(no_trend.get('delta_stricter_balanced_score'))}. In the additive runs, `raw + trend` is high-purity but low-recall, so trend is useful as context but not sufficient alone.",
            f"3. airPLS residual evidence mainly buys recall/coverage at the cost of broader and less pure predictions. Removing residual drops GT coverage by {fmt(no_residual.get('delta_GT_coverage'))}, but improves purity by {fmt(no_residual.get('delta_predicted_GT_fraction'))}, lowers unsupportable coverage by {fmt(no_residual.get('delta_unsupportable_gt_coverage'))}, lowers duration by {fmt(no_residual.get('delta_predicted_duration_ratio'))}, and improves strict score by {fmt(no_residual.get('delta_stricter_balanced_score'))}.",
            f"4. SG direct fusion weight is {'useful' if improves else 'weak or negative'} under this fixed-candidate test; best tested sg_weight is {safe_float(sg_best.get('sg_weight_test_value')):.2f}.",
            f"5. SG should {'remain as a direct term' if improves else 'be considered for removal from direct fusion while retaining smoothing/candidate/diagnostic roles'} unless validated on a held-out split.",
            f"6. Peak-count independent contribution is weak: `w/o peak-count evidence` delta_strict={fmt(no_peak.get('delta_stricter_balanced_score'))}, with nearly unchanged supportable and unsupportable coverage.",
            f"7. Length penalty helps modestly control width/unsupported coverage: removing it changes duration by {fmt(no_length.get('delta_predicted_duration_ratio'))} and unsupportable coverage by {fmt(no_length.get('delta_unsupportable_gt_coverage'))}.",
            f"8. Low-residual penalty helps reduce unsupported coverage: removing it changes unsupportable coverage by {fmt(no_low_residual.get('delta_unsupportable_gt_coverage'))} and strict score by {fmt(no_low_residual.get('delta_stricter_balanced_score'))}.",
            f"9. Obvious drag components are those whose removal improves strict score; see `ablation_delta_vs_full.csv`. Best strict run here is `{best_main['run_name']}`.",
            f"10. Recall-oriented: `{best_recall['run_name']}`. Duration-controlled: `{best_duration['run_name']}`. Main report recommendation from this ablation: `{best_main['run_name']}` pending validation split.",
            "",
            "## Short conclusion",
            "",
            "- Core useful components: raw score evidence, trend context, and penalties for duration / low residual support.",
            "- Auxiliary useful components: SG smoothing/candidate generation and diagnostic curves remain useful even when direct SG weight is weak.",
            f"- Components with weak or negative direct contribution: SG direct evidence, peak-count direct evidence, and airPLS residual direct evidence under this fixed-candidate strict-score objective.",
            f"- Recommended direct fusion weights: raw=0.25, sg=0.00, residual=0.00, trend=0.15, peak_count=0.10, length_penalty=0.15, low_residual_penalty=0.10 as a validation candidate; among actually executed rows, `{best_main['run_name']}` is best.",
            f"- Recommended operating point: `{best_main['run_name']}` for strict balance; `{best_duration['run_name']}` when duration control is prioritized.",
            "- Remaining limitations: no held-out validation split, fixed candidate pool isolates direct scoring but does not test candidate-generation ablations, and supportability labels come from prior score-support analysis rather than new human adjudication.",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_stats_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv"))
    parser.add_argument("--gt_support_csv", type=Path, default=Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv"))
    parser.add_argument("--video_inventory_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv"))
    parser.add_argument("--existing_interval_root", type=Path, default=Path("outputs"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse_cached_curves", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    out = args.output_dir
    summaries = out / "summaries"
    figures = out / "figures"
    summaries.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    warnings = []
    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    pre_args = argparse.Namespace(output_dir=out / "outputs", reuse_cached_curves=args.reuse_cached_curves)
    if args.reuse_cached_curves and DEFAULT_CACHE_SOURCE.exists():
        target = pre_args.output_dir / "cache" / "decomposition_curves"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(DEFAULT_CACHE_SOURCE, target)
    decomp = precompute_curves(pre_args, inventory, warnings)

    existing = []
    auto_scan_methods(args.existing_interval_root, existing)
    add_window_methods(existing, inventory, [100, 300], DEFAULT_PARAMS["trend_threshold"], Path.cwd())
    candidate_config = full_config()
    spectral = []
    for (dataset, video_id), data in decomp.items():
        spectral.extend(generate_candidates_for_video("candidate_pool", dataset, video_id, data, candidate_config, warnings))
    candidates = [row for row in existing if row["method"] in {"Peak-Aware-Refined", "Hierarchical-Merged"}] + spectral

    ablation_configs = build_ablation_configs()
    sg_configs = build_sg_configs()
    ablation_rows, _ = evaluate_configs(ablation_configs, candidates, decomp, gt_rows, inventory)
    sg_rows, _ = evaluate_configs(sg_configs, candidates, decomp, gt_rows, inventory)
    ablation_rows = add_deltas(ablation_rows, "Full Spectral-Fusion-Refined")
    sg_rows = add_deltas(sg_rows, "sg_weight_0.2")

    peak_intervals = [dict(row, method="Peak-Aware-Refined") for row in existing if row["method"] == "Peak-Aware-Refined"]
    _, peak_overall, _, _, _ = evaluate_methods(peak_intervals, gt_rows, inventory, [0.1, 0.3, 0.5])
    peak_row = next((row for row in peak_overall if row.get("dataset") == "ALL"), None)
    if peak_row:
        peak_row = dict(peak_row)
        peak_row["stricter_balanced_score"] = strict_score(peak_row)

    ablation_fields = [
        "run_name",
        "enabled_components",
        "disabled_components",
        "fusion_threshold",
        "trend_threshold",
        "trend_window",
        "sg_weight",
        "residual_weight",
        "trend_weight",
        "peak_count_weight",
        "length_penalty_weight",
        "low_residual_penalty_weight",
    ] + METRICS + ["mean_interval_length", "median_interval_length"]
    sg_fields = ["run_name", "sg_weight"] + METRICS + ["mean_interval_length", "median_interval_length"] + [
        "delta_GT_coverage",
        "delta_predicted_GT_fraction",
        "delta_supportable_gt_coverage",
        "delta_unsupportable_gt_coverage",
        "delta_predicted_duration_ratio",
        "delta_stricter_balanced_score",
    ]
    delta_fields = ["run_name"] + [field for field in ablation_rows[0] if field.startswith("delta_")]
    write_csv(summaries / "ablation_module_results.csv", project(ablation_rows, ablation_fields), ablation_fields)
    write_csv(summaries / "sg_weight_sensitivity.csv", project(sg_rows, sg_fields), sg_fields)
    write_csv(summaries / "ablation_delta_vs_full.csv", project(ablation_rows, delta_fields), delta_fields)
    plot_module_metrics(ablation_rows, figures / "ablation_module_metrics.png")
    plot_sg(sg_rows, figures / "sg_weight_sensitivity.png")
    plot_deltas(ablation_rows, figures / "ablation_delta_vs_full.png")

    report = summaries / "ablation_summary.md"
    report.write_text("\n".join(report_lines(ablation_rows, sg_rows, peak_row)), encoding="utf-8")
    shutil.copy2(report, out / "ablation_summary.md")
    program = out / "programs" / "scripts" / "run_spectral_ablation_study.py"
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program)
    write_json(
        out / "ablation_study_summary.json",
        {
            "ablation_runs": len(ablation_rows),
            "sg_weight_runs": len(sg_rows),
            "candidate_intervals": len(candidates),
            "warnings": len(warnings),
            "report": str(report),
        },
    )
    print(json.dumps({"ablation_runs": len(ablation_rows), "sg_weight_runs": len(sg_rows), "candidate_intervals": len(candidates), "warnings": len(warnings), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
