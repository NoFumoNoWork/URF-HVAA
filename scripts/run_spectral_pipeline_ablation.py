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

from scripts.anomaly_utils import write_json  # noqa: E402
from scripts.evaluate_interval_methods import (  # noqa: E402
    add_window_methods,
    aggregate_rows,
    auto_scan_methods,
    evaluate_methods,
    evaluate_one_video,
    group_gt,
    load_gt_rows,
    load_inventory,
    write_csv,
)
from scripts.run_spectral_ablation_study import full_config, score_fusion  # noqa: E402
from scripts.run_spectral_param_scan import (  # noqa: E402
    DEFAULT_PARAMS,
    generate_candidates_for_video,
    precompute_curves,
    safe_float,
    strict_score,
)


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-19-51-spectral-pipeline-ablation")
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


def fmt(value) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    except (TypeError, ValueError):
        return "NA"


def config(
    run_name: str,
    configuration_type: str,
    *,
    sg_weight: float | None = None,
    residual_weight: float | None = None,
    trend_weight: float | None = None,
    peak_count_weight: float | None = None,
    fusion_threshold: float | None = None,
    trend_threshold: float | None = None,
    trend_window: int | None = None,
    length_penalty_weight: float | None = None,
    low_residual_penalty_weight: float | None = None,
    use_sg_candidates: bool = True,
    use_residual_candidates: bool = True,
    use_trend_candidates: bool = True,
    direct_raw: bool = True,
    notes: str = "",
) -> dict:
    out = full_config(run_name)
    out["configuration_type"] = configuration_type
    out["sg_smoothing_enabled"] = True
    out["sg_candidate_generation_enabled"] = use_sg_candidates
    out["residual_candidate_generation_enabled"] = use_residual_candidates
    out["trend_candidate_generation_enabled"] = use_trend_candidates
    out["notes"] = notes
    if sg_weight is not None:
        out["sg_weight"] = sg_weight
    if residual_weight is not None:
        out["residual_weight"] = residual_weight
    if trend_weight is not None:
        out["trend_weight"] = trend_weight
    if peak_count_weight is not None:
        out["peak_count_weight"] = peak_count_weight
    if fusion_threshold is not None:
        out["fusion_threshold"] = fusion_threshold
    if trend_threshold is not None:
        out["trend_threshold"] = trend_threshold
    if trend_window is not None:
        out["trend_window"] = trend_window
    if length_penalty_weight is not None:
        out["length_penalty_weight"] = length_penalty_weight
    if low_residual_penalty_weight is not None:
        out["low_residual_penalty_weight"] = low_residual_penalty_weight
    if not direct_raw:
        out["raw_weight"] = 0.0
    return out


def build_sg_ablation_configs() -> list[dict]:
    return [
        config("Full default", "default", sg_weight=0.20, use_sg_candidates=True),
        config("Spectral-Fusion-SG0", "SG0", sg_weight=0.00, use_sg_candidates=True, notes="New direct fusion default candidate: SG smoothing and SG candidates retained; direct SG evidence weight is zero."),
        config("Pipeline w/o SG candidates", "pipeline-sg-ablation", sg_weight=0.20, use_sg_candidates=False),
        config("Pipeline w/o SG candidates + SG0", "pipeline-sg-ablation", sg_weight=0.00, use_sg_candidates=False),
    ]


def build_residual_ablation_configs() -> list[dict]:
    return [
        config("Full default", "default", residual_weight=0.25, use_residual_candidates=True),
        config("w/o residual direct evidence", "pipeline-residual-ablation", residual_weight=0.00, use_residual_candidates=True),
        config("w/o residual candidates", "pipeline-residual-ablation", residual_weight=0.25, use_residual_candidates=False),
        config("w/o residual candidates + w/o residual direct evidence", "pipeline-residual-ablation", residual_weight=0.00, use_residual_candidates=False),
        config("SG0 + w/o residual direct evidence", "pipeline-residual-ablation", sg_weight=0.00, residual_weight=0.00, use_residual_candidates=True),
        config("SG0 + w/o residual candidates + w/o residual direct evidence", "pipeline-residual-ablation", sg_weight=0.00, residual_weight=0.00, use_residual_candidates=False),
    ]


def build_final_configs() -> list[dict]:
    return [
        {"run_name": "Peak-Aware-Refined baseline", "configuration_type": "baseline", "baseline_method": "Peak-Aware-Refined"},
        config("Full Spectral-Fusion-Refined default", "default", sg_weight=0.20, residual_weight=0.25, trend_weight=0.15, length_penalty_weight=0.15, low_residual_penalty_weight=0.10, fusion_threshold=0.35),
        config("Spectral-Fusion-SG0", "SG0", sg_weight=0.00),
        config("Strict SG0 residual0 peak_count_keep", "strict-oriented", sg_weight=0.00, residual_weight=0.00, peak_count_weight=0.10),
        config("Strict SG0 residual0 peak_count0", "strict-oriented", sg_weight=0.00, residual_weight=0.00, peak_count_weight=0.00),
        config("Duration combo SG0 residual0", "duration-controlled", sg_weight=0.00, residual_weight=0.00, fusion_threshold=0.45, trend_window=50, trend_weight=0.35, length_penalty_weight=0.30),
        config("Duration combo SG0 residual0.10", "duration-controlled", sg_weight=0.00, residual_weight=0.10, fusion_threshold=0.45, trend_window=50, trend_weight=0.35, length_penalty_weight=0.30),
        config("Recall trend0.5 SG0 residual0.10", "recall-oriented", sg_weight=0.00, residual_weight=0.10, trend_threshold=0.50),
        config("Recall trend0.5 SG0 residual0.25", "recall-oriented", sg_weight=0.00, residual_weight=0.25, trend_threshold=0.50),
        config("Raw trend residual penalties SG0 residual0.25 peak0", "strict-oriented", sg_weight=0.00, residual_weight=0.25, peak_count_weight=0.00),
        config("Raw trend residual penalties SG0 residual0.10 peak0", "strict-oriented", sg_weight=0.00, residual_weight=0.10, peak_count_weight=0.00),
    ]


def direct_weights(config_row: dict) -> str:
    return (
        f"raw={config_row.get('raw_weight', 0)}, sg={config_row.get('sg_weight', 0)}, "
        f"residual={config_row.get('residual_weight', 0)}, trend={config_row.get('trend_weight', 0)}, "
        f"peak_count={config_row.get('peak_count_weight', 0)}, length_penalty={config_row.get('length_penalty_weight', 0)}, "
        f"low_residual_penalty={config_row.get('low_residual_penalty_weight', 0)}"
    )


def source_names(config_row: dict, enabled: bool = True) -> str:
    base = {
        "Peak-Aware-Refined": True,
        "Hierarchical-Merged": True,
        "SG-Peak": bool(config_row.get("sg_candidate_generation_enabled", True)),
        "AirPLS-Residual": bool(config_row.get("residual_candidate_generation_enabled", True)),
        "Trend-Guided": bool(config_row.get("trend_candidate_generation_enabled", True)),
    }
    return ", ".join(name for name, is_enabled in base.items() if is_enabled == enabled)


def filter_candidates(config_row: dict, base_candidates: list[dict]) -> list[dict]:
    rows = []
    for row in base_candidates:
        method = row.get("method", "")
        if method == "SG-Peak" and not config_row.get("sg_candidate_generation_enabled", True):
            continue
        if method == "AirPLS-Residual" and not config_row.get("residual_candidate_generation_enabled", True):
            continue
        if method.startswith("Trend-Guided") and not config_row.get("trend_candidate_generation_enabled", True):
            continue
        rows.append(row)
    return rows


def empty_metrics(run_name: str, gt_rows: list[dict], inventory: dict) -> dict:
    gt_by_video = group_gt(gt_rows)
    per_video = [
        evaluate_one_video(run_name, dataset, video_id, video_gt_rows, [], inventory, [0.1, 0.3, 0.5])
        for (dataset, video_id), video_gt_rows in sorted(gt_by_video.items())
    ]
    metrics = aggregate_rows(per_video, ("method",), [0.1, 0.3, 0.5])
    metrics["dataset"] = "ALL"
    return metrics


def make_candidates_for_config(cfg: dict, existing_rows: list[dict], decomp: dict, warnings: list[dict]) -> list[dict]:
    spectral = []
    for (dataset, video_id), data in decomp.items():
        spectral.extend(generate_candidates_for_video(cfg["run_name"], dataset, video_id, data, cfg, warnings))
    candidates = [row for row in existing_rows if row["method"] in {"Peak-Aware-Refined", "Hierarchical-Merged"}] + spectral
    return filter_candidates(cfg, candidates)


def evaluate_config_set(configs: list[dict], existing_rows: list[dict], decomp: dict, gt_rows: list[dict], inventory: dict, warnings: list[dict]) -> list[dict]:
    all_intervals = []
    pending = []
    rows_by_name = {}
    for cfg in configs:
        run_name = cfg["run_name"]
        if cfg.get("baseline_method") == "Peak-Aware-Refined":
            intervals = [dict(row, method=run_name) for row in existing_rows if row["method"] == "Peak-Aware-Refined"]
        else:
            candidates = make_candidates_for_config(cfg, existing_rows, decomp, warnings)
            intervals, _ = score_fusion(run_name, candidates, decomp, cfg)
        if intervals:
            all_intervals.extend(intervals)
            pending.append(cfg)
        else:
            rows_by_name[run_name] = empty_metrics(run_name, gt_rows, inventory)
    if all_intervals:
        _, overall, _, _, _ = evaluate_methods(all_intervals, gt_rows, inventory, [0.1, 0.3, 0.5])
        rows_by_name.update({row["method"]: row for row in overall if row.get("dataset") == "ALL"})
    out = []
    for cfg in configs:
        row = dict(cfg)
        metrics = rows_by_name.get(cfg["run_name"], {})
        row.update(metrics)
        row["run_name"] = cfg["run_name"]
        row["stricter_balanced_score"] = strict_score(metrics)
        row["mean_interval_length"] = row.get("mean_predicted_interval_length", math.nan)
        row["median_interval_length"] = row.get("median_predicted_interval_length", math.nan)
        row["enabled_candidate_sources"] = "Peak-Aware-Refined only" if cfg.get("baseline_method") else source_names(cfg, True)
        row["disabled_candidate_sources"] = "" if cfg.get("baseline_method") else source_names(cfg, False)
        row["direct_weights"] = "NA" if cfg.get("baseline_method") else direct_weights(cfg)
        out.append(row)
    return out


def add_deltas(rows: list[dict], full_name: str = "Full default") -> list[dict]:
    full = next((row for row in rows if row["run_name"] == full_name), rows[0])
    for row in rows:
        for field in [
            "GT_coverage",
            "predicted_GT_fraction",
            "supportable_gt_coverage",
            "unsupportable_gt_coverage",
            "predicted_duration_ratio",
            "stricter_balanced_score",
        ]:
            row[f"delta_{field}"] = safe_float(row.get(field), 0.0) - safe_float(full.get(field), 0.0)
    return rows


def project(rows: list[dict], fields: list[str]) -> list[dict]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def plot_rows(rows: list[dict], path: Path, title: str) -> None:
    metrics = ["GT_coverage", "predicted_GT_fraction", "supportable_gt_coverage", "unsupportable_gt_coverage", "predicted_duration_ratio"]
    labels = [row["run_name"] for row in rows]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.2), 7))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#7F7F7F"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, [safe_float(row.get(metric)) for row in rows], width=width, label=metric, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [safe_float(row.get("stricter_balanced_score")) for row in rows]
    ys = [safe_float(row.get("GT_coverage")) for row in rows]
    sizes = [50 + 500 * safe_float(row.get("predicted_duration_ratio")) for row in rows]
    colors = [safe_float(row.get("unsupportable_gt_coverage")) for row in rows]
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="magma_r", alpha=0.7)
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(row["run_name"], (x, y), fontsize=7)
    ax.set_xlabel("stricter_balanced_score")
    ax.set_ylabel("GT_coverage")
    ax.set_title("Recall vs strict-score trade-off")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="unsupportable_gt_coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_recommendation(path: Path, sg_rows: list[dict], residual_rows: list[dict], final_rows: list[dict]) -> None:
    peak = next(row for row in final_rows if row["configuration_type"] == "baseline")
    full = next(row for row in final_rows if row["run_name"] == "Full Spectral-Fusion-Refined default")
    sg0 = next(row for row in final_rows if row["run_name"] == "Spectral-Fusion-SG0")
    recall_candidates = [row for row in final_rows if row["configuration_type"] == "recall-oriented"]
    strict_candidates = [row for row in final_rows if row["configuration_type"] in {"strict-oriented", "duration-controlled", "SG0"}]
    recall = max(recall_candidates, key=lambda r: (safe_float(r.get("GT_coverage")), safe_float(r.get("supportable_gt_coverage"))))
    strict = max(strict_candidates, key=lambda r: safe_float(r.get("stricter_balanced_score")))
    viable_duration_pool = [row for row in strict_candidates if safe_float(row.get("GT_coverage")) >= 0.50]
    duration = min(viable_duration_pool or strict_candidates, key=lambda r: (safe_float(r.get("predicted_duration_ratio"), 999), -safe_float(r.get("stricter_balanced_score"))))
    no_sg_candidates = next(row for row in sg_rows if row["run_name"] == "Pipeline w/o SG candidates + SG0")
    no_residual_direct = next(row for row in residual_rows if row["run_name"] == "w/o residual direct evidence")
    no_residual_candidates = next(row for row in residual_rows if row["run_name"] == "w/o residual candidates")

    def metric_row(row: dict) -> str:
        return (
            f"GT={fmt(row.get('GT_coverage'))}, purity={fmt(row.get('predicted_GT_fraction'))}, "
            f"supportable={fmt(row.get('supportable_gt_coverage'))}, unsupported={fmt(row.get('unsupportable_gt_coverage'))}, "
            f"duration={fmt(row.get('predicted_duration_ratio'))}, strict={fmt(row.get('stricter_balanced_score'))}"
        )

    lines = [
        "# Final Spectral Fusion Configuration Recommendation",
        "",
        "## New Direct Fusion Candidate: Spectral-Fusion-SG0",
        "",
        "- Changed parameter: `sg_weight` from 0.20 to 0.00.",
        "- SG smoothing still runs; cached `sg_score_{window}_{poly}` curves remain available.",
        "- SG candidate intervals still generate by default through `SG-Peak` unless explicitly disabled in pipeline ablations.",
        "- `sg_evidence` can still be computed as a feature, but it contributes zero direct positive score in `Spectral-Fusion-SG0`.",
        "",
        "```text",
        "fusion_score =",
        "    raw_weight * raw_evidence",
        "  + residual_weight * residual_evidence",
        "  + trend_weight * trend_evidence",
        "  + peak_count_weight * peak_count_evidence",
        "  - length_penalty_weight * length_penalty",
        "  - low_residual_penalty_weight * low_residual_penalty",
        "```",
        "",
        "## Pipeline SG Candidate Ablation",
        "",
        f"- Full default: {metric_row(next(row for row in sg_rows if row['run_name'] == 'Full default'))}.",
        f"- Spectral-Fusion-SG0: {metric_row(next(row for row in sg_rows if row['run_name'] == 'Spectral-Fusion-SG0'))}.",
        f"- Removing SG candidates with SG0: {metric_row(no_sg_candidates)}.",
        "- Interpretation: `sg_weight=0` tests direct evidence; removing SG candidates tests candidate-generation contribution. Compare the CSV deltas to decide whether SG is useful as source generation even when it is not useful as direct score.",
        "",
        "## Pipeline Residual Candidate Ablation",
        "",
        f"- w/o residual direct evidence: {metric_row(no_residual_direct)}.",
        f"- w/o residual candidates: {metric_row(no_residual_candidates)}.",
        "- Interpretation: residual direct evidence trades coverage against duration/unsupported coverage. With the old SG-positive score, setting residual direct evidence to zero improves strict score by shrinking intervals. Under the SG0 candidate family, however, completely zeroing residual direct evidence can collapse coverage, so residual direct weight should be tuned rather than blindly removed.",
        "",
        "## Final Candidate Comparison",
        "",
        "| run | type | GT | purity | supportable | unsupported | duration | strict |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| `{row['run_name']}` | {row['configuration_type']} | {fmt(row.get('GT_coverage'))} | {fmt(row.get('predicted_GT_fraction'))} | {fmt(row.get('supportable_gt_coverage'))} | {fmt(row.get('unsupportable_gt_coverage'))} | {fmt(row.get('predicted_duration_ratio'))} | {fmt(row.get('stricter_balanced_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Recall-Oriented Configuration",
            "",
            f"- Configuration name: `{recall['run_name']}`.",
            f"- Candidate sources: {recall['enabled_candidate_sources']}.",
            f"- Direct weights: {recall['direct_weights']}.",
            f"- Key thresholds: fusion_threshold={recall.get('fusion_threshold')}, trend_threshold={recall.get('trend_threshold')}, trend_window={recall.get('trend_window')}.",
            f"- Metrics: {metric_row(recall)}.",
            f"- Relative to Full default: GT {fmt(safe_float(recall.get('GT_coverage')) - safe_float(full.get('GT_coverage')))}, supportable {fmt(safe_float(recall.get('supportable_gt_coverage')) - safe_float(full.get('supportable_gt_coverage')))}, purity {fmt(safe_float(recall.get('predicted_GT_fraction')) - safe_float(full.get('predicted_GT_fraction')))}, unsupported {fmt(safe_float(recall.get('unsupportable_gt_coverage')) - safe_float(full.get('unsupportable_gt_coverage')))}, duration {fmt(safe_float(recall.get('predicted_duration_ratio')) - safe_float(full.get('predicted_duration_ratio')))}.",
            f"- Relative to Peak-Aware: GT {fmt(safe_float(recall.get('GT_coverage')) - safe_float(peak.get('GT_coverage')))}, supportable {fmt(safe_float(recall.get('supportable_gt_coverage')) - safe_float(peak.get('supportable_gt_coverage')))}, purity {fmt(safe_float(recall.get('predicted_GT_fraction')) - safe_float(peak.get('predicted_GT_fraction')))}, unsupported {fmt(safe_float(recall.get('unsupportable_gt_coverage')) - safe_float(peak.get('unsupportable_gt_coverage')))}, duration {fmt(safe_float(recall.get('predicted_duration_ratio')) - safe_float(peak.get('predicted_duration_ratio')))}.",
            "- Why recall-oriented: it prioritizes GT/supportable coverage and accepts some loss in purity or duration control.",
            "",
            "## Strict-Oriented / Duration-Controlled Configuration",
            "",
            f"- Configuration name: `{strict['run_name']}`.",
            f"- Candidate sources: {strict['enabled_candidate_sources']}.",
            f"- Direct weights: {strict['direct_weights']}.",
            f"- Key thresholds: fusion_threshold={strict.get('fusion_threshold')}, trend_threshold={strict.get('trend_threshold')}, trend_window={strict.get('trend_window')}.",
            f"- Metrics: {metric_row(strict)}.",
            f"- Viable duration-controlled alternative: `{duration['run_name']}` with {metric_row(duration)}.",
            f"- Relative to Full default: GT {fmt(safe_float(strict.get('GT_coverage')) - safe_float(full.get('GT_coverage')))}, supportable {fmt(safe_float(strict.get('supportable_gt_coverage')) - safe_float(full.get('supportable_gt_coverage')))}, purity {fmt(safe_float(strict.get('predicted_GT_fraction')) - safe_float(full.get('predicted_GT_fraction')))}, unsupported {fmt(safe_float(strict.get('unsupportable_gt_coverage')) - safe_float(full.get('unsupportable_gt_coverage')))}, duration {fmt(safe_float(strict.get('predicted_duration_ratio')) - safe_float(full.get('predicted_duration_ratio')))}.",
            f"- Relative to Peak-Aware: GT {fmt(safe_float(strict.get('GT_coverage')) - safe_float(peak.get('GT_coverage')))}, supportable {fmt(safe_float(strict.get('supportable_gt_coverage')) - safe_float(peak.get('supportable_gt_coverage')))}, purity {fmt(safe_float(strict.get('predicted_GT_fraction')) - safe_float(peak.get('predicted_GT_fraction')))}, unsupported {fmt(safe_float(strict.get('unsupportable_gt_coverage')) - safe_float(peak.get('unsupportable_gt_coverage')))}, duration {fmt(safe_float(strict.get('predicted_duration_ratio')) - safe_float(peak.get('predicted_duration_ratio')))}.",
            "- Why strict-oriented: it prefers higher purity, lower unsupported coverage, and lower duration even if GT coverage drops.",
            "",
            "## Final Judgments",
            "",
            "1. `sg_weight` should be set to 0 for the next default direct-fusion candidate.",
            "2. SG smoothing should be retained.",
            "3. SG candidate generation should be retained if removing it lowers GT/supportable coverage without enough strict-score gain; use `pipeline_sg_candidate_ablation.csv` for the exact trade-off.",
            "4. Residual direct fusion evidence should not be removed globally. For the SG0 strict-balanced candidate, a positive residual direct weight preserves coverage; for duration-controlled variants, lower residual weight such as 0.10 can be tested.",
            "5. Residual candidate generation should be retained. Removing residual candidates slightly lowers coverage and did not produce a better strict operating point in the final comparison.",
            "6. Trend evidence is a stabilizing context term; it is more important in strict/duration-controlled configurations where residual is downweighted.",
            "7. Penalties should be retained because they control duration and unsupported coverage.",
            "8. Peak-count direct evidence is weak but cheap; keep it only if validation confirms no harm, otherwise set it to 0 in strict mode.",
            "9. Two operating points are recommended: one strict/main-report configuration and one supplementary recall-oriented configuration.",
            f"10. Main report candidate: `{strict['run_name']}` pending validation. Supplementary recall-oriented result: `{recall['run_name']}`.",
            "",
            "## Limitations",
            "",
            "- These are offline post-processing ablations on the same data, not held-out validation results.",
            "- Candidate-generation ablations still reuse existing Peak-Aware and Hierarchical candidates.",
            "- Supportable/unsupportable labels come from prior score-support classification.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    base_candidate_config = full_config("candidate_pool")
    base_candidates = make_candidates_for_config(base_candidate_config, existing, decomp, warnings)

    sg_rows = add_deltas(evaluate_config_set(build_sg_ablation_configs(), existing, decomp, gt_rows, inventory, warnings), "Full default")
    residual_rows = add_deltas(evaluate_config_set(build_residual_ablation_configs(), existing, decomp, gt_rows, inventory, warnings), "Full default")
    final_rows = evaluate_config_set(build_final_configs(), existing, decomp, gt_rows, inventory, warnings)

    sg_fields = [
        "run_name",
        "sg_weight",
        "sg_smoothing_enabled",
        "sg_candidate_generation_enabled",
    ] + METRICS + [
        "mean_interval_length",
        "median_interval_length",
        "delta_GT_coverage",
        "delta_predicted_GT_fraction",
        "delta_supportable_gt_coverage",
        "delta_unsupportable_gt_coverage",
        "delta_predicted_duration_ratio",
        "delta_stricter_balanced_score",
    ]
    residual_fields = [
        "run_name",
        "sg_weight",
        "residual_weight",
        "residual_candidate_generation_enabled",
    ] + METRICS + ["mean_interval_length", "median_interval_length"]
    final_fields = [
        "run_name",
        "configuration_type",
        "enabled_candidate_sources",
        "disabled_candidate_sources",
        "direct_weights",
        "fusion_threshold",
        "trend_threshold",
        "trend_window",
    ] + METRICS + ["mean_interval_length", "median_interval_length"]
    write_csv(summaries / "pipeline_sg_candidate_ablation.csv", project(sg_rows, sg_fields), sg_fields)
    write_csv(summaries / "pipeline_residual_candidate_ablation.csv", project(residual_rows, residual_fields), residual_fields)
    write_csv(summaries / "final_candidate_config_comparison.csv", project(final_rows, final_fields), final_fields)
    write_recommendation(summaries / "final_config_recommendation.md", sg_rows, residual_rows, final_rows)
    shutil.copy2(summaries / "final_config_recommendation.md", out / "final_config_recommendation.md")

    plot_rows(sg_rows, figures / "fig_pipeline_sg_candidate_ablation.png", "Pipeline SG candidate ablation")
    plot_rows(residual_rows, figures / "fig_pipeline_residual_candidate_ablation.png", "Pipeline residual candidate ablation")
    plot_rows(final_rows, figures / "fig_final_candidate_config_comparison.png", "Final candidate configuration comparison")
    plot_tradeoff(final_rows, figures / "fig_recall_vs_strict_tradeoff.png")

    program = out / "programs" / "scripts" / "run_spectral_pipeline_ablation.py"
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program)
    summary = {
        "sg_ablation_runs": len(sg_rows),
        "residual_ablation_runs": len(residual_rows),
        "final_candidate_runs": len(final_rows),
        "candidate_intervals": len(base_candidates),
        "warnings": len(warnings),
        "report": str(summaries / "final_config_recommendation.md"),
    }
    write_json(out / "pipeline_ablation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
