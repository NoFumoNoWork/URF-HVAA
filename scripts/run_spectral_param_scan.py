import argparse
import csv
import itertools
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import write_json  # noqa: E402
from scripts.evaluate_interval_methods import (  # noqa: E402
    add_window_methods,
    as_float,
    auto_scan_methods,
    clean_interval,
    evaluate_methods,
    load_gt_rows,
    load_inventory,
    merge_ranges,
    read_csv,
    write_csv,
)
from scripts.run_spectral_score_decomposition import (  # noqa: E402
    airpls,
    compute_sg,
    estimate_stride,
    expand_peak,
    group_positive_runs,
    interval_query,
    load_score_series,
    local_find_peaks,
    mad_threshold,
    rolling_mean_by_frames,
    safe_name,
)


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-18-52-spectral-param-scan/outputs")
DEFAULT_PARAMS = {
    "fusion_threshold": 0.35,
    "trend_window": 100,
    "trend_threshold": 0.60,
    "airpls_lambda": 1000,
    "peak_mad_k": 3.0,
    "sg_window_length": 17,
    "sg_polyorder": 2,
    "length_penalty_weight": 0.15,
    "low_residual_penalty_weight": 0.10,
    "residual_weight": 0.25,
    "trend_weight": 0.15,
    "sg_weight": 0.20,
    "raw_weight": 0.25,
    "peak_count_weight": 0.10,
    "residual_mad_k": 3.0,
    "peak_stop_ratio": 0.25,
    "merge_gap_frames": 48,
}
ONE_FACTOR_SPACE = {
    "fusion_threshold": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "trend_window": [30, 50, 100, 150, 300],
    "trend_threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
    "airpls_lambda": [100, 1000, 10000, 100000],
    "peak_mad_k": [1.5, 2.0, 2.5, 3.0],
    "sg_window_length": [9, 17, 31, 51],
    "sg_polyorder": [2, 3],
    "length_penalty_weight": [0.05, 0.10, 0.15, 0.20, 0.30],
    "low_residual_penalty_weight": [0.05, 0.10, 0.15, 0.20],
    "residual_weight": [0.10, 0.20, 0.25, 0.35, 0.45],
    "trend_weight": [0.10, 0.15, 0.25, 0.35, 0.45],
    "sg_weight": [0.00, 0.10, 0.20, 0.30],
}
COMBO_SPACE = {
    "fusion_threshold": [0.30, 0.35, 0.40, 0.45],
    "trend_window": [50, 100, 150],
    "trend_weight": [0.15, 0.25, 0.35],
    "length_penalty_weight": [0.10, 0.20, 0.30],
}
METRIC_FIELDS = [
    "GT_coverage",
    "predicted_GT_fraction",
    "supportable_gt_coverage",
    "unsupportable_gt_coverage",
    "uncertain_gt_coverage",
    "predicted_duration_ratio",
    "gt_event_hit_ratio",
    "gt_event_missed_ratio",
    "pred_interval_gt_overlap_ratio",
    "mean_predicted_interval_length",
    "median_predicted_interval_length",
    "num_predicted_intervals",
]


def fmt_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def fmt_metric(value) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    except (TypeError, ValueError):
        return "NA"


def strict_score(row: dict) -> float:
    return (
        0.30 * safe_float(row.get("GT_coverage"))
        + 0.25 * safe_float(row.get("predicted_GT_fraction"))
        + 0.25 * safe_float(row.get("supportable_gt_coverage"))
        - 0.10 * safe_float(row.get("predicted_duration_ratio"))
        - 0.10 * safe_float(row.get("unsupportable_gt_coverage"))
    )


def safe_float(value, default: float = 0.0) -> float:
    out = as_float(value, math.nan)
    return default if math.isnan(out) or math.isinf(out) else out


def run_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_")[:90]


def param_summary(config: dict) -> str:
    keys = [
        "fusion_threshold",
        "trend_window",
        "trend_threshold",
        "airpls_lambda",
        "peak_mad_k",
        "sg_window_length",
        "sg_polyorder",
        "length_penalty_weight",
        "low_residual_penalty_weight",
        "residual_weight",
        "trend_weight",
        "sg_weight",
    ]
    return "; ".join(f"{key}={fmt_value(config[key])}" for key in keys if key in config)


def build_runs(scan_mode: str, max_runs: int | None) -> list[dict]:
    runs = []

    def add(mode: str, name: str, config: dict, changed_param: str = "", changed_value="") -> None:
        idx = len(runs) + 1
        run_id = f"run_{idx:04d}_{run_slug(name)}"
        item = {
            "run_id": run_id,
            "display_name": name,
            "scan_mode": mode,
            "changed_param": changed_param,
            "changed_value": changed_value,
            "config": dict(config),
        }
        runs.append(item)

    default = dict(DEFAULT_PARAMS)
    ablation_specs = [
        ("Peak-Aware-Refined", {"direct_method": "Peak-Aware-Refined", "enable_sg": False, "enable_airpls": False, "enable_trend": False}),
        ("Spectral-Fusion without SG", {"enable_sg": False, "enable_airpls": True, "enable_trend": True}),
        ("Spectral-Fusion without airPLS", {"enable_sg": True, "enable_airpls": False, "enable_trend": True}),
        ("Spectral-Fusion without trend", {"enable_sg": True, "enable_airpls": True, "enable_trend": False}),
        ("Spectral-Fusion SG only", {"enable_sg": True, "enable_airpls": False, "enable_trend": False}),
        ("Spectral-Fusion airPLS only", {"enable_sg": False, "enable_airpls": True, "enable_trend": False}),
        ("Spectral-Fusion trend only", {"enable_sg": False, "enable_airpls": False, "enable_trend": True}),
        ("Spectral-Fusion SG + airPLS", {"enable_sg": True, "enable_airpls": True, "enable_trend": False}),
        ("Spectral-Fusion SG + trend", {"enable_sg": True, "enable_airpls": False, "enable_trend": True}),
        ("Spectral-Fusion airPLS + trend", {"enable_sg": False, "enable_airpls": True, "enable_trend": True}),
        ("Full Spectral-Fusion-Refined", {"enable_sg": True, "enable_airpls": True, "enable_trend": True}),
    ]
    if scan_mode in {"ablation", "all"}:
        for name, overrides in ablation_specs:
            config = dict(default)
            config.update(overrides)
            add("ablation", name, config)

    if scan_mode in {"one_factor", "all"}:
        for param, values in ONE_FACTOR_SPACE.items():
            for value in values:
                config = dict(default)
                config.update({"enable_sg": True, "enable_airpls": True, "enable_trend": True})
                config[param] = value
                add("one_factor", f"{param}_{fmt_value(value)}", config, param, value)

    combo_runs = []
    if scan_mode in {"combo", "all"}:
        keys = list(COMBO_SPACE)
        for values in itertools.product(*(COMBO_SPACE[key] for key in keys)):
            config = dict(default)
            config.update({"enable_sg": True, "enable_airpls": True, "enable_trend": True})
            config.update(dict(zip(keys, values)))
            combo_runs.append(("combo_" + "_".join(f"{k}{fmt_value(v)}" for k, v in zip(keys, values)), config))
        if max_runs and max_runs < len(combo_runs):
            random.seed(20260707)
            positions = sorted(random.sample(range(len(combo_runs)), max_runs))
            combo_runs = [combo_runs[pos] for pos in positions]
        for name, config in combo_runs:
            add("combo", name, config)
    return runs


def load_cached_curves(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    if not rows or "frame" not in columns:
        return None
    frames = np.asarray([int(row["frame"]) for row in rows], dtype=int)
    curves = {}
    for col in columns:
        if col == "frame":
            continue
        curves[col] = np.asarray([safe_float(row.get(col), 0.0) for row in rows], dtype=float)
    return frames, curves


def save_curves(path: Path, frames: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["frame"] + sorted(curves)
    rows = []
    for idx, frame in enumerate(frames):
        row = {"frame": int(frame)}
        for key in sorted(curves):
            row[key] = round(float(curves[key][idx]), 6)
        rows.append(row)
    write_json(path, {"columns": columns, "rows": rows})


def precompute_curves(args, inventory: dict, warnings: list[dict]) -> dict:
    sg_windows = sorted(set(ONE_FACTOR_SPACE["sg_window_length"] + [DEFAULT_PARAMS["sg_window_length"]]))
    sg_polys = sorted(set(ONE_FACTOR_SPACE["sg_polyorder"] + [DEFAULT_PARAMS["sg_polyorder"]]))
    air_lambdas = sorted(set(ONE_FACTOR_SPACE["airpls_lambda"] + [DEFAULT_PARAMS["airpls_lambda"]]))
    trend_windows = sorted(set(ONE_FACTOR_SPACE["trend_window"] + COMBO_SPACE["trend_window"] + [DEFAULT_PARAMS["trend_window"]]))
    cache_dir = args.output_dir / "cache" / "decomposition_curves"
    root = Path.cwd()
    decomp = {}
    for key, meta in sorted(inventory.items()):
        dataset, video_id = key
        cache_path = cache_dir / dataset / f"{safe_name(video_id)}_curves.json"
        cached = load_cached_curves(cache_path) if args.reuse_cached_curves else None
        score_path = Path(meta.get("score_json_path", ""))
        if not score_path.is_absolute():
            score_path = root / score_path
        if cached is not None:
            frames, curves = cached
        else:
            if not score_path.exists():
                warnings.append(warn("", "load_score", f"missing score file: {score_path}", key))
                continue
            try:
                frames, values = load_score_series(score_path)
            except Exception as exc:  # noqa: BLE001
                warnings.append(warn("", "load_score", f"failed to read score file: {exc}", key))
                continue
            if len(values) < 3:
                warnings.append(warn("", "load_score", f"too few score points: {len(values)}", key))
                continue
            curves = {"raw_score": values}
            curves.update(compute_sg(values, sg_windows, sg_polys, warnings, f"{dataset}/{video_id}"))
            for win in trend_windows:
                curves[f"rolling_mean_{win}"] = rolling_mean_by_frames(frames, values, win)
            for lam in air_lambdas:
                baseline = airpls(values, float(lam), 2, 20, warnings, f"{dataset}/{video_id}")
                curves[f"airpls_baseline_{int(lam)}"] = baseline
                curves[f"airpls_residual_{int(lam)}"] = values - baseline
            save_curves(cache_path, frames, curves)
        stride = estimate_stride(frames, meta.get("score_stride_est", 16))
        decomp[key] = {"frames": frames, "curves": curves, "stride": stride}
    return decomp


def warn(run_id: str, stage: str, message: str, key: tuple[str, str] | None = None) -> dict:
    dataset, video_id = key if key else ("", "")
    return {"run_id": run_id, "dataset": dataset, "video_id": video_id, "stage": stage, "warning": message}


def add_interval(rows: list[dict], method: str, dataset: str, video_id: str, start: int, end: int, source: str) -> None:
    interval = clean_interval(start, end)
    if interval:
        rows.append({"method": method, "dataset": dataset, "video_id": video_id, "start": interval[0], "end": interval[1], "source_path": source})


def selected_curve(curves: dict[str, np.ndarray], config: dict, group: str) -> np.ndarray | None:
    if group == "sg":
        return curves.get(f"sg_score_{int(config['sg_window_length'])}_{int(config['sg_polyorder'])}")
    if group == "residual":
        return curves.get(f"airpls_residual_{int(config['airpls_lambda'])}")
    if group == "trend":
        return curves.get(f"rolling_mean_{int(config['trend_window'])}")
    return None


def generate_candidates_for_video(run_id: str, dataset: str, video_id: str, data: dict, config: dict, warnings: list[dict]) -> list[dict]:
    frames = data["frames"]
    curves = data["curves"]
    stride = data["stride"]
    raw = curves["raw_score"]
    rows = []

    if config.get("enable_sg", True):
        sg = selected_curve(curves, config, "sg")
        if sg is None:
            warnings.append(warn(run_id, "sg", "selected SG curve missing; raw score fallback used", (dataset, video_id)))
            sg = raw
        threshold = mad_threshold(sg, float(config["peak_mad_k"]))
        spread = np.abs(sg - np.median(sg))
        prom = max(1e-6, mad_threshold(spread, float(config["peak_mad_k"])) - float(np.median(spread)))
        residual = selected_curve(curves, config, "residual")
        positive = np.maximum(residual, 0) if residual is not None else sg
        for peak_idx in local_find_peaks(sg, threshold, prom):
            start, end = expand_peak(frames, positive, peak_idx, stride, float(config["peak_stop_ratio"]))
            add_interval(rows, "SG-Peak", dataset, video_id, start, end, "spectral_param_scan_sg")

    if config.get("enable_airpls", True):
        residual = selected_curve(curves, config, "residual")
        if residual is None:
            warnings.append(warn(run_id, "airpls", "selected airPLS residual missing; skipped residual candidates", (dataset, video_id)))
        else:
            positive = np.maximum(residual, 0)
            threshold = mad_threshold(positive, float(config["residual_mad_k"]))
            for start, end in group_positive_runs(frames, positive > threshold, stride, int(config["merge_gap_frames"])):
                add_interval(rows, "AirPLS-Residual", dataset, video_id, start, end, "spectral_param_scan_airpls")

    if config.get("enable_trend", True):
        trend = selected_curve(curves, config, "trend")
        if trend is None:
            warnings.append(warn(run_id, "trend", "selected trend curve missing; skipped trend candidates", (dataset, video_id)))
        else:
            for start, end in group_positive_runs(frames, trend > float(config["trend_threshold"]), stride, int(config["merge_gap_frames"])):
                raw_vals = interval_query(frames, raw, start, end)
                if len(raw_vals) and float(np.max(raw_vals)) >= float(config["trend_threshold"]):
                    add_interval(rows, "Trend-Guided", dataset, video_id, start, end, "spectral_param_scan_trend")
    return rows


def feature_rows(intervals: list[dict], decomp: dict, config: dict) -> list[dict]:
    rows = []
    for item in intervals:
        key = (item["dataset"], item["video_id"])
        if key not in decomp:
            continue
        data = decomp[key]
        frames = data["frames"]
        curves = data["curves"]
        raw_curve = curves["raw_score"]
        sg_curve = selected_curve(curves, config, "sg")
        residual_curve = selected_curve(curves, config, "residual")
        trend_curve = selected_curve(curves, config, "trend")
        start = int(item["start"])
        end = int(item["end"])
        raw = interval_query(frames, raw_curve, start, end)
        sg = interval_query(frames, sg_curve if sg_curve is not None else raw_curve, start, end)
        residual_full = residual_curve if residual_curve is not None else raw_curve - np.median(raw_curve)
        residual = interval_query(frames, residual_full, start, end)
        positive = np.maximum(residual, 0)
        trend = interval_query(frames, trend_curve if trend_curve is not None else raw_curve, start, end)
        residual_threshold = mad_threshold(np.maximum(residual_full, 0), float(config["residual_mad_k"]))
        rows.append(
            {
                "method": item["method"],
                "dataset": item["dataset"],
                "video_id": item["video_id"],
                "start": start,
                "end": end,
                "source_path": item.get("source_path", ""),
                "raw_max": max_or_zero(raw),
                "sg_max": max_or_zero(sg),
                "airpls_residual_max": max_or_zero(residual),
                "trend_mean": mean_or_zero(trend),
                "residual_peak_count": int(np.sum(positive > residual_threshold)) if len(positive) else 0,
                "interval_length": max(1, end - start),
                "low_residual_ratio": float(np.mean(positive <= 0.05)) if len(positive) else 1.0,
            }
        )
    return rows


def max_or_zero(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0


def mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def normalize(values: list[float]) -> list[float]:
    arr = np.asarray([0.0 if math.isnan(v) or math.isinf(v) else float(v) for v in values], dtype=float)
    if len(arr) == 0:
        return []
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return [0.0 for _ in arr]
    return [float((v - lo) / (hi - lo)) for v in arr]


def build_fusion(run_id: str, existing: list[dict], spectral: list[dict], decomp: dict, config: dict) -> tuple[list[dict], list[dict]]:
    sources = ["Peak-Aware-Refined", "Hierarchical-Merged"]
    candidates = [row for row in existing if row["method"] in sources] + spectral
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
        score = float(config["raw_weight"]) * norms["raw_max"][idx]
        if config.get("enable_sg", True):
            score += float(config["sg_weight"]) * norms["sg_max"][idx]
        if config.get("enable_airpls", True):
            score += float(config["residual_weight"]) * norms["airpls_residual_max"][idx]
            score += float(config["peak_count_weight"]) * norms["residual_peak_count"][idx]
        if config.get("enable_trend", True):
            score += float(config["trend_weight"]) * norms["trend_mean"][idx]
        score -= float(config["length_penalty_weight"]) * norms["interval_length"][idx]
        score -= float(config["low_residual_penalty_weight"]) * safe_float(row["low_residual_ratio"], 1.0)
        out = dict(row)
        out["run_id"] = run_id
        out["fusion_score"] = round(score, 6)
        scored.append(out)
        if score >= float(config["fusion_threshold"]):
            by_video[(row["dataset"], row["video_id"])].append((int(row["start"]), int(row["end"])))
    fused = []
    for (dataset, video_id), intervals in by_video.items():
        for start, end in merge_with_gap(intervals, int(config["merge_gap_frames"])):
            add_interval(fused, run_id, dataset, video_id, start, end, "spectral_param_scan_fusion")
    return fused, scored


def merge_with_gap(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def create_run_outputs(run: dict, intervals: list[dict], scored: list[dict], output_dir: Path) -> None:
    run_dir = output_dir / "runs" / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(run_dir / "spectral_fusion_intervals.csv", intervals, ["method", "dataset", "video_id", "start", "end", "source_path"])
    write_csv(run_dir / "fusion_candidates_scored.csv", scored[:5000])
    write_csv(run_dir / "warnings.csv", [], ["run_id", "dataset", "video_id", "stage", "warning"])


def evaluate_all_runs(all_intervals: list[dict], gt_rows: list[dict], inventory: dict, run_meta: dict) -> list[dict]:
    _, overall, _, _, _ = evaluate_methods(all_intervals, gt_rows, inventory, [0.1, 0.3, 0.5])
    rows = []
    for row in overall:
        if row.get("dataset") != "ALL":
            continue
        meta = run_meta.get(row["method"], {})
        out = {}
        out.update(meta)
        out.update(meta.get("config", {}))
        out.update(row)
        out["run_id"] = row["method"]
        out["display_name"] = meta.get("display_name", row["method"])
        out["current_balanced_score"] = row.get("balanced_score")
        out["stricter_balanced_score"] = strict_score(row)
        out["event_hit_ratio"] = row.get("gt_event_hit_ratio")
        out["event_missed_ratio"] = row.get("gt_event_missed_ratio")
        rows.append(out)
    return rows


def pareto_flags(rows: list[dict]) -> set[str]:
    frontier = set()
    maximize = ["GT_coverage", "predicted_GT_fraction", "supportable_gt_coverage"]
    minimize = ["predicted_duration_ratio", "unsupportable_gt_coverage"]
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            at_least = True
            better = False
            for field in maximize:
                a = safe_float(other.get(field))
                b = safe_float(row.get(field))
                at_least = at_least and a >= b
                better = better or a > b
            for field in minimize:
                a = safe_float(other.get(field), 999.0)
                b = safe_float(row.get(field), 999.0)
                at_least = at_least and a <= b
                better = better or a < b
            if at_least and better:
                dominated = True
                break
        if not dominated:
            frontier.add(row["run_id"])
    return frontier


def write_summary_tables(rows: list[dict], output_dir: Path) -> dict:
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frontier = pareto_flags(rows)
    for row in rows:
        row["is_pareto_frontier"] = row["run_id"] in frontier
        row["config_summary"] = param_summary(row)
    write_csv(summary_dir / "param_scan_all_runs.csv", rows)
    write_csv(summary_dir / "ablation_summary.csv", [r for r in rows if r.get("scan_mode") == "ablation"])
    write_csv(summary_dir / "one_factor_sensitivity_summary.csv", [r for r in rows if r.get("scan_mode") == "one_factor"])
    write_csv(summary_dir / "combo_scan_summary.csv", [r for r in rows if r.get("scan_mode") == "combo"])

    pareto_fields = ["run_id", "display_name", "scan_mode", "config_summary"] + METRIC_FIELDS + ["current_balanced_score", "stricter_balanced_score", "is_pareto_frontier"]
    write_csv(summary_dir / "pareto_frontier_runs.csv", project_fields([r for r in rows if r["is_pareto_frontier"]], pareto_fields), pareto_fields)
    write_csv(summary_dir / "top_runs_by_gt_coverage.csv", project_fields(top(rows, "GT_coverage"), pareto_fields), pareto_fields)
    write_csv(summary_dir / "top_runs_by_purity.csv", project_fields(top(rows, "predicted_GT_fraction"), pareto_fields), pareto_fields)
    write_csv(summary_dir / "top_runs_by_supportable_coverage.csv", project_fields(top(rows, "supportable_gt_coverage"), pareto_fields), pareto_fields)
    write_csv(summary_dir / "top_runs_by_stricter_balanced_score.csv", project_fields(top(rows, "stricter_balanced_score"), pareto_fields), pareto_fields)
    picks = choose_operating_points(rows)
    write_csv(summary_dir / "best_operating_points.csv", project_fields(picks, ["selection", "selection_rule"] + pareto_fields), ["selection", "selection_rule"] + pareto_fields)
    return {"frontier_count": len(frontier), "picks": picks}


def project_fields(rows: list[dict], fields: list[str]) -> list[dict]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def top(rows: list[dict], field: str, n: int = 20) -> list[dict]:
    return sorted(rows, key=lambda r: safe_float(r.get(field), -999), reverse=True)[:n]


def choose_operating_points(rows: list[dict]) -> list[dict]:
    non_peak = [r for r in rows if r.get("display_name") != "Peak-Aware-Refined"]
    candidates = non_peak or rows
    picks = []

    def add(name: str, rule: str, row: dict | None) -> None:
        if not row:
            return
        out = dict(row)
        out["selection"] = name
        out["selection_rule"] = rule
        picks.append(out)

    add("best_recall_oriented", "Maximize GT_coverage excluding the standalone Peak-Aware baseline.", max(candidates, key=lambda r: safe_float(r.get("GT_coverage"), -999), default=None))
    add("best_purity_oriented", "Maximize predicted_GT_fraction excluding the standalone Peak-Aware baseline.", max(candidates, key=lambda r: safe_float(r.get("predicted_GT_fraction"), -999), default=None))
    add("best_supportable_oriented", "Maximize supportable_gt_coverage excluding the standalone Peak-Aware baseline.", max(candidates, key=lambda r: safe_float(r.get("supportable_gt_coverage"), -999), default=None))
    duration_pool = [r for r in candidates if safe_float(r.get("predicted_duration_ratio"), 999) <= 0.55]
    add("best_duration_controlled", "Maximize stricter_balanced_score among runs with predicted_duration_ratio <= 0.55.", max(duration_pool, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999), default=None))
    add("best_balanced", "Maximize stricter_balanced_score excluding the standalone Peak-Aware baseline.", max(candidates, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999), default=None))
    frontier = [r for r in candidates if r.get("is_pareto_frontier")]
    rec_pool = [r for r in frontier if safe_float(r.get("predicted_duration_ratio"), 999) <= 0.65] or frontier or candidates
    add("recommended_operating_point", "Choose the best stricter_balanced_score on the Pareto frontier, preferring predicted_duration_ratio <= 0.65.", max(rec_pool, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999), default=None))
    return picks


def plot_metric_bars(rows: list[dict], path: Path, title: str) -> None:
    metrics = ["GT_coverage", "predicted_GT_fraction", "supportable_gt_coverage", "unsupportable_gt_coverage", "predicted_duration_ratio"]
    labels = [r["display_name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.16
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#7F7F7F"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 2) * width, [safe_float(r.get(metric)) for r in rows], width=width, label=metric, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_sensitivity(rows: list[dict], param: str, path: Path) -> None:
    subset = sorted([r for r in rows if r.get("changed_param") == param], key=lambda r: safe_float(r.get("changed_value"), 0))
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [safe_float(r.get("changed_value")) for r in subset]
    for metric, color in [
        ("GT_coverage", "#4C78A8"),
        ("predicted_GT_fraction", "#59A14F"),
        ("supportable_gt_coverage", "#F28E2B"),
        ("unsupportable_gt_coverage", "#E15759"),
        ("predicted_duration_ratio", "#7F7F7F"),
    ]:
        ax.plot(xs, [safe_float(r.get(metric)) for r in subset], marker="o", label=metric, color=color)
    ax.set_xlabel(param)
    ax.set_ylabel("metric value")
    ax.set_title(f"One-factor sensitivity: {param}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_figures(rows: list[dict], picks: list[dict], output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ablation = [r for r in rows if r.get("scan_mode") == "ablation"]
    if ablation:
        plot_metric_bars(ablation, fig_dir / "fig_ablation_metrics.png", "Ablation metrics")
    name_map = {
        "fusion_threshold": "fig_param_sensitivity_fusion_threshold.png",
        "trend_window": "fig_param_sensitivity_trend_window.png",
        "airpls_lambda": "fig_param_sensitivity_airpls_lambda.png",
        "peak_mad_k": "fig_param_sensitivity_peak_mad_k.png",
        "length_penalty_weight": "fig_param_sensitivity_length_penalty.png",
    }
    for param, name in name_map.items():
        plot_sensitivity(rows, param, fig_dir / name)

    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [safe_float(r.get("predicted_GT_fraction")) for r in rows]
    ys = [safe_float(r.get("GT_coverage")) for r in rows]
    sizes = [35 + 350 * min(1.0, safe_float(r.get("predicted_duration_ratio"))) for r in rows]
    colors = [safe_float(r.get("stricter_balanced_score")) for r in rows]
    sc = ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.58, cmap="viridis")
    ax.set_xlabel("predicted_GT_fraction")
    ax.set_ylabel("GT_coverage")
    ax.set_title("Coverage-purity Pareto field")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="stricter_balanced_score")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_pareto_coverage_purity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter([safe_float(r.get("supportable_gt_coverage")) for r in rows], [safe_float(r.get("unsupportable_gt_coverage")) for r in rows], alpha=0.58, s=80, color="#4C78A8")
    ax.set_xlabel("supportable_gt_coverage")
    ax.set_ylabel("unsupportable_gt_coverage")
    ax.set_title("Supportable vs unsupportable coverage")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_supportable_vs_unsupportable.png", dpi=180)
    plt.close(fig)

    top10 = top(rows, "stricter_balanced_score", 10)
    plot_metric_bars(top10, fig_dir / "fig_top10_configs.png", "Top 10 configurations by stricter balanced score")
    compare_names = {"Peak-Aware-Refined", "Full Spectral-Fusion-Refined"}
    compare = [r for r in rows if r["display_name"] in compare_names]
    seen = {r["run_id"] for r in compare}
    for pick in picks:
        if pick["run_id"] not in seen:
            compare.append(pick)
            seen.add(pick["run_id"])
    plot_metric_bars(compare, fig_dir / "fig_default_vs_best.png", "Default vs best operating points")


def summarize_param_sensitivity(rows: list[dict]) -> list[str]:
    lines = []
    for param in ONE_FACTOR_SPACE:
        subset = [r for r in rows if r.get("changed_param") == param]
        if not subset:
            continue
        gt_vals = [safe_float(r.get("GT_coverage")) for r in subset]
        purity_vals = [safe_float(r.get("predicted_GT_fraction")) for r in subset]
        strict_vals = [safe_float(r.get("stricter_balanced_score")) for r in subset]
        best = max(subset, key=lambda r: safe_float(r.get("stricter_balanced_score"), -999))
        lines.append(
            f"- `{param}`: GT range {max(gt_vals)-min(gt_vals):.3f}, purity range {max(purity_vals)-min(purity_vals):.3f}, stricter-score range {max(strict_vals)-min(strict_vals):.3f}; best value by stricter score = `{best.get('changed_value')}`."
        )
    return lines


def write_report(rows: list[dict], picks: list[dict], summary: dict, output_dir: Path) -> Path:
    report = output_dir / "spectral_param_scan_report.md"
    by_name = {r["display_name"]: r for r in rows}
    default = by_name.get("Full Spectral-Fusion-Refined")
    peak = by_name.get("Peak-Aware-Refined")
    recommended = next((p for p in picks if p["selection"] == "recommended_operating_point"), None)
    best = next((p for p in picks if p["selection"] == "best_balanced"), None)
    lines = [
        "# Spectral Parameter Scan Report",
        "",
        "## Executive summary",
        "",
        f"- Runs evaluated: {len(rows)}; Pareto frontier runs: {summary.get('frontier_count', 0)}.",
        f"- Default Full Spectral-Fusion-Refined: GT_coverage={fmt_metric(default.get('GT_coverage') if default else math.nan)}, predicted_GT_fraction={fmt_metric(default.get('predicted_GT_fraction') if default else math.nan)}, supportable_gt_coverage={fmt_metric(default.get('supportable_gt_coverage') if default else math.nan)}, unsupportable_gt_coverage={fmt_metric(default.get('unsupportable_gt_coverage') if default else math.nan)}, predicted_duration_ratio={fmt_metric(default.get('predicted_duration_ratio') if default else math.nan)}.",
        f"- Best stricter-balanced run: `{best['display_name'] if best else 'NA'}` with stricter_balanced_score={fmt_metric(best.get('stricter_balanced_score') if best else math.nan)}.",
        f"- Recommended operating point: `{recommended['display_name'] if recommended else 'NA'}`. It is chosen from the Pareto frontier when possible and penalizes over-wide / unsupportable coverage.",
        "- The balanced scores are auxiliary ranking tools, not absolute accuracy measures.",
        "",
        "## Scan design",
        "",
        "This stage deliberately uses ablation, one-factor sensitivity, and a small combo grid instead of Bayesian optimization. The goal is to explain which parameters move the coverage-purity trade-off, not to find a black-box optimum before the objective and validation split are settled.",
        "",
        "- Ablation toggles SG, airPLS residual, and trend evidence around the current default.",
        "- One-factor scans change one parameter at a time while holding the default fixed.",
        "- Combo scan covers the main operating controls: fusion threshold, trend window, trend weight, and length penalty.",
        "",
        "## Ablation results",
        "",
    ]
    for row in [r for r in rows if r.get("scan_mode") == "ablation"]:
        lines.append(f"- `{row['display_name']}`: GT={fmt_metric(row.get('GT_coverage'))}, purity={fmt_metric(row.get('predicted_GT_fraction'))}, supportable={fmt_metric(row.get('supportable_gt_coverage'))}, unsupportable={fmt_metric(row.get('unsupportable_gt_coverage'))}, duration={fmt_metric(row.get('predicted_duration_ratio'))}, strict={fmt_metric(row.get('stricter_balanced_score'))}.")
    lines.extend(["", "## One-factor sensitivity", ""])
    lines.extend(summarize_param_sensitivity(rows))
    lines.extend(
        [
            "",
            "## Combo scan and Pareto frontier",
            "",
            f"- Pareto frontier size: {summary.get('frontier_count', 0)}.",
            "- Pareto objectives maximize GT_coverage, predicted_GT_fraction, and supportable_gt_coverage, while minimizing predicted_duration_ratio and unsupportable_gt_coverage.",
            "- See `summaries/pareto_frontier_runs.csv` and the coverage-purity figure for runs that improve one axis without being dominated on the others.",
            "",
            "## Default vs best comparison",
            "",
            "| selection | run | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | stricter_balanced_score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    comparison = []
    if peak:
        comparison.append(dict(peak, selection="Peak-Aware-Refined"))
    if default:
        comparison.append(dict(default, selection="default Spectral-Fusion-Refined"))
    comparison.extend(picks)
    seen = set()
    for row in comparison:
        key = (row.get("selection"), row.get("run_id"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {row.get('selection', '')} | `{row.get('display_name', row.get('run_id'))}` | {fmt_metric(row.get('GT_coverage'))} | {fmt_metric(row.get('predicted_GT_fraction'))} | {fmt_metric(row.get('supportable_gt_coverage'))} | {fmt_metric(row.get('unsupportable_gt_coverage'))} | {fmt_metric(row.get('predicted_duration_ratio'))} | {fmt_metric(row.get('stricter_balanced_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Use `{recommended['display_name'] if recommended else 'NA'}` as the recommended operating point for the current report if the validation protocol accepts parameter selection on this dataset.",
            "- Keep recall-oriented, purity-oriented, and supportable-oriented picks as supplementary operating points rather than replacing the main result silently.",
            "- Do not run Bayesian optimization yet. The objective function is still being interpreted, many controls are discrete module switches, and no train/validation/test split has been enforced.",
            "",
            "## Limitations",
            "",
            "- Without a train/validation/test split, parameter scanning may overfit the current dataset.",
            "- `balanced_score` and `stricter_balanced_score` use human-set weights and are not absolute accuracy.",
            "- Human GT and VDA scores are not absolute truth and can disagree temporally.",
            "- Over-tuning post-processing can hide limitations in VDA scoring.",
            "- Score-unsupported GT cannot be restored reliably by interval post-processing alone.",
            "",
            "## Next steps",
            "",
            "- Split a validation set before any further optimization.",
            "- Consider Bayesian optimization only after the key parameters and objective are fixed.",
            "- Prioritize error taxonomy and case studies for missed score-unsupported GT.",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(report, output_dir.parent / "spectral_param_scan_report.md")
    return report


def archive_outputs(output_dir: Path) -> None:
    root = Path.cwd().resolve()
    archive_root = output_dir.parent.resolve()
    program = archive_root / "programs" / "scripts" / "run_spectral_param_scan.py"
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "scripts" / "run_spectral_param_scan.py", program)
    manifest = [
        "# spectral-param-scan",
        "",
        f"- archive_folder: `{archive_root.relative_to(root).as_posix()}`",
        "- primary_report: `spectral_param_scan_report.md`",
        "",
        "## Contents",
        "",
        "- `programs/`: copied script used for this scan.",
        "- `outputs/cache/`: reusable decomposition curves.",
        "- `outputs/runs/`: per-run config, intervals, warnings, and sampled scored candidates.",
        "- `outputs/summaries/`: all-run, ablation, one-factor, combo, Pareto, top-list, and operating-point CSV files.",
        "- `outputs/figures/`: PNG plots for ablation, sensitivity, Pareto, supportability, top configs, and default-vs-best comparisons.",
    ]
    (archive_root / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    layout = root / "reports" / "ARTIFACT_LAYOUT.md"
    line = f"- `{archive_root.relative_to(root).as_posix()}/`"
    if layout.exists():
        text = layout.read_text(encoding="utf-8")
        if line not in text:
            layout.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_root", type=Path, default=Path("."))
    parser.add_argument("--gt_stats_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv"))
    parser.add_argument("--gt_support_csv", type=Path, default=Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv"))
    parser.add_argument("--video_inventory_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv"))
    parser.add_argument("--existing_interval_root", type=Path, default=Path("outputs"))
    parser.add_argument("--base_spectral_output_dir", type=Path, default=Path("outputs/26-07-07-spectral-score-decomposition/outputs"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scan_mode", choices=["ablation", "one_factor", "combo", "all"], default="all")
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_runs", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reuse_cached_curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse_existing_intervals", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args.scan_mode, args.max_runs)
    if args.dry_run:
        for run in runs:
            print(f"{run['run_id']}: {run['display_name']} | {param_summary(run['config'])}")
        print(json.dumps({"planned_runs": len(runs), "scan_mode": args.scan_mode}, indent=2))
        return

    warnings = []
    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    decomp = precompute_curves(args, inventory, warnings)

    existing = []
    skipped = auto_scan_methods(args.existing_interval_root, existing)
    warnings.extend(warn("", "existing_interval_scan", row["reason"], None) for row in skipped if not row.get("parsed"))
    add_window_methods(existing, inventory, [100, 300], DEFAULT_PARAMS["trend_threshold"], Path.cwd())

    all_intervals = []
    run_meta = {run["run_id"]: run for run in runs}
    failed = 0
    for run in runs:
        run_id = run["run_id"]
        config = run["config"]
        try:
            if config.get("direct_method") == "Peak-Aware-Refined":
                intervals = [dict(row, method=run_id, source_path=row.get("source_path", "")) for row in existing if row["method"] == "Peak-Aware-Refined"]
                scored = []
            else:
                spectral = []
                for (dataset, video_id), data in decomp.items():
                    spectral.extend(generate_candidates_for_video(run_id, dataset, video_id, data, config, warnings))
                intervals, scored = build_fusion(run_id, existing, spectral, decomp, config)
                if not intervals:
                    warnings.append(warn(run_id, "fusion", "run generated no fused intervals", None))
            all_intervals.extend(intervals)
            create_run_outputs(run, intervals, scored, args.output_dir)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            warnings.append(warn(run_id, "run", f"run failed: {exc}", None))
            create_run_outputs(run, [], [], args.output_dir)

    rows = evaluate_all_runs(all_intervals, gt_rows, inventory, run_meta)
    summary = write_summary_tables(rows, args.output_dir)
    picks = summary["picks"]
    if args.make_plots:
        write_figures(rows, picks, args.output_dir)
    write_csv(args.output_dir / "param_scan_warnings.csv", warnings, ["run_id", "dataset", "video_id", "stage", "warning"])
    report = write_report(rows, picks, summary, args.output_dir)
    archive_outputs(args.output_dir)
    result = {
        "scan_mode": args.scan_mode,
        "planned_runs": len(runs),
        "evaluated_runs": len(rows),
        "failed_runs": failed,
        "warning_count": len(warnings),
        "report": str(report),
    }
    write_json(args.output_dir / "spectral_param_scan_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
