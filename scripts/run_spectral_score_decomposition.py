import argparse
import csv
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
from scripts.anomaly_utils import write_json  # noqa: E402


try:
    from scipy.signal import find_peaks, savgol_filter

    SCIPY_SIGNAL_AVAILABLE = True
except Exception:  # noqa: BLE001
    find_peaks = None
    savgol_filter = None
    SCIPY_SIGNAL_AVAILABLE = False

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    SCIPY_SPARSE_AVAILABLE = True
except Exception:  # noqa: BLE001
    sparse = None
    spsolve = None
    SCIPY_SPARSE_AVAILABLE = False


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-spectral-score-decomposition/outputs")
EVAL_OLD_METHODS = {
    "Hierarchical-Merged",
    "Peak-Aware-Refined",
    "Window-100F",
    "Window-300F",
    "TopK-10",
    "Random-Same-Length",
}


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_score_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = read_json(path)
    frames = []
    scores = []
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    for key, value in items:
        try:
            frame = int(key)
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(score) or math.isinf(score):
            continue
        frames.append(frame)
        scores.append(score)
    order = np.argsort(frames)
    return np.asarray(frames, dtype=int)[order], np.asarray(scores, dtype=float)[order]


def estimate_stride(frames: np.ndarray, fallback: int = 16) -> int:
    if len(frames) < 2:
        return fallback
    diffs = np.diff(frames)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[int(np.argmax(counts))])


def rolling_mean_by_frames(frames: np.ndarray, values: np.ndarray, window_frames: int) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=float)
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    out = np.zeros_like(values, dtype=float)
    half = window_frames / 2.0
    for idx, frame in enumerate(frames):
        lo = int(np.searchsorted(frames, math.ceil(frame - half), side="left"))
        hi = int(np.searchsorted(frames, math.floor(frame + half), side="right"))
        if hi <= lo:
            out[idx] = values[idx]
        else:
            out[idx] = (prefix[hi] - prefix[lo]) / (hi - lo)
    return out


def mad_threshold(values: np.ndarray, k: float = 3.0) -> float:
    if len(values) == 0:
        return math.inf
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad == 0:
        mad = float(np.std(values))
    return med + k * (mad if mad > 0 else 1e-6)


def compute_sg(values: np.ndarray, windows: list[int], polys: list[int], warnings: list[dict], video_key: str) -> dict[str, np.ndarray]:
    curves = {}
    if not SCIPY_SIGNAL_AVAILABLE:
        warnings.append({"video_key": video_key, "stage": "sg", "warning": "scipy.signal unavailable; skipped Savitzky-Golay curves"})
        return curves
    for win in windows:
        local_win = win if win % 2 == 1 else win + 1
        if local_win >= len(values):
            local_win = len(values) - 1 if len(values) % 2 == 0 else len(values)
        if local_win < 3 or local_win % 2 == 0:
            warnings.append({"video_key": video_key, "stage": "sg", "warning": f"illegal window length {win} for score length {len(values)}"})
            continue
        for poly in polys:
            if poly >= local_win:
                warnings.append({"video_key": video_key, "stage": "sg", "warning": f"polyorder {poly} >= window {local_win}"})
                continue
            try:
                curves[f"sg_score_{local_win}_{poly}"] = savgol_filter(values, local_win, poly, mode="interp")
            except Exception as exc:  # noqa: BLE001
                warnings.append({"video_key": video_key, "stage": "sg", "warning": f"savgol failed window={local_win} poly={poly}: {exc}"})
    return curves


def difference_penalty(n: int, order: int):
    if order <= 1:
        return sparse.diags([-1, 1], [0, 1], shape=(n - 1, n), format="csc")
    return sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc")


def airpls(values: np.ndarray, lam: float, order: int, itermax: int, warnings: list[dict], video_key: str) -> np.ndarray:
    n = len(values)
    if n < max(4, order + 2):
        warnings.append({"video_key": video_key, "stage": "airpls", "warning": f"too few points for airPLS lambda={lam}"})
        return values.copy()
    if np.allclose(values, values[0]):
        return np.full_like(values, float(values[0]), dtype=float)

    if SCIPY_SPARSE_AVAILABLE:
        try:
            dmat = difference_penalty(n, order)
            penalty = lam * (dmat.T @ dmat)
            weights = np.ones(n, dtype=float)
            baseline = values.copy()
            for iteration in range(1, itermax + 1):
                wmat = sparse.diags(weights, 0, shape=(n, n), format="csc")
                baseline = np.asarray(spsolve(wmat + penalty, weights * values), dtype=float)
                residual = values - baseline
                negative = residual[residual < 0]
                denom = float(np.sum(np.abs(negative)))
                if denom < 1e-8:
                    break
                weights[residual >= 0] = 0.01
                weights[residual < 0] = np.exp(np.minimum(50.0, iteration * np.abs(residual[residual < 0]) / denom))
                weights[0] = weights[-1] = max(weights[0], weights[-1], 1.0)
            return np.nan_to_num(baseline, nan=float(np.median(values)))
        except Exception as exc:  # noqa: BLE001
            warnings.append({"video_key": video_key, "stage": "airpls", "warning": f"sparse solve failed lambda={lam}: {exc}"})

    if n > 1500:
        warnings.append({"video_key": video_key, "stage": "airpls", "warning": f"dense fallback avoided for n={n}; used rolling-min baseline lambda={lam}"})
        width = max(5, min(n // 5, int(math.sqrt(lam))))
        return rolling_minimum(values, width)

    try:
        dmat = np.diff(np.eye(n), n=order, axis=0)
        penalty = lam * (dmat.T @ dmat)
        weights = np.ones(n, dtype=float)
        baseline = values.copy()
        for iteration in range(1, itermax + 1):
            amat = np.diag(weights) + penalty
            baseline = np.linalg.solve(amat, weights * values)
            residual = values - baseline
            negative = residual[residual < 0]
            denom = float(np.sum(np.abs(negative)))
            if denom < 1e-8:
                break
            weights[residual >= 0] = 0.01
            weights[residual < 0] = np.exp(np.minimum(50.0, iteration * np.abs(residual[residual < 0]) / denom))
        return np.nan_to_num(baseline, nan=float(np.median(values)))
    except Exception as exc:  # noqa: BLE001
        warnings.append({"video_key": video_key, "stage": "airpls", "warning": f"dense solve failed lambda={lam}: {exc}"})
        return rolling_minimum(values, max(5, min(n // 5, int(math.sqrt(lam)))))


def rolling_minimum(values: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    half = max(1, width // 2)
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out[idx] = float(np.min(values[lo:hi]))
    return rolling_mean_by_index(out, max(3, half))


def rolling_mean_by_index(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) == 0:
        return values
    width = max(1, int(width))
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def build_decomposition(frames: np.ndarray, values: np.ndarray, args, warnings: list[dict], video_key: str) -> dict[str, np.ndarray]:
    curves = {"raw_score": values}
    if args.enable_sg:
        curves.update(compute_sg(values, args.sg_window_lengths, args.sg_polyorders, warnings, video_key))
    for win in args.trend_windows:
        curves[f"rolling_mean_{win}"] = rolling_mean_by_frames(frames, values, win)
    if args.enable_airpls:
        for lam in args.airpls_lambdas:
            baseline = airpls(values, lam, args.airpls_order, args.airpls_itermax, warnings, video_key)
            curves[f"airpls_baseline_{int(lam)}"] = baseline
            curves[f"airpls_residual_{int(lam)}"] = values - baseline
    return curves


def save_curves(path: Path, frames: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    keys = list(curves)
    for idx, frame in enumerate(frames):
        row = {"frame": int(frame)}
        for key in keys:
            row[key] = round(float(curves[key][idx]), 6)
        rows.append(row)
    write_json(path, {"columns": ["frame"] + keys, "rows": rows})


def primary_curve(curves: dict[str, np.ndarray], prefix: str, preferred: str | None = None) -> tuple[str, np.ndarray] | tuple[None, None]:
    if preferred and preferred in curves:
        return preferred, curves[preferred]
    keys = sorted(k for k in curves if k.startswith(prefix))
    if not keys:
        return None, None
    return keys[len(keys) // 2], curves[keys[len(keys) // 2]]


def group_positive_runs(frames: np.ndarray, mask: np.ndarray, stride: int, max_gap_frames: int = 32) -> list[tuple[int, int]]:
    intervals = []
    start = None
    last = None
    for frame, keep in zip(frames, mask):
        if keep:
            if start is None:
                start = int(frame)
            elif last is not None and int(frame) - last > max_gap_frames:
                intervals.append((start, last + stride))
                start = int(frame)
            last = int(frame)
    if start is not None and last is not None:
        intervals.append((start, last + stride))
    return merge_ranges(intervals)


def expand_peak(frames: np.ndarray, signal: np.ndarray, peak_idx: int, stride: int, stop_ratio: float = 0.25) -> tuple[int, int]:
    peak_value = float(signal[peak_idx])
    cutoff = max(0.0, stop_ratio * peak_value)
    left = peak_idx
    right = peak_idx
    while left > 0 and signal[left] > cutoff:
        left -= 1
    while right < len(signal) - 1 and signal[right] > cutoff:
        right += 1
    return int(frames[left]), int(frames[right] + stride)


def local_find_peaks(signal: np.ndarray, threshold: float, prominence: float) -> list[int]:
    if SCIPY_SIGNAL_AVAILABLE and find_peaks is not None:
        peaks, props = find_peaks(signal, height=threshold, prominence=max(prominence, 1e-6))
        return [int(p) for p in peaks]
    peaks = []
    for idx in range(1, len(signal) - 1):
        if signal[idx] >= threshold and signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1]:
            peaks.append(idx)
    return peaks


def add_interval(rows: list[dict], method: str, dataset: str, video_id: str, start: int, end: int, source: str) -> None:
    interval = clean_interval(start, end)
    if not interval:
        return
    rows.append(
        {
            "method": method,
            "dataset": dataset,
            "video_id": video_id,
            "start": interval[0],
            "end": interval[1],
            "source_path": source,
        }
    )


def generate_spectral_intervals(dataset: str, video_id: str, frames: np.ndarray, curves: dict[str, np.ndarray], stride: int, args) -> list[dict]:
    rows = []
    _, sg = primary_curve(curves, "sg_score_", "sg_score_17_2")
    _, residual = primary_curve(curves, "airpls_residual_", "airpls_residual_1000")
    trend100 = curves.get("rolling_mean_100")
    if residual is None:
        residual = curves["raw_score"] - np.median(curves["raw_score"])
    positive_residual = np.maximum(residual, 0)

    peak_signal = sg if sg is not None else curves["raw_score"]
    peak_threshold = mad_threshold(peak_signal, args.peak_mad_k)
    peak_prom = max(1e-6, mad_threshold(np.abs(peak_signal - np.median(peak_signal)), args.peak_mad_k) - np.median(np.abs(peak_signal - np.median(peak_signal))))
    for peak_idx in local_find_peaks(peak_signal, peak_threshold, peak_prom):
        start, end = expand_peak(frames, positive_residual if residual is not None else peak_signal, peak_idx, stride, args.peak_stop_ratio)
        add_interval(rows, "SG-Peak", dataset, video_id, start, end, "spectral_decomposition")

    residual_threshold = mad_threshold(positive_residual, args.residual_mad_k)
    for start, end in group_positive_runs(frames, positive_residual > residual_threshold, stride, args.merge_gap_frames):
        add_interval(rows, "AirPLS-Residual", dataset, video_id, start, end, "spectral_decomposition")

    if trend100 is not None:
        for start, end in group_positive_runs(frames, trend100 > args.score_threshold, stride, args.merge_gap_frames):
            overlap_support = np.max(positive_residual[(frames >= start) & (frames < end)]) if np.any((frames >= start) & (frames < end)) else 0.0
            if overlap_support > 0 or np.max(curves["raw_score"][(frames >= start) & (frames < end)]) >= args.score_threshold:
                add_interval(rows, "Trend-Guided-100F", dataset, video_id, start, end, "spectral_decomposition")
    return rows


def interval_query(frames: np.ndarray, values: np.ndarray, start: int, end: int) -> np.ndarray:
    lo = int(np.searchsorted(frames, start, side="left"))
    hi = int(np.searchsorted(frames, end, side="left"))
    return values[lo:hi]


def peak_count_in_interval(frames: np.ndarray, signal: np.ndarray, start: int, end: int, threshold: float) -> tuple[int, float]:
    peaks = local_find_peaks(signal, threshold, threshold - float(np.median(signal)) if len(signal) else 0.0)
    vals = [float(signal[idx]) for idx in peaks if start <= frames[idx] < end]
    return len(vals), max(vals) if vals else 0.0


def extract_features(intervals: list[dict], decomp: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for item in intervals:
        key = (item["dataset"], item["video_id"])
        if key not in decomp:
            continue
        data = decomp[key]
        frames = data["frames"]
        curves = data["curves"]
        start = int(item["start"])
        end = int(item["end"])
        raw = interval_query(frames, curves["raw_score"], start, end)
        sg_name, sg = primary_curve(curves, "sg_score_", "sg_score_17_2")
        base_name, baseline = primary_curve(curves, "airpls_baseline_", "airpls_baseline_1000")
        res_name, residual = primary_curve(curves, "airpls_residual_", "airpls_residual_1000")
        trend100 = curves.get("rolling_mean_100", curves["raw_score"])
        trend300 = curves.get("rolling_mean_300", curves["raw_score"])
        sg_vals = interval_query(frames, sg if sg is not None else curves["raw_score"], start, end)
        baseline_vals = interval_query(frames, baseline if baseline is not None else np.zeros_like(curves["raw_score"]), start, end)
        residual_curve = residual if residual is not None else curves["raw_score"] - np.median(curves["raw_score"])
        residual_vals = interval_query(frames, residual_curve, start, end)
        pos_res = np.maximum(residual_vals, 0)
        trend100_vals = interval_query(frames, trend100, start, end)
        trend300_vals = interval_query(frames, trend300, start, end)
        residual_threshold = mad_threshold(np.maximum(residual_curve, 0), 3.0)
        pcount, pmax = peak_count_in_interval(frames, np.maximum(residual_curve, 0), start, end, residual_threshold)
        sg_slope = np.abs(np.diff(sg_vals)) if len(sg_vals) > 1 else np.asarray([], dtype=float)
        length = max(1, end - start)
        rows.append(
            {
                "method": item["method"],
                "dataset": item["dataset"],
                "video_id": item["video_id"],
                "start": start,
                "end": end,
                "source_path": item.get("source_path", ""),
                "raw_mean": mean_or_nan(raw),
                "raw_max": max_or_nan(raw),
                "sg_curve": sg_name or "",
                "sg_mean": mean_or_nan(sg_vals),
                "sg_max": max_or_nan(sg_vals),
                "sg_slope_abs_mean": mean_or_nan(sg_slope),
                "airpls_baseline_curve": base_name or "",
                "airpls_baseline_mean": mean_or_nan(baseline_vals),
                "airpls_residual_curve": res_name or "",
                "airpls_residual_mean": mean_or_nan(residual_vals),
                "airpls_residual_max": max_or_nan(residual_vals),
                "airpls_residual_area": float(np.sum(pos_res)) if len(pos_res) else 0.0,
                "residual_peak_count": pcount,
                "residual_peak_max_prominence": pmax,
                "trend_100_mean": mean_or_nan(trend100_vals),
                "trend_100_max": max_or_nan(trend100_vals),
                "trend_300_mean": mean_or_nan(trend300_vals),
                "trend_300_max": max_or_nan(trend300_vals),
                "interval_length": length,
                "low_residual_ratio": float(np.mean(pos_res <= 0.05)) if len(pos_res) else 1.0,
                "isolated_spike_count": pcount,
                "micro_density": "",
                "gap_count": "",
            }
        )
    return rows


def mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def max_or_nan(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else math.nan


def normalize_feature(values: list[float]) -> list[float]:
    arr = np.asarray([0.0 if math.isnan(v) or math.isinf(v) else float(v) for v in values], dtype=float)
    lo = float(np.min(arr)) if len(arr) else 0.0
    hi = float(np.max(arr)) if len(arr) else 1.0
    if hi <= lo:
        return [0.0 for _ in arr]
    return [float((v - lo) / (hi - lo)) for v in arr]


def build_fusion(features: list[dict], args) -> list[dict]:
    source_methods = {"Peak-Aware-Refined", "Hierarchical-Merged", "SG-Peak", "AirPLS-Residual", "Trend-Guided-100F"}
    candidates = [row for row in features if row["method"] in source_methods]
    fields = {
        "raw_max": normalize_feature([as_float(r.get("raw_max"), 0) for r in candidates]),
        "sg_max": normalize_feature([as_float(r.get("sg_max"), 0) for r in candidates]),
        "airpls_residual_max": normalize_feature([as_float(r.get("airpls_residual_max"), 0) for r in candidates]),
        "trend_100_mean": normalize_feature([as_float(r.get("trend_100_mean"), 0) for r in candidates]),
        "residual_peak_count": normalize_feature([as_float(r.get("residual_peak_count"), 0) for r in candidates]),
        "interval_length": normalize_feature([as_float(r.get("interval_length"), 0) for r in candidates]),
    }
    by_video = defaultdict(list)
    scored = []
    for idx, row in enumerate(candidates):
        score = (
            0.25 * fields["raw_max"][idx]
            + 0.20 * fields["sg_max"][idx]
            + 0.25 * fields["airpls_residual_max"][idx]
            + 0.15 * fields["trend_100_mean"][idx]
            + 0.10 * fields["residual_peak_count"][idx]
            - 0.15 * fields["interval_length"][idx]
            - 0.10 * as_float(row.get("low_residual_ratio"), 1.0)
        )
        out = dict(row)
        out["fusion_score"] = round(float(score), 6)
        scored.append(out)
        if score >= args.fusion_threshold:
            by_video[(row["dataset"], row["video_id"])].append((int(row["start"]), int(row["end"])))

    fused = []
    for (dataset, video_id), intervals in by_video.items():
        for start, end in merge_with_gap(intervals, args.merge_gap_frames):
            add_interval(fused, "Spectral-Fusion-Refined", dataset, video_id, start, end, "spectral_fusion")
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


def plot_video(path: Path, dataset: str, video_id: str, gt_rows: list[dict], interval_rows: list[dict], data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = data["frames"]
    curves = data["curves"]
    fig, axes = plt.subplots(7, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 3]})
    tracks = [
        ("GT", gt_rows, "#D55E00"),
        ("Peak-Aware", [r for r in interval_rows if r["method"] == "Peak-Aware-Refined"], "#009E73"),
        ("SG-Peak", [r for r in interval_rows if r["method"] == "SG-Peak"], "#CC79A7"),
        ("AirPLS", [r for r in interval_rows if r["method"] == "AirPLS-Residual"], "#0072B2"),
        ("Trend", [r for r in interval_rows if r["method"] == "Trend-Guided-100F"], "#E69F00"),
        ("Fusion", [r for r in interval_rows if r["method"] == "Spectral-Fusion-Refined"], "#000000"),
    ]
    for ax, (label, rows, color) in zip(axes[:6], tracks):
        for row in rows:
            start = int(row.get("start", row.get("gt_start", 0)))
            end = int(row.get("end", row.get("gt_end", 0)))
            ax.axvspan(start, end, ymin=0.15, ymax=0.85, color=color, alpha=0.75)
        ax.set_ylabel(label)
        ax.set_yticks([])
    ax = axes[-1]
    ax.plot(frames, curves["raw_score"], label="raw", color="#4C78A8", linewidth=1.0)
    sg_name, sg = primary_curve(curves, "sg_score_", "sg_score_17_2")
    if sg is not None:
        ax.plot(frames, sg, label=sg_name, color="#F58518", linewidth=1.0)
    base_name, baseline = primary_curve(curves, "airpls_baseline_", "airpls_baseline_1000")
    if baseline is not None:
        ax.plot(frames, baseline, label=base_name, color="#54A24B", linewidth=1.0)
    res_name, residual = primary_curve(curves, "airpls_residual_", "airpls_residual_1000")
    if residual is not None:
        ax.plot(frames, np.maximum(residual, 0), label=f"positive {res_name}", color="#E45756", linewidth=1.0)
    if "rolling_mean_100" in curves:
        ax.plot(frames, curves["rolling_mean_100"], label="trend 100F", color="#72B7B2", linewidth=1.0)
    if "rolling_mean_300" in curves:
        ax.plot(frames, curves["rolling_mean_300"], label="trend 300F", color="#B279A2", linewidth=1.0)
    ax.set_ylabel("score")
    ax.set_xlabel("frame")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.suptitle(f"{dataset} | {video_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_plot(rows: list[dict], x_field: str, y_field: str, path: Path, title: str) -> None:
    all_rows = [r for r in rows if r["dataset"] == "ALL"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for row in all_rows:
        x = as_float(row.get(x_field), math.nan)
        y = as_float(row.get(y_field), math.nan)
        if math.isnan(x) or math.isnan(y):
            continue
        ax.scatter([x], [y], s=90)
        ax.annotate(row["method"], (x, y), fontsize=8)
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bar_plot(rows: list[dict], field: str, path: Path, title: str) -> None:
    all_rows = sorted([r for r in rows if r["dataset"] == "ALL"], key=lambda r: as_float(r.get("balanced_score"), -999), reverse=True)
    labels = [r["method"] for r in all_rows]
    values = [as_float(r.get(field), 0) for r in all_rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(field)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, args, summary: dict, overall: list[dict], warnings: list[dict]) -> None:
    all_rows = {r["method"]: r for r in overall if r["dataset"] == "ALL"}
    fusion = all_rows.get("Spectral-Fusion-Refined", {})
    peak = all_rows.get("Peak-Aware-Refined", {})
    best_new = max((all_rows.get(m, {}) for m in ["SG-Peak", "AirPLS-Residual", "Trend-Guided-100F", "Spectral-Fusion-Refined"] if all_rows.get(m)), key=lambda r: as_float(r.get("balanced_score"), -999), default={})
    verdict = "did not beat" if as_float(fusion.get("balanced_score"), -999) <= as_float(peak.get("balanced_score"), -999) else "beat"
    lines = [
        "# Spectral Score Decomposition Report",
        "",
        "## Motivation",
        "",
        "The anomaly score curve is treated as a spectroscopy-like signal containing noise, baseline/trend, local peaks, and broad bands. The goal is not to train a new model, but to test interpretable preprocessing and interval fusion rules.",
        "",
        "## Literature-inspired design",
        "",
        "- Savitzky-Golay smoothing uses local least-squares polynomial fitting to smooth noise while preserving peak shape better than a simple moving average.",
        "- airPLS estimates an adaptive baseline with iteratively reweighted penalized least squares, reducing the influence of high peak points on the baseline.",
        "- Multiple preprocessing permutations are compared downstream instead of selecting one curve by intuition.",
        "",
        "## Parameters",
        "",
        f"- SG windows: {args.sg_window_lengths}; SG polyorders: {args.sg_polyorders}.",
        f"- airPLS lambdas: {args.airpls_lambdas}; order={args.airpls_order}; itermax={args.airpls_itermax}.",
        f"- trend windows: {args.trend_windows} frames.",
        f"- score_threshold: {args.score_threshold}; fusion_threshold: {args.fusion_threshold}.",
        "",
        "## Method",
        "",
        "- `SG-Peak`: detects peaks on the primary SG-smoothed curve using median + k*MAD, then expands boundaries until residual evidence decays.",
        "- `AirPLS-Residual`: detects contiguous high positive residual regions after baseline subtraction.",
        "- `Trend-Guided-100F`: selects 100F trend-positive regions and requires raw/residual support.",
        "- `Spectral-Fusion-Refined`: fuses existing peak-aware/hierarchical intervals with spectral candidates using normalized raw, SG, residual, trend, peak-count, length, and low-residual evidence.",
        "",
        "## Results",
        "",
        f"- Videos processed: {summary['video_count']}; GT intervals: {summary['gt_interval_count']}; warnings: {len(warnings)}.",
        f"- Best new spectral method by balanced_score: `{best_new.get('method', 'NA')}`.",
        "",
        "| method | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | balanced_score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["Peak-Aware-Refined", "SG-Peak", "AirPLS-Residual", "Trend-Guided-100F", "Spectral-Fusion-Refined", "Window-100F", "Window-300F", "TopK-10"]:
        row = all_rows.get(method)
        if row:
            lines.append(
                f"| `{method}` | {fmt(row.get('GT_coverage'))} | {fmt(row.get('predicted_GT_fraction'))} | "
                f"{fmt(row.get('supportable_gt_coverage'))} | {fmt(row.get('unsupportable_gt_coverage'))} | "
                f"{fmt(row.get('predicted_duration_ratio'))} | {fmt(row.get('balanced_score'))} |"
            )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"`Spectral-Fusion-Refined` {verdict} `Peak-Aware-Refined` by the current balanced_score. If it failed to improve, the likely reason is that the VDA score evidence and peak-aware rules already capture most recoverable signal, while extra residual/trend intervals add duration or miss sparse GT intervals.",
            "",
            "Random-Same-Length is retained only as a sanity baseline and must not be recommended as a detector.",
            "",
            "## Limitations",
            "",
            "- GT intervals with no VDA score response cannot be recovered by post-processing alone.",
            "- airPLS lambda strongly affects baseline and residual estimates.",
            "- Large SG windows can smooth away short anomalies.",
            "- Fusion weights are fixed rules, not learned on a validation set.",
            "- Score stride and sparse labels can distort short-interval evidence.",
            "",
            "## Next steps",
            "",
            "- Sweep fusion weights and residual thresholds on a validation split.",
            "- Keep the coverage-purity frontier as the operating-point selection tool.",
            "- If spectral decomposition cannot beat peak-aware refinement, return to VDA score generation and annotation consistency rather than only post-processing.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    except (TypeError, ValueError):
        return "NA"


def archive_outputs(root: Path, output_dir: Path, report: Path) -> None:
    root = root.resolve()
    archive_root = output_dir.parent.resolve()
    shutil.copy2(report, archive_root / "spectral-score-decomposition_report.md")
    program = archive_root / "programs/scripts/run_spectral_score_decomposition.py"
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "scripts/run_spectral_score_decomposition.py", program)
    manifest = [
        "# spectral-score-decomposition",
        "",
        f"- archive_folder: `{archive_root.relative_to(root).as_posix()}`",
        "- primary_report: `spectral-score-decomposition_report.md`",
        "",
        "## Contents",
        "",
        "- `programs/`: copied script used for this experiment.",
        "- `outputs/`: decomposition curves, interval features, generated intervals, evaluation CSVs, figures, warnings, and report.",
    ]
    (archive_root / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    layout = root / "reports/ARTIFACT_LAYOUT.md"
    if layout.exists():
        line = f"- `{archive_root.relative_to(root).as_posix()}/`"
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
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--enable_sg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_airpls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sg_window_lengths", default="9,17,31")
    parser.add_argument("--sg_polyorders", default="2,3")
    parser.add_argument("--airpls_lambdas", default="100,1000,10000")
    parser.add_argument("--airpls_order", type=int, default=2)
    parser.add_argument("--airpls_itermax", type=int, default=20)
    parser.add_argument("--trend_windows", default="30,100,300")
    parser.add_argument("--score_threshold", type=float, default=0.6)
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_plots", type=int, default=40)
    parser.add_argument("--peak_mad_k", type=float, default=3.0)
    parser.add_argument("--residual_mad_k", type=float, default=3.0)
    parser.add_argument("--peak_stop_ratio", type=float, default=0.25)
    parser.add_argument("--merge_gap_frames", type=int, default=48)
    parser.add_argument("--fusion_threshold", type=float, default=0.35)
    args = parser.parse_args()
    args.sg_window_lengths = parse_int_list(args.sg_window_lengths)
    args.sg_polyorders = parse_int_list(args.sg_polyorders)
    args.airpls_lambdas = parse_float_list(args.airpls_lambdas)
    args.trend_windows = parse_int_list(args.trend_windows)

    root = Path.cwd()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    gt_by_video = defaultdict(list)
    for row in gt_rows:
        gt_by_video[(row["dataset"], row["video_id"])].append(row)

    decomp = {}
    spectral_intervals = []
    for key, meta in sorted(inventory.items()):
        dataset, video_id = key
        score_path = Path(meta.get("score_json_path", ""))
        if not score_path.is_absolute():
            score_path = root / score_path
        if not score_path.exists():
            warnings.append({"video_key": f"{dataset}/{video_id}", "stage": "load_score", "warning": f"missing score file: {score_path}"})
            continue
        frames, values = load_score_series(score_path)
        if len(values) < 3:
            warnings.append({"video_key": f"{dataset}/{video_id}", "stage": "load_score", "warning": f"too few score points: {len(values)}"})
            continue
        stride = estimate_stride(frames, meta.get("score_stride_est", 16))
        curves = build_decomposition(frames, values, args, warnings, f"{dataset}/{video_id}")
        decomp[key] = {"frames": frames, "curves": curves, "stride": stride}
        save_curves(output_dir / "curves" / dataset / f"{safe_name(video_id)}_spectral_decomposition.json", frames, curves)
        spectral_intervals.extend(generate_spectral_intervals(dataset, video_id, frames, curves, stride, args))

    existing_intervals = []
    skipped = auto_scan_methods(args.existing_interval_root, existing_intervals)
    warnings.extend({"video_key": "", "stage": "existing_interval_scan", "warning": row["reason"], "source_path": row["source_path"]} for row in skipped if not row.get("parsed"))
    add_window_methods(existing_intervals, inventory, [100, 300], args.score_threshold, root)
    feature_candidates = existing_intervals + spectral_intervals
    features = extract_features(feature_candidates, decomp)
    fusion_intervals, scored_fusion_candidates = build_fusion(features, args)
    all_new = spectral_intervals + fusion_intervals
    all_for_eval = [row for row in existing_intervals if row["method"] in EVAL_OLD_METHODS] + all_new

    per_video, overall, support_rows, iou_rows, ranking = evaluate_methods(all_for_eval, gt_rows, inventory, [0.1, 0.3, 0.5])

    write_csv(output_dir / "spectral_generated_intervals.csv", all_new)
    write_json(output_dir / "spectral_fusion_intervals.json", {"intervals": fusion_intervals})
    write_csv(output_dir / "spectral_fusion_candidates_scored.csv", scored_fusion_candidates)
    write_csv(output_dir / "spectral_interval_features.csv", features)
    write_csv(output_dir / "spectral_method_per_video_metrics.csv", per_video)
    write_csv(output_dir / "spectral_method_overall_metrics.csv", overall)
    write_csv(output_dir / "spectral_method_supportability_metrics.csv", support_rows)
    write_csv(output_dir / "spectral_method_iou_event_metrics.csv", iou_rows)
    write_csv(output_dir / "spectral_method_ranking.csv", ranking)
    write_csv(output_dir / "spectral_decomposition_warnings.csv", warnings, ["video_key", "stage", "warning", "source_path"])

    if args.make_plots:
        aggregate_plot(overall, "predicted_GT_fraction", "GT_coverage", output_dir / "fig_spectral_coverage_vs_purity.png", "Spectral Coverage-Purity Trade-off")
        bar_plot(overall, "predicted_duration_ratio", output_dir / "fig_spectral_predicted_duration_ratio.png", "Predicted Duration Ratio")
        aggregate_plot(overall, "supportable_gt_coverage", "unsupportable_gt_coverage", output_dir / "fig_spectral_supportable_vs_unsupportable.png", "Supportable vs Unsupportable Coverage")
        plotted = 0
        by_video_intervals = defaultdict(list)
        for row in all_for_eval:
            by_video_intervals[(row["dataset"], row["video_id"])].append(row)
        for key in sorted(decomp):
            if plotted >= args.max_plots:
                break
            if key not in gt_by_video:
                continue
            dataset, video_id = key
            plot_video(
                output_dir / "visualizations" / dataset / f"{safe_name(video_id)}_spectral_decomposition.png",
                dataset,
                video_id,
                gt_by_video[key],
                by_video_intervals.get(key, []),
                decomp[key],
            )
            plotted += 1

    summary = {
        "video_count": len(decomp),
        "gt_interval_count": len(gt_rows),
        "generated_interval_count": len(all_new),
        "fusion_interval_count": len(fusion_intervals),
        "feature_row_count": len(features),
        "warning_count": len(warnings),
        "scipy_signal_available": SCIPY_SIGNAL_AVAILABLE,
        "scipy_sparse_available": SCIPY_SPARSE_AVAILABLE,
    }
    write_json(output_dir / "spectral_score_decomposition_summary.json", summary)
    report = output_dir / "spectral_score_decomposition_report.md"
    write_report(report, args, summary, overall, warnings)
    archive_outputs(root, output_dir, report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
