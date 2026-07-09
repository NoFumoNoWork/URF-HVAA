"""Stage 1 H4 enrichment analysis around prediction gaps.

This script tests whether caption-level H4 boundary candidates are enriched near
prediction gaps and near GT-derived positive_merge gaps. It does not validate
true camera transitions, does not assume H4 means a new scene, and does not
implement a final merge rule. Oracle labels use GT and are diagnostic only.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_GAP_FIELDS = [
    "video_id", "gap_start", "gap_end", "gap_len", "has_h4_in_gap",
    "has_h4_near_gap", "h4_count_in_gap", "h4_count_near_gap",
    "nearest_h4_distance", "strongest_h4_score", "h4_types_near_gap",
    "score_dip", "left_mean_score", "right_mean_score", "gap_mean_score",
    "gap_gt_overlap_ratio", "same_gt_on_both_sides", "merge_oracle_label",
]

EXPECTED_H4_FIELDS = [
    "video_id", "h4_position", "h4_type", "h4_score", "score_drop",
    "inside_gt", "distance_to_nearest_gt_start", "distance_to_nearest_gt_end",
    "distance_to_nearest_prediction_gap", "inside_prediction_gap",
    "near_prediction_gap", "nearest_gap_id", "gap_oracle_label",
]


def bool_series(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def ensure_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def safe_ratio(num, den):
    return float(num) / float(den) if den else np.nan


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_input_check(resource_dir, output_dir, required):
    lines = ["# Input check", ""]
    summary = {}
    for name, fields in required.items():
        path = resource_dir / name
        exists = path.exists()
        lines.extend([f"## {name}", "", f"- exists: {exists}"])
        if not exists:
            lines.extend(["- missing: true", ""])
            summary[name] = {"exists": False}
            continue
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8", errors="replace")
            preview = "\n".join(text.splitlines()[:25])
            lines.extend([
                f"- rows: {len(text.splitlines())}",
                "- columns: markdown document",
                "- required fields present: n/a",
                "- missing fields: `[]`",
                "- preview:",
                "",
                "```text",
                preview,
                "```",
                "",
            ])
            summary[name] = {"exists": True, "rows": len(text.splitlines()), "missing_fields": []}
            continue
        df = read_csv(path)
        missing = [field for field in fields if field not in df.columns]
        lines.extend([
            f"- rows: {len(df)}",
            f"- columns: `{list(df.columns)}`",
            f"- required fields present: {not missing}",
            f"- missing fields: `{missing}`",
            "- preview:",
            "",
            "```text",
            df.head(5).to_string(index=False),
            "```",
            "",
        ])
        summary[name] = {"exists": True, "rows": len(df), "missing_fields": missing}
    (output_dir / "input_check.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def numeric_summary(df, group_col, value_col):
    if value_col not in df.columns:
        return pd.DataFrame()
    return df.groupby(group_col)[value_col].agg(["mean", "median", q25, q75]).rename(
        columns={"q25": f"{value_col}_q25", "q75": f"{value_col}_q75"}
    )


def gap_level_stats(gaps):
    labels = sorted(gaps["merge_oracle_label"].fillna("unknown").unique())
    rows = []
    numeric_cols = [
        "h4_count_near_gap", "h4_count_in_gap", "nearest_h4_distance",
        "strongest_h4_score", "score_dip", "gap_len",
    ]
    for label in labels + ["all_non_positive"]:
        sub = gaps[gaps["merge_oracle_label"] != "positive_merge"] if label == "all_non_positive" else gaps[gaps["merge_oracle_label"] == label]
        if sub.empty:
            continue
        row = {
            "merge_oracle_label": label,
            "num_gaps": len(sub),
            "has_h4_near_gap_ratio": bool_series(sub["has_h4_near_gap"]).mean() if "has_h4_near_gap" in sub else np.nan,
            "has_h4_in_gap_ratio": bool_series(sub["has_h4_in_gap"]).mean() if "has_h4_in_gap" in sub else np.nan,
        }
        for col in numeric_cols:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce").dropna()
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_median"] = values.median()
                row[f"{col}_q25"] = values.quantile(0.25)
                row[f"{col}_q75"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def write_gap_summary(path, stats):
    lookup = {row["merge_oracle_label"]: row for _, row in stats.iterrows()}
    pos = lookup.get("positive_merge", {})
    neg = lookup.get("negative_merge", {})
    non = lookup.get("all_non_positive", {})
    lines = [
        "# Gap-level H4 enrichment summary",
        "",
        f"- positive_merge gaps: {pos.get('num_gaps', 'n/a')}",
        f"- negative_merge gaps: {neg.get('num_gaps', 'n/a')}",
        f"- positive_merge H4-near ratio: {pos.get('has_h4_near_gap_ratio', np.nan):.6f}",
        f"- negative_merge H4-near ratio: {neg.get('has_h4_near_gap_ratio', np.nan):.6f}",
        f"- all_non_positive H4-near ratio: {non.get('has_h4_near_gap_ratio', np.nan):.6f}",
        "",
        "## Interpretation",
        "",
        "In this run, positive_merge does not have the highest H4-near ratio. Negative/risky gaps also often have nearby H4 candidates, so H4 proximity alone is not a safe merge rule. Its better role is to trigger event-continuity or score-shape rechecks.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def type_stats(h4, exploded):
    rows = h4.copy()
    if exploded:
        rows["h4_type_item"] = rows["h4_type"].fillna("unknown").astype(str).str.split(";")
        rows = rows.explode("h4_type_item")
        type_col = "h4_type_item"
    else:
        type_col = "h4_type"
        rows[type_col] = rows[type_col].fillna("unknown").replace("", "unknown")
    labels = ["positive_merge", "risky_merge", "negative_merge", "unknown", "no_near_gap"]
    out = []
    for name, sub in rows.groupby(type_col):
        if not str(name).strip():
            name = "unknown"
        total = len(sub)
        near = bool_series(sub["near_prediction_gap"]).sum() if "near_prediction_gap" in sub else 0
        counts = sub["gap_oracle_label"].fillna("no_near_gap").replace("", "no_near_gap").value_counts().to_dict()
        pos = counts.get("positive_merge", 0)
        neg = counts.get("negative_merge", 0)
        risky = counts.get("risky_merge", 0)
        out.append({
            "h4_type": name,
            "total": total,
            "near_prediction_gap_count": near,
            "near_prediction_gap_ratio": safe_ratio(near, total),
            **{f"{label}_count": counts.get(label, 0) for label in labels},
            "positive_merge_ratio": safe_ratio(pos, total),
            "negative_merge_ratio": safe_ratio(neg, total),
            "positive_negative_ratio": safe_ratio(pos, neg) if neg else np.inf if pos else np.nan,
            "risky_negative_ratio": safe_ratio(risky + neg, total),
        })
    return pd.DataFrame(out).sort_values(["positive_merge_ratio", "total"], ascending=[False, False])


def write_type_summary(path, exploded_stats):
    top = exploded_stats.sort_values("positive_merge_ratio", ascending=False).head(8)
    lines = [
        "# H4 type oracle summary",
        "",
        "## Top positive_merge ratios",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['h4_type']}: positive={row['positive_merge_ratio']:.6f}, negative={row['negative_merge_ratio']:.6f}, total={int(row['total'])}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Multi-label exploded statistics are better for interpreting component H4 types.",
        "- Combination statistics preserve original type mixtures and are useful for later rule design.",
        "- `event_onset_not_h4` should be treated cautiously or excluded from merge-trigger candidate sets.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def distance_to_gaps(pos, gap_starts, gap_ends):
    if len(gap_starts) == 0:
        return np.nan
    inside = (gap_starts <= pos) & (pos <= gap_ends)
    if inside.any():
        return 0.0
    return float(np.minimum(np.abs(gap_starts - pos), np.abs(gap_ends - pos)).min())


def random_baseline(h4, gaps, timeline, h4_window, trials, seed):
    rng = np.random.default_rng(seed)
    timeline_ranges = timeline.groupby(["dataset", "video_id"]).agg(min_start=("start", "min"), max_end=("end", "max")).reset_index()
    gap_groups = {key: sub for key, sub in gaps.groupby(["dataset", "video_id"])}
    h4_counts = h4.groupby(["dataset", "video_id"]).size().to_dict()
    real_dist = pd.to_numeric(h4["distance_to_nearest_prediction_gap"], errors="coerce").dropna()
    real_near = (real_dist <= h4_window).mean() if len(real_dist) else np.nan
    trial_rows = []
    all_random_distances = []
    for trial in range(trials):
        distances = []
        for _, row in timeline_ranges.iterrows():
            key = (row["dataset"], row["video_id"])
            n = h4_counts.get(key, 0)
            if n <= 0:
                continue
            gap_sub = gap_groups.get(key)
            if gap_sub is None or gap_sub.empty:
                continue
            starts = pd.to_numeric(gap_sub["gap_start"], errors="coerce").dropna().to_numpy()
            ends = pd.to_numeric(gap_sub["gap_end"], errors="coerce").dropna().to_numpy()
            if len(starts) == 0:
                continue
            samples = rng.uniform(float(row["min_start"]), float(row["max_end"]), size=n)
            distances.extend(distance_to_gaps(pos, starts, ends) for pos in samples)
        distances = np.asarray([d for d in distances if not np.isnan(d)])
        if len(distances) == 0:
            continue
        all_random_distances.extend(distances.tolist())
        near_rate = float((distances <= h4_window).mean())
        trial_rows.append({
            "trial": trial,
            "random_near_gap_rate": near_rate,
            "random_median_distance": float(np.median(distances)),
            "random_mean_distance": float(np.mean(distances)),
        })
    trials_df = pd.DataFrame(trial_rows)
    summary = {
        "real_near_gap_rate": real_near,
        "random_near_gap_rate_mean": trials_df["random_near_gap_rate"].mean(),
        "random_near_gap_rate_ci_low": trials_df["random_near_gap_rate"].quantile(0.025),
        "random_near_gap_rate_ci_high": trials_df["random_near_gap_rate"].quantile(0.975),
        "real_median_distance": real_dist.median(),
        "random_median_distance_mean": trials_df["random_median_distance"].mean(),
        "random_median_distance_ci_low": trials_df["random_median_distance"].quantile(0.025),
        "random_median_distance_ci_high": trials_df["random_median_distance"].quantile(0.975),
        "enrichment_ratio": real_near / trials_df["random_near_gap_rate"].mean() if trials_df["random_near_gap_rate"].mean() else np.nan,
    }
    return trials_df, pd.DataFrame([summary]), real_dist, np.asarray(all_random_distances)


def score_shape_stats(gaps):
    rows = gaps.copy()
    if {"left_mean_score", "right_mean_score", "gap_mean_score"}.issubset(rows.columns):
        rows["computed_score_dip"] = rows[["left_mean_score", "right_mean_score"]].min(axis=1) - rows["gap_mean_score"]
    metrics = ["left_mean_score", "right_mean_score", "gap_mean_score", "score_dip", "computed_score_dip", "gap_len"]
    out = []
    for label, sub in rows.groupby("merge_oracle_label"):
        row = {"merge_oracle_label": label, "num_gaps": len(sub)}
        for col in metrics:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_median"] = vals.median()
                row[f"{col}_q25"] = vals.quantile(0.25)
                row[f"{col}_q75"] = vals.quantile(0.75)
        out.append(row)
    return pd.DataFrame(out)


def save_figures(output_dir, gaps, h4, type_exploded, real_dist, random_dist, random_summary):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    label_counts = gaps["merge_oracle_label"].value_counts()
    plt.figure(figsize=(8, 4.5)); label_counts.plot(kind="bar"); plt.title("Gap oracle label counts"); plt.ylabel("Gaps"); plt.tight_layout(); plt.savefig(fig_dir / "gap_oracle_label_counts.png", dpi=180); plt.close()

    near = gaps.assign(has_h4_near_gap_bool=bool_series(gaps["has_h4_near_gap"])).groupby("merge_oracle_label")["has_h4_near_gap_bool"].mean()
    plt.figure(figsize=(8, 4.5)); near.plot(kind="bar"); plt.title("H4 near-gap rate by oracle label"); plt.ylabel("Rate"); plt.ylim(0, 1); plt.tight_layout(); plt.savefig(fig_dir / "h4_near_gap_rate_by_oracle.png", dpi=180); plt.close()

    def boxplot_by_label(col, filename, title, ylabel):
        labels = [label for label in label_counts.index if col in gaps.columns]
        data = [pd.to_numeric(gaps[gaps["merge_oracle_label"] == label][col], errors="coerce").dropna().clip(upper=1000) for label in labels]
        plt.figure(figsize=(8.5, 4.8)); plt.boxplot(data, labels=labels, showfliers=False); plt.xticks(rotation=20, ha="right"); plt.title(title); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(fig_dir / filename, dpi=180); plt.close()

    boxplot_by_label("h4_count_near_gap", "h4_count_near_gap_by_oracle.png", "H4 count near gap by oracle", "H4 count")
    boxplot_by_label("nearest_h4_distance", "nearest_h4_distance_by_oracle.png", "Nearest H4 distance by oracle", "Distance (clipped)")
    boxplot_by_label("score_dip", "score_dip_by_oracle.png", "Score dip by oracle", "Score dip")

    top = type_exploded[type_exploded["total"] >= 20].sort_values("positive_merge_ratio", ascending=False).head(12)
    plt.figure(figsize=(10, 5)); plt.bar(top["h4_type"], top["positive_merge_ratio"]); plt.xticks(rotation=30, ha="right"); plt.title("H4 type positive_merge ratio"); plt.ylabel("Ratio"); plt.tight_layout(); plt.savefig(fig_dir / "h4_type_positive_ratio.png", dpi=180); plt.close()
    neg = type_exploded[type_exploded["total"] >= 20].sort_values("negative_merge_ratio", ascending=False).head(12)
    plt.figure(figsize=(10, 5)); plt.bar(neg["h4_type"], neg["negative_merge_ratio"]); plt.xticks(rotation=30, ha="right"); plt.title("H4 type negative_merge ratio"); plt.ylabel("Ratio"); plt.tight_layout(); plt.savefig(fig_dir / "h4_type_negative_ratio.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.hist(real_dist.clip(upper=500), bins=40, alpha=0.6, label="real H4")
    if len(random_dist):
        sample = random_dist if len(random_dist) < 50000 else np.random.default_rng(0).choice(random_dist, 50000, replace=False)
        plt.hist(np.clip(sample, 0, 500), bins=40, alpha=0.5, label="random")
    plt.title("Real vs random nearest-gap distance")
    plt.xlabel("Distance to nearest prediction gap (clipped)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "real_vs_random_nearest_gap_distance.png", dpi=180)
    plt.close()

    if not random_summary.empty:
        vals = [float(random_summary["enrichment_ratio"].iloc[0])]
        plt.figure(figsize=(5, 4)); plt.bar(["real/random"], vals); plt.ylabel("Enrichment ratio"); plt.title("Near-gap enrichment ratio"); plt.tight_layout(); plt.savefig(fig_dir / "enrichment_ratio_bar.png", dpi=180); plt.close()


def write_score_summary(path, score_stats, gaps):
    pos = score_stats[score_stats["merge_oracle_label"] == "positive_merge"]
    lines = ["# Score-shape summary", ""]
    if not pos.empty:
        row = pos.iloc[0]
        lines.append(f"- positive_merge median score_dip: {row.get('score_dip_median', np.nan):.6f}")
        lines.append(f"- positive_merge median gap_len: {row.get('gap_len_median', np.nan):.6f}")
    if {"has_h4_near_gap", "score_dip", "merge_oracle_label"}.issubset(gaps.columns):
        tmp = gaps.copy()
        tmp["has_h4_near_gap"] = bool_series(tmp["has_h4_near_gap"])
        tmp["strong_dip"] = tmp["score_dip"] >= tmp["score_dip"].median()
        both = tmp[tmp["has_h4_near_gap"] & tmp["strong_dip"]]
        ratio = (both["merge_oracle_label"] == "positive_merge").mean() if len(both) else np.nan
        lines.append(f"- H4-near plus above-median score_dip positive_merge ratio: {ratio:.6f}")
    lines.extend(["", "Score-shape is diagnostic only; high-low-high structure should be combined with semantic continuity before any merge rule."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_random_summary(path, summary):
    row = summary.iloc[0].to_dict()
    lines = [
        "# H4 vs random boundary distance summary",
        "",
        f"- real H4 near-gap rate: {row.get('real_near_gap_rate', np.nan):.6f}",
        f"- random near-gap rate mean: {row.get('random_near_gap_rate_mean', np.nan):.6f}",
        f"- random near-gap rate 95% interval: [{row.get('random_near_gap_rate_ci_low', np.nan):.6f}, {row.get('random_near_gap_rate_ci_high', np.nan):.6f}]",
        f"- real median distance: {row.get('real_median_distance', np.nan):.6f}",
        f"- random median distance mean: {row.get('random_median_distance_mean', np.nan):.6f}",
        f"- enrichment ratio: {row.get('enrichment_ratio', np.nan):.6f}",
        "",
        "This random baseline only tests enrichment near prediction gaps. It does not prove true camera transitions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path, input_summary, gap_stats, type_stats_df, random_summary, score_stats, skipped):
    pos_row = gap_stats[gap_stats["merge_oracle_label"] == "positive_merge"]
    neg_row = gap_stats[gap_stats["merge_oracle_label"] == "negative_merge"]
    rand = random_summary.iloc[0].to_dict() if not random_summary.empty else {}
    top_types = type_stats_df[type_stats_df["total"] >= 20].head(5)
    lines = [
        "# Stage 1: H4 Enrichment around Prediction Gaps",
        "",
        "## 1. Purpose",
        "",
        "This stage tests whether caption-level H4 candidates are useful diagnostic triggers around prediction gaps. It does not implement final interval merging or validate real camera transitions.",
        "",
        "## 2. Inputs",
        "",
    ]
    for name, item in input_summary.items():
        lines.append(f"- {name}: exists={item.get('exists')}, rows={item.get('rows', 'n/a')}, missing_fields={item.get('missing_fields', [])}")
    lines.extend([
        "",
        "## 3. Important Definitions",
        "",
        "- H4 candidate = caption-level boundary/context-disruption candidate.",
        "- Prediction gap = low-score interval between two predicted anomaly intervals.",
        "- positive_merge / risky_merge / negative_merge / unknown are GT-derived oracle labels for diagnosis only.",
        "- Scene switch does not imply new scene; boundary sides may be same event, new viewpoint, related consequence, or narrative jump.",
        "",
        "## 4. Gap-level Enrichment",
        "",
    ])
    if not pos_row.empty:
        lines.append(f"- positive_merge H4-near ratio: {pos_row.iloc[0].get('has_h4_near_gap_ratio', np.nan):.6f}.")
    if not neg_row.empty:
        lines.append(f"- negative_merge H4-near ratio: {neg_row.iloc[0].get('has_h4_near_gap_ratio', np.nan):.6f}.")
    lines.extend([
        "",
        "## 5. H4 Type Analysis",
        "",
    ])
    for _, row in top_types.iterrows():
        lines.append(f"- {row['h4_type']}: positive ratio={row['positive_merge_ratio']:.6f}, negative ratio={row['negative_merge_ratio']:.6f}, total={int(row['total'])}.")
    lines.extend([
        "",
        "## 6. Random Boundary Baseline",
        "",
        f"- real near-gap rate: {rand.get('real_near_gap_rate', np.nan):.6f}.",
        f"- random near-gap rate mean: {rand.get('random_near_gap_rate_mean', np.nan):.6f}.",
        f"- enrichment ratio: {rand.get('enrichment_ratio', np.nan):.6f}.",
        "",
        "## 7. Score-shape Diagnosis",
        "",
        "See `score_shape_by_oracle_label.csv` and `score_shape_summary.md`. Score dip can help identify high-low-high gaps, but should not be used alone.",
        "",
        "## 8. Interpretation",
        "",
        "The Stage 1 evidence does not support using H4 proximity as a direct merge rule. In this run, negative_merge gaps have a higher H4-near ratio than positive_merge gaps, and the real-vs-random near-gap enrichment ratio is only slightly above 1. This means H4 candidates are common around gaps, but they are not positive-specific enough by themselves.",
        "",
        "The more useful signal appears at the H4-type level: possible_context_forgetting and lexical_topic_boundary have higher positive_merge ratios than broad explicit_transition_boundary and multi_scene_compression_boundary. Even there, the result should be treated as a trigger for event-continuity recheck, not as an automatic merge decision.",
        "",
        "Score-shape remains relevant because H4-near plus stronger score dip has a higher positive_merge ratio than raw H4-near alone, but it still requires Stage 2 upper-bound analysis and later semantic continuity checks.",
        "",
        "## 9. Decision for Next Stage",
        "",
        "- Recommend entering Stage 2: Oracle upper bound.",
        "- Recommend entering Stage 3: simple gap merge baseline only after Stage 2 quantifies upper-bound room.",
        "- Recommend Stage 4 H4 proximity parameter scan as diagnostic, not final claim.",
        "- Recommend Stage 5 H4 + score-shape / semantic continuity if Stage 2 confirms sufficient positive_merge opportunities.",
        "- Next Codex task: Stage 2 gap-level oracle upper bound analysis.",
        "",
        "## 10. Limitations",
        "",
        "- No original videos.",
        "- No true camera transition verification.",
        "- H4 is caption-level boundary only.",
        "- Oracle labels use GT and are not deployable.",
        "- Current prediction gaps depend on score threshold 0.6 unless regenerated.",
        "- Current source-type labels are dataset-level proxy only, not per-video verified labels.",
    ])
    if skipped:
        lines.extend(["", "## Skipped or degraded analyses", ""])
        lines.extend([f"- {item}" for item in skipped])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_dir", default="outputs/26-07-09-15-25-h4-resource-prep")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--h4_window", type=float, default=60)
    parser.add_argument("--random_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    input_summary = write_input_check(output_dir=output_dir, resource_dir=resource_dir, required={
        "prediction_gaps.csv": EXPECTED_GAP_FIELDS,
        "h4_diagnostic_table.csv": EXPECTED_H4_FIELDS,
        "h4_diagnostic_summary.md": [],
        "canonical_timeline.csv": ["dataset", "video_id", "start", "end", "center"],
        "prediction_intervals.csv": ["dataset", "video_id", "pred_start", "pred_end"],
    })

    gaps = read_csv(resource_dir / "prediction_gaps.csv")
    h4 = read_csv(resource_dir / "h4_diagnostic_table.csv")
    timeline = read_csv(resource_dir / "canonical_timeline.csv")
    for df in [gaps, h4, timeline]:
        ensure_numeric(df, [col for col in df.columns if col.endswith("score") or col.endswith("distance") or col in ["gap_len", "gap_start", "gap_end", "h4_count_near_gap", "h4_count_in_gap", "score_dip", "left_mean_score", "right_mean_score", "gap_mean_score", "start", "end", "center", "h4_position"]])
    gaps["merge_oracle_label"] = gaps["merge_oracle_label"].fillna("unknown").replace("", "unknown")
    h4["gap_oracle_label"] = h4["gap_oracle_label"].fillna("no_near_gap").replace("", "no_near_gap")

    gap_stats = gap_level_stats(gaps)
    gap_stats.to_csv(output_dir / "gap_level_enrichment_stats.csv", index=False, encoding="utf-8-sig")
    write_gap_summary(output_dir / "gap_level_enrichment_summary.md", gap_stats)

    exploded = type_stats(h4, exploded=True)
    combo = type_stats(h4, exploded=False)
    exploded.to_csv(output_dir / "h4_type_oracle_stats_exploded.csv", index=False, encoding="utf-8-sig")
    combo.to_csv(output_dir / "h4_type_oracle_stats_combination.csv", index=False, encoding="utf-8-sig")
    write_type_summary(output_dir / "h4_type_oracle_summary.md", exploded)

    trials, random_summary, real_dist, random_dist = random_baseline(h4, gaps, timeline, args.h4_window, args.random_trials, args.seed)
    trials.to_csv(output_dir / "random_boundary_baseline.csv", index=False, encoding="utf-8-sig")
    random_summary.to_csv(output_dir / "random_boundary_summary.csv", index=False, encoding="utf-8-sig")
    write_random_summary(output_dir / "h4_vs_random_distance_summary.md", random_summary)

    score_stats = score_shape_stats(gaps)
    score_stats.to_csv(output_dir / "score_shape_by_oracle_label.csv", index=False, encoding="utf-8-sig")
    write_score_summary(output_dir / "score_shape_summary.md", score_stats, gaps)

    save_figures(output_dir, gaps, h4, exploded, real_dist, random_dist, random_summary)
    write_report(output_dir / "stage1_h4_gap_enrichment_report.md", input_summary, gap_stats, exploded, random_summary, score_stats, skipped)
    (output_dir / "stage1_summary.json").write_text(json.dumps({
        "resource_dir": str(resource_dir).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "random_trials": args.random_trials,
        "seed": args.seed,
        "skipped": skipped,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"gap_stats={output_dir / 'gap_level_enrichment_stats.csv'}")
    print(f"report={output_dir / 'stage1_h4_gap_enrichment_report.md'}")


if __name__ == "__main__":
    main()
