import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-09-23-11-h4-report-integration")
DEFAULT_FP_REDUCTION_DIR = Path("outputs/26-07-09-16-49-h4-fp-reduction-test")
DEFAULT_VIS_DIR = Path("outputs/26-07-09-16-18-low-fp-h4-visualization")
DEFAULT_STAGE1_DIR = Path("outputs/26-07-09-15-48-h4-gap-enrichment")
DEFAULT_RESOURCE_DIR = Path("outputs/26-07-09-15-25-h4-resource-prep")
DEFAULT_FINAL_REPORT = Path("papers/SMILES_2026.md")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4) -> str:
    value = as_float(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def draw_workflow(path_png: Path, path_svg: Path, path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    main = [
        ("Score /\nanomaly score", 0.5, 3.25),
        ("Candidate interval\ngeneration", 2.6, 3.25),
        ("Multi-evidence\nfeature extraction", 4.8, 3.25),
        ("Precision-first\nevidence fusion", 7.0, 3.25),
        ("Low-FP\nfiltering", 9.2, 3.25),
        ("Valley cut /\nnegative evidence", 11.2, 3.25),
        ("Final prediction\nintervals", 13.7, 3.25),
    ]
    branch = [
        ("VLM caption", 0.8, 5.75),
        ("H4 caption-level\nboundary screening", 3.2, 5.75),
        ("H4 candidate /\ngap-level signal", 6.0, 5.75),
        ("H4 added-value\ntest", 8.8, 5.75),
        ("Not selected as\nfinal low-FP module", 12.0, 5.75),
    ]

    def box(text, x, y, w=1.65, h=0.82, fc="#E9F2FB", ec="#4E79A7", ls="-", alpha=1.0):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.055,rounding_size=0.08",
            linewidth=1.3,
            edgecolor=ec,
            facecolor=fc,
            linestyle=ls,
            alpha=alpha,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
        return patch

    def arrow(x1, y1, x2, y2, color="#4E79A7", ls="-", lw=1.5):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color, linestyle=ls))

    for i, (text, x, y) in enumerate(main):
        box(text, x, y)
        if i:
            prev = main[i - 1]
            arrow(prev[1] + 1.65, prev[2] + 0.41, x, y + 0.41)

    for i, (text, x, y) in enumerate(branch):
        box(text, x, y, w=2.0 if i in {1, 2, 3, 4} else 1.55, fc="#F3F3F3", ec="#777777", ls="--", alpha=0.96)
        if i:
            prev = branch[i - 1]
            prev_w = 2.0 if i - 1 in {1, 2, 3, 4} else 1.55
            arrow(prev[1] + prev_w, prev[2] + 0.41, x, y + 0.41, color="#777777", ls="--", lw=1.3)

    ax.text(8.0, 4.82, "H4 branch is evaluated separately against score-only / valley-only controls", fontsize=9, color="#666666", ha="center", va="center")
    ax.text(
        8.0,
        1.25,
        "H4 = caption-level boundary candidate; not verified camera transition; used for added-value test only; not included in final low-FP configuration.",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E7", edgecolor="#D4A72C", linewidth=1.0),
    )
    ax.text(8.0, 6.9, "Evidence-Stratified Low-FP Interval Refinement with H4 Side-Branch", ha="center", va="center", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    fig.savefig(path_pdf)
    plt.close(fig)


def redraw_pareto(rows: list[dict], output_path: Path) -> None:
    families = {
        "M0_baseline": ("#111111", "Baseline"),
        "M1_score_only_stricter_threshold": ("#4E79A7", "Score-only"),
        "M2_valley_cut_only_stronger": ("#59A14F", "Valley-only"),
        "M3_simple_gap_score_shape_merge": ("#9C755F", "Simple merge"),
        "M4_h4_suppression_cut": ("#E15759", "H4 suppression"),
        "M5_h4_gated_valley_cut": ("#B07AA1", "H4-gated valley"),
        "M6_h4_veto_merge": ("#F28E2B", "H4 veto merge"),
    }
    plt.figure(figsize=(9.4, 6.2))
    seen = set()
    for row in rows:
        family = row["method_family"]
        color, label = families.get(family, ("#888888", family))
        label = label if family not in seen else None
        seen.add(family)
        size = 90 if family == "M0_baseline" else 34
        alpha = 0.9 if family == "M0_baseline" or family.startswith("M4") or family.startswith("M5") or family.startswith("M6") else 0.72
        plt.scatter(as_float(row["TP_retention"]), as_float(row["FP"]), s=size, color=color, alpha=alpha, label=label)
    labels = [
        "M0_baseline_low_fp",
        "M4_h4_cut_all_h4_score_0.45",
        "M2_valley_only_low_0.22_len_32",
        "M2_valley_only_low_0.18_len_48",
        "M1_score_only_mean_ge_0.36",
    ]
    for row in rows:
        if row["method_name"] in labels:
            plt.annotate(row["method_name"], (as_float(row["TP_retention"]), as_float(row["FP"])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.axvline(0.98, color="#777777", linestyle="--", linewidth=0.9)
    plt.text(0.981, max(as_float(r["FP"]) for r in rows) * 0.998, "TP retention = 0.98", fontsize=8, color="#666666")
    plt.xlabel("TP retention vs low-FP baseline")
    plt.ylabel("FP duration")
    plt.title("H4 added-value test: TP-preserving FP reduction")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def method_row(rows: list[dict], name: str) -> dict:
    for row in rows:
        if row["method_name"] == name:
            return row
    raise KeyError(name)


def make_tables(rows: list[dict], output_dir: Path) -> tuple[list[dict], list[dict]]:
    baseline = method_row(rows, "M0_baseline_low_fp")
    best_h4 = method_row(rows, "M4_h4_cut_all_h4_score_0.45")
    best_valley = method_row(rows, "M2_valley_only_low_0.22_len_32")
    core_rows = [
        {
            "Method": "Low-FP baseline",
            "H4 used": "No",
            "TP retention": "1.0000",
            "FP": baseline["FP"],
            "FP reduction": "0",
            "Precision": fmt(baseline["precision"]),
            "Recall": fmt(baseline["recall"]),
            "F1": fmt(baseline["F1"]),
        },
        {
            "Method": "Best H4 suppression",
            "H4 used": "Yes",
            "TP retention": fmt(best_h4["TP_retention"]),
            "FP": best_h4["FP"],
            "FP reduction": best_h4["FP_reduction"],
            "Precision": fmt(best_h4["precision"]),
            "Recall": fmt(best_h4["recall"]),
            "F1": fmt(best_h4["F1"]),
        },
        {
            "Method": "Best valley-only control",
            "H4 used": "No",
            "TP retention": fmt(best_valley["TP_retention"]),
            "FP": best_valley["FP"],
            "FP reduction": best_valley["FP_reduction"],
            "Precision": fmt(best_valley["precision"]),
            "Recall": fmt(best_valley["recall"]),
            "F1": fmt(best_valley["F1"]),
        },
    ]
    fields = ["Method", "H4 used", "TP retention", "FP", "FP reduction", "Precision", "Recall", "F1"]
    write_csv(output_dir / "h4_added_value_core_table.csv", core_rows, fields)

    type_specs = [
        ("all H4", "M4_h4_cut_all_h4_score_0.45"),
        ("possible_context_forgetting + lexical_topic_boundary", "M4_h4_cut_possible_context_forgetting+lexical_topic_boundary_score_0.45"),
        ("explicit + multi-scene", "M4_h4_cut_explicit_transition_boundary+multi_scene_compression_boundary_score_0.45"),
        ("lexical_topic_boundary", "M4_h4_cut_lexical_topic_boundary_score_0.45"),
    ]
    type_rows = []
    for label, method in type_specs:
        row = method_row(rows, method)
        type_rows.append(
            {
                "H4 type filter": label,
                "TP retention": fmt(row["TP_retention"]),
                "FP reduction": row["FP_reduction"],
                "Precision": fmt(row["precision"]),
                "Recall": fmt(row["recall"]),
            }
        )
    write_csv(output_dir / "h4_type_added_value_table.csv", type_rows, ["H4 type filter", "TP retention", "FP reduction", "Precision", "Recall"])
    return core_rows, type_rows


def md_table(rows: list[dict], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] + ["---:" for _ in fields[1:]]) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def build_report(output_dir: Path, resources: list[dict], core_rows: list[dict], type_rows: list[dict], decision: dict) -> None:
    lines = [
        "# H4 Report Integration Package",
        "",
        "## 1. Purpose",
        "",
        "This package collects the minimum report-ready materials needed to integrate the H4 analysis into the formal report for Evidence-Stratified Low-FP Interval Refinement for Zero-Shot Video Anomaly Detection. H4 is not positioned as a final method module. It is a caption-level boundary feature from VLM captions, evaluated through an added-value test that asks whether it can reduce false positives while preserving TP/recall.",
        "",
        "## 2. Input resources checked",
        "",
        "| Resource | Status | Use |",
        "| --- | --- | --- |",
    ]
    for item in resources:
        lines.append(f"| `{item['path']}` | {item['status']} | {item['use']} |")
    lines.extend(
        [
            "",
            "## 3. How H4 should be positioned in the final report",
            "",
            "- H4 comes from VLM captions, not from raw visual shot-boundary annotation.",
            "- H4 should be described as a caption-level boundary candidate or context-boundary proxy.",
            "- H4 is not a verified camera transition and must not be described as detecting true scene switches.",
            "- H4 is not included in the final low-FP configuration.",
            "- H4 is used to test whether caption-level boundary information adds FP-reduction value beyond score-only or valley-only controls.",
            "- The result is a negative/boundary result: current H4 reduces some FP, but it does not provide independent added value over non-H4 controls under TP-preserving constraints.",
            "- H4 is still useful as motivation for future event-continuity modeling, especially if future VLM outputs explicitly label same-event continuation, viewpoint shift, related consequence, or new event.",
            "",
            "## 4. Required changes to the final report",
            "",
            "### 4.1 Introduction",
            "",
            "Paste-ready paragraph:",
            "",
            "> VLM captions may contain context-boundary cues that are not explicit in the scalar anomaly score. We therefore examined an auxiliary H4 signal, defined as caption-level boundary candidates that may indicate topic shifts, multi-scene compression, context forgetting, or explicit transition language. H4 is not treated as a camera-transition detector and is not assumed to mark true visual shot boundaries. Instead, we use it as a diagnostic caption-level feature and ask whether it provides added value for the main objective of this work: reducing false-positive interval duration while preserving true-positive coverage under a precision-first low-FP operating point.",
            "",
            "### 4.2 Method workflow figure",
            "",
            "The workflow figure should keep the main method as evidence-stratified low-FP refinement: score -> candidate interval generation -> multi-evidence feature extraction -> precision-first evidence fusion -> low-FP filtering -> valley cut / negative evidence refinement -> final prediction intervals. H4 should appear only as a dashed side branch from VLM caption to H4 screening, H4 candidate/gap signal, and added-value test. The branch should not connect into final prediction intervals.",
            "",
            "- Figure path: `figures/fig_workflow_with_h4_branch.png`.",
            "",
            "### 4.3 Method section",
            "",
            "Paste-ready subsection:",
            "",
            "#### 2.3 Caption-level H4 boundary screening and added-value test",
            "",
            "> In addition to score-derived evidence, we examined a caption-level H4 boundary signal extracted from VLM captions. H4 candidates include lexical topic boundaries, explicit transition boundaries, multi-scene compression boundaries, and possible context-forgetting boundaries. These candidates are not interpreted as verified camera transitions or true scene switches; they only indicate that the caption stream may contain a context boundary or event-continuity disruption. We evaluated H4 in several interval-level roles, including merge diagnostics, suppression/cutting of suspicious long intervals, H4-gated valley refinement, and vetoing naive gap merges. The success criterion was deliberately precision-oriented: H4 is useful only if it reduces false-positive duration while preserving baseline TP/recall. We therefore compared H4 variants against score-only stricter filtering and valley-only stronger refinement. If H4 does not outperform these non-H4 controls under comparable TP-retention constraints, it is not included in the final low-FP configuration.",
            "",
            "### 4.4 Result and Discussion section",
            "",
            "Paste-ready subsection:",
            "",
            "#### 3.5 Caption-level H4 boundary signal does not provide independent FP-reduction gain",
            "",
            "> We tested whether the caption-level H4 boundary signal provides added value beyond the low-FP interval refinement pipeline. The baseline low-FP system achieved precision 0.5520, recall 0.6874, and F1 0.6123 with FP duration 347264. The best H4 suppression variant reduced FP duration to 329953 while retaining 98.63% of baseline TP, yielding precision 0.5612 and F1 0.6141. However, a non-H4 valley-only stronger control achieved lower FP duration, 313112, under a comparable TP-retention constraint of 98.10%, with higher precision 0.5728 and F1 0.6194. Thus, although H4 can remove some false-positive duration, it is dominated by a score/valley-only control for the TP-preserving FP-reduction objective. We therefore treat H4 as a diagnostic caption-level boundary proxy rather than as a selected final module. This negative result suggests that current H4 candidates do not reliably determine whether the two sides of a caption boundary belong to the same event, a viewpoint change, a related consequence, or a genuinely new event.",
            "",
            "Core table:",
            "",
            *md_table(core_rows, ["Method", "H4 used", "TP retention", "FP", "FP reduction", "Precision", "Recall", "F1"]),
            "",
            "### 4.5 Limitation section",
            "",
            "Paste-ready paragraph:",
            "",
            "> The H4 analysis is limited by its caption-level nature. A caption boundary indicates that the textual description changes, but it does not prove a camera transition, visual shot boundary, or event discontinuity. Without raw video and event-continuity labels, H4 cannot distinguish same-event continuation from a new viewpoint, related consequence, narrative jump, caption phrasing change, or a truly new event. In the current experiments, H4 did not provide independent FP-reduction gain beyond score-only and valley-only controls, so it should be reported as a diagnostic analysis and future direction rather than as an effective module in the final pipeline.",
            "",
            "### 4.6 Conclusion",
            "",
            "Paste-ready paragraph:",
            "",
            "> We additionally evaluated H4 caption-level boundary candidates as a possible source of structural evidence. Under TP-preserving FP-reduction criteria, H4 did not outperform non-H4 score/valley controls and was therefore not integrated into the final low-FP configuration. Future work could revisit H1-H4 style caption features if raw videos, VLM prompts, VAL/VAU prompts, full inference traces, and explicit event-continuity labels are available. For H4 in particular, a useful next step would be to ask the VLM to explicitly output event-continuity judgments rather than relying on caption-boundary proxies.",
            "",
            "## 5. Figures and tables prepared",
            "",
            "- `figures/fig_workflow_with_h4_branch.png`: updated workflow with H4 as a dashed side branch, not a final module.",
            "- `h4_added_value_core_table.csv`: baseline vs best H4 suppression vs best valley-only control.",
            "- `figures/fig_h4_pareto_fp_vs_tp_retention.png`: TP-retention vs FP-duration comparison across H4 and non-H4 controls.",
            "- `h4_type_added_value_table.csv`: compact H4 subtype comparison.",
            "- `figures/fig_low_fp_h4_overlay_case.png`: copied diagnostic overlay case; it illustrates H4 as boundary/gap overlay, not as proof of final effectiveness.",
            "",
            "## 6. Recommended final report structure",
            "",
            "3.1 原文标注和评分之间的不一致性  ",
            "3.2 基本提取能力与基线比较  ",
            "3.3 消融实验：各组件贡献  ",
            "3.4 参数扫描：关键参数的重要程度  ",
            "3.5 Caption-level H4 边界信号的附加价值检验  ",
            "3.6 能力边界与局限",
            "",
            "## 7. Decision on additional experiments",
            "",
            "- Do not continue large-scale H4 parameter scanning for the current report.",
            "- Do not add H4 to the final main method.",
            "- Do not directly quote the original paper's multi-dataset results as this report's own results.",
            "- The original multi-dataset evaluation can be cited only as background; this report should present only datasets for which intermediate artifacts and post-processing evaluations are actually available.",
            "- To extend H4 rigorously, future work needs raw videos, VLM prompts, VAL/VAU prompts, the complete inference chain, and event-continuity labels.",
            "",
            "## 8. Final checklist",
            "",
            "- [ ] Introduction 已加入 H4 动机",
            "- [ ] workflow 图已加入 H4 旁路",
            "- [ ] Method 已加入 H4 screening / added-value test",
            "- [ ] Result 已加入 H4 negative result",
            "- [ ] Limitation 已解释 H4 边界",
            "- [ ] Conclusion 已加入 event-continuity future work",
            "- [ ] H4 未被误写成最终有效模块",
            "- [ ] H4 未被误写成真实 camera transition",
            "",
            "## Decision JSON summary",
            "",
            "```json",
            json.dumps(decision, indent=2, ensure_ascii=False),
            "```",
            "",
            "## H4 type compact table",
            "",
            *md_table(type_rows, ["H4 type filter", "TP retention", "FP reduction", "Precision", "Recall"]),
        ]
    )
    (output_dir / "stage_h4_report_integration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fp_reduction_dir", type=Path, default=DEFAULT_FP_REDUCTION_DIR)
    parser.add_argument("--visualization_dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--stage1_dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--resource_dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--final_report", type=Path, default=DEFAULT_FINAL_REPORT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    resources = [
        {"path": args.final_report.as_posix(), "status": "found" if args.final_report.exists() else "missing", "use": "current formal report draft"},
        {"path": (args.fp_reduction_dir / "stage_h4_fp_reduction_report.md").as_posix(), "status": "found" if (args.fp_reduction_dir / "stage_h4_fp_reduction_report.md").exists() else "missing", "use": "H4 added-value conclusion"},
        {"path": (args.fp_reduction_dir / "method_metrics.csv").as_posix(), "status": "found" if (args.fp_reduction_dir / "method_metrics.csv").exists() else "missing", "use": "method metrics for core/Pareto tables"},
        {"path": (args.fp_reduction_dir / "best_methods_by_constraint.csv").as_posix(), "status": "found" if (args.fp_reduction_dir / "best_methods_by_constraint.csv").exists() else "missing", "use": "TP-retention constrained best methods"},
        {"path": (args.fp_reduction_dir / "h4_added_value_decision.json").as_posix(), "status": "found" if (args.fp_reduction_dir / "h4_added_value_decision.json").exists() else "missing", "use": "machine-readable H4 decision"},
        {"path": (args.fp_reduction_dir / "figures/pareto_fp_vs_tp_retention.png").as_posix(), "status": "found" if (args.fp_reduction_dir / "figures/pareto_fp_vs_tp_retention.png").exists() else "missing", "use": "prior Pareto figure"},
        {"path": (args.visualization_dir / "fig_low_fp_case_visualization.png").as_posix(), "status": "found" if (args.visualization_dir / "fig_low_fp_case_visualization.png").exists() else "missing", "use": "low-FP + H4 diagnostic overlay case"},
        {"path": (args.resource_dir / "prediction_gaps.csv").as_posix(), "status": "found" if (args.resource_dir / "prediction_gaps.csv").exists() else "missing", "use": "H4 gap resources"},
        {"path": (args.resource_dir / "h4_diagnostic_table.csv").as_posix(), "status": "found" if (args.resource_dir / "h4_diagnostic_table.csv").exists() else "missing", "use": "H4 candidates/types"},
        {"path": (args.stage1_dir / "stage1_h4_gap_enrichment_report.md").as_posix(), "status": "found" if (args.stage1_dir / "stage1_h4_gap_enrichment_report.md").exists() else "missing", "use": "Stage 1 context/limitations"},
        {"path": (args.stage1_dir / "h4_type_oracle_summary.md").as_posix(), "status": "found" if (args.stage1_dir / "h4_type_oracle_summary.md").exists() else "missing", "use": "H4 type context"},
        {"path": (args.stage1_dir / "h4_vs_random_distance_summary.md").as_posix(), "status": "found" if (args.stage1_dir / "h4_vs_random_distance_summary.md").exists() else "missing", "use": "random-boundary context"},
    ]
    missing_required = [item for item in resources[:5] if item["status"] != "found"]
    if missing_required:
        raise FileNotFoundError("Missing required resources: " + ", ".join(item["path"] for item in missing_required))

    rows = read_csv(args.fp_reduction_dir / "method_metrics.csv")
    decision = json.loads((args.fp_reduction_dir / "h4_added_value_decision.json").read_text(encoding="utf-8"))
    core_rows, type_rows = make_tables(rows, args.output_dir)
    draw_workflow(figures / "fig_workflow_with_h4_branch.png", figures / "fig_workflow_with_h4_branch.svg", figures / "fig_workflow_with_h4_branch.pdf")
    redraw_pareto(rows, figures / "fig_h4_pareto_fp_vs_tp_retention.png")
    copy_if_exists(args.visualization_dir / "fig_low_fp_case_visualization.png", figures / "fig_low_fp_h4_overlay_case.png")
    copy_if_exists(args.fp_reduction_dir / "figures/precision_recall_tradeoff_h4_vs_baselines.png", figures / "fig_h4_precision_recall_tradeoff.png")
    copy_if_exists(args.fp_reduction_dir / "figures/h4_type_fp_reduction.png", figures / "fig_h4_type_fp_reduction.png")
    build_report(args.output_dir, resources, core_rows, type_rows, decision)
    (args.output_dir / "h4_report_integration_manifest.json").write_text(
        json.dumps(
            {
                "output_dir": args.output_dir.as_posix(),
                "main_report": (args.output_dir / "stage_h4_report_integration.md").as_posix(),
                "figures": sorted(path.name for path in figures.glob("*")),
                "tables": ["h4_added_value_core_table.csv", "h4_type_added_value_table.csv"],
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    program_dir = args.output_dir / "programs" / "scripts"
    program_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program_dir / Path(__file__).name)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir.as_posix(),
                "main_report": (args.output_dir / "stage_h4_report_integration.md").as_posix(),
                "figures": sorted(path.name for path in figures.glob("*")),
                "h4_positioning": "caption-level diagnostic added-value test; not final module",
                "more_experiments_needed": False,
                "next_step": "Merge paste-ready paragraphs from stage_h4_report_integration.md into the formal report.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
