import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Experiment:
    timestamp: str
    name: str
    report: str
    extra_reports: list[str]
    programs: list[str]
    outputs: list[str]


EXPERIMENTS = [
    Experiment(
        timestamp="26-07-05-17-46",
        name="multi-anomaly-miss",
        report="reports/multi_anomaly_miss_test_report.md",
        extra_reports=[],
        programs=["scripts/analyze_multi_anomaly_misses.py"],
        outputs=[
            "outputs/multi_anomaly_miss_analysis.json",
            "outputs/multi_anomaly_miss_analysis_verify.json",
        ],
    ),
    Experiment(
        timestamp="26-07-05-21-46",
        name="long-video-diagnosis",
        report="reports/long_video_multi_anomaly_diagnosis.md",
        extra_reports=[
            "reports/data_inventory.md",
            "reports/anomaly_event_index_summary.md",
            "reports/missed_interval_score_analysis.md",
            "reports/topk_coverage_report.md",
            "reports/multiscale_coverage_report.md",
        ],
        programs=[
            "scripts/anomaly_utils.py",
            "scripts/inventory_available_data.py",
            "scripts/build_anomaly_event_index.py",
            "scripts/plot_anomaly_timeline.py",
            "scripts/analyze_missed_interval_scores.py",
            "scripts/generate_topk_intervals.py",
            "scripts/evaluate_topk_coverage.py",
            "scripts/generate_multiscale_intervals.py",
            "src/topk_score_filter.py",
        ],
        outputs=[
            "outputs/data_inventory.json",
            "outputs/anomaly_event_index.json",
            "outputs/anomaly_event_index.csv",
            "outputs/missed_interval_score_analysis.json",
            "outputs/missed_interval_score_analysis.csv",
            "outputs/topk_coverage_results.json",
            "outputs/topk_coverage_curve.png",
            "outputs/topk_intervals",
            "outputs/multiscale_intervals.json",
            "outputs/multiscale_coverage_curve.png",
            "outputs/timeline_plots",
        ],
    ),
    Experiment(
        timestamp="26-07-05-23-42",
        name="wmax-replacement-baseline",
        report="reports/wmax_replacement_baseline_experiment.md",
        extra_reports=[],
        programs=[
            "src/baseline_interval_filters.py",
            "scripts/evaluate_interval_baselines.py",
        ],
        outputs=["outputs/wmax_replacement_baselines"],
    ),
    Experiment(
        timestamp="26-07-06-00-21",
        name="adaptive-interval-selection",
        report="reports/adaptive_interval_selection_report.md",
        extra_reports=[],
        programs=[
            "src/adaptive_interval_selection.py",
            "scripts/generate_adaptive_intervals.py",
            "scripts/evaluate_adaptive_intervals.py",
        ],
        outputs=[
            "outputs/adaptive_intervals",
            "outputs/adaptive_interval_tradeoff.png",
            "outputs/adaptive_timeline_plots",
        ],
    ),
    Experiment(
        timestamp="26-07-06-00-37",
        name="adaptive-param-grid",
        report="reports/adaptive_param_grid_report.md",
        extra_reports=[],
        programs=[
            "src/adaptive_interval_selection.py",
            "scripts/sweep_adaptive_params.py",
        ],
        outputs=[
            "outputs/adaptive_param_grid",
            "outputs/adaptive_param_grid_tradeoff.png",
        ],
    ),
    Experiment(
        timestamp="26-07-06-08-39",
        name="hierarchical-intervals",
        report="reports/hierarchical_interval_report.md",
        extra_reports=[],
        programs=[
            "src/adaptive_interval_selection.py",
            "scripts/anomaly_utils.py",
            "scripts/generate_hierarchical_intervals.py",
            "scripts/evaluate_hierarchical_intervals.py",
            "scripts/plot_hierarchical_timeline.py",
        ],
        outputs=["outputs/hierarchical_intervals"],
    ),
    Experiment(
        timestamp="26-07-07-01-08",
        name="peak-aware-refinement",
        report="outputs/26-07-07-01-08-peak-aware-refinement/peak-aware-refinement_report.md",
        extra_reports=[],
        programs=[
            "src/peak_refinement.py",
            "scripts/run_peak_refinement.py",
            "scripts/sanity_check_peak_refinement.py",
        ],
        outputs=["outputs/26-07-07-01-08-peak-aware-refinement/outputs/peak_refinement"],
    ),
]


def copy_path(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"missing: {src}"
    if src.resolve() == dst.resolve():
        return f"already in place: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return f"copied: {src} -> {dst}"


def archive_experiment(root: Path, exp: Experiment) -> list[str]:
    folder = root / "outputs" / f"{exp.timestamp}-{exp.name}"
    logs = []
    logs.append(copy_path(root / exp.report, folder / f"{exp.name}_report.md"))

    for report in exp.extra_reports:
        logs.append(copy_path(root / report, folder / "reports" / Path(report).name))

    for program in exp.programs:
        logs.append(copy_path(root / program, folder / "programs" / program))

    for output in exp.outputs:
        src = root / output
        if src.is_dir():
            dst = folder / "outputs" / src.name
        else:
            dst = folder / "outputs" / Path(output).name
        logs.append(copy_path(src, dst))

    manifest = [
        f"# {exp.name}",
        "",
        f"- archive_folder: `outputs/{exp.timestamp}-{exp.name}`",
        f"- primary_report: `{exp.name}_report.md`",
        "",
        "## Contents",
        "",
        "- `programs/`: copied scripts/modules needed to reproduce or inspect this experiment.",
        "- `outputs/`: copied generated data, plots, and result files.",
        "- `reports/`: supporting reports when the experiment is a multi-stage bundle.",
        "",
        "## Copy Log",
        "",
    ]
    manifest.extend([f"- {line}" for line in logs])
    (folder / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    return logs


def write_index(root: Path, archived: list[Experiment]) -> None:
    lines = [
        "# Experiment Artifact Layout",
        "",
        "New experiment artifacts should be archived under:",
        "",
        "`outputs/yy-mm-dd-hh-min-test-name/`",
        "",
        "Each archive folder should contain:",
        "",
        "- `<test-name>_report.md`",
        "- `programs/`",
        "- `outputs/`",
        "- `MANIFEST.md`",
        "",
        "## Current Archives",
        "",
    ]
    for exp in archived:
        lines.append(f"- `outputs/{exp.timestamp}-{exp.name}/`")
    (root / "reports" / "ARTIFACT_LAYOUT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--only", help="Archive only the named experiment.")
    args = parser.parse_args()
    root = args.root.resolve()

    selected = [exp for exp in EXPERIMENTS if args.only in (None, exp.name)]
    if args.only and not selected:
        raise SystemExit(f"unknown experiment: {args.only}")

    for exp in selected:
        archive_experiment(root, exp)
    write_index(root, EXPERIMENTS)
    print(f"Archived {len(selected)} experiments under outputs/")


if __name__ == "__main__":
    main()
