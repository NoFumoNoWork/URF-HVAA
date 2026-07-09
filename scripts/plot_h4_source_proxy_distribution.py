import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


FIELDS = [
    "dataset",
    "source_proxy",
    "num_videos",
    "total_candidates",
    "strong_candidates",
    "possible_h4_candidates",
    "candidates_per_video",
    "strong_per_video",
    "possible_h4_per_video",
]

SOURCE_PROXY = {
    "XD-Violence": "edited/movie/web-video proxy",
    "UCF-Crime": "surveillance/crime-video proxy",
    "MSAD": "surveillance-like proxy",
    "UBNormal": "surveillance-like proxy",
}


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=6):
    return round(float(value), digits)


def possible_h4_counts(strong_rows):
    counts = defaultdict(int)
    for row in strong_rows:
        label = str(row.get("preliminary_h4_label", "")).strip()
        if label != "likely_event_onset_not_h4":
            counts[row.get("dataset", "")] += 1
    return counts


def build_summary(dataset_summary_path, strong_candidates_path):
    dataset_rows = read_csv(dataset_summary_path)
    strong_rows = read_csv(strong_candidates_path)
    possible_counts = possible_h4_counts(strong_rows)
    out = []
    for row in dataset_rows:
        dataset = row["dataset"]
        videos = as_float(row.get("num_videos"))
        total = as_float(row.get("num_candidates"))
        strong = as_float(row.get("num_strong"))
        possible = float(possible_counts.get(dataset, 0))
        out.append({
            "dataset": dataset,
            "source_proxy": SOURCE_PROXY.get(dataset, "unknown proxy"),
            "num_videos": int(videos),
            "total_candidates": int(total),
            "strong_candidates": int(strong),
            "possible_h4_candidates": int(possible),
            "candidates_per_video": fmt(total / videos if videos else 0),
            "strong_per_video": fmt(strong / videos if videos else 0),
            "possible_h4_per_video": fmt(possible / videos if videos else 0),
        })
    return out


def aggregate_by_proxy(rows):
    grouped = {}
    for row in rows:
        proxy = row["source_proxy"]
        item = grouped.setdefault(proxy, {
            "source_proxy": proxy,
            "num_videos": 0,
            "total_candidates": 0,
            "strong_candidates": 0,
            "possible_h4_candidates": 0,
        })
        item["num_videos"] += int(row["num_videos"])
        item["total_candidates"] += int(row["total_candidates"])
        item["strong_candidates"] += int(row["strong_candidates"])
        item["possible_h4_candidates"] += int(row["possible_h4_candidates"])
    for item in grouped.values():
        videos = item["num_videos"]
        item["candidates_per_video"] = fmt(item["total_candidates"] / videos if videos else 0)
        item["strong_per_video"] = fmt(item["strong_candidates"] / videos if videos else 0)
        item["possible_h4_per_video"] = fmt(item["possible_h4_candidates"] / videos if videos else 0)
    return list(grouped.values())


def plot_by_proxy(proxy_rows, output_dir):
    proxies = [row["source_proxy"] for row in proxy_rows]
    total = [int(row["total_candidates"]) for row in proxy_rows]
    strong = [int(row["strong_candidates"]) for row in proxy_rows]
    possible = [int(row["possible_h4_candidates"]) for row in proxy_rows]
    x = list(range(len(proxies)))
    width = 0.25
    plt.figure(figsize=(11, 5.5))
    plt.bar([i - width for i in x], total, width=width, label="total candidates")
    plt.bar(x, strong, width=width, label="strong candidates")
    plt.bar([i + width for i in x], possible, width=width, label="possible H4 candidates")
    plt.xticks(x, proxies, rotation=18, ha="right")
    plt.ylabel("Candidate count")
    plt.title("H4 boundary candidates by source proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_candidates_by_source_proxy.png", dpi=180)
    plt.close()


def plot_per_video(rows, output_dir):
    rows = sorted(rows, key=lambda row: float(row["candidates_per_video"]), reverse=True)
    datasets = [row["dataset"] for row in rows]
    values = [float(row["candidates_per_video"]) for row in rows]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(datasets, values)
    plt.ylabel("Candidates per video")
    plt.title("H4 boundary candidate density by dataset")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_h4_candidates_per_video_by_dataset.png", dpi=180)
    plt.close()


def write_report(path, rows, proxy_rows):
    by_total = sorted(rows, key=lambda row: int(row["total_candidates"]), reverse=True)
    by_density = sorted(rows, key=lambda row: float(row["candidates_per_video"]), reverse=True)
    by_strong_density = sorted(rows, key=lambda row: float(row["strong_per_video"]), reverse=True)
    by_possible_density = sorted(rows, key=lambda row: float(row["possible_h4_per_video"]), reverse=True)
    proxy_density = sorted(proxy_rows, key=lambda row: float(row["candidates_per_video"]), reverse=True)
    proxy_strong_density = sorted(proxy_rows, key=lambda row: float(row["strong_per_video"]), reverse=True)
    proxy_possible_density = sorted(proxy_rows, key=lambda row: float(row["possible_h4_per_video"]), reverse=True)
    lines = [
        "# H4 source proxy distribution report",
        "",
        "## Data and proxy definition",
        "",
        "This analysis uses the existing caption-level H4 boundary candidate exports. `dataset_summary.csv` provides dataset-level video and candidate counts, and `h4_strong_candidates.csv` is used to estimate possible H4 strong candidates after excluding rows preliminarily labeled as likely event-onset-not-H4.",
        "",
        "The `source_proxy` field is a coarse proxy based on dataset identity and file/source conventions: XD-Violence is treated as an edited/movie/web-video proxy; UCF-Crime as a surveillance/crime-video proxy; MSAD and UBNormal as surveillance-like proxies. This is not manual per-video source labeling and must not be interpreted as a strict movie-versus-surveillance annotation.",
        "",
        "## Candidate volume",
        "",
        f"- The largest total contributor is {by_total[0]['dataset']} with {by_total[0]['total_candidates']} total candidates.",
        f"- Dataset order by total candidates: {', '.join(f'{row['dataset']}={row['total_candidates']}' for row in by_total)}.",
        "",
        "## Candidate density",
        "",
        f"- By candidates per video, the highest dataset is {by_density[0]['dataset']} with {by_density[0]['candidates_per_video']} candidates/video.",
        f"- Dataset order by candidates/video: {', '.join(f'{row['dataset']}={row['candidates_per_video']}' for row in by_density)}.",
        f"- Proxy order by candidates/video: {', '.join(f'{row['source_proxy']}={row['candidates_per_video']}' for row in proxy_density)}.",
        f"- By strong candidates per video, the highest dataset is {by_strong_density[0]['dataset']} with {by_strong_density[0]['strong_per_video']} strong candidates/video.",
        f"- By possible H4 candidates per video, the highest dataset is {by_possible_density[0]['dataset']} with {by_possible_density[0]['possible_h4_per_video']} possible H4 candidates/video.",
        f"- Proxy order by strong candidates/video: {', '.join(f'{row['source_proxy']}={row['strong_per_video']}' for row in proxy_strong_density)}.",
        f"- Proxy order by possible H4 candidates/video: {', '.join(f'{row['source_proxy']}={row['possible_h4_per_video']}' for row in proxy_possible_density)}.",
        "",
        "## Interpretation",
        "",
        "The results support a cautious caption-level proxy statement, but with an important distinction. XD-Violence, the edited/movie/web-video proxy, contributes the largest total number of H4 boundary candidates. For raw total candidates per video, UCF-Crime is slightly higher than XD-Violence. However, for stronger H4 signals, XD-Violence is clearly highest: it has the highest strong-candidate density and the highest possible-H4-candidate density. Therefore the evidence is better phrased as: edited/movie/web-video proxy data contains denser strong caption-level H4 boundary signals, while raw total-candidate density is not exclusively higher for that proxy.",
        "",
        "However, this is not a visual shot-boundary conclusion. The counts are derived from caption text and heuristic candidate labels, not from original-video shot detection or manual video-source annotation. Caption density can be affected by VLM phrasing, prompt behavior, dataset naming, event type, and video duration. Strict verification would require original videos plus per-video source-type labels and shot-boundary/event-continuity annotations.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(dataset_summary, strong_candidates, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary(dataset_summary, strong_candidates)
    proxy_rows = aggregate_by_proxy(rows)
    write_csv(output_dir / "h4_source_proxy_summary.csv", rows, FIELDS)
    write_csv(output_dir / "h4_source_proxy_aggregate.csv", proxy_rows, ["source_proxy"] + FIELDS[2:])
    plot_by_proxy(proxy_rows, output_dir)
    plot_per_video(rows, output_dir)
    write_report(output_dir / "h4_source_proxy_distribution_report.md", rows, proxy_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summary", default="outputs/26-07-09-00-35-caption_boundary_screen/dataset_summary.csv")
    parser.add_argument("--strong-candidates", default="outputs/26-07-09-00-35-caption_boundary_screen/h4_strong_candidates.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(Path(args.dataset_summary), Path(args.strong_candidates), Path(args.output_dir))


if __name__ == "__main__":
    main()
