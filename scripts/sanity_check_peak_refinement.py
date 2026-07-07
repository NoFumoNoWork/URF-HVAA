import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.peak_refinement import (  # noqa: E402
    detect_peaks,
    estimate_baseline,
    expand_peak_intervals,
    rescue_peak_intervals,
    split_merged_intervals_by_peak_gaps,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_single_spike() -> None:
    scores = [0.1, 0.1, 0.9, 0.1]
    baseline = estimate_baseline(scores, window=3)
    peaks = detect_peaks(scores, baseline=baseline, min_width=0.5, min_distance=1, mad_k=1.0)
    peak_intervals = expand_peak_intervals(scores, baseline, peaks, min_len=1)
    rescued = rescue_peak_intervals(peak_intervals, [], [], min_prominence=0.1, min_area=0.1)
    assert_true(len(peaks) == 1, "single spike should produce one peak")
    assert_true(rescued and rescued[0]["confidence_hint"] == "low_isolated_sharp_spike", "single narrow spike should be marked isolated/low confidence")


def test_wide_peak() -> None:
    scores = [0.1, 0.1, 0.6, 0.8, 0.8, 0.7, 0.1, 0.1]
    baseline = estimate_baseline(scores, window=3, method="quantile", quantile=0.3)
    peaks = detect_peaks(scores, baseline=baseline, min_width=1, min_distance=1, mad_k=1.0)
    peak_intervals = expand_peak_intervals(scores, baseline, peaks, min_len=3)
    assert_true(peaks, "wide high-score region should produce a peak")
    assert_true(any(item["end"] - item["start"] >= 3 for item in peak_intervals), "wide peak should expand to interval")


def test_long_low_gap_split() -> None:
    scores = [0.8] * 8 + [0.05] * 8 + [0.85] * 8
    baseline = [0.1] * len(scores)
    peaks = [
        {"peak_index": 3, "prominence": 0.7, "area_residual": 4.0},
        {"peak_index": 19, "prominence": 0.75, "area_residual": 4.0},
    ]
    merged = [{"start": 0, "end": 24}]
    micro = [{"start": 0, "end": 8}, {"start": 16, "end": 24}]
    refined, diagnostics = split_merged_intervals_by_peak_gaps(merged, micro, peaks, scores, baseline, min_gap_len=5, low_score_quantile=0.5)
    assert_true(len(diagnostics) == 1, "long low residual gap should split")
    assert_true(len(refined) == 2, "split should create two refined intervals")


def test_short_or_peak_gap_no_split() -> None:
    scores = [0.8] * 8 + [0.7, 0.75] + [0.85] * 8
    baseline = [0.1] * len(scores)
    peaks = [{"peak_index": 8, "prominence": 0.6, "area_residual": 2.0}]
    merged = [{"start": 0, "end": 18}]
    micro = [{"start": 0, "end": 8}, {"start": 10, "end": 18}]
    refined, diagnostics = split_merged_intervals_by_peak_gaps(merged, micro, peaks, scores, baseline, min_gap_len=5, low_score_quantile=0.5)
    assert_true(len(diagnostics) == 0, "short gap or gap with peak should not split")
    assert_true(len(refined) == 1, "no split should preserve merged interval")


def test_empty_and_zero_scores() -> None:
    assert_true(len(estimate_baseline([])) == 0, "empty baseline should be empty")
    assert_true(detect_peaks([], baseline=[]) == [], "empty scores should produce no peaks")
    scores = [0, 0, 0, 0]
    baseline = estimate_baseline(scores, window=3)
    assert_true(detect_peaks(scores, baseline=baseline, min_width=1, min_distance=1) == [], "all-zero scores should produce no peaks")


def main() -> None:
    tests = [
        test_single_spike,
        test_wide_peak,
        test_long_low_gap_split,
        test_short_or_peak_gap_no_split,
        test_empty_and_zero_scores,
    ]
    for test in tests:
        test()
    print(f"peak refinement sanity ok {len(tests)}")


if __name__ == "__main__":
    main()
