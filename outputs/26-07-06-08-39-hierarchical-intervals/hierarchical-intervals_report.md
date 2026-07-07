# Hierarchical Interval Report

## Input

- prediction_file: `outputs\hierarchical_intervals\hierarchical_intervals.json`

## Coverage

- merged event segment miss rate: 17.79%
- merged event mean coverage: 0.735
- micro interval segment miss rate: 18.36%
- micro interval mean coverage: 0.705

## Structure

- mean events per video: 2.087
- mean micro intervals per event: 38.894
- mean gap duration inside event: 76.000
- mean GT intervals covered by each merged event: 1.015
- mean micro intervals inside each GT interval: 28.174

## Questions

- micro proposals capture score spikes: Yes. The direct signal is micro miss rate and micro mean coverage.
- merged events cover GT blocks: Yes. The direct signal is merged miss rate and merged mean coverage.
- average micro intervals per merged event: 38.894.
- over-merging exists: Yes. Events covering multiple GT intervals: 212.
- over-fragmentation exists: Yes. GT intervals with at least 8 overlapping micro intervals: 923.
- cases needing agent same-event/separate-event judgement: 1186.

## Agent Review Cases

- XD-Violence `Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6` item=1: merged_event_spans_multiple_gt_or_large_internal_gaps
- XD-Violence `Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6` item=2: gt_interval_has_many_micro_fragments
- XD-Violence `Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6` item=3: gt_interval_has_many_micro_fragments
- XD-Violence `Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0` item=1: merged_event_spans_multiple_gt_or_large_internal_gaps
- XD-Violence `Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0` item=1: merged_event_spans_multiple_gt_or_large_internal_gaps
- XD-Violence `Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0` item=3: merged_event_spans_multiple_gt_or_large_internal_gaps
- XD-Violence `Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0` item=2: gt_interval_has_many_micro_fragments
- XD-Violence `Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0` item=3: gt_interval_has_many_micro_fragments
- XD-Violence `Black.Hawk.Down.2001__#01-13-59_01-14-49_label_B2-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Black.Hawk.Down.2001__#01-32-40_01-34-00_label_B4-0-0` item=3: merged_event_spans_multiple_gt_or_large_internal_gaps
- XD-Violence `Black.Hawk.Down.2001__#01-32-40_01-34-00_label_B4-0-0` item=2: gt_interval_has_many_micro_fragments
- XD-Violence `Black.Hawk.Down.2001__#01-32-40_01-34-00_label_B4-0-0` item=4: gt_interval_has_many_micro_fragments
- XD-Violence `Black.Hawk.Down.2001__#01-42-58_01-43-58_label_G-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Black.Hawk.Down.2001__#02-00-12_02-01-29_label_B2-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Braveheart.1995__#00-56-30_00-57-20_label_B1-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Braveheart.1995__#01-26-50_01-32-30_label_B1-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Braveheart.1995__#02-05-34_02-06-40_label_B1-0-0` item=1: gt_interval_has_many_micro_fragments
- XD-Violence `Braveheart.1995__#02-07-00_02-08-15_label_B1-0-0` item=2: gt_interval_has_many_micro_fragments