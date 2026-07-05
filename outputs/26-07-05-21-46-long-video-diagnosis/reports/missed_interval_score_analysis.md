# Missed Interval Score Analysis

- Total GT intervals with scores: 1394
- Missed by single highest interval: 864
- Missed but high score percentile (gt_max >= 80th percentile): 664
- Missed and low/moderate score percentile: 199

## By Dataset

- UCF-Crime: total=156, missed=86, missed_high_score=77
- XD-Violence: total=1238, missed=778, missed_high_score=587

## Interpretation

- Missed intervals with high score percentiles indicate the output structure is a bottleneck: the score curve saw the event, but one Wmax could not keep multiple regions.
- Missed intervals with low score percentiles indicate additional score/caption recognition weakness or unsuitable temporal scale.