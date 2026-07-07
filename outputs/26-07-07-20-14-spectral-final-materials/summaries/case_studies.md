# Case Studies

These cases are selected to explain behavior, not to estimate generalization.

## Case 1: over-wide full corrected by SG0/Strict

- Video: `XD-Violence / v=15wDrZJQpsw__#00-00-00_00-00-51_label_B6-0-0`.
- Why selected: Full default predicts a broader interval footprint than SG0/Strict while the conservative variants retain some GT overlap.
- GT intervals: 434-562 [supportable]; 749-866 [unsupportable]
- Peak-Aware prediction: 48-1040
- Full default prediction: 48-1040
- SG0 prediction: 416-432; 528-544; 976-992
- Recall-oriented prediction: 48-1040
- Strict-oriented prediction: 416-432; 528-544; 976-992
- Score curve summary: score_points=77.000; max_score=0.800; mean_score=0.196; frac_score_ge_0.6=0.091
- Figure: `outputs\26-07-07-20-14-spectral-final-materials\figures\case_studies\01_XD-Violence_v=15wDrZJQpsw__#00-00-00_00-00-51_label_B6-0-0.png`

## Case 2: recall-oriented recovers supportable GT

- Video: `XD-Violence / The.Hurt.Locker.2008__#0-19-22_0-22-32_label_B2-B1-0`.
- Why selected: Recall-oriented configuration improves supportable GT coverage relative to Full default.
- GT intervals: 2094-2125 [uncertain]; 2244-2270 [unsupportable]; 2532-2575 [uncertain]; 3580-3750 [supportable]
- Peak-Aware prediction: 240-352; 784-992; 1088-1136; 1760-3568
- Full default prediction: 240-368; 480-992; 1088-1408; 1744-3568
- SG0 prediction: 240-368; 480-992; 1088-1408; 1744-3568
- Recall-oriented prediction: 208-1024; 1088-1520; 1696-3856
- Strict-oriented prediction: 240-368; 480-992; 1088-1408; 1744-3568
- Score curve summary: score_points=286.000; max_score=1.000; mean_score=0.579; frac_score_ge_0.6=0.615
- Figure: `outputs\26-07-07-20-14-spectral-final-materials\figures\case_studies\02_XD-Violence_The.Hurt.Locker.2008__#0-19-22_0-22-32_label_B2-B1-0.png`

## Case 3: residual direct evidence over-extension

- Video: `XD-Violence / v=9Jk2sIp5MRQ__#1_label_G-0-0`.
- Why selected: Full default expands more than SG0, illustrating how direct evidence can increase duration and unsupported coverage.
- GT intervals: 175-215 [uncertain]; 273-567 [supportable]
- Peak-Aware prediction: 0-576
- Full default prediction: 0-812
- SG0 prediction: 240-256
- Recall-oriented prediction: 240-256
- Strict-oriented prediction: 240-256
- Score curve summary: score_points=36.000; max_score=0.700; mean_score=0.203; frac_score_ge_0.6=0.056
- Figure: `outputs\26-07-07-20-14-spectral-final-materials\figures\case_studies\03_XD-Violence_v=9Jk2sIp5MRQ__#1_label_G-0-0.png`

## Case 4: score-unsupported GT partially covered

- Video: `XD-Violence / Crank.Dircut.2006__#0-27-42_0-29-01_label_B1-0-0`.
- Why selected: The video has score-unsupported GT that is nevertheless partially covered, useful for explaining unsupported coverage as diagnostic rather than purely negative.
- GT intervals: 347-445 [supportable]; 560-1427 [supportable]; 1886-1894 [unsupportable]
- Peak-Aware prediction: 0-128; 320-560; 720-1536; 1696-1728
- Full default prediction: 0-592; 672-1904
- SG0 prediction: 0-592; 672-1904
- Recall-oriented prediction: 0-1904
- Strict-oriented prediction: 0-592; 672-1904
- Score curve summary: score_points=119.000; max_score=1.000; mean_score=0.750; frac_score_ge_0.6=0.874
- Figure: `outputs\26-07-07-20-14-spectral-final-materials\figures\case_studies\04_XD-Violence_Crank.Dircut.2006__#0-27-42_0-29-01_label_B1-0-0.png`

## Case 5: score no-response failure

- Video: `UCF-Crime / Abuse028_x264`.
- Why selected: All operating points have very low GT coverage and/or the score curve has weak response, so post-processing cannot reliably recover the event.
- GT intervals: 165-240 [uncertain]
- Peak-Aware prediction: 544-1200
- Full default prediction: 544-1200
- SG0 prediction: none
- Recall-oriented prediction: none
- Strict-oriented prediction: none
- Score curve summary: score_points=89.000; max_score=0.700; mean_score=0.203; frac_score_ge_0.6=0.056
- Figure: `outputs\26-07-07-20-14-spectral-final-materials\figures\case_studies\05_UCF-Crime_Abuse028_x264.png`
