# Case Study Index

## Case 01: over-wide full corrected by SG0/Strict

| case_id | video_id | dataset | case_type | why_selected | GT intervals summary | score curve summary | Peak-Aware prediction summary | Full default prediction summary | SG0 prediction summary | Recall-oriented prediction summary | Strict-oriented prediction summary | figure_path | one_sentence_conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | v=15wDrZJQpsw__#00-00-00_00-00-51_label_B6-0-0 | XD-Violence | over-wide full corrected by SG0/Strict | Full default predicts a broader interval footprint than SG0/Strict while the conservative variants retain some GT overlap. | 434-562 [supportable]; 749-866 [unsupportable] |  |  | 48-1040 | 416-432; 528-544; 976-992 | 48-1040 | 416-432; 528-544; 976-992 | figures/final_report/fig_08_case_01.png | Full default predicts a broader interval footprint than SG0/Strict while the conservative variants retain some GT overlap. |

## Case 02: recall-oriented recovers supportable GT

| case_id | video_id | dataset | case_type | why_selected | GT intervals summary | score curve summary | Peak-Aware prediction summary | Full default prediction summary | SG0 prediction summary | Recall-oriented prediction summary | Strict-oriented prediction summary | figure_path | one_sentence_conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 02 | The.Hurt.Locker.2008__#0-19-22_0-22-32_label_B2-B1-0 | XD-Violence | recall-oriented recovers supportable GT | Recall-oriented configuration improves supportable GT coverage relative to Full default. | 2094-2125 [uncertain]; 2244-2270 [unsupportable]; 2532-2575 [uncertain]; 3580-3750 [supportable] |  |  | 240-368; 480-992; 1088-1408; 1744-3568 | 240-368; 480-992; 1088-1408; 1744-3568 | 208-1024; 1088-1520; 1696-3856 | 240-368; 480-992; 1088-1408; 1744-3568 | figures/final_report/fig_08_case_02.png | Recall-oriented configuration improves supportable GT coverage relative to Full default. |

## Case 03: residual direct evidence over-extension

| case_id | video_id | dataset | case_type | why_selected | GT intervals summary | score curve summary | Peak-Aware prediction summary | Full default prediction summary | SG0 prediction summary | Recall-oriented prediction summary | Strict-oriented prediction summary | figure_path | one_sentence_conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 03 | v=9Jk2sIp5MRQ__#1_label_G-0-0 | XD-Violence | residual direct evidence over-extension | Full default expands more than SG0, illustrating how direct evidence can increase duration and unsupported coverage. | 175-215 [uncertain]; 273-567 [supportable] |  |  | 0-812 | 240-256 | 240-256 | 240-256 | figures/final_report/fig_08_case_03.png | Full default expands more than SG0, illustrating how direct evidence can increase duration and unsupported coverage. |

## Case 04: score-unsupported GT partially covered

| case_id | video_id | dataset | case_type | why_selected | GT intervals summary | score curve summary | Peak-Aware prediction summary | Full default prediction summary | SG0 prediction summary | Recall-oriented prediction summary | Strict-oriented prediction summary | figure_path | one_sentence_conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04 | Crank.Dircut.2006__#0-27-42_0-29-01_label_B1-0-0 | XD-Violence | score-unsupported GT partially covered | The video has score-unsupported GT that is nevertheless partially covered, useful for explaining unsupported coverage as diagnostic rather than purely negative. | 347-445 [supportable]; 560-1427 [supportable]; 1886-1894 [unsupportable] |  |  | 0-592; 672-1904 | 0-592; 672-1904 | 0-1904 | 0-592; 672-1904 | figures/final_report/fig_08_case_04.png | The video has score-unsupported GT that is nevertheless partially covered, useful for explaining unsupported coverage as diagnostic rather than purely negative. |

## Case 05: score no-response failure

| case_id | video_id | dataset | case_type | why_selected | GT intervals summary | score curve summary | Peak-Aware prediction summary | Full default prediction summary | SG0 prediction summary | Recall-oriented prediction summary | Strict-oriented prediction summary | figure_path | one_sentence_conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 05 | Abuse028_x264 | UCF-Crime | score no-response failure | All operating points have very low GT coverage and/or the score curve has weak response, so post-processing cannot reliably recover the event. | 165-240 [uncertain] |  |  | 544-1200 | none | none | none | figures/final_report/fig_08_case_05.png | All operating points have very low GT coverage and/or the score curve has weak response, so post-processing cannot reliably recover the event. |
