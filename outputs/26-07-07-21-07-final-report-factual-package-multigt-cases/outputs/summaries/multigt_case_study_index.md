# Multi-GT Case Visualization Index

- Total multi-GT videos: 276.
- Plotted videos: 276.
- Missing plots: 0.

Each figure contains raw anomaly score, GT intervals, and predictions from Peak-Aware, Full, SG0, Recall, and Strict operating points.

## Selection Hints

- Prefer cases with several GT intervals and visible score response if you want visually reliable examples.
- Use `Recall_GT_coverage` and `Strict_purity` together instead of relying on one metric.
- Treat `unsupportable_gt_count` as a diagnostic field, not a label-error claim.

## Preview

| case_id | dataset | video_id | gt_interval_count | supportable_gt_count | unsupportable_gt_count | max_score | Full_GT_coverage | Recall_GT_coverage | Strict_purity | figure_path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0001 | UCF-Crime | Arson011_x264 | 2 | 2 | 0 | 1.000 | 0.686 | 0.686 | 0.693 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0001_UCF-Crime_Arson011_x264.png |
| 0002 | UCF-Crime | Assault010_x264 | 2 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.156 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0002_UCF-Crime_Assault010_x264.png |
| 0003 | UCF-Crime | Burglary021_x264 | 2 | 1 | 0 | 0.800 | 0.925 | 0.925 | 0.481 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0003_UCF-Crime_Burglary021_x264.png |
| 0004 | UCF-Crime | Burglary037_x264 | 2 | 2 | 0 | 1.000 | 0.823 | 0.823 | 0.863 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0004_UCF-Crime_Burglary037_x264.png |
| 0005 | UCF-Crime | Explosion033_x264 | 2 | 2 | 0 | 0.900 | 0.619 | 0.619 | 0.769 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0005_UCF-Crime_Explosion033_x264.png |
| 0006 | UCF-Crime | Shooting046_x264 | 2 | 2 | 0 | 1.000 | 0.769 | 1.000 | 0.103 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0006_UCF-Crime_Shooting046_x264.png |
| 0007 | UCF-Crime | Shooting047_x264 | 2 | 2 | 0 | 1.000 | 0.525 | 0.525 | 0.470 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0007_UCF-Crime_Shooting047_x264.png |
| 0008 | UCF-Crime | Shoplifting007_x264 | 2 | 0 | 2 | 0.900 | 0.580 | 0.580 | 0.165 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0008_UCF-Crime_Shoplifting007_x264.png |
| 0009 | UCF-Crime | Shoplifting010_x264 | 2 | 1 | 1 | 0.500 | 0.000 | 0.000 | NA | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0009_UCF-Crime_Shoplifting010_x264.png |
| 0010 | UCF-Crime | Shoplifting022_x264 | 2 | 0 | 1 | 0.900 | 1.000 | 0.444 | 0.132 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0010_UCF-Crime_Shoplifting022_x264.png |
| 0011 | UCF-Crime | Shoplifting027_x264 | 2 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.299 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0011_UCF-Crime_Shoplifting027_x264.png |
| 0012 | UCF-Crime | Stealing019_x264 | 2 | 0 | 0 | 0.800 | 0.658 | 0.658 | 0.124 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0012_UCF-Crime_Stealing019_x264.png |
| 0013 | UCF-Crime | Stealing079_x264 | 2 | 2 | 0 | 0.600 | 0.039 | 0.000 | NA | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0013_UCF-Crime_Stealing079_x264.png |
| 0014 | UCF-Crime | Vandalism017_x264 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.171 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0014_UCF-Crime_Vandalism017_x264.png |
| 0015 | UCF-Crime | Vandalism028_x264 | 2 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.279 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0015_UCF-Crime_Vandalism028_x264.png |
| 0016 | UCF-Crime | Vandalism036_x264 | 2 | 1 | 1 | 1.000 | 0.727 | 0.727 | 0.375 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0016_UCF-Crime_Vandalism036_x264.png |
| 0017 | XD-Violence | Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6 | 4 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.505 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0017_XD-Violence_Bad.Boys.1995___01-11-55_01-12-40_label_G-B2-B6.png |
| 0018 | XD-Violence | Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0 | 2 | 1 | 0 | 1.000 | 1.000 | 1.000 | 0.702 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0018_XD-Violence_Bad.Boys.1995___01-33-51_01-34-37_label_B2-0-0.png |
| 0019 | XD-Violence | Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0 | 3 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.315 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0019_XD-Violence_Bad.Boys.II.2003___00-06-42_00-10-00_label_B2-G-0.png |
| 0020 | XD-Violence | Black.Hawk.Down.2001__#01-32-40_01-34-00_label_B4-0-0 | 4 | 4 | 0 | 1.000 | 0.871 | 1.000 | 0.776 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0020_XD-Violence_Black.Hawk.Down.2001___01-32-40_01-34-00_label_B4-0-0.png |
| 0021 | XD-Violence | Braveheart.1995__#02-07-00_02-08-15_label_B1-0-0 | 2 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.413 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0021_XD-Violence_Braveheart.1995___02-07-00_02-08-15_label_B1-0-0.png |
| 0022 | XD-Violence | Brick.Mansions.2014__#00-41-25_00-42-36_label_B1-0-0 | 2 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.501 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0022_XD-Violence_Brick.Mansions.2014___00-41-25_00-42-36_label_B1-0-0.png |
| 0023 | XD-Violence | Bullet.in.the.Head.1990__#00-17-20_00-18-55_label_B1-0-0 | 2 | 2 | 0 | 1.000 | 0.996 | 0.996 | 0.784 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0023_XD-Violence_Bullet.in.the.Head.1990___00-17-20_00-18-55_label_B1-0-0.png |
| 0024 | XD-Violence | Bullet.in.the.Head.1990__#00-41-30_00-44-16_label_B4-G-0 | 2 | 1 | 0 | 1.000 | 0.713 | 0.756 | 0.946 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0024_XD-Violence_Bullet.in.the.Head.1990___00-41-30_00-44-16_label_B4-G-0.png |
| 0025 | XD-Violence | Bullet.in.the.Head.1990__#02-02-00_02-05-22_label_B2-B6-G | 6 | 5 | 0 | 1.000 | 0.784 | 0.886 | 0.493 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0025_XD-Violence_Bullet.in.the.Head.1990___02-02-00_02-05-22_label_B2-B6-G.png |
| 0026 | XD-Violence | Casino.Royale.2006__#00-50-05_00-51-16_label_B1-B2-B6 | 6 | 5 | 0 | 1.000 | 1.000 | 1.000 | 0.527 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0026_XD-Violence_Casino.Royale.2006___00-50-05_00-51-16_label_B1-B2-B6.png |
| 0027 | XD-Violence | Casino.Royale.2006__#00-51-16_00-52-41_label_B1-B6-0 | 3 | 3 | 0 | 1.000 | 1.000 | 1.000 | 0.624 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0027_XD-Violence_Casino.Royale.2006___00-51-16_00-52-41_label_B1-B6-0.png |
| 0028 | XD-Violence | City.Of.Men.2007__#00-51-50_00-53-31_label_B2-0-0 | 2 | 1 | 0 | 1.000 | 1.000 | 1.000 | 0.155 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0028_XD-Violence_City.Of.Men.2007___00-51-50_00-53-31_label_B2-0-0.png |
| 0029 | XD-Violence | City.Of.Men.2007__#00-57-37_00-58-27_label_B2-0-0 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.105 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0029_XD-Violence_City.Of.Men.2007___00-57-37_00-58-27_label_B2-0-0.png |
| 0030 | XD-Violence | City.of.God.2002__#00-40-16_00-41-30_label_B2-0-0 | 3 | 1 | 0 | 1.000 | 0.363 | 0.363 | 0.073 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0030_XD-Violence_City.of.God.2002___00-40-16_00-41-30_label_B2-0-0.png |
| 0031 | XD-Violence | City.of.God.2002__#01-52-20_01-54-32_label_B2-0-0 | 3 | 1 | 0 | 1.000 | 0.831 | 0.831 | 0.150 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0031_XD-Violence_City.of.God.2002___01-52-20_01-54-32_label_B2-0-0.png |
| 0032 | XD-Violence | Crank.Dircut.2006__#0-27-42_0-29-01_label_B1-0-0 | 3 | 2 | 1 | 1.000 | 0.918 | 1.000 | 0.490 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0032_XD-Violence_Crank.Dircut.2006___0-27-42_0-29-01_label_B1-0-0.png |
| 0033 | XD-Violence | Deadpool.2.2018__#0-04-46_0-05-01_label_B2-0-0 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.345 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0033_XD-Violence_Deadpool.2.2018___0-04-46_0-05-01_label_B2-0-0.png |
| 0034 | XD-Violence | Deadpool.2.2018__#0-50-30_0-51-20_label_B1-0-0 | 3 | 2 | 1 | 0.900 | 0.694 | 0.694 | 0.513 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0034_XD-Violence_Deadpool.2.2018___0-50-30_0-51-20_label_B1-0-0.png |
| 0035 | XD-Violence | Deadpool.2016__#0-18-58_0-19-20_label_B1-0-0 | 2 | 1 | 0 | 1.000 | 0.815 | 0.815 | 0.478 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0035_XD-Violence_Deadpool.2016___0-18-58_0-19-20_label_B1-0-0.png |
| 0036 | XD-Violence | Death.Proof.2007__#00-45-05_00-47-36_label_B5-0-0 | 2 | 1 | 0 | 1.000 | 0.176 | 0.176 | 0.027 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0036_XD-Violence_Death.Proof.2007___00-45-05_00-47-36_label_B5-0-0.png |
| 0037 | XD-Violence | Death.Proof.2007__#01-40-41_01-42-17_label_B5-B6-0 | 2 | 1 | 0 | 0.900 | 1.000 | 1.000 | 0.304 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0037_XD-Violence_Death.Proof.2007___01-40-41_01-42-17_label_B5-B6-0.png |
| 0038 | XD-Violence | Desperado.1995__#00-16-48_00-18-52_label_B1-0-0 | 5 | 2 | 1 | 1.000 | 0.968 | 0.983 | 0.733 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0038_XD-Violence_Desperado.1995___00-16-48_00-18-52_label_B1-0-0.png |
| 0039 | XD-Violence | Desperado.1995__#00-38-36_00-39-21_label_B1-B2-0 | 4 | 2 | 1 | 1.000 | 1.000 | 1.000 | 0.418 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0039_XD-Violence_Desperado.1995___00-38-36_00-39-21_label_B1-B2-0.png |
| 0040 | XD-Violence | Desperado.1995__#01-14-11_01-17-28_label_B2-G-0 | 6 | 4 | 0 | 1.000 | 0.979 | 1.000 | 0.373 | outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/outputs/figures/multigt_case_studies/0040_XD-Violence_Desperado.1995___01-14-11_01-17-28_label_B2-G-0.png |

Preview truncated to 40 rows. Full table: `outputs/summaries/multigt_case_study_index.csv`.