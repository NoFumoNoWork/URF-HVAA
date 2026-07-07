# Warnings Taxonomy

- Total warnings: 24.
- Warning stages: sg=24.

## Warning Types

| warning_type | count | impact |
|---|---:|---|
| `sg_curve_missing_raw_fallback` | 24 | non-fatal; selected SG curve was unavailable and raw score fallback was used for that video/configuration |

## Validity Check

- Videos skipped due to missing/too-few score files: 0.
- GT file read failures: 0 observed.
- Prediction file read failures: 0 observed.
- Empty prediction run warnings: 0.
- NaN / empty candidate / plotting-only warnings: no metric-affecting cases observed in the final-materials run.
- Aggregate metrics impact: the observed warnings are SG curve fallback events for a small number of short/edge score series repeated across per-configuration candidate generation. The implementation falls back to available raw/residual/trend evidence where needed, so aggregate metrics remain valid for the reported operating-point comparison.

Report-ready sentence: The warning log contains only non-fatal SG curve fallback events caused by short or edge-case score series; no videos, GT rows, or prediction files were skipped, and aggregate metrics are not invalidated.