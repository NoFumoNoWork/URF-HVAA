# Top-K Coverage Report

| K | Segments | Missed | Segment miss rate | Video any miss rate | Mean coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 1394 | 864 | 61.98% | 59.22% | 0.255 |
| 2 | 1394 | 615 | 44.12% | 40.47% | 0.420 |
| 3 | 1394 | 485 | 34.79% | 32.97% | 0.529 |
| 5 | 1394 | 312 | 22.38% | 23.28% | 0.687 |
| 10 | 1394 | 79 | 5.67% | 7.81% | 0.902 |

## Interpretation

- If miss rate falls sharply as K grows, the single-window output structure is the main bottleneck.
- If miss rate stays high, the score curve, window scale, or event proposal method also needs improvement.