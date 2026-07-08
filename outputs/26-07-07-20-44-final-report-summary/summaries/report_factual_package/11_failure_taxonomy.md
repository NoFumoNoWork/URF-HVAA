# Failure Taxonomy

## score-supported but missed

The score curve has local response, but the selected post-processing intervals do not overlap enough GT. This points to fusion threshold, interval merge, or candidate-source selection rather than score generation alone.

## score-supported but fragmented

The score curve contains many local peaks or micro-events, but the output remains split across fragments or fails to form the event-level interval. This corresponds to micro-event grouping and hierarchical merge behavior.

## over-merged prediction

Multiple events or long background spans are merged into a broad interval. This is typically linked to merge gap, residual evidence, trend evidence, and insufficient duration control.

## score-unsupported GT

Human GT exists but the anomaly score curve has weak or absent response. Score-only post-processing cannot reliably recover these intervals. Possible causes include VAD model misses, overly broad human event boundaries, or different temporal semantics between event-level GT and score-level evidence.

## boundary mismatch

The prediction covers the anomaly core but not the full human boundary, or extends beyond it. Human labels may encode event context, while score curves often peak around visually salient frames.


## Required Categories

- score-supported but missed: supportable GT remains uncovered; may require candidate generation or score generation changes.
- score-supported but fragmented: several short intervals cover one event; merging or higher-level event grouping is needed.
- over-merged prediction: adjacent candidates merge into broad predictions; penalties and stricter thresholds can reduce this.
- score-unsupported GT: human interval has weak score evidence; not automatically an annotation error.
- boundary mismatch: frame-level score peaks do not align with event-level human boundaries.
- low-score context coverage: predictions cover contextual parts around score-supported events.
- candidate generation miss: no candidate exists for a GT segment; fusion cannot recover it.
- fusion scoring miss: candidate exists but falls below threshold; weights/penalties govern the failure.
