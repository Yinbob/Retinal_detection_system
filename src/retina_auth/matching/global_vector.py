from __future__ import annotations

import time

import numpy as np

from ..types import GlobalFeatures, PairResult


def compare_global(
    query: GlobalFeatures,
    reference: GlobalFeatures,
) -> PairResult:
    started = time.perf_counter()
    valid = query.valid and reference.valid
    reason = None
    if not query.valid:
        reason = f"query:{query.reason or 'invalid_feature'}"
    elif not reference.valid:
        reason = f"reference:{reference.reason or 'invalid_feature'}"
    score = float("nan")
    if valid:
        score = float(np.dot(query.vector, reference.vector))
        if not np.isfinite(score):
            valid = False
            reason = "non_finite_similarity"
    return PairResult(
        score=score if valid else 0.0,
        valid=valid,
        reason=reason,
        n_keypoints_q=0,
        n_keypoints_r=0,
        n_ratio_matches=0,
        n_inliers=0,
        inlier_ratio=0.0,
        median_reprojection_error=None,
        transform=None,
        match_time_ms=(time.perf_counter() - started) * 1000.0,
    )

