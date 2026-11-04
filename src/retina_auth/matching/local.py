from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ..types import LocalFeatures, PairResult


LOCAL_MATCH_DEFAULTS: dict[str, Any] = {
    "ratio": 0.75,
    "ransac_px": 3.0,
    "max_iters": 2000,
    "confidence": 0.995,
    "min_matches": 8,
    "min_inliers": 8,
    "bidirectional": False,
}


def _invalid_result(
    reason: str,
    *,
    started: float,
    q_features: LocalFeatures,
    r_features: LocalFeatures,
    matches: int = 0,
    inliers: int = 0,
) -> PairResult:
    return PairResult(
        score=float(inliers),
        valid=False,
        reason=reason,
        n_keypoints_q=len(q_features.keypoints),
        n_keypoints_r=len(r_features.keypoints),
        n_ratio_matches=matches,
        n_inliers=inliers,
        inlier_ratio=(inliers / matches) if matches else 0.0,
        median_reprojection_error=None,
        transform=None,
        match_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def _ratio_matches(
    query_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
    good = [
        pair[0]
        for pair in forward
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    return good


def compare_local(
    query: LocalFeatures,
    reference: LocalFeatures,
    config: dict[str, Any] | None = None,
) -> PairResult:
    started = time.perf_counter()
    merged = {**LOCAL_MATCH_DEFAULTS, **(config or {})}
    if not query.valid:
        return _invalid_result(
            f"query:{query.reason or 'invalid_feature'}",
            started=started,
            q_features=query,
            r_features=reference,
        )
    if not reference.valid:
        return _invalid_result(
            f"reference:{reference.reason or 'invalid_feature'}",
            started=started,
            q_features=query,
            r_features=reference,
        )
    if len(query.descriptors) < 2 or len(reference.descriptors) < 2:
        return _invalid_result(
            "too_few_descriptors",
            started=started,
            q_features=query,
            r_features=reference,
        )

    ratio = float(merged["ratio"])
    good = _ratio_matches(query.descriptors, reference.descriptors, ratio)
    if bool(merged.get("bidirectional", False)):
        reverse = _ratio_matches(reference.descriptors, query.descriptors, ratio)
        reverse_pairs = {
            (match.trainIdx, match.queryIdx) for match in reverse
        }
        good = [
            match
            for match in good
            if (match.queryIdx, match.trainIdx) in reverse_pairs
        ]
    if len(good) < int(merged["min_matches"]):
        return _invalid_result(
            "too_few_matches",
            started=started,
            q_features=query,
            r_features=reference,
            matches=len(good),
        )

    source = np.asarray(
        [query.keypoints[match.queryIdx] for match in good], dtype=np.float32
    ).reshape(-1, 1, 2)
    destination = np.asarray(
        [reference.keypoints[match.trainIdx] for match in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    cv2.setRNGSeed(20261102)
    try:
        transform, inlier_mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            float(merged["ransac_px"]),
            maxIters=int(merged["max_iters"]),
            confidence=float(merged["confidence"]),
        )
    except cv2.error:
        return _invalid_result(
            "homography_error",
            started=started,
            q_features=query,
            r_features=reference,
            matches=len(good),
        )
    if transform is None or inlier_mask is None:
        return _invalid_result(
            "homography_failed",
            started=started,
            q_features=query,
            r_features=reference,
            matches=len(good),
        )
    if not np.isfinite(transform).all():
        return _invalid_result(
            "non_finite_homography",
            started=started,
            q_features=query,
            r_features=reference,
            matches=len(good),
        )
    inliers = int(inlier_mask.ravel().sum())
    if inliers < int(merged["min_inliers"]):
        return _invalid_result(
            "too_few_inliers",
            started=started,
            q_features=query,
            r_features=reference,
            matches=len(good),
            inliers=inliers,
        )

    source_points = source.reshape(-1, 2)
    destination_points = destination.reshape(-1, 2)
    homogeneous = np.column_stack(
        [source_points, np.ones(len(source_points), dtype=np.float32)]
    )
    projected = homogeneous @ transform.T
    if np.any(~np.isfinite(projected)) or np.any(
        np.abs(projected[:, 2]) < 1e-12
    ):
        return _invalid_result(
            "non_finite_projection",
            started=started,
            q_features=query,
            r_features=reference,
            matches=len(good),
            inliers=inliers,
        )
    projected_xy = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected_xy - destination_points, axis=1)
    inlier_errors = errors[inlier_mask.ravel().astype(bool)]
    median_error = float(np.median(inlier_errors))
    return PairResult(
        score=float(inliers),
        valid=True,
        reason=None,
        n_keypoints_q=len(query.keypoints),
        n_keypoints_r=len(reference.keypoints),
        n_ratio_matches=len(good),
        n_inliers=inliers,
        inlier_ratio=inliers / len(good),
        median_reprojection_error=median_error,
        transform=tuple(
            tuple(float(value) for value in row) for row in transform
        ),
        match_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def inlier_correspondences(
    query: LocalFeatures,
    reference: LocalFeatures,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return processed-coordinate keypoint pairs that support the homography."""
    merged = {**LOCAL_MATCH_DEFAULTS, **(config or {})}
    if (
        not query.valid
        or not reference.valid
        or len(query.descriptors) < 2
        or len(reference.descriptors) < 2
    ):
        return np.empty((0, 2)), np.empty((0, 2))

    ratio = float(merged["ratio"])
    good = _ratio_matches(query.descriptors, reference.descriptors, ratio)
    if bool(merged.get("bidirectional", False)):
        reverse = _ratio_matches(reference.descriptors, query.descriptors, ratio)
        reverse_pairs = {
            (match.trainIdx, match.queryIdx) for match in reverse
        }
        good = [
            match
            for match in good
            if (match.queryIdx, match.trainIdx) in reverse_pairs
        ]
    if len(good) < int(merged["min_matches"]):
        return np.empty((0, 2)), np.empty((0, 2))

    source = np.asarray(
        [query.keypoints[match.queryIdx] for match in good], dtype=np.float32
    ).reshape(-1, 1, 2)
    destination = np.asarray(
        [reference.keypoints[match.trainIdx] for match in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    cv2.setRNGSeed(20261102)
    try:
        _, mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            float(merged["ransac_px"]),
            maxIters=int(merged["max_iters"]),
            confidence=float(merged["confidence"]),
        )
    except cv2.error:
        return np.empty((0, 2)), np.empty((0, 2))
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2))
    inliers = mask.ravel().astype(bool)
    return (
        source.reshape(-1, 2)[inliers],
        destination.reshape(-1, 2)[inliers],
    )

