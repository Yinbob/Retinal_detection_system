from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..types import LocalFeatures, ProcessedImage


SIFT_CONFIG_DEFAULTS: dict[str, Any] = {
    "nfeatures": 1500,
    "contrast_threshold": 0.02,
    "edge_threshold": 10,
    "sigma": 1.6,
}


def extract_sift(
    processed: ProcessedImage,
    config: dict[str, Any] | None = None,
) -> LocalFeatures:
    merged = {**SIFT_CONFIG_DEFAULTS, **(config or {})}
    gray = processed.gray.copy()
    gray[processed.fov == 0] = 0
    try:
        sift = cv2.SIFT_create(
            nfeatures=int(merged["nfeatures"]),
            contrastThreshold=float(merged["contrast_threshold"]),
            edgeThreshold=int(merged["edge_threshold"]),
            sigma=float(merged["sigma"]),
        )
        keypoints, descriptors = sift.detectAndCompute(gray, processed.fov)
    except cv2.error as error:
        return LocalFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            valid=False,
            reason=f"opencv_error:{error.code}",
        )
    if descriptors is None or len(descriptors) < 2:
        return LocalFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            valid=False,
            reason="too_few_descriptors",
        )
    points = np.asarray(
        [keypoint.pt for keypoint in keypoints], dtype=np.float32
    )
    descriptors = np.asarray(descriptors, dtype=np.float32)
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]
    descriptors = descriptors[order]
    if not np.isfinite(points).all() or not np.isfinite(descriptors).all():
        return LocalFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            valid=False,
            reason="non_finite_sift_feature",
        )
    return LocalFeatures(
        keypoints=points,
        descriptors=descriptors,
        valid=True,
    )

