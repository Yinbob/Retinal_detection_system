from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.feature import hog

from ..types import GlobalFeatures, ProcessedImage


HOG_CONFIG_DEFAULTS: dict[str, Any] = {
    "image_size": 512,
    "orientations": 9,
    "pixels_per_cell": [16, 16],
    "cells_per_block": [2, 2],
    "block_norm": "L2-Hys",
}


def extract_hog(
    processed: ProcessedImage,
    config: dict[str, Any] | None = None,
) -> GlobalFeatures:
    merged = {**HOG_CONFIG_DEFAULTS, **(config or {})}
    size = int(merged["image_size"])
    gray = cv2.resize(
        processed.gray,
        (size, size),
        interpolation=cv2.INTER_AREA if size < processed.gray.shape[0] else cv2.INTER_LINEAR,
    )
    mask = cv2.resize(processed.fov, (size, size), interpolation=cv2.INTER_NEAREST)
    gray[mask == 0] = 0
    try:
        vector = hog(
            gray,
            orientations=int(merged["orientations"]),
            pixels_per_cell=tuple(int(value) for value in merged["pixels_per_cell"]),
            cells_per_block=tuple(
                int(value) for value in merged["cells_per_block"]
            ),
            block_norm=str(merged["block_norm"]),
            feature_vector=True,
        ).astype(np.float32)
    except (ValueError, TypeError) as error:
        return GlobalFeatures(
            vector=np.empty(0, dtype=np.float32),
            valid=False,
            reason=f"hog_error:{type(error).__name__}",
        )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        return GlobalFeatures(
            vector=np.empty(0, dtype=np.float32),
            valid=False,
            reason="invalid_global_vector",
        )
    return GlobalFeatures(vector=(vector / norm).astype(np.float32), valid=True)

