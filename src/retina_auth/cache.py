from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .types import GlobalFeatures, LocalFeatures
from .utils import stable_hash


FEATURE_IMPLEMENTATION_VERSION = "features-v2"


def feature_cache_key(
    pixel_sha256: str,
    preprocess_config: dict[str, Any],
    feature_config: dict[str, Any],
    implementation_version: str = FEATURE_IMPLEMENTATION_VERSION,
) -> str:
    return stable_hash(
        {
            "pixel_sha256": pixel_sha256,
            "preprocess": preprocess_config,
            "feature": feature_config,
            "implementation_version": implementation_version,
        }
    )


class FeatureCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / key[:2] / f"{key}.npz"

    def load_local(self, key: str) -> LocalFeatures | None:
        path = self.path("sift", key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return LocalFeatures(
                    keypoints=np.asarray(data["keypoints"], dtype=np.float32),
                    descriptors=np.asarray(data["descriptors"], dtype=np.float32),
                    valid=bool(data["valid"]),
                    reason=(
                        None
                        if str(data["reason"]) == ""
                        else str(data["reason"])
                    ),
                )
        except (OSError, ValueError, KeyError):
            return None

    def save_local(self, key: str, features: LocalFeatures) -> None:
        path = self.path("sift", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            keypoints=features.keypoints.astype(np.float32),
            descriptors=features.descriptors.astype(np.float32),
            valid=np.asarray(features.valid),
            reason=np.asarray(features.reason or ""),
        )
        os.replace(temporary, path)

    def load_global(self, key: str) -> GlobalFeatures | None:
        path = self.path("hog", key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return GlobalFeatures(
                    vector=np.asarray(data["vector"], dtype=np.float32),
                    valid=bool(data["valid"]),
                    reason=(
                        None
                        if str(data["reason"]) == ""
                        else str(data["reason"])
                    ),
                )
        except (OSError, ValueError, KeyError):
            return None

    def save_global(self, key: str, features: GlobalFeatures) -> None:
        path = self.path("hog", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            vector=features.vector.astype(np.float32),
            valid=np.asarray(features.valid),
            reason=np.asarray(features.reason or ""),
        )
        os.replace(temporary, path)

