from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from ..cache import FeatureCache, feature_cache_key
import numpy as np

from ..preprocessing import PreprocessConfig, preprocess_array, preprocess_record
from ..types import GlobalFeatures, ImageRecord, LocalFeatures, ProcessedImage
from ..utils import read_gray, stable_hash
from .hog import HOG_CONFIG_DEFAULTS, extract_hog
from .sift import SIFT_CONFIG_DEFAULTS, extract_sift


def local_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = {**SIFT_CONFIG_DEFAULTS, **config}
    return {
        key: merged[key]
        for key in (
            "nfeatures",
            "contrast_threshold",
            "edge_threshold",
            "sigma",
        )
    }


def global_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = {**HOG_CONFIG_DEFAULTS, **config}
    return {
        key: merged[key]
        for key in (
            "image_size",
            "orientations",
            "pixels_per_cell",
            "cells_per_block",
            "block_norm",
        )
    }


class FeatureRepository:
    def __init__(
        self,
        records: Iterable[ImageRecord],
        dataset_root: str | Path,
        preprocess_config: PreprocessConfig,
        cache: FeatureCache,
        progress: Callable[[int, str], None] | None = None,
    ) -> None:
        self.record_by_id = {record.image_id: record for record in records}
        self.dataset_root = Path(dataset_root)
        self.preprocess_config = preprocess_config
        self.cache = cache
        self.progress = progress
        self._processed: dict[str, ProcessedImage] = {}
        self._local: dict[tuple[str, str], LocalFeatures] = {}
        self._global: dict[tuple[str, str], GlobalFeatures] = {}
        self.processed_count = 0
        self.local_computed = 0
        self.global_computed = 0

    def _notify(self, processed: int, method: str) -> None:
        if self.progress is not None:
            self.progress(processed, method)

    def processed(self, record: ImageRecord) -> ProcessedImage:
        cached = self._processed.get(record.pixel_sha256)
        if cached is not None:
            return cached
        image = preprocess_record(
            record, self.dataset_root, self.preprocess_config
        )
        self._processed[record.pixel_sha256] = image
        self.processed_count += 1
        self._notify(self.processed_count, "preprocess")
        return image

    def register_array(self, record: ImageRecord, rgb: np.ndarray) -> None:
        """Make an in-memory RGB query addressable by its pixel hash.

        Used for uploads: the image is preprocessed like any dataset record but
        never written to the manifest or protocol.
        """
        if record.pixel_sha256 in self._processed:
            return
        mask = None
        if record.mask_path is not None:
            mask = read_gray(self.dataset_root / record.mask_path)
        self._processed[record.pixel_sha256] = preprocess_array(
            rgb,
            pixel_hash=record.pixel_sha256,
            config=self.preprocess_config,
            external_mask=mask,
        )
        self.processed_count += 1
        self._notify(self.processed_count, "preprocess")

    def local(
        self,
        record: ImageRecord,
        config: dict[str, Any],
    ) -> LocalFeatures:
        feature_config = local_feature_config(config)
        key = feature_cache_key(
            record.pixel_sha256,
            self.preprocess_config.to_dict(),
            feature_config,
        )
        cache_key = (record.pixel_sha256, key)
        cached = self._local.get(cache_key)
        if cached is not None:
            return cached
        loaded = self.cache.load_local(key)
        if loaded is None:
            loaded = extract_sift(self.processed(record), feature_config)
            self.cache.save_local(key, loaded)
        self._local[cache_key] = loaded
        self.local_computed += 1
        self._notify(self.local_computed, "sift")
        return loaded

    def global_feature(
        self,
        record: ImageRecord,
        config: dict[str, Any],
    ) -> GlobalFeatures:
        feature_config = global_feature_config(config)
        key = feature_cache_key(
            record.pixel_sha256,
            self.preprocess_config.to_dict(),
            feature_config,
        )
        cache_key = (record.pixel_sha256, key)
        cached = self._global.get(cache_key)
        if cached is not None:
            return cached
        loaded = self.cache.load_global(key)
        if loaded is None:
            loaded = extract_hog(self.processed(record), feature_config)
            self.cache.save_global(key, loaded)
        self._global[cache_key] = loaded
        self.global_computed += 1
        self._notify(self.global_computed, "hog")
        return loaded

    def config_fingerprint(
        self,
        method: str,
        config: dict[str, Any],
    ) -> str:
        if method == "sift":
            feature_config = local_feature_config(config)
        elif method == "hog":
            feature_config = global_feature_config(config)
        else:
            feature_config = config
        return stable_hash(
            {
                "method": method,
                "preprocess": self.preprocess_config.to_dict(),
                "feature": feature_config,
            }
        )

