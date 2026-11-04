"""Query-time identification shared by the CLI and the local web service."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache import FeatureCache
from .experiments import (
    _classify_error_row,
    load_artifact_state,
    load_config,
    load_frozen,
)
from .features.repository import FeatureRepository
from .matching import compare_global, compare_local
from .matching.local import inlier_correspondences
from .preprocessing import PreprocessConfig
from .retrieval import retrieve
from .types import DatasetManifest, ImageRecord, RetrievalProtocol
from .utils import pixel_sha256_rgb, read_rgb, sha256_bytes


METHODS = ("sift", "sift_bidirectional", "hog")
DATASETS = ("vistel", "fire")


@dataclass
class DatasetState:
    manifest: DatasetManifest
    gallery: tuple[ImageRecord, ...]
    repository: FeatureRepository
    protocol: RetrievalProtocol
    fire_pair_protocol: RetrievalProtocol | None = None
    protocols: dict[str, RetrievalProtocol] = field(default_factory=dict)


@dataclass
class IdentificationService:
    """Loads artifacts once and answers identification queries.

    Feature extraction for gallery images is cached on disk and in memory, so
    repeated queries against the same dataset do not reprocess the gallery.
    """

    artifacts_dir: Path
    config: dict[str, Any]
    frozen: dict[str, Any]
    manifests: dict[str, DatasetManifest]
    protocols: dict[str, RetrievalProtocol]
    states: dict[str, DatasetState]

    @classmethod
    def load(
        cls,
        artifacts_dir: str | Path,
        config_path: str | Path,
        frozen_path: str | Path | None = None,
    ) -> "IdentificationService":
        artifacts = Path(artifacts_dir)
        config = load_config(config_path)
        frozen = load_frozen(frozen_path or artifacts / "frozen_config.json")
        manifests, protocols = load_artifact_state(artifacts)
        preprocess = PreprocessConfig.from_dict(config["preprocessing"])
        cache = FeatureCache(artifacts / "cache" / "features")
        states: dict[str, DatasetState] = {}
        for dataset in DATASETS:
            manifest = manifests[dataset]
            gallery = tuple(
                record for record in manifest.records if record.role == "gallery"
            )
            repository = FeatureRepository(
                manifest.records, manifest.root, preprocess, cache
            )
            if dataset == "vistel":
                protocol = protocols["vistel_course"]
            else:
                protocol = protocols["fire_pair"]
            states[dataset] = DatasetState(
                manifest=manifest,
                gallery=gallery,
                repository=repository,
                protocol=protocol,
                protocols={
                    name: p for name, p in protocols.items() if p.dataset == dataset
                },
            )
        return cls(
            artifacts_dir=artifacts,
            config=config,
            frozen=frozen,
            manifests=manifests,
            protocols=protocols,
            states=states,
        )

    def method_setting(self, method: str) -> tuple[dict[str, Any], float]:
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")
        if (
            method == "sift_bidirectional"
            and "sift_bidirectional" not in self.frozen.get("methods", {})
            and "sift" in self.frozen.get("methods", {})
        ):
            bi_cand = self.frozen["methods"]["sift"]["bidirectional_candidate"]
            raw = float(bi_cand["threshold"]["threshold"])
            eff = (
                raw
                if math.isfinite(raw)
                else float(bi_cand["config"].get("min_inliers", 6))
            )
            return {**bi_cand["config"], "bidirectional": True}, eff
        selected = self.frozen["methods"][method]["selected"]
        raw = float(selected["threshold"]["threshold"])
        eff = (
            raw
            if math.isfinite(raw)
            else float(selected["config"].get("min_inliers", 6))
        )
        cfg = dict(selected["config"])
        if method == "sift_bidirectional":
            cfg["bidirectional"] = True
        return cfg, eff

    def protocol_queries(self, dataset: str) -> list[dict[str, Any]]:
        state = self.states[dataset]
        records = state.manifest.by_id()
        rows = []
        for query_id in state.protocol.query_ids:
            record = records[query_id]
            rows.append(
                {
                    "image_id": query_id,
                    "subject_id": record.subject_id,
                    "eye": record.eye,
                    "pair_id": record.pair_id,
                    "category": record.category,
                    "relation_group": record.relation_group,
                }
            )
        return rows

    def _compute_inlier_overlay(
        self,
        state: DatasetState,
        query: ImageRecord,
        reference: ImageRecord,
        method_config: dict[str, Any],
        max_points: int = 48,
    ) -> list[dict[str, float]]:
        q_feat = state.repository.local(query, method_config)
        r_feat = state.repository.local(reference, method_config)
        src, dst = inlier_correspondences(q_feat, r_feat, method_config)
        if len(src) == 0:
            return []
        q_proc = state.repository.processed(query)
        r_proc = state.repository.processed(reference)
        q_inv = np.linalg.inv(q_proc.original_to_processed)
        r_inv = np.linalg.inv(r_proc.original_to_processed)
        q_orig = (
            np.column_stack([src, np.ones(len(src))]) @ q_inv.T
        )[:, :2]
        r_orig = (
            np.column_stack([dst, np.ones(len(dst))]) @ r_inv.T
        )[:, :2]
        qw, qh = q_proc.original_size
        rw, rh = r_proc.original_size
        step = max(1, len(src) // max_points)
        points: list[dict[str, float]] = []
        for idx in range(0, len(src), step):
            if len(points) >= max_points:
                break
            points.append(
                {
                    "qx": round(float(np.clip(q_orig[idx, 0] / max(1, qw), 0.0, 1.0)), 4),
                    "qy": round(float(np.clip(q_orig[idx, 1] / max(1, qh), 0.0, 1.0)), 4),
                    "rx": round(float(np.clip(r_orig[idx, 0] / max(1, rw), 0.0, 1.0)), 4),
                    "ry": round(float(np.clip(r_orig[idx, 1] / max(1, rh), 0.0, 1.0)), 4),
                }
            )
        return points

    def identify_record(
        self,
        dataset: str,
        query: ImageRecord,
        method: str,
        *,
        protocol_name: str | None = None,
    ) -> dict[str, Any]:
        state = self.states[dataset]
        method_config, threshold = self.method_setting(method)
        repository = state.repository
        gallery = state.gallery
        placeholder = (
            state.protocol
            if protocol_name is None
            else self.protocols[protocol_name]
        )
        allowed_ids = (
            placeholder.allowed_gallery(query.image_id)
            if query.image_id in placeholder.query_ids
            else tuple(record.image_id for record in gallery)
        )
        started = time.perf_counter()
        if method in ("sift", "sift_bidirectional"):
            query_feature = repository.local(query, method_config)
            pairs = {
                record.image_id: compare_local(
                    query_feature,
                    repository.local(record, method_config),
                    method_config,
                )
                for record in gallery
            }
        else:
            query_feature = repository.global_feature(query, method_config)
            pairs = {
                record.image_id: compare_global(
                    query_feature,
                    repository.global_feature(record, method_config),
                )
                for record in gallery
            }
        elapsed_ms = (time.perf_counter() - started) * 1000
        results = {
            (query.image_id, reference_id): result
            for reference_id, result in pairs.items()
        }
        prediction = retrieve(
            query.image_id,
            allowed_ids,
            results,
            placeholder,
            threshold,
            elapsed_ms=elapsed_ms,
        )
        best_pair = (
            pairs.get(prediction.reference_id)
            if prediction.reference_id is not None
            else None
        )
        inliers_overlay: list[dict[str, float]] = []
        if (
            method in ("sift", "sift_bidirectional")
            and prediction.reference_id is not None
        ):
            ref_record = state.manifest.by_id().get(prediction.reference_id)
            if ref_record is not None:
                inliers_overlay = self._compute_inlier_overlay(
                    state, query, ref_record, method_config
                )
        return {
            "dataset": dataset,
            "protocol": placeholder.name,
            "method": method,
            "threshold": threshold,
            "status": prediction.status,
            "accepted": prediction.accepted,
            "reference_id": prediction.reference_id,
            "score": prediction.score,
            "second_score": prediction.second_score,
            "score_margin": prediction.score_margin,
            "top_candidates": [
                {"image_id": image_id, "score": score}
                for image_id, score in prediction.top_candidates
            ],
            "elapsed_ms": elapsed_ms,
            "gallery_size": len(allowed_ids),
            "pair_stats": (
                None
                if best_pair is None
                else {
                    "n_keypoints_q": best_pair.n_keypoints_q,
                    "n_keypoints_r": best_pair.n_keypoints_r,
                    "n_ratio_matches": best_pair.n_ratio_matches,
                    "n_inliers": best_pair.n_inliers,
                    "inlier_ratio": best_pair.inlier_ratio,
                    "median_reprojection_error": best_pair.median_reprojection_error,
                }
            ),
            "inlier_correspondences": inliers_overlay,
        }

    def identify_protocol_query(
        self,
        dataset: str,
        image_id: str,
        method: str,
        *,
        protocol_name: str | None = None,
    ) -> dict[str, Any]:
        state = self.states[dataset]
        records = state.manifest.by_id()
        protocol = (
            state.protocol
            if protocol_name is None
            else self.protocols[protocol_name]
        )
        if image_id not in protocol.query_ids:
            raise KeyError(f"{image_id} is not a query in the {dataset} protocol")
        result = self.identify_record(
            dataset,
            records[image_id],
            method,
            protocol_name=protocol.name,
        )
        positives = protocol.positive_reference_ids.get(image_id, frozenset())
        confirmed = protocol.confirmed_relation_reference_ids.get(
            image_id, frozenset()
        )
        ref_id = result["reference_id"]
        top1_pos = bool(ref_id is not None and ref_id in positives)
        acc_pos = bool(result["accepted"] and top1_pos)
        result["query_id"] = image_id
        result["positive_reference_ids"] = sorted(positives)
        result["top1_correct"] = top1_pos
        result["designated_correct"] = acc_pos
        if confirmed:
            top1_conf = bool(ref_id is not None and ref_id in confirmed)
            acc_conf = bool(result["accepted"] and top1_conf)
            result["top1_confirmed_same_eye_correct"] = top1_conf
            result["confirmed_same_eye_correct"] = acc_conf
            result["confirmed_reference_ids"] = sorted(confirmed)
            result["same_eye_status"] = (
                "confirmed_same_eye" if top1_conf else "same_eye_unconfirmed"
            )
        else:
            result["same_eye_status"] = "correct" if top1_pos else "incorrect"
        if (
            result["status"] != "accepted"
            or not result["designated_correct"]
            or result.get("same_eye_status") == "same_eye_unconfirmed"
        ):
            err_cat, err_note = _classify_error_row(result, records, dataset)
            result["error_category"] = err_cat
            result["diagnostic_note"] = err_note
        return result

    def identify_upload(
        self, dataset: str, data: bytes, method: str
    ) -> dict[str, Any]:
        if not data:
            raise ValueError("Uploaded file is empty")
        from .utils import decode_rgb_bytes

        rgb = decode_rgb_bytes(data)
        query = ImageRecord(
            image_id="__upload__",
            dataset=dataset,  # type: ignore[arg-type]
            image_path="",
            content_sha256=sha256_bytes(data),
            pixel_sha256=pixel_sha256_rgb(rgb),
            role="query",
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        )
        state = self.states[dataset]
        if dataset == "fire":
            query = ImageRecord(
                **{**query.to_dict(), "mask_path": state.gallery[0].mask_path}
            )
        state.repository.register_array(query, rgb)
        return self.identify_record(dataset, query, method)

    def image_path(self, dataset: str, image_id: str) -> Path:
        state = self.states[dataset]
        record = state.manifest.by_id().get(image_id)
        if record is None:
            raise KeyError(f"Unknown image: {image_id}")
        return state.manifest.resolve(record)
