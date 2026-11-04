from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


DatasetName = Literal["vistel", "fire"]
Role = Literal["query", "gallery"]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    dataset: DatasetName
    image_path: str
    content_sha256: str
    pixel_sha256: str
    role: Role
    subject_id: str | None = None
    eye: str | None = None
    pair_id: str | None = None
    category: str | None = None
    relation_group: str | None = None
    vessel_path: str | None = None
    mask_path: str | None = None
    sample_index: int | None = None
    width: int | None = None
    height: int | None = None
    zip_member: str | None = None

    @property
    def identity_id(self) -> str | None:
        if self.dataset != "vistel" or self.subject_id is None or self.eye is None:
            return None
        return f"vistel:{self.subject_id}:{self.eye}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["identity_id"] = self.identity_id
        return data


@dataclass(frozen=True)
class DatasetManifest:
    dataset: DatasetName
    root: str
    records: tuple[ImageRecord, ...]
    archive_sha256: str
    summary: dict[str, Any] = field(default_factory=dict)
    audit_date: str | None = None

    def by_id(self) -> dict[str, ImageRecord]:
        return {record.image_id: record for record in self.records}

    def resolve(self, record_or_path: ImageRecord | str) -> Path:
        relative = (
            record_or_path.image_path
            if isinstance(record_or_path, ImageRecord)
            else record_or_path
        )
        return Path(self.root) / relative

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "root": self.root,
            "archive_sha256": self.archive_sha256,
            "summary": self.summary,
            "audit_date": self.audit_date,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class RetrievalProtocol:
    name: str
    dataset: DatasetName
    query_ids: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    positive_reference_ids: dict[str, frozenset[str]]
    label_semantics: str
    protocol_hash: str
    excluded_reference_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    confirmed_relation_reference_ids: dict[str, frozenset[str]] = field(
        default_factory=dict
    )

    def allowed_gallery(self, query_id: str) -> tuple[str, ...]:
        excluded = self.excluded_reference_ids.get(query_id, frozenset())
        return tuple(ref for ref in self.gallery_ids if ref not in excluded)

    def is_positive(self, query_id: str, reference_id: str) -> bool:
        return reference_id in self.positive_reference_ids.get(query_id, frozenset())

    def is_confirmed_relation(
        self, query_id: str, reference_id: str
    ) -> bool:
        return reference_id in self.confirmed_relation_reference_ids.get(
            query_id, frozenset()
        )


@dataclass(frozen=True)
class ProcessedImage:
    gray: np.ndarray
    fov: np.ndarray
    original_to_processed: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    source_pixel_sha256: str
    crop_xywh: tuple[int, int, int, int]
    scale: float
    padding: tuple[int, int]


@dataclass(frozen=True)
class LocalFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class GlobalFeatures:
    vector: np.ndarray
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class PairResult:
    score: float
    valid: bool
    reason: str | None
    n_keypoints_q: int
    n_keypoints_r: int
    n_ratio_matches: int
    n_inliers: int
    inlier_ratio: float
    median_reprojection_error: float | None
    transform: tuple[tuple[float, ...], ...] | None
    match_time_ms: float

    @property
    def rank_key(self) -> tuple[float, float, float]:
        reprojection = (
            self.median_reprojection_error
            if self.median_reprojection_error is not None
            else float("inf")
        )
        return (self.score, self.inlier_ratio, -reprojection)

    def rank_order_key(self, image_id: str) -> tuple[float, float, float, str]:
        score, inlier_ratio, negative_error = self.rank_key
        return (-score, -inlier_ratio, -negative_error, image_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transform"] = (
            None if self.transform is None else [list(row) for row in self.transform]
        )
        return data


@dataclass(frozen=True)
class Prediction:
    query_id: str
    reference_id: str | None
    score: float | None
    accepted: bool
    status: str
    positive_reference_id: str | None
    same_eye_status: str
    elapsed_ms: float
    top_candidates: tuple[tuple[str, float], ...] = ()
    second_score: float | None = None
    score_margin: float | None = None
    top1_correct: bool | None = None
    correct_with_reject: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["top_candidates"] = [list(item) for item in self.top_candidates]
        return data

