from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import (
    PreprocessConfig,
    preprocess_record,
    transform_points,
    transform_to_original,
)
from .types import DatasetManifest, PairResult, ProcessedImage, RetrievalProtocol


def read_control_points(path: str | Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected N x 4 FIRE control points, got {points.shape}")
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"Invalid FIRE control points: {path}")
    return points


def map_reference_points_to_query(
    reference_points: np.ndarray,
    query_to_reference: np.ndarray,
    reference_processed: ProcessedImage,
    query_processed: ProcessedImage,
) -> np.ndarray:
    reference_in_processed = transform_points(
        reference_processed.original_to_processed,
        reference_points,
    )
    inverse = np.linalg.inv(query_to_reference)
    query_in_processed = transform_points(inverse, reference_in_processed)
    return transform_to_original(query_processed, query_in_processed)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if len(values) == 0:
        return None
    return float(np.percentile(values, percentile))


def analyze_fire_registration(
    manifest: DatasetManifest,
    protocol: RetrievalProtocol,
    pair_results: dict[tuple[str, str], PairResult],
    preprocess_config: PreprocessConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = manifest.by_id()
    processed_by_hash: dict[str, ProcessedImage] = {}
    rows: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []

    def processed(record):
        cached = processed_by_hash.get(record.pixel_sha256)
        if cached is None:
            cached = preprocess_record(record, manifest.root, preprocess_config)
            processed_by_hash[record.pixel_sha256] = cached
        return cached

    for query_id in protocol.query_ids:
        query = by_id[query_id]
        pair_name = (query.pair_id or "").split(":")[-1]
        positive_ids = sorted(
            protocol.positive_reference_ids.get(query_id, frozenset())
        )
        reference_id = positive_ids[0] if positive_ids else None
        row: dict[str, Any] = {
            "pair_id": pair_name,
            "category": query.category,
            "query_id": query_id,
            "reference_id": reference_id,
            "status": "evaluated",
            "n_points": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "max_error_px": None,
        }
        if pair_name == "P37":
            row["status"] = "excluded_p37"
            rows.append(row)
            continue
        if reference_id is None:
            row["status"] = "missing_positive_reference"
            rows.append(row)
            continue

        result = pair_results.get((query_id, reference_id))
        if result is None or not result.valid or result.transform is None:
            row["status"] = "transform_unavailable"
            row["reason"] = None if result is None else result.reason
            rows.append(row)
            continue

        control_path = (
            Path(manifest.root)
            / "Ground Truth"
            / f"control_points_{pair_name}_1_2.txt"
        )
        try:
            control_points = read_control_points(control_path)
            reference = by_id[reference_id]
            predicted_query = map_reference_points_to_query(
                control_points[:, :2],
                np.asarray(result.transform, dtype=np.float64),
                processed(reference),
                processed(query),
            )
            errors = np.linalg.norm(
                predicted_query - control_points[:, 2:4],
                axis=1,
            )
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            row["status"] = "analysis_failed"
            row["reason"] = str(error)
            rows.append(row)
            continue

        all_errors.append(errors)
        row.update(
            {
                "n_points": len(errors),
                "mean_error_px": float(np.mean(errors)),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(np.max(errors)),
            }
        )
        rows.append(row)

    values = np.concatenate(all_errors) if all_errors else np.empty(0)
    evaluated_rows = [row for row in rows if row["status"] == "evaluated"]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "ground_truth_pairs": len(rows),
        "excluded_pairs": ["P37"],
        "eligible_pairs": len(rows) - 1,
        "evaluated_pairs": len(evaluated_rows),
        "failed_pairs": len(rows) - 1 - len(evaluated_rows),
        "control_points": int(len(values)),
        "mean_error_px": float(np.mean(values)) if len(values) else None,
        "median_error_px": float(np.median(values)) if len(values) else None,
        "p95_error_px": _percentile(values, 95),
        "max_error_px": float(np.max(values)) if len(values) else None,
        "within_1px": float(np.mean(values <= 1.0)) if len(values) else None,
        "within_3px": float(np.mean(values <= 3.0)) if len(values) else None,
        "within_5px": float(np.mean(values <= 5.0)) if len(values) else None,
        "status_counts": dict(sorted(status_counts.items())),
    }
    return summary, rows
