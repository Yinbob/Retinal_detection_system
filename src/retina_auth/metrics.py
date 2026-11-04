from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from .types import ImageRecord, Prediction, RetrievalProtocol
from .utils import median_and_p95


def _group_metrics(
    predictions: list[Prediction],
    protocol: RetrievalProtocol,
) -> dict[str, Any]:
    total = len(predictions)
    top1_designated = sum(
        prediction.reference_id is not None
        and protocol.is_positive(prediction.query_id, prediction.reference_id)
        for prediction in predictions
    )
    designated_correct = sum(
        prediction.reference_id is not None
        and prediction.accepted
        and protocol.is_positive(prediction.query_id, prediction.reference_id)
        for prediction in predictions
    )
    rejected = sum(prediction.status == "rejected" for prediction in predictions)
    failures = sum(
        prediction.status == "no_valid_candidate" for prediction in predictions
    )
    elapsed = [prediction.elapsed_ms for prediction in predictions]
    median_ms, p95_ms = median_and_p95(elapsed)
    result: dict[str, Any] = {
        "queries": total,
        "correct": designated_correct,
        "accuracy": designated_correct / total if total else 0.0,
        "top1_correct": top1_designated,
        "top1_accuracy": top1_designated / total if total else 0.0,
        "rejected": rejected,
        "failures": failures,
        "status_counts": dict(
            sorted(Counter(prediction.status for prediction in predictions).items())
        ),
        "median_query_ms": median_ms,
        "p95_query_ms": p95_ms,
    }
    if protocol.confirmed_relation_reference_ids:
        top1_confirmed = sum(
            prediction.reference_id is not None
            and protocol.is_confirmed_relation(
                prediction.query_id, prediction.reference_id
            )
            for prediction in predictions
        )
        confirmed = sum(
            prediction.reference_id is not None
            and prediction.accepted
            and protocol.is_confirmed_relation(
                prediction.query_id, prediction.reference_id
            )
            for prediction in predictions
        )
        rejected_confirmed = top1_confirmed - confirmed
        unconfirmed = sum(
            prediction.same_eye_status == "same_eye_unconfirmed"
            for prediction in predictions
        )
        result.update(
            {
                "designated_pair_correct": designated_correct,
                "designated_pair_accuracy": (
                    designated_correct / total if total else 0.0
                ),
                "top1_designated_pair_correct": top1_designated,
                "top1_designated_pair_accuracy": (
                    top1_designated / total if total else 0.0
                ),
                "confirmed_same_eye_correct": confirmed,
                "confirmed_same_eye_accuracy": confirmed / total if total else 0.0,
                "top1_confirmed_same_eye_correct": top1_confirmed,
                "top1_confirmed_same_eye_accuracy": (
                    top1_confirmed / total if total else 0.0
                ),
                "rejected_confirmed_same_eye": rejected_confirmed,
                "same_eye_unconfirmed": unconfirmed,
                "same_eye_accuracy_is_lower_bound": True,
            }
        )
    return result


def evaluate(
    predictions: Iterable[Prediction],
    protocol: RetrievalProtocol,
    records: Iterable[ImageRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_list = sorted(predictions, key=lambda item: item.query_id)
    by_query = {prediction.query_id: prediction for prediction in prediction_list}
    missing = sorted(set(protocol.query_ids) - set(by_query))
    extra = sorted(set(by_query) - set(protocol.query_ids))
    if missing or extra:
        raise ValueError(f"Prediction/query mismatch: missing={missing}, extra={extra}")
    record_by_id = {record.image_id: record for record in records}
    metrics = _group_metrics(prediction_list, protocol)
    metrics.update(
        {
            "protocol": protocol.name,
            "protocol_hash": protocol.protocol_hash,
            "dataset": protocol.dataset,
            "label_semantics": protocol.label_semantics,
        }
    )

    category_predictions: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in prediction_list:
        category = record_by_id[prediction.query_id].category
        if category:
            category_predictions[category].append(prediction)
    if category_predictions:
        metrics["categories"] = {
            category: _group_metrics(items, protocol)
            for category, items in sorted(category_predictions.items())
        }

    rows: list[dict[str, Any]] = []
    for prediction in prediction_list:
        row = prediction.to_dict()
        is_pos = (
            prediction.reference_id is not None
            and protocol.is_positive(
                prediction.query_id, prediction.reference_id
            )
        )
        row.update(
            {
                "protocol": protocol.name,
                "top1_correct": bool(is_pos),
                "correct_with_reject": bool(prediction.accepted and is_pos),
                "designated_correct": bool(prediction.accepted and is_pos),
            }
        )
        if protocol.confirmed_relation_reference_ids:
            is_conf = (
                prediction.reference_id is not None
                and protocol.is_confirmed_relation(
                    prediction.query_id, prediction.reference_id
                )
            )
            row["top1_confirmed_same_eye_correct"] = bool(is_conf)
            row["confirmed_same_eye_correct"] = (
                bool(prediction.accepted and is_conf)
            )
        rows.append(row)
    return metrics, rows


def pair_score_summary(
    pair_results: dict[tuple[str, str], Any],
    protocol: RetrievalProtocol,
) -> dict[str, Any]:
    genuine: list[float] = []
    impostor: list[float] = []
    invalid = 0
    for query_id in protocol.query_ids:
        allowed = set(protocol.allowed_gallery(query_id))
        for reference_id in protocol.gallery_ids:
            result = pair_results.get((query_id, reference_id))
            if result is None or not result.valid:
                invalid += 1
                continue
            if reference_id not in allowed:
                continue
            if protocol.is_positive(query_id, reference_id):
                genuine.append(result.score)
            else:
                impostor.append(result.score)
    return {
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "invalid_or_blocked_pairs": invalid,
        "genuine_median": float(np.median(genuine)) if genuine else None,
        "impostor_median": float(np.median(impostor)) if impostor else None,
    }

