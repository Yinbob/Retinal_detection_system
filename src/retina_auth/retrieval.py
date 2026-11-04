from __future__ import annotations

from collections.abc import Mapping

from .types import PairResult, Prediction, RetrievalProtocol


def retrieve(
    query_id: str,
    gallery_ids: tuple[str, ...],
    pair_results: Mapping[tuple[str, str], PairResult],
    protocol: RetrievalProtocol,
    threshold: float | None,
    elapsed_ms: float = 0.0,
) -> Prediction:
    candidates: list[tuple[tuple[float, float, float], str, PairResult]] = []
    for reference_id in gallery_ids:
        result = pair_results.get((query_id, reference_id))
        if result is None or not result.valid:
            continue
        candidates.append((result.rank_key, reference_id, result))
    candidates.sort(key=lambda item: item[2].rank_order_key(item[1]))
    top_candidates = tuple(
        (reference_id, result.score)
        for _, reference_id, result in candidates[:5]
    )
    positives = protocol.positive_reference_ids.get(query_id, frozenset())
    positive_id = sorted(positives)[0] if positives else None
    if not candidates:
        return Prediction(
            query_id=query_id,
            reference_id=None,
            score=None,
            accepted=False,
            status="no_valid_candidate",
            positive_reference_id=positive_id,
            same_eye_status="failure",
            elapsed_ms=elapsed_ms,
            top_candidates=top_candidates,
            second_score=None,
            score_margin=None,
            top1_correct=False,
            correct_with_reject=False,
        )

    _, best_id, best = candidates[0]
    second_score = candidates[1][2].score if len(candidates) > 1 else None
    score_margin = (
        float(best.score - second_score)
        if second_score is not None
        else float(best.score)
    )
    accepted = threshold is None or best.score >= threshold
    if not accepted:
        status = "rejected"
    else:
        status = "accepted"
    is_pos = protocol.is_positive(query_id, best_id)
    if protocol.confirmed_relation_reference_ids:
        same_eye_status = (
            "confirmed_same_eye"
            if protocol.is_confirmed_relation(query_id, best_id)
            else "same_eye_unconfirmed"
        )
    else:
        same_eye_status = (
            "correct" if is_pos else "incorrect"
        )
    return Prediction(
        query_id=query_id,
        reference_id=best_id,
        score=best.score,
        accepted=accepted,
        status=status,
        positive_reference_id=positive_id,
        same_eye_status=same_eye_status,
        elapsed_ms=elapsed_ms,
        top_candidates=top_candidates,
        second_score=second_score,
        score_margin=score_margin,
        top1_correct=is_pos,
        correct_with_reject=bool(accepted and is_pos),
    )

