from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .features.repository import FeatureRepository
from .matching import compare_global, compare_local
from .types import PairResult, RetrievalProtocol


def select_threshold(
    pair_results: dict[tuple[str, str], PairResult],
    protocol: RetrievalProtocol,
    target_far: float,
) -> dict[str, Any]:
    """Pick the lowest threshold whose empirical FAR stays within target_far.

    Invalid pairs stay in the denominators: an invalid impostor is never
    accepted (correct rejection), and an invalid genuine pair is a false
    rejection. Dropping them would inflate both FAR and FRR's basis.
    """
    genuine_valid: list[float] = []
    impostor_valid: list[float] = []
    genuine_total = 0
    impostor_total = 0
    for query_id in protocol.query_ids:
        allowed = set(protocol.allowed_gallery(query_id))
        for reference_id in protocol.gallery_ids:
            if reference_id not in allowed:
                continue
            result = pair_results.get((query_id, reference_id))
            if result is None:
                raise ValueError(f"Missing pair result for {(query_id, reference_id)}")
            usable = result.valid
            if protocol.is_positive(query_id, reference_id):
                genuine_total += 1
                if usable:
                    genuine_valid.append(result.score)
            else:
                impostor_total += 1
                if usable:
                    impostor_valid.append(result.score)
    if genuine_total == 0:
        raise ValueError("Calibration requires at least one valid genuine score")
    genuine_array = np.asarray(genuine_valid, dtype=np.float64)
    impostor_array = np.sort(np.asarray(impostor_valid, dtype=np.float64))[::-1]
    allowed_false = int(math.floor(target_far * impostor_total))
    if allowed_false >= len(impostor_array):
        threshold = -math.inf
    else:
        cutoff = impostor_array[allowed_false]
        threshold = float(np.nextafter(cutoff, math.inf))
    false_accepts = int(np.sum(impostor_array >= threshold))
    accepted_genuine = int(np.sum(genuine_array >= threshold))
    false_rejects = genuine_total - accepted_genuine
    return {
        "threshold": threshold,
        "target_far": target_far,
        "empirical_far": false_accepts / impostor_total if impostor_total else 0.0,
        "empirical_frr": false_rejects / genuine_total,
        "genuine_pairs": genuine_total,
        "impostor_pairs": impostor_total,
        "valid_genuine_pairs": len(genuine_valid),
        "valid_impostor_pairs": len(impostor_valid),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def _matching_ms(pair_results: dict[tuple[str, str], PairResult]) -> float:
    """Total matcher time; excludes feature extraction, which is cache-dependent."""
    return float(sum(result.match_time_ms for result in pair_results.values()))


def compute_pair_results(
    protocol: RetrievalProtocol,
    repository: FeatureRepository,
    *,
    method: str,
    config: dict[str, Any],
    progress: Callable[[int, int], None] | None = None,
) -> dict[tuple[str, str], PairResult]:
    query_ids = protocol.query_ids
    gallery_ids = protocol.gallery_ids
    if method in ("sift", "sift_bidirectional"):
        effective_config = (
            {**config, "bidirectional": True}
            if method == "sift_bidirectional"
            else config
        )
        local_by_id = {
            image_id: repository.local(
                repository.record_by_id[image_id],
                effective_config,
            )
            for image_id in sorted(set(query_ids) | set(gallery_ids))
        }
        compare = lambda q, r: compare_local(q, r, effective_config)
    elif method == "hog":
        global_by_id = {
            image_id: repository.global_feature(
                repository.record_by_id[image_id],
                config,
            )
            for image_id in sorted(set(query_ids) | set(gallery_ids))
        }
        compare = lambda q, r: compare_global(q, r)
    else:
        raise ValueError(f"Unknown method: {method}")

    results: dict[tuple[str, str], PairResult] = {}
    shared: dict[tuple[str, str], PairResult] = {}
    total = len(query_ids) * len(gallery_ids)
    completed = 0
    for query_id in query_ids:
        query_record = repository.record_by_id[query_id]
        for reference_id in gallery_ids:
            reference_record = repository.record_by_id[reference_id]
            content_key = (
                query_record.pixel_sha256,
                reference_record.pixel_sha256,
            )
            result = shared.get(content_key)
            if result is None:
                if method in ("sift", "sift_bidirectional"):
                    result = compare(
                        local_by_id[query_id], local_by_id[reference_id]
                    )
                else:
                    result = compare(
                        global_by_id[query_id], global_by_id[reference_id]
                    )
                shared[content_key] = result
            results[(query_id, reference_id)] = result
            completed += 1
            if progress is not None:
                progress(completed, total)
    return results


def top1_correct_count(
    pair_results: dict[tuple[str, str], PairResult],
    protocol: RetrievalProtocol,
) -> int:
    correct = 0
    for query_id in protocol.query_ids:
        best: tuple[tuple[float, float, float, str], str] | None = None
        for reference_id in protocol.allowed_gallery(query_id):
            result = pair_results.get((query_id, reference_id))
            if result is None or not result.valid:
                continue
            candidate = (result.rank_order_key(reference_id), reference_id)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is not None and protocol.is_positive(query_id, best[1]):
            correct += 1
    return correct


def calibrate_sift(
    proxy_protocol: RetrievalProtocol,
    repository: FeatureRepository,
    calibration_config: dict[str, Any],
    base_config: dict[str, Any],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    ratios = calibration_config["sift_ratios"]
    residuals = calibration_config["sift_ransac_px"]
    for ratio in ratios:
        for residual in residuals:
            config = {
                **base_config,
                "ratio": float(ratio),
                "ransac_px": float(residual),
                "min_matches": 8,
                "min_inliers": 8,
                "bidirectional": False,
            }
            results = compute_pair_results(
                proxy_protocol,
                repository,
                method="sift",
                config=config,
                progress=progress,
            )
            threshold = select_threshold(
                results,
                proxy_protocol,
                float(calibration_config["target_far"]),
            )
            candidates.append(
                {
                    "config": config,
                    "top1_correct": top1_correct_count(results, proxy_protocol),
                    "elapsed_ms": _matching_ms(results),
                    "threshold": threshold,
                }
            )

    def selection_key(candidate: dict[str, Any]) -> tuple:
        serialized = repr(sorted(candidate["config"].items()))
        return (
            -candidate["top1_correct"],
            candidate["elapsed_ms"],
            serialized,
        )

    base_winner = min(candidates, key=selection_key)
    inlier_candidates: list[dict[str, Any]] = []
    for minimum in calibration_config["sift_min_inliers"]:
        config = {
            **base_winner["config"],
            "min_inliers": int(minimum),
            "min_matches": 8,
        }
        results = compute_pair_results(
            proxy_protocol,
            repository,
            method="sift",
            config=config,
            progress=progress,
        )
        threshold = select_threshold(
            results,
            proxy_protocol,
            float(calibration_config["target_far"]),
        )
        inlier_candidates.append(
            {
                "config": config,
                "top1_correct": top1_correct_count(results, proxy_protocol),
                "elapsed_ms": _matching_ms(results),
                "threshold": threshold,
            }
        )
    unidirectional_winner = min(inlier_candidates, key=selection_key)

    bidirectional_config = {
        **unidirectional_winner["config"],
        "bidirectional": True,
    }
    bidirectional_results = compute_pair_results(
        proxy_protocol,
        repository,
        method="sift",
        config=bidirectional_config,
        progress=progress,
    )
    bidirectional_threshold = select_threshold(
        bidirectional_results,
        proxy_protocol,
        float(calibration_config["target_far"]),
    )
    bidirectional = {
        "config": bidirectional_config,
        "top1_correct": top1_correct_count(
            bidirectional_results, proxy_protocol
        ),
        "elapsed_ms": _matching_ms(bidirectional_results),
        "threshold": bidirectional_threshold,
    }
    improvement = (
        bidirectional["top1_correct"]
        > unidirectional_winner["top1_correct"]
    )
    selected = bidirectional if improvement else unidirectional_winner
    return {
        "method": "sift",
        "search_candidates": candidates,
        "min_inlier_candidates": inlier_candidates,
        "selected_base": unidirectional_winner,
        "bidirectional_candidate": bidirectional,
        "bidirectional_adopted": improvement,
        "selected": selected,
    }


def calibrate_hog(
    proxy_protocol: RetrievalProtocol,
    repository: FeatureRepository,
    calibration_config: dict[str, Any],
    base_config: dict[str, Any],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for cell in calibration_config["hog_cells"]:
        config = {
            **base_config,
            "pixels_per_cell": [int(cell), int(cell)],
        }
        results = compute_pair_results(
            proxy_protocol,
            repository,
            method="hog",
            config=config,
            progress=progress,
        )
        threshold = select_threshold(
            results,
            proxy_protocol,
            float(calibration_config["target_far"]),
        )
        candidates.append(
            {
                "config": config,
                "top1_correct": top1_correct_count(results, proxy_protocol),
                "elapsed_ms": _matching_ms(results),
                "threshold": threshold,
            }
        )
    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate["top1_correct"],
            candidate["elapsed_ms"],
            repr(sorted(candidate["config"].items())),
        ),
    )
    return {
        "method": "hog",
        "search_candidates": candidates,
        "selected": selected,
    }

