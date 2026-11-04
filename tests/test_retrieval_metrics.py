from __future__ import annotations

from retina_auth.calibration import select_threshold, top1_correct_count
from retina_auth.manifest import finalize_protocol
from retina_auth.metrics import evaluate
from retina_auth.retrieval import retrieve
from retina_auth.types import ImageRecord, PairResult, Prediction


def pair_result(
    *,
    score: float,
    valid: bool = True,
    inlier_ratio: float = 0.0,
    error: float | None = None,
) -> PairResult:
    return PairResult(
        score=score,
        valid=valid,
        reason=None if valid else "invalid",
        n_keypoints_q=10,
        n_keypoints_r=10,
        n_ratio_matches=10,
        n_inliers=int(score),
        inlier_ratio=inlier_ratio,
        median_reprojection_error=error,
        transform=None,
        match_time_ms=1.0,
    )


def make_protocol(
    *,
    query_ids: tuple[str, ...] = ("q",),
    gallery_ids: tuple[str, ...] = ("a", "b"),
    positives: dict[str, set[str]] | None = None,
):
    positives = positives or {"q": {"a"}}
    return finalize_protocol(
        name="test",
        dataset="vistel",
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        positive_reference_ids=positives,
        label_semantics="same_subject_same_eye",
    )


def test_ties_are_deterministic_by_error_then_image_id() -> None:
    protocol = make_protocol(gallery_ids=("a", "b", "z"))
    results = {
        ("q", "a"): pair_result(score=8, inlier_ratio=0.8, error=1.0),
        ("q", "b"): pair_result(score=8, inlier_ratio=0.8, error=0.5),
        ("q", "z"): pair_result(score=8, inlier_ratio=0.8, error=0.5),
    }
    prediction = retrieve(
        "q", protocol.gallery_ids, results, protocol, threshold=None
    )
    assert prediction.reference_id == "b"
    assert top1_correct_count(results, protocol) == 0


def test_image_id_breaks_exact_tie_ascending() -> None:
    protocol = make_protocol(gallery_ids=("z", "a"))
    results = {
        ("q", "z"): pair_result(score=8, inlier_ratio=0.8, error=1.0),
        ("q", "a"): pair_result(score=8, inlier_ratio=0.8, error=1.0),
    }
    prediction = retrieve(
        "q", protocol.gallery_ids, results, protocol, threshold=None
    )
    assert prediction.reference_id == "a"
    assert top1_correct_count(results, protocol) == 1


def test_no_valid_candidate_is_a_failure_not_first_gallery_item() -> None:
    protocol = make_protocol()
    results = {
        ("q", "a"): pair_result(score=0, valid=False),
        ("q", "b"): pair_result(score=0, valid=False),
    }
    prediction = retrieve(
        "q", protocol.gallery_ids, results, protocol, threshold=None
    )
    assert prediction.reference_id is None
    assert prediction.status == "no_valid_candidate"
    assert prediction.same_eye_status == "failure"


def test_rejections_and_failures_stay_in_metric_denominator() -> None:
    protocol = make_protocol(
        query_ids=("q1", "q2", "q3"),
        gallery_ids=("a", "b"),
        positives={"q1": {"a"}, "q2": {"a"}, "q3": {"a"}},
    )
    predictions = [
        Prediction(
            query_id="q1",
            reference_id="a",
            score=10.0,
            accepted=True,
            status="accepted",
            positive_reference_id="a",
            same_eye_status="correct",
            elapsed_ms=1.0,
        ),
        Prediction(
            query_id="q2",
            reference_id="b",
            score=2.0,
            accepted=False,
            status="rejected",
            positive_reference_id="a",
            same_eye_status="incorrect",
            elapsed_ms=2.0,
        ),
        Prediction(
            query_id="q3",
            reference_id=None,
            score=None,
            accepted=False,
            status="no_valid_candidate",
            positive_reference_id="a",
            same_eye_status="failure",
            elapsed_ms=3.0,
        ),
    ]
    records = [
        ImageRecord(
            image_id=query_id,
            dataset="vistel",
            image_path=f"{query_id}.png",
            content_sha256=query_id,
            pixel_sha256=query_id,
            role="query",
        )
        for query_id in protocol.query_ids
    ]
    metrics, rows = evaluate(predictions, protocol, records=records)
    assert metrics["queries"] == 3
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 1 / 3
    assert metrics["rejected"] == 1
    assert metrics["failures"] == 1
    assert len(rows) == 3


def test_threshold_selection_reports_original_counts() -> None:
    protocol = make_protocol(
        query_ids=("q1", "q2"),
        gallery_ids=("a", "b"),
        positives={"q1": {"a"}, "q2": {"a"}},
    )
    results = {
        ("q1", "a"): pair_result(score=9),
        ("q1", "b"): pair_result(score=10),
        ("q2", "a"): pair_result(score=8),
        ("q2", "b"): pair_result(score=7),
    }
    threshold = select_threshold(results, protocol, target_far=0.5)
    assert threshold["genuine_pairs"] == 2
    assert threshold["impostor_pairs"] == 2
    assert threshold["false_accepts"] == 1
    assert threshold["false_rejects"] == 0


def test_invalid_impostors_stay_in_far_denominator() -> None:
    protocol = make_protocol()
    results = {
        ("q", "a"): pair_result(score=12, inlier_ratio=1.0, error=0.1),
        ("q", "b"): pair_result(score=0, valid=False),
    }
    threshold = select_threshold(results, protocol, target_far=0.01)
    # Only one impostor exists and it is invalid, so nothing can be falsely accepted.
    assert threshold["threshold"] == float("-inf")
    assert threshold["genuine_pairs"] == 1
    assert threshold["impostor_pairs"] == 1
    assert threshold["valid_impostor_pairs"] == 0
    assert threshold["false_accepts"] == 0
    assert threshold["empirical_far"] == 0.0


def test_invalid_genuine_pairs_count_as_false_rejects() -> None:
    protocol = make_protocol()
    results = {
        ("q", "a"): pair_result(score=0, valid=False),
        ("q", "b"): pair_result(score=5, valid=True),
    }
    threshold = select_threshold(results, protocol, target_far=0.5)
    assert threshold["genuine_pairs"] == 1
    assert threshold["valid_genuine_pairs"] == 0
    assert threshold["false_rejects"] == 1
    assert threshold["empirical_frr"] == 1.0


def test_far_uses_full_impostor_population_when_some_invalid() -> None:
    protocol = make_protocol(
        query_ids=("q",),
        gallery_ids=("a", "b", "c", "d"),
        positives={"q": {"a"}},
    )
    results = {
        ("q", "a"): pair_result(score=20),
        ("q", "b"): pair_result(score=3),
        ("q", "c"): pair_result(score=0, valid=False),
        ("q", "d"): pair_result(score=0, valid=False),
    }
    threshold = select_threshold(results, protocol, target_far=0.01)
    assert threshold["impostor_pairs"] == 3
    assert threshold["valid_impostor_pairs"] == 1
    assert threshold["genuine_pairs"] == 1
    assert threshold["false_accepts"] == 0
    assert threshold["false_rejects"] == 0
