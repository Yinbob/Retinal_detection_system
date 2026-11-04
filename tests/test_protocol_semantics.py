from __future__ import annotations

from pathlib import Path

from retina_auth.experiments import load_artifact_state


ROOT = Path(__file__).resolve().parents[1]


def test_vistel_requires_same_subject_and_eye() -> None:
    manifests, protocols = load_artifact_state(ROOT / "artifacts")
    manifest = manifests["vistel"]
    protocol = protocols["vistel_course"]
    by_id = manifest.by_id()

    query = by_id[protocol.query_ids[0]]
    same_eye = [
        record
        for record in manifest.records
        if record.role == "gallery"
        and record.subject_id == query.subject_id
        and record.eye == query.eye
    ]
    other_eye = [
        record
        for record in manifest.records
        if record.role == "gallery"
        and record.subject_id == query.subject_id
        and record.eye != query.eye
    ]

    assert same_eye
    assert other_eye
    assert all(protocol.is_positive(query.image_id, record.image_id) for record in same_eye)
    assert all(
        not protocol.is_positive(query.image_id, record.image_id)
        for record in other_eye
    )


def test_fire_positive_aliases_share_designated_content() -> None:
    manifests, protocols = load_artifact_state(ROOT / "artifacts")
    manifest = manifests["fire"]
    protocol = protocols["fire_pair"]
    by_id = manifest.by_id()

    for query_id in protocol.query_ids:
        positive_ids = protocol.positive_reference_ids[query_id]
        assert positive_ids
        hashes = {by_id[reference_id].content_sha256 for reference_id in positive_ids}
        assert len(hashes) == 1
        assert by_id[query_id].content_sha256 not in hashes


def test_fire_excluding_self_copies_preserves_every_positive() -> None:
    _, protocols = load_artifact_state(ROOT / "artifacts")
    original = protocols["fire_pair"]
    excluded = protocols["fire_pair_exclude_self"]

    assert original.query_ids == excluded.query_ids
    assert original.gallery_ids == excluded.gallery_ids
    assert sum(
        len(values) for values in excluded.excluded_reference_ids.values()
    ) == 110
    for query_id in excluded.query_ids:
        allowed = set(excluded.allowed_gallery(query_id))
        positives = set(excluded.positive_reference_ids[query_id])
        assert positives & allowed


def test_fire_cross_group_has_no_claimed_negative_label() -> None:
    manifests, protocols = load_artifact_state(ROOT / "artifacts")
    manifest = manifests["fire"]
    protocol = protocols["fire_pair"]
    by_id = manifest.by_id()
    query_id = protocol.query_ids[0]

    other_group_id = next(
        reference_id
        for reference_id in protocol.gallery_ids
        if by_id[reference_id].relation_group != by_id[query_id].relation_group
    )
    assert not protocol.is_confirmed_relation(query_id, other_group_id)
