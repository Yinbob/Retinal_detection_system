from __future__ import annotations

from collections import defaultdict

from ..manifest import finalize_protocol
from ..types import DatasetManifest, ImageRecord, RetrievalProtocol


def _by_id(records: tuple[ImageRecord, ...]) -> dict[str, ImageRecord]:
    return {record.image_id: record for record in records}


def build_vistel_course_protocol(
    manifest: DatasetManifest,
) -> RetrievalProtocol:
    queries = [
        record.image_id for record in manifest.records if record.role == "query"
    ]
    gallery = [
        record.image_id for record in manifest.records if record.role == "gallery"
    ]
    gallery_by_identity: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        if record.role == "gallery" and record.identity_id is not None:
            gallery_by_identity[record.identity_id].add(record.image_id)
    positives = {}
    for query_id in queries:
        query = _by_id(manifest.records)[query_id]
        positives[query_id] = gallery_by_identity[query.identity_id or ""]
    return finalize_protocol(
        name="vistel_course",
        dataset="vistel",
        query_ids=queries,
        gallery_ids=gallery,
        positive_reference_ids=positives,
        label_semantics="same_subject_same_eye",
    )


def build_vistel_proxy_protocol(
    manifest: DatasetManifest,
) -> RetrievalProtocol:
    by_identity: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in manifest.records:
        if record.role == "gallery" and record.identity_id is not None:
            by_identity[record.identity_id].append(record)

    query_ids: list[str] = []
    gallery_ids: set[str] = set()
    proxies: set[str] = set()
    for identity, records in sorted(by_identity.items()):
        if len(records) < 2:
            continue
        ordered = sorted(
            records, key=lambda record: (record.sample_index or 0, record.image_id)
        )
        proxy = ordered[0]
        query_ids.append(proxy.image_id)
        proxies.add(proxy.image_id)
        gallery_ids.update(record.image_id for record in ordered[1:])

    for record in manifest.records:
        if record.role == "gallery" and record.image_id not in proxies:
            gallery_ids.add(record.image_id)

    identity_by_query = {
        record.image_id: record.identity_id
        for record in manifest.records
        if record.image_id in query_ids
    }
    identity_by_gallery: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        if record.image_id in gallery_ids and record.identity_id is not None:
            identity_by_gallery[record.identity_id].add(record.image_id)
    positives = {
        query_id: identity_by_gallery[identity_by_query[query_id] or ""]
        for query_id in query_ids
    }
    protocol = finalize_protocol(
        name="vistel_proxy",
        dataset="vistel",
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        positive_reference_ids=positives,
        label_semantics="same_subject_same_eye",
    )
    if (len(protocol.query_ids), len(protocol.gallery_ids)) != (33, 125):
        raise ValueError(
            "Unexpected Vistel proxy split: "
            f"{len(protocol.query_ids)}/{len(protocol.gallery_ids)}"
        )
    return protocol


def build_fire_protocols(
    manifest: DatasetManifest,
) -> tuple[RetrievalProtocol, RetrievalProtocol]:
    by_id = _by_id(manifest.records)
    query_records = [
        record for record in manifest.records if record.role == "query"
    ]
    gallery_records = [
        record for record in manifest.records if record.role == "gallery"
    ]
    gallery_by_hash: dict[str, set[str]] = defaultdict(set)
    gallery_by_relation: dict[str, set[str]] = defaultdict(set)
    for record in gallery_records:
        gallery_by_hash[record.content_sha256].add(record.image_id)
        if record.relation_group is not None:
            gallery_by_relation[record.relation_group].add(record.image_id)

    designated: dict[str, set[str]] = {}
    confirmed: dict[str, set[str]] = {}
    excluded: dict[str, set[str]] = {}
    for query in query_records:
        pair_name = (query.pair_id or "").split(":")[-1]
        designated_ref = by_id[f"{pair_name}_1"]
        if (
            by_id[designated_ref.image_id].content_sha256
            == query.content_sha256
        ):
            raise ValueError(f"Self-paired FIRE positive: {query.image_id}")
        designated[query.image_id] = set(
            gallery_by_hash[designated_ref.content_sha256]
        )
        confirmed[query.image_id] = set(
            gallery_by_relation[query.relation_group or ""]
        )
        excluded[query.image_id] = set(gallery_by_hash[query.content_sha256])
        if not designated[query.image_id] - excluded[query.image_id]:
            raise ValueError(
                f"FIRE query lacks non-self positive: {query.image_id}"
            )

    original = finalize_protocol(
        name="fire_pair",
        dataset="fire",
        query_ids=[record.image_id for record in query_records],
        gallery_ids=[record.image_id for record in gallery_records],
        positive_reference_ids=designated,
        confirmed_relation_reference_ids=confirmed,
        label_semantics="designated_pair_and_confirmed_relation",
    )
    excluded_protocol = finalize_protocol(
        name="fire_pair_exclude_self",
        dataset="fire",
        query_ids=[record.image_id for record in query_records],
        gallery_ids=[record.image_id for record in gallery_records],
        positive_reference_ids=designated,
        excluded_reference_ids=excluded,
        confirmed_relation_reference_ids=confirmed,
        label_semantics="designated_pair_and_confirmed_relation",
    )
    blocked = sum(
        len(value)
        for value in excluded_protocol.excluded_reference_ids.values()
    )
    if blocked != 110:
        raise ValueError(f"Expected 110 blocked FIRE combinations, got {blocked}")
    for protocol in (original, excluded_protocol):
        for query_id in protocol.query_ids:
            allowed = set(protocol.allowed_gallery(query_id))
            if not allowed.intersection(protocol.positive_reference_ids[query_id]):
                raise ValueError(f"No allowed FIRE positive for {query_id}")
    return original, excluded_protocol

