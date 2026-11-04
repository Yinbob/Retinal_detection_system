from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .types import DatasetManifest, ImageRecord, RetrievalProtocol
from .utils import stable_hash, write_json


RECORD_FIELDS = [
    "image_id",
    "dataset",
    "image_path",
    "content_sha256",
    "pixel_sha256",
    "role",
    "subject_id",
    "eye",
    "identity_id",
    "pair_id",
    "category",
    "relation_group",
    "vessel_path",
    "mask_path",
    "sample_index",
    "width",
    "height",
    "zip_member",
]


def write_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    write_json(path, manifest.to_dict())
    csv_path = Path(path).with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in manifest.records:
            writer.writerow(record.to_dict())


def load_manifest(path: str | Path) -> DatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        ImageRecord(
            **{
                key: item.get(key)
                for key in asdict(ImageRecord(
                    image_id="",
                    dataset="vistel",
                    image_path="",
                    content_sha256="",
                    pixel_sha256="",
                    role="gallery",
                )).keys()
                if key in item
            }
        )
        for item in data["records"]
    )
    return DatasetManifest(
        dataset=data["dataset"],
        root=data["root"],
        records=records,
        archive_sha256=data["archive_sha256"],
        summary=data.get("summary", {}),
        audit_date=data.get("audit_date"),
    )


def protocol_payload(protocol: RetrievalProtocol) -> dict:
    return {
        "name": protocol.name,
        "dataset": protocol.dataset,
        "query_ids": list(protocol.query_ids),
        "gallery_ids": list(protocol.gallery_ids),
        "positive_reference_ids": {
            key: sorted(value)
            for key, value in sorted(protocol.positive_reference_ids.items())
        },
        "confirmed_relation_reference_ids": {
            key: sorted(value)
            for key, value in sorted(
                protocol.confirmed_relation_reference_ids.items()
            )
        },
        "excluded_reference_ids": {
            key: sorted(value)
            for key, value in sorted(protocol.excluded_reference_ids.items())
        },
        "label_semantics": protocol.label_semantics,
    }


def finalize_protocol(
    *,
    name: str,
    dataset: str,
    query_ids: Iterable[str],
    gallery_ids: Iterable[str],
    positive_reference_ids: dict[str, Iterable[str]],
    label_semantics: str,
    excluded_reference_ids: dict[str, Iterable[str]] | None = None,
    confirmed_relation_reference_ids: dict[str, Iterable[str]] | None = None,
) -> RetrievalProtocol:
    query_tuple = tuple(sorted(set(query_ids)))
    gallery_tuple = tuple(sorted(set(gallery_ids)))
    positive = {
        query: frozenset(sorted(values))
        for query, values in sorted(positive_reference_ids.items())
    }
    excluded = {
        query: frozenset(sorted(values))
        for query, values in sorted((excluded_reference_ids or {}).items())
        if values
    }
    confirmed = {
        query: frozenset(sorted(values))
        for query, values in sorted(
            (confirmed_relation_reference_ids or {}).items()
        )
        if values
    }
    draft = RetrievalProtocol(
        name=name,
        dataset=dataset,  # type: ignore[arg-type]
        query_ids=query_tuple,
        gallery_ids=gallery_tuple,
        positive_reference_ids=positive,
        label_semantics=label_semantics,
        protocol_hash="",
        excluded_reference_ids=excluded,
        confirmed_relation_reference_ids=confirmed,
    )
    protocol_hash = stable_hash(protocol_payload(draft))
    return RetrievalProtocol(
        name=draft.name,
        dataset=draft.dataset,
        query_ids=draft.query_ids,
        gallery_ids=draft.gallery_ids,
        positive_reference_ids=draft.positive_reference_ids,
        label_semantics=draft.label_semantics,
        protocol_hash=protocol_hash,
        excluded_reference_ids=draft.excluded_reference_ids,
        confirmed_relation_reference_ids=draft.confirmed_relation_reference_ids,
    )


def write_protocol(protocol: RetrievalProtocol, path: str | Path) -> None:
    payload = protocol_payload(protocol)
    payload["protocol_hash"] = protocol.protocol_hash
    write_json(path, payload)
