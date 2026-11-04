from __future__ import annotations

from pathlib import Path

from ..manifest import write_manifest, write_protocol
from ..types import DatasetManifest, RetrievalProtocol
from ..utils import write_json
from .fire import build_fire_manifest, extract_fire
from .protocol import (
    build_fire_protocols,
    build_vistel_course_protocol,
    build_vistel_proxy_protocol,
)
from .vistel import build_vistel_manifest, extract_vistel


def prepare_datasets(
    data_dir: str | Path,
    artifacts_dir: str | Path,
) -> tuple[
    dict[str, DatasetManifest],
    dict[str, RetrievalProtocol],
]:
    data_dir = Path(data_dir).resolve()
    artifacts_dir = Path(artifacts_dir).resolve()
    extracted_root = artifacts_dir / "datasets"
    manifest_root = artifacts_dir / "manifests"
    protocol_root = artifacts_dir / "protocols"
    vistel_archive = data_dir / "vistel_staff_fundus.zip"
    fire_archive = data_dir / "FIRE.zip"
    if not vistel_archive.exists() or not fire_archive.exists():
        raise FileNotFoundError("Both Vistel and FIRE ZIP archives are required")

    vistel_root = extract_vistel(vistel_archive, extracted_root / "vistel")
    fire_root = extract_fire(fire_archive, extracted_root / "fire")
    manifests = {
        "vistel": build_vistel_manifest(vistel_root, vistel_archive),
        "fire": build_fire_manifest(fire_root, fire_archive),
    }

    fire_original, fire_excluded = build_fire_protocols(manifests["fire"])
    protocols = {
        "vistel_course": build_vistel_course_protocol(manifests["vistel"]),
        "vistel_proxy": build_vistel_proxy_protocol(manifests["vistel"]),
        "fire_pair": fire_original,
        "fire_pair_exclude_self": fire_excluded,
    }
    for name, protocol in protocols.items():
        query_count = len(protocol.query_ids)
        gallery_count = len(protocol.gallery_ids)
        allowed_count = sum(
            len(protocol.allowed_gallery(query)) for query in protocol.query_ids
        )
        if name == "vistel_proxy":
            proxy_genuine = sum(
                len(protocol.positive_reference_ids[query])
                for query in protocol.query_ids
            )
            if (query_count, gallery_count, allowed_count, proxy_genuine) != (
                33,
                125,
                4125,
                118,
            ):
                raise AssertionError(
                    "Unexpected proxy counts: "
                    f"{query_count}/{gallery_count}/{allowed_count}/{proxy_genuine}"
                )

    for name, manifest in manifests.items():
        write_manifest(manifest, manifest_root / f"{name}.json")
    for name, protocol in protocols.items():
        write_protocol(protocol, protocol_root / f"{name}.json")

    summary = {
        "datasets": {
            name: {
                "summary": manifest.summary,
                "archive_sha256": manifest.archive_sha256,
            }
            for name, manifest in manifests.items()
        },
        "protocols": {
            name: {
                "hash": protocol.protocol_hash,
                "queries": len(protocol.query_ids),
                "gallery": len(protocol.gallery_ids),
                "allowed_pairs": sum(
                    len(protocol.allowed_gallery(query))
                    for query in protocol.query_ids
                ),
            }
            for name, protocol in protocols.items()
        },
        "checks": {
            "fire_images_268_not_536": manifests["fire"].summary["images"] == 268,
            "fire_unique_content_124": (
                manifests["fire"].summary["unique_content"] == 124
            ),
            "fire_relation_groups_42": (
                manifests["fire"].summary["relation_groups"] == 42
            ),
            "fire_queries_with_copies_49": (
                manifests["fire"].summary["queries_with_exact_gallery_copy"] == 49
            ),
            "fire_blocked_combinations_110": (
                manifests["fire"].summary["blocked_self_copy_combinations"] == 110
            ),
            "vistel_split_40_158": (
                manifests["vistel"].summary["queries"] == 40
                and manifests["vistel"].summary["gallery"] == 158
            ),
            "protocol_hashes_present": all(
                len(protocol.protocol_hash) == 64
                for protocol in protocols.values()
            ),
        },
    }
    if not all(summary["checks"].values()):
        raise AssertionError(f"Data preparation checks failed: {summary['checks']}")
    write_json(artifacts_dir / "data_audit.json", summary)
    return manifests, protocols

