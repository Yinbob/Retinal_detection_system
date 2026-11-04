from __future__ import annotations

import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

from ..types import DatasetManifest, ImageRecord
from ..utils import (
    is_system_file,
    pixel_sha256_rgb,
    read_rgb,
    safe_extract,
    sha256_bytes,
    sha256_file,
)


FIRE_PATTERN = re.compile(
    r"^(?P<pair>[SPA]\d+)_(?P<slot>[12])\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)
FIRE_ARCHIVE_ROOT = "FIRE"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def extract_fire(archive_path: str | Path, destination: str | Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not is_system_file(name)
            and name.startswith(FIRE_ARCHIVE_ROOT + "/")
            and (
                name.startswith(FIRE_ARCHIVE_ROOT + "/Images/")
                or name.startswith(FIRE_ARCHIVE_ROOT + "/Masks/")
                or name.startswith(FIRE_ARCHIVE_ROOT + "/Ground Truth/")
                or name == FIRE_ARCHIVE_ROOT + "/README.txt"
            )
        ]
    safe_extract(archive_path, destination, members)
    return Path(destination) / FIRE_ARCHIVE_ROOT


def _assign_relation_groups(records: list[ImageRecord]) -> list[ImageRecord]:
    union = UnionFind()
    by_pair: dict[str, dict[str, ImageRecord]] = defaultdict(dict)
    for record in records:
        union.find(record.content_sha256)
        if record.pair_id is not None:
            pair_name = record.pair_id.split(":")[-1]
            slot = record.image_id[-1]
            by_pair[pair_name][slot] = record
    for pair in by_pair.values():
        if "1" in pair and "2" in pair:
            union.union(pair["1"].content_sha256, pair["2"].content_sha256)

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[union.find(record.content_sha256)].append(record.image_id)
    ordered_groups = sorted(grouped.items(), key=lambda item: min(item[1]))
    group_id_by_hash = {
        content_hash: f"fire_relation_{index:03d}"
        for index, (content_hash, _) in enumerate(ordered_groups)
    }
    return [
        replace(
            record,
            relation_group=group_id_by_hash[union.find(record.content_sha256)],
        )
        for record in records
    ]


def build_fire_manifest(
    root: str | Path, archive_path: str | Path
) -> DatasetManifest:
    root = Path(root).resolve()
    image_dir = root / "Images"
    mask_path = root / "Masks" / "mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing FIRE mask: {mask_path}")

    records: list[ImageRecord] = []
    pairs: dict[str, dict[str, ImageRecord]] = defaultdict(dict)
    for path in image_dir.iterdir():
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        match = FIRE_PATTERN.match(path.name)
        if match is None:
            continue
        rgb = read_rgb(path)
        pair_root = match.group("pair").upper()
        slot = match.group("slot")
        record = ImageRecord(
            image_id=path.stem.upper(),
            dataset="fire",
            image_path=str(path.relative_to(root)),
            content_sha256=sha256_bytes(path.read_bytes()),
            pixel_sha256=pixel_sha256_rgb(rgb),
            role="gallery" if slot == "1" else "query",
            pair_id=f"FIRE:{pair_root}",
            category=pair_root[0],
            mask_path=str(mask_path.relative_to(root)),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            zip_member=f"{FIRE_ARCHIVE_ROOT}/Images/{path.name}",
        )
        records.append(record)
        pairs[pair_root][slot] = record

    if len(pairs) != 134 or len(records) != 268:
        raise ValueError(f"Unexpected FIRE counts: {len(pairs)}/{len(records)}")
    if any(set(value) != {"1", "2"} for value in pairs.values()):
        raise ValueError("Every FIRE pair must contain one _1 and one _2 image")

    records.sort(key=lambda record: record.image_id)
    records = _assign_relation_groups(records)
    by_id = {record.image_id: record for record in records}
    query_records = [record for record in records if record.role == "query"]
    gallery_records = [record for record in records if record.role == "gallery"]
    gallery_hashes = {record.content_sha256 for record in gallery_records}
    gallery_ids_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in gallery_records:
        gallery_ids_by_hash[record.content_sha256].add(record.image_id)

    queries_with_copy: set[str] = set()
    blocked_combinations = 0
    for query in query_records:
        if query.content_sha256 in gallery_hashes:
            queries_with_copy.add(query.image_id)
            blocked_combinations += len(
                gallery_ids_by_hash[query.content_sha256]
            )

    relation_groups = {record.relation_group for record in records}
    categories = Counter(record.category for record in gallery_records)
    duplicate_gallery_references = {
        content_hash
        for content_hash, ids in gallery_ids_by_hash.items()
        if len(ids) > 1
    }
    duplicate_query_contents = len(query_records) - len(
        {record.content_sha256 for record in query_records}
    )
    duplicate_gallery_contents = len(gallery_records) - len(gallery_hashes)
    unique_content = len({record.content_sha256 for record in records})
    if unique_content != 124:
        raise ValueError(f"Expected 124 FIRE contents, found {unique_content}")
    if len(relation_groups) != 42:
        raise ValueError(
            f"Expected 42 FIRE relation groups, found {len(relation_groups)}"
        )
    if len(queries_with_copy) != 49 or blocked_combinations != 110:
        raise ValueError(
            "Unexpected FIRE duplicate protocol: "
            f"{len(queries_with_copy)} queries, {blocked_combinations} blocks"
        )
    if any(
        by_id[query.image_id].content_sha256
        == by_id[pairs[query.pair_id.split(":")[-1]]["1"].image_id].content_sha256
        for query in query_records
    ):
        raise ValueError("A FIRE query's designated reference must not be itself")

    summary = {
        "images": len(records),
        "queries": len(query_records),
        "gallery": len(gallery_records),
        "pairs": len(pairs),
        "unique_content": unique_content,
        "relation_groups": len(relation_groups),
        "duplicate_query_contents": duplicate_query_contents,
        "duplicate_gallery_contents": duplicate_gallery_contents,
        "duplicate_gallery_groups": len(duplicate_gallery_references),
        "queries_with_exact_gallery_copy": len(queries_with_copy),
        "blocked_self_copy_combinations": blocked_combinations,
        "remaining_combinations_excluding_self": (
            len(query_records) * len(gallery_records) - blocked_combinations
        ),
        "category_pairs": dict(sorted(categories.items())),
        "raw_image_entries": 536,
        "ignored_appledouble_entries": 268,
    }
    return DatasetManifest(
        dataset="fire",
        root=str(root),
        records=tuple(records),
        archive_sha256=sha256_file(archive_path),
        summary=summary,
        audit_date=date.today().isoformat(),
    )

