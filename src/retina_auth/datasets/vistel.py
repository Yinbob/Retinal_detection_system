from __future__ import annotations

import re
import zipfile
from collections import Counter
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


VISTEL_PATTERN = re.compile(
    r"^vistel(?P<subject>\d+)_(?P<eye>left|right)_(?P<index>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)
VISTEL_ARCHIVE_ROOT = "vistel_staff_fundus"


def extract_vistel(archive_path: str | Path, destination: str | Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not is_system_file(name)
            and name.startswith(VISTEL_ARCHIVE_ROOT + "/")
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    safe_extract(archive_path, destination, members)
    return Path(destination) / VISTEL_ARCHIVE_ROOT


def build_vistel_manifest(
    root: str | Path, archive_path: str | Path
) -> DatasetManifest:
    root = Path(root).resolve()
    photo_dir = root / "ImageData_mask"
    vessel_dir = root / "ImageData_mask_vesselseg"
    grouped: dict[tuple[str, str], list[tuple[int, Path, Path]]] = {}

    for photo in photo_dir.glob("*.jpg"):
        match = VISTEL_PATTERN.match(photo.name)
        if match is None:
            continue
        subject = match.group("subject")
        eye = match.group("eye").lower()
        sample_index = int(match.group("index"))
        vessel = vessel_dir / f"{photo.stem}.png"
        if not vessel.exists():
            raise FileNotFoundError(f"Missing vessel image for {photo.name}")
        grouped.setdefault((subject, eye), []).append(
            (sample_index, photo, vessel)
        )

    if len(grouped) != 40:
        raise ValueError(f"Expected 40 Vistel eye identities, found {len(grouped)}")

    records: list[ImageRecord] = []
    all_hashes: list[str] = []
    all_pixels: list[str] = []
    for (subject, eye), items in sorted(grouped.items()):
        items.sort(key=lambda item: (item[0], item[1].name))
        for position, (sample_index, photo, vessel) in enumerate(items):
            rgb = read_rgb(photo)
            content_hash = sha256_bytes(photo.read_bytes())
            pixel_hash = pixel_sha256_rgb(rgb)
            all_hashes.append(content_hash)
            all_pixels.append(pixel_hash)
            records.append(
                ImageRecord(
                    image_id=photo.stem,
                    dataset="vistel",
                    image_path=str(photo.relative_to(root)),
                    content_sha256=content_hash,
                    pixel_sha256=pixel_hash,
                    role="query" if position == 0 else "gallery",
                    subject_id=subject,
                    eye=eye,
                    relation_group=f"vistel:{subject}:{eye}",
                    vessel_path=str(vessel.relative_to(root)),
                    sample_index=sample_index,
                    width=int(rgb.shape[1]),
                    height=int(rgb.shape[0]),
                    zip_member=(
                        f"{VISTEL_ARCHIVE_ROOT}/ImageData_mask/{photo.name}"
                    ),
                )
            )

    records.sort(key=lambda record: record.image_id)
    query_count = sum(record.role == "query" for record in records)
    gallery_count = sum(record.role == "gallery" for record in records)
    if (len(records), query_count, gallery_count) != (198, 40, 158):
        raise ValueError(
            "Unexpected Vistel split: "
            f"{len(records)}/{query_count}/{gallery_count}"
        )

    group_sizes = Counter(len(items) for items in grouped.values())
    summary = {
        "images": len(records),
        "subjects": len({key[0] for key in grouped}),
        "eyes": len(grouped),
        "queries": query_count,
        "gallery": gallery_count,
        "content_hash_duplicates": len(all_hashes) - len(set(all_hashes)),
        "pixel_hash_duplicates": len(all_pixels) - len(set(all_pixels)),
        "group_size_distribution": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
    }
    return DatasetManifest(
        dataset="vistel",
        root=str(root),
        records=tuple(records),
        archive_sha256=sha256_file(archive_path),
        summary=summary,
        audit_date=date.today().isoformat(),
    )

