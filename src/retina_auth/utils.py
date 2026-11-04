from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: str | Path) -> np.ndarray:
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_gray(path: str | Path) -> np.ndarray:
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unable to decode image: {path}")
    return gray


def decode_rgb_bytes(data: bytes) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Uploaded data is not a decodable image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pixel_sha256_rgb(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    import yaml

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def median_and_p95(values: Iterable[float]) -> tuple[float | None, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None, None
    return float(np.median(array)), float(np.percentile(array, 95))


def safe_extract(
    archive_path: str | Path,
    destination: str | Path,
    members: Iterable[str],
) -> None:
    import zipfile

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        available = set(archive.namelist())
        for member in members:
            if member not in available or member.endswith("/"):
                continue
            relative = Path(member)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ZIP member: {member}")
            target = (destination / relative).resolve()
            if destination not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)


def is_system_file(name: str) -> bool:
    parts = Path(name).parts
    return (
        "__MACOSX" in parts
        or Path(name).name == ".DS_Store"
        or Path(name).name.startswith("._")
    )


def sidecar_path(root: str | Path, path: str | Path, suffix: str) -> Path:
    relative = Path(path)
    return Path(root) / relative.with_suffix(suffix)

