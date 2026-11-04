from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import ImageRecord, ProcessedImage
from .utils import read_gray, read_rgb


@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 1024
    brightness_threshold: int = 10
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    boundary_erode_px: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessConfig":
        return cls(
            size=int(data.get("size", 1024)),
            brightness_threshold=int(data.get("brightness_threshold", 10)),
            clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
            clahe_grid_size=int(data.get("clahe_grid_size", 8)),
            boundary_erode_px=int(data.get("boundary_erode_px", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "brightness_threshold": self.brightness_threshold,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_grid_size": self.clahe_grid_size,
            "boundary_erode_px": self.boundary_erode_px,
        }


def _largest_fov(gray: np.ndarray, threshold: int) -> np.ndarray:
    initial = np.where(gray > threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(initial, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No valid field of view found")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < gray.size * 0.05:
        raise ValueError(f"Field of view is too small: {area / gray.size:.4f}")
    fov = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(fov, [contour], -1, 255, cv2.FILLED)
    return fov


def _resize_and_pad(
    image: np.ndarray,
    fov: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], float]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    target_w = max(1, min(size, int(round(width * scale))))
    target_h = max(1, min(size, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    resized_fov = cv2.resize(fov, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    pad_x = (size - target_w) // 2
    pad_y = (size - target_h) // 2
    canvas = np.zeros((size, size), dtype=np.uint8)
    fov_canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[pad_y : pad_y + target_h, pad_x : pad_x + target_w] = resized
    fov_canvas[
        pad_y : pad_y + target_h, pad_x : pad_x + target_w
    ] = resized_fov
    return canvas, fov_canvas, fov, (pad_x, pad_y), scale


def preprocess_array(
    rgb: np.ndarray,
    *,
    pixel_hash: str,
    config: PreprocessConfig,
    external_mask: np.ndarray | None = None,
) -> ProcessedImage:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {rgb.shape}")
    height, width = rgb.shape[:2]
    green = rgb[:, :, 1]
    if external_mask is None:
        fov_original = _largest_fov(green, config.brightness_threshold)
    else:
        if external_mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {external_mask.shape} does not match image {(height, width)}"
            )
        fov_original = np.where(external_mask > 0, 255, 0).astype(np.uint8)
        if not np.any(fov_original):
            raise ValueError("External mask is empty")

    x, y, box_w, box_h = cv2.boundingRect(fov_original)
    if box_w < 2 or box_h < 2:
        raise ValueError("Field of view has an invalid bounding box")
    cropped = green[y : y + box_h, x : x + box_w]
    cropped_mask = fov_original[y : y + box_h, x : x + box_w]
    resized, resized_mask, _, padding, scale = _resize_and_pad(
        cropped, cropped_mask, config.size
    )

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
    )
    enhanced = clahe.apply(resized)
    enhanced[resized_mask == 0] = 0
    erode_size = max(0, config.boundary_erode_px)
    if erode_size:
        kernel = np.ones((erode_size * 2 + 1, erode_size * 2 + 1), np.uint8)
        valid_fov = cv2.erode(resized_mask, kernel, iterations=1)
    else:
        valid_fov = resized_mask
    if not np.any(valid_fov):
        raise ValueError("Field of view became empty after boundary erosion")
    enhanced[valid_fov == 0] = 0

    pad_x, pad_y = padding
    transform = np.array(
        [
            [scale, 0.0, pad_x - x * scale],
            [0.0, scale, pad_y - y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ProcessedImage(
        gray=enhanced,
        fov=valid_fov,
        original_to_processed=transform,
        original_size=(width, height),
        processed_size=(config.size, config.size),
        source_pixel_sha256=pixel_hash,
        crop_xywh=(x, y, box_w, box_h),
        scale=scale,
        padding=padding,
    )


def preprocess_record(
    record: ImageRecord,
    dataset_root: str | Path,
    config: PreprocessConfig,
) -> ProcessedImage:
    root = Path(dataset_root)
    rgb = read_rgb(root / record.image_path)
    external_mask = None
    if record.mask_path is not None:
        external_mask = read_gray(root / record.mask_path)
    return preprocess_array(
        rgb,
        pixel_hash=record.pixel_sha256,
        config=config,
        external_mask=external_mask,
    )


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape N x 2")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ matrix.T
    denominator = transformed[:, 2:3]
    if np.any(~np.isfinite(denominator)) or np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Invalid projective transform")
    return transformed[:, :2] / denominator


def transform_to_original(
    processed: ProcessedImage, points: np.ndarray
) -> np.ndarray:
    inverse = np.linalg.inv(processed.original_to_processed)
    return transform_points(inverse, points)

