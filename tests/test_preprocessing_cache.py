from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retina_auth.cache import FeatureCache, feature_cache_key
from retina_auth.experiments import load_artifact_state
from retina_auth.features.repository import FeatureRepository
from retina_auth.preprocessing import (
    PreprocessConfig,
    preprocess_array,
    transform_points,
    transform_to_original,
)
from retina_auth.registration import (
    map_reference_points_to_query,
    read_control_points,
)


ROOT = Path(__file__).resolve().parents[1]


def test_coordinate_transform_round_trip() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[:, :, 1] = np.arange(200, dtype=np.uint8)[None, :]
    processed = preprocess_array(
        rgb,
        pixel_hash="synthetic",
        config=PreprocessConfig(
            size=100,
            brightness_threshold=1,
            boundary_erode_px=0,
        ),
        external_mask=np.full((100, 200), 255, dtype=np.uint8),
    )
    points = np.asarray([[0.0, 0.0], [199.0, 99.0], [50.0, 25.0]])
    processed_points = transform_points(processed.original_to_processed, points)
    restored = transform_to_original(processed, processed_points)
    assert np.allclose(restored, points, atol=1e-6)


def test_cache_key_changes_with_each_identity_component() -> None:
    base = feature_cache_key(
        "pixel-a",
        {"size": 1024},
        {"nfeatures": 1500},
        implementation_version="v1",
    )
    assert base == feature_cache_key(
        "pixel-a",
        {"size": 1024},
        {"nfeatures": 1500},
        implementation_version="v1",
    )
    assert base != feature_cache_key(
        "pixel-b",
        {"size": 1024},
        {"nfeatures": 1500},
        implementation_version="v1",
    )
    assert base != feature_cache_key(
        "pixel-a",
        {"size": 512},
        {"nfeatures": 1500},
        implementation_version="v1",
    )
    assert base != feature_cache_key(
        "pixel-a",
        {"size": 1024},
        {"nfeatures": 800},
        implementation_version="v1",
    )
    assert base != feature_cache_key(
        "pixel-a",
        {"size": 1024},
        {"nfeatures": 1500},
        implementation_version="v2",
    )


def test_duplicate_fire_content_reuses_processed_image_and_features(
    tmp_path: Path,
) -> None:
    manifests, _ = load_artifact_state(ROOT / "artifacts")
    manifest = manifests["fire"]
    by_hash: dict[str, list] = {}
    for record in manifest.records:
        by_hash.setdefault(record.pixel_sha256, []).append(record)
    aliases = next(records for records in by_hash.values() if len(records) > 1)
    first, second = aliases[:2]

    repository = FeatureRepository(
        aliases,
        manifest.root,
        PreprocessConfig(size=256, boundary_erode_px=2),
        FeatureCache(tmp_path / "cache"),
    )
    first_features = repository.local(first, {"nfeatures": 200})
    second_features = repository.local(second, {"nfeatures": 200})

    assert repository.processed_count == 1
    assert first_features.valid and second_features.valid
    np.testing.assert_array_equal(
        first_features.keypoints, second_features.keypoints
    )
    np.testing.assert_array_equal(
        first_features.descriptors, second_features.descriptors
    )


def test_sift_keypoints_stay_inside_fov() -> None:
    from retina_auth.features.sift import extract_sift

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    processed = preprocess_array(
        rgb,
        pixel_hash="synthetic-fov",
        config=PreprocessConfig(size=256, boundary_erode_px=0),
        external_mask=mask,
    )
    features = extract_sift(processed, {"nfeatures": 300})
    assert features.valid
    assert len(features.keypoints) > 0
    xy = np.round(features.keypoints).astype(int)
    inside = processed.fov[xy[:, 1], xy[:, 0]] > 0
    assert inside.all()


def test_external_mask_shape_is_checked() -> None:
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        preprocess_array(
            rgb,
            pixel_hash="synthetic",
            config=PreprocessConfig(size=64),
            external_mask=np.ones((10, 10), dtype=np.uint8),
        )


def test_fire_reference_points_round_trip_with_known_transform() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[:, :, 1] = np.arange(200, dtype=np.uint8)[None, :]
    config = PreprocessConfig(
        size=200,
        brightness_threshold=1,
        boundary_erode_px=0,
    )
    mask = np.full((100, 200), 255, dtype=np.uint8)
    reference = preprocess_array(
        rgb,
        pixel_hash="reference",
        config=config,
        external_mask=mask,
    )
    query = preprocess_array(
        rgb,
        pixel_hash="query",
        config=config,
        external_mask=mask,
    )
    reference_points = np.asarray(
        [[10.0, 20.0], [150.0, 60.0], [75.0, 25.0]]
    )
    query_to_reference = np.asarray(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]]
    )
    mapped = map_reference_points_to_query(
        reference_points,
        query_to_reference,
        reference,
        query,
    )
    np.testing.assert_allclose(
        mapped,
        reference_points - np.asarray([10.0, 5.0]),
        atol=1e-6,
    )


def test_fire_control_points_require_four_columns(tmp_path: Path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text(
        "10 20 11 21\n30 40 31 41\n",
        encoding="utf-8",
    )
    points = read_control_points(valid)
    assert points.shape == (2, 4)

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("10 20 11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="N x 4"):
        read_control_points(invalid)
