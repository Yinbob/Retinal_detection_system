from __future__ import annotations

from pathlib import Path

from retina_auth.experiments import load_artifact_state
from retina_auth.utils import is_system_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

EXPECTED_PROTOCOL_HASHES = {
    "vistel_course": "4ad6d0063b8661870970e46e5e2d58bff76ab32a4c40680e682f160ef13ae56c",
    "vistel_proxy": "140f2efc69aac60b5c712effd9209ff2851dac0ec47dd1d2ba02bcbdadd4e6ea",
    "fire_pair": "1513347f6dce7b96586f8cc9da64e14953eb437c527e5eba21be764928afe7cc",
    "fire_pair_exclude_self": (
        "827539bbb8ec4d8c928dfd358b963bcfad0f7bb48f27d15a69404facbc6875e8"
    ),
}


def test_system_files_are_filtered() -> None:
    assert is_system_file("__MACOSX/FIRE/Images/._S1_1.jpg")
    assert is_system_file("FIRE/Images/._S1_1.jpg")
    assert is_system_file("FIRE/.DS_Store")
    assert not is_system_file("FIRE/Images/S1_1.jpg")


def test_generated_manifest_and_protocol_counts() -> None:
    manifests, protocols = load_artifact_state(ARTIFACTS)

    vistel = manifests["vistel"]
    assert vistel.summary["images"] == 198
    assert vistel.summary["queries"] == 40
    assert vistel.summary["gallery"] == 158
    assert vistel.summary["subjects"] == 20
    assert vistel.summary["eyes"] == 40

    fire = manifests["fire"]
    assert fire.summary["images"] == 268
    assert fire.summary["queries"] == 134
    assert fire.summary["gallery"] == 134
    assert fire.summary["unique_content"] == 124
    assert fire.summary["relation_groups"] == 42
    assert fire.summary["queries_with_exact_gallery_copy"] == 49
    assert fire.summary["blocked_self_copy_combinations"] == 110
    assert fire.summary["remaining_combinations_excluding_self"] == 17846
    assert fire.summary["category_pairs"] == {"A": 14, "P": 49, "S": 71}

    assert len(protocols["vistel_course"].query_ids) == 40
    assert len(protocols["vistel_course"].gallery_ids) == 158
    assert len(protocols["vistel_proxy"].query_ids) == 33
    assert len(protocols["vistel_proxy"].gallery_ids) == 125
    assert len(protocols["fire_pair"].query_ids) == 134
    assert len(protocols["fire_pair"].gallery_ids) == 134
    assert len(protocols["fire_pair_exclude_self"].allowed_gallery(
        protocols["fire_pair_exclude_self"].query_ids[0]
    )) == 134


def test_protocol_hashes_are_stable() -> None:
    _, protocols = load_artifact_state(ARTIFACTS)
    assert {
        name: protocol.protocol_hash for name, protocol in protocols.items()
    } == EXPECTED_PROTOCOL_HASHES
