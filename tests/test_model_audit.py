from retina_auth.experiments import _superretina_audit


def test_superretina_audit_reports_missing_artifacts_before_disabled(tmp_path) -> None:
    audit = _superretina_audit(
        {
            "superretina": {
                "enabled": False,
                "source_dir": str(tmp_path / "missing-source"),
                "weights": str(tmp_path / "missing-weights.pth"),
            }
        }
    )

    assert audit["status"] == "skipped"
    assert audit["reason"] == "official_source_or_weights_missing"
    assert audit["source_present"] is False
    assert audit["weights_present"] is False
    assert audit["random_initialization_used"] is False
