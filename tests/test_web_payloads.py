"""Browser payloads must use strict JSON, including legacy calibration files."""
import json
import math
from pathlib import Path

from retina_auth.web.server import _finite_json, _summary_payload


def test_legacy_calibration_infinities_are_valid_browser_json():
    payload = {"threshold": -math.inf, "search": [{"far": math.nan, "threshold": math.inf}], "score": 0.938}
    encoded = json.dumps(_finite_json(payload), allow_nan=False)
    assert json.loads(encoded) == {"threshold": None, "search": [{"far": None, "threshold": None}], "score": 0.938}
    assert payload["threshold"] == -math.inf  # Stored experiment data is preserved.


def test_summary_exposes_calibration_and_diagnostic_data(tmp_path: Path):
    (tmp_path / "reports").mkdir()
    report = {"rows": [{"method": "sift"}], "error_analysis": {"by_run": {"example": {"cases": [{"query_id": "A02_2"}]}}}}
    (tmp_path / "reports" / "report.json").write_text(json.dumps(report))
    (tmp_path / "calibration.json").write_text(json.dumps({"methods": {"sift": {"selected": {"threshold": {"threshold": -math.inf}}, "search_candidates": [{"top1_correct": 33}], "bidirectional_candidate": {"top1_correct": 33}}}}))
    summary = json.loads(json.dumps(_finite_json(_summary_payload(tmp_path)), allow_nan=False))
    assert summary["report"]["error_analysis"] == report["error_analysis"]
    assert summary["calibration"]["methods"]["sift"]["selected"]["threshold"]["threshold"] is None
    assert summary["calibration"]["methods"]["sift"]["search_candidates"] == [{"top1_correct": 33}]
    assert summary["data_audit"] is None
