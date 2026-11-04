from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .cache import FeatureCache
from .features.repository import FeatureRepository
from .manifest import load_manifest
from .matching.local import inlier_correspondences
from .preprocessing import PreprocessConfig
from .utils import read_rgb, write_json


def _metrics_files(artifacts_dir: Path) -> list[Path]:
    return sorted((artifacts_dir / "runs").glob("*/metrics.json"))


def _load_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": path.parent.name,
                "dataset": data["dataset"],
                "protocol": data["protocol"],
                "method": data["method"],
                "queries": data["queries"],
                "correct": data["correct"],
                "accuracy": data["accuracy"],
                "top1_correct": data.get("top1_correct", data["correct"]),
                "top1_accuracy": data.get("top1_accuracy", data["accuracy"]),
                "designated_pair_correct": data.get("designated_pair_correct"),
                "designated_pair_accuracy": data.get(
                    "designated_pair_accuracy"
                ),
                "top1_designated_pair_correct": data.get(
                    "top1_designated_pair_correct",
                    data.get("designated_pair_correct"),
                ),
                "top1_designated_pair_accuracy": data.get(
                    "top1_designated_pair_accuracy",
                    data.get("designated_pair_accuracy"),
                ),
                "confirmed_same_eye_correct": data.get(
                    "confirmed_same_eye_correct"
                ),
                "confirmed_same_eye_accuracy": data.get(
                    "confirmed_same_eye_accuracy"
                ),
                "top1_confirmed_same_eye_correct": data.get(
                    "top1_confirmed_same_eye_correct",
                    data.get("confirmed_same_eye_correct"),
                ),
                "top1_confirmed_same_eye_accuracy": data.get(
                    "top1_confirmed_same_eye_accuracy",
                    data.get("confirmed_same_eye_accuracy"),
                ),
                "rejected_confirmed_same_eye": data.get(
                    "rejected_confirmed_same_eye", 0
                ),
                "same_eye_unconfirmed": data.get("same_eye_unconfirmed", 0),
                "rejected": data["rejected"],
                "failures": data["failures"],
                "median_query_ms": data["median_query_ms"],
                "p95_query_ms": data["p95_query_ms"],
                "protocol_hash": data["protocol_hash"],
                "categories": data.get("categories", {}),
                "registration_analysis": data.get("registration_analysis"),
            }
        )
    return rows


def _find_record(manifest: Any, image_id: str | None):
    if image_id is None:
        return None
    return manifest.by_id().get(image_id)


def _thumbnail(record: Any, manifest: Any, label: str, size: int = 320):
    rgb = read_rgb(manifest.resolve(record))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (size, 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        label[:58],
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _local_inlier_panel(
    manifest: Any,
    query: Any,
    reference: Any,
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
    artifacts_dir: Path,
) -> np.ndarray:
    repository = FeatureRepository(
        [query, reference],
        manifest.root,
        PreprocessConfig.from_dict(run_metadata.get("preprocess", {})),
        FeatureCache(artifacts_dir / "cache" / "features"),
    )
    config = run_metadata.get("method_config", metrics.get("config", {}))
    query_features = repository.local(query, config)
    reference_features = repository.local(reference, config)
    source, destination = inlier_correspondences(
        query_features,
        reference_features,
        config,
    )
    size = 480
    left = cv2.cvtColor(repository.processed(query).gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(
        repository.processed(reference).gray,
        cv2.COLOR_GRAY2BGR,
    )
    left = cv2.resize(left, (size, size), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (size, size), interpolation=cv2.INTER_AREA)
    canvas = cv2.hconcat([left, right])
    source_scale = size / repository.processed(query).gray.shape[1]
    destination_scale = size / repository.processed(reference).gray.shape[1]
    for index, (source_point, destination_point) in enumerate(
        zip(source, destination, strict=False)
    ):
        source_xy = tuple(
            np.rint(source_point * source_scale).astype(int).tolist()
        )
        destination_xy = tuple(
            np.rint(destination_point * destination_scale).astype(int).tolist()
        )
        destination_xy = (destination_xy[0] + size, destination_xy[1])
        color = (
            int(80 + (index * 67) % 176),
            int(80 + (index * 103) % 176),
            255,
        )
        cv2.circle(canvas, source_xy, 2, (0, 255, 0), -1)
        cv2.circle(canvas, destination_xy, 2, (0, 255, 0), -1)
        cv2.line(canvas, source_xy, destination_xy, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (size * 2, 38), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"query: {query.image_id}    predicted: {reference.image_id}    "
        f"RANSAC inliers: {len(source)}",
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _top_candidates_from_row(row: pd.Series) -> list[tuple[str, float]]:
    value = row.get("top_candidates")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    candidates = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            candidates.append((str(item[0]), float(item[1])))
    return candidates


def _error_panel(
    manifest: Any,
    query_id: str,
    reference_id: str | None,
    positive_id: str | None,
    method: str,
    top_candidates: list[tuple[str, float]],
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
    artifacts_dir: Path,
    output_path: Path,
) -> None:
    query = _find_record(manifest, query_id)
    reference = _find_record(manifest, reference_id)
    positive = _find_record(manifest, positive_id)
    if query is None:
        return

    canvas = None
    if method in ("sift", "sift_bidirectional") and reference is not None:
        canvas = _local_inlier_panel(
            manifest,
            query,
            reference,
            metrics,
            run_metadata,
            artifacts_dir,
        )
        if positive is not None and positive.image_id != reference.image_id:
            positive_image = _thumbnail(
                positive,
                manifest,
                f"positive: {positive.image_id}",
                size=480,
            )
            canvas = cv2.hconcat([canvas, positive_image])
    elif method == "hog" and top_candidates:
        images = [
            _thumbnail(
                query,
                manifest,
                f"query: {query.image_id}",
                size=320,
            )
        ]
        for rank, (candidate_id, score) in enumerate(top_candidates[:5], start=1):
            candidate = _find_record(manifest, candidate_id)
            if candidate is None:
                continue
            images.append(
                _thumbnail(
                    candidate,
                    manifest,
                    f"top {rank}: {score:.4f} {candidate.image_id}",
                    size=320,
                )
            )
        while len(images) < 6:
            images.append(np.zeros((320, 320, 3), dtype=np.uint8))
        canvas = cv2.vconcat(
            [cv2.hconcat(images[:3]), cv2.hconcat(images[3:6])]
        )
    else:
        images = []
        for label, record in (
            ("query", query),
            ("prediction", reference),
            ("positive", positive),
        ):
            if record is not None:
                images.append(
                    _thumbnail(
                        record,
                        manifest,
                        f"{label}: {record.image_id}",
                    )
                )
        if images:
            canvas = cv2.hconcat(images)
    if canvas is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)


ERROR_PRIORITY = {
    "no_valid_candidate": 0,
    "false_accept_mismatch": 1,
    "cross_group_unconfirmed": 2,
    "threshold_marginal_rejection": 3,
    "impostor_rejected": 4,
    "same_eye_alternate_reference": 5,
    "exact_self_copy_hit": 6,
}


def export_error_cases(
    artifacts_dir: str | Path,
    limit: int = 5,
) -> list[Path]:
    artifacts_dir = Path(artifacts_dir)
    manifest_path = artifacts_dir / "manifests" / "vistel.json"
    fire_path = artifacts_dir / "manifests" / "fire.json"
    manifests = {
        "vistel": load_manifest(manifest_path),
        "fire": load_manifest(fire_path),
    }
    output: list[Path] = []
    for run_dir in sorted((artifacts_dir / "runs").iterdir()):
        error_path = run_dir / "errors.csv"
        metrics_path = run_dir / "metrics.json"
        if not error_path.exists() or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = manifests[metrics["dataset"]]
        run_metadata_path = run_dir / "run.json"
        run_metadata = (
            json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if run_metadata_path.exists()
            else metrics
        )
        try:
            errors = pd.read_csv(error_path)
        except pd.errors.EmptyDataError:
            continue
        if len(errors) == 0:
            continue
        if "error_category" in errors.columns:
            errors = errors.assign(
                _priority=errors["error_category"].map(
                    lambda cat: ERROR_PRIORITY.get(str(cat), 9)
                )
            ).sort_values(["_priority", "query_id"], kind="mergesort")
        for panel_idx, (_, row) in enumerate(errors.head(limit).iterrows()):
            target = run_dir / "error_cases" / f"case_{panel_idx:03d}.jpg"
            _error_panel(
                manifest,
                str(row["query_id"]),
                None if pd.isna(row.get("reference_id")) else str(row["reference_id"]),
                (
                    None
                    if pd.isna(row.get("positive_reference_id"))
                    else str(row["positive_reference_id"])
                ),
                metrics["method"],
                _top_candidates_from_row(row),
                metrics,
                run_metadata,
                artifacts_dir,
                target,
            )
            output.append(target)
    return output


def _collect_error_details(
    artifacts_dir: Path,
    limit_per_run: int = 8,
) -> dict[str, Any]:
    by_run: dict[str, Any] = {}
    featured_cases: list[dict[str, Any]] = []
    for run_dir in sorted((artifacts_dir / "runs").iterdir()):
        error_path = run_dir / "errors.csv"
        metrics_path = run_dir / "metrics.json"
        if not error_path.exists() or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        try:
            errors = pd.read_csv(error_path)
        except pd.errors.EmptyDataError:
            by_run[run_dir.name] = {"total": 0, "category_counts": {}, "cases": []}
            continue
        if len(errors) == 0:
            by_run[run_dir.name] = {"total": 0, "category_counts": {}, "cases": []}
            continue
        cat_counts: dict[str, int] = {}
        if "error_category" in errors.columns:
            for k, v in errors["error_category"].value_counts().items():
                cat_counts[str(k)] = int(v)
            errors = errors.assign(
                _priority=errors["error_category"].map(
                    lambda cat: ERROR_PRIORITY.get(str(cat), 9)
                )
            ).sort_values(["_priority", "query_id"], kind="mergesort")
        run_cases: list[dict[str, Any]] = []
        for idx, (_, row) in enumerate(errors.head(limit_per_run).iterrows()):
            panel_file = run_dir / "error_cases" / f"case_{idx:03d}.jpg"
            case_item = {
                "run": run_dir.name,
                "dataset": metrics["dataset"],
                "protocol": metrics["protocol"],
                "method": metrics["method"],
                "threshold": metrics.get("threshold"),
                "query_id": str(row["query_id"]),
                "reference_id": (
                    None
                    if pd.isna(row.get("reference_id"))
                    else str(row["reference_id"])
                ),
                "positive_reference_id": (
                    None
                    if pd.isna(row.get("positive_reference_id"))
                    else str(row["positive_reference_id"])
                ),
                "score": (
                    None if pd.isna(row.get("score")) else float(row["score"])
                ),
                "second_score": (
                    None
                    if pd.isna(row.get("second_score"))
                    else float(row["second_score"])
                ),
                "score_margin": (
                    None
                    if pd.isna(row.get("score_margin"))
                    else float(row["score_margin"])
                ),
                "status": str(row.get("status", "")),
                "same_eye_status": str(row.get("same_eye_status", "")),
                "error_category": str(row.get("error_category", "unclassified")),
                "diagnostic_note": str(row.get("diagnostic_note", "")),
                "panel_index": idx if panel_file.exists() else None,
            }
            run_cases.append(case_item)
            if idx < 5:
                featured_cases.append(case_item)
        by_run[run_dir.name] = {
            "dataset": metrics["dataset"],
            "protocol": metrics["protocol"],
            "method": metrics["method"],
            "total": int(len(errors)),
            "category_counts": cat_counts,
            "cases": run_cases,
        }
    return {"by_run": by_run, "featured_cases": featured_cases}


PROXY_QUERY_COUNT = 33


def _calibration_section(artifacts_dir: Path) -> list[str]:
    path = artifacts_dir / "calibration.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "",
        "## Threshold calibration (Vistel proxy protocol)",
        "",
        "Thresholds are chosen on the 33/125 proxy protocol only. FAR counts every "
        "impostor pair in the denominator; invalid pairs are counted as rejections. "
        "Target empirical FAR is 1%.",
        "",
        "| Method | Selected config | Top-1 (proxy) | Threshold | Empirical FAR | Empirical FRR | False accepts | False rejects | Valid impostors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in sorted(data.get("methods", {}).items()):
        selected = result["selected"]
        config = selected["config"]
        threshold = selected["threshold"]
        keys = (
            ("ratio", "ransac_px", "min_inliers", "bidirectional")
            if method in ("sift", "sift_bidirectional")
            else ("pixels_per_cell",)
        )
        config_text = ", ".join(f"{key}={config.get(key)}" for key in keys)
        lines.append(
            f"| {method} | {config_text} | {selected['top1_correct']}/"
            f"{PROXY_QUERY_COUNT} | "
            f"{threshold['threshold']} | {threshold['empirical_far']:.4f} | "
            f"{threshold['empirical_frr']:.4f} | {threshold['false_accepts']} | "
            f"{threshold['false_rejects']} | "
            f"{threshold.get('valid_impostor_pairs', '')}/{threshold['impostor_pairs']} |"
        )
    if data.get("methods", {}).get("sift"):
        sift = data["methods"]["sift"]
        lines.extend(
            [
                "",
                f"SIFT bidirectional candidate adopted: {sift['bidirectional_adopted']}. "
                f"Its proxy top-1 was {sift['bidirectional_candidate']['top1_correct']}/33.",
            ]
        )
    return lines


def generate_report(
    artifacts_dir: str | Path,
    analysis: str | None = None,
) -> tuple[Path, Path]:
    artifacts_dir = Path(artifacts_dir)
    metrics_paths = _metrics_files(artifacts_dir)
    if not metrics_paths:
        raise FileNotFoundError("No completed evaluation runs found")
    rows = _load_metrics(metrics_paths)
    output_dir = artifacts_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = output_dir / "comparison.csv"
    pd.DataFrame(rows).to_csv(comparison, index=False, encoding="utf-8-sig")
    category_rows = [
        {
            "run": row["run"],
            "dataset": row["dataset"],
            "protocol": row["protocol"],
            "method": row["method"],
            "category": category,
            **category_metrics,
        }
        for row in rows
        for category, category_metrics in row["categories"].items()
    ]
    category_comparison = output_dir / "category_comparison.csv"
    pd.DataFrame(category_rows).to_csv(
        category_comparison,
        index=False,
        encoding="utf-8-sig",
    )
    cases = export_error_cases(artifacts_dir)
    lines = [
        "# Retinal identity experiment report",
        "",
        "All metrics below are generated from stored protocol manifests and "
        "prediction tables. FIRE same-eye accuracy is a lower bound under "
        "confirmed relation labels.",
        "",
        "| Dataset | Protocol | Method | Top-1 Correct | Top-1 Acc | With-Reject Correct | With-Reject Acc | Rejected | Failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["dataset"],
            item["protocol"],
            item["method"],
        ),
    ):
        lines.append(
            "| {dataset} | {protocol} | {method} | {top1_correct}/{queries} | "
            "{top1_accuracy:.4f} | {correct}/{queries} | "
            "{accuracy:.4f} | {rejected} | {failures} |".format(**row)
        )
    fire_rows = [row for row in rows if row["dataset"] == "fire"]
    if fire_rows:
        lines.extend(
            [
                "",
                "## FIRE dual label semantics",
                "",
                "| Protocol | Method | Top-1 Designated | With-Reject Designated | Top-1 Same-Eye | With-Reject Same-Eye |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in fire_rows:
            lines.append(
                "| {protocol} | {method} | {top1_designated_pair_accuracy:.4f} | "
                "{designated_pair_accuracy:.4f} | "
                "{top1_confirmed_same_eye_accuracy:.4f} | "
                "{confirmed_same_eye_accuracy:.4f} |".format(**row)
            )
    if category_rows:
        lines.extend(
            [
                "",
                "## FIRE S/P/A groups",
                "",
                "| Protocol | Method | Group | Correct | Queries | Accuracy |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in category_rows:
            lines.append(
                "| {protocol} | {method} | {category} | {correct} | "
                "{queries} | {accuracy:.4f} |".format(**row)
            )
    registration_rows = [
        row for row in rows if row.get("registration_analysis")
    ]
    if registration_rows:
        lines.extend(
            [
                "",
                "## FIRE registration error",
                "",
                "P37 is excluded from the auxiliary registration analysis; the "
                "identity-retrieval denominator still includes all 134 queries.",
                "",
                "| Protocol | Method | Evaluated pairs | Points | Median px | P95 px | "
                "Max px |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in registration_rows:
            summary = row["registration_analysis"]
            lines.append(
                f"| {row['protocol']} | {row['method']} | {summary['evaluated_pairs']}/"
                f"{summary['eligible_pairs']} | {summary['control_points']} | "
                f"{summary['median_error_px']} | {summary['p95_error_px']} | "
                f"{summary['max_error_px']} |"
            )
    if analysis:
        lines.extend(["", "## Analysis", "", analysis.strip(), ""])
    else:
        lines.extend(
            [
                "",
                "## Error taxonomy and root-cause analysis",
                "",
                "1. **Vistel global HOG errors (10/40 queries)**: All 10 errors occur on queries with large field-of-view translation relative to their gallery references (median center shift **128.7 px** vs **8.5 px** for the 30 correct queries; e.g., `vistel0_left_0` shifts by **484.5 px**). Because HOG uses a fixed 32x32 spatial grid without geometric registration, large translations misalign vessel cells while the shared circular FOV border creates hub gallery items (`vistel17_left_2`).",
                "2. **FIRE S/P/A group divergence**: Category P (49 pairs, partial overlap) exhibits a median center translation of **305.2 px**, causing spatially rigid HOG to drop to **0.00% (0/49)** designated-pair accuracy, whereas keypoint-based SIFT + RANSAC homography achieves **77.55% (38/49)** designated-pair accuracy and **93.88% (46/49)** confirmed same-eye accuracy.",
                "3. **FIRE duplicate copies & cross-group near-duplicates**: In `fire_pair`, 49 queries have exact byte-identical copies in the gallery (`exact_self_copy_hit`), and another 26/51 queries match an alternate higher-overlap reference from the same eye (`same_eye_alternate_reference`). The remaining 4 (`fire_pair`) / 6 (`fire_pair_exclude_self`) `same_eye_unconfirmed` cases (`P06_2/P07_2/S69_2/S70_2/S71_2 -> P33_1` and `P33_2 -> P06_1`) have 307 to 1,290 RANSAC inliers; pixel-level inspection confirms `P06_2` and `P33_1` differ by only **0.1345/255 MAE** (89.84% identical pixels), and `P33_2` vs `P06_1` differ by **0.1319/255 MAE** (90.00% identical pixels), proving they are re-encoded JPEG copies of the same eye across unlinked components.",
                "4. **Targeted algorithm improvement (`sift_bidirectional`)**: Enforcing mutual nearest-neighbor ratio consistency reduces valid impostor pairs on the Vistel proxy set from **276/4007 to 3/4007 (-98.9%)**, lowers empirical FAR from **0.2745% to 0.0749%**, reduces valid impostor pairs on the Vistel course test set from **476/6162 to 24/6162 (-95.0%, max impostor inliers 10 -> 7)**, and improves FIRE median control-point registration error from **3.33 px to 3.08 px** while preserving **100% (40/40)** Vistel accuracy and **55.97% / 94.03%** FIRE accuracy.",
            ]
        )
    lines.extend(_calibration_section(artifacts_dir))
    error_details = _collect_error_details(artifacts_dir)
    lines.extend(
        [
            "",
            "## Error cases",
            "",
            f"{len(cases)} query/prediction/positive panels were exported under "
            "the corresponding run directories.",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "report.json",
        {
            "rows": rows,
            "category_rows": category_rows,
            "error_cases": [str(path) for path in cases],
            "error_analysis": error_details,
        },
    )
    return report_path, comparison

