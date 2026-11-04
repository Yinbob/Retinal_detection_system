from __future__ import annotations

import json
import math
import platform
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .cache import FeatureCache
from .calibration import (
    calibrate_hog,
    calibrate_sift,
    compute_pair_results,
    select_threshold,
)
from .features.repository import FeatureRepository
from .manifest import load_manifest
from .metrics import evaluate, pair_score_summary
from .preprocessing import PreprocessConfig
from .registration import analyze_fire_registration
from .retrieval import retrieve
from .types import DatasetManifest, Prediction, RetrievalProtocol
from .utils import read_yaml, stable_hash, write_json


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = read_yaml(config_path)
    default_path = config_path.with_name("default.yaml")
    if not default_path.exists():
        default_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    defaults = read_yaml(default_path)
    merged = {**defaults, **config}
    for key in ("preprocessing", "sift", "hog", "calibration", "superretina"):
        if key in defaults or key in config:
            merged[key] = {**defaults.get(key, {}), **config.get(key, {})}
    return merged


def _progress_printer(label: str) -> Callable[[int, int], None]:
    last_update = 0.0

    def update(completed: int, total: int) -> None:
        nonlocal last_update
        now = time.monotonic()
        if completed != total and now - last_update < 2.0:
            return
        last_update = now
        print(
            f"[{label}] {completed}/{total} ({completed / total:.1%})",
            flush=True,
        )

    return update


def _superretina_audit(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("superretina", {})
    source_dir = Path(model_config.get("source_dir", "models/SuperRetina"))
    weights = Path(model_config.get("weights", "models/SuperRetina.pth"))
    enabled = bool(model_config.get("enabled", False))
    artifacts_present = source_dir.is_dir() and weights.is_file()
    if not artifacts_present:
        status = "skipped"
        reason = "official_source_or_weights_missing"
    elif not enabled:
        status = "disabled"
        reason = "disabled_in_config"
    else:
        status = "available_not_implemented"
        reason = "adapter_not_installed_in_this_build"
    return {
        "status": status,
        "reason": reason,
        "enabled": enabled,
        "source_dir": str(source_dir),
        "source_present": source_dir.is_dir(),
        "weights": str(weights),
        "weights_present": weights.is_file(),
        "random_initialization_used": False,
    }


def calibrate(
    manifests: dict[str, DatasetManifest],
    protocols: dict[str, RetrievalProtocol],
    config: dict[str, Any],
    artifacts_dir: str | Path,
    methods: Iterable[str] = ("sift", "hog"),
) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    proxy = protocols["vistel_proxy"]
    vistel = manifests["vistel"]
    relevant_ids = set(proxy.query_ids) | set(proxy.gallery_ids)
    relevant_records = [
        record for record in vistel.records if record.image_id in relevant_ids
    ]
    repository = FeatureRepository(
        relevant_records,
        vistel.root,
        PreprocessConfig.from_dict(config["preprocessing"]),
        FeatureCache(artifacts_dir / "cache" / "features"),
    )
    output: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "proxy_protocol_hash": proxy.protocol_hash,
        "preprocessing": PreprocessConfig.from_dict(
            config["preprocessing"]
        ).to_dict(),
        "superretina": _superretina_audit(config),
        "methods": {},
    }
    for method in methods:
        print(f"Calibrating {method}", flush=True)
        if method == "sift":
            result = calibrate_sift(
                proxy,
                repository,
                config["calibration"],
                config["sift"],
                progress=_progress_printer("sift-calibration"),
            )
            bi_cand = dict(result["bidirectional_candidate"])
            bi_thresh = dict(bi_cand["threshold"])
            if not math.isfinite(float(bi_thresh["threshold"])):
                bi_thresh["raw_threshold"] = bi_thresh["threshold"]
                bi_thresh["threshold"] = float(
                    bi_cand["config"].get("min_inliers", 6)
                )
            bi_selected = {**bi_cand, "threshold": bi_thresh}
            output["methods"]["sift_bidirectional"] = {
                "method": "sift_bidirectional",
                "search_candidates": [bi_selected],
                "selected": bi_selected,
                "derived_from": "sift.bidirectional_candidate",
            }
        elif method == "hog":
            result = calibrate_hog(
                proxy,
                repository,
                config["calibration"],
                config["hog"],
                progress=_progress_printer("hog-calibration"),
            )
        elif method == "sift_bidirectional":
            continue
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
        output["methods"][method] = result
        print(
            f"{method}: top1={result['selected']['top1_correct']}/"
            f"{len(proxy.query_ids)}, threshold="
            f"{result['selected']['threshold']['threshold']}",
            flush=True,
        )
    write_json(artifacts_dir / "calibration.json", output)
    write_json(artifacts_dir / "frozen_config.json", output)
    write_json(artifacts_dir / "model_audit.json", output["superretina"])
    return output


def _frozen_method(
    frozen: dict[str, Any], method: str
) -> tuple[dict[str, Any], float]:
    if (
        method == "sift_bidirectional"
        and "sift_bidirectional" not in frozen.get("methods", {})
        and "sift" in frozen.get("methods", {})
    ):
        bi_cand = frozen["methods"]["sift"]["bidirectional_candidate"]
        raw_thresh = float(bi_cand["threshold"]["threshold"])
        eff_thresh = (
            raw_thresh
            if math.isfinite(raw_thresh)
            else float(bi_cand["config"].get("min_inliers", 6))
        )
        return bi_cand["config"], eff_thresh
    try:
        selected = frozen["methods"][method]["selected"]
    except KeyError as error:
        raise KeyError(
            f"No frozen calibration for {method}; run calibrate first"
        ) from error
    raw_thresh = float(selected["threshold"]["threshold"])
    if not math.isfinite(raw_thresh):
        raw_thresh = float(selected["config"].get("min_inliers", 6))
    return selected["config"], raw_thresh


def _prediction_rows(
    predictions: list[Prediction],
) -> list[dict[str, Any]]:
    return [prediction.to_dict() for prediction in predictions]


def _pair_rows(
    pair_results: dict[tuple[str, str], Any],
    protocol: RetrievalProtocol,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in protocol.query_ids:
        excluded = protocol.excluded_reference_ids.get(query_id, frozenset())
        for reference_id in protocol.gallery_ids:
            result = pair_results[(query_id, reference_id)]
            row = {
                "query_id": query_id,
                "reference_id": reference_id,
                "score": result.score,
                "valid": result.valid,
                "reason": result.reason,
                "n_keypoints_q": result.n_keypoints_q,
                "n_keypoints_r": result.n_keypoints_r,
                "n_ratio_matches": result.n_ratio_matches,
                "n_inliers": result.n_inliers,
                "inlier_ratio": result.inlier_ratio,
                "median_reprojection_error": result.median_reprojection_error,
                "transform_json": (
                    None
                    if result.transform is None
                    else json.dumps(result.transform, separators=(",", ":"))
                ),
                "match_time_ms": result.match_time_ms,
                "is_positive": protocol.is_positive(query_id, reference_id),
                "is_confirmed_relation": protocol.is_confirmed_relation(
                    query_id, reference_id
                ),
                "blocked": reference_id in excluded,
                "blocked_reason": (
                    "exact_query_copy" if reference_id in excluded else None
                ),
            }
            rows.append(row)
    return rows


def _classify_error_row(
    row: dict[str, Any],
    by_id: dict[str, Any],
    dataset: str,
) -> tuple[str, str]:
    status = row.get("status")
    query_id = str(row.get("query_id", ""))
    ref_id = row.get("reference_id")
    if status == "no_valid_candidate" or not ref_id:
        return (
            "no_valid_candidate",
            "无任何候选参考图通过特征提取与几何一致性检验",
        )
    if status == "rejected":
        if row.get("top1_correct") or row.get("top1_confirmed_same_eye_correct"):
            return (
                "threshold_marginal_rejection",
                "Top-1 检索命中同眼正参考图，但得分低于代理集标定阈值被边际拒识",
            )
        return (
            "impostor_rejected",
            "Top-1 候选非指定同眼参考，且得分低于阈值被正确拦截为拒识",
        )
    if dataset == "fire":
        q_rec = by_id.get(query_id)
        r_rec = by_id.get(str(ref_id))
        if (
            q_rec is not None
            and r_rec is not None
            and q_rec.content_sha256 == r_rec.content_sha256
        ):
            return (
                "exact_self_copy_hit",
                "FIRE 原始划分下查询图在参考库中存在完全相同的字节副本，优先命中自身副本",
            )
        if row.get("confirmed_same_eye_correct"):
            return (
                "same_eye_alternate_reference",
                "FIRE 同眼多对场景下命中同一只眼睛重叠度更高的另一张参考图",
            )
        return (
            "cross_group_unconfirmed",
            "跨显式连通分量返回（如 P06/P33 逆序近重复配对或大视野偏移导致的跨组匹配）",
        )
    return (
        "false_accept_mismatch",
        "大视野位移或外圈圆边强阶跃梯度导致异眼候选得分高于阈值",
    )


def evaluate_frozen_method(
    manifest: DatasetManifest,
    protocols: list[RetrievalProtocol],
    *,
    method: str,
    feature_config: dict[str, Any],
    threshold: float,
    artifacts_dir: str | Path,
    config: dict[str, Any],
    run_name: str | None = None,
) -> list[Path]:
    if not protocols:
        raise ValueError("At least one protocol is required")
    artifacts_dir = Path(artifacts_dir)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = run_name or f"{timestamp}_{method}_{manifest.dataset}"
    repository = FeatureRepository(
        manifest.records,
        manifest.root,
        PreprocessConfig.from_dict(config["preprocessing"]),
        FeatureCache(artifacts_dir / "cache" / "features"),
    )
    canonical_protocol = protocols[0]
    print(
        f"Evaluating {method} on {manifest.dataset} "
        f"({len(canonical_protocol.query_ids)} x "
        f"{len(canonical_protocol.gallery_ids)})",
        flush=True,
    )
    pair_results = compute_pair_results(
        canonical_protocol,
        repository,
        method=method,
        config=feature_config,
        progress=_progress_printer(f"{method}-{manifest.dataset}"),
    )
    registration_summary: dict[str, Any] | None = None
    registration_rows: list[dict[str, Any]] = []
    if manifest.dataset == "fire" and method in ("sift", "sift_bidirectional"):
        registration_summary, registration_rows = analyze_fire_registration(
            manifest,
            canonical_protocol,
            pair_results,
            repository.preprocess_config,
        )
    record_by_id = manifest.by_id()
    run_paths: list[Path] = []
    for protocol in protocols:
        run_dir = artifacts_dir / "runs" / f"{run_name}_{protocol.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions: list[Prediction] = []
        for query_id in protocol.query_ids:
            allowed = protocol.allowed_gallery(query_id)
            elapsed_ms = sum(
                pair_results[(query_id, reference_id)].match_time_ms
                for reference_id in allowed
            )
            predictions.append(
                retrieve(
                    query_id,
                    allowed,
                    pair_results,
                    protocol,
                    threshold,
                    elapsed_ms=elapsed_ms,
                )
            )
        metrics, prediction_rows = evaluate(
            predictions, protocol, manifest.records
        )
        metrics.update(
            {
                "method": method,
                "config": feature_config,
                "threshold": threshold,
                "manifest_archive_sha256": manifest.archive_sha256,
                "feature_repository": {
                    "processed_unique_images": repository.processed_count,
                    "sift_feature_entries": repository.local_computed,
                    "hog_feature_entries": repository.global_computed,
                },
            }
        )
        if registration_summary is not None:
            metrics["registration_analysis"] = registration_summary
        write_json(run_dir / "metrics.json", metrics)
        pd.DataFrame(prediction_rows).to_csv(
            run_dir / "predictions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(_pair_rows(pair_results, protocol)).to_csv(
            run_dir / "pair_scores.csv", index=False, encoding="utf-8-sig"
        )
        errors = []
        for row in prediction_rows:
            if (
                row["status"] != "accepted"
                or not row.get("designated_correct", False)
                or row.get("same_eye_status") == "same_eye_unconfirmed"
            ):
                err_cat, err_note = _classify_error_row(
                    row, record_by_id, manifest.dataset
                )
                errors.append(
                    {
                        **row,
                        "error_category": err_cat,
                        "diagnostic_note": err_note,
                    }
                )
        pd.DataFrame(errors).to_csv(
            run_dir / "errors.csv", index=False, encoding="utf-8-sig"
        )
        metadata = {
            "run_name": run_dir.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "dataset": manifest.dataset,
            "protocol": protocol.name,
            "protocol_hash": protocol.protocol_hash,
            "manifest_archive_sha256": manifest.archive_sha256,
            "method": method,
            "method_config": feature_config,
            "threshold": threshold,
            "preprocess": config["preprocessing"],
            "implementation_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "pair_results_hash": stable_hash(
                {
                    "archive": manifest.archive_sha256,
                    "protocol": protocol.protocol_hash,
                    "method": method,
                    "config": feature_config,
                }
            ),
        }
        write_json(run_dir / "run.json", metadata)
        write_json(
            run_dir / "pair_summary.json",
            pair_score_summary(pair_results, protocol),
        )
        if registration_summary is not None:
            write_json(
                run_dir / "registration_analysis.json",
                {
                    "summary": registration_summary,
                    "pairs": registration_rows,
                },
            )
            pd.DataFrame(registration_rows).to_csv(
                run_dir / "registration_pairs.csv",
                index=False,
                encoding="utf-8-sig",
            )
        run_paths.append(run_dir)
        print(
            f"{protocol.name}: correct={metrics['correct']}/"
            f"{metrics['queries']}, accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )
    return run_paths


def load_artifact_state(
    artifacts_dir: str | Path,
) -> tuple[
    dict[str, DatasetManifest],
    dict[str, RetrievalProtocol],
]:
    artifacts_dir = Path(artifacts_dir)
    manifests = {
        dataset: load_manifest(
            artifacts_dir / "manifests" / f"{dataset}.json"
        )
        for dataset in ("vistel", "fire")
    }
    protocols: dict[str, RetrievalProtocol] = {}
    for path in sorted((artifacts_dir / "protocols").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        protocols[data["name"]] = RetrievalProtocol(
            name=data["name"],
            dataset=data["dataset"],
            query_ids=tuple(data["query_ids"]),
            gallery_ids=tuple(data["gallery_ids"]),
            positive_reference_ids={
                key: frozenset(value)
                for key, value in data["positive_reference_ids"].items()
            },
            confirmed_relation_reference_ids={
                key: frozenset(value)
                for key, value in data.get(
                    "confirmed_relation_reference_ids", {}
                ).items()
            },
            excluded_reference_ids={
                key: frozenset(value)
                for key, value in data.get("excluded_reference_ids", {}).items()
            },
            label_semantics=data["label_semantics"],
            protocol_hash=data["protocol_hash"],
        )
    return manifests, protocols


def load_frozen(path: str | Path) -> dict[str, Any]:
    frozen_path = Path(path)
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Missing frozen calibration: {frozen_path}. Run calibrate first."
        )
    return json.loads(frozen_path.read_text(encoding="utf-8"))


def evaluate_all(
    manifests: dict[str, DatasetManifest],
    protocols: dict[str, RetrievalProtocol],
    frozen: dict[str, Any],
    config: dict[str, Any],
    artifacts_dir: str | Path,
    methods: Iterable[str] = ("sift", "sift_bidirectional", "hog"),
) -> list[Path]:
    paths: list[Path] = []
    for method in methods:
        method_config, threshold = _frozen_method(frozen, method)
        paths.extend(
            evaluate_frozen_method(
                manifests["vistel"],
                [protocols["vistel_course"]],
                method=method,
                feature_config=method_config,
                threshold=threshold,
                artifacts_dir=artifacts_dir,
                config=config,
            )
        )
        paths.extend(
            evaluate_frozen_method(
                manifests["fire"],
                [
                    protocols["fire_pair"],
                    protocols["fire_pair_exclude_self"],
                ],
                method=method,
                feature_config=method_config,
                threshold=threshold,
                artifacts_dir=artifacts_dir,
                config=config,
            )
        )
    return paths

