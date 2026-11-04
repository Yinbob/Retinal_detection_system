from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .calibration import compute_pair_results
from .cache import FeatureCache
from .datasets import prepare_datasets
from .experiments import (
    calibrate,
    evaluate_all,
    load_artifact_state,
    load_config,
    load_frozen,
)
from .features.repository import FeatureRepository
from .preprocessing import PreprocessConfig
from .reporting import generate_report
from .retrieval import retrieve
from .types import ImageRecord
from .utils import sha256_bytes, write_json


ROOT = Path(__file__).resolve().parents[2]


def _config(args: argparse.Namespace) -> dict:
    return load_config(Path(args.config))


def _prepare(args: argparse.Namespace) -> None:
    manifests, protocols = prepare_datasets(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {
                name: {
                    "queries": len(protocol.query_ids),
                    "gallery": len(protocol.gallery_ids),
                    "hash": protocol.protocol_hash,
                }
                for name, protocol in protocols.items()
            },
            indent=2,
        )
    )


def _calibrate(args: argparse.Namespace) -> None:
    manifests, protocols = load_artifact_state(args.artifacts_dir)
    methods = (
        ["sift", "hog"]
        if args.method == "all"
        else ["sift"]
        if args.method == "sift_bidirectional"
        else [args.method]
    )
    calibrate(
        manifests,
        protocols,
        _config(args),
        args.artifacts_dir,
        methods=methods,
    )


def _evaluate(args: argparse.Namespace) -> None:
    manifests, protocols = load_artifact_state(args.artifacts_dir)
    frozen = load_frozen(args.frozen)
    methods = (
        ["sift", "sift_bidirectional", "hog"]
        if args.method == "all"
        else [args.method]
    )
    paths = evaluate_all(
        manifests,
        protocols,
        frozen,
        _config(args),
        args.artifacts_dir,
        methods=methods,
    )
    for path in paths:
        print(path)


def _report(args: argparse.Namespace) -> None:
    report_path, comparison_path = generate_report(
        args.artifacts_dir, args.analysis
    )
    print(report_path)
    print(comparison_path)


def _make_query_record(path: Path, dataset: str) -> ImageRecord:
    from .utils import pixel_sha256_rgb, read_rgb

    rgb = read_rgb(path)
    return ImageRecord(
        image_id="__query__",
        dataset="vistel" if dataset == "vistel" else "fire",
        image_path=str(path),
        content_sha256=sha256_bytes(path.read_bytes()),
        pixel_sha256=pixel_sha256_rgb(rgb),
        role="query",
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
    )


def _predict(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    config = _config(args)
    manifests, protocols = load_artifact_state(args.artifacts_dir)
    manifest = manifests[args.dataset]
    frozen = load_frozen(args.frozen)
    method_config = frozen["methods"][args.method]["selected"]["config"]
    threshold = float(
        frozen["methods"][args.method]["selected"]["threshold"]["threshold"]
    )
    query_path = Path(args.query).resolve()
    query_record = _make_query_record(query_path, args.dataset)
    gallery_records = [
        record for record in manifest.records if record.role == "gallery"
    ]
    if args.dataset == "fire":
        query_record = ImageRecord(
            **{
                **query_record.to_dict(),
                "mask_path": gallery_records[0].mask_path,
            }
        )
    repository = FeatureRepository(
        [query_record, *gallery_records],
        manifest.root,
        PreprocessConfig.from_dict(config["preprocessing"]),
        FeatureCache(Path(args.artifacts_dir) / "cache" / "features"),
    )
    from .matching import compare_global, compare_local

    results = {}
    if args.method == "sift":
        query_feature = repository.local(query_record, method_config)
        for record in gallery_records:
            results[(query_record.image_id, record.image_id)] = compare_local(
                query_feature,
                repository.local(record, method_config),
                method_config,
            )
    elif args.method == "sift_bidirectional":
        bi_config = {**method_config, "bidirectional": True}
        query_feature = repository.local(query_record, bi_config)
        for record in gallery_records:
            results[(query_record.image_id, record.image_id)] = compare_local(
                query_feature,
                repository.local(record, bi_config),
                bi_config,
            )
    elif args.method == "hog":
        query_feature = repository.global_feature(query_record, method_config)
        for record in gallery_records:
            results[(query_record.image_id, record.image_id)] = compare_global(
                query_feature,
                repository.global_feature(record, method_config),
            )
    else:
        raise ValueError("predict supports sift, sift_bidirectional, and hog")
    placeholder = protocols[
        "fire_pair" if args.dataset == "fire" else "vistel_course"
    ]
    prediction = retrieve(
        query_record.image_id,
        tuple(record.image_id for record in gallery_records),
        results,
        placeholder,
        threshold,
        elapsed_ms=(time.perf_counter() - started_at) * 1000,
    )
    output = prediction.to_dict()
    output["same_eye_status"] = "not_evaluated"
    output["query_path"] = str(query_path)
    output["dataset"] = args.dataset
    output["method"] = args.method
    output["threshold"] = threshold
    output["gallery"] = len(gallery_records)
    output["reference_path"] = (
        None
        if prediction.reference_id is None
        else str(
            manifest.resolve(manifest.by_id()[prediction.reference_id])
        )
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retina-auth")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="YAML configuration path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-dir", default=str(ROOT / "dataset"))
    prepare.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    prepare.set_defaults(func=_prepare)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument(
        "--method",
        choices=["all", "sift", "sift_bidirectional", "hog"],
        default="all",
    )
    calibrate_parser.add_argument(
        "--artifacts-dir", default=str(ROOT / "artifacts")
    )
    calibrate_parser.set_defaults(func=_calibrate)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--method",
        choices=["all", "sift", "sift_bidirectional", "hog"],
        default="all",
    )
    evaluate_parser.add_argument(
        "--artifacts-dir", default=str(ROOT / "artifacts")
    )
    evaluate_parser.add_argument(
        "--frozen", default=str(ROOT / "artifacts" / "frozen_config.json")
    )
    evaluate_parser.set_defaults(func=_evaluate)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument(
        "--artifacts-dir", default=str(ROOT / "artifacts")
    )
    report_parser.add_argument("--analysis")
    report_parser.set_defaults(func=_report)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--query", required=True)
    predict_parser.add_argument("--dataset", choices=["vistel", "fire"], required=True)
    predict_parser.add_argument(
        "--method",
        choices=["sift", "sift_bidirectional", "hog"],
        required=True,
    )
    predict_parser.add_argument(
        "--artifacts-dir", default=str(ROOT / "artifacts")
    )
    predict_parser.add_argument(
        "--frozen", default=str(ROOT / "artifacts" / "frozen_config.json")
    )
    predict_parser.set_defaults(func=_predict)
    _add_serve_parser(subparsers)
    return parser


def _serve(args: argparse.Namespace) -> None:
    from .web.server import serve

    serve(
        args.artifacts_dir,
        args.config,
        host=args.host,
        port=args.port,
    )


def _add_serve_parser(subparsers) -> None:
    serve_parser = subparsers.add_parser("serve", help="start the local web console")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--artifacts-dir", default=str(ROOT / "artifacts")
    )
    serve_parser.set_defaults(func=_serve)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

