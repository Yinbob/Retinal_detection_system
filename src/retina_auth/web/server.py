"""Local HTTP server for the retinal identification web console.

Standard library only. Binds to loopback by default. Image access is limited to
image ids present in the stored manifests, never arbitrary filesystem paths.
"""

from __future__ import annotations

import json
import math
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..identification import DATASETS, METHODS, IdentificationService
from ..utils import read_rgb


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
REPORT_FILE = "report.json"
CALIBRATION_FILE = "calibration.json"


def _finite_json(value: Any) -> Any:
    """Keep legacy calibration infinities out of browser JSON responses."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


@dataclass
class AppState:
    service: IdentificationService
    artifacts_dir: Path
    lock: Lock


def _encode_jpeg(rgb, max_side: int = 512) -> bytes:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Failed to encode thumbnail")
    return encoded.tobytes()


def _summary_payload(artifacts_dir: Path) -> dict[str, Any]:
    report_path = artifacts_dir / "reports" / REPORT_FILE
    calibration_path = artifacts_dir / CALIBRATION_FILE
    audit_path = artifacts_dir / "data_audit.json"
    payload: dict[str, Any] = {
        "report": None,
        "calibration": None,
        "data_audit": None,
    }
    if report_path.exists():
        payload["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    if audit_path.exists():
        payload["data_audit"] = json.loads(
            audit_path.read_text(encoding="utf-8")
        )
    if calibration_path.exists():
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        payload["calibration"] = {
            "proxy_protocol_hash": data.get("proxy_protocol_hash"),
            "preprocessing": data.get("preprocessing"),
            "superretina": data.get("superretina"),
            "methods": {
                name: {
                    "selected": result["selected"],
                    "search_candidates": result.get("search_candidates", []),
                    "min_inlier_candidates": result.get(
                        "min_inlier_candidates", []
                    ),
                    "bidirectional_candidate": result.get(
                        "bidirectional_candidate"
                    ),
                    "bidirectional_adopted": result.get("bidirectional_adopted"),
                }
                for name, result in data.get("methods", {}).items()
            },
        }
    return payload


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RetinaConsole/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        # -- helpers -------------------------------------------------------
        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(_finite_json(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def _query_params(self) -> dict[str, str]:
            parsed = parse_qs(urlparse(self.path).query)
            return {key: values[-1] for key, values in parsed.items()}

        def _check_dataset_method(self, dataset: str, method: str) -> None:
            if dataset not in DATASETS:
                raise ValueError(f"Unknown dataset: {dataset}")
            if method not in METHODS:
                raise ValueError(f"Unknown method: {method}")

        # -- routes --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/" or path == "/index.html":
                    return self._serve_static("index.html")
                if path.startswith("/static/"):
                    return self._serve_static(path[len("/static/") :])
                if path == "/api/health":
                    return self._send_json(HTTPStatus.OK, {"status": "ok"})
                if path == "/api/datasets":
                    return self._send_json(HTTPStatus.OK, self._datasets())
                if path == "/api/protocol-queries":
                    params = self._query_params()
                    dataset = params.get("dataset", "")
                    if dataset not in DATASETS:
                        raise ValueError(f"Unknown dataset: {dataset}")
                    return self._send_json(
                        HTTPStatus.OK,
                        {"queries": state.service.protocol_queries(dataset)},
                    )
                if path == "/api/image":
                    return self._serve_image()
                if path == "/api/error-panel":
                    return self._serve_error_panel()
                if path == "/api/summary":
                    return self._send_json(
                        HTTPStatus.OK, _summary_payload(state.artifacts_dir)
                    )
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            except KeyError as error:
                return self._error(HTTPStatus.NOT_FOUND, str(error.args[0]))
            except ValueError as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/api/identify/protocol":
                    return self._identify_protocol()
                if path == "/api/identify/upload":
                    return self._identify_upload()
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            except KeyError as error:
                return self._error(HTTPStatus.NOT_FOUND, str(error.args[0]))
            except ValueError as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))

        def _datasets(self) -> dict[str, Any]:
            info = {}
            for dataset in DATASETS:
                manifest = state.service.manifests[dataset]
                protocol = state.service.states[dataset].protocol
                ds_protocols = state.service.states[dataset].protocols
                info[dataset] = {
                    "summary": manifest.summary,
                    "protocol": protocol.name,
                    "available_protocols": [
                        {
                            "name": p.name,
                            "queries": len(p.query_ids),
                            "gallery": len(p.gallery_ids),
                            "allowed_pairs": sum(
                                len(p.allowed_gallery(q)) for q in p.query_ids
                            ),
                            "hash": p.protocol_hash,
                        }
                        for p in ds_protocols.values()
                    ],
                    "queries": len(protocol.query_ids),
                    "gallery": len(protocol.gallery_ids),
                    "label_semantics": protocol.label_semantics,
                }
            return {
                "datasets": info,
                "methods": {
                    method: {
                        "threshold": state.service.method_setting(method)[1],
                        "config": state.service.method_setting(method)[0],
                    }
                    for method in METHODS
                },
            }

        def _serve_static(self, relative: str) -> None:
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            if not target.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            return self._send_bytes(HTTPStatus.OK, target.read_bytes(), content_type)

        def _serve_image(self) -> None:
            params = self._query_params()
            dataset = params.get("dataset", "")
            image_id = params.get("id", "")
            stage = params.get("stage", "rgb")
            if dataset not in DATASETS:
                raise ValueError(f"Unknown dataset: {dataset}")
            ds_state = state.service.states[dataset]
            record = ds_state.manifest.by_id().get(image_id)
            if record is None:
                raise KeyError(f"Unknown image: {image_id}")
            path = state.service.image_path(dataset, image_id)
            rgb = read_rgb(path)
            if stage == "green":
                g = rgb[:, :, 1]
                rgb = np.stack([g, g, g], axis=-1)
            elif stage == "fov":
                proc = ds_state.repository.processed(record)
                fov_vis = np.zeros((*proc.fov.shape, 3), dtype=np.uint8)
                fov_vis[:, :, 1] = np.where(proc.fov > 0, proc.gray, 0)
                fov_vis[:, :, 0] = np.where(proc.fov > 0, proc.gray // 3, 0)
                fov_vis[:, :, 2] = np.where(proc.fov > 0, proc.gray // 2, 0)
                contours, _ = cv2.findContours(
                    proc.fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(fov_vis, contours, -1, (110, 231, 183), 3)
                rgb = fov_vis
            elif stage == "clahe":
                proc = ds_state.repository.processed(record)
                rgb = np.stack([proc.gray, proc.gray, proc.gray], axis=-1)
            return self._send_bytes(HTTPStatus.OK, _encode_jpeg(rgb), "image/jpeg")

        def _serve_error_panel(self) -> None:
            params = self._query_params()
            run_name = params.get("run", "")
            idx_str = params.get("index", "0")
            if not run_name or "/" in run_name or ".." in run_name:
                raise ValueError("Invalid run name")
            try:
                idx = int(idx_str)
            except ValueError as err:
                raise ValueError("Invalid panel index") from err
            runs_root = (state.artifacts_dir / "runs").resolve()
            panel_path = (
                runs_root / run_name / "error_cases" / f"case_{idx:03d}.jpg"
            ).resolve()
            if runs_root not in panel_path.parents or not panel_path.is_file():
                raise KeyError(f"Error panel not found: {run_name} #{idx}")
            return self._send_bytes(
                HTTPStatus.OK, panel_path.read_bytes(), "image/jpeg"
            )

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                raise ValueError("Empty request body")
            if length > MAX_UPLOAD_BYTES:
                raise ValueError("Upload exceeds 20 MB limit")
            return self.rfile.read(length)

        def _identify_protocol(self) -> None:
            params = self._query_params()
            dataset = params.get("dataset", "")
            method = params.get("method", "")
            image_id = params.get("id", "")
            protocol_name = params.get("protocol") or None
            self._check_dataset_method(dataset, method)
            if protocol_name is not None:
                ds_protocols = state.service.states[dataset].protocols
                if protocol_name not in ds_protocols:
                    raise ValueError(
                        f"Unknown protocol {protocol_name} for {dataset}"
                    )
            with state.lock:
                result = state.service.identify_protocol_query(
                    dataset, image_id, method, protocol_name=protocol_name
                )
            result["reference_thumb_url"] = _thumb_url(result.get("reference_id"), dataset)
            positives = result.get("positive_reference_ids") or []
            result["positive_thumb_url"] = (
                _thumb_url(positives[0], dataset) if positives else None
            )
            return self._send_json(HTTPStatus.OK, _jsonable(result, dataset))

        def _identify_upload(self) -> None:
            params = self._query_params()
            dataset = params.get("dataset", "")
            method = params.get("method", "")
            self._check_dataset_method(dataset, method)
            data = self._read_body()
            with state.lock:
                result = state.service.identify_upload(dataset, data, method)
            return self._send_json(HTTPStatus.OK, _jsonable(result, dataset))

    return Handler


def _thumb_url(image_id: str | None, dataset: str) -> str | None:
    if image_id is None:
        return None
    from urllib.parse import urlencode

    return "/api/image?" + urlencode({"dataset": dataset, "id": image_id})


def _jsonable(result: dict[str, Any], dataset: str) -> dict[str, Any]:
    result = dict(result)
    result["top_candidates"] = [
        {**item, "thumb_url": _thumb_url(item["image_id"], dataset)}
        for item in result.get("top_candidates", [])
    ]
    return result


def build_server(
    artifacts_dir: str | Path,
    config_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    artifacts = Path(artifacts_dir)
    service = IdentificationService.load(artifacts, config_path)
    state = AppState(service=service, artifacts_dir=artifacts, lock=Lock())
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server


def serve(
    artifacts_dir: str | Path,
    config_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The console only binds to loopback addresses")
    server = build_server(artifacts_dir, config_path, host=host, port=port)
    print(f"Retinal console: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
