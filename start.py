"""Cross-platform bootstrap; only the Python standard library is needed."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up and start the retinal web console")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rebuild", action="store_true", help="rerun preparation, calibration and evaluation after changing data/config/code")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required.", file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    missing = [name for name in ("FIRE.zip", "vistel_staff_fundus.zip")
               if not (ROOT / "dataset" / name).is_file()]
    if missing:
        print(f"Place these original ZIP files in {ROOT / 'dataset'}: " + ", ".join(missing), file=sys.stderr)
        return 1
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Creating .venv ...", flush=True)
        run(sys.executable, "-m", "venv", str(ROOT / ".venv"))
    # Check the existing environment too: the launcher may use a newer interpreter.
    run(str(python), "-c", "import sys; assert sys.version_info >= (3, 11), 'Recreate .venv with Python 3.11+'")
    fingerprint = hashlib.sha256(
        (ROOT / "requirements.lock.txt").read_bytes() + (ROOT / "pyproject.toml").read_bytes()
    ).hexdigest()
    stamp = ROOT / ".venv" / ".retina-dependencies"
    if not stamp.exists() or stamp.read_text() != fingerprint:
        print("Installing dependencies (internet required) ...", flush=True)
        run(str(python), "-m", "pip", "install", "-r", "requirements.lock.txt")
        run(str(python), "-m", "pip", "install", "-e", ".", "--no-deps")
        stamp.write_text(fingerprint)
    artifacts = ROOT / "artifacts"
    preparation = ["manifests/vistel.json", "manifests/fire.json", "protocols/vistel_course.json",
                   "protocols/vistel_proxy.json", "protocols/fire_pair.json", "protocols/fire_pair_exclude_self.json"]
    prepared = args.rebuild or not all((artifacts / p).exists() for p in preparation)
    calibrated = prepared or not (artifacts / "frozen_config.json").exists()
    if prepared:
        run(str(python), "-m", "retina_auth.cli", "prepare")
    if calibrated:
        run(str(python), "-m", "retina_auth.cli", "calibrate")
    if calibrated or not (artifacts / "reports/report.json").exists():
        run(str(python), "-m", "retina_auth.cli", "evaluate")
        run(str(python), "-m", "retina_auth.cli", "report")
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.", flush=True)
    run(str(python), "-m", "retina_auth.cli", "serve", "--port", str(args.port))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Startup failed: {exc}\nFix the error above and run the launcher again.", file=sys.stderr)
        raise SystemExit(1)
