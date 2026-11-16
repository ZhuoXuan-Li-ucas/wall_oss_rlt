#!/usr/bin/env python3
"""Build or resume a validation job, exporting only after verified success."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def process_identity(pid):
    try:
        # Field 22, accounting for spaces in the parenthesized command name.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main(args):
    root = Path(__file__).resolve().parents[1]
    state = {"build_pid": args.build_pid, "report": str(args.report), "bundle": str(args.bundle)}
    build_process = None
    env = dict(os.environ, WALL_REPORT=str(args.report), WALL_DATASET=str(args.dataset),
               WALL_ASSETS=str(args.assets), WALL_BUNDLE=str(args.bundle))
    # Reuse is explicit, never inherited accidentally from the caller's environment.
    env.pop("WALL_REUSE_BUILD_REPORT", None)
    env.pop("WALL_REUSE_IMAGE_ID", None)
    if args.reuse_build_report is not None:
        env.update(WALL_REUSE_BUILD_REPORT=str(args.reuse_build_report),
                   WALL_REUSE_IMAGE_ID=args.reuse_image_id)
        state.update(mode="reuse_built_image", reuse_build_report=str(args.reuse_build_report),
                     reuse_image_id=args.reuse_image_id)

    def record(status, **fields):
        state.update(status=status, checked_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = args.job_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.job_file)

    marker = args.report / "validated-image-id.txt"
    args.job_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.bundle.exists():
            raise RuntimeError("Refusing to overwrite an existing output bundle")
        if args.build_pid is None:
            if args.reuse_build_report is None:
                if not env.get("WALL_WHEELS"):
                    raise RuntimeError("Set WALL_WHEELS to the verified local wheel cache")
                if not env.get("WALL_PYTHON_BOOTSTRAP"):
                    raise RuntimeError("Set WALL_PYTHON_BOOTSTRAP to the relocated Python bootstrap")
            if args.report.exists():
                raise RuntimeError("A new validation run requires a new report directory")
            build_process = subprocess.Popen(
                ["bash", str(root / "scripts/docker_build_validate.sh")], env=env
            )
            args.build_pid = build_process.pid
            state["build_pid"] = args.build_pid
        identity = process_identity(args.build_pid)
        record("WAITING_FOR_VALIDATION")
        while not marker.is_file():
            if identity is None or process_identity(args.build_pid) != identity:
                raise RuntimeError("Build/validation process ended without a success marker; inspect build.log/runtime.log")
            time.sleep(10)
            record("WAITING_FOR_VALIDATION")
            if build_process is not None and build_process.poll() is not None and not marker.is_file():
                raise RuntimeError(f"Build/validation failed with exit {build_process.returncode}; inspect the report logs")
        if build_process is not None and build_process.wait() != 0:
            raise RuntimeError("Build/validation process did not finish successfully")
        for name in ("setup-verification.json", "dataset-verification.json"):
            report = json.loads((args.report / name).read_text())
            if report.get("status") != "PASS":
                raise RuntimeError(f"Required validation is not PASS: {name}")
        validated_image_id = marker.read_text().strip()
        if args.reuse_image_id is not None and validated_image_id != args.reuse_image_id:
            raise RuntimeError("Validated image ID differs from the explicitly requested reused image")
        record("EXPORTING", source_image_id=validated_image_id)
        with args.job_file.with_suffix(".log").open("a") as log:
            subprocess.run(["bash", str(root / "scripts/docker_export_bundle.sh")],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for required in ("image-tag.txt", "image-identity.json", "image_archive_identity.py", "SHA256SUMS"):
            if not (args.bundle / required).is_file():
                raise RuntimeError(f"Export did not provide required deployment metadata: {required}")
        portable_identity = json.loads((args.bundle / "image-identity.json").read_text())
        if portable_identity["source_image_id"] != state["source_image_id"]:
            raise RuntimeError("Export identity does not match the validated source image")
        preflight_path = root / "metadata/migration-preflight.json"
        preflight = json.loads(preflight_path.read_text())
        preflight["docker_build_and_runtime"] = {
            "status": "PASS", "scope": "component and historical smoke-data validation only",
            "image_exported": True, "source_image_id": state["source_image_id"],
            "transfer_tag": portable_identity["transfer_tag"],
            "portable_identity": str(args.bundle / "image-identity.json"),
            "evidence": str(args.report), "bundle": str(args.bundle),
        }
        preflight["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
        record("BUNDLE_READY", h100_validation="NOT_RUN", windows_validation="NOT_REQUIRED",
               server_transfer="NOT_STARTED")
        return 0
    except Exception as error:
        record("FAILED", error=str(error))
        preflight_path = root / "metadata/migration-preflight.json"
        if preflight_path.exists():
            preflight = json.loads(preflight_path.read_text())
            preflight["docker_build_and_runtime"] = {
                "status": "FAILED", "image_exported": False, "reason": str(error),
                "evidence": str(args.report), "background_job": str(args.job_file),
            }
            preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
        print(str(error), flush=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-pid", type=int, help="Resume this running build; omit to start a new build")
    parser.add_argument("--reuse-build-report", type=lambda value: Path(value).resolve(),
                        help="Reuse a completed build from this report; rerun all validation without building")
    parser.add_argument("--reuse-image-id", help="Exact sha256 image ID recorded by the completed build")
    for name in ("report", "dataset", "assets", "bundle", "job-file"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    if (args.reuse_build_report is None) != (args.reuse_image_id is None):
        parser.error("--reuse-build-report and --reuse-image-id must be supplied together")
    if args.build_pid is not None and args.reuse_build_report is not None:
        parser.error("--build-pid cannot be combined with completed-image reuse")
    raise SystemExit(main(args))
