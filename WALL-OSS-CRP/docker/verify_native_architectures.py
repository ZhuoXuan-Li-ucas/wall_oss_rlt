#!/usr/bin/env python3
"""Require H100 machine code in installed native extensions without using a GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MODULES = (
    "flash_attn_2_cuda",
    "wall_x.model.core.ops._cuda_ext_bin",
)


def extension_path(module_name):
    """Find the active binary without importing Torch or executing package code."""
    parts = module_name.split(".")
    spec = importlib.util.find_spec(parts[0])
    if spec is None:
        raise ModuleNotFoundError(parts[0])
    if len(parts) == 1:
        if not spec.origin or not any(
            spec.origin.endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            raise RuntimeError(f"Expected a native extension for {module_name}")
        binary = Path(spec.origin).resolve()
        if not binary.is_file():
            raise FileNotFoundError(binary)
        return binary
    for location in spec.submodule_search_locations or ():
        parent = Path(location).joinpath(*parts[1:-1])
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = parent / (parts[-1] + suffix)
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"No active native binary found for {module_name}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def inspect_extension(module_name, cuobjdump, required_architectures):
    result = {"module": module_name, "status": "FAIL", "commands": []}
    try:
        binary = extension_path(module_name)
        result.update(path=str(binary), bytes=binary.stat().st_size, sha256=sha256(binary))
        for label, option in (("elf", "--list-elf"), ("ptx", "--list-ptx")):
            command = [cuobjdump, "--all-fatbin", option, str(binary)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            result["commands"].append({
                "command": command, "returncode": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr,
            })
            if run.returncode:
                raise RuntimeError(f"cuobjdump {option} failed with exit {run.returncode}")
            result[f"{label}_architectures"] = sorted(set(
                re.findall(r"\b(?:sm|compute)_\d+[a-z]?\b", run.stdout)
            ))
            result[f"{label}_entries"] = len(run.stdout.splitlines())
        result["missing_architectures"] = sorted(
            set(required_architectures) - set(result["elf_architectures"])
        )
        if result["missing_architectures"]:
            raise RuntimeError(
                "Required cubin architecture missing: "
                + ", ".join(result["missing_architectures"])
            )
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main(args):
    required = args.required_arch or ["sm_90"]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "native cubin architecture coverage; no GPU kernels, model inference or training",
        "python_executable": sys.executable,
        "required_architectures": required,
        "status": "FAIL",
        "extensions": [],
    }
    try:
        cuobjdump = shutil.which(args.cuobjdump)
        if not cuobjdump and args.cuobjdump == "cuobjdump":
            candidate = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/cuobjdump"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                cuobjdump = str(candidate)
        if not cuobjdump:
            raise FileNotFoundError(f"cuobjdump executable unavailable: {args.cuobjdump}")
        report["cuobjdump"] = str(Path(cuobjdump).resolve())
        report["extensions"] = [
            inspect_extension(name, cuobjdump, required) for name in MODULES
        ]
        if all(item["status"] == "PASS" for item in report["extensions"]):
            report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    # Full command output remains in the report; keep terminal output compact.
    summary = {
        "status": report["status"], "report": str(args.output),
        "required_architectures": required,
        "extensions": [
            {key: item[key] for key in (
                "module", "status", "path", "sha256", "elf_architectures",
                "ptx_architectures", "missing_architectures", "error",
            ) if key in item}
            for item in report["extensions"]
        ],
    }
    if "error" in report:
        summary["error"] = report["error"]
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--required-arch", action="append", choices=("sm_80", "sm_86", "sm_90", "sm_120"),
                        help="Require this cubin architecture in both binaries; repeatable, default sm_90")
    raise SystemExit(main(parser.parse_args()))
