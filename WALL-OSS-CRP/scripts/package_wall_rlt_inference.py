#!/usr/bin/env python3
"""Package/verify/extract offline RTX 5070 Ti build sources, never model/data bytes.

Creation uses Git only to verify the already patched local WALL worktree. Target
verification/extraction needs Python's standard library only, and the container
build uses complete sources without .git or a dependency download.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = "sha256:2eaee50dfe782ceec5f3c29ade76fdb4b641d92229a18404f1cb5001bba353c3"
WALL_COMMIT = "ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f"
ARCHIVE = "inference-sources.tar.gz"
TOOLS = {
    "Dockerfile.inference5070": "docker/Dockerfile.inference5070",
    "build_inference5070.sh": "docker/build_inference5070.sh",
    "image_archive_identity.py": "docker/image_archive_identity.py",
    "package_wall_rlt_inference.py": "scripts/package_wall_rlt_inference.py",
}
BUNDLE_FILES = set(TOOLS) | {ARCHIVE, "MANIFEST.json", "SHA256SUMS", "base-image-identity.json"}
OMIT_DIRS = {"build", "dist", "__pycache__", "venv", "env", "node_modules", "logs", "weights"}
OMIT_SUFFIXES = {".so", ".o", ".a", ".pyc", ".pyo", ".whl", ".pt", ".pth", ".safetensors",
                 ".npz", ".npy", ".mp4", ".avi", ".pem", ".key", ".p12"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or name == "." or "\\" in name or path.is_absolute()
            or ".." in path.parts or path.as_posix() != name
            or any(ord(char) < 32 for char in name)):
        raise ValueError(f"unsafe or noncanonical path: {name!r}")
    return path


def regular_file(root, name):
    path = Path(root) / safe_relative(name).as_posix()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {name}")
    root = Path(root).resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"input escapes its source root: {name}")
    if any(parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root)):
        raise ValueError(f"symlinked input directory: {name}")
    return path


def omitted(path):
    return (any(part.startswith(".") or part in OMIT_DIRS or part.endswith(".egg-info")
                for part in path.parts)
            or path.suffix.lower() in OMIT_SUFFIXES
            or ".so." in path.name
            or path.name.lower() in {"id_rsa", "id_ed25519", "credentials", "credentials.json", "secrets.json"})


def source_files(root):
    """Retain complete ordinary source trees; prune generated/private artifacts."""
    root = Path(root).resolve()
    result = []
    for current, directories, files in os.walk(root, followlinks=False):
        relative = Path(current).relative_to(root)
        kept = []
        for name in sorted(directories):
            item = relative / name
            if omitted(item):
                continue
            if (root / item).is_symlink():
                raise ValueError(f"symlinked source directory: {item}")
            kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            item = relative / name
            if not omitted(item):
                result.append((item.as_posix(), regular_file(root, item.as_posix())))
    return result


def git(source, *args):
    return subprocess.run(["git", "-C", str(source), *args], check=True,
                          capture_output=True).stdout


def wall_sources(project, source):
    lock = json.loads(regular_file(project, "metadata/wall-rlt-sources.lock.json").read_text())
    patch = regular_file(project, "patches/wall-x-rlt-prefix.patch")
    if lock["wall_commit"] != WALL_COMMIT or sha256(patch) != lock["wall_patch_sha256"]:
        raise ValueError("WALL source lock or reviewed patch differs")
    if git(source, "rev-parse", "HEAD").decode().strip() != WALL_COMMIT:
        raise ValueError("WALL source is not the locked commit")
    if git(source, "diff", "--cached", "--binary"):
        raise ValueError("WALL source has staged changes")
    if git(source, "diff", "--binary", "HEAD", "--") != patch.read_bytes():
        raise ValueError("WALL source differs from the exact reviewed patch")
    # Restrict to tracked source: host extensions, cache files and local secrets
    # cannot be added to the image even if they are ignored by Git.
    files = []
    for raw in git(source, "ls-files", "-z").split(b"\0"):
        if raw:
            name = raw.decode()
            if not omitted(PurePosixPath(name)):
                files.append((name, regular_file(source, name)))
    names = {name for name, _ in files}
    for name in ("setup.py", "pyproject.toml", "wall_x/model/core/ops/csrc/binding.cu"):
        if name not in names:
            raise ValueError(f"WALL source incomplete: {name}")
    return files, {"commit": WALL_COMMIT, "patch_sha256": sha256(patch), "exact_patch_verified": True}


def flash_sources(source):
    files = source_files(source)
    names = {name for name, _ in files}
    for name in ("setup.py", "flash_attn/__init__.py", "csrc/flash_attn/flash_api.cpp",
                 "csrc/cutlass/include/cutlass/cutlass.h"):
        if name not in names:
            raise ValueError(f"FlashAttention source incomplete: {name}")
    if not re.search(r'__version__\s*=\s*[\'\"]2\.8\.3[\'\"]',
                     regular_file(source, "flash_attn/__init__.py").read_text()):
        raise ValueError("FlashAttention source must be version 2.8.3")
    return files


def project_files(project):
    names = set()
    for directory in ("wall_rlt", "configs/rlt", "configs/inference", "tests/rlt", "docker"):
        if not (project / directory).is_dir():
            continue
        for name, path in source_files(project / directory):
            if path.suffix in {".py", ".json", ".yaml", ".yml", ".txt", ".md", ".sh"} or path.name.startswith(("LICENSE", "Dockerfile")):
                names.add(f"{directory}/{name}")
    names.update(path.relative_to(project).as_posix()
                 for path in (project / "scripts").glob("*wall_rlt*.py"))
    names.update(path.name for path in project.glob("WALL_RLT*.md"))
    for name in ("README.md", "数据集格式说明.md", "patches/wall-x-rlt-prefix.patch",
                 "metadata/wall-rlt-sources.lock.json",
                 "metadata/rlt-top-error-selected-20261029/normalization.ar5_joint_next_v1.json",
                 "metadata/rlt-top-error-selected-20261029/episodes.json",
                 "logs/rlt-inference-20261029/cpu-check/report.json"):
        if (project / name).is_file():
            names.add(name)
    for required in ("wall_rlt/__init__.py", "docker/verify_native_architectures.py", "docker/verify_full_model.py"):
        if required not in names:
            raise ValueError(f"project source incomplete: {required}")
    return [(name, regular_file(project, name)) for name in sorted(names)]


def validate_identity(identity):
    if (identity.get("schema_version") != 1 or identity.get("config_digest") != BASE_CONFIG
            or identity.get("platform") != "linux/amd64"
            or BASE_CONFIG not in identity.get("accepted_image_ids", [])
            or not identity.get("rootfs_diff_ids")):
        raise ValueError("base identity is not the validated D0 config/platform")


def make_archive(path, files):
    members = {}
    folded = set()
    with tarfile.open(path, "w:gz") as archive:
        for name, source in sorted(files):
            safe_relative(name)
            if name.casefold() in folded:
                raise ValueError("duplicate/case-colliding archive member")
            folded.add(name.casefold())
            before = source.stat()
            info = tarfile.TarInfo(name)
            info.size = before.st_size
            info.mode = 0o755 if before.st_mode & 0o111 else 0o644
            info.mtime = 0
            digest = hashlib.sha256()

            class Reader:
                def read(self, size=-1):
                    value = stream.read(size)
                    digest.update(value)
                    return value

            with source.open("rb") as stream:
                archive.addfile(info, Reader())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"source changed during packaging: {name}")
            members[name] = {"bytes": info.size, "sha256": digest.hexdigest(), "mode": info.mode}
    return {"sha256": sha256(path), "bytes": path.stat().st_size, "members": members}


def package(project, output, *, base_identity, flash_source=None, wall_source=None):
    project, output = Path(project).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a NEW directory; existing packages are never replaced")
    if output.resolve().is_relative_to(Path(base_identity).resolve().parent):
        raise ValueError("output must be outside the immutable base bundle")
    flash_source = Path(flash_source or project / ".cache/flash-attn-2.8.3").resolve()
    wall_source = Path(wall_source or project / ".cache/wall-x-rlt").resolve()
    identity = json.loads(Path(base_identity).read_text())
    validate_identity(identity)
    wall, provenance = wall_sources(project, wall_source)
    files = [("flash-attn/" + name, path) for name, path in flash_sources(flash_source)]
    files += [("wall-x/" + name, path) for name, path in wall]
    files += [("project/" + name, path) for name, path in project_files(project)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inference-package-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for name, relative in TOOLS.items():
            shutil.copyfile(regular_file(project, relative), stage / name)
        write_json(stage / "base-image-identity.json", identity)
        record = {
            "schema_version": 1, "status": "SOURCE_PACKAGE_NOT_GPU_VALIDATED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": {"os": "Ubuntu", "gpu": "RTX 5070 Ti", "compute_capability": "12.0", "cubin": "sm_120"},
            "base_config_digest": BASE_CONFIG, "wall_source": provenance, "flash_attention_version": "2.8.3",
            "scope": "offline source build tools only; no model weights, recordings, RL actor, or training dependencies",
            "payload": make_archive(stage / ARCHIVE, files),
        }
        write_json(stage / "MANIFEST.json", record)
        (stage / "SHA256SUMS").write_text("".join(
            f"{sha256(stage / name)}  {name}\n" for name in sorted(BUNDLE_FILES - {"SHA256SUMS"})))
        verify_bundle(stage)
        stage.rename(output)
    return {"status": record["status"], "output": str(output), "files": len(BUNDLE_FILES),
            "bytes": sum(path.stat().st_size for path in output.iterdir()),
            "source_members": len(record["payload"]["members"])}


def verify_archive(path, expected, destination=None):
    folded = set()
    for name, spec in expected.items():
        relative = safe_relative(name)
        if relative.parts[0] not in {"flash-attn", "wall-x", "project"} or len(relative.parts) < 2:
            raise ValueError("unexpected archive source root")
        if name.casefold() in folded or not isinstance(spec["bytes"], int) or spec["bytes"] < 0:
            raise ValueError("invalid or duplicate archive inventory")
        folded.add(name.casefold())
        if spec.get("mode") not in {0o644, 0o755} or not re.fullmatch("[0-9a-f]{64}", spec.get("sha256", "")):
            raise ValueError("invalid archive member metadata")
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = safe_relative(member.name).as_posix()
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.linkname or member.sparse:
                raise ValueError("archive accepts ordinary regular files only, no links")
            if name in seen or name not in expected:
                raise ValueError("unexpected or duplicate archive member")
            seen.add(name)
            spec = expected[name]
            if member.size != spec["bytes"] or member.mode != spec["mode"]:
                raise ValueError("archive member size/mode differs")
            digest = hashlib.sha256()
            target = None
            try:
                if destination is not None:
                    target_path = Path(destination) / name
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target = target_path.open("xb")
                source = archive.extractfile(member)
                for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
                    digest.update(block)
                    if target is not None:
                        target.write(block)
            finally:
                if target is not None:
                    target.close()
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"archive member SHA256 differs: {name}")
            if destination is not None:
                target_path.chmod(spec["mode"])
    if seen != expected.keys():
        raise ValueError("archive member inventory is incomplete")


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    if {path.name for path in bundle.iterdir()} != BUNDLE_FILES:
        raise ValueError("bundle inventory differs; keep build results outside this directory")
    for name in BUNDLE_FILES:
        regular_file(bundle, name)
    entries = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  [A-Za-z0-9_.-]+", line):
            raise ValueError("invalid SHA256SUMS entry")
        digest, name = line.split("  ")
        if name in entries:
            raise ValueError("duplicate SHA256SUMS entry")
        entries[name] = digest
    if entries.keys() != BUNDLE_FILES - {"SHA256SUMS"}:
        raise ValueError("checksum inventory differs")
    for name, digest in entries.items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"bundle SHA256 differs: {name}")
    record = json.loads((bundle / "MANIFEST.json").read_text())
    if (record.get("schema_version") != 1 or record.get("base_config_digest") != BASE_CONFIG
            or record.get("status") != "SOURCE_PACKAGE_NOT_GPU_VALIDATED"):
        raise ValueError("unsupported inference package manifest")
    validate_identity(json.loads((bundle / "base-image-identity.json").read_text()))
    payload = record["payload"]
    if (entries[ARCHIVE] != payload["sha256"] or (bundle / ARCHIVE).stat().st_size != payload["bytes"]):
        raise ValueError("archive digest/size differs from manifest")
    verify_archive(bundle / ARCHIVE, payload["members"])
    return record


def extract_bundle(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("extraction output must be a NEW directory")
    if output.resolve().is_relative_to(bundle):
        raise ValueError("extraction output must be outside the immutable source bundle")
    record = verify_bundle(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inference-extract-", dir=output.parent) as temporary:
        stage = Path(temporary) / "context"
        stage.mkdir()
        verify_archive(bundle / ARCHIVE, record["payload"]["members"], stage)
        shutil.copyfile(bundle / "Dockerfile.inference5070", stage / "Dockerfile.inference5070")
        shutil.copyfile(bundle / "MANIFEST.json", stage / "SOURCE-MANIFEST.json")
        stage.rename(output)
    return {"status": "SOURCES_VERIFIED_AND_EXTRACTED", "output": str(output),
            "source_members": len(record["payload"]["members"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--project", type=Path, default=ROOT)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--base-identity", type=Path, required=True)
    create.add_argument("--flash-source", type=Path)
    create.add_argument("--wall-source", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--bundle", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        result = package(args.project, args.output, base_identity=args.base_identity,
                         flash_source=args.flash_source, wall_source=args.wall_source)
    elif args.command == "extract":
        result = extract_bundle(args.bundle, args.output)
    else:
        record = verify_bundle(args.bundle)
        result = {"status": "INTEGRITY_PASS", "bundle": str(args.bundle),
                  "source_members": len(record["payload"]["members"]), "gpu_validated": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
