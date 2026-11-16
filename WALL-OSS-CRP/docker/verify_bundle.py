#!/usr/bin/env python3
"""Create or verify a complete transfer/assets bundle using only the stdlib."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "walloss-offline-bundle"
MANIFEST = "SHA256SUMS"
READY = "BUNDLE_READY.json"
COMMON = ["models.tar.gz", "smoke-data.tar.gz", "model_files.manifest.json",
          "sources.lock.json", "verify_bundle.py"]
TRANSFER = ["environment-image.tar.gz", "image-id.txt", "image-tag.txt",
            "image-identity.json", "image_archive_identity.py",
            "verify_native_architectures.py", "verify_h100_bundle.sh", "OFFLINE-DEPLOY.md"]
REPORTS = ["build.log", "image-inspect.json", "runtime.log", "setup-verification.json",
           "dataset-verification.json", "dependencies.json", "validated-dataset-root.txt",
           "validated-assets.sha256", "validated-image-id.txt", "native-architectures.json",
           "native-architectures.log"]
ASSETS = ["source-assets.sha256", "dataset-root.txt", "source-check.log",
          "migration-preflight.json", "STATUS.txt", "Verify-Bundle.ps1"]


def required_files(kind):
    return sorted(COMMON + (TRANSFER + ["reports/" + name for name in REPORTS]
                            if kind == "transfer" else ASSETS))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def valid_path(name):
    parts = name.split("/")
    if (not name or "\\" in name or ":" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in parts)
            or any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
                   for part in parts)):
        raise ValueError(f"Unsafe or nonportable bundle path: {name!r}")
    if name.lower().endswith(".partial"):
        raise ValueError(f"Incomplete temporary file remains: {name}")


def inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Bundle must be a real directory")
    names, case_names = set(), set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            name = path.relative_to(root).as_posix()
            valid_path(name)
            if path.is_symlink():
                raise ValueError(f"Symlink is not allowed in a portable bundle: {name}")
            folded = name.casefold()
            if folded in case_names:
                raise ValueError(f"Case-insensitive duplicate bundle path: {name}")
            case_names.add(folded)
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                if name not in (MANIFEST, READY):
                    names.add(name)
            else:
                raise ValueError(f"Non-regular bundle file: {name}")
    return names


def parse_manifest(raw):
    records, case_names = {}, set()
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Invalid SHA256SUMS line")
        value, name = match.groups()
        valid_path(name)
        if name in (MANIFEST, READY) or name.casefold() in case_names:
            raise ValueError(f"Duplicate or reserved manifest path: {name}")
        records[name] = value
        case_names.add(name.casefold())
    if not records:
        raise ValueError("SHA256SUMS must not be empty")
    return records


def identity(root, records, kind):
    if kind == "assets":
        return None
    data = json.loads((root / "image-identity.json").read_text())
    source = data["source_image_id"]
    tag = data["transfer_tag"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", source) or not isinstance(tag, str) or not tag:
        raise ValueError("Invalid source image identity")
    if ((root / "image-id.txt").read_text().strip() != source
            or (root / "reports/validated-image-id.txt").read_text().strip() != source
            or (root / "image-tag.txt").read_text().strip() != tag):
        raise ValueError("Source image identity files disagree")
    return {"filename": "image-identity.json", "sha256": records["image-identity.json"],
            "source_image_id": source, "transfer_tag": tag}


def expected_archives(records, kind):
    names = ["models.tar.gz", "smoke-data.tar.gz"]
    if kind == "transfer":
        names.append("environment-image.tar.gz")
    return {name: records[name] for name in sorted(names)}


def verify(root, kind):
    actual = inventory(root)
    marker = json.loads((root / READY).read_text())
    if (marker.get("schema") != SCHEMA or marker.get("version") != 1
            or marker.get("status") != "READY" or marker.get("package_kind") != kind):
        raise ValueError("Missing or unsupported complete-bundle marker")
    raw = (root / MANIFEST).read_bytes()
    records = parse_manifest(raw)
    expected = required_files(kind)
    if marker.get("required_files") != expected or not set(expected).issubset(records):
        raise ValueError("Bundle is missing its required file set")
    if set(records) != actual:
        raise ValueError(f"Manifest differs from directory: missing={sorted(set(records) - actual)}, "
                         f"unlisted={sorted(actual - set(records))}")
    if marker.get("manifest") != {"filename": MANIFEST, "sha256": hashlib.sha256(raw).hexdigest(),
                                  "file_count": len(records)}:
        raise ValueError("Manifest digest/count differs from completion marker")
    if marker.get("archives") != expected_archives(records, kind):
        raise ValueError("Archive checksums differ from completion marker")
    if marker.get("image_identity") != identity(root, records, kind):
        raise ValueError("Image identity differs from completion marker")
    for name, expected_digest in records.items():
        if digest(root / name) != expected_digest:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Complete {kind} bundle verified: {len(records)} files; {root}")


def finalize(root, kind):
    if (root / MANIFEST).exists() or (root / READY).exists():
        raise ValueError("Refusing to overwrite an existing manifest/completion marker")
    actual = inventory(root)
    required = required_files(kind)
    if not set(required).issubset(actual):
        raise ValueError(f"Missing required files: {sorted(set(required) - actual)}")
    records = {name: digest(root / name) for name in sorted(actual)}
    raw = "".join(f"{value}  {name}\n" for name, value in records.items()).encode("utf-8")
    marker = {"schema": SCHEMA, "version": 1, "status": "READY", "package_kind": kind,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest": {"filename": MANIFEST, "sha256": hashlib.sha256(raw).hexdigest(),
                           "file_count": len(records)},
              "required_files": required, "archives": expected_archives(records, kind),
              "image_identity": identity(root, records, kind)}
    temporary = root / (MANIFEST + ".partial")
    temporary.write_bytes(raw)
    temporary.replace(root / MANIFEST)
    # Check the entire published manifest before publishing the success marker.
    # A killed process leaves no ready marker (and may leave a .partial file).
    if inventory(root) != actual:
        raise ValueError("Bundle file set changed while finalizing")
    for name, expected_digest in records.items():
        if digest(root / name) != expected_digest:
            raise ValueError(f"Checksum changed while finalizing: {name}")
    temporary = root / (READY + ".partial")
    temporary.write_text(json.dumps(marker, indent=2) + "\n")
    temporary.replace(root / READY)
    print(f"Complete {kind} bundle finalized: {len(records)} files; {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("finalize", "verify"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--kind", choices=("transfer", "assets"), default="transfer")
    args = parser.parse_args()
    try:
        (finalize if args.action == "finalize" else verify)(args.bundle.absolute(), args.kind)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Bundle verification failed: {error}")
