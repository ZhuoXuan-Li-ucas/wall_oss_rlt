#!/usr/bin/env python3
"""Download pinned public inference assets and verify them against HF metadata."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("all", "checkpoint", "processor"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = []
    for metadata_name, dirname, processor_only in [
        ("wall-model-api.json", "wall-oss-0.5", False),
        ("qwen-processor-api.json", "Qwen2.5-VL-3B-Instruct", True),
    ]:
        if args.component == "processor" and not processor_only:
            continue
        if args.component == "checkpoint" and processor_only:
            continue
        metadata = json.loads((ROOT / "metadata" / metadata_name).read_text())
        files = [
            f for f in metadata["siblings"]
            if not (processor_only and (f["rfilename"].endswith(".safetensors")
                    or f["rfilename"].endswith(".safetensors.index.json")))
        ]
        target = ROOT / "models" / dirname
        if not args.verify_only:
            snapshot_download(
                repo_id=metadata["id"], revision=metadata["sha"],
                local_dir=target, allow_patterns=[f["rfilename"] for f in files],
                max_workers=2,
            )
        for entry in files:
            path = target / entry["rfilename"]
            size = path.stat().st_size
            if size != entry["size"]:
                raise ValueError(f"File size mismatch: {path}")
            sha256 = hashlib.sha256()
            blob = hashlib.sha1(f"blob {size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    sha256.update(block)
                    blob.update(block)
            expected = entry.get("lfs", {}).get("sha256")
            if expected and sha256.hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {path}")
            if not expected and blob.hexdigest() != entry["blobId"]:
                raise ValueError(f"Git blob SHA1 mismatch: {path}")
            manifest.append({
                "path": str(path.relative_to(ROOT)), "size_bytes": size,
                "sha256": sha256.hexdigest(), "repo_id": metadata["id"],
                "revision": metadata["sha"],
                "verified_against": "HF LFS SHA256" if expected else "HF Git blob SHA1",
                "status": "PASS",
            })
            print(f"Verified {path.relative_to(ROOT)} ({size:,} bytes)", flush=True)
    manifest_path = ROOT / "model_files.manifest.json"
    if manifest_path.exists():
        updated_repos = {entry["repo_id"] for entry in manifest}
        manifest += [entry for entry in json.loads(manifest_path.read_text())
                     if entry["repo_id"] not in updated_repos]
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("All selected model and processor files verified.", flush=True)


if __name__ == "__main__":
    main()
