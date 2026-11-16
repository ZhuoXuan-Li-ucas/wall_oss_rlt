#!/usr/bin/env python3
"""Recover an exact official CUDA base from matching blobs in a local image archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        self.bytes += len(data)
        return data


def recover(args):
    manifest_data = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_data).hexdigest()
    if manifest_hash != args.manifest_sha256:
        raise ValueError("Official manifest hash mismatch")
    manifest = json.loads(manifest_data)
    config_data = args.config.read_bytes()
    config_hash = hashlib.sha256(config_data).hexdigest()
    if config_hash != manifest["config"]["digest"].removeprefix("sha256:"):
        raise ValueError("Official config hash mismatch")
    if len(config_data) != manifest["config"]["size"]:
        raise ValueError("Official config size mismatch")
    config = json.loads(config_data)
    if (config["os"], config["architecture"]) != ("linux", "amd64"):
        raise ValueError("Expected linux/amd64 base")
    if len(config["rootfs"]["diff_ids"]) != len(manifest["layers"]):
        raise ValueError("Layer count mismatch")
    if args.output.exists() or args.output.with_suffix(".partial").exists():
        raise FileExistsError("Use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial")
    layers = {"blobs/sha256/" + item["digest"].split(":", 1)[1]: item for item in manifest["layers"]}
    needed = set(layers)
    verified = []
    config_name = "blobs/sha256/" + config_hash
    manifest_name = "blobs/sha256/" + manifest_hash
    docker_manifest = [{"Config": config_name, "RepoTags": [args.tag], "Layers": list(layers)}]
    registry_tag = args.tag if args.tag.startswith("docker.io/") else "docker.io/" + args.tag
    index = {"schemaVersion": 2, "manifests": [{
        "mediaType": manifest["mediaType"], "digest": "sha256:" + manifest_hash,
        "size": len(manifest_data), "platform": {"os": "linux", "architecture": "amd64"},
        "annotations": {"io.containerd.image.name": registry_tag, "org.opencontainers.image.ref.name": args.tag},
    }]}
    report = {
        "status": "IN_PROGRESS", "source": str(args.source), "output": str(args.output),
        "official_manifest_sha256": manifest_hash, "image_id": "sha256:" + config_hash,
        "tag": args.tag, "verified_layers": verified,
        "scope": "Official CUDA base only; no layers or configuration from the old project",
    }

    def save_report():
        report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_report()
    try:
        with tarfile.open(partial, "w", copybufsize=1024 * 1024) as output:
            for name, data in [(config_name, config_data), (manifest_name, manifest_data),
                               ("manifest.json", json.dumps(docker_manifest).encode()),
                               ("index.json", json.dumps(index).encode()),
                               ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}')]:
                item = tarfile.TarInfo(name)
                item.size = len(data)
                output.addfile(item, io.BytesIO(data))
            with tarfile.open(args.source, "r|gz", bufsize=1024 * 1024) as source:
                for member in source:
                    if member.name not in needed:
                        continue
                    expected = layers[member.name]
                    if not member.isfile() or member.size != expected["size"]:
                        raise ValueError(f"Source layer type/size mismatch: {member.name}")
                    reader = HashedReader(source.extractfile(member))
                    clean = tarfile.TarInfo(member.name)
                    clean.size = member.size
                    output.addfile(clean, reader)
                    if reader.bytes != member.size or "sha256:" + reader.hash.hexdigest() != expected["digest"]:
                        raise ValueError(f"Layer SHA256 mismatch: {member.name}")
                    verified.append({"digest": expected["digest"], "size": reader.bytes})
                    needed.remove(member.name)
                    save_report()
                    print(f"Verified {len(verified)}/{len(layers)} layers: {reader.bytes} bytes", flush=True)
                    if not needed:
                        break
            if needed:
                raise ValueError(f"Missing {len(needed)} official base layers")
        partial.replace(args.output)
        report["status"] = "PASS"
        report["archive_size"] = args.output.stat().st_size
        save_report()
        print(f"Recovered exact official base: {args.output}", flush=True)
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        save_report()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest", "config", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tag", default="nvidia/cuda:12.8.1-devel-ubuntu22.04")
    recover(parser.parse_args())
