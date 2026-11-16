#!/usr/bin/env python3
"""Verify docker-save identity across classic and containerd image stores.

Uses only the standard library, never extracts archive files, and never prints
image environment variables. The bundle's SHA256SUMS protects the whole archive;
this utility checks the selected image's config and reachable image descriptors.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile


DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
MAX_JSON = 16 * 1024 * 1024
MAX_METADATA = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_tag(value):
    require(isinstance(value, str) and re.fullmatch(
        r"[a-z0-9][a-z0-9./:_-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", value
    ), "Invalid explicit image tag")
    require(".." not in value and "//" not in value, "Invalid image tag path")
    first, separator, rest = value.partition("/")
    if not separator or not ("." in first or ":" in first or first == "localhost"):
        value = "docker.io/" + value
    if value.startswith("index.docker.io/"):
        value = "docker.io/" + value[len("index.docker.io/"):]
    if value.startswith("docker.io/") and value.count("/") == 1:
        value = "docker.io/library/" + value[len("docker.io/"):]
    return value


def same_tag(value, expected):
    try:
        return canonical_tag(value) == expected
    except ValueError:
        return False


def safe_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts,
            "Unsafe archive member path")
    return str(path)


def archive_metadata(archive):
    """One sequential pass; large layer blobs are skipped without extraction."""
    data = {}
    total = 0
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            name = safe_name(member.name)
            candidate = name.endswith(".json") or re.fullmatch(
                r"blobs/sha256/[0-9a-f]{64}", name
            )
            if not candidate or member.size > MAX_JSON:
                continue
            require(member.isfile(), "Image metadata must be a regular archive member")
            require(name not in data, "Duplicate image metadata archive member")
            source = stream.extractfile(member)
            prefix = source.read(min(member.size, 4096))
            if name.startswith("blobs/sha256/") and not prefix.lstrip().startswith((b"{", b"[")):
                # Small layer blobs share this namespace; do not keep them in RAM.
                continue
            total += member.size
            require(total <= MAX_METADATA, "Image metadata exceeds size limit")
            data[name] = prefix + source.read()
    return data


def load_json(data, name):
    require(name in data, "Archive is missing required image metadata: " + name)
    return json.loads(data[name])


def image_identity(archive, tag, source_id):
    tag = canonical_tag(tag)
    require(DIGEST.fullmatch(source_id) is not None, "Invalid source image ID")
    data = archive_metadata(archive)
    legacy = load_json(data, "manifest.json")
    require(isinstance(legacy, list) and len(legacy) == 1,
            "Expected one image in Docker-compatible manifest.json")
    entry = legacy[0]
    require(any(same_tag(value, tag) for value in entry.get("RepoTags", [])),
            "Selected transfer tag is absent from manifest.json")
    config_name = safe_name(entry["Config"])
    config = load_json(data, config_name)
    config_digest = digest(data[config_name])
    # Check known Docker/OCI content-addressed config filenames as well as bytes.
    embedded = re.fullmatch(r"(?:blobs/sha256/)?([0-9a-f]{64})(?:\.json)?", config_name)
    if embedded:
        require(config_digest == "sha256:" + embedded.group(1),
                "Config payload does not match its archive digest")
    require(config.get("os") == "linux" and config.get("architecture") == "amd64",
            "Selected image must be linux/amd64")
    require(not config.get("variant"), "Unexpected amd64 platform variant")
    rootfs = config.get("rootfs", {})
    diff_ids = rootfs.get("diff_ids")
    require(rootfs.get("type") == "layers" and isinstance(diff_ids, list)
            and all(isinstance(value, str) and DIGEST.fullmatch(value) for value in diff_ids),
            "Invalid ordered rootfs diff_ids")
    require(isinstance(entry.get("Layers"), list) and len(entry["Layers"]) == len(diff_ids),
            "Docker manifest layer count differs from config rootfs")
    accepted = {config_digest}
    descriptors = {}

    def descriptor_bytes(descriptor):
        value = descriptor.get("digest", "")
        require(isinstance(value, str) and DIGEST.fullmatch(value),
                "Invalid descriptor digest")
        name = "blobs/sha256/" + value.split(":", 1)[1]
        require(name in data, "Missing selected image descriptor payload")
        payload = data[name]
        require(digest(payload) == value and descriptor.get("size") == len(payload),
                "Descriptor payload hash or size differs")
        return value, json.loads(payload)

    def visit(descriptor, ancestors):
        value, obj = descriptor_bytes(descriptor)
        require(value not in ancestors, "Cycle in image descriptor graph")
        media_type = descriptor.get("mediaType")
        require(obj.get("schemaVersion") == 2, "Unsupported image descriptor schema")
        require(obj.get("mediaType", media_type) == media_type,
                "Image descriptor media type differs from payload")
        if media_type in MANIFEST_TYPES:
            child_digest, _ = descriptor_bytes(obj["config"])
            require(child_digest == config_digest,
                    "Selected image descriptor points to a different config")
            require(isinstance(obj.get("layers"), list) and len(obj["layers"]) == len(diff_ids),
                    "OCI manifest layer count differs from config rootfs")
        elif media_type in INDEX_TYPES:
            children = obj.get("manifests")
            require(isinstance(children, list) and children,
                    "Selected image index is empty")
            for child in children:
                # Every descendant must resolve to this same selected config.
                visit(child, ancestors | {value})
        else:
            raise ValueError("Unsupported reachable image descriptor media type")
        accepted.add(value)
        descriptors[value] = media_type

    if "index.json" in data:
        index = load_json(data, "index.json")
        require(index.get("schemaVersion") == 2, "Invalid OCI archive index")
        selected = []
        for descriptor in index.get("manifests", []):
            annotations = descriptor.get("annotations", {})
            if any(same_tag(annotations.get(key), tag) for key in (
                "io.containerd.image.name", "org.opencontainers.image.ref.name"
            )):
                selected.append(descriptor)
        require(selected, "Selected transfer tag is absent from OCI archive index")
        for descriptor in selected:
            visit(descriptor, set())

    return {
        "schema_version": 1,
        "transfer_tag": tag,
        "source_image_id": source_id,
        "platform": "linux/amd64",
        "config_digest": config_digest,
        "rootfs_diff_ids": diff_ids,
        "accepted_image_ids": sorted(accepted),
        "reachable_image_descriptors": [
            {"digest": value, "media_type": descriptors[value]}
            for value in sorted(descriptors)
        ],
    }


def check_loaded(identity, inspected):
    require(isinstance(inspected, list) and len(inspected) == 1,
            "Expected exactly one inspected image")
    image = inspected[0]
    image_id = image.get("Id")
    require(image_id in identity["accepted_image_ids"],
            "Loaded image ID is not the selected archive config/manifest/index")
    require(image.get("Os") == "linux" and image.get("Architecture") == "amd64"
            and not image.get("Variant"), "Loaded image platform differs")
    require(image.get("RootFS", {}).get("Type") == "layers"
            and image["RootFS"].get("Layers") == identity["rootfs_diff_ids"],
            "Loaded image ordered rootfs diff_ids differ")
    require(any(same_tag(value, identity["transfer_tag"]) for value in image.get("RepoTags", [])),
            "Loaded image does not retain the selected transfer tag")
    return image_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--archive", required=True)
    create.add_argument("--tag", required=True)
    create.add_argument("--source-image-id", required=True)
    create.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--identity", required=True)
    loaded = commands.add_parser("check-loaded")
    loaded.add_argument("--identity", required=True)
    loaded.add_argument("--report")
    args = parser.parse_args()
    try:
        if args.command == "create":
            result = image_identity(args.archive, args.tag, args.source_image_id)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print("Archive image identity verified: " + result["config_digest"])
        else:
            identity = json.loads(Path(args.identity).read_text())
            require(identity.get("schema_version") == 1, "Unsupported identity schema")
            if args.command == "verify":
                actual = image_identity(args.archive, identity["transfer_tag"], identity["source_image_id"])
                require(actual == identity, "Archive image identity differs from recorded metadata")
                print("Archive image identity matches: " + actual["config_digest"])
            else:
                image_id = check_loaded(identity, json.load(sys.stdin))
                if args.report:
                    Path(args.report).write_text(json.dumps({
                        "status": "PASS", "target_image_id": image_id,
                        "transfer_tag": identity["transfer_tag"],
                        "config_digest": identity["config_digest"],
                        "platform": identity["platform"],
                        "rootfs_diff_ids": identity["rootfs_diff_ids"],
                    }, indent=2) + "\n")
                print(image_id)
    except (ValueError, KeyError, TypeError, AttributeError, OSError, tarfile.TarError) as exc:
        print("Image identity verification failed: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
