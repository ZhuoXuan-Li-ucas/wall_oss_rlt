"""Portable inference sources: integrity, extraction boundaries and build selection."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import package_wall_rlt_inference as pack


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = tmp_path / "wall"
    source.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True,
                              capture_output=True).stdout
    git("init")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    for name in ("setup.py", "pyproject.toml", "wall_x/model/core/ops/csrc/binding.cu"):
        write(source / name, "original source\n")
    git("add", ".")
    git("commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(pack, "WALL_COMMIT", commit)
    write(source / "setup.py", "reviewed patched source\n")
    patch = git("diff", "--binary", "HEAD", "--")
    write(project / "patches/wall-x-rlt-prefix.patch", patch.decode())
    write(project / "metadata/wall-rlt-sources.lock.json", json.dumps({
        "wall_commit": commit, "wall_patch_sha256": hashlib.sha256(patch).hexdigest()}))
    write(source / "wall_x/model/core/ops/old.so", "do not include host binary")
    write(source / "private-secret.json", "do not include untracked content")
    flash = tmp_path / "flash"
    for name in ("setup.py", "csrc/flash_attn/flash_api.cpp", "csrc/cutlass/include/cutlass/cutlass.h"):
        write(flash / name, "source\n")
    write(flash / "flash_attn/__init__.py", '__version__ = "2.8.3"\n')
    write(flash / "build/large.o", "do not include compiled cache")
    write(flash / "flash_attn.egg-info/PKG-INFO", "do not include generated metadata")
    write(flash / ".git/config", "do not require git on target")
    write(flash / "old.so", "do not include host extension")
    for name, relative in pack.TOOLS.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    for name in ("wall_rlt/__init__.py", "docker/verify_native_architectures.py", "docker/verify_full_model.py",
                 "WALL_RLT_Implementation_Explained.md", "README.md"):
        write(project / name, "source\n")
    write(project / "wall_rlt/old.so", "excluded")
    write(project / "wall_rlt/id_rsa", "excluded")
    write(project / "models/wall-oss-0.5/model.safetensors", "excluded")
    write(project / "env/private-config.json", "excluded")
    write(project / "data/top_error/episode.npz", "excluded")
    identity = {
        "schema_version": 1, "platform": "linux/amd64", "config_digest": pack.BASE_CONFIG,
        "source_image_id": "sha256:" + "1" * 64,
        "transfer_tag": "docker.io/library/wall-oss-crp-transfer:validated-fixture",
        "accepted_image_ids": [pack.BASE_CONFIG, "sha256:" + "1" * 64],
        "rootfs_diff_ids": ["sha256:" + "2" * 64],
    }
    identity_path = write(tmp_path / "d0/identity.json", json.dumps(identity))
    return project, source, flash, identity_path, identity


def create(inputs, tmp_path):
    project, source, flash, identity_path, _ = inputs
    bundle = tmp_path / "bundle"
    result = pack.package(project, bundle, base_identity=identity_path,
                          flash_source=flash, wall_source=source)
    assert result["status"] == "SOURCE_PACKAGE_NOT_GPU_VALIDATED"
    return bundle


def test_source_package_roundtrip_excludes_models_data_and_native_outputs(inputs, tmp_path):
    bundle = create(inputs, tmp_path)
    record = pack.verify_bundle(bundle)
    names = record["payload"]["members"]
    assert "flash-attn/csrc/cutlass/include/cutlass/cutlass.h" in names
    assert "project/WALL_RLT_Implementation_Explained.md" in names
    assert "project/docker/verify_full_model.py" in names
    assert not any("/.git" in name or "/build/" in name or "egg-info" in name
                   or name.endswith((".so", ".safetensors", ".npz")) or "secret" in name
                   or "/env/" in name or "id_rsa" in name for name in names)
    extracted = tmp_path / "extracted"
    result = pack.extract_bundle(bundle, extracted)
    assert result["status"] == "SOURCES_VERIFIED_AND_EXTRACTED"
    assert (extracted / "wall-x/setup.py").read_text() == "reviewed patched source\n"
    assert (extracted / "Dockerfile.inference5070").read_bytes() == (bundle / "Dockerfile.inference5070").read_bytes()
    assert {path.name for path in bundle.iterdir()} == pack.BUNDLE_FILES
    with pytest.raises(ValueError, match="NEW directory"):
        pack.extract_bundle(bundle, extracted)
    project, source, flash, identity, _ = inputs
    with pytest.raises(ValueError, match="NEW directory"):
        pack.package(project, bundle, base_identity=identity, flash_source=flash, wall_source=source)
    with pytest.raises(ValueError, match="immutable source bundle"):
        pack.extract_bundle(bundle, bundle / "build-context")
    with pytest.raises(ValueError, match="immutable base bundle"):
        pack.package(project, identity.parent / "new-child", base_identity=identity,
                     flash_source=flash, wall_source=source)


@pytest.mark.parametrize("change", ["source", "patch", "base", "cutlass"])
def test_refuses_unreviewed_or_incomplete_inputs(inputs, tmp_path, change):
    project, source, flash, identity_path, identity = inputs
    if change == "source":
        write(source / "setup.py", "unexpected change\n")
    elif change == "patch":
        write(project / "patches/wall-x-rlt-prefix.patch", "unreviewed")
    elif change == "base":
        identity["config_digest"] = "sha256:" + "3" * 64
        write(identity_path, json.dumps(identity))
    else:
        (flash / "csrc/cutlass/include/cutlass/cutlass.h").unlink()
    with pytest.raises(ValueError):
        pack.package(project, tmp_path / "output", base_identity=identity_path,
                     flash_source=flash, wall_source=source)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("change", ["bytes", "extra", "symlink", "duplicate_checksum"])
def test_bundle_integrity_fails_before_extraction(inputs, tmp_path, change):
    bundle = create(inputs, tmp_path)
    if change == "bytes":
        with (bundle / pack.ARCHIVE).open("ab") as target:
            target.write(b"damage")
    elif change == "extra":
        write(bundle / "unlisted.txt", "extra")
    elif change == "symlink":
        (bundle / "Dockerfile.inference5070").unlink()
        (bundle / "Dockerfile.inference5070").symlink_to(ROOT / "docker/Dockerfile.inference5070")
    else:
        content = (bundle / "SHA256SUMS").read_text()
        write(bundle / "SHA256SUMS", content + content.splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        pack.extract_bundle(bundle, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "duplicate", "mismatch", "case_collision"])
def test_untrusted_tar_members_never_escape_new_context(tmp_path, kind):
    archive_path = tmp_path / "untrusted.tar.gz"
    payload = b"source"
    name = {"traversal": "../escape", "absolute": "/escape"}.get(kind, "project/source.py")
    info = tarfile.TarInfo(name)
    info.mode = 0o644
    info.size = len(payload)
    if kind in {"symlink", "hardlink"}:
        info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        info.linkname = "../../escape"
        info.size = 0
    expected = {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": info.size, "mode": 0o644}}
    if kind == "mismatch":
        expected[name]["sha256"] = "0" * 64
    if kind == "case_collision":
        expected["project/SOURCE.py"] = dict(expected[name])
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
        if kind == "duplicate":
            archive.addfile(info, io.BytesIO(payload))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError):
        pack.verify_archive(archive_path, expected, output)
    assert not (tmp_path / "escape").exists()


def fake_docker(tmp_path, identity, *, exists=False, wrong=False):
    binary = tmp_path / "bin/docker"
    inspected = [{"Id": identity["config_digest"], "Os": "linux", "Architecture": "amd64",
                  "RootFS": {"Type": "layers", "Layers": identity["rootfs_diff_ids"]},
                  "RepoTags": [identity["transfer_tag"]]}]
    if wrong:
        inspected[0]["RootFS"]["Layers"] = ["sha256:" + "9" * 64]
    metadata = write(tmp_path / "inspect.json", json.dumps(inspected))
    log = tmp_path / "docker-calls.jsonl"
    write(binary, "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
          f"with Path({str(log)!r}).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
          "a=sys.argv[1:]\n"
          "if a[0]=='info': print('29.0'); sys.exit(0)\n"
          f"if a[:2]==['image','inspect'] and a[2]=='candidate:test': sys.exit({0 if exists else 1})\n"
          f"if a[:2]==['image','inspect']: print(Path({str(metadata)!r}).read_text()); sys.exit(0)\n"
          "if a[0]=='build': print('mock build invocation, no actual image'); sys.exit(0)\n"
          "sys.exit(90)\n")
    binary.chmod(0o755)
    env = dict(os.environ, PATH=str(binary.parent) + os.pathsep + os.environ["PATH"])
    return env, log


def test_build_entrypoint_pins_verified_local_base_and_disables_network(inputs, tmp_path):
    bundle = create(inputs, tmp_path)
    env, log = fake_docker(tmp_path, inputs[-1])
    context = tmp_path / "build-context"
    run = subprocess.run(["bash", str(bundle / "build_inference5070.sh"),
        "--base-image", "my-local-validated-d0:tag", "--tag", "candidate:test",
        "--context-dir", str(context)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    build = next(call for call in calls if call[0] == "build")
    assert "--pull=false" in build and "--network=none" in build
    assert "BASE_IMAGE=" + pack.BASE_CONFIG in build
    assert "GPU kernel/inference acceptance remains pending" in run.stdout
    assert (context / "SOURCE-MANIFEST.json").is_file()


@pytest.mark.parametrize("failure", ["wrong_base", "existing_tag", "missing_base"])
def test_build_entrypoint_stops_before_build_on_wrong_selection(inputs, tmp_path, failure):
    bundle = create(inputs, tmp_path)
    env, log = fake_docker(tmp_path, inputs[-1], exists=failure == "existing_tag", wrong=failure == "wrong_base")
    command = ["bash", str(bundle / "build_inference5070.sh"), "--tag", "candidate:test"]
    if failure != "missing_base":
        command += ["--base-image", "local:base"]
    run = subprocess.run(command, env=env, capture_output=True, text=True)
    assert run.returncode != 0
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    assert not any(call[0] == "build" for call in calls)


def test_native_helper_cli_accepts_sm120_and_records_static_failure(tmp_path):
    """Exercise the real CLI without requiring Torch, CUDA, cuobjdump or a GPU."""
    report = tmp_path / "native.json"
    run = subprocess.run([sys.executable, str(ROOT / "docker/verify_native_architectures.py"),
        "--required-arch", "sm_120", "--cuobjdump", str(tmp_path / "absent-cuobjdump"),
        "--output", str(report)], capture_output=True, text=True)
    assert run.returncode == 1  # Accepted CLI, explicit compatibility failure.
    value = json.loads(report.read_text())
    assert value["required_architectures"] == ["sm_120"]
    assert value["status"] == "FAIL" and value["extensions"] == []
    assert "cuobjdump executable unavailable" in value["error"]
