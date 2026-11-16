#!/usr/bin/env python3
"""Losslessly relocate this task's unused regular wheel cache after validation."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone

source = Path('/home/pangk/workspaces/WALL-OSS-CRP/wheelhouse')
destination = Path('/media/pangk/My Passport/WALL-OSS-CRP-build-cache-20261028/legacy-cu130-wheelhouse')
log = source.parent / 'logs/legacy-cu130-wheelhouse-migration-20261028.json'
staging_source = source.with_name('.wheelhouse-verified-migration-20261028')
report = {'source': str(source), 'destination': str(destination), 'status': 'STARTED',
          'started_utc': datetime.now(timezone.utc).isoformat(), 'files': [],
          'active_build_context': '/media/pangk/My Passport/WALL-OSS-CRP-build-cache-20261028/wheelhouse',
          'original_wheelhouse_excluded_by_dockerignore': True}

def save():
    log.write_text(json.dumps(report, indent=2) + '\n')

def fingerprint(path):
    item = path.lstat()
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise RuntimeError('Unexpected file type/hard links: ' + path.name)
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

try:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError('Source must be the original regular directory')
    if destination.exists() or destination.is_symlink() or staging_source.exists():
        raise RuntimeError('Destination or temporary source path already exists; no overwrite')
    paths = sorted(source.iterdir())
    if not paths or any(p.suffix != '.whl' for p in paths):
        raise RuntimeError('Expected only flat regular wheel files')
    initial = {p.name: fingerprint(p) for p in paths}
    destination.mkdir()
    save()
    for number, path in enumerate(paths, 1):
        if fingerprint(path) != initial[path.name]:
            raise RuntimeError('Source changed before copy: ' + path.name)
        hasher = hashlib.sha256()
        with path.open('rb') as incoming, (destination / path.name).open('xb') as outgoing:
            for block in iter(lambda: incoming.read(4 * 1024 * 1024), b''):
                hasher.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if fingerprint(path) != initial[path.name]:
            raise RuntimeError('Source changed during copy: ' + path.name)
        copied_hash = hasher.hexdigest()
        destination_hash = digest(destination / path.name)
        source_hash = digest(path)
        if not copied_hash == destination_hash == source_hash:
            raise RuntimeError('Copied/source/destination SHA256 differs: ' + path.name)
        report['files'].append({'relative_path': path.name, 'bytes': initial[path.name][2],
                                'source_sha256': source_hash, 'destination_sha256': destination_hash,
                                'verified': True})
        save()
        print(f'Verified {number}/{len(paths)}: {path.name}', flush=True)
    if set(p.name for p in source.iterdir()) != set(initial):
        raise RuntimeError('Source path set changed during copy')
    if set(p.name for p in destination.iterdir()) != set(initial):
        raise RuntimeError('Destination relative path set differs')
    if any(fingerprint(source / name) != before for name, before in initial.items()):
        raise RuntimeError('Source files changed during verification')
    if any(fingerprint(destination / name)[2] != initial[name][2] for name in initial):
        raise RuntimeError('Destination files changed during verification')
    report.update(status='COPY_VERIFIED', relative_path_set_verified=True,
                  total_bytes=sum(p['bytes'] for p in report['files']))
    save()
    # Retain the complete source until the verified target is accessible through
    # the original path. If the symlink cannot be installed, restore the source.
    source.rename(staging_source)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except Exception:
        staging_source.rename(source)
        raise
    if any(fingerprint(staging_source / name) != before for name, before in initial.items()):
        source.unlink()
        staging_source.rename(source)
        raise RuntimeError('Source changed before removal; original directory restored')
    for name in sorted(initial):
        (staging_source / name).unlink()
    staging_source.rmdir()
    if source.resolve() != destination.resolve() or set(p.name for p in source.iterdir()) != set(initial):
        raise RuntimeError('Final compatibility symlink check failed')
    report.update(status='PASS', source_regular_files_removed=True,
                  compatibility_symlink_verified=True,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps({'status': report['status'], 'files': len(paths),
                      'bytes': report['total_bytes'], 'log': str(log)}, indent=2), flush=True)
except Exception as exc:
    report.update(status='STOPPED', error=str(exc))
    save()
    raise
