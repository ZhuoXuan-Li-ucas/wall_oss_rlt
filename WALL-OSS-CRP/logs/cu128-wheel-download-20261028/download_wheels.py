#!/usr/bin/env python3
"""Resume official, hash-verified wheels with at most eight Range requests."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CACHE = Path('/media/pangk/My Passport/WALL-OSS-CRP-build-cache-20261028/wheelhouse')
PACKAGES = json.loads((ROOT / 'download-manifest.json').read_text())
EVENTS = ROOT / 'download-events.jsonl'
FIRST = 1024 * 1024
CHUNK = 4 * FIRST
START = time.monotonic()


def emit(event):
    event['elapsed_seconds'] = round(time.monotonic() - START, 2)
    with EVENTS.open('a') as f:
        f.write(json.dumps(event) + '\n')
    print(json.dumps(event), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * FIRST), b''):
            h.update(data)
    return h.hexdigest()


def segments(package):
    size = package['size']
    yield 0, min(FIRST, size) - 1
    for start in range(FIRST, size, CHUNK):
        yield start, min(start + CHUNK, size) - 1


def part_path(package, start):
    return CACHE / (package['filename'] + '.parts') / f'{start:012d}.part'


def download(job):
    package, start, end = job
    part = part_path(package, start)
    length = end - start + 1
    if part.exists() and part.stat().st_size == length:
        return package, length, True
    part.parent.mkdir(exist_ok=True)
    temporary = part.with_suffix('.tmp')
    wanted = f'bytes {start}-{end}/{package["size"]}'
    last = {}
    for attempt in range(3):
        command = [
            'curl', '-q', '-f', '-sS', '-L', '--proxy', 'http://127.0.0.1:7897',
            '--noproxy', '', '--connect-timeout', '15', '--max-time', '90',
            '--speed-limit', '16384', '--speed-time', '25',
            '--range', f'{start}-{end}', '--max-filesize', str(length),
            '--output', str(temporary), '--dump-header', '/dev/stderr',
            '--write-out', '%{http_code} %{size_download} %{time_total}',
            package['url'],
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        fields = result.stdout.split()
        status = int(fields[0]) if fields else 0
        ranges = re.findall(r'(?im)^content-range:\s*(.*?)\s*$', result.stderr)
        observed = ranges[-1] if ranges else None
        if (result.returncode == 0 and status == 206 and observed == wanted
                and temporary.stat().st_size == length):
            temporary.replace(part)
            return package, length, False
        # Some servers return the complete small wheel despite a Range request.
        if (result.returncode == 0 and status == 200 and start == 0
                and length == package['size'] and temporary.stat().st_size == length
                and digest(temporary) == package['sha256']):
            temporary.replace(part)
            return package, length, False
        last = {'curl_rc': result.returncode, 'http_status': status,
                'observed_range': observed, 'wanted_range': wanted}
        time.sleep(1 + attempt)
    raise RuntimeError(f"{package['filename']} range {start}-{end}: {last}")


def finalize(package):
    target = CACHE / package['filename']
    if target.exists():
        if target.stat().st_size == package['size'] and digest(target) == package['sha256']:
            emit({'event': 'verified_existing', 'package': package['package'], 'filename': package['filename']})
            return
        raise RuntimeError(f'Existing complete wheel has an unexpected digest: {target.name}')
    temporary = CACHE / (package['filename'] + '.assembling')
    h = hashlib.sha256()
    with temporary.open('wb') as output:
        for start, end in segments(package):
            with part_path(package, start).open('rb') as source:
                for data in iter(lambda: source.read(FIRST), b''):
                    output.write(data)
                    h.update(data)
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != package['size'] or h.hexdigest() != package['sha256']:
        raise RuntimeError(f"Final SHA256 mismatch: {package['filename']}")
    temporary.replace(target)
    emit({'event': 'verified_complete', 'package': package['package'],
          'filename': package['filename'], 'bytes': package['size'], 'sha256': h.hexdigest()})


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    active = []
    remaining = {}
    for package in sorted(PACKAGES, key=lambda p: p['size']):
        target = CACHE / package['filename']
        if target.exists():
            finalize(package)
            continue
        jobs = [(package, start, end) for start, end in segments(package)]
        remaining[package['filename']] = len(jobs)
        active.extend(jobs)
    emit({'event': 'start', 'workers': 8, 'segments': len(active),
          'total_bytes': sum(p['size'] for p in PACKAGES)})
    completed_bytes = 0
    downloaded_bytes = 0
    last_report = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, job) for job in active]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            package, length, cached = future.result()
            completed_bytes += length
            if not cached:
                downloaded_bytes += length
            remaining[package['filename']] -= 1
            if remaining[package['filename']] == 0:
                finalize(package)
            if time.monotonic() - last_report >= 15 or count == len(futures):
                elapsed = time.monotonic() - START
                emit({'event': 'progress', 'segments_done': count, 'segments_total': len(futures),
                      'completed_bytes': completed_bytes, 'downloaded_bytes': downloaded_bytes,
                      'downloaded_bytes_per_second': round(downloaded_bytes / elapsed)})
                last_report = time.monotonic()
    emit({'event': 'all_downloads_verified'})


if __name__ == '__main__':
    main()
