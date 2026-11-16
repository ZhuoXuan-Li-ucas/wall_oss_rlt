#!/usr/bin/env python3
"""Resumable HTTP range download, with exact size and SHA256 verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def download(url, output, size, sha256, workers=8, chunk_size=32 * 1024 * 1024):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == size:
        with output.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)
        if actual == sha256:
            print(f"Already verified: {output.name}", flush=True)
            return
    partial = output.with_name(output.name + ".part")
    state_path = output.with_name(output.name + ".parts.json")
    if chunk_size <= 0 or workers <= 0:
        raise ValueError("chunk_size and workers must be positive")
    state = {"sha256": sha256, "size": size, "chunk_size": chunk_size, "done": []}
    if partial.exists() and state_path.exists():
        old = json.loads(state_path.read_text())
        if all(old.get(k) == state[k] for k in ("sha256", "size", "chunk_size")):
            state = old
    # Resolve signed redirects in memory; never write signed URLs into logs.
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=45) as response:
        resolved_url = response.url
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    total = (size + chunk_size - 1) // chunk_size
    start_time = time.monotonic()

    def fetch(index):
        first = index * chunk_size
        last = min(size - 1, first + chunk_size - 1)
        offset = first
        for attempt in range(8):
            try:
                request = urllib.request.Request(resolved_url, headers={"Range": f"bytes={offset}-{last}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {offset}-{last}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor exact HTTP range")
                    while offset <= last:
                        block = response.read(min(64 * 1024, last + 1 - offset))
                        if not block:
                            raise IOError("Incomplete HTTP range")
                        written = os.pwrite(fd, block, offset)
                        if written != len(block):
                            raise IOError("Short disk write")
                        offset += written
                    if offset != last + 1:
                        raise IOError("Incomplete HTTP range")
                with lock:
                    state["done"].append(index)
                    tmp = state_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(state))
                    tmp.replace(state_path)
                    print(f"{output.name}: {len(state['done'])}/{total} parts, {time.monotonic()-start_time:.0f}s", flush=True)
                return
            except Exception as error:
                print(f"{output.name}: part {index} attempt {attempt + 1}: {type(error).__name__}", flush=True)
                if attempt == 7:
                    raise RuntimeError(f"Failed part {index}: {type(error).__name__}") from None
                time.sleep(min(2 ** attempt, 16))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, [i for i in range(total) if i not in set(state["done"])]))
    finally:
        os.close(fd)
    with partial.open("rb") as stream:
        actual = _hash(stream)
    if actual != sha256:
        raise ValueError(f"SHA256 mismatch: {output.name}")
    partial.replace(output)
    state_path.unlink(missing_ok=True)
    print(f"Verified SHA256: {output.name}", flush=True)


def _hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("size", type=int)
    parser.add_argument("sha256")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()
    download(args.url, args.output, args.size, args.sha256, args.workers, args.chunk_size)
