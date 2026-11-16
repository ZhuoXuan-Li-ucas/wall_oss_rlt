#!/usr/bin/env python3
"""Fetch the pinned CUDA 13.0 PyTorch wheels from official package hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.parse
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from download_artifact import download

ROOT = Path(__file__).resolve().parents[1]
TAGS = set(sys_tags())
PACKAGES = {
    "cuda-bindings": "13.0.3", "nvidia-cuda-nvrtc": "13.0.88",
    "nvidia-cuda-runtime": "13.0.96", "nvidia-cuda-cupti": "13.0.85",
    "nvidia-cudnn-cu13": "9.15.1.9", "nvidia-cublas": "13.1.0.3",
    "nvidia-cufft": "12.0.0.61", "nvidia-curand": "10.4.0.35",
    "nvidia-cusolver": "12.0.4.66", "nvidia-cusparse": "12.6.3.3",
    "nvidia-cusparselt-cu13": "0.8.0", "nvidia-nccl-cu13": "2.28.9",
    "nvidia-nvshmem-cu13": "3.4.5", "nvidia-nvtx": "13.0.85",
    "nvidia-nvjitlink": "13.0.88", "nvidia-cufile": "1.15.1.6",
    "triton": "3.6.0",
}


def metadata(item):
    name, version = item
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=45) as response:
        data = json.load(response)
    candidates = [f for f in data["urls"] if f["filename"].endswith(".whl")
                  and parse_wheel_filename(f["filename"])[3] & TAGS]
    if len(candidates) != 1:
        raise ValueError(f"Expected one compatible wheel for {name}, got {len(candidates)}")
    f = candidates[0]
    return {"name": name, "version": version, "filename": f["filename"],
            "url": f["url"], "size": f["size"], "sha256": f["digests"]["sha256"]}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.append(dict(attrs).get("href", ""))


def torch_metadata(name, version):
    index = f"https://download.pytorch.org/whl/cu130/{name}/"
    with urllib.request.urlopen(index, timeout=45) as response:
        text = response.read().decode()
    parser = Links()
    parser.feed(text)
    for link in parser.urls:
        filename = urllib.parse.unquote(urllib.parse.urlsplit(link).path.split("/")[-1])
        if not filename.endswith(".whl"):
            continue
        _, candidate_version, _, tags = parse_wheel_filename(filename)
        if str(candidate_version) == version and tags & TAGS:
            url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(index, link))
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=45) as response:
                size = int(response.headers["Content-Length"])
            return {"name": name, "version": version, "filename": filename,
                    "url": url, "size": size,
                    "sha256": urllib.parse.parse_qs(fragment)["sha256"][0]}
    raise ValueError(f"No compatible official wheel found: {name} {version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    lock = ROOT / "metadata/pytorch-wheels.json"
    if not lock.exists():
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs = list(pool.map(metadata, PACKAGES.items()))
        jobs += [torch_metadata(n, v) for n, v in (
            ("torch", "2.10.0+cu130"), ("torchvision", "0.25.0+cu130"),
            ("torchaudio", "2.10.0+cu130"))]
        lock.write_text(json.dumps(jobs, indent=2) + "\n")
    jobs = json.loads(lock.read_text())
    print(f"Pinned {len(jobs)} wheels, {sum(j['size'] for j in jobs)/1e9:.2f} GB", flush=True)
    if args.prepare:
        return
    def fetch(job):
        download(job["url"], ROOT / "wheelhouse" / job["filename"],
                 job["size"], job["sha256"], workers=8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, jobs))
    print("All pinned PyTorch CUDA wheels verified.", flush=True)


if __name__ == "__main__":
    main()
