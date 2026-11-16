#!/usr/bin/env bash
# Offline component checks only. Full-model inference is a separate explicit step.
set -euo pipefail
WALL_BUNDLE="${WALL_BUNDLE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${WALL_RUN:?Set WALL_RUN to a new directory for extracted assets and H100 reports}"
WALL_GPU="${WALL_GPU:-0}"
for command_name in docker nvidia-smi sha256sum gzip tar realpath python3; do
  command -v "$command_name" >/dev/null || {
    echo "Missing host command: $command_name. Prepare the Ubuntu host before offline validation." >&2
    echo 'This bundle does not install Docker, NVIDIA Container Toolkit or the NVIDIA driver.' >&2
    exit 1
  }
done
WALL_BUNDLE="$(realpath -- "$WALL_BUNDLE")"
WALL_RUN="$(realpath -m -- "$WALL_RUN")"
[[ ! -e $WALL_RUN && ! -L $WALL_RUN ]] || {
  echo "Refusing to overwrite existing run directory: $WALL_RUN" >&2
  exit 1
}
cd "$WALL_BUNDLE"
for required in BUNDLE_READY.json SHA256SUMS verify_bundle.py image-id.txt image-tag.txt image-identity.json image_archive_identity.py verify_native_architectures.py \
  environment-image.tar.gz models.tar.gz smoke-data.tar.gz; do
  test -f "$required" || { echo "Incomplete bundle: missing $required" >&2; exit 1; }
done
python3 verify_bundle.py verify --bundle "$WALL_BUNDLE"
gzip -t environment-image.tar.gz models.tar.gz smoke-data.tar.gz
python3 image_archive_identity.py verify --archive environment-image.tar.gz --identity image-identity.json
WALL_IMAGE_TAG="$(python3 - <<'PY'
import json
from pathlib import Path
identity = json.loads(Path('image-identity.json').read_text())
if Path('image-id.txt').read_text().strip() != identity['source_image_id']:
    raise SystemExit('Source image ID differs from archive identity metadata.')
if Path('image-tag.txt').read_text().strip() != identity['transfer_tag']:
    raise SystemExit('Transfer tag differs from archive identity metadata.')
print(identity['transfer_tag'])
PY
)"
docker info >/dev/null || {
  echo 'Docker is unavailable to this user. Check the daemon and existing access permissions.' >&2
  exit 1
}
nvidia-smi >/dev/null || {
  echo 'The host NVIDIA driver is not working. Resolve host GPU access before validation.' >&2
  exit 1
}
# mkdir without -p rejects existing directories, including concurrent attempts.
mkdir -- "$WALL_RUN"
mkdir -p "$WALL_RUN/data/smoke" "$WALL_RUN/reports"
cp image-id.txt "$WALL_RUN/reports/source-image-id.txt"
cp image-tag.txt image-identity.json "$WALL_RUN/reports/"
nvidia-smi > "$WALL_RUN/reports/host-nvidia-smi.txt"
docker version > "$WALL_RUN/reports/host-docker-version.txt"
docker info --format '{{json .Runtimes}}' > "$WALL_RUN/reports/host-docker-runtimes.json"
docker load -i environment-image.tar.gz 2>&1 | tee "$WALL_RUN/reports/docker-load.log"
WALL_IMAGE_ID="$(docker image inspect "$WALL_IMAGE_TAG" | \
  python3 image_archive_identity.py check-loaded --identity image-identity.json \
  --report "$WALL_RUN/reports/loaded-image-identity.json")"
printf '%s\n' "$WALL_IMAGE_ID" > "$WALL_RUN/reports/image-id.txt"
tar --no-same-owner -xzf models.tar.gz -C "$WALL_RUN"
tar --no-same-owner -xzf smoke-data.tar.gz -C "$WALL_RUN/data/smoke"
test -f "$WALL_RUN/models/wall-oss-0.5/model.safetensors"
test -f "$WALL_RUN/data/smoke/meta/info.json"
if ! docker run --rm --pull=never --network none --gpus "device=$WALL_GPU" \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  -v "$WALL_RUN/models:/opt/walloss/models:ro" \
  -v "$WALL_RUN/data/smoke:/data/smoke:ro" \
  -v "$WALL_RUN/reports:/reports" \
  -v "$WALL_BUNDLE/verify_native_architectures.py:/checks/verify_native_architectures.py:ro" \
  "$WALL_IMAGE_ID" bash -euc '
    python /checks/verify_native_architectures.py --output /reports/native-architectures.json
    python scripts/verify_setup.py --output /reports/setup-verification.json
    python docker/verify_dataset.py --root /data/smoke --output /reports/dataset-verification.json
    python docker/check_dependencies.py --output /reports/dependencies.json
  ' 2>&1 | tee "$WALL_RUN/reports/components.log"; then
  echo "H100 component validation failed. See $WALL_RUN/reports/components.log" >&2
  echo 'For GPU runtime errors, check the existing NVIDIA Container Toolkit/runtime configuration.' >&2
  exit 1
fi
printf '%s\n' "$WALL_IMAGE_ID" > "$WALL_RUN/reports/components-passed-image-id.txt"
echo "Offline component and smoke-data checks passed: $WALL_RUN/reports"
echo 'Known dependency declaration failures remain recorded separately in dependencies.json.'
echo 'Full-model inference, service round trips, training and robot operation were NOT run.'
