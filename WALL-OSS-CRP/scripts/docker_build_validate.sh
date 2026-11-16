#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WALL_DATASET:?Set WALL_DATASET to the smoke dataset directory}"
WALL_REUSE_BUILD_REPORT="${WALL_REUSE_BUILD_REPORT:-}"
WALL_REUSE_IMAGE_ID="${WALL_REUSE_IMAGE_ID:-}"
if [[ -n $WALL_REUSE_BUILD_REPORT || -n $WALL_REUSE_IMAGE_ID ]]; then
  : "${WALL_REUSE_BUILD_REPORT:?Set both WALL_REUSE_BUILD_REPORT and WALL_REUSE_IMAGE_ID}"
  : "${WALL_REUSE_IMAGE_ID:?Set both WALL_REUSE_BUILD_REPORT and WALL_REUSE_IMAGE_ID}"
  [[ $WALL_REUSE_IMAGE_ID =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'An exact sha256 image ID is required for reuse.' >&2; exit 1; }
else
  : "${WALL_WHEELS:?Set WALL_WHEELS to the verified offline wheel cache}"
  : "${WALL_PYTHON_BOOTSTRAP:?Set WALL_PYTHON_BOOTSTRAP to the relocated Python bootstrap directory}"
fi
WALL_IMAGE="${WALL_IMAGE:-walloss-crp:cu128-validation-20261028}"
WALL_GPU="${WALL_GPU:-0}"
WALL_REPORT="${WALL_REPORT:-$TASK_ROOT/logs/docker-validation-$(date -u +%Y%m%dT%H%M%SZ)}"
command -v docker >/dev/null || { echo 'Docker is not installed.' >&2; exit 1; }
docker info >/dev/null
test -f "$WALL_DATASET/meta/info.json"
test -f "$TASK_ROOT/models/wall-oss-0.5/model.safetensors"
test ! -e "$WALL_REPORT" || { echo 'Validation requires a new report directory.' >&2; exit 1; }
mkdir -p "$WALL_REPORT"
if [[ -n $WALL_REUSE_BUILD_REPORT ]]; then
  # Preserve original build evidence, but never reuse prior runtime results.
  python3 - "$WALL_REUSE_BUILD_REPORT" "$WALL_REUSE_IMAGE_ID" "$WALL_REPORT" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

source, image_id, target = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
build_bytes = (source / 'build.log').read_bytes()
inspect_bytes = (source / 'image-inspect.json').read_bytes()
images = json.loads(inspect_bytes)
if (not isinstance(images, list) or len(images) != 1
        or images[0].get('Id') != image_id
        or images[0].get('Os') != 'linux' or images[0].get('Architecture') != 'amd64'):
    raise SystemExit('Reuse image ID/platform does not match the original image-inspect.json')
log = build_bytes.decode('utf-8')
confirmed = False
for stage in re.findall(r'^#(\d+) exporting to image\s*$', log, re.MULTILINE):
    export = re.search(rf'^#{stage} exporting (?:manifest(?: list)?|config) {re.escape(image_id)}(?:\s|$)', log, re.MULTILINE)
    done = re.search(rf'^#{stage} DONE(?:\s|$)', log, re.MULTILINE)
    if export and done and done.start() > export.start():
        confirmed = True
if not confirmed:
    raise SystemExit('Original build.log has no completed image export matching the exact image ID')
(target / 'build.log').write_bytes(build_bytes)
(target / 'reused-build-image-inspect.json').write_bytes(inspect_bytes)
if (source / 'asset-archive-portability.json').is_file():
    shutil.copyfile(source / 'asset-archive-portability.json', target / 'asset-archive-portability.json')
provenance = {
    'mode': 'reuse_built_image', 'source_report': str(source), 'source_image_id': image_id,
    'build_log_sha256': hashlib.sha256(build_bytes).hexdigest(),
    'source_image_inspect_sha256': hashlib.sha256(inspect_bytes).hexdigest(),
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'build evidence only; component, dataset, dependency and source-asset validation rerun',
}
(target / 'build-reuse.json').write_text(json.dumps(provenance, indent=2) + '\n')
PY
  WALL_IMAGE_ID="$WALL_REUSE_IMAGE_ID"
  docker image inspect "$WALL_IMAGE_ID" > "$WALL_REPORT/image-inspect.json"
  python3 - "$WALL_REPORT/image-inspect.json" "$WALL_IMAGE_ID" <<'PY'
import json
from pathlib import Path
import sys
images = json.loads(Path(sys.argv[1]).read_text())
if (not isinstance(images, list) or len(images) != 1
        or images[0].get('Id') != sys.argv[2]
        or images[0].get('Os') != 'linux' or images[0].get('Architecture') != 'amd64'):
    raise SystemExit('The locally available image does not match the original exact image ID/platform')
PY
else
  # All build steps use local verified inputs with network disabled.
  docker build --pull=false --platform linux/amd64 --provenance=false --network none \
    --build-context "wall_wheels=$WALL_WHEELS" \
    --build-context "wall_python=$WALL_PYTHON_BOOTSTRAP" \
    -f "$TASK_ROOT/docker/Dockerfile.h100" \
    -t "$WALL_IMAGE" "$TASK_ROOT" 2>&1 | tee "$WALL_REPORT/build.log"
  docker image inspect "$WALL_IMAGE" > "$WALL_REPORT/image-inspect.json"
  WALL_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$WALL_IMAGE")"
fi
docker run --rm --pull=never --network none --gpus "device=$WALL_GPU" \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  -v "$TASK_ROOT/models:/opt/walloss/models:ro" \
  -v "$WALL_DATASET:/data/smoke:ro" -v "$WALL_REPORT:/reports" \
  "$WALL_IMAGE_ID" bash -euc '
    python scripts/verify_setup.py --output /reports/setup-verification.json
    python docker/verify_dataset.py --root /data/smoke --output /reports/dataset-verification.json
    python docker/check_dependencies.py --output /reports/dependencies.json
  ' 2>&1 | tee "$WALL_REPORT/runtime.log"
realpath "$WALL_DATASET" > "$WALL_REPORT/validated-dataset-root.txt"
find "$TASK_ROOT/models" "$(realpath "$WALL_DATASET")" -type d -name .cache -prune -o -type f -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$WALL_REPORT/validated-assets.sha256"
printf '%s\n' "$WALL_IMAGE_ID" > "$WALL_REPORT/validated-image-id.txt"
echo "Component and dataset checks passed: $WALL_REPORT"
echo 'Full model inference, H100 execution and training remain unverified.'
