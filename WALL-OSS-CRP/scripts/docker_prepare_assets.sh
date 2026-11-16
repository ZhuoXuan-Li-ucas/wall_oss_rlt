#!/usr/bin/env bash
# Stage immutable model/data archives while Docker installation is pending.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WALL_DATASET:?Set WALL_DATASET to the selected smoke dataset}"
: "${WALL_ASSETS:?Set WALL_ASSETS to a new staging directory on My Passport}"
test -f "$WALL_DATASET/meta/info.json"
test ! -e "$WALL_ASSETS" || { echo 'Use a new assets directory.' >&2; exit 1; }
mkdir -p "$WALL_ASSETS"
find "$TASK_ROOT/models" "$(realpath "$WALL_DATASET")" -type d -name .cache -prune -o -type f -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$WALL_ASSETS/source-assets.sha256"
realpath "$WALL_DATASET" > "$WALL_ASSETS/dataset-root.txt"
tar --exclude='*/.cache' -C "$TASK_ROOT" -cf - models | gzip -1 > "$WALL_ASSETS/models.tar.gz.partial"
mv "$WALL_ASSETS/models.tar.gz.partial" "$WALL_ASSETS/models.tar.gz"
tar --exclude='*/.cache' -C "$WALL_DATASET" -cf - . | gzip -1 > "$WALL_ASSETS/smoke-data.tar.gz.partial"
mv "$WALL_ASSETS/smoke-data.tar.gz.partial" "$WALL_ASSETS/smoke-data.tar.gz"
sha256sum -c "$WALL_ASSETS/source-assets.sha256" > "$WALL_ASSETS/source-check.log"
cp "$TASK_ROOT/model_files.manifest.json" "$TASK_ROOT/sources.lock.json" "$WALL_ASSETS/"
cp "$TASK_ROOT/metadata/migration-preflight.json" "$WALL_ASSETS/"
cp "$TASK_ROOT/docker/Verify-Bundle.ps1" "$TASK_ROOT/docker/verify_bundle.py" "$WALL_ASSETS/"
printf '%s\n' 'ASSETS ONLY: no Docker image in this directory.' \
  'Historical smoke dataset; not AR5 demonstrations.' \
  'The validated Docker image will be exported separately after Docker is installed.' \
  'Windows verification: powershell -NoProfile -File .\Verify-Bundle.ps1 -AssetsOnly' \
  > "$WALL_ASSETS/STATUS.txt"
gzip -t "$WALL_ASSETS/models.tar.gz" "$WALL_ASSETS/smoke-data.tar.gz"
python3 "$TASK_ROOT/docker/verify_bundle.py" finalize --bundle "$WALL_ASSETS" --kind assets
echo "Assets staged: $WALL_ASSETS"
