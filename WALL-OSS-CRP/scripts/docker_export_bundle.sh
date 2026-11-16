#!/usr/bin/env bash
# Export only the exact image whose component checks have succeeded.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WALL_REPORT:?Set WALL_REPORT to the successful docker-validation report directory}"
: "${WALL_DATASET:?Set WALL_DATASET to the same smoke dataset used for validation}"
: "${WALL_BUNDLE:?Set WALL_BUNDLE to a new directory on My Passport}"
test -f "$WALL_REPORT/validated-image-id.txt"
test -f "$WALL_DATASET/meta/info.json"
test "$(realpath "$WALL_DATASET")" = "$(cat "$WALL_REPORT/validated-dataset-root.txt")"
sha256sum -c "$WALL_REPORT/validated-assets.sha256"
test ! -e "$WALL_BUNDLE" || { echo 'Use a new bundle directory.' >&2; exit 1; }
WALL_IMAGE_ID="$(cat "$WALL_REPORT/validated-image-id.txt")"
[[ $WALL_IMAGE_ID =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Invalid validated source image ID.' >&2; exit 1; }
[[ $(docker image inspect --format '{{.Id}}' "$WALL_IMAGE_ID") == "$WALL_IMAGE_ID" ]]
# Local GPU execution alone cannot prove that the rebuilt binaries contain H100
# machine code. Inspect the exact validated image before creating the bundle.
test -f "$TASK_ROOT/docker/verify_native_architectures.py"
docker run --rm --pull=never --network none \
  --user "$(id -u):$(id -g)" \
  -v "$TASK_ROOT/docker/verify_native_architectures.py:/checks/verify_native_architectures.py:ro" \
  -v "$WALL_REPORT:/reports" \
  "$WALL_IMAGE_ID" python /checks/verify_native_architectures.py \
  --output /reports/native-architectures.json \
  2>&1 | tee "$WALL_REPORT/native-architectures.log"
# A Docker 29/containerd ID may be a manifest/index digest; classic Docker uses
# the config digest. Save a named image so both stores restore the same tag.
WALL_IMAGE_TAG="docker.io/library/wall-oss-crp-transfer:validated-${WALL_IMAGE_ID:7:12}"
docker image tag "$WALL_IMAGE_ID" "$WALL_IMAGE_TAG"
[[ $(docker image inspect --format '{{.Id}}' "$WALL_IMAGE_TAG") == "$WALL_IMAGE_ID" ]]
mkdir -p "$WALL_BUNDLE/reports"
docker image save --platform linux/amd64 "$WALL_IMAGE_TAG" | gzip -1 > "$WALL_BUNDLE/environment-image.tar.gz.partial"
python3 "$TASK_ROOT/docker/image_archive_identity.py" create \
  --archive "$WALL_BUNDLE/environment-image.tar.gz.partial" \
  --tag "$WALL_IMAGE_TAG" --source-image-id "$WALL_IMAGE_ID" \
  --output "$WALL_BUNDLE/image-identity.json"
mv "$WALL_BUNDLE/environment-image.tar.gz.partial" "$WALL_BUNDLE/environment-image.tar.gz"
if [[ -n ${WALL_ASSETS:-} ]]; then
  (cd "$WALL_ASSETS" && sha256sum -c SHA256SUMS)
  # Compare every hash/path entry and its multiplicity, independent of locale order.
  (
    WALL_ASSET_COMPARISON="$(mktemp -d)"
    trap 'rm -rf -- "$WALL_ASSET_COMPARISON"' EXIT
    LC_ALL=C sort "$WALL_ASSETS/source-assets.sha256" > "$WALL_ASSET_COMPARISON/source"
    LC_ALL=C sort "$WALL_REPORT/validated-assets.sha256" > "$WALL_ASSET_COMPARISON/validated"
    cmp "$WALL_ASSET_COMPARISON/source" "$WALL_ASSET_COMPARISON/validated"
  )
  for archive in models.tar.gz smoke-data.tar.gz; do
    cp "$WALL_ASSETS/$archive" "$WALL_BUNDLE/$archive.partial"
    mv "$WALL_BUNDLE/$archive.partial" "$WALL_BUNDLE/$archive"
  done
else
  tar --exclude='*/.cache' -C "$TASK_ROOT" -cf - models | gzip -1 > "$WALL_BUNDLE/models.tar.gz.partial"
  mv "$WALL_BUNDLE/models.tar.gz.partial" "$WALL_BUNDLE/models.tar.gz"
  tar --exclude='*/.cache' -C "$WALL_DATASET" -cf - . | gzip -1 > "$WALL_BUNDLE/smoke-data.tar.gz.partial"
  mv "$WALL_BUNDLE/smoke-data.tar.gz.partial" "$WALL_BUNDLE/smoke-data.tar.gz"
fi
cp -R --no-preserve=ownership,mode "$WALL_REPORT/." "$WALL_BUNDLE/reports/"
cp "$TASK_ROOT/model_files.manifest.json" "$TASK_ROOT/sources.lock.json" "$WALL_BUNDLE/"
cp "$TASK_ROOT/docker/OFFLINE-DEPLOY.md" \
  "$TASK_ROOT/docker/verify_h100_bundle.sh" \
  "$TASK_ROOT/docker/verify_bundle.py" \
  "$TASK_ROOT/docker/verify_native_architectures.py" \
  "$TASK_ROOT/docker/image_archive_identity.py" "$WALL_BUNDLE/"
# The source ID is audit evidence only; run using the verified target host ID.
printf '%s\n' "$WALL_IMAGE_ID" > "$WALL_BUNDLE/image-id.txt"
printf '%s\n' "$WALL_IMAGE_TAG" > "$WALL_BUNDLE/image-tag.txt"
gzip -t "$WALL_BUNDLE/environment-image.tar.gz" "$WALL_BUNDLE/models.tar.gz" "$WALL_BUNDLE/smoke-data.tar.gz"
# The completion marker is published last, after the manifest is atomically
# installed. Neither the manifest nor marker recursively checksum themselves.
python3 "$TASK_ROOT/docker/verify_bundle.py" finalize --bundle "$WALL_BUNDLE"
echo "Bundle ready for SSH transfer to the Ubuntu server: $WALL_BUNDLE"
echo 'H100 GPU/full-model verification must still be performed after docker load.'
