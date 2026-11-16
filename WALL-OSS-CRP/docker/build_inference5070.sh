#!/usr/bin/env bash
# Run from the new inference source bundle after validating/loading the D0 bundle.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash build_inference5070.sh --base-image LOCAL_VALIDATED_D0_TAG_OR_ID --tag NEW_TAG [--context-dir NEW_DIRECTORY]
Requires a locally loaded, already validated D0 image. The script verifies its
recorded config/manifest identity and ordered filesystem layers before building.
No pull, network access, GPU execution, host driver changes, or D0 overwrite.
The context and Docker build log are kept outside the immutable source bundle.
EOF
}

base_image=""
image_tag=""
context_dir=""
while (($#)); do
  case "$1" in
    --base-image|--tag|--context-dir)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --base-image) base_image="$2" ;;
        --tag) image_tag="$2" ;;
        --context-dir) context_dir="$2" ;;
      esac
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$base_image" && -n "$image_tag" ]] || { usage >&2; exit 2; }
[[ "$base_image" != -* && "$image_tag" != -* && "$image_tag" == *:* && "$image_tag" != *@* ]] || {
  echo 'Supply an explicit local base reference and a new explicit output tag.' >&2; exit 2;
}
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python3 "$bundle_dir/package_wall_rlt_inference.py" verify --bundle "$bundle_dir"
# Verify daemon access separately: an inaccessible daemon must not look like an
# unused output tag. Passing --base-image is the operator's explicit selection.
docker info --format '{{.ServerVersion}}' >/dev/null
if docker image inspect "$image_tag" >/dev/null 2>&1; then
  echo 'Output tag already exists; choose a new tag. Existing images are preserved.' >&2
  exit 2
fi
base_id="$(docker image inspect "$base_image" | python3 "$bundle_dir/image_archive_identity.py" \
  check-loaded --identity "$bundle_dir/base-image-identity.json")"
[[ "$base_id" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Invalid verified base image ID.' >&2; exit 2; }
if [[ -z "$context_dir" ]]; then
  context_dir="${bundle_dir}-build-$(date +%Y%m%d-%H%M%S)"
fi
python3 "$bundle_dir/package_wall_rlt_inference.py" extract --bundle "$bundle_dir" --output "$context_dir"
context_dir="$(cd -- "$context_dir" && pwd -P)"
echo "Building $image_tag from verified local base $base_id; compilation uses one worker."
docker build --pull=false --network=none --build-arg "BASE_IMAGE=$base_id" \
  --tag "$image_tag" --file "$context_dir/Dockerfile.inference5070" "$context_dir" \
  2>&1 | tee "${context_dir}.build.log"
echo "Image built: $image_tag. Static sm_120 coverage passed; GPU kernel/inference acceptance remains pending."
echo "Build log: ${context_dir}.build.log"
