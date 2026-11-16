#!/usr/bin/env bash
set -euo pipefail
cd /home/pangk/workspaces/WALL-OSS-CRP
test "$(id -u)" = 0
: "${1:?New report directory is required}"
WALL_SWAP_PATH=/home/pangk/workspaces/WALL-OSS-CRP/.cache/wall-build-20261029.swap
test ! -e "$WALL_SWAP_PATH"
test ! -L "$WALL_SWAP_PATH"
cleanup_swap() {
  WALL_EXIT_CODE=$?
  trap - EXIT
  if awk '{print $1}' /proc/swaps | rg -Fxq "$WALL_SWAP_PATH"; then
    if ! swapoff "$WALL_SWAP_PATH"; then
      echo "Temporary build swap remains active: $WALL_SWAP_PATH" >&2
      exit 1
    fi
  fi
  rm -f -- "$WALL_SWAP_PATH"
  echo 'Temporary build swap removed.'
  exit "$WALL_EXIT_CODE"
}
trap cleanup_swap EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
install -m 600 /dev/null "$WALL_SWAP_PATH"
fallocate -l 4G "$WALL_SWAP_PATH"
mkswap "$WALL_SWAP_PATH"
swapon "$WALL_SWAP_PATH"
swapon --show
python3 scripts/docker_finish_pending_bundle.py \
  --report "$1" \
  --dataset '/media/pangk/My Passport/H100_lerobot/Datasets/stack_the_base_llx_04251' \
  --assets '/media/pangk/My Passport/WALL-OSS-CRP-assets-20261028' \
  --bundle '/media/pangk/My Passport/WALL-OSS-CRP-transfer-20261028' \
  --job-file /home/pangk/workspaces/WALL-OSS-CRP/logs/docker-bundle-job-20261028.json
