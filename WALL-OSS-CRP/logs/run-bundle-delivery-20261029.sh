#!/usr/bin/env bash
set -euo pipefail
cd /home/pangk/workspaces/WALL-OSS-CRP
exec python3 /home/pangk/workspaces/WALL-OSS-CRP/scripts/docker_finish_pending_bundle.py --reuse-build-report /home/pangk/workspaces/WALL-OSS-CRP/logs/docker-validation-final-20261029T011859Z --reuse-image-id sha256:8f067a0eb220228f444811532cc8ec0a328e4d46b7983637c1c4ea7743ef9ee8 --report /home/pangk/workspaces/WALL-OSS-CRP/logs/docker-validation-delivery-20261029 --dataset '/media/pangk/My Passport/H100_lerobot/Datasets/stack_the_base_llx_04251' --assets '/media/pangk/My Passport/WALL-OSS-CRP-assets-20261028' --bundle '/media/pangk/My Passport/WALL-OSS-CRP-transfer-20261028' --job-file /home/pangk/workspaces/WALL-OSS-CRP/logs/docker-bundle-job-20261028.json
