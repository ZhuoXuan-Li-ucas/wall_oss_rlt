#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STAGE_ROOT}/scripts/activate.sh"
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  "${STAGE_ROOT}"/wheelhouse/*.whl
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -r "${STAGE_ROOT}/wall-x/requirements.txt" \
  -r "${STAGE_ROOT}/configs/requirements-stage1-extra.txt" \
  'setuptools==80.9.0' wheel
# DMuon is installed for consistency with the README; training is outside scope.
# Its optional training-only fast-clip extension is explicitly disabled.
DMUON_BUILD_FAST_CLIP=0 python -m pip install --no-build-isolation --no-deps \
  -e "${STAGE_ROOT}/dmuon"
python -m pip install --no-deps -e "${STAGE_ROOT}/lerobot"
if [[ -f "${STAGE_ROOT}/artifacts/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl" ]]; then
  python -m pip install --no-deps "${STAGE_ROOT}/artifacts/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl"
elif [[ -f "${STAGE_ROOT}/.cache/flash-attn-2.8.3/setup.py" ]]; then
  python -m pip install --no-build-isolation --no-deps "${STAGE_ROOT}/.cache/flash-attn-2.8.3"
else
  python -m pip install --no-cache-dir --no-build-isolation 'flash-attn==2.8.3'
fi
python -m pip install --no-build-isolation --no-deps -e "${STAGE_ROOT}/wall-x"
python -m pip freeze --all > "${STAGE_ROOT}/environment.freeze.txt"
conda list --explicit > "${STAGE_ROOT}/environment.conda-explicit.txt"
