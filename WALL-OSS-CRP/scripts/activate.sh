# Source this file from bash.
_WALL_CRP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH PYTHONHOME
source "${_WALL_CRP_ROOT}/env/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME="${_WALL_CRP_ROOT}/.cache/huggingface"
if [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; fi
export MAX_JOBS="${MAX_JOBS:-1}" NVCC_THREADS="${NVCC_THREADS:-1}"
unset _WALL_CRP_ROOT
