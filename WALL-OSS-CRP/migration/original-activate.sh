# Source this file in bash: source /home/pangk/workspaces/WALL-OSS-CRP/scripts/activate.sh
_WALL_CRP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/pangk/miniforge3/etc/profile.d/conda.sh
conda activate "${_WALL_CRP_ROOT}/env"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="${_WALL_CRP_ROOT}/.cache/huggingface"
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=8.6
# FlashAttention 2.8.3 accepts 80/90/100/120; sm80 binaries run on sm86.
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=1
export NVCC_THREADS=1
unset _WALL_CRP_ROOT
