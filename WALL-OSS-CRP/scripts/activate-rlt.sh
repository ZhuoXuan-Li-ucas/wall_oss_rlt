# Source this file from bash.
_WALL_RLT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_WALL_RLT_ROOT}/scripts/activate.sh"
source "${_WALL_RLT_ROOT}/.venv-rlt/bin/activate"
unset _WALL_RLT_ROOT
