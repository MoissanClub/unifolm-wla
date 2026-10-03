#!/usr/bin/env bash
# Reuse the proven conda environment and its CUDA/cuDSS/TensorRT activation hooks.
# Usage: CONDA_ENV=g1fetch bash deploy/g1/setup.sh client|model|export [--check|--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=${1:-client}
[[ $# == 0 ]] || shift
case "$kind" in
  client|model|export) ;;
  *) echo "Usage: CONDA_ENV=g1fetch bash deploy/g1/setup.sh client|model|export [--check|--dry-run]" >&2; exit 2 ;;
esac
env_name=${CONDA_ENV:-g1fetch}
conda_exe=${CONDA_EXE:-}
if [[ -z "$conda_exe" ]]; then
  conda_exe=$(type -P conda || true)
fi
if [[ -z "$conda_exe" && -x "$HOME/miniconda3/bin/conda" ]]; then
  conda_exe="$HOME/miniconda3/bin/conda"
fi
if [[ -z "$conda_exe" ]]; then
  echo "Conda not found. Set CONDA_EXE to your conda executable." >&2
  exit 1
fi
conda_base=$("$conda_exe" info --base)
source "$conda_base/etc/profile.d/conda.sh"
# Activation matters: invoking env/bin/python alone misses the Jetson library hooks.
conda activate "$env_name"
python -u -m deploy.g1.dependencies "$kind" "$@"
echo "Environment: conda activate $env_name (run modules from this repository root)"
