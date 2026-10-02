#!/usr/bin/env bash
# Use an existing CUDA-enabled Python on Jetson. Never replace Jetson torch/CUDA/TensorRT.
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=${1:-client}
python_bin=${PYTHON_BIN:-python3}
venv_path=${VENV_PATH:-.venv-g1-$kind}
case "$kind" in client|model) ;; *) echo "Usage: PYTHON_BIN=... bash deploy/g1/setup.sh client|model" >&2; exit 2;; esac
"$python_bin" -m venv --system-site-packages "$venv_path"
python_env="$venv_path/bin/python"
if [[ "$kind" == model ]]; then
  "$python_env" - <<'PY'
import sys, torch, torchvision
if sys.version_info < (3, 10):
    raise SystemExit('Python >=3.10 needed; upstream reference environment is 3.12')
if not torch.cuda.is_available():
    raise SystemExit('Select PYTHON_BIN from a working Jetson CUDA PyTorch environment first')
if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 8):
    raise SystemExit('Need PyTorch >=2.8; use the NVIDIA wheel for this JetPack/Python, not a generic replacement')
x = torch.ones((32,32), device='cuda', dtype=torch.bfloat16)
assert (x @ x)[0,0].item() == 32
print('Validated', torch.__version__, torchvision.__version__)
PY
fi
# Keep inherited binary packages at their current versions during dependency resolution.
"$python_env" - <<'PY' > "$venv_path/binary-constraints.txt"
from importlib.metadata import version, PackageNotFoundError
for package in ('torch', 'torchvision', 'numpy', 'scipy'):
    try: print(f'{package}=={version(package)}')
    except PackageNotFoundError: pass
PY
"$python_env" -m pip install -c "$venv_path/binary-constraints.txt" -r "deploy/g1/requirements-$kind.txt"
"$python_env" -m deploy.g1.assets preflight
echo "Ready: $python_env (run modules from this repository root)"
