#!/usr/bin/env bash
# RegTR-Colour-Fusion pod setup: conda env + PyTorch + PyTorch3D + pip deps.
# Run from repo root:  bash scripts/install_pod_deps.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${REGTR_ENV_NAME:-regtr}"
PYTHON_VER="${REGTR_PYTHON:-3.10}"
# Pin torch 2.4 + cu124 (matches prebuilt PyTorch3D wheel below).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
# Meta's fbaipublicfiles wheel index often returns AccessDenied on cloud pods; use a direct wheel.
PYTORCH3D_WHEEL_URL="${PYTORCH3D_WHEEL_URL:-https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9+pt2.4.0cu124-cp310-cp310-linux_x86_64.whl}"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi
conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel

echo "==> PyTorch 2.4 + cu124 (matches PyTorch3D prebuilt wheels)"
pip install torch==2.4.0 torchvision==0.19.0 --index-url "$TORCH_INDEX"

echo "==> pip requirements (vtk>=9.2.4, etc.)"
pip install -r src/requirements.txt

echo "==> PyTorch3D (direct wheel; Meta CDN index often blocked on pods)"
pip install "$PYTORCH3D_WHEEL_URL"

python -c "
import torch, pytorch3d
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('pytorch3d', pytorch3d.__version__)
"

echo "Done. Activate with: conda activate ${ENV_NAME}"
