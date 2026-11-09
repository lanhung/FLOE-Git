#!/usr/bin/env bash
# Build the FLOE training environment as a self-contained conda prefix.
#
#   bash tools/system/setup_env.sh [PREFIX]
#
# PREFIX defaults to /root/autodl-tmp/WannianLi/FLOE/envs/ice-binary. Versions are
# pinned to what the code was written against: segmentation-models-pytorch 0.5.0
# (MiT encoder forward convention, see floe/models/sota/segformer.py), mmcv 1.x
# (mmcv.Config, pure-Python build without CUDA ops), mmengine. The CUDA 12.4 torch
# build runs on drivers >= 12.4 (Ada and earlier; for Blackwell/RTX 50xx use a
# cu128 build of torch >= 2.7 instead).
set -euo pipefail
# Shared server: root's ~/.local site-packages (other projects) would otherwise come
# before this env on sys.path.
export PYTHONNOUSERSITE=1

PREFIX=${1:-/root/autodl-tmp/WannianLi/FLOE/envs/ice-binary}
CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}

source "$CONDA_SH"
if [ ! -x "$PREFIX/bin/python" ]; then
    conda create -y -p "$PREFIX" python=3.10
fi
PY="$PREFIX/bin/python"
PIP="$PY -m pip"

$PIP install --upgrade pip wheel "setuptools<70"
$PIP install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX"
$PIP install \
    "numpy==1.26.4" "scipy<1.15" "h5py>=3.10" "xarray<2025" h5netcdf netCDF4 \
    "matplotlib<3.10" seaborn tqdm "wandb==0.18.7" icecream \
    "torchmetrics==1.4.3" "timm==1.0.11" "segmentation-models-pytorch==0.5.0" \
    "mmengine==0.10.5" "yapf==0.40.1" addict opencv-python-headless \
    fvcore thop "huggingface-hub<1.0" "hdf5plugin==5.0.0"
# mmcv 1.x without compiled ops (only mmcv.Config / mkdir_or_exist are used).
MMCV_WITH_OPS=0 $PIP install --no-build-isolation "mmcv==1.7.2"

$PY - <<'EOF'
import torch, torchvision, segmentation_models_pytorch as smp, timm, torchmetrics, mmcv, mmengine, numpy, h5py, xarray
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(),
      'devices', torch.cuda.device_count())
print('torchvision', torchvision.__version__, '| smp', smp.__version__, '| timm', timm.__version__,
      '| torchmetrics', torchmetrics.__version__, '| mmcv', mmcv.__version__, '| mmengine', mmengine.__version__)
print('numpy', numpy.__version__, '| h5py', h5py.__version__, '| xarray', xarray.__version__)
from mmcv import Config  # noqa: F401  (mmcv 1.x API used by every entry point)
import sys
assert not any('.local' in p for p in sys.path), f'user site leaked into sys.path: {sys.path}'
EOF
$PIP freeze > "$PREFIX/pip-freeze.txt"
echo "ENV_READY $PREFIX"
