# Source before any FLOE command on the server:  source tools/system/floe_env.sh
# Every value can be overridden by exporting it first.
FLOE_ROOT=${FLOE_ROOT:-/root/autodl-tmp/WannianLi/FLOE}
# Shared server: keep root's ~/.local site-packages (other projects) off sys.path.
export PYTHONNOUSERSITE=1
# Training reads the compact copy (tools.data.recompress: continuous channels on a
# 2^-10 z-unit grid, labels/mask bit-exact, zstd). FLOE_DATA_ORIG is the original.
export FLOE_DATA_ORIG=${FLOE_DATA_ORIG:-$FLOE_ROOT/Data/FLOE_datanc}
export FLOE_DATA_DIR=${FLOE_DATA_DIR:-$FLOE_ROOT/data_q10}
export FLOE_TILE_INDEX=${FLOE_TILE_INDEX:-$FLOE_ROOT/index/floe_tile_index_417.npz}   # rebuilt from data_q10, same tiles
export FLOE_CM_DIR=${FLOE_CM_DIR:-$FLOE_ROOT/cm}
# AI4Arctic / AutoICE benchmark in FLOE format (tools.data.convert_ai4arctic)
export FLOE_A4A_DIR=${FLOE_A4A_DIR:-$FLOE_ROOT/ai4arctic/floe_format}
export FLOE_A4A_TILE_INDEX=${FLOE_A4A_TILE_INDEX:-$FLOE_ROOT/ai4arctic/floe_tile_index_a4a.npz}
export FLOE_POOL_DIR=${FLOE_POOL_DIR:-$FLOE_ROOT/pooled}            # symlinks to both archives
export FLOE_POOL_TILE_INDEX=${FLOE_POOL_TILE_INDEX:-$FLOE_ROOT/pooled_tile_index.npz}
export FLOE_CHART_DIR=${FLOE_CHART_DIR:-$FLOE_ROOT/charts}           # <scene>.chart.h5 (egg-code compositions)
# Rebuilt archive (tools/build): scenes, composition sidecars, grouped split, tile index
export FLOE_V2_DIR=${FLOE_V2_DIR:-$FLOE_ROOT/data_v2}
export FLOE_V2_CHART_DIR=${FLOE_V2_CHART_DIR:-$FLOE_ROOT/charts_v2}
export FLOE_V2_SPLIT=${FLOE_V2_SPLIT:-datalists/v2_2023}
export FLOE_V2_TILE_INDEX=${FLOE_V2_TILE_INDEX:-$FLOE_ROOT/index/v2_2023_tile_index.npz}
export FLOE_FIG_DIR=${FLOE_FIG_DIR:-$FLOE_ROOT/figs}
export FLOE_LOG_DIR=${FLOE_LOG_DIR:-$FLOE_ROOT/logs}
# huggingface.co is not reachable from this server; the mirror is.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# Pretrained encoders (smp-hub/mit_b2.imagenet, smp-hub/resnet50.imagenet) are cached
# in ~/.cache/huggingface; stay offline so concurrent runs never hit the mirror.
# Set HF_HUB_OFFLINE=0 once to fetch a new encoder.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# No W&B login on this server: log locally (sync later with `wandb sync` if wanted).
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$FLOE_ROOT}
FLOE_ENV=${FLOE_ENV:-$FLOE_ROOT/envs/ice-binary}
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "$FLOE_ENV"
fi
# Blosc/zstd HDF5 filters come from `import hdf5plugin` in the readers. Do not also
# export HDF5_PLUGIN_PATH: registering the same filter twice makes dataset creation
# fail ("error during user callback").
unset HDF5_PLUGIN_PATH
mkdir -p "$FLOE_CM_DIR" "$FLOE_FIG_DIR" "$FLOE_LOG_DIR"
echo "[floe_env] data=$FLOE_DATA_DIR env=$FLOE_ENV wandb=$WANDB_MODE"
