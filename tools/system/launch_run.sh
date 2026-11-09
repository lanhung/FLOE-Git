#!/usr/bin/env bash
# Launch one non-fold run: train with the config's own lists, then evaluate the
# selected checkpoint on its test list with a per-scene CM dump.
#
#   bash tools/system/launch_run.sh TAG CONFIG SEED GPU
#   e.g. bash tools/system/launch_run.sh a4a_baseline configs/AI4Arctic/a4a_baseline.py 42 2
#
# Writes $FLOE_ROOT/work_dir/floe_<TAG>_s<SEED>/, $FLOE_LOG_DIR/floe_<TAG>_s<SEED>.log
# and $FLOE_CM_DIR/<TAG>_s<SEED>_{val,test}.npz. Refuses if the GPU is busy or the
# run directory exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null
TAG=$1; CFG=$2; SEED=$3; GPU=$4
RUN=floe_${TAG}_s${SEED}
WD=$FLOE_ROOT/work_dir/$RUN
LOG=$FLOE_LOG_DIR/$RUN.log
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
if [ -e "$WD" ]; then echo "[launch] $RUN: $WD exists, skipping"; exit 0; fi
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -gt 1000 ]; then echo "[launch] $RUN: GPU $GPU busy (${USED} MiB), skipping"; exit 0; fi
read -r VAL_LIST TEST_LIST < <(python - "$CFG" <<'EOF'
import sys
from mmcv import Config
o = Config.fromfile(sys.argv[1]).train_options
print(o['val_path'], o['test_path'])
EOF
)
mkdir -p "$FLOE_ROOT/work_dir" "$FLOE_CM_DIR"
tmux new-session -d -s "floe-${TAG}-s${SEED}" "
  source tools/system/floe_env.sh >/dev/null
  export CUDA_VISIBLE_DEVICES=$GPU
  {
    echo \"[launch] \$(date -Is) $RUN on GPU $GPU, config $CFG\"
    python -m tools.train $CFG --seed $SEED --wandb-project MyDS-FLOE --wandb-name $RUN --work-dir $WD
    echo \"EXIT_TRAIN=\$?\"
    for split in val test; do
      L=\$([ \$split = val ] && echo $VAL_LIST || echo $TEST_LIST)
      python -m tools.evaluate --config $CFG --checkpoint $WD/best_model_$RUN.pth \
          --val-list \$L --dump-cm $FLOE_CM_DIR/${TAG}_s${SEED}_\$split.npz --output-dir $WD/eval_\$split
      echo \"EXIT_EVAL_\$split=\$?\"
    done
    echo \"[launch] \$(date -Is) done\"
  } > $LOG 2>&1"
echo "[launch] $RUN started on GPU $GPU (tmux floe-${TAG}-s${SEED}, log $LOG)"
