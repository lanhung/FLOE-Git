#!/usr/bin/env bash
# Launch paper runs for one fold: train, then per-scene evaluation with CM dump.
#
#   bash tools/system/launch_fold.sh FOLD SEED TAG:GPU [TAG:GPU ...]
#   e.g. bash tools/system/launch_fold.sh 0 42 baseline:0 pgsg:1 pgsg_noev:2 gate_noev:3
#
# Tags -> configs:
#   baseline   configs/FLOE/sota_segformer_in1k_dlr_417.py   (a)
#   evgate     configs/FLOE/pgsg_evidence_gate.py             (b)
#   gate_noev  configs/FLOE/pgsg_gate_noevidence.py           (b')
#   pgsg       configs/FLOE/pgsg.py                           (c)
#   pgsg_noev  configs/FLOE/pgsg_noevidence.py                (c')
#   dl3        configs/FLOE/sota_deeplabv3_in1k_dlr.py        comparison model
#   res80      configs/FLOE/res80_noglcm.py                   resolution exp., 80 m, no GLCM
#   res160     configs/FLOE/res160_noglcm.py                  resolution exp., 160 m, no GLCM
#   pooled     configs/Pooled/pooled_noglcm.py                + all AutoICE train scenes
#   comp       configs/FLOE/comp.py                           (a) + chart-composition supervision
#   comp_noord configs/FLOE/comp_noord.py                     comp without the ordinal term
#   comp_allce configs/FLOE/comp_allce.py                     comp with CE on all polygons
#
# Each run gets its own tmux session floe-<tag>-f<K>-s<seed>, writes
#   $FLOE_ROOT/work_dir/floe_<tag>_cv5f<K>_s<seed>/   (checkpoint, offline wandb)
#   $FLOE_LOG_DIR/floe_<tag>_cv5f<K>_s<seed>.log     (train + eval log, EXIT= lines)
#   $FLOE_CM_DIR/cv5/<tag>_s<seed>_f<K>.npz          (per-scene CM for paired_cv)
#   $FLOE_CM_DIR/a4a/<tag>_s<seed>_f<K>_test.npz     (res80, pooled: AutoICE test scenes)
#   $FLOE_CM_DIR/poly/<tag>_s<seed>_f<K>.npz         (per-polygon histograms, if $FLOE_CHART_DIR exists)
# Refuses to start if a fold scene is missing or partial, the GPU is busy, or the
# run directory already exists (no silent overwrite or resume).
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null

FOLD=$1; SEED=$2; shift 2
[ $# -ge 1 ] || { echo "need at least one TAG:GPU"; exit 2; }

declare -A CFG=(
  [baseline]=configs/FLOE/sota_segformer_in1k_dlr_417.py
  [evgate]=configs/FLOE/pgsg_evidence_gate.py
  [gate_noev]=configs/FLOE/pgsg_gate_noevidence.py
  [pgsg]=configs/FLOE/pgsg.py
  [pgsg_noev]=configs/FLOE/pgsg_noevidence.py
  [dl3]=configs/FLOE/sota_deeplabv3_in1k_dlr.py
  [res80]=configs/FLOE/res80_noglcm.py
  [res160]=configs/FLOE/res160_noglcm.py
  [pooled]=configs/Pooled/pooled_noglcm.py
  [comp]=configs/FLOE/comp.py
  [comp_noord]=configs/FLOE/comp_noord.py
  [comp_allce]=configs/FLOE/comp_allce.py
)
# No-GLCM models are also scored on the 20 AutoICE test scenes (the pooled config sees both archives).
declare -A XEVAL=([res80]=1 [pooled]=1)
TRAIN_LIST=datalists/cv5/fold${FOLD}_train.json
VAL_LIST=datalists/cv5/fold${FOLD}_val.json
[ -f "$TRAIN_LIST" ] && [ -f "$VAL_LIST" ] || { echo "missing fold $FOLD lists"; exit 2; }

# All scenes of this fold must be present and complete.
python - "$TRAIN_LIST" "$VAL_LIST" <<'EOF'
import json, os, sys
d = os.environ['FLOE_DATA_DIR']
names = set(json.load(open(sys.argv[1]))) | set(json.load(open(sys.argv[2])))
on_disk = set(os.listdir(d))
missing = sorted(n for n in names if n not in on_disk)
partial = sorted(n for n in names if any(f.startswith('.' + n) for f in on_disk))
if missing or partial:
    sys.exit(f'fold scenes not ready: {len(missing)} missing, {len(partial)} partial '
             f'(e.g. {(missing or partial)[:2]})')
print(f'[launch] all {len(names)} fold scenes present')
EOF

mkdir -p "$FLOE_CM_DIR/cv5" "$FLOE_CM_DIR/a4a" "$FLOE_CM_DIR/poly" "$FLOE_ROOT/work_dir"
if [[ " $* " == *" pooled:"* ]]; then   # (re)build the symlink dir and merged tile index
  python -m tools.data.pool_archives --archive "$FLOE_DATA_DIR" "$FLOE_TILE_INDEX" \
      --archive "$FLOE_A4A_DIR" "$FLOE_A4A_TILE_INDEX" --dst "$FLOE_POOL_DIR" --index-out "$FLOE_POOL_TILE_INDEX"
fi
for pair in "$@"; do
  TAG=${pair%%:*}; GPU=${pair##*:}
  [ -n "${CFG[$TAG]:-}" ] || { echo "unknown tag $TAG"; exit 2; }
  RUN=floe_${TAG}_cv5f${FOLD}_s${SEED}
  WD=$FLOE_ROOT/work_dir/$RUN
  LOG=$FLOE_LOG_DIR/$RUN.log
  CM=$FLOE_CM_DIR/cv5/${TAG}_s${SEED}_f${FOLD}.npz
  XCMD=true
  if [ -n "${XEVAL[$TAG]:-}" ]; then
    XCMD="python -m tools.evaluate --config configs/Pooled/pooled_noglcm.py --checkpoint $WD/best_model_$RUN.pth \
          --val-list datalists/ai4arctic/test.json --dump-cm $FLOE_CM_DIR/a4a/${TAG}_s${SEED}_f${FOLD}_test.npz \
          --output-dir $WD/eval_a4a_test; echo EXIT_EVAL_A4A=\$?"
  fi
  if [ -e "$WD" ]; then echo "[launch] $RUN: $WD exists, skipping"; continue; fi
  USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$USED" -gt 1000 ]; then echo "[launch] $RUN: GPU $GPU busy (${USED} MiB), skipping"; continue; fi
  tmux new-session -d -s "floe-${TAG}-f${FOLD}-s${SEED}" "
    source tools/system/floe_env.sh >/dev/null
    export CUDA_VISIBLE_DEVICES=$GPU
    {
      echo \"[launch] \$(date -Is) $RUN on GPU $GPU, config ${CFG[$TAG]}\"
      python -m tools.train ${CFG[$TAG]} --fold $FOLD --seed $SEED \
          --wandb-project MyDS-FLOE --wandb-name $RUN --work-dir $WD
      echo \"EXIT_TRAIN=\$?\"
      CA=; [ -d \"\$FLOE_CHART_DIR\" ] && CA=\"--composition-dir \$FLOE_CHART_DIR --dump-polygons $FLOE_CM_DIR/poly/${TAG}_s${SEED}_f${FOLD}.npz\"
      python -m tools.evaluate --config ${CFG[$TAG]} --checkpoint $WD/best_model_$RUN.pth \
          --val-list $VAL_LIST --dump-cm $CM --output-dir $WD/eval \$CA
      echo \"EXIT_EVAL=\$?\"
      $XCMD
      echo \"[launch] \$(date -Is) done\"
    } > $LOG 2>&1"
  echo "[launch] $RUN started on GPU $GPU (tmux floe-${TAG}-f${FOLD}-s${SEED}, log $LOG)"
done
