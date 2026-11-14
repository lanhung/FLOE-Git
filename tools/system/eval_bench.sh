#!/usr/bin/env bash
# Score finished benchmark checkpoints without the tile gate, with per-scene CM and
# per-polygon dumps (Table 7 of the dataset paper, tools.analysis.baseline_table).
#
#   bash tools/system/eval_bench.sh test    FOLD SEED GPU      baseline f<K> s<seed> on the 2026 test scenes
#   bash tools/system/eval_bench.sh valpoly FOLD SEED GPU      the same model on its validation fold again
#   bash tools/system/eval_bench.sh model NAME CONFIG CHECKPOINT GPU   any model on the 2026 test scenes
#   bash tools/system/eval_bench.sh screen NAME CONFIG CHECKPOINT GPU  a 2023 fold-0 model on its validation
#        scenes (paper 2 screening, paper_method/DECISIONS.md), -> $FLOE_CM_DIR/v2_full/<NAME>.npz
#
# Writes (refuses if the CM or polygon dump exists; no overwrite):
#   test     $FLOE_CM_DIR/v2bench_test/baseline_s<seed>_f<K>.npz, poly/v2bench_test/... alike
#   valpoly  $FLOE_CM_DIR/v2bench_full/baseline_s<seed>_f<K>.npz, poly/v2bench_full/... alike
#   model    $FLOE_CM_DIR/v2bench_test/<NAME>.npz, poly/v2bench_test/<NAME>.npz
#   log      $FLOE_LOG_DIR/<run or NAME>_<test2026|valfull>.log with EXIT_EVAL= and a
#            final "[launch] ... done" line (run_queue.sh done:<that log name minus .log>)
# The baselines are evaluated with configs/V2/baseline_evalfull.py (gate and on-demand
# reading off; see there why). The 2026 modes refuse to start until all 15 benchmark
# baseline runs (folds 0-4 x seeds 42-44) have ended with EXIT_TRAIN=0 and EXIT_EVAL=0:
# the held-out year is scored only after every model is fixed.
# A tmux session does not inherit the caller's environment: everything is exported into it.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null

MODE=$1; shift
SPLIT=datalists/v2_bench
TILE_INDEX=$FLOE_ROOT/index/v2_bench_tile_index.npz
EVAL_CFG=configs/V2/baseline_evalfull.py
case $MODE in
  test|valpoly)
    [ $# -eq 3 ] || { echo "usage: $0 $MODE FOLD SEED GPU"; exit 2; }
    FOLD=$1; SEED=$2; GPU=$3
    RUN=floe_baseline_v2benchf${FOLD}_s${SEED}
    CFG=$EVAL_CFG
    CKPT=$FLOE_ROOT/work_dir/$RUN/best_model_$RUN.pth
    TAG=baseline_s${SEED}_f${FOLD}
    if [ "$MODE" = test ]; then LIST=$SPLIT/test.json; SUB=v2bench_test; SUFFIX=test2026
    else LIST=$SPLIT/fold${FOLD}_val.json; SUB=v2bench_full; SUFFIX=valfull; fi
    NAME=$RUN ;;
  model)
    [ $# -eq 4 ] || { echo "usage: $0 model NAME CONFIG CHECKPOINT GPU"; exit 2; }
    TAG=$1; CFG=$2; CKPT=$3; GPU=$4
    LIST=$SPLIT/test.json; SUB=v2bench_test; SUFFIX=test2026; NAME=floe_$TAG ;;
  screen)
    [ $# -eq 4 ] || { echo "usage: $0 screen NAME CONFIG CHECKPOINT GPU"; exit 2; }
    TAG=$1; CFG=$2; CKPT=$3; GPU=$4
    SPLIT=datalists/v2_2023; TILE_INDEX=$FLOE_ROOT/index/v2_2023_tile_index.npz
    LIST=$SPLIT/fold0_val.json; SUB=v2_full; SUFFIX=valfull2023; NAME=floe_$TAG ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
CM=$FLOE_CM_DIR/$SUB/$TAG.npz
POLY=$FLOE_CM_DIR/poly/$SUB/$TAG.npz
LOG=$FLOE_LOG_DIR/${NAME}_${SUFFIX}.log
OUT=$FLOE_ROOT/work_dir/eval_bench/${NAME}_${SUFFIX}
SESSION=floe-evb-${TAG}-${SUFFIX}

[ -f "$CFG" ] || { echo "[eval_bench] $TAG: no config $CFG"; exit 2; }
[ -f "$LIST" ] || { echo "[eval_bench] $TAG: no list $LIST"; exit 2; }
[ -f "$CKPT" ] || { echo "[eval_bench] $TAG: no checkpoint $CKPT"; exit 2; }
for f in "$CM" "$POLY"; do
  if [ -e "$f" ]; then echo "[eval_bench] $TAG: $f exists, skipping (no overwrite)"; exit 0; fi
done
if [ "$SUFFIX" = test2026 ]; then
  for S in 42 43 44; do for K in 0 1 2 3 4; do
    L=$FLOE_LOG_DIR/floe_baseline_v2benchf${K}_s${S}.log
    if ! { grep -aq '^\[launch\] .* done$' "$L" && grep -aq '^EXIT_TRAIN=0$' "$L" && grep -aq '^EXIT_EVAL=0$' "$L"; } 2>/dev/null; then
      echo "[eval_bench] $TAG: benchmark run f${K}_s${S} has not ended cleanly ($L); the 2026 scenes are not scored yet"
      exit 3
    fi
  done; done
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "[eval_bench] $TAG: tmux $SESSION exists, skipping"; exit 0; fi
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -gt 1000 ]; then echo "[eval_bench] $TAG: GPU $GPU busy (${USED} MiB), skipping"; exit 0; fi
# an earlier attempt that left no CM: keep its log and outputs, renamed
STAMP=$(date +%Y%m%dT%H%M%S)
[ -e "$LOG" ] && mv "$LOG" "$LOG.$STAMP"
[ -e "$OUT" ] && mv "$OUT" "$OUT.$STAMP"
mkdir -p "$FLOE_CM_DIR/$SUB" "$FLOE_CM_DIR/poly/$SUB" "$(dirname "$OUT")"

tmux new-session -d -s "$SESSION" "
  cd $PWD
  source tools/system/floe_env.sh >/dev/null
  export CUDA_VISIBLE_DEVICES=$GPU FLOE_ROOT=$FLOE_ROOT FLOE_CM_DIR=$FLOE_CM_DIR FLOE_LOG_DIR=$FLOE_LOG_DIR
  export FLOE_V2_DIR=$FLOE_V2_DIR FLOE_V2_CHART_DIR=$FLOE_V2_CHART_DIR FLOE_V2_SPLIT=$SPLIT FLOE_V2_TILE_INDEX=$TILE_INDEX
  export FLOE_A4A_DIR=$FLOE_A4A_DIR FLOE_A4A_V2_DIR=$FLOE_A4A_V2_DIR FLOE_A4A_TILE_INDEX=$FLOE_A4A_TILE_INDEX
  {
    echo \"[launch] \$(date -Is) $NAME $MODE on GPU $GPU, config $CFG, list $LIST\"
    if [ -e $CM ] || [ -e $POLY ]; then echo 'output appeared in the meantime, not overwritten'; echo EXIT_EVAL=9
    else
      python -m tools.evaluate --config $CFG --checkpoint $CKPT --val-list $LIST \
          --dump-cm $CM --composition-dir $FLOE_V2_CHART_DIR --dump-polygons $POLY --output-dir $OUT
      echo \"EXIT_EVAL=\$?\"
    fi
    echo \"[launch] \$(date -Is) done\"
  } > $LOG 2>&1"
echo "[eval_bench] $TAG: $MODE started on GPU $GPU (tmux $SESSION, log $LOG)"
