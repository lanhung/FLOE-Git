#!/usr/bin/env bash
# One priority queue for all GPU runs.
#
#   tmux new -d -s floe-queue "bash tools/system/run_queue.sh QUEUE_FILE"
#   bash tools/system/run_queue.sh QUEUE_FILE --check     # readiness report only
#
# QUEUE_FILE: one entry per line, "READY | COMMAND", processed strictly in order.
#   READY   a readiness check: fold<K> (all fold-K scenes converted and readable),
#           a4a (AutoICE scenes converted and tile index built), done:<run> (the launch
#           of that run has ended, see launch_fold.sh), or always.
#   COMMAND run with {GPU} replaced by the first idle GPU (memory.used < 1000 MiB).
# Lines starting with # are ignored. After each launch the script waits until the
# run occupies its GPU (max 30 min), then STAGGER_MIN (default 20) minutes, so no
# GPU is double-booked and validation memory peaks do not coincide.
# Progress: $FLOE_LOG_DIR/run_queue.log. Cancel: tmux kill-session -t floe-queue
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null
QUEUE=$1
STAGGER=${STAGGER_MIN:-20}
LOG=$FLOE_LOG_DIR/run_queue.log
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

ready_fold() {   # all scenes of fold $1 present in FLOE_DATA_DIR and openable (cached)
  local K=$1 marker=$FLOE_LOG_DIR/.ready_fold$1
  [ -f "$marker" ] && return 0
  python - "datalists/cv5/fold${K}_train.json" "datalists/cv5/fold${K}_val.json" <<'EOF' || return 1
import json, os, sys
from tools.data.check_transfer import check_file
d = os.environ['FLOE_DATA_DIR']
names = sorted(set(json.load(open(sys.argv[1]))) | set(json.load(open(sys.argv[2]))))
on_disk = set(os.listdir(d))
if any(n not in on_disk for n in names):
    sys.exit(1)
bad = [n for n in names if not check_file(os.path.join(d, n))['ok']]
if bad:
    print('UNREADABLE', bad[:5]); sys.exit(2)
EOF
  touch "$marker"
}
ready_a4a() { [ -f "$FLOE_A4A_TILE_INDEX" ] && [ -f datalists/ai4arctic/test.json ]; }
ready_done() { grep -aq '^\[launch\] .* done$' "$FLOE_LOG_DIR/$1.log" 2>/dev/null; }
is_ready() {
  case $1 in
    always) return 0 ;;
    a4a) ready_a4a ;;
    done:*) ready_done "${1#done:}" ;;
    fold*) ready_fold "${1#fold}" ;;
    *) say "unknown READY '$1'"; return 1 ;;
  esac
}
gpu_mem() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits; }
# A GPU counts as idle only if it is still below 1000 MiB 90 s later: between the training
# and the evaluation step of a run the card is empty for a few seconds, and a launch in
# that gap was refused by launch_fold.sh (GPU busy) and the entry lost (12 Nov 2026).
idle_gpu() {
  local g
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
      awk -F', ' '$2 < 1000 {print $1; exit}')
  [ -n "$g" ] || return 0
  sleep 90
  [ "$(gpu_mem "$g")" -lt 1000 ] && echo "$g"
  return 0
}

if [ "${2:-}" = "--check" ]; then   # report only, launch nothing
  echo "idle GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {printf "%s ", $1}')"
  grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$' | while IFS='|' read -r READY CMD; do
    READY=$(echo "$READY" | xargs)
    if is_ready "$READY" < /dev/null; then st=READY; else st=waiting; fi
    printf '%-8s %-7s %s\n' "$st" "$READY" "$(echo "$CMD" | sed 's/^ *//')"
  done
  exit 0
fi
say "queue start: $QUEUE"
grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$' | while IFS='|' read -r READY CMD; do
  READY=$(echo "$READY" | xargs); CMD=$(echo "$CMD" | sed 's/^ *//')
  until is_ready "$READY"; do sleep 600; done
  while true; do
    GPU=$(idle_gpu)
    while [ -z "$GPU" ]; do sleep 300; GPU=$(idle_gpu); done
    say "launch on GPU $GPU: ${CMD//\{GPU\}/$GPU}"
    OUT=$(bash -c "${CMD//\{GPU\}/$GPU}" < /dev/null 2>&1)
    echo "$OUT" | tee -a "$LOG"
    # launch_fold.sh refuses a busy card; keep the entry and try again instead of dropping it
    if echo "$OUT" | grep -q "GPU $GPU busy"; then
      say "GPU $GPU was busy at launch; retrying this entry in 5 min"
      sleep 300; continue
    fi
    break
  done
  for i in $(seq 1 60); do
    [ "$(gpu_mem "$GPU")" -gt 1000 ] && break
    sleep 30
  done
  say "GPU $GPU now $(gpu_mem "$GPU") MiB; staggering ${STAGGER} min"
  sleep $((STAGGER * 60))
done
say "queue finished"
