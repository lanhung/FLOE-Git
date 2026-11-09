#!/usr/bin/env bash
# Convert scenes into the compact copy as they arrive; stop when all are done.
#
#   tmux new -d -s floe-compress "bash tools/system/compress_loop.sh"
#
# Every 10 minutes runs tools.data.recompress over the complete source files (rsync
# temporaries and already converted scenes are skipped). Exits once every scene of
# the 417-scene lists exists in the destination. Sources are never modified.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null
SRC=${FLOE_DATA_ORIG:?}
DST=${FLOE_DATA_DIR:?}
WORKERS=${COMPRESS_WORKERS:-12}
LOG=$FLOE_LOG_DIR/compress_loop.log
LISTS="datalists/train_417.json datalists/val_417.json"
echo "[$(date -Is)] compress loop: $SRC -> $DST (workers $WORKERS)" | tee -a "$LOG"
while true; do
  python -m tools.data.recompress --src "$SRC" --dst "$DST" --q-exp -10 --codec zstd \
      --workers "$WORKERS" --lists $LISTS >> "$LOG" 2>&1
  N=$(python - $LISTS <<EOF
import json, os, sys
names = set()
for p in sys.argv[1:]:
    names |= set(json.load(open(p)))
print(sum(os.path.exists(os.path.join("$DST", n)) for n in names), len(names))
EOF
)
  echo "[$(date -Is)] converted ${N% *}/${N#* }" | tee -a "$LOG"
  [ "${N% *}" = "${N#* }" ] && break
  sleep 600
done
echo "[$(date -Is)] all scenes converted" | tee -a "$LOG"
