#!/usr/bin/env bash
# After tools/data/fetch_ai4arctic.sh has verified the download: convert all scenes
# to FLOE format, write datalists/ai4arctic/{train,val,test}.json, build the tile
# index. Waits for the verification marker; refuses to run on a failed check.
#
#   tmux new -d -s floe-a4a-prep "bash tools/data/prepare_ai4arctic.sh"
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null
A4A=${A4A_DIR:-$FLOE_ROOT/ai4arctic}
LOG=$FLOE_LOG_DIR/prepare_ai4arctic.log
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for verified download in $A4A/rtt"
until [ -f "$A4A/rtt/.verified" ] || [ -f "$A4A/rtt/.verify_failed" ]; do sleep 300; done
if [ -f "$A4A/rtt/.verify_failed" ]; then say "download verification FAILED; not converting"; exit 1; fi

say "converting to FLOE format -> $FLOE_A4A_DIR"
python -m tools.data.convert_ai4arctic --src "$A4A/rtt" --dst "$FLOE_A4A_DIR" \
    --lists-out datalists/ai4arctic --workers ${A4A_WORKERS:-12} >> "$LOG" 2>&1 \
  || { say "conversion FAILED"; exit 1; }
tail -2 "$LOG"

say "building tile index -> $FLOE_A4A_TILE_INDEX"
python -m tools.data.scan_tiles --data-dir "$FLOE_A4A_DIR" --out "$FLOE_A4A_TILE_INDEX" \
    --patch-size 256 --workers 16 >> "$LOG" 2>&1 || { say "tile index FAILED"; exit 1; }
say "DONE (queue entries 'a4a' are now ready)"
