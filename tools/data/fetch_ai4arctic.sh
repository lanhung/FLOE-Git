#!/usr/bin/env bash
# Fetch the AI4Arctic ready-to-train benchmark (train v3 + test v2) and verify it.
#
#   tmux new -d -s floe-a4a "bash tools/data/fetch_ai4arctic.sh"
#
# Source: TorchGeo re-host on Hugging Face (Figshare/DTU are behind a bot challenge
# from the server), via HF_ENDPOINT (hf-mirror). Steps, each resumable:
#   1. download the three tarball parts with resume + retries;
#   2. check each part's SHA-256 against the Hugging Face LFS pointer;
#   3. extract to $A4A/rtt/{train,test};
#   4. check every extracted .nc against the official DTU MD5 manifests in
#      datalists/ai4arctic/ (train v3, test v2) and report any mismatch;
#   5. delete the tarballs only if step 4 passed.
set -uo pipefail
cd "$(dirname "$0")/../.."
source tools/system/floe_env.sh >/dev/null
export HF_HUB_OFFLINE=0
A4A=${A4A_DIR:-$FLOE_ROOT/ai4arctic}
REPO=torchgeo/ai4artic-sea-ice-challenge
BASE=${HF_ENDPOINT:-https://huggingface.co}
LOG=$FLOE_LOG_DIR/fetch_ai4arctic.log
mkdir -p "$A4A/dl" "$A4A/rtt"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "fetching $REPO via $BASE into $A4A"
curl -s -A "Mozilla/5.0" "$BASE/api/datasets/$REPO/tree/main?recursive=true" -o "$A4A/dl/tree.json"
PARTS="test.tar.gz train.tar.gzaa train.tar.gzab"
for p in $PARTS; do
  want=$(python -c "import json; t=json.load(open('$A4A/dl/tree.json')); print([x['lfs']['oid'] for x in t if x['path']=='$p'][0])")
  if [ -f "$A4A/dl/$p.ok" ]; then say "$p already verified"; continue; fi
  for attempt in 1 2 3 4 5; do
    say "download $p (attempt $attempt)"
    curl -L -C - --retry 20 --retry-delay 10 -A "Mozilla/5.0" -o "$A4A/dl/$p" \
         "$BASE/datasets/$REPO/resolve/main/$p" >> "$LOG" 2>&1
    got=$(sha256sum "$A4A/dl/$p" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then touch "$A4A/dl/$p.ok"; say "$p sha256 OK"; break; fi
    say "$p sha256 mismatch ($got != $want)"
    [ $attempt -ge 3 ] && rm -f "$A4A/dl/$p"
  done
  [ -f "$A4A/dl/$p.ok" ] || { say "FAILED to fetch $p"; exit 1; }
done

if [ ! -f "$A4A/rtt/.extracted" ]; then
  say "extracting"
  tar -xzf "$A4A/dl/test.tar.gz" -C "$A4A/rtt" && \
  cat "$A4A/dl/train.tar.gzaa" "$A4A/dl/train.tar.gzab" | tar -xz -C "$A4A/rtt" && \
  touch "$A4A/rtt/.extracted" || { say "extraction FAILED"; exit 1; }
fi

say "checking extracted files against official DTU MD5 manifests"
python - "$A4A/rtt" <<'EOF' 2>&1 | tee -a "$LOG"
import csv, hashlib, os, sys
root = sys.argv[1]
found = {}
for d, _, fs in os.walk(root):
    for f in fs:
        if f.endswith('.nc'):
            found.setdefault(f, []).append(os.path.join(d, f))
bad = 0
for man in ('datalists/ai4arctic/manifest_rtt_train_v3.csv', 'datalists/ai4arctic/manifest_rtt_test_v2.csv'):
    rows = [r for r in csv.DictReader(open(man)) if r['name'].endswith('.nc')]
    ok = mism = miss = 0
    for r in rows:
        paths = found.get(r['name'])
        if not paths:
            miss += 1; print('  missing', r['name']); continue
        h = hashlib.md5()
        with open(paths[0], 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 24), b''):
                h.update(b)
        if h.hexdigest() == r['md5']:
            ok += 1
        else:
            mism += 1; print('  md5 differs from official:', r['name'])
    bad += mism + miss
    print(f'{man}: {ok} identical to official, {mism} differ, {miss} missing (of {len(rows)})')
open(os.path.join(root, '.verified' if bad == 0 else '.verify_failed'), 'w').write(str(bad))
sys.exit(1 if bad else 0)
EOF
if [ -f "$A4A/rtt/.verified" ]; then
  say "all files identical to the official release; removing tarballs"
  rm -f "$A4A/dl/"*.tar.gz*
  say "DONE"
else
  say "verification found differences; tarballs kept, see log"
fi
