"""Download, build and clean up every paired frame (tools.build.pair output).

Each worker downloads one frame from ASF, builds the scene with tools.build.make_scene,
and deletes the zip. Scenes with too little charted ice to be useful for floe size
(--min-ice-frac of valid pixels with a FLOE class 1-5) are removed again and logged.
Restartable: frames with a scene file or a 'skipped' log entry are not redone.

    python -m tools.build.run_archive --pairs $FLOE_ROOT/raw/pairs.json --charts $FLOE_ROOT/raw/dmi \
        --out $FLOE_ROOT/data_v2 --chart-out $FLOE_ROOT/charts_v2 --tmp $FLOE_ROOT/raw/s1 --workers 12
"""
import argparse
import json
import os
import os.path as osp
import time
import traceback
from multiprocessing import Pool

import numpy as np

from tools.build import asf
from tools.build import make_scene as ms

_SESSION = None
_INDEX = None


def _init(creds, chart_root):
    global _SESSION, _INDEX
    _SESSION = asf.EarthdataSession(*creds)
    _INDEX = json.load(open(osp.join(chart_root, 'index.json')))


def _work(args):
    name, info, a = args
    t0 = time.time()
    zp = osp.join(a['tmp'], name + '.zip')
    try:
        _, st = asf.download(name, info, a['tmp'], _SESSION)
        if st == 'FAILED':
            return {'product': name, 'status': 'download_failed'}
        r = ms.build_one(zp, a['charts'], _INDEX, a['out'], a['chart_out'], a['max_minutes'])
        if r['status'] == 'ok':
            h = np.asarray(r['floe_hist'])
            n_valid = r['shape'][0] * r['shape'][1]
            ice = h[1:6].sum() / max(n_valid, 1)
            r['ice_frac'] = float(ice)
            if ice < a['min_ice_frac']:
                for p in (osp.join(a['out'], name + '.nc'), osp.join(a['chart_out'], name + '.chart.h5')):
                    if osp.exists(p):
                        os.remove(p)
                r['status'] = 'skipped_little_ice'
        return {**r, 'total_seconds': round(time.time() - t0, 1)}
    except Exception as e:
        return {'product': name, 'status': 'error', 'error': repr(e), 'trace': traceback.format_exc()[-1500:]}
    finally:
        if osp.exists(zp) and not a['keep_zips']:
            os.remove(zp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pairs', required=True, nargs='+', help='one or more pair.py outputs')
    ap.add_argument('--charts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chart-out', required=True)
    ap.add_argument('--tmp', required=True, help='download directory (zips are deleted after the build)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-minutes', type=float, default=30.0)
    ap.add_argument('--min-ice-frac', type=float, default=0.01)
    ap.add_argument('--keep-zips', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    for d in (args.out, args.chart_out, args.tmp):
        os.makedirs(d, exist_ok=True)

    frames = {}
    for p in args.pairs:
        frames.update(json.load(open(p))['frames'])
    log_path = osp.join(args.out, 'archive_log.jsonl')
    done = set()
    if osp.exists(log_path):
        for line in open(log_path):
            r = json.loads(line)
            if r['status'] in ('ok', 'skipped_little_ice', 'no_chart'):
                done.add(r['product'])
    todo = [n for n in sorted(frames) if n not in done and not osp.exists(osp.join(args.out, n + '.nc'))]
    if args.limit:
        todo = todo[:args.limit]
    gb = sum(frames[n]['bytes'] for n in todo) / 1e9
    print(f'{len(frames)} paired frames, {len(todo)} to do ({gb:.0f} GB to download)', flush=True)

    a = {'tmp': args.tmp, 'charts': args.charts, 'out': args.out, 'chart_out': args.chart_out,
         'max_minutes': args.max_minutes, 'min_ice_frac': args.min_ice_frac, 'keep_zips': args.keep_zips}
    jobs = [(n, frames[n], a) for n in todo]
    counts = {}
    with open(log_path, 'a') as log, Pool(args.workers, initializer=_init,
                                          initargs=(asf.credentials(), args.charts)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs), 1):
            log.write(json.dumps(r) + '\n')
            log.flush()
            counts[r['status']] = counts.get(r['status'], 0) + 1
            if i % 10 == 0 or i == len(jobs):
                print(f'[{i}/{len(jobs)}] {counts}', flush=True)
    print('done', counts)


if __name__ == '__main__':
    main()
