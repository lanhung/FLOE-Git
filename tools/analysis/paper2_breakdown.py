"""Paper 2, exploratory (not fixed in advance): where the gain of the benchmark arm comes from.

Per-class IoU, and the hard-ice IoU (small and medium floes) by chart region and by season, for
the baseline and the arm, with the per-scene confusion matrices summed over seeds (held-out
folds) or over the 15 models (2026), as in tools/analysis/paper2_compare.py. Region: the region
of the scene's first chart; season: DJF, MAM, JJA, SON of the acquisition month.

    python tools/analysis/paper2_breakdown.py --arm multitask_comp --json paper2_breakdown.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from tools.analysis.paired_cv import metrics  # noqa: E402

SEEDS, FOLDS = (42, 43, 44), (0, 1, 2, 3, 4)
NAMES = ('water', 'cake', 'small', 'medium', 'big', 'vast')
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA',
          9: 'SON', 10: 'SON', 11: 'SON'}


def summed(paths):
    tot = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for n, c in zip(z['scenes'], z['cm']):
            n = str(n)
            tot[n] = tot.get(n, 0) + c.astype(np.float64)
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', default='multitask_comp')
    ap.add_argument('--cm-dir', default=os.environ.get('FLOE_CM_DIR'))
    ap.add_argument('--log', default=os.path.join(os.environ.get('FLOE_V2_DIR', ''), 'archive_log.jsonl'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    region = {}
    for line in open(args.log):
        r = json.loads(line)
        if r.get('charts'):
            region[r['product']] = r['charts'][0].split('_')[1]
    out = {}
    sets = {'folds': ('v2bench_full', [(s, k) for s in SEEDS for k in FOLDS]),
            '2026': ('v2bench_test', [(s, k) for s in SEEDS for k in FOLDS])}
    for name, (sub, units) in sets.items():
        a = summed([os.path.join(args.cm_dir, sub, f'baseline_s{s}_f{k}.npz') for s, k in units])
        b = summed([os.path.join(args.cm_dir, sub, f'{args.arm}_s{s}_f{k}.npz') for s, k in units])
        names = sorted(a)
        ma, mb = metrics(sum(a[n] for n in names)), metrics(sum(b[n] for n in names))
        res = {'per_class': {c: [float(ma['iou'][i]), float(mb['iou'][i])] for i, c in enumerate(NAMES)}}
        print(f'\n=== {name}: {len(names)} scenes; per-class IoU baseline -> {args.arm} (diff)')
        print('  ' + '  '.join('%s %.3f->%.3f (%+.3f)' % (c, *res['per_class'][c], res['per_class'][c][1] - res['per_class'][c][0])
                               for c in NAMES))
        for key, fn in (('region', lambda n: region.get(n, '?')), ('season', lambda n: SEASON[int(n[21:23])])):
            grp = defaultdict(list)
            for n in names:
                grp[fn(n)].append(n)
            rows = {}
            print(f'  hard-ice IoU by {key}:')
            for g in sorted(grp):
                ca, cb = sum(a[n] for n in grp[g]), sum(b[n] for n in grp[g])
                hard_px = ca[2:4].sum()
                xa, xb = metrics(ca), metrics(cb)
                ha, hb = float(np.mean(xa['iou'][2:4])), float(np.mean(xb['iou'][2:4]))
                rows[g] = {'scenes': len(grp[g]), 'hard_px': float(hard_px), 'base': ha, 'arm': hb,
                           'miou_base': xa['miou'], 'miou_arm': xb['miou']}
                print('    %-13s %4d scenes  hard px %6.1fM  hard %.3f -> %.3f (%+.3f)   mIoU %.3f -> %.3f (%+.3f)' % (
                    g, len(grp[g]), hard_px / 1e6, ha, hb, hb - ha, xa['miou'], xb['miou'], xb['miou'] - xa['miou']))
            res[key] = rows
        out[name] = res
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)
        print('->', args.json)


if __name__ == '__main__':
    main()
