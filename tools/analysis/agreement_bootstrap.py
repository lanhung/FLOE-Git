"""Intervals for the chart-agreement scores of tools.build.chart_agreement.

The pairs of charts are not independent: charts of one pass share the image, and passes
of one day see the same ice. The acquisition day is therefore resampled, with all its
pairs, and the scores are recomputed from the summed matrices.

    python -m tools.analysis.agreement_bootstrap --prefix $FLOE_ROOT/logs/chart_agreement_same_image_v3
"""
import argparse
import csv
import json

import numpy as np

from tools.build.chart_agreement import FLOE_NAMES, scores


def stats(cm):
    s = scores(cm, 6)
    c = (cm + cm.T)[:6, :6] / 2
    row = c / np.maximum(c.sum(1, keepdims=True), 1e-12)
    out = {'oa': s['oa'], 'miou': s['miou'], 'miou_ice': float(s['iou'][1:6].mean())}
    out.update({f'iou_{FLOE_NAMES[k]}': float(s['iou'][k]) for k in range(6)})
    out['cake_to_water'] = float(row[1, 0])
    out['big_to_vast'] = float(row[4, 5])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', required=True, help='chart_agreement output prefix (.npz and .csv)')
    ap.add_argument('--kind', default='neighbours')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.prefix + '.npz')
    rows = list(csv.DictReader(open(args.prefix + '.csv')))
    cms = z['pair_floe']
    assert len(rows) == len(cms), (len(rows), len(cms))
    keep = np.array([r['kind'] == args.kind for r in rows])
    cms = cms[keep]
    days = np.array([r['chart_a'][:8] for r, k in zip(rows, keep) if k])
    uniq, inv = np.unique(days, return_inverse=True)
    per_day = np.zeros((len(uniq),) + cms.shape[1:])
    np.add.at(per_day, inv, cms)

    point = stats(per_day.sum(0))
    rng = np.random.default_rng(args.seed)
    draws = {k: [] for k in point}
    for _ in range(args.n):
        pick = rng.integers(0, len(uniq), len(uniq))
        for k, v in stats(per_day[pick].sum(0)).items():
            draws[k].append(v)

    with_floe = (cms[:, :6, :6].sum((1, 2)) > 0).sum()
    print(f'{args.kind}: {len(cms)} pairs ({with_floe} with a floe class in both charts) on {len(uniq)} days, '
          f'{per_day[:, :6, :6].sum():.0f} km2 with a floe class in both; {args.n} resamples of days')
    res = {}
    for k, v in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        res[k] = {'value': v, 'lo': float(lo), 'hi': float(hi)}
        print(f'  {k:14s} {v:.3f}  [{lo:.3f}, {hi:.3f}]')
    area = (per_day.sum(0) + per_day.sum(0).T)[:6, :6].sum(1) / 2
    print('  area per class (km2): ' + ', '.join(f'{FLOE_NAMES[k]} {area[k]:.0f}' for k in range(6)))
    json.dump(res, open(f'{args.prefix}_bootstrap_{args.kind}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
