"""Agreement between DMI charts where they overlap: an upper bound for models.

DMI's regional charts overlap at their borders, and charts of neighbouring regions are
often drawn on the same Sentinel-1 image (same time stamp). Where two such charts cover
the same sea they are two readings of one image, so their agreement is what the labels
themselves allow a model to reach. For every pair of charts between --min-minutes and
--max-minutes apart whose polygons overlap, the area (EASE-Grid 2.0 North, equal area)
of every pair of intersecting polygons is added to confusion matrices of the FLOE class
(AutoICE dominance rule), SIC (tenths) and SOD, and the egg-code floe-size mixes are
compared. With --min-minutes 600 --max-minutes 2160 the same tool gives the change
between consecutive charts of one area (drift and real change included).

Overlaps where both charts carry the same polygon (shapes within 5 % IoU of each other,
e.g. copied from the neighbouring chart) are counted separately, because they agree by
construction and say little about independent judgement. Pairs of the same DMI region
(a chart issued again a few minutes later, or the next chart of that region) are
reported apart from pairs of neighbouring regions for the same reason.

    python -m tools.build.chart_agreement --charts $FLOE_ROOT/raw/dmi \
        --out $FLOE_ROOT/logs/chart_agreement_same_image
"""
import argparse
import csv
import json
import os.path as osp
from datetime import datetime
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import shapely
from shapely.geometry import box

from tools.build import charts as ch

FLOE_NAMES = ('water', 'cake', 'small', 'medium', 'big', 'vast', 'bergs')
SOD_N, SIC_N, FLOE_N = 6, 11, 7
_ROOT = None


@lru_cache(maxsize=64)
def _chart(path):
    g = ch.read_chart(osp.join(_ROOT, path)).to_crs(6931)
    geoms = shapely.make_valid(g.geometry.values)
    comp = [None if c is None else np.asarray(c, np.float64) for c in g['comp']]
    return (geoms, g['floe'].to_numpy(np.int64), g['sic'].to_numpy(np.int64),
            g['sod'].to_numpy(np.int64), comp, g['unknown'].to_numpy(np.float64))


def _init(root):
    global _ROOT
    _ROOT = root


def compare(job):
    ra, rb, dt = job
    ga, fa, sa, da, ca, ua = _chart(ra['path'])
    gb, fb, sb, db, cb, ub = _chart(rb['path'])
    ia, ib = shapely.STRtree(gb).query(ga, predicate='intersects')
    out = {'floe': np.zeros((FLOE_N, FLOE_N)), 'sic': np.zeros((SIC_N, SIC_N)),
           'sod': np.zeros((SOD_N, SOD_N)), 'floe_copied': np.zeros((FLOE_N, FLOE_N)),
           'tv': 0.0, 'tv_area': 0.0, 'water_diff': 0.0, 'water_area': 0.0, 'area': 0.0,
           'copied_area': 0.0}
    kind = 'same_region' if ra['region'] == rb['region'] else 'neighbours'
    if not len(ia):
        return ra['name'], rb['name'], dt, kind, out
    inter = shapely.area(shapely.intersection(ga[ia], gb[ib])) / 1e6        # km2
    union = shapely.area(ga[ia]) / 1e6 + shapely.area(gb[ib]) / 1e6 - inter
    copied = inter / np.maximum(union, 1e-9) > 0.95
    for k in np.nonzero(inter > 0)[0]:
        a, b, w = ia[k], ib[k], inter[k]
        if fa[a] < FLOE_N and fb[b] < FLOE_N:
            out['floe_copied' if copied[k] else 'floe'][fa[a], fb[b]] += w
        if copied[k]:
            out['copied_area'] += w
            continue
        out['area'] += w
        if sa[a] < SIC_N and sb[b] < SIC_N:
            out['sic'][sa[a], sb[b]] += w
        if da[a] < SOD_N and db[b] < SOD_N:
            out['sod'][da[a], db[b]] += w
        qa, qb = ca[a], cb[b]
        if qa is not None and qb is not None and ua[a] <= 0.5 and ub[b] <= 0.5:
            ia_, ib_ = qa[1:].sum(), qb[1:].sum()
            out['water_diff'] += w * abs(qa[0] - qb[0])
            out['water_area'] += w
            if ia_ > 0 and ib_ > 0:
                out['tv'] += w * 0.5 * np.abs(qa[1:] / ia_ - qb[1:] / ib_).sum()
                out['tv_area'] += w
    return ra['name'], rb['name'], dt, kind, out


def chart_pairs(index, lo, hi):
    recs = sorted(index, key=lambda r: r['time'])
    t = [datetime.fromisoformat(r['time']) for r in recs]
    for i, r in enumerate(recs):
        bi = box(*r['bbox'])
        for j in range(i + 1, len(recs)):
            dt = (t[j] - t[i]).total_seconds() / 60
            if dt > hi:
                break
            if dt >= lo and bi.intersects(box(*recs[j]['bbox'])):
                yield r, recs[j], dt


def scores(cm, n_eval):
    """Agreement scores of a confusion matrix between two readers, symmetrised."""
    c = (cm + cm.T)[:n_eval, :n_eval] / 2
    tp = np.diag(c)
    iou = tp / np.maximum(c.sum(0) + c.sum(1) - tp, 1e-12)
    f1 = 2 * tp / np.maximum(c.sum(0) + c.sum(1), 1e-12)
    w = c.sum(1) / max(c.sum(), 1e-12)
    present = c.sum(1) > 0
    return {'oa': float(tp.sum() / max(c.sum(), 1e-12)), 'miou': float(iou[present].mean()),
            'f1_weighted': float((w * f1).sum()), 'iou': iou, 'share': w}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--charts', required=True, help='DMI chart root with index.json')
    ap.add_argument('--out', required=True, help='output prefix (.npz, .csv)')
    ap.add_argument('--min-minutes', type=float, default=0.0)
    ap.add_argument('--max-minutes', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    index = [r for r in json.load(open(osp.join(args.charts, 'index.json'))) if r['name'] not in ch.EXCLUDED_CHARTS]
    jobs = list(chart_pairs(index, args.min_minutes, args.max_minutes))
    print(f'{len(index)} charts, {len(jobs)} pairs {args.min_minutes:.0f}-{args.max_minutes:.0f} min '
          'apart with overlapping bounding boxes', flush=True)
    tot = {}
    rows, pair_floe = [], []
    with Pool(args.workers, initializer=_init, initargs=(args.charts,)) as pool:
        for na, nb, dt, kind, o in pool.imap_unordered(compare, jobs, chunksize=4):
            if kind not in tot:
                tot[kind] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.items()}
            else:
                for k, v in o.items():
                    tot[kind][k] = tot[kind][k] + v
            if o['area'] + o['copied_area'] > 0:
                f = o['floe'][:6, :6]
                rows.append({'chart_a': na, 'chart_b': nb, 'kind': kind, 'dt_minutes': round(dt, 1),
                             'overlap_km2': round(o['area'], 1), 'copied_km2': round(o['copied_area'], 1),
                             'floe_agree': round(float(np.trace(f) / f.sum()), 4) if f.sum() else ''})
                pair_floe.append(o['floe'])

    # pair_floe: one FLOE matrix per row of the csv, for bootstrapping over chart pairs
    np.savez(args.out + '.npz', pair_floe=np.array(pair_floe), pair_kind=np.array([r['kind'] for r in rows]),
             **{f'{kind}_{k}': np.asarray(v) for kind, t in tot.items() for k, v in t.items()})
    with open(args.out + '.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['chart_a'])
        w.writeheader()
        w.writerows(rows)

    for kind, t in sorted(tot.items()):
        n_pairs = sum(r['kind'] == kind for r in rows)
        print(f'\n== {kind}: {n_pairs} pairs overlap, {t["area"]:.0f} km2 drawn separately, '
              f'{t["copied_area"]:.0f} km2 the same polygon in both charts')
        for name, cm, n in (('FLOE', t['floe'], 6), ('SOD', t['sod'], SOD_N), ('SIC', t['sic'], SIC_N)):
            s = scores(cm, n)
            print(f'{name}: OA {s["oa"]:.3f}  mIoU {s["miou"]:.3f}  weighted F1 {s["f1_weighted"]:.3f}')
            if name == 'FLOE':
                for k in range(6):
                    print(f'   {FLOE_NAMES[k]:7s} IoU {s["iou"][k]:.3f}  area share {s["share"][k]:.3f}')
        if t['tv_area']:
            print(f'floe-size mix of the ice: mean total-variation distance {t["tv"] / t["tv_area"]:.3f}; '
                  f'open-water share: mean abs. difference {t["water_diff"] / max(t["water_area"], 1e-9):.3f}')
    print(f'-> {args.out}.npz, {args.out}.csv')

if __name__ == '__main__':
    main()
