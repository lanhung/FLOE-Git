"""How wrong are dominant-class floe labels, taking the chart's own egg code as truth?

Every pixel of a chart polygon gets the polygon's dominant FLOE class, but the egg code
says the polygon is a mix (e.g. 6/10 medium, 3/10 big, 1/10 water). If the pixels
really follow that mix, a perfect pixel classifier would disagree with the labels on
every pixel outside the dominant share. This script builds that implied confusion
matrix (rows: label, columns: class implied by the egg code, pixel counts) over the
scenes of the given datalists and reports the scores a perfect model would get against
the labels, once with the open-water share of ice polygons as water and once with only
the floe-size mix of the ice (water share removed).

    python -m tools.build.label_purity --data-dir $FLOE_ROOT/data_v2 \
        --lists datalists/v2_2023/fold0_val.json --out $FLOE_ROOT/logs/label_purity_v2_2023_f0val
"""
import argparse
import json
import os.path as osp
from multiprocessing import Pool

import h5py
import hdf5plugin  # noqa: F401  zstd filter
import numpy as np

NAMES = ('water', 'cake', 'small', 'medium', 'big', 'vast')


def one(path):
    with h5py.File(path, 'r') as f:
        poly = f['polygon_id'][...]
        lab = f['FLOE'][...]
        floe = f['poly_floe'][...].astype(np.int64)
        comp = f['poly_composition'][...].astype(np.float64)
        unk = f['poly_unknown'][...].astype(np.float64)
    m = (poly >= 0) & (lab < 6)
    n = np.bincount(poly[m], minlength=len(floe)).astype(np.float64)
    full = np.zeros((6, 6))
    ice = np.zeros((6, 6))
    skipped = 0.0
    for p in np.nonzero(n)[0]:
        L = floe[p]
        if L >= 6:
            continue
        if L == 0:                                   # open-water polygon
            full[0, 0] += n[p]
            ice[0, 0] += n[p]
            continue
        q = comp[p]
        if not np.isfinite(q).all() or unk[p] > 0.5 or q[1:].sum() <= 0:
            skipped += n[p]                          # mostly unknown ice: no implied truth
            continue
        full[L] += n[p] * q / q.sum()
        ice[L, 1:] += n[p] * q[1:] / q[1:].sum()
    return full, ice, skipped


def scores(cm):
    tp = np.diag(cm)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1e-12)
    purity = tp / np.maximum(cm.sum(1), 1e-12)               # share of a label that is right
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-12)
    w = cm.sum(1) / cm.sum()
    present = cm.sum(1) > 0
    return iou, purity, float(iou[present].mean()), float((w * f1).sum()), float(tp.sum() / cm.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--lists', nargs='+', required=True)
    ap.add_argument('--out', required=True, help='output prefix (.npz)')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    names = sorted({n for p in args.lists for n in json.load(open(p))})
    paths = [osp.join(args.data_dir, n) for n in names]
    full, ice, skipped = np.zeros((6, 6)), np.zeros((6, 6)), 0.0
    with Pool(args.workers) as pool:
        for f, i, s in pool.imap_unordered(one, paths):
            full += f
            ice += i
            skipped += s
    np.savez(args.out + '.npz', full=full, ice=ice, skipped=skipped, scenes=np.array(names))
    print(f'{len(names)} scenes; {ice.sum() / 1e6:.0f} Mpx labelled with an implied truth, '
          f'{skipped / 1e6:.0f} Mpx skipped (mostly unknown ice)')
    for title, cm in (('open-water share counted as water', full), ('floe-size mix of the ice only', ice)):
        iou, pur, miou, f1w, oa = scores(cm)
        print(f'\nA perfect pixel model against the dominant labels ({title}):')
        print(f'  mIoU {miou:.3f}  weighted F1 {f1w:.3f}  OA {oa:.3f}')
        for k in range(6):
            print(f'  {NAMES[k]:7s} IoU {iou[k]:.3f}  label purity {pur[k]:.3f}  label share {cm[k].sum() / cm.sum():.3f}')
    print(f'-> {args.out}.npz')


if __name__ == '__main__':
    main()
