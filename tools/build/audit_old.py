"""Audit an existing labelled archive against the DMI charts, scene by scene.

For every scene (file name = Sentinel-1 product id) the DMI charts within --max-minutes
that overlap its footprint (from the CDSE catalogue) are clipped to the footprint, and
their area-weighted FLOE class shares (dominant labels and egg-code compositions) are
compared with the class shares of the archive's own labels (from its tile index).
No SAR is needed.

    python -m tools.build.audit_old --tile-index $FLOE_TILE_INDEX --charts $FLOE_ROOT/raw/dmi \
        --out $FLOE_ROOT/logs/audit_old.csv
"""
import argparse
import csv
import json
import os.path as osp
import re
from datetime import datetime, timezone

import numpy as np
import shapely
from pyproj import Transformer

from tools.build import cdse
from tools.build import charts as ch

NAMES = ['water', 'cake', 'small', 'medium', 'big', 'vast']
TO_3413 = Transformer.from_crs(4326, 3413, always_xy=True)


def product_times(name):
    m = re.search(r'_(\d{8}T\d{6})_(\d{8}T\d{6})_', name)
    f = lambda s: datetime.strptime(s, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    return f(m.group(1)), f(m.group(2))


def footprint_of(entry):
    wkt = entry['Footprint'].split(';', 1)[-1].rstrip("'")
    return shapely.from_wkt(wkt)


def chart_shares(paths, fp):
    """Area-weighted dominant-label and composition shares of the charts inside fp."""
    fp_m = shapely.transform(fp, lambda c: np.stack(TO_3413.transform(c[:, 0], c[:, 1]), 1))
    dom = np.zeros(7)
    comp = np.zeros(6)
    unk = 0.0
    for p in paths:
        g = ch.read_chart(p)
        for _, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            gm = shapely.transform(geom, lambda c: np.stack(TO_3413.transform(c[:, 0], c[:, 1]), 1))
            if not gm.is_valid:
                gm = shapely.make_valid(gm)
            a = gm.intersection(fp_m).area / 1e6
            if a <= 0:
                continue
            if row['floe'] < 7:
                dom[row['floe']] += a
            if row['comp'] is not None:
                comp += a * np.asarray(row['comp'])
                unk += a * row['unknown']
    return dom, comp, unk


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tile-index', required=True)
    ap.add_argument('--charts', required=True)
    ap.add_argument('--max-minutes', type=float, default=30.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    idx = np.load(args.tile_index, allow_pickle=False)
    files = [str(f) for f in idx['files']]
    per = np.zeros((len(files), idx['hist'].shape[1]), np.int64)
    np.add.at(per, idx['scene_id'], idx['hist'].astype(np.int64))
    names = [osp.splitext(f)[0] for f in files]
    entries = cdse.lookup(names)
    index = ch.build_index(args.charts) if not osp.exists(osp.join(args.charts, 'index.json')) \
        else json.load(open(osp.join(args.charts, 'index.json')))

    rows = []
    for i, n in enumerate(names):
        e = entries.get(n)
        if e is None:
            continue
        start, stop = product_times(n)
        fp = footprint_of(e)
        m = ch.match(index, start, stop, fp, args.max_minutes)
        old = per[i, :6] / max(per[i, :6].sum(), 1)
        rec = {'product': n, 'n_charts': len(m), 'charts': ' '.join(r['name'] for r in m),
               'dt_min': min((r['dt_minutes'] for r in m), default=''),
               **{f'old_{k}': round(float(v), 4) for k, v in zip(NAMES, old)}}
        if m:
            dom, comp, unk = chart_shares([osp.join(args.charts, r['path']) for r in m], fp)
            d = dom[:6] / max(dom[:6].sum(), 1e-9)
            c = comp / max(comp.sum(), 1e-9)
            rec.update({f'chart_dom_{k}': round(float(v), 4) for k, v in zip(NAMES, d)})
            rec.update({f'chart_comp_{k}': round(float(v), 4) for k, v in zip(NAMES, c)})
            rec['chart_unknown_frac'] = round(float(unk / max(comp.sum() + unk, 1e-9)), 4)
        rows.append(rec)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'product', k))
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    matched = [r for r in rows if r['n_charts']]
    print(f'{len(rows)} scenes; {len(matched)} with a DMI chart within {args.max_minutes:.0f} min')
    if matched:
        for k in NAMES:
            o = np.mean([r[f'old_{k}'] for r in matched])
            d = np.mean([r[f'chart_dom_{k}'] for r in matched])
            c = np.mean([r[f'chart_comp_{k}'] for r in matched])
            print(f'  {k:7s} old labels {o:.4f} | chart dominant {d:.4f} | chart composition {c:.4f}')
        bv = [r for r in matched if r['chart_dom_big'] + r['chart_dom_vast'] > 0.02
              and r['old_big'] + r['old_vast'] == 0]
        print(f'  scenes where the chart has > 2% big/vast but the archive has none: {len(bv)}')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
