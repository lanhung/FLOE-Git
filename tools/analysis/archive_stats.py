"""Numbers that describe the rebuilt archive: pairing, quality control, classes, polygons.

    python -m tools.analysis.archive_stats --table $FLOE_LOG_DIR/polygons_v2.npz \
        --log $FLOE_V2_DIR/archive_log.jsonl --pairs $FLOE_ROOT/raw/pairs_20*.json [--json stats.json]

The table is the output of tools.build.polygon_table. Everything printed here is what
the dataset paper quotes, so the paper can be checked against one command.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

NAMES = ('water', 'cake', 'small', 'medium', 'big', 'vast')
KM2 = 0.08 ** 2


def pct(a, q):
    return [float(v) for v in np.percentile(a, q)] if len(a) else [float('nan')] * len(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--table', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--pairs', nargs='*', default=[])
    ap.add_argument('--json')
    args = ap.parse_args()
    out = {}

    # pairing and build outcome, per year
    status, rec = defaultdict(Counter), {}
    for line in open(args.log):
        r = json.loads(line)
        rec[r['product']] = r
    for name, r in rec.items():
        status[name.split('_')[4][:4]][r['status']] += 1
    charts = defaultdict(lambda: [0, 0, 0])
    for p in args.pairs:
        d = json.load(open(p))
        for c in d['unpaired_charts']:
            charts[c[:4]][0] += 1
        for c in d['charts']:
            charts[c[:4]][0] += 1
            charts[c[:4]][1] += 1
        for f in d['frames']:
            charts[f.split('_')[4][:4]][2] += 1
    print('year  charts  paired  frames | built: ' + '  '.join(sorted({s for c in status.values() for s in c})))
    for y in sorted(set(charts) | set(status)):
        print(f'{y}  {charts[y][0]:6d}  {charts[y][1]:6d}  {charts[y][2]:6d} | {dict(status[y])}')
    out['pairing'] = {y: {'charts': charts[y][0], 'charts_paired': charts[y][1], 'frames': charts[y][2],
                          **status[y]} for y in sorted(set(charts) | set(status))}

    ok = [r for r in rec.values() if r['status'] == 'ok']
    dt = np.array([min(abs(v) for v in r['dt_minutes']) for r in ok])
    err = np.array([r['georef_err_px'] for r in ok])
    hv = np.array([r['nonpositive_replaced']['hv'] for r in ok])
    out['qc'] = {'scenes': len(ok), 'dt_minutes': pct(dt, [50, 95, 100]), 'georef_err_px': pct(err, [50, 95, 100]),
                 'hv_replaced': pct(hv, [50, 95, 100]), 'ipf': dict(Counter(r['ipf'] for r in ok)),
                 'mission': dict(Counter(r['product'][:3] for r in ok)),
                 'charts_per_scene': dict(Counter(len(r['charts']) for r in ok))}
    print('\nquality control of the kept scenes:', json.dumps(out['qc'], indent=1))

    t = np.load(args.table)
    scenes = t['scenes']
    year = np.array([int(n.split('_')[4][:4]) for n in scenes])[t['scene']]
    floe, n, comp, unk = t['floe'], t['labelled'].astype(np.float64), t['composition'].astype(np.float64), t['unknown']
    ice = (floe >= 1) & (floe <= 5)
    has_comp = ice & (unk <= 0.5) & (comp[:, 1:].sum(1) > 0)
    q = comp / comp.sum(1, keepdims=True).clip(1e-12)
    print(f'\n{len(scenes)} scenes, {len(floe)} polygons ({int(ice.sum())} with an ice floe class), '
          f'{t["pixels"].sum() * KM2 / 1e6:.2f} M km2 charted, {n.sum() * KM2 / 1e6:.2f} M km2 with a floe class')
    out['totals'] = {'scenes': len(scenes), 'polygons': int(len(floe)), 'ice_polygons': int(ice.sum()),
                     'charted_km2': float(t['pixels'].sum() * KM2), 'labelled_km2': float(n.sum() * KM2)}

    print('\nlabel pixels (Mpx) by year and class; last rows: all years, and what the egg codes imply')
    out['classes'] = {}
    for y in list(np.unique(year)) + ['all']:
        m = np.ones(len(floe), bool) if y == 'all' else year == y
        lab = np.array([n[m & (floe == k)].sum() for k in range(6)])
        out['classes'][str(y)] = {'scenes': int(len(np.unique(t['scene'][m]))), 'label_px': lab.tolist()}
        print(f'{y!s:>5} {len(np.unique(t["scene"][m])):5d} scenes  ' + '  '.join(f'{NAMES[k]} {lab[k] / 1e6:7.0f}' for k in range(6)))
    implied = (n[has_comp, None] * q[has_comp]).sum(0)
    implied[0] += n[floe == 0].sum()
    out['classes']['implied_px'] = implied.tolist()
    print('egg code            ' + '  '.join(f'{NAMES[k]} {implied[k] / 1e6:7.0f}' for k in range(6)))

    # how mixed the ice polygons are
    top = q[has_comp].max(1)
    top_ice = (q[has_comp][:, 1:] / q[has_comp][:, 1:].sum(1, keepdims=True)).max(1)
    w = n[has_comp]
    out['mixing'] = {
        'ice_polygons_with_composition': int(has_comp.sum()),
        'pure_polygons_ge0.9': float((top >= 0.9).mean()), 'pure_area_ge0.9': float(w[top >= 0.9].sum() / w.sum()),
        'pure_ice_mix_ge0.9': float((top_ice >= 0.9).mean()), 'pure_ice_mix_area_ge0.9': float(w[top_ice >= 0.9].sum() / w.sum()),
        'single_floe_class': float((top_ice >= 0.999).mean()),
        'mean_water_share_by_label': [float((w * q[has_comp][:, 0])[floe[has_comp] == k].sum() / max(w[floe[has_comp] == k].sum(), 1))
                                      for k in range(1, 6)],
        'masked_ice_polygons': int((floe == 255).sum()), 'masked_ice_km2': float(t['pixels'][floe == 255].sum() * KM2),
    }
    print('\nmixing of ice polygons:', json.dumps(out['mixing'], indent=1))
    size = t['pixels'] * KM2
    out['polygon_km2'] = pct(size, [10, 50, 90])
    out['polygons_per_scene'] = pct(np.bincount(t['scene']), [10, 50, 90])
    print(f'polygon area in a scene (km2) p10/p50/p90: {out["polygon_km2"]};  polygons per scene: {out["polygons_per_scene"]}')

    print('\nscenes and ice by chart region')
    out['regions'] = {}
    for reg in sorted(set(t['region'])):
        m = t['region'] == reg
        lab = np.array([n[m & (floe == k)].sum() for k in range(6)])
        out['regions'][str(reg)] = {'scenes': int(len(np.unique(t['scene'][m]))), 'label_px': lab.tolist()}
        print(f'{reg!s:13s} {len(np.unique(t["scene"][m])):5d} scenes  ' + '  '.join(f'{lab[k] / 1e6:7.0f}' for k in range(6)))
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
