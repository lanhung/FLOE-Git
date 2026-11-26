"""Paper 2: gather everything the figures and the appendix tables need into one JSON.

Read-only over the benchmark outputs (confusion matrices, polygon dumps, training logs, polygon
table). Nothing here changes a result; the comparison itself is tools/analysis/paper2_compare.py.

  confusion      6x6 matrices summed over runs (held-out folds, 2026) for both arms
  runs           every run: stop epoch, best validation mIoU at training time, scores on its
                 held-out fold, on 2026 and on the AutoICE test scenes
  curves         smoothed validation mIoU per validation (every 2 epochs) of every run
  mixed          class make-up of the mixed polygons: chart vs prediction (pixel-weighted)
  coverage       share of the charted ice area that carries each kind of chart supervision
  autoice        AutoICE test scores by ice service
  examples       2026 scenes with the largest labelled area of small and medium floes on the
                 east and on the west coast (chosen by content, before looking at predictions)

    python tools/analysis/paper2_figdata.py --arm multitask_comp --extra-arms multitask --json paper2_figdata.json

`arms` in the JSON is baseline, --arm, then --extra-arms (the ablation arm of DECISIONS.md item 14);
the figures of the primary comparison use the first two.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from floe.composition import composition_confusion, confusion_scores  # noqa: E402
from tools.analysis.paired_cv import metrics  # noqa: E402

SEEDS, FOLDS = (42, 43, 44), (0, 1, 2, 3, 4)
EAST = ('NorthEast', 'CentralEast', 'SouthEast', 'North')
WEST = ('Qaanaaq', 'NorthWest', 'CentralWest', 'SouthWest')


def cm_of(path):
    z = np.load(path, allow_pickle=True)
    return [str(s) for s in z['scenes']], z['cm'].astype(np.float64)


def score(cm):
    m = metrics(cm)
    return {'miou': float(m['miou']), 'hard': float(np.mean(m['iou'][2:4])), 'ice': float(np.mean(m['iou'][1:])),
            'wf1': float(m['weighted_f1']), 'iou': [float(x) for x in m['iou']]}


def comp_of(path):
    d = np.load(path, allow_pickle=True)
    c = composition_confusion(d['hist'], d['q'], d['u'])
    return float(confusion_scores(c['full'])['miou']), float(confusion_scores(c['ice'])['miou'])


def mixed_mask(h, q, u):
    n, ice = h.sum(1), q[:, 1:]
    return (n >= 64) & (u <= 0.5) & (ice.sum(1) > 0) & (ice.max(1) < 0.65 * ice.sum(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', default='multitask_comp')
    ap.add_argument('--extra-arms', nargs='*', default=[], help='further arms, e.g. the ablation arm multitask')
    ap.add_argument('--root', default=os.environ.get('FLOE_ROOT'))
    ap.add_argument('--table', default=None, help='polygon table of the whole dataset')
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    R = args.root
    C, L = os.path.join(R, 'cm'), os.path.join(R, 'logs')
    arms = ('baseline', args.arm, *args.extra_arms)
    out = {'arms': list(arms)}

    # confusion matrices and runs
    conf = {'folds': {}, '2026': {}}
    runs = []
    for arm in arms:
        tot = {'folds': 0, '2026': 0}
        for s in SEEDS:
            for k in FOLDS:
                log = open(os.path.join(L, f'floe_{arm}_v2benchf{k}_s{s}.log'), errors='ignore').read()
                stop = re.search(r'Early stopping triggered at epoch (\d+)\. Best FLOE mIoU: ([0-9.]+)', log)
                curve = [float(x) for x in re.findall(r'Smoothed val mIoU \(w=3\): ([0-9.]+)', log)]
                row = {'arm': arm, 'seed': s, 'fold': k,
                       'stop_epoch': int(stop.group(1)) if stop else 200,
                       'best_val_miou': float(stop.group(2)) if stop else max(curve), 'curve': curve}
                for name, sub in (('folds', 'v2bench_full'), ('2026', 'v2bench_test')):
                    _, cm = cm_of(os.path.join(C, sub, f'{arm}_s{s}_f{k}.npz'))
                    tot[name] = tot[name] + cm.sum(0)
                    sc = score(cm.sum(0))
                    sc.pop('iou')
                    sc['comp'], sc['comp_ice'] = comp_of(os.path.join(C, 'poly', sub, f'{arm}_s{s}_f{k}.npz'))
                    row[name] = sc
                names, cm = cm_of(os.path.join(C, 'a4a', f'v2bench_{arm}_s{s}_f{k}_test.npz'))
                row['autoice'] = {}
                for svc in ('all', 'dmi', 'cis'):
                    idx = [i for i, n in enumerate(names) if svc == 'all' or svc in n]
                    m = metrics(cm[idx].sum(0))
                    row['autoice'][svc] = {'miou': float(m['miou']), 'wf1': float(m['weighted_f1'])}
                runs.append(row)
        for name in conf:
            conf[name][arm] = {'cm': tot[name].tolist(), **score(tot[name])}
    out['confusion'], out['runs'] = conf, runs

    # mixed polygons: chart vs predicted class make-up, pixel-weighted, summed over runs
    mixed = {}
    for name, sub in (('folds', 'v2bench_full'), ('2026', 'v2bench_test')):
        mixed[name] = {}
        for arm in arms:
            pred, chart, npoly, npx, tv = np.zeros(6), np.zeros(6), 0, 0.0, []
            for s in SEEDS:
                for k in FOLDS:
                    d = np.load(os.path.join(C, 'poly', sub, f'{arm}_s{s}_f{k}.npz'), allow_pickle=True)
                    h, q, u = d['hist'].astype(np.float64), d['q'].astype(np.float64), d['u'].astype(np.float64)
                    m = mixed_mask(h, q, u)
                    qn = q[m] / q[m].sum(1, keepdims=True)
                    pred += h[m].sum(0)
                    chart += (qn * h[m].sum(1, keepdims=True)).sum(0)
                    npoly += int(m.sum())
                    npx += float(h[m].sum())
                    hi = h[m][:, 1:] / h[m][:, 1:].sum(1, keepdims=True).clip(1e-9)     # predicted mix of the ice
                    qi = q[m][:, 1:] / q[m][:, 1:].sum(1, keepdims=True)                # chart mix of the ice
                    w = h[m].sum(1)
                    tv.append(float((0.5 * np.abs(hi - qi).sum(1) * w).sum() / w.sum()))
            mixed[name][arm] = {'pred': (pred / pred.sum()).tolist(), 'chart': (chart / chart.sum()).tolist(),
                                'polygons_per_run': npoly / 15.0, 'mpx_per_run': npx / 15.0 / 1e6,
                                'tv_ice_mix': float(np.mean(tv))}
    out['mixed'] = mixed

    # coverage of the charted ice area by each kind of supervision (whole dataset)
    if args.table:
        t = np.load(args.table)
        px = t['pixels'].astype(np.float64)
        floe, sic, sod = t['floe'], t['sic'], t['sod']
        comp, unk = t['composition'].astype(np.float64), t['unknown'].astype(np.float64)
        ice = floe != 0                                          # every polygon that is not open water
        area = px[ice].sum()
        usable = np.nan_to_num(unk, nan=1.0) <= 0.5
        has_mix = np.nan_to_num(comp[:, 1:].sum(1)) > 0
        out['coverage'] = {
            'ice_polygons': int(ice.sum()), 'ice_mpx': float(area / 1e6),
            'floe_label': float(px[ice & (floe >= 1) & (floe <= 5)].sum() / area),
            'sic_label': float(px[ice & (sic != 255)].sum() / area),
            'sod_label': float(px[ice & (sod != 255)].sum() / area),
            'composition': float(px[ice & usable & has_mix].sum() / area),
            'composition_without_floe_label': float(px[ice & usable & has_mix & ~((floe >= 1) & (floe <= 5))].sum() / area),
            'sic_without_floe_label': float(px[ice & (sic != 255) & ~((floe >= 1) & (floe <= 5))].sum() / area),
            'sod_without_floe_label': float(px[ice & (sod != 255) & ~((floe >= 1) & (floe <= 5))].sum() / area),
            'sic_values': sorted(int(v) for v in np.unique(sic)), 'sod_values': sorted(int(v) for v in np.unique(sod)),
        }

    # AutoICE by service
    a4a = {}
    for svc in ('all', 'dmi', 'cis'):
        a4a[svc] = {}
        for arm in arms:
            v = np.array([[r['autoice'][svc]['miou'], r['autoice'][svc]['wf1']] for r in runs if r['arm'] == arm])
            a4a[svc][arm] = {'miou': [float(v[:, 0].mean()), float(v[:, 0].std(ddof=1))],
                             'wf1': [float(v[:, 1].mean()), float(v[:, 1].std(ddof=1))]}
    out['autoice'] = a4a

    # example scenes: most labelled small+medium pixels on each coast in 2026
    region = {}
    for line in open(os.path.join(R, 'data_v2', 'archive_log.jsonl')):
        r = json.loads(line)
        if r.get('charts'):
            region[r['product']] = r['charts'][0].split('_')[1]
    names, cm = cm_of(os.path.join(C, 'v2bench_test', 'baseline_s42_f0.npz'))
    hard_px = cm[:, 2:4, :].sum((1, 2))
    ex = {}
    for coast, regs in (('east', EAST), ('west', WEST)):
        idx = [i for i, n in enumerate(names) if region.get(n) in regs]
        best = max(idx, key=lambda i: hard_px[i])
        ex[coast] = {'scene': names[best], 'region': region[names[best]], 'hard_mpx': float(hard_px[best] / 1e6)}
    out['examples'] = ex

    json.dump(out, open(args.json, 'w'))
    print('coverage', out.get('coverage'))
    print('examples', ex)
    print('mixed 2026', {a: [round(x, 3) for x in mixed['2026'][a]['pred']] for a in arms}, 'chart',
          [round(x, 3) for x in mixed['2026']['baseline']['chart']])
    print('->', args.json)


if __name__ == '__main__':
    main()
