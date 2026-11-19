"""Paper 2: the benchmark arm against the baseline, as fixed in paper_method/DECISIONS.md.

Written and committed before the 2026 scores of either arm were read for paper 2.

Co-primary measures (DECISIONS.md item 6): (a) mean IoU over six classes against the dominant-class
labels; (b) the full composition score (floe.composition, polygons >= 64 px, at most half unknown).
Secondary: mean IoU of the ice classes, hard-ice IoU (small, medium), weighted F1, per-class IoU.

For each test set (held-out folds; 2026) and both arms:
  - per seed (folds) or per model (2026: 15 = 5 folds x 3 seeds), the paired difference arm - baseline;
    mean and a t interval over seeds (folds) or over models (2026);
  - per-scene confusion matrices and per-polygon histograms summed over seeds (folds) or models
    (2026), and a paired bootstrap over the scene groups of tools.build.split (DECISIONS.md item 7).
AutoICE test: mean and sd over the 15 models of each arm, paired differences, by ice service (no
composition score: the AutoICE files carry no chart compositions).

Reading of item 6, written down here before the results (the decision did not say how
"improves" is judged): the arm is called better only if the composition difference is positive AND
the 95 % group-bootstrap interval of the composition difference lies above zero, AND the mean-IoU
difference is not below -0.005. Everything is reported whatever the outcome.

    python tools/analysis/paper2_compare.py --arm multitask_comp --json paper2_compare.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from floe.composition import composition_confusion, confusion_scores  # noqa: E402
from tools.analysis.paired_cv import metrics  # noqa: E402
from tools.build.split import groups_of, load_scenes  # noqa: E402

SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
MARGIN = 0.005
T975 = {2: 4.303, 14: 2.145}
KEYS = ('miou', 'comp', 'miou_ice', 'hard', 'wf1')


def load_cm(path):
    z = np.load(path, allow_pickle=True)
    return [str(s) for s in z['scenes']], z['cm'].astype(np.float64)


def load_poly(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in ('scene', 'polygon', 'hist', 'q', 'u')}


def cat_poly(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def scores(cm_sum, poly):
    m = metrics(cm_sum)
    out = {'miou': m['miou'], 'miou_ice': float(np.mean(m['iou'][1:])), 'hard': float(np.mean(m['iou'][2:4])),
           'wf1': m['weighted_f1'], 'iou': [float(x) for x in m['iou']]}
    out['comp'] = (confusion_scores(composition_confusion(poly['hist'], poly['q'], poly['u'])['full'])['miou']
                   if poly is not None else float('nan'))
    return out


def poly_key(p):
    return np.char.add(np.char.add(p['scene'].astype(str), '#'), p['polygon'].astype(str))


def summed_poly(polys):
    """Sum the predicted histograms of the same polygons over several runs (q and u are the chart's)."""
    base = polys[0]
    key = poly_key(base)
    order = np.argsort(key)
    hist = base['hist'].astype(np.float64).copy()
    for p in polys[1:]:
        k = poly_key(p)
        if not np.array_equal(np.sort(k), key[order]):
            raise SystemExit('polygon dumps of the runs do not hold the same polygons')
        idx = np.argsort(k)
        h = np.empty_like(hist)
        h[order] = p['hist'][idx]
        hist += h
    return {'scene': base['scene'], 'polygon': base['polygon'], 'hist': hist, 'q': base['q'], 'u': base['u']}


def bootstrap(names, cm_a, cm_b, poly_a, poly_b, group_of, n_boot, rng):
    """Paired bootstrap over scene groups of (b - a) for every key; cm_* [S, C, C] per scene."""
    gid = np.array([group_of[n] for n in names])
    groups = np.unique(gid)
    rows_of = {g: np.where(gid == g)[0] for g in groups}
    pscene_a = poly_a['scene'].astype(str)
    prow_of_scene = {}
    for i, s in enumerate(pscene_a):
        prow_of_scene.setdefault(s, []).append(i)
    prows_of_group = {g: np.array([r for i in rows_of[g] for r in prow_of_scene.get(names[i], [])], dtype=int)
                      for g in groups}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(groups, len(groups))
        rows = np.concatenate([rows_of[g] for g in pick])
        prow = np.concatenate([prows_of_group[g] for g in pick])
        sel = lambda p: {k: v[prow] for k, v in p.items()}  # noqa: E731
        sa, sb = scores(cm_a[rows].sum(0), sel(poly_a)), scores(cm_b[rows].sum(0), sel(poly_b))
        draws.append([sb[k] - sa[k] for k in KEYS])
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(KEYS)}


def t_interval(d):
    d = np.asarray(d, float)
    n = len(d)
    half = T975[n - 1] * d.std(ddof=1) / np.sqrt(n)
    return [float(d.mean() - half), float(d.mean() + half)]


def evaluate_set(name, files, group_of, n_boot, rng):
    """files: list of (arm_cm, arm_poly, base_cm, base_poly) per unit (seed for folds, model for 2026)."""
    print(f'\n=== {name} ===')
    per_unit, names_ref = [], None
    sum_a = sum_b = None
    polys_a, polys_b = [], []
    for unit, (cb, pb, ca, pa) in files.items():
        parts_a = [load_cm(c) for c in ca]
        parts_b = [load_cm(c) for c in cb]
        names = [n for p in parts_a for n in p[0]]
        if [n for p in parts_b for n in p[0]] != names:
            raise SystemExit(f'{name} {unit}: the two arms were not scored on the same scenes in the same order')
        cm_a = np.concatenate([p[1] for p in parts_a])
        cm_b = np.concatenate([p[1] for p in parts_b])
        poly_a = cat_poly([load_poly(p) for p in pa])
        poly_b = cat_poly([load_poly(p) for p in pb])
        sa, sb = scores(cm_a.sum(0), poly_a), scores(cm_b.sum(0), poly_b)
        per_unit.append({'unit': unit, 'base': sa, 'arm': sb, 'diff': {k: sb[k] - sa[k] for k in KEYS}})
        if names_ref is None:
            names_ref = names
            sum_a, sum_b = np.zeros_like(cm_a), np.zeros_like(cm_b)
        order = {n: i for i, n in enumerate(names)}
        idx = [order[n] for n in names_ref]
        sum_a += cm_a[idx]
        sum_b += cm_b[idx]
        polys_a.append(poly_a)
        polys_b.append(poly_b)
    pa_sum, pb_sum = summed_poly(polys_a), summed_poly(polys_b)
    pooled_a, pooled_b = scores(sum_a.sum(0), pa_sum), scores(sum_b.sum(0), pb_sum)
    boot = bootstrap(names_ref, sum_a, sum_b, pa_sum, pb_sum, group_of, n_boot, rng)
    res = {'units': len(per_unit), 'scenes': len(names_ref), 'per_unit': per_unit, 'bootstrap_groups': boot,
           'pooled': {'base': pooled_a, 'arm': pooled_b}}
    print('%-9s %-16s %-16s %-10s %-22s %-22s' % ('measure', 'baseline mean', 'arm mean', 'diff', 't interval', 'group bootstrap'))
    for k in KEYS:
        d = [u['diff'][k] for u in per_unit]
        ti = t_interval(d)
        res.setdefault('diff', {})[k] = {'mean': float(np.mean(d)), 't95': ti, 'boot95': boot[k], 'per_unit': d}
        print('%-9s %.4f ± %.4f  %.4f ± %.4f  %+.4f   [%+.4f, %+.4f]     [%+.4f, %+.4f]' % (
            k, np.mean([u['base'][k] for u in per_unit]), np.std([u['base'][k] for u in per_unit], ddof=1),
            np.mean([u['arm'][k] for u in per_unit]), np.std([u['arm'][k] for u in per_unit], ddof=1),
            np.mean(d), *ti, *boot[k]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', default='multitask_comp')
    ap.add_argument('--cm-dir', default=os.environ.get('FLOE_CM_DIR'))
    ap.add_argument('--log', default=os.path.join(os.environ.get('FLOE_V2_DIR', ''), 'archive_log.jsonl'))
    ap.add_argument('--data-dir', default=os.environ.get('FLOE_V2_DIR'))
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--skip-2026', action='store_true', help='only the held-out folds and AutoICE')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    C, rng = args.cm_dir, np.random.default_rng(0)

    def p(sub, arm, s, k, poly=False):
        return os.path.join(C, 'poly' if poly else '', sub, f'{arm}_s{s}_f{k}.npz').replace('//', '/')

    all_scenes = load_scenes(args.log, args.data_dir)
    out = {'arm': args.arm}

    # held-out folds: unit = seed (five folds pooled)
    fold_files = {s: ([p('v2bench_full', args.arm, s, k) for k in FOLDS], [p('v2bench_full', args.arm, s, k, True) for k in FOLDS],
                      [p('v2bench_full', 'baseline', s, k) for k in FOLDS], [p('v2bench_full', 'baseline', s, k, True) for k in FOLDS])
                  for s in SEEDS}
    missing = [f for v in fold_files.values() for l in v for f in l if not os.path.exists(f)]
    if missing:
        raise SystemExit(f'missing fold dumps: {missing[:3]} ... ({len(missing)})')
    fold_names = sorted({n for f in fold_files[42][2] for n in load_cm(f)[0]})
    fold_groups = {n: i for i, g in enumerate(groups_of({n: all_scenes[n] for n in fold_names})) for n in g}
    out['folds'] = evaluate_set('held-out folds (units: seeds; folds pooled per seed)', fold_files, fold_groups, args.n_boot, rng)

    if not args.skip_2026:
        test_files = {f'f{k}s{s}': ([p('v2bench_test', args.arm, s, k)], [p('v2bench_test', args.arm, s, k, True)],
                                    [p('v2bench_test', 'baseline', s, k)], [p('v2bench_test', 'baseline', s, k, True)])
                      for s in SEEDS for k in FOLDS}
        missing = [f for v in test_files.values() for l in v for f in l if not os.path.exists(f)]
        if missing:
            raise SystemExit(f'missing 2026 dumps: {missing[:3]} ... ({len(missing)})')
        test_names = load_cm(test_files['f0s42'][2][0])[0]
        test_groups = {n: i for i, g in enumerate(groups_of({n: all_scenes[n] for n in test_names})) for n in g}
        out['2026'] = evaluate_set('2026 (units: the 15 models)', test_files, test_groups, args.n_boot, rng)

    # AutoICE test: per model, by service
    print('\n=== AutoICE test (20 scenes; 8 DMI, 12 CIS), mean ± sd over 15 models ===')
    a4a = {}
    for svc in ('all', 'dmi', 'cis'):
        rows = {'base': [], 'arm': []}
        for arm, key in ((args.arm, 'arm'), ('baseline', 'base')):
            for s in SEEDS:
                for k in FOLDS:
                    names, cm = load_cm(os.path.join(C, 'a4a', f'v2bench_{arm}_s{s}_f{k}_test.npz'))
                    idx = [i for i, n in enumerate(names) if svc == 'all' or svc in n]
                    m = metrics(cm[idx].sum(0))
                    rows[key].append((m['miou'], m['weighted_f1']))
        b, a = np.array(rows['base']), np.array(rows['arm'])
        d = a - b
        a4a[svc] = {'base_mean': b.mean(0).tolist(), 'arm_mean': a.mean(0).tolist(), 'diff_mean': d.mean(0).tolist(),
                    'diff_t95_miou': t_interval(d[:, 0]), 'diff_t95_wf1': t_interval(d[:, 1])}
        print('  %-4s mIoU base %.4f arm %.4f diff %+.4f [%+.4f, %+.4f] | wF1 base %.3f arm %.3f diff %+.3f' % (
            svc, b[:, 0].mean(), a[:, 0].mean(), d[:, 0].mean(), *t_interval(d[:, 0]), b[:, 1].mean(), a[:, 1].mean(), d[:, 1].mean()))
    out['autoice'] = a4a

    # the decision (DECISIONS.md item 6, as read in the module docstring)
    for setname in ('folds', '2026'):
        if setname not in out:
            continue
        dc, dm = out[setname]['diff']['comp'], out[setname]['diff']['miou']
        better = dc['mean'] > 0 and dc['boot95'][0] > 0 and dm['mean'] >= -MARGIN
        out[setname]['verdict_better'] = bool(better)
        print(f'\n{setname}: composition {dc["mean"]:+.4f} (bootstrap [{dc["boot95"][0]:+.4f}, {dc["boot95"][1]:+.4f}]), '
              f'mIoU {dm["mean"]:+.4f} -> {"BETTER than the baseline" if better else "not shown to be better"} by the rule')
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)
        print('->', args.json)


if __name__ == '__main__':
    main()
