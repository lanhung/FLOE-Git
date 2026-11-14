"""What a model gains on validation scenes whose neighbours were in its training set.

Two models of one configuration: one trained on the grouped fold, one on its leaking
variant (tools.build.leak_split), in which the training set holds scenes of the same
pass, chart or region and week as part of the validation scenes. Both are scored on
the same scenes from the per-scene confusion matrices of tools.evaluate:

  exposed   scenes with a neighbour in the leaking training set;
  control   scenes without one. A difference here comes from the changed training
            set (and from run-to-run noise), not from leakage.

Intervals: paired bootstrap over the groups of the grouped split, since scenes of one
group are not independent.

With one dump per seed (--grouped and --leaking take several files, paired in order), the
per-scene matrices are summed over seeds, and a further section estimates the part of the
gain that leakage explains: (exposed gain) - (control gain). Its interval resamples the
groups of both roles jointly (each scene keeps its role), and the same difference is given
per seed with a t interval over seeds, since the group bootstrap does not see seed noise.

    python -m tools.analysis.leak_effect \
        --grouped $FLOE_CM_DIR/v2/baseline_s{42,43,44}_f0.npz \
        --leaking $FLOE_CM_DIR/v2leak/baseline_s{42,43,44}_f0.npz \
        --roles datalists/v2_2023_leak/leak_fold0.json --json leak_effect_3seeds.json

    python -m tools.analysis.leak_effect --grouped $FLOE_CM_DIR/v2/baseline_s42_f0.npz \
        --leaking $FLOE_CM_DIR/v2leak/baseline_s42_f0.npz --roles datalists/v2_2023_leak/leak_fold0.json
"""
import argparse
import json

import numpy as np

from tools.analysis.paired_cv import NAMES, load_cm, metrics

KEYS = ('miou', 'weighted_f1', 'oa')


def row(m):
    return [m[k] for k in KEYS] + list(m['iou'])


# Two-sided 95 % t quantiles for df = 1..10, for the interval over seeds.
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]


def pooled(paths):
    """Per-scene confusion matrices summed over the dumps of several seeds."""
    dumps = [dict(zip(*load_cm(p))) for p in paths]
    names = set(dumps[0])
    for p, d in zip(paths[1:], dumps[1:]):
        if set(d) != names:
            raise SystemExit(f'{p}: scenes differ from {paths[0]}')
    return {s: sum(d[s] for d in dumps) for s in names}, dumps


def excess(a, b, role, idx):
    """(gain on exposed) - (gain on control) for the scenes idx; None if a role is empty."""
    gains = []
    for r in ('exposed', 'control'):
        sel = idx[role[idx] == r]
        if not len(sel):
            return None
        gains.append(np.subtract(row(metrics(b[sel].sum(0))), row(metrics(a[sel].sum(0)))))
    return gains[0] - gains[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--grouped', required=True, nargs='+',
                    help='CM dump(s) of the model trained on the grouped fold, one per seed')
    ap.add_argument('--leaking', required=True, nargs='+',
                    help='CM dump(s) of the model trained on the leaking variant, same seeds in the same order')
    ap.add_argument('--roles', required=True, help='leak_fold<K>.json of tools.build.leak_split')
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if len(args.grouped) != len(args.leaking):
        raise SystemExit('--grouped and --leaking need the same number of dumps (one per seed)')
    roles = json.load(open(args.roles))
    cm_g, seeds_g = pooled(args.grouped)
    cm_l, seeds_l = pooled(args.leaking)
    if len(args.grouped) > 1:
        print(f'{len(args.grouped)} seeds; per-scene matrices summed over seeds')
    rng = np.random.default_rng(args.seed)
    cols = KEYS + tuple(NAMES)
    out = {}
    for role in ('exposed', 'control'):
        scenes = roles[role]
        lost = [s for s in scenes if s not in cm_g or s not in cm_l]
        if lost:
            raise SystemExit(f'{role}: {len(lost)} scenes without a confusion matrix, e.g. {lost[:2]}')
        a = np.stack([cm_g[s] for s in scenes])
        b = np.stack([cm_l[s] for s in scenes])
        if not np.array_equal(a.sum(2), b.sum(2)):
            raise SystemExit(f'{role}: the two dumps do not hold the same labels')
        ma, mb = metrics(a.sum(0)), metrics(b.sum(0))
        if ma['absent']:
            print(f'[WARN] {role}: no pixels of {[NAMES[i] for i in ma["absent"]]}; their IoU counts as 0 in the mean')

        by = {}
        for i, s in enumerate(scenes):
            by.setdefault(roles['group'][s], []).append(i)
        units = [np.array(v) for v in by.values()]
        diff = np.empty((args.bootstrap, len(cols)))
        for r in range(args.bootstrap):
            idx = np.concatenate([units[j] for j in rng.integers(0, len(units), len(units))])
            diff[r] = np.subtract(row(metrics(b[idx].sum(0))), row(metrics(a[idx].sum(0))))
        lo, hi = np.percentile(diff, [2.5, 97.5], axis=0) if args.bootstrap else (None, None)

        print(f'\n== {role}: {len(scenes)} scenes in {len(units)} groups')
        print('%-12s %9s %9s %9s   %s' % ('', 'grouped', 'leaking', 'gain', '95% interval'))
        res = {}
        for j, c in enumerate(cols):
            va, vb = row(ma)[j], row(mb)[j]
            ci = [float(lo[j]), float(hi[j])] if args.bootstrap else None
            print('%-12s %9.4f %9.4f %+9.4f   %s' % (c, va, vb, vb - va, '[%+.4f, %+.4f]' % tuple(ci) if ci else ''))
            res[c] = {'grouped': va, 'leaking': vb, 'gain': vb - va, 'ci95': ci}
        out[role] = {'scenes': len(scenes), 'groups': len(units), 'scores': res}
    # The part of the gain that leakage explains: exposed gain minus control gain.
    scenes = list(roles['exposed']) + list(roles['control'])
    role = np.array(['exposed'] * len(roles['exposed']) + ['control'] * len(roles['control']))
    a = np.stack([cm_g[s] for s in scenes])
    b = np.stack([cm_l[s] for s in scenes])
    by = {}
    for i, s in enumerate(scenes):
        by.setdefault(roles['group'][s], []).append(i)
    units = [np.array(v) for v in by.values()]
    mixed = sum(len(set(role[u])) > 1 for u in units)
    point = excess(a, b, role, np.arange(len(scenes)))
    draws, empty = [], 0
    for _ in range(args.bootstrap):
        idx = np.concatenate([units[j] for j in rng.integers(0, len(units), len(units))])
        d = excess(a, b, role, idx)
        if d is None:
            empty += 1
        else:
            draws.append(d)
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0) if draws else (None, None)
    per_seed = []
    for dg, dl in zip(seeds_g, seeds_l):
        sa = np.stack([dg[s] for s in scenes])
        sb = np.stack([dl[s] for s in scenes])
        per_seed.append(excess(sa, sb, role, np.arange(len(scenes))))
    per_seed = np.array(per_seed)
    n = len(per_seed)
    if n > 1:
        half = _T975[n - 2] * per_seed.std(0, ddof=1) / np.sqrt(n)
        mean = per_seed.mean(0)
    print(f'\n== leakage-specific gain: (exposed gain) - (control gain); {len(units)} groups '
          f'resampled jointly ({mixed} hold both roles), {n} seed(s)')
    if empty:
        print(f'[WARN] {empty} of {args.bootstrap} resamples had no scene of one role and were dropped')
    hdr = '%-12s %9s   %-20s' % ('', 'excess', '95% group bootstrap')
    if n > 1:
        hdr += '   %-28s %s' % ('per seed', '95% t interval over seeds')
    print(hdr)
    res = {}
    for j, c in enumerate(cols):
        ci = [float(lo[j]), float(hi[j])] if draws else None
        line = '%-12s %+9.4f   %-20s' % (c, point[j], '[%+.4f, %+.4f]' % tuple(ci) if ci else '')
        r = {'excess': float(point[j]), 'ci95_groups': ci, 'per_seed': [float(x) for x in per_seed[:, j]]}
        if n > 1:
            tci = [float(mean[j] - half[j]), float(mean[j] + half[j])]
            line += '   %-28s [%+.4f, %+.4f]' % (' '.join('%+.4f' % x for x in per_seed[:, j]), *tci)
            r['ci95_seeds'] = tci
        print(line)
        res[c] = r
    out['excess'] = {'groups': len(units), 'mixed_groups': int(mixed), 'seeds': n,
                     'dropped_resamples': empty, 'scores': res}
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)
        print('->', args.json)


if __name__ == '__main__':
    main()
