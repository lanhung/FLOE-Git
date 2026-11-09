#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired K-fold comparison of two models from per-scene confusion matrices.

Inputs: one `tools.evaluate --dump-cm` .npz per (model, fold) with keys `scenes` and `cm`
(S, C, C; rows = true); both models must cover the same scenes in each fold. Reports per-fold
metrics (hard-ice = mean IoU of small and medium) and B - A, the across-fold mean with a t-based
95% CI (df = K-1), and a paired bootstrap CI on the pooled CM, resampled within folds. Without
--groups, neighbouring scenes are resampled independently and the bootstrap CI is optimistic.

  python -m tools.analysis.paired_cv --a 'cm/cv5/baseline_f{K}.npz' --b 'cm/cv5/pgsg_f{K}.npz' \
      --labels baseline pgsg [--groups groups.json] [--json-out out.json]
  python -m tools.analysis.paired_cv --self-test
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

NAMES = ['water', 'cake', 'small', 'medium', 'big', 'vast']
HARD = (2, 3)
# Two-sided 95% t quantiles, df = 1..30 (avoids a scipy dependency).
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    return _T975[df - 1] if 1 <= df <= len(_T975) else 1.96


def scene_key(name):
    s = str(name)
    return s[:-3] if s.endswith('.nc') else s


def metrics(cm):
    """All metrics from one (C, C) confusion matrix. Absent classes (no GT and no
    prediction) get IoU 0 and count in the macro mean, matching the paper tables;
    `absent` lists them so that the caller can warn."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt = cm.sum(1)
    pr = cm.sum(0)
    union = gt + pr - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    f1d = gt + pr
    f1 = np.divide(2 * tp, f1d, out=np.zeros_like(tp), where=f1d > 0)
    tot = cm.sum()
    return {
        'miou': float(iou.mean()),
        'hard_ice': float(iou[list(HARD)].mean()),
        'iou': iou.tolist(),
        'macro_f1': float(f1.mean()),
        'weighted_f1': float((f1 * gt).sum() / tot) if tot > 0 else float('nan'),
        'oa': float(tp.sum() / tot) if tot > 0 else float('nan'),
        'absent': [int(i) for i in np.where(gt == 0)[0]],
    }


def load_cm(path):
    d = np.load(path, allow_pickle=True)
    scenes = [scene_key(s) for s in d['scenes']]
    cm = np.asarray(d['cm'], dtype=np.float64)
    assert cm.ndim == 3 and cm.shape[0] == len(scenes), f'{path}: bad shapes'
    order = np.argsort(scenes)
    return [scenes[i] for i in order], cm[order]


def load_pair(path_a, path_b):
    sa, ca = load_cm(path_a)
    sb, cb = load_cm(path_b)
    if sa != sb:
        only_a = sorted(set(sa) - set(sb))[:5]
        only_b = sorted(set(sb) - set(sa))[:5]
        raise ValueError(f'unpaired scenes: {path_a} vs {path_b}; '
                         f'only A {only_a} only B {only_b}')
    for s, a_, b_ in zip(sa, ca, cb):
        if a_.sum(1).tolist() != b_.sum(1).tolist():
            raise ValueError(f'{s}: GT row sums differ between A and B (different labels/masks?)')
    return sa, ca, cb


SCALARS = ('miou', 'hard_ice', 'macro_f1', 'weighted_f1', 'oa')


def diff_vector(ma, mb):
    return np.array([mb[k] - ma[k] for k in SCALARS] + list(np.subtract(mb['iou'], ma['iou'])))


def bootstrap(folds, groups, n_boot, seed):
    """folds: list of (scenes, cm_a, cm_b). Paired, stratified by fold."""
    rng = np.random.default_rng(seed)
    units = []   # per fold: list of index arrays (one per resampling unit)
    for scenes, _, _ in folds:
        if groups is None:
            units.append([np.array([i]) for i in range(len(scenes))])
        else:
            by = {}
            for i, s in enumerate(scenes):
                if s not in groups:
                    raise KeyError(f'scene {s} missing from --groups')
                by.setdefault(groups[s], []).append(i)
            units.append([np.array(v) for v in by.values()])
    out = np.empty((n_boot, len(SCALARS) + len(NAMES)))
    for r in range(n_boot):
        tot_a = 0.0
        tot_b = 0.0
        for (scenes, ca, cb), u in zip(folds, units):
            pick = rng.integers(0, len(u), size=len(u))
            idx = np.concatenate([u[j] for j in pick])
            tot_a = tot_a + ca[idx].sum(0)
            tot_b = tot_b + cb[idx].sum(0)
        out[r] = diff_vector(metrics(tot_a), metrics(tot_b))
    return out


def run(paths_a, paths_b, fold_ids, labels, groups=None, n_boot=10000, seed=0, quiet=False):
    say = (lambda *a, **k: None) if quiet else print
    la, lb = labels
    folds, per_fold = [], []
    seen = set()
    for k, pa, pb in zip(fold_ids, paths_a, paths_b):
        scenes, ca, cb = load_pair(pa, pb)
        dup = seen & set(scenes)
        if dup:
            raise ValueError(f'fold {k}: scenes already validated in another fold: {sorted(dup)[:5]}')
        seen |= set(scenes)
        ma, mb = metrics(ca.sum(0)), metrics(cb.sum(0))
        if ma['absent']:
            say(f'[WARN] fold {k}: classes with no GT pixels {[NAMES[i] for i in ma["absent"]]} '
                f'(IoU counted as 0 in macro mIoU)')
        folds.append((scenes, ca, cb))
        per_fold.append({'fold': k, 'n_scenes': len(scenes), la: ma, lb: mb,
                         'diff': diff_vector(ma, mb).tolist()})

    keys = list(SCALARS) + [f'iou_{n}' for n in NAMES]
    D = np.array([f['diff'] for f in per_fold])          # (K, M)
    K = len(per_fold)
    mean = D.mean(0)
    se = D.std(0, ddof=1) / np.sqrt(K) if K > 1 else np.full(D.shape[1], np.nan)
    tq = t975(K - 1) if K > 1 else np.nan
    wins = (D > 0).sum(0)

    pa_cm = sum(c[1].sum(0) for c in folds)
    pb_cm = sum(c[2].sum(0) for c in folds)
    pooled_a, pooled_b = metrics(pa_cm), metrics(pb_cm)
    boot = bootstrap(folds, groups, n_boot, seed) if n_boot > 0 else None

    say(f'\n=== per fold ({lb} - {la}) ===')
    say('%-5s %4s | %8s %8s %8s | %8s %8s %8s' % ('fold', 'n', f'{la[:8]}', f'{lb[:8]}', 'Δmiou',
                                                   'hardA', 'hardB', 'Δhard'))
    for f in per_fold:
        a, b = f[la], f[lb]
        say('%-5s %4d | %8.4f %8.4f %+8.4f | %8.4f %8.4f %+8.4f' % (
            f['fold'], f['n_scenes'], a['miou'], b['miou'], b['miou'] - a['miou'],
            a['hard_ice'], b['hard_ice'], b['hard_ice'] - a['hard_ice']))

    say(f'\n=== across folds (K={K}): mean ± SE, t-95% CI, folds with {lb} > {la} ===')
    summary = {}
    for j, key in enumerate(keys):
        ci = (mean[j] - tq * se[j], mean[j] + tq * se[j])
        entry = {'mean_diff': float(mean[j]), 'se': float(se[j]),
                 't95_ci': [float(ci[0]), float(ci[1])], 'folds_better': int(wins[j])}
        if boot is not None:
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            entry['pooled_a'] = (pooled_a[key] if key in pooled_a
                                 else pooled_a['iou'][NAMES.index(key[4:])])
            entry['pooled_b'] = (pooled_b[key] if key in pooled_b
                                 else pooled_b['iou'][NAMES.index(key[4:])])
            entry['boot95_ci'] = [float(lo), float(hi)]
            entry['p_boot_gt0'] = float((boot[:, j] > 0).mean())
        summary[key] = entry
        say('%-12s %+8.4f ± %.4f  CI [%+.4f, %+.4f]  %d/%d%s' % (
            key, mean[j], se[j], ci[0], ci[1], wins[j], K,
            ('  | pooled %.4f→%.4f boot CI [%+.4f, %+.4f] P(>0)=%.3f' % (
                entry['pooled_a'], entry['pooled_b'], *entry['boot95_ci'], entry['p_boot_gt0'])
             if boot is not None else '')))
    if boot is not None:
        unit = 'groups' if groups is not None else 'scenes (no --groups: CI is optimistic)'
        say(f'\nbootstrap: {n_boot} paired replicates, resampling {unit} within each fold')
    return {'labels': labels, 'folds': per_fold, 'summary': summary,
            'pooled': {la: pooled_a, lb: pooled_b},
            'bootstrap': {'n_boot': n_boot, 'unit': 'group' if groups else 'scene', 'seed': seed}}


def self_test(tmpdir):
    """Synthetic check: B = A with extra correct pixels in class 'big' on every fold
    ⇒ Δ big IoU > 0 in all folds, Δ for untouched classes = 0, CI excludes 0."""
    rng = np.random.default_rng(1)
    paths_a, paths_b = [], []
    for k in range(5):
        S = 20
        cm_a = rng.integers(0, 50, size=(S, 6, 6)).astype(np.float64)
        cm_a[:, np.arange(6), np.arange(6)] += 400
        cm_b = cm_a.copy()
        # move 30 px per scene from big->vast confusion to the big diagonal
        move = np.minimum(cm_b[:, 4, 5], 30)
        cm_b[:, 4, 5] -= move
        cm_b[:, 4, 4] += move
        scenes = np.array([f'S1A_f{k}_s{i:02d}' for i in range(S)])
        for tag, cm, lst in (('a', cm_a, paths_a), ('b', cm_b, paths_b)):
            p = osp.join(tmpdir, f'{tag}_f{k}.npz')
            np.savez(p, scenes=scenes, cm=cm)
            lst.append(p)
    res = run(paths_a, paths_b, list(range(5)), ('A', 'B'), n_boot=500, quiet=True)
    sm = res['summary']
    assert sm['iou_big']['folds_better'] == 5 and sm['iou_big']['t95_ci'][0] > 0, sm['iou_big']
    assert sm['iou_big']['boot95_ci'][0] > 0, sm['iou_big']
    for n in ('water', 'cake', 'small', 'medium'):
        assert abs(sm[f'iou_{n}']['mean_diff']) < 1e-12, n
    assert sm['miou']['mean_diff'] > 0
    # grouped bootstrap path
    groups = {f'S1A_f{k}_s{i:02d}': f'g{k}_{i // 2}' for k in range(5) for i in range(20)}
    res_g = run(paths_a, paths_b, list(range(5)), ('A', 'B'), groups=groups, n_boot=200, quiet=True)
    assert res_g['bootstrap']['unit'] == 'group'
    # pairing violations must be rejected
    np.savez(osp.join(tmpdir, 'bad.npz'), scenes=np.array(['x']), cm=np.ones((1, 6, 6)))
    try:
        run(paths_a[:1], [osp.join(tmpdir, 'bad.npz')], [0], ('A', 'B'), n_boot=0, quiet=True)
        raise AssertionError('unpaired input not rejected')
    except ValueError:
        pass
    try:
        run(paths_a[:2], paths_b[:1] + paths_b[:1], [0, 1], ('A', 'B'), n_boot=0, quiet=True)
        raise AssertionError('mismatched fold pairing not rejected')
    except ValueError:
        pass
    print('[self-test] PASS paired_cv: per-fold diffs, t-CI, paired bootstrap (scene and '
          'group units), pairing checks')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--a', help="model A npz path template with {K}, e.g. 'cm/base_f{K}.npz'")
    p.add_argument('--b', help='model B npz path template with {K}')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--labels', nargs=2, default=['A', 'B'])
    p.add_argument('--groups', default=None,
                   help='JSON {scene_name: group_id} for group-level bootstrap')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json-out', default=None)
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self_test(d)
        return
    assert args.a and args.b and '{K}' in args.a and '{K}' in args.b, \
        '--a/--b must be path templates containing {K}'
    paths_a = [args.a.replace('{K}', str(k)) for k in args.folds]
    paths_b = [args.b.replace('{K}', str(k)) for k in args.folds]
    groups = None
    if args.groups:
        groups = {scene_key(k): v for k, v in json.load(open(args.groups)).items()}
    res = run(paths_a, paths_b, args.folds, tuple(args.labels), groups=groups,
              n_boot=args.n_boot, seed=args.seed)
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
    sys.exit(0)
