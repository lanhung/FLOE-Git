#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 7 of the dataset paper ("Baseline scores", tab:baseline) from the benchmark runs.

Rows (paper_dataset/main.tex, Sect. "Baseline"):
  cv     FLOE, five folds   -> held-out fold   15 runs floe_baseline_v2benchf{K}_s{seed}
  test   FLOE, 2022-2025    -> 2026            the same 15 checkpoints on datalists/v2_bench/test.json
  a4a    FLOE, 2022-2025    -> AutoICE test    the same 15 checkpoints on the 20 AutoICE test scenes
  rev    AutoICE train      -> FLOE 2026       model(s) trained on the AutoICE training scenes (--rev)

Columns (Sect. "Scores"; every score is computed from a summed confusion matrix):
  weighted F1      F1 per class weighted by reference pixels (the AutoICE floe score), in %;
  mean IoU         mean IoU over the 6 classes against the dominant-class raster, icebergs and
                   masked pixels ignored; a class without reference pixels counts as 0 and is
                   reported (tools.analysis.paired_cv.metrics, the same as tools.evaluate);
  mean IoU, ice    mean of the 5 ice-class IoUs (cake, small, medium, big, vast) of the same CM;
  composition IoU  mean IoU of the 'full' composition confusion matrix of
                   floe.composition.composition_confusion: per chart polygon with n predicted
                   pixels and egg-code composition q, t = n q pixels are expected per class;
                   min(t_k, h_k) count as correct and the rest of t is spread over the rest of
                   the predicted histogram h in proportion; summed over polygons. Polygons with
                   < 64 valid pixels or > 0.5 unknown mass are left out (defaults of
                   composition_confusion, as in tools.analysis.composition_eval). Its mean is
                   over the classes with reference mass (floe.composition.confusion_scores).
                   The ice-mix variant ('ice' matrix) is written to the JSON as well.
Inputs are the per-scene CMs of `tools.evaluate --dump-cm` and the per-polygon dumps of
`--composition-dir ... --dump-polygons`. No composition exists for the AutoICE test scenes
(no chart sidecars), so that cell of row 'a4a' is n/a.

Aggregation:
  cv     for each seed, pool (sum) the per-scene CMs of the five validation folds and compute
         the scores (the composition CM likewise: summed over the polygons of the five folds);
         report mean and sample standard deviation (ddof = 1) over the three seeds.
  test, a4a   score each of the 15 models on the whole test set (CM summed over its scenes);
         report mean and sample sd over the 15, and the per-seed means over the 5 folds.
  rev    each listed model scored on 2026; mean and sd over models (sd n/a for one model).

Every expected file is checked first; a missing file stops the script with a list of what is
missing. --allow-partial computes from what exists instead: cells are marked PARTIAL (printed,
in the JSON, and wrapped in \\pending{} in the LaTeX), a cv seed is pooled over the folds it
has. Every CM must hold exactly the scenes of its list (fold val list, 2026 list, 20 AutoICE
scenes); a mismatch is an error even with --allow-partial.

Polygon dumps: the validation dumps written by launch_fold.sh were made with the tile gate of
tools.evaluate on (configs/FLOE/_sota_base.py val_tile_gate): tiles whose dominant-class labels
are all masked are not run and predict class 0, so masked (mixed) polygons are scored as water.
The composition score must come from dumps made without the gate (configs/V2/baseline_evalfull.py,
tools/system/eval_bench.sh). Those are read from $FLOE_CM_DIR/poly/v2bench_full (validation)
and poly/v2bench_test (2026). A heuristic check (masked-looking polygons predicted 100 % water)
warns if a dump looks gated.

    python -m tools.analysis.baseline_table --json $FLOE_ROOT/logs/baseline_table.json \
        --tex $FLOE_ROOT/logs/baseline_table.tex [--allow-partial]
    python -m tools.analysis.baseline_table --self-test
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

from floe.composition import composition_confusion, confusion_scores
from tools.analysis.composition_eval import load as load_poly
from tools.analysis.paired_cv import NAMES, load_cm, metrics, scene_key

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (42, 43, 44)
COLS = ('weighted_f1', 'miou', 'miou_ice', 'comp_miou')
ROWS = (
    ('cv', r'\dsname, five folds', 'held-out fold'),
    ('test', r'\dsname, 2022--2025', '2026'),
    ('a4a', r'\dsname, 2022--2025', 'AutoICE test'),
    ('rev', 'AutoICE train', r'\dsname{} 2026'),
)
ROW_TITLE = {'cv': 'FLOE 5 folds -> held-out fold', 'test': 'FLOE 2022-25 -> 2026',
             'a4a': 'FLOE 2022-25 -> AutoICE test', 'rev': 'AutoICE train -> FLOE 2026'}


class Missing(Exception):
    pass


# ---------------------------------------------------------------- inputs

def paths(cm_dir, run_prefix='baseline', val_poly_sub='v2bench_full'):
    """Expected input files per row. cv/test/a4a: {(seed, fold): (cm, poly or None)}."""
    p = {'cv': {}, 'test': {}, 'a4a': {}}
    for s in SEEDS:
        for k in FOLDS:
            tag = f'{run_prefix}_s{s}_f{k}'
            p['cv'][(s, k)] = (osp.join(cm_dir, 'v2bench', tag + '.npz'),
                               osp.join(cm_dir, 'poly', val_poly_sub, tag + '.npz'))
            p['test'][(s, k)] = (osp.join(cm_dir, 'v2bench_test', tag + '.npz'),
                                 osp.join(cm_dir, 'poly', 'v2bench_test', tag + '.npz'))
            p['a4a'][(s, k)] = (osp.join(cm_dir, 'a4a', f'v2bench_{tag}_test.npz'), None)
    return p


def read_list(path):
    return sorted(scene_key(s) for s in json.load(open(path)))


def load_checked(path, expected, what):
    scenes, cm = load_cm(path)
    if scenes != expected:
        only_cm = sorted(set(scenes) - set(expected))[:3]
        only_list = sorted(set(expected) - set(scenes))[:3]
        raise ValueError(f'{what}: {path} holds {len(scenes)} scenes, the list {len(expected)}; '
                         f'only in CM {only_cm}, only in list {only_list}')
    return cm.sum(0)


def composition_cms(poly_path, scenes):
    """Composition CMs ('full', 'ice') of one polygon dump, restricted to the given scenes."""
    d = load_poly(poly_path)
    extra = sorted(set(d['scene']) - set(scenes))
    if extra:
        raise ValueError(f'{poly_path}: polygons of {len(extra)} scenes not in the scored list, e.g. {extra[:3]}')
    c = composition_confusion(d['hist'], d['q'], d['u'])
    return {'full': c['full'], 'ice': c['ice'], 'polygons': c['polygons'],
            'scenes_with_polygons': int(len(set(d['scene']))), 'gated_suspect': gated_suspect(d)}


def gated_suspect(d, min_px=10000):
    """Share of large polygons whose chart says < 10 % water but whose prediction is all
    class 0: the signature of the tile gate (skipped tiles predict 0). Near 0 for an
    ungated dump; the gated validation dumps of Nov 2026 have many."""
    n = d['hist'].sum(1)
    q = d['q'] / np.clip(d['q'].sum(1, keepdims=True), 1e-9, None)
    big = (n >= min_px) & (q.sum(1) > 0)
    if not big.any():
        return 0.0
    s = big & (q[:, 0] < 0.1) & (d['hist'][:, 0] == n)
    return float(s.sum() / big.sum())


# ---------------------------------------------------------------- scores

def scores(cm, comp=None):
    m = metrics(cm)
    out = {'weighted_f1': 100.0 * m['weighted_f1'], 'miou': m['miou'],
           'miou_ice': float(np.mean(m['iou'][1:])), 'hard_ice': m['hard_ice'],
           'macro_f1': 100.0 * m['macro_f1'], 'oa': 100.0 * m['oa'], 'iou': m['iou'],
           'absent': [NAMES[i] for i in m['absent']]}
    if comp is not None:
        f, i = confusion_scores(comp['full']), confusion_scores(comp['ice'])
        out.update({'comp_miou': f['miou'], 'comp_weighted_f1': 100.0 * f['f1_weighted'],
                    'comp_iou': f['iou'].tolist(), 'comp_ice_miou': i['miou'],
                    'comp_ice_weighted_f1': 100.0 * i['f1_weighted'], 'comp_polygons': comp['polygons']})
    return out


def mean_sd(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None), int(len(v))


def summarise(units, keys):
    """units: list of score dicts -> {key: {mean, sd, n}}."""
    out = {}
    for k in keys:
        m, s, n = mean_sd([u.get(k) for u in units])
        out[k] = {'mean': m, 'sd': s, 'n': n}
    return out


SUMMARY_KEYS = COLS + ('hard_ice', 'macro_f1', 'oa', 'comp_weighted_f1', 'comp_ice_miou', 'comp_ice_weighted_f1')


# ---------------------------------------------------------------- rows

def build(cm_dir, split_dir, a4a_list, rev_names, allow_partial, say=print, val_poly_sub='v2bench_full'):
    p = paths(cm_dir, val_poly_sub=val_poly_sub)
    lists = {k: read_list(osp.join(split_dir, f'fold{k}_val.json')) for k in FOLDS}
    test_list = read_list(osp.join(split_dir, 'test.json'))
    a4a_scenes = read_list(a4a_list)
    seen = set()
    for k in FOLDS:
        if seen & set(lists[k]):
            raise ValueError(f'fold {k} validation scenes overlap another fold')
        seen |= set(lists[k])
    if seen & set(test_list):
        raise ValueError('2026 test scenes overlap the validation folds')

    missing, warnings = [], []
    res = {'rows': {}, 'missing': missing, 'warnings': warnings, 'partial': False}

    def have(path, what):
        if path is None:
            return False
        if not osp.exists(path):
            missing.append(f'{what}: {path}')
            return False
        return True

    def warn(msg):
        warnings.append(msg)
        say('[warn] ' + msg)

    def check_scores(s, what):
        if s['absent']:
            warn(f'{what}: no reference pixels of {s["absent"]} (IoU counted as 0 in the mean)')
        return s

    # -- row cv: pool the five folds per seed
    per_seed, cv_units = {}, []
    for s in SEEDS:
        cm_tot, comp_tot, folds_cm, folds_comp = 0.0, None, [], []
        for k in FOLDS:
            cm_p, poly_p = p['cv'][(s, k)]
            if not have(cm_p, f'cv seed {s} fold {k} CM'):
                continue
            cm_tot = cm_tot + load_checked(cm_p, lists[k], f'cv s{s} f{k}')
            folds_cm.append(k)
            if have(poly_p, f'cv seed {s} fold {k} polygons (ungated)'):
                c = composition_cms(poly_p, lists[k])
                if c['gated_suspect'] > 0.01:
                    warn(f'{poly_p}: {100 * c["gated_suspect"]:.1f} % of large polygons look tile-gated '
                         f'(predicted all water, chart < 10 % water)')
                comp_tot = c if comp_tot is None else {
                    'full': comp_tot['full'] + c['full'], 'ice': comp_tot['ice'] + c['ice'],
                    'polygons': comp_tot['polygons'] + c['polygons']}
                folds_comp.append(k)
        if not folds_cm:
            continue
        sc = check_scores(scores(cm_tot, comp_tot), f'cv seed {s}')
        if folds_comp != folds_cm:   # composition only if it covers the same folds
            sc.pop('comp_miou', None)
        sc.update({'seed': s, 'folds_cm': folds_cm, 'folds_comp': folds_comp,
                   'partial': len(folds_cm) < len(FOLDS) or len(folds_comp) < len(FOLDS)})
        per_seed[s] = sc
        cv_units.append(sc)
    res['rows']['cv'] = {'units': per_seed, 'summary': summarise(cv_units, SUMMARY_KEYS),
                         'partial': len(cv_units) < len(SEEDS) or any(u['partial'] for u in cv_units),
                         'aggregation': 'per seed: CMs of the five folds summed; mean and sd (ddof=1) over seeds'}

    # -- rows test and a4a: 15 models scored on the whole test set
    for row, scenes in (('test', test_list), ('a4a', a4a_scenes)):
        units = {}
        for s in SEEDS:
            for k in FOLDS:
                cm_p, poly_p = p[row][(s, k)]
                if not have(cm_p, f'{row} seed {s} fold {k} CM'):
                    continue
                comp = None
                if poly_p is not None and have(poly_p, f'{row} seed {s} fold {k} polygons'):
                    comp = composition_cms(poly_p, scenes)
                    if comp['gated_suspect'] > 0.01:
                        warn(f'{poly_p}: {100 * comp["gated_suspect"]:.1f} % of large polygons look tile-gated')
                sc = check_scores(scores(load_checked(cm_p, scenes, f'{row} s{s} f{k}'), comp), f'{row} s{s} f{k}')
                sc.update({'seed': s, 'fold': k})
                units[f's{s}_f{k}'] = sc
        lst = list(units.values())
        res['rows'][row] = {
            'units': units, 'summary': summarise(lst, SUMMARY_KEYS),
            'per_seed': {s: summarise([u for u in lst if u['seed'] == s], COLS) for s in SEEDS},
            'partial': len(lst) < len(SEEDS) * len(FOLDS),
            'aggregation': 'each of the 15 models scored on all test scenes; mean and sd (ddof=1) over models'}
        if row == 'test':
            comp_n = sum('comp_miou' in u for u in lst)
            if lst and comp_n < len(lst):
                res['rows'][row]['partial'] = True

    # -- row rev: AutoICE-trained model(s) on 2026
    units = {}
    for name in rev_names:
        cm_p = osp.join(cm_dir, 'v2bench_test', name + '.npz')
        poly_p = osp.join(cm_dir, 'poly', 'v2bench_test', name + '.npz')
        if not have(cm_p, f'rev {name} CM'):
            continue
        comp = composition_cms(poly_p, test_list) if have(poly_p, f'rev {name} polygons') else None
        units[name] = check_scores(scores(load_checked(cm_p, test_list, f'rev {name}'), comp), f'rev {name}')
    res['rows']['rev'] = {'units': units, 'summary': summarise(list(units.values()), SUMMARY_KEYS),
                          'partial': (len(units) < len(rev_names)
                                      or any('comp_miou' not in u for u in units.values())),
                          'aggregation': 'each listed model scored on all 2026 scenes; mean (and sd if > 1 model)'}

    res['partial'] = bool(missing) or any(r['partial'] for r in res['rows'].values())
    if missing and not allow_partial:
        raise Missing('\n  '.join([f'{len(missing)} input file(s) missing:'] + missing))
    return res


# ---------------------------------------------------------------- output

def fmt(entry, col, tex=False):
    m, s = entry['mean'], entry['sd']
    if m is None:
        return '--' if tex else 'n/a'
    d = 1 if col == 'weighted_f1' else 3
    pm = r' $\pm$ ' if tex else ' ± '
    return f'{m:.{d}f}' + (f'{pm}{s:.{d}f}' if s is not None else '')


def print_table(res, say=print):
    say('\n=== Table 7: baseline scores (mean ± sd; weighted F1 in %) ===')
    if res['partial']:
        say('*** PARTIAL: inputs missing, cells marked * are not the protocol numbers ***')
    say(f'{"row":32s} {"wF1":>13s} {"mIoU":>15s} {"mIoU ice":>15s} {"comp. IoU":>15s}  n')
    for key, _, _ in ROWS:
        r = res['rows'][key]
        sm = r['summary']
        mark = '*' if r['partial'] else ' '
        say(f'{ROW_TITLE[key]:31s}{mark} ' + ' '.join(f'{fmt(sm[c], c):>15s}' for c in COLS)
            + f'  {sm["miou"]["n"]}')
    for key in ('test', 'a4a'):
        r = res['rows'][key]
        for s, sm in r['per_seed'].items():
            if sm['miou']['n']:
                say(f'  {key} seed {s} ({sm["miou"]["n"]} folds): ' + '  '.join(
                    f'{c} {fmt(sm[c], c)}' for c in COLS))
    for s, u in res['rows']['cv']['units'].items():
        say(f'  cv seed {s}: folds {u["folds_cm"]} (composition {u["folds_comp"]})  '
            + '  '.join(f'{c} {u[c]:.4f}' for c in COLS if c in u))


def latex(res):
    lines = ['% Table 7 rows, written by tools.analysis.baseline_table',
             '% mean $\\pm$ sd: five folds -> over 3 seeds (folds pooled per seed); 2026 and AutoICE -> over 15 models']
    if res['partial']:
        lines.append('% PARTIAL: some inputs are missing; not for the paper (cells in \\pending{})')
    for key, trained, tested in ROWS:
        r = res['rows'][key]
        cells = []
        for c in COLS:
            v = fmt(r['summary'][c], c, tex=True)
            cells.append(rf'\pending{{{v}}}' if (r['partial'] or v == '--') and not
                         (key == 'a4a' and c == 'comp_miou') else v)
        lines.append(f'{trained} & {tested} & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(lines) + '\n'


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------- self-test

def self_test(tmp):
    """Synthetic run tree: checks pooling per seed, mean/sd over seeds and models, the
    composition column, the missing-file and scene checks, and the LaTeX fragment."""
    rng = np.random.default_rng(3)
    cm_dir, split = osp.join(tmp, 'cm'), osp.join(tmp, 'split')
    for d in ('v2bench', 'v2bench_test', 'a4a', 'poly/v2bench_full', 'poly/v2bench_test'):
        os.makedirs(osp.join(cm_dir, d), exist_ok=True)
    os.makedirs(split)
    val = {k: [f'S1A_f{k}_{i:02d}.nc' for i in range(8)] for k in FOLDS}
    test = [f'S1A_2026_{i:02d}.nc' for i in range(10)]
    a4a = [f'A4A_{i:02d}.nc' for i in range(4)]
    for k in FOLDS:
        json.dump(val[k], open(osp.join(split, f'fold{k}_val.json'), 'w'))
    json.dump(test, open(osp.join(split, 'test.json'), 'w'))
    json.dump(a4a, open(osp.join(tmp, 'a4a_test.json'), 'w'))

    def cms(n):
        c = rng.integers(0, 40, size=(n, 6, 6)).astype(np.float64)
        c[:, np.arange(6), np.arange(6)] += 300
        return c

    def poly(scenes, pure):
        """Pure one-hot polygons with a histogram that agrees on 90 % of the pixels."""
        rows = []
        for sc in scenes:
            for j in range(3):
                q = np.zeros(6)
                if pure:
                    q[(j + len(rows)) % 6] = 1.0
                else:
                    q[[1, 2]] = 0.5
                h = np.zeros(6, np.int64)
                h[np.argmax(q)] = 900
                h[(np.argmax(q) + 1) % 6] = 100
                rows.append((sc[:-3], q, h))
        return dict(scene=np.array([r[0] for r in rows]), polygon=np.arange(len(rows)),
                    hist=np.array([r[2] for r in rows]), q=np.array([r[1] for r in rows], np.float32),
                    u=np.zeros(len(rows), np.float32), checkpoint=np.array('x'))

    p = paths(cm_dir)
    store = {}
    for s in SEEDS:
        for k in FOLDS:
            c = cms(8)
            store[(s, k)] = c
            np.savez(p['cv'][(s, k)][0], scenes=np.array(val[k]), cm=c)
            np.savez(p['cv'][(s, k)][1], **poly(val[k], pure=True))
            np.savez(p['test'][(s, k)][0], scenes=np.array(test), cm=cms(10))
            np.savez(p['test'][(s, k)][1], **poly(test, pure=False))
            np.savez(p['a4a'][(s, k)][0], scenes=np.array(a4a), cm=cms(4))
    np.savez(osp.join(cm_dir, 'v2bench_test', 'rev.npz'), scenes=np.array(test), cm=cms(10))
    quiet = lambda *a, **k: None
    a4a_list = osp.join(tmp, 'a4a_test.json')

    res = build(cm_dir, split, a4a_list, ['rev'], allow_partial=True, say=quiet)
    # (1) pooling per seed = metrics of the summed fold CMs; mean/sd over the three seeds
    per = [metrics(sum(store[(s, k)].sum(0) for k in FOLDS))['miou'] for s in SEEDS]
    sm = res['rows']['cv']['summary']['miou']
    assert abs(sm['mean'] - np.mean(per)) < 1e-12 and abs(sm['sd'] - np.std(per, ddof=1)) < 1e-12, sm
    assert res['rows']['test']['summary']['miou']['n'] == 15
    # (2) composition on pure polygons equals the pixel score: 90 % agreement on each
    #     one-hot polygon -> with all classes used equally, IoU = 900 / (900 + 100 + 100)
    cv_comp = res['rows']['cv']['summary']['comp_miou']['mean']
    assert abs(cv_comp - 900 / 1100) < 1e-9, cv_comp
    #     mixed 50/50 cake/small polygon predicted 90 % cake, 10 % small: all of small
    #     agrees and the surplus cake goes to small -> cake IoU 500/900, small 100/500
    t_comp = res['rows']['test']['summary']['comp_miou']['mean']
    assert abs(t_comp - (500 / 900 + 100 / 500) / 2) < 1e-9, t_comp
    # (3) the only gap: rev has no polygons -> missing reported, row marked partial
    assert any('rev rev polygons' in m for m in res['missing']), res['missing']
    assert res['partial'] and res['rows']['rev']['partial'] and not res['rows']['cv']['partial']
    try:
        build(cm_dir, split, a4a_list, ['rev'], allow_partial=False, say=quiet)
        raise AssertionError('missing input not reported')
    except Missing:
        pass
    # (4) missing fold -> partial cv; wrong scene set -> error even with --allow-partial
    os.remove(p['cv'][(44, 3)][0])
    r2 = build(cm_dir, split, a4a_list, ['rev'], allow_partial=True, say=quiet)
    assert r2['rows']['cv']['units'][44]['folds_cm'] == [0, 1, 2, 4] and r2['rows']['cv']['partial']
    np.savez(p['cv'][(44, 3)][0], scenes=np.array(val[2]), cm=cms(8))
    try:
        build(cm_dir, split, a4a_list, ['rev'], allow_partial=True, say=quiet)
        raise AssertionError('wrong scene set not rejected')
    except ValueError:
        pass
    # (5) LaTeX: four rows, partial cells in \pending, AutoICE composition '--'
    tex = latex(res)
    rows = [l for l in tex.splitlines() if l.endswith(r'\\')]
    assert len(rows) == 4 and r'\pending' in rows[3] and rows[2].rstrip(r' \\').endswith('--'), tex
    print('[self-test] PASS baseline_table: seed pooling, mean/sd, composition column, '
          'missing and scene checks, LaTeX rows')


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cm-dir', default=os.environ.get('FLOE_CM_DIR'))
    ap.add_argument('--split', default='datalists/v2_bench', help='fold{K}_val.json and test.json')
    ap.add_argument('--a4a-list', default='datalists/ai4arctic/test.json')
    ap.add_argument('--rev', nargs='*', default=['a4a_v0map_s42'],
                    help='AutoICE-trained models scored on 2026: $FLOE_CM_DIR/v2bench_test/<name>.npz '
                         '(and poly/v2bench_test/<name>.npz); see tools/system/queues/v2_bench_test2026.txt')
    ap.add_argument('--val-poly-sub', default='v2bench_full',
                    help='validation polygon dumps under $FLOE_CM_DIR/poly/<this>; v2bench holds the gated '
                         'dumps of launch_fold.sh (for checks only, see the module docstring)')
    ap.add_argument('--allow-partial', action='store_true',
                    help='compute from the inputs that exist; affected cells are marked PARTIAL')
    ap.add_argument('--json', default=None)
    ap.add_argument('--tex', default=None)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self_test(d)
        return 0
    if not args.cm_dir:
        sys.exit('--cm-dir or $FLOE_CM_DIR needed')
    try:
        res = build(args.cm_dir, args.split, args.a4a_list, args.rev, args.allow_partial,
                    val_poly_sub=args.val_poly_sub)
    except Missing as e:
        print(f'[error] {e}\n(use --allow-partial to compute from what exists, marked PARTIAL)')
        return 2
    if res['missing']:
        print(f'[partial] {len(res["missing"])} input file(s) missing:')
        for m in res['missing']:
            print('  ' + m)
    print_table(res)
    tex = latex(res)
    print('\n' + tex)
    res['inputs'] = {'val_poly_sub': args.val_poly_sub, 'cm_dir': args.cm_dir, 'split': args.split, 'a4a_list': args.a4a_list, 'rev': args.rev}
    if args.json:
        json.dump(to_json(res), open(args.json, 'w'), indent=1)
        print(f'-> {args.json}')
    if args.tex:
        open(args.tex, 'w').write(tex)
        print(f'-> {args.tex}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
