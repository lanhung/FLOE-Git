"""Paper 2, DECISIONS.md item 14: LaTeX rows of the ablation tables, printed from the result files.

    python tools/analysis/paper2_ablation_table.py \
        --comp-vs-base paper2_compare.json --mt-vs-base paper2_compare_multitask.json \
        --comp-vs-mt paper2_compare_comp_vs_mt.json --figdata paper2_figdata.json \
        [--ensembles paper2_ensembles.json]

Prints (i) the scores of the three models (levels), (ii) the three paired comparisons with their
intervals and the verdict of the rule of item 6, (iii) AutoICE by service, (iv) the runs of the
ablation arm, (v) the ensembles. Nothing is typed by hand.
"""
import argparse
import json

import numpy as np

MEAS = (('comp', 'Composition score'), ('miou', 'Mean IoU'), ('miou_ice', 'IoU, ice classes'),
        ('hard', 'IoU, small+medium'), ('wf1', 'Weighted F1'))
NAME = {'baseline': 'Baseline', 'multitask': 'Decoders only', 'multitask_comp': 'Combined'}


def ms(v, scale=1.0, nd=3):
    v = np.asarray(v, dtype=float) * scale
    sd = v.std(ddof=1)
    sdd = nd + 1 if round(sd, nd) == 0 else nd          # as in the paper's table: 0.0004, not 0.000
    return f'${v.mean():.{nd}f}\\pm{sd:.{sdd}f}$'


def sgn(x, nd=4):
    return f'{x:+.{nd}f}'


def interval(iv, scale=1.0, nd=4):
    return f'$[{iv[0] * scale:+.{nd}f}, {iv[1] * scale:+.{nd}f}]$'


def unit_scores(cmp_, key, side):
    return [u[side] for u in cmp_[key]['per_unit']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--comp-vs-base', required=True)
    ap.add_argument('--mt-vs-base', required=True)
    ap.add_argument('--comp-vs-mt', required=True)
    ap.add_argument('--figdata', required=True)
    ap.add_argument('--ensembles', default=None)
    args = ap.parse_args()
    cb, mb, cm = (json.load(open(f)) for f in (args.comp_vs_base, args.mt_vs_base, args.comp_vs_mt))
    fd = json.load(open(args.figdata))
    assert cb['arm'] == 'multitask_comp' and mb['arm'] == 'multitask' and cm['arm'] == 'multitask_comp'
    assert cm.get('base') == 'multitask', 'third file must be multitask_comp against multitask (--base multitask)'

    # (i) levels: baseline and combined from the primary file, decoders only from its own file
    print('%% (i) levels: test set & model & mean IoU & IoU ice & small+medium & weighted F1 (%) & composition')
    src = {'baseline': (cb, 'base'), 'multitask': (mb, 'arm'), 'multitask_comp': (cb, 'arm')}
    for key, label in (('folds', 'Held-out folds'), ('2026', '2026')):
        for i, arm in enumerate(('baseline', 'multitask', 'multitask_comp')):
            c, side = src[arm]
            u = unit_scores(c, key, side)
            cells = [ms([x[m] for x in u], 100.0 if m == 'wf1' else 1.0, 1 if m == 'wf1' else 3)
                     for m in ('miou', 'miou_ice', 'hard', 'wf1', 'comp')]
            print(f"{label if i == 0 else '':15s} & {NAME[arm]} & " + ' & '.join(cells) + ' \\\\')
    # consistency: baseline units of the two files vs baseline must be identical
    for key in ('folds', '2026'):
        a = [x['miou'] for x in unit_scores(cb, key, 'base')]
        b = [x['miou'] for x in unit_scores(mb, key, 'base')]
        assert np.allclose(a, b), f'baseline units differ between files on {key}'
        a = [x['miou'] for x in unit_scores(cb, key, 'arm')]
        b = [x['miou'] for x in unit_scores(cm, key, 'arm')]
        assert np.allclose(a, b), f'combined units differ between files on {key}'
        a = [x['miou'] for x in unit_scores(mb, key, 'arm')]
        b = [x['miou'] for x in unit_scores(cm, key, 'base')]
        assert np.allclose(a, b), f'decoders-only units differ between files on {key}'
    print('%% consistency: the same runs enter every file (checked)')

    # (ii) paired comparisons
    print('\n%% (ii) paired differences: comparison & measure & folds: diff, seeds, groups & 2026: diff, models, groups')
    for title, c in (('Decoders only $-$ baseline', mb), ('Combined $-$ baseline', cb),
                     ('Combined $-$ decoders only', cm)):
        print(f'\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\')
        for m, mname in MEAS:
            sc, nd = (100.0, 2) if m == 'wf1' else (1.0, 4)
            row = [mname]
            for key in ('folds', '2026'):
                d = c[key]['diff'][m]
                row += [f'${d["mean"] * sc:+.{nd}f}$', interval(d['t95'], sc, nd), interval(d['boot95'], sc, nd)]
            print(' & '.join(row) + ' \\\\')
        for key in ('folds', '2026'):
            pooled = {m: c[key]['pooled']['arm'][m] - c[key]['pooled']['base'][m] for m, _ in MEAS}
            print(f"%%   {title} {key}: pooled differences " + ', '.join(f'{m} {v:+.4f}' for m, v in pooled.items())
                  + f"; verdict of the rule: {'BETTER' if c[key]['verdict_better'] else 'not shown to be better'}")
    wins = {}
    for title, c in (('mt>base', mb), ('comp>base', cb), ('comp>mt', cm)):
        for key in ('folds', '2026'):
            wins[(title, key)] = {m: int(sum(u['diff'][m] > 0 for u in c[key]['per_unit']))
                                  for m in ('miou', 'hard', 'comp')}
    print('%% pairs with a positive difference (of 3 seeds on folds, of 15 models on 2026):')
    for k, v in wins.items():
        print(f'%%   {k[0]:10s} {k[1]:5s} {v}')

    # (iii) AutoICE
    print('\n%% (iii) AutoICE (mean +- sd over 15 models): model & all wF1 & all mIoU & DMI wF1 & DMI mIoU & CIS wF1 & CIS mIoU')
    for arm in ('baseline', 'multitask', 'multitask_comp'):
        cells = []
        for svc in ('all', 'dmi', 'cis'):
            a = fd['autoice'][svc][arm]
            cells += [f"${100 * a['wf1'][0]:.1f}\\pm{100 * a['wf1'][1]:.1f}$", f"${a['miou'][0]:.3f}\\pm{a['miou'][1]:.3f}$"]
        print(f'{NAME[arm]} & ' + ' & '.join(cells) + ' \\\\')
    for title, c in (('Decoders only - baseline', mb), ('Combined - baseline', cb), ('Combined - decoders only', cm)):
        for svc in ('all', 'dmi', 'cis'):
            a = c['autoice'][svc]
            print(f"%%   {title:26s} {svc}: mIoU {a['diff_mean'][0]:+.3f} [{a['diff_t95_miou'][0]:+.3f}, {a['diff_t95_miou'][1]:+.3f}]"
                  f", wF1 {100 * a['diff_mean'][1]:+.1f} [{100 * a['diff_t95_wf1'][0]:+.1f}, {100 * a['diff_t95_wf1'][1]:+.1f}]")

    # (iv) runs of the ablation arm
    print('\n%% (iv) decoders-only runs: seed & fold & stop epoch & fold mIoU & 2026 mIoU & 2026 small+medium & 2026 composition & AutoICE wF1')
    run = {(r['arm'], r['seed'], r['fold']): r for r in fd['runs']}
    stops = []
    for s in (42, 43, 44):
        for k in range(5):
            r = run[('multitask', s, k)]
            stops.append(r['stop_epoch'])
            print(f"{s} & {k} & {r['stop_epoch']} & {r['folds']['miou']:.3f} & {r['2026']['miou']:.3f} & "
                  f"{r['2026']['hard']:.3f} & {r['2026']['comp']:.3f} & {100 * r['autoice']['all']['wf1']:.1f} \\\\")
        if s != 44:
            print('\\middlehline')
    print(f'%% stop epochs {min(stops)} to {max(stops)}, mean {np.mean(stops):.0f}')
    comp_ice = {a: np.mean([r['2026']['comp_ice'] for r in fd['runs'] if r['arm'] == a]) for a in NAME}
    print('%% 2026 ice-only composition, mean over 15 models:', {a: round(float(v), 4) for a, v in comp_ice.items()})
    if 'mixed' in fd:
        for name in ('folds', '2026'):
            m = fd['mixed'][name]
            print(f'%% mixed polygons {name}: predicted water share',
                  {a: round(100 * m[a]['pred'][0], 1) for a in NAME if a in m}, 'chart',
                  round(100 * m['baseline']['chart'][0], 1), '| ice-mix TV distance',
                  {a: round(m[a]['tv_ice_mix'], 3) for a in NAME if a in m})
    for key in ('folds', '2026'):
        print(f'%% 2026/folds per-class IoU, pooled ({key}):')
        for arm in NAME:
            if arm in fd['confusion'][key]:
                print(f"%%   {arm:15s}", [round(x, 3) for x in fd['confusion'][key][arm]['iou']])

    # (v) ensembles
    if args.ensembles:
        en = json.load(open(args.ensembles))
        print('\n%% (v) ensembles of 15 models vs the mean single model: model & 2026 mIoU & ice & small+medium & wF1 & '
              'composition & AutoICE wF1 & AutoICE mIoU')
        for arm in ('baseline', 'multitask', 'multitask_comp'):
            e = en.get(arm, {})
            if '2026' not in e:
                print(f'%% {arm}: no ensemble result')
                continue
            t, a = e['2026'], e.get('autoice', {}).get('all')
            single = [r for r in fd['runs'] if r['arm'] == arm]
            sm = {m: np.mean([r['2026'][m] for r in single]) for m in ('miou', 'ice', 'hard', 'wf1', 'comp')}
            print(f"{NAME[arm]} & {t['miou']:.3f} & {t['ice']:.3f} & {t['hard']:.3f} & {100 * t['wf1']:.1f} & {t['comp']:.3f} & "
                  + (f"{100 * a['wf1']:.1f} & {a['miou']:.3f}" if a else '-- & --') + ' \\\\')
            print(f"%%   mean single model: " + ', '.join(f'{m} {v:.4f}' for m, v in sm.items())
                  + f"; ensemble gain mIoU {t['miou'] - sm['miou']:+.4f}, composition {t['comp'] - sm['comp']:+.4f}")


if __name__ == '__main__':
    main()
