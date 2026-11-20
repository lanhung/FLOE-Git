"""Paper 2: LaTeX rows of the appendix tables, printed from the result files (no hand-typed numbers).

    python tools/analysis/paper2_tables.py --figdata paper2_figdata.json --breakdown paper2_breakdown.json
"""
import argparse
import json

import numpy as np

REGIONS = ('Qaanaaq', 'NorthWest', 'CentralWest', 'SouthWest', 'CapeFarewell', 'SouthEast', 'CentralEast',
           'NorthEast', 'North')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--figdata', required=True)
    ap.add_argument('--breakdown', required=True)
    args = ap.parse_args()
    fd, bd = json.load(open(args.figdata)), json.load(open(args.breakdown))
    base, arm = fd['arms']
    run = {(r['arm'], r['seed'], r['fold']): r for r in fd['runs']}
    print('%% all runs: seed, fold | stop epoch B C | fold mIoU B C | 2026 mIoU B C | 2026 small+medium B C | '
          '2026 composition B C | AutoICE wF1 B C')
    for s in (42, 43, 44):
        for k in range(5):
            a, b = run[(base, s, k)], run[(arm, s, k)]
            print(f"{s} & {k} & {a['stop_epoch']} & {b['stop_epoch']} & {a['folds']['miou']:.3f} & {b['folds']['miou']:.3f} & "
                  f"{a['2026']['miou']:.3f} & {b['2026']['miou']:.3f} & {a['2026']['hard']:.3f} & {b['2026']['hard']:.3f} & "
                  f"{a['2026']['comp']:.3f} & {b['2026']['comp']:.3f} & {100 * a['autoice']['all']['wf1']:.1f} & "
                  f"{100 * b['autoice']['all']['wf1']:.1f} \\\\")
        if s != 44:
            print('\\middlehline')
    wins = {k: sum(run[(arm, s, f)]['2026'][k] > run[(base, s, f)]['2026'][k] for s in (42, 43, 44) for f in range(5))
            for k in ('miou', 'hard', 'comp')}
    print('%% pairs (of 15) in which the combined model is higher on 2026:', wins)
    fw = {k: sum(run[(arm, s, f)]['folds'][k] > run[(base, s, f)]['folds'][k] for s in (42, 43, 44) for f in range(5))
          for k in ('miou', 'hard', 'comp')}
    print('%% pairs (of 15) in which the combined model is higher on its held-out fold:', fw)
    print('\n%% region and season: scenes folds | hard B C | scenes 2026 | hard B C')
    for group, names in (('region', REGIONS), ('season', ('DJF', 'MAM', 'JJA', 'SON'))):
        for n in names:
            f, t = bd['folds'][group][n], bd['2026'][group][n]
            print(f"{n} & {f['scenes']} & {f['base']:.3f} & {f['arm']:.3f} & {f['arm'] - f['base']:+.3f} & "
                  f"{t['scenes']} & {t['base']:.3f} & {t['arm']:.3f} & {t['arm'] - t['base']:+.3f} \\\\")
        if group == 'region':
            print('\\middlehline')
    print('\n%% AutoICE by service (mean ± sd over 15 models): wF1 %, mIoU')
    for svc in ('all', 'dmi', 'cis'):
        a, b = fd['autoice'][svc][base], fd['autoice'][svc][arm]
        print(f"{svc}: wF1 {100 * a['wf1'][0]:.1f} ± {100 * a['wf1'][1]:.1f} -> {100 * b['wf1'][0]:.1f} ± {100 * b['wf1'][1]:.1f}; "
              f"mIoU {a['miou'][0]:.3f} ± {a['miou'][1]:.3f} -> {b['miou'][0]:.3f} ± {b['miou'][1]:.3f}")
    print('\n%% mixed polygons, class shares (water cake small medium big vast), %')
    for name in ('folds', '2026'):
        m = fd['mixed'][name]
        print(name, 'chart', [round(100 * x, 1) for x in m[base]['chart']], '| baseline', [round(100 * x, 1) for x in m[base]['pred']],
              '| combined', [round(100 * x, 1) for x in m[arm]['pred']], '| polygons/run %.0f, Mpx/run %.0f' % (m[base]['polygons_per_run'], m[base]['mpx_per_run']))
    cm = {a: np.array(fd['confusion']['2026'][a]['cm']) for a in (base, arm)}
    r = {a: 100 * c / c.sum(1, keepdims=True) for a, c in cm.items()}
    print('\n%% 2026 confusion, row-normalised %, diagonal baseline -> combined:',
          [f"{r[base][i, i]:.1f}->{r[arm][i, i]:.1f}" for i in range(6)])
    print('%% small->cake %.1f->%.1f, medium->big %.1f->%.1f, big->vast %.1f->%.1f, medium->vast %.1f->%.1f' % (
        r[base][2, 1], r[arm][2, 1], r[base][3, 4], r[arm][3, 4], r[base][4, 5], r[arm][4, 5], r[base][3, 5], r[arm][3, 5]))
    sd = {a: np.std([x['2026']['miou'] for x in fd['runs'] if x['arm'] == a], ddof=1) for a in (base, arm)}
    cs = {a: np.std([x['2026']['comp'] for x in fd['runs'] if x['arm'] == a], ddof=1) for a in (base, arm)}
    print('%% sd over 15 models on 2026: mIoU', {a: round(float(v), 5) for a, v in sd.items()}, 'composition',
          {a: round(float(v), 5) for a, v in cs.items()})


if __name__ == '__main__':
    main()
