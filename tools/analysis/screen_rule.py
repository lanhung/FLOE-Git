"""Apply the paper 2 screening rule (paper_method/DECISIONS.md, item 10) on 2023 fold 0.

Written before the screening scores were read. For each model (baseline, multitask,
comp_aux, multitask_comp) and seed 42/43/44 it reads the gate-free dumps of
tools/system/eval_bench.sh screen (CM: $FLOE_CM_DIR/v2_full/<tag>_v2f0_s<seed>.npz, polygon
histograms: $FLOE_CM_DIR/poly/v2_full/<tag>_v2f0_s<seed>.npz) and computes
  (a) six-class mean IoU against the dominant-class labels (tools.analysis.paired_cv.metrics);
  (b) the full composition score (floe.composition.composition_confusion -> confusion_scores,
      'full', polygons >= 64 px with at most half unknown ice, as in Table 7).
The rule: the benchmark arm is multitask_comp if, against multitask, its mean (b) over the
three seeds is higher and its mean (a) is not lower by more than 0.005; otherwise multitask.
Also printed (not part of the rule): every model against the baseline, with per-seed values.

    python tools/analysis/screen_rule.py --json screen_rule.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from floe.composition import composition_confusion, confusion_scores  # noqa: E402
from tools.analysis.paired_cv import metrics  # noqa: E402

MODELS = ('baseline', 'multitask', 'comp_aux', 'multitask_comp')
SEEDS = (42, 43, 44)
MARGIN = 0.005


def scores(cm_dir, tag, seed):
    name = f'{tag}_v2f0_s{seed}.npz'
    cm_p, poly_p = os.path.join(cm_dir, 'v2_full', name), os.path.join(cm_dir, 'poly', 'v2_full', name)
    if not (os.path.exists(cm_p) and os.path.exists(poly_p)):
        return None
    z = np.load(cm_p, allow_pickle=True)
    miou = metrics(z['cm'].astype(np.float64).sum(0))['miou']
    d = np.load(poly_p, allow_pickle=True)
    comp = confusion_scores(composition_confusion(d['hist'], d['q'], d['u'])['full'])['miou']
    return float(miou), float(comp), [str(s) for s in z['scenes']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cm-dir', default=os.environ.get('FLOE_CM_DIR'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    res, missing, scene_sets = {}, [], set()
    for m in MODELS:
        res[m] = {}
        for s in SEEDS:
            r = scores(args.cm_dir, m, s)
            if r is None:
                missing.append(f'{m} s{s}')
                continue
            res[m][s] = {'miou': r[0], 'comp': r[1]}
            scene_sets.add(tuple(r[2]))
    if missing:
        raise SystemExit(f'missing screening dumps: {missing}')
    if len(scene_sets) != 1:
        raise SystemExit('the dumps do not cover the same validation scenes')

    mean = {m: {k: float(np.mean([res[m][s][k] for s in SEEDS])) for k in ('miou', 'comp')} for m in MODELS}
    print(f'2023 fold 0, {len(next(iter(scene_sets)))} validation scenes, seeds {SEEDS}')
    print('%-15s %-28s %-28s' % ('model', 'mIoU (per seed) mean', 'composition (per seed) mean'))
    for m in MODELS:
        print('%-15s %s %.4f   %s %.4f' % (
            m, ' '.join('%.4f' % res[m][s]['miou'] for s in SEEDS), mean[m]['miou'],
            ' '.join('%.4f' % res[m][s]['comp'] for s in SEEDS), mean[m]['comp']))
    print('\ndifferences to the baseline (mean over seeds):')
    for m in MODELS[1:]:
        print('  %-15s mIoU %+.4f   composition %+.4f' % (
            m, mean[m]['miou'] - mean['baseline']['miou'], mean[m]['comp'] - mean['baseline']['comp']))

    d_comp = mean['multitask_comp']['comp'] - mean['multitask']['comp']
    d_miou = mean['multitask_comp']['miou'] - mean['multitask']['miou']
    take = d_comp > 0 and d_miou >= -MARGIN
    arm = 'multitask_comp' if take else 'multitask'
    print(f'\nrule (multitask_comp vs multitask): composition {d_comp:+.4f} (must be > 0), '
          f'mIoU {d_miou:+.4f} (must be >= {-MARGIN})')
    print(f'-> benchmark arm: {arm}')
    if args.json:
        json.dump({'per_seed': res, 'mean': mean, 'rule': {'d_comp': d_comp, 'd_miou': d_miou,
                   'margin': MARGIN, 'arm': arm}}, open(args.json, 'w'), indent=1)
        print('->', args.json)


if __name__ == '__main__':
    main()
