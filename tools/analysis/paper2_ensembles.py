"""Paper 2, DECISIONS.md item 14b: the ensemble of each arm's 15 benchmark models (logits averaged).

One number per arm and test set, no interval (secondary). Reads the dumps of
`ARM=<arm> bash tools/system/eval_bench.sh ensemble test2026|autoice GPU`.

    python tools/analysis/paper2_ensembles.py --json paper2_ensembles.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from floe.composition import composition_confusion, confusion_scores  # noqa: E402
from tools.analysis.paired_cv import metrics  # noqa: E402

ARMS = ('baseline', 'multitask', 'multitask_comp')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cm-dir', default=os.environ.get('FLOE_CM_DIR'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    C, out = args.cm_dir, {}
    for arm in ARMS:
        r = {}
        p = os.path.join(C, 'v2bench_test', f'ens_{arm}.npz')
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            m = metrics(z['cm'].astype(np.float64).sum(0))
            d = np.load(os.path.join(C, 'poly', 'v2bench_test', f'ens_{arm}.npz'), allow_pickle=True)
            c = composition_confusion(d['hist'], d['q'], d['u'])
            r['2026'] = {'scenes': int(len(z['scenes'])), 'miou': float(m['miou']), 'ice': float(np.mean(m['iou'][1:])),
                         'hard': float(np.mean(m['iou'][2:4])), 'wf1': float(m['weighted_f1']),
                         'comp': float(confusion_scores(c['full'])['miou']),
                         'comp_ice': float(confusion_scores(c['ice'])['miou']), 'iou': [float(x) for x in m['iou']]}
        p = os.path.join(C, 'a4a', f'v2bench_ens_{arm}_test.npz')
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            names, cm = [str(x) for x in z['scenes']], z['cm'].astype(np.float64)
            r['autoice'] = {}
            for svc in ('all', 'dmi', 'cis'):
                idx = [i for i, n in enumerate(names) if svc == 'all' or svc in n]
                m = metrics(cm[idx].sum(0))
                r['autoice'][svc] = {'miou': float(m['miou']), 'wf1': float(m['weighted_f1'])}
        out[arm] = r
        if '2026' in r:
            e = r['2026']
            print(f"{arm:15s} 2026: mIoU {e['miou']:.4f} ice {e['ice']:.4f} small+medium {e['hard']:.4f} "
                  f"wF1 {100 * e['wf1']:.2f} composition {e['comp']:.4f} (ice {e['comp_ice']:.4f})")
        if 'autoice' in r:
            a = r['autoice']
            print(f"{'':15s} AutoICE wF1 all {100 * a['all']['wf1']:.1f} DMI {100 * a['dmi']['wf1']:.1f} "
                  f"CIS {100 * a['cis']['wf1']:.1f}; mIoU all {a['all']['miou']:.3f}")
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)
        print('->', args.json)


if __name__ == '__main__':
    main()
