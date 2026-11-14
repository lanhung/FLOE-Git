"""Score predictions against the chart's egg codes instead of the dominant class.

Input: polygon dumps of tools.evaluate (--composition-dir ... --dump-polygons x.npz),
i.e. per chart polygon the histogram of predicted classes and the chart composition.
A dominant-class label calls every pixel of a polygon the same class although the egg
code says it is a mix, so the ordinary pixel scores punish a model for mapping what the
chart describes. Three scores avoid that (floe.composition.composition_confusion):

  composition  all 6 classes against the egg-code composition, pixels inside a polygon
               arranged in the most favourable way (the chart does not place them);
  ice mix      the same for the floe-size mix of the ice alone, water share removed;
  pure         ordinary pixel scores on polygons that hold one class only (>= 0.9).

With several dumps the scenes are bootstrapped jointly, so the intervals of a
difference between two models are paired.

    python -m tools.analysis.composition_eval --poly $FLOE_CM_DIR/poly/v2/baseline_s42_f0.npz \
        $FLOE_CM_DIR/poly/v2/comp_aux_s42_f0.npz --labels baseline comp_aux
"""
import argparse
import json

import numpy as np

from floe.composition import composition_confusion, composition_metrics, confusion_scores

NAMES = ('water', 'cake', 'small', 'medium', 'big', 'vast')
VIEWS = (('composition', 'full', NAMES), ('ice mix', 'ice', NAMES[1:]), ('pure polygons', 'pure', NAMES))


def load(path):
    d = np.load(path, allow_pickle=False)
    return {'scene': d['scene'].astype(str), 'hist': d['hist'], 'q': d['q'], 'u': d['u']}


def per_scene(d, **kw):
    """Scene -> confusion matrices, so scenes can be resampled without redoing the polygons."""
    out = {}
    order = np.argsort(d['scene'], kind='stable')
    scenes, start = np.unique(d['scene'][order], return_index=True)
    for sc, idx in zip(scenes, np.split(order, start[1:])):
        out[sc] = composition_confusion(d['hist'][idx], d['q'][idx], d['u'][idx], **kw)
    return out


def total(cms, scenes, key):
    return sum(cms[s][key] for s in scenes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--poly', nargs='+', required=True, help='polygon dumps of tools.evaluate')
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--pure-threshold', type=float, default=0.9)
    ap.add_argument('--min-pixels', type=int, default=64)
    ap.add_argument('--bootstrap', type=int, default=1000, help='scene resamples for the 95 %% intervals (0 = off)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None, help='also write the scores here')
    args = ap.parse_args()
    labels = args.labels or [p.split('/')[-1].replace('.npz', '') for p in args.poly]
    assert len(labels) == len(args.poly)

    dumps = [load(p) for p in args.poly]
    cms = [per_scene(d, pure_threshold=args.pure_threshold, min_pixels=args.min_pixels) for d in dumps]
    scenes = sorted(set.intersection(*(set(c) for c in cms)))
    for lab, c in zip(labels, cms):
        if len(c) != len(scenes):
            print(f'[warn] {lab}: {len(c)} scenes, {len(scenes)} common to all dumps are scored')
    rng = np.random.default_rng(args.seed)
    draws = [rng.choice(len(scenes), len(scenes)) for _ in range(args.bootstrap)]

    result = {}
    for title, key, names in VIEWS:
        print(f'\n== {title} ({len(scenes)} scenes)')
        print(f'{"model":14s} {"mIoU":>6s} {"95% CI":>15s} {"wF1":>6s}  ' + ' '.join(f'{n:>7s}' for n in names))
        boot = []
        for lab, c in zip(labels, cms):
            s = confusion_scores(total(c, scenes, key))
            b = np.array([confusion_scores(sum(c[scenes[i]][key] for i in d))['miou'] for d in draws])
            boot.append(b)
            ci = f'[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]' if len(b) else ''
            print(f'{lab:14s} {s["miou"]:6.3f} {ci:>15s} {s["f1_weighted"]:6.3f}  '
                  + ' '.join(f'{v:7.3f}' for v in s['iou']))
            result.setdefault(lab, {})[key] = {'miou': s['miou'], 'f1_weighted': s['f1_weighted'],
                                               'iou': s['iou'].tolist(),
                                               'miou_ci': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else None}
        if len(labels) == 2 and args.bootstrap:
            d = boot[1] - boot[0]
            print(f'{labels[1]} - {labels[0]}: {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, '
                  f'{np.percentile(d, 97.5):+.4f}], P(>0) = {(d > 0).mean():.3f}')

    print('\n== polygon compositions')
    for lab, d in zip(labels, dumps):
        keep = np.isin(d['scene'], scenes)
        m = composition_metrics(d['hist'][keep], d['q'][keep], d['u'][keep],
                                pure_threshold=args.pure_threshold, min_pixels=args.min_pixels)
        print(f'{lab:14s} ' + '  '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in m.items()))
        result[lab]['polygons'] = m
    if args.json:
        json.dump(result, open(args.json, 'w'), indent=1)
        print(f'-> {args.json}')


if __name__ == '__main__':
    main()
