#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recover the frozen K-fold split (datalists/cv5) from per-fold confusion-matrix dumps.

Fallback when the original fold files are missing: each `tools.evaluate --dump-cm` .npz
stores its fold's validation scenes under key `scenes`. Nothing is written unless the folds
are disjoint and cover the archive (default train_417 + val_417) and, with --reference,
the scene-name prefixes found in that file match the recovered --reference-fold exactly.

  python -m tools.data.folds_from_cm --cm 'cm/cv5/b2dlr_f{K}.npz' --out datalists/cv5 \
      [--reference-fold 0 --reference docs/logs/paper_radio_f0.log] [--dry-run]
"""
import argparse
import json
import os
import os.path as osp
import re
import sys

import numpy as np

SCENE_RE = re.compile(r'S1[ABC]_EW_GRDM_1SDH_[0-9T_]+')


def nc_name(s):
    s = str(s)
    return s if s.endswith('.nc') else s + '.nc'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cm', required=True, help="path template with {K}")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--archive', nargs='+',
                    default=['datalists/train_417.json', 'datalists/val_417.json'])
    ap.add_argument('--out', default='datalists/cv5')
    ap.add_argument('--reference', default=None,
                    help='text file whose scene-name prefixes are one fold\'s val set')
    ap.add_argument('--reference-fold', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    archive = set()
    for p in args.archive:
        archive |= set(json.load(open(p)))
    folds = {}
    for k in args.folds:
        path = args.cm.replace('{K}', str(k))
        folds[k] = sorted(nc_name(s) for s in np.load(path, allow_pickle=True)['scenes'])
        print(f'fold {k}: {len(folds[k])} val scenes  ({path})')

    ok = True
    seen = {}
    for k, scenes in folds.items():
        for s in scenes:
            if s in seen:
                print(f'✗ {s} in fold {seen[s]} and fold {k}')
                ok = False
            seen[s] = k
    union = set(seen)
    if union != archive:
        print(f'✗ union {len(union)} != archive {len(archive)}: '
              f'missing {sorted(archive - union)[:3]} extra {sorted(union - archive)[:3]}')
        ok = False

    if args.reference:
        ref = sorted(set(SCENE_RE.findall(open(args.reference, encoding='utf-8').read())))
        got = folds[args.reference_fold]
        n = min(len(r) for r in ref) if ref else 0
        ref_p = sorted(set(r[:n] for r in ref))
        got_p = sorted(set(g[:n] for g in got))
        if ref_p != got_p:
            print(f'✗ fold {args.reference_fold} differs from reference ({len(ref_p)} vs '
                  f'{len(got_p)} prefixes): only ref {sorted(set(ref_p) - set(got_p))[:3]} '
                  f'only recovered {sorted(set(got_p) - set(ref_p))[:3]}')
            ok = False
        else:
            print(f'✓ fold {args.reference_fold} matches the reference ({len(ref_p)} scenes)')

    if not ok:
        sys.exit('not writing fold files: checks failed')
    print('✓ folds disjoint and cover the archive')
    if args.dry_run:
        return
    os.makedirs(args.out, exist_ok=True)
    for k, val in folds.items():
        train = sorted(archive - set(val))
        for kind, lst in (('train', train), ('val', val)):
            p = osp.join(args.out, f'fold{k}_{kind}.json')
            if osp.exists(p):
                old = json.load(open(p))
                if sorted(old) != lst:
                    sys.exit(f'{p} exists with different content; refusing to overwrite')
                continue
            with open(p, 'w') as f:
                json.dump(lst, f, indent=1)
        print(f'fold {k}: wrote train {len(train)} / val {len(val)}')


if __name__ == '__main__':
    main()
