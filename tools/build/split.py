"""Grouped K-fold split of the rebuilt archive (tools.build.run_archive output).

Frames are not independent: consecutive frames of one pass share a chart, and the same
pack is imaged again on the next days and charted again. Frames are therefore joined
into groups (union-find) when they share a datatake, share a chart, or have a chart of
the same DMI region in the same ISO week, and whole groups are dealt to the folds.
Groups never span two ISO weeks, so a fold boundary is at worst a week boundary.

Groups are assigned greedily, largest first, to the fold whose floe-size pixel totals
are furthest below their 1/K share (sum over classes of the shortfall, weighted by the
inverse class frequency so rare classes count as much as common ones).

    python -m tools.build.split --log $FLOE_ROOT/data_v2/archive_log.jsonl \
        --data-dir $FLOE_ROOT/data_v2 --out datalists/v2 --folds 5

Writes fold<K>_train.json / fold<K>_val.json (file names, as the loaders expect) and
split_report.md with per-fold class shares and the group size distribution.
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

import numpy as np

NAMES = ('water', 'cake', 'small', 'medium', 'big', 'vast')


class _Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def join(self, i, j):
        self.p[self.find(i)] = self.find(j)


def load_scenes(log_path, data_dir):
    """Latest 'ok' record per product whose scene file exists."""
    rec = {}
    for line in open(log_path):
        r = json.loads(line)
        if r['status'] == 'ok':
            rec[r['product']] = r
        else:
            rec.pop(r['product'], None)
    return {n: r for n, r in rec.items() if osp.exists(osp.join(data_dir, n + '.nc'))}


def chart_region(chart):
    return chart.split('_')[1].lower()          # 202301020755_NorthEast_RIC -> northeast


def groups_of(scenes):
    names = sorted(scenes)
    u = _Union(len(names))
    first = {}
    for i, n in enumerate(names):
        r = scenes[n]
        week = datetime.strptime(n.split('_')[4][:8], '%Y%m%d').isocalendar()[:2]
        keys = [('take', n.split('_')[0], n.split('_')[7])]          # mission + datatake id
        keys += [('chart', c) for c in r['charts']]
        keys += [('region-week', chart_region(c), week) for c in r['charts']]
        for k in keys:
            if k in first:
                u.join(i, first[k])
            else:
                first[k] = i
    out = defaultdict(list)
    for i, n in enumerate(names):
        out[u.find(i)].append(n)
    return list(out.values())


def assign(groups, scenes, k, seed=0):
    hist = {n: np.asarray(scenes[n]['floe_hist'][:6], np.float64) for n in scenes}
    gh = [sum(hist[n] for n in g) for g in groups]
    total = sum(gh)
    w = 1.0 / np.maximum(total, 1)                 # rare classes weigh as much as common ones
    rng = np.random.default_rng(seed)
    order = sorted(range(len(groups)), key=lambda i: (-(gh[i] * w).sum(), rng.random()))
    fold_h = np.zeros((k, 6))
    fold_n = np.zeros(k)
    fold_of = {}
    for i in order:
        short = np.maximum(total / k - fold_h, 0) * w          # remaining room per fold and class
        score = (np.minimum(short, gh[i] * w)).sum(1) - 1e-3 * fold_n / max(len(scenes), 1)
        f = int(np.argmax(score))
        fold_h[f] += gh[i]
        fold_n[f] += len(groups[i])
        for n in groups[i]:
            fold_of[n] = f
    return fold_of, fold_h, fold_n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--log', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--years', type=int, nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    scenes = load_scenes(args.log, args.data_dir)
    if args.years:
        scenes = {n: r for n, r in scenes.items() if int(n.split('_')[4][:4]) in args.years}
    groups = groups_of(scenes)
    fold_of, fold_h, fold_n = assign(groups, scenes, args.folds, args.seed)

    os.makedirs(args.out, exist_ok=True)
    for f in range(args.folds):
        val = sorted(n + '.nc' for n, g in fold_of.items() if g == f)
        train = sorted(n + '.nc' for n, g in fold_of.items() if g != f)
        json.dump(train, open(osp.join(args.out, f'fold{f}_train.json'), 'w'), indent=1)
        json.dump(val, open(osp.join(args.out, f'fold{f}_val.json'), 'w'), indent=1)

    sizes = sorted((len(g) for g in groups), reverse=True)
    share = fold_h / np.maximum(fold_h.sum(0, keepdims=True), 1)
    lines = [f'# Grouped {args.folds}-fold split of {osp.basename(osp.normpath(args.data_dir))}', '',
             f'{len(scenes)} scenes in {len(groups)} groups (largest {sizes[:5]}, '
             f'median {int(np.median(sizes))}); built by tools/build/split.py.', '',
             '| fold | scenes | ' + ' | '.join(f'{c} share' for c in NAMES) + ' | ice Mpx |',
             '|---|---|' + '---|' * len(NAMES) + '---|']
    for f in range(args.folds):
        lines.append(f'| {f} | {int(fold_n[f])} | ' + ' | '.join(f'{s:.2f}' for s in share[f])
                     + f' | {fold_h[f, 1:].sum() / 1e6:.0f} |')
    tot = fold_h.sum(0)
    lines += ['', 'Archive class totals (Mpx): ' + ', '.join(f'{c} {v / 1e6:.1f}' for c, v in zip(NAMES, tot))]
    open(osp.join(args.out, 'split_report.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
