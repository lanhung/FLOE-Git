#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check a (possibly still running) data transfer against the scene lists.

Read-only. Reports present / missing / unexpected scenes and partial files (temporaries,
tiny files, or files still growing with --watch-seconds); with --open N (-1 = all) also
opens N files with h5py and checks the variables and z-score attributes training reads.

  python -m tools.data.check_transfer --data-dir DIR [--lists ...] [--open 20] [--watch-seconds 30]
"""
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import sys
import time

REQUIRED_VARS = [
    'nersc_sar_primary', 'nersc_sar_secondary', 'sar_incidenceangle',
    'glcm_sigma0_hh_contrast', 'glcm_sigma0_hh_dissimilarity',
    'glcm_sigma0_hh_homogeneity', 'global_valid_mask', 'FLOE',
]
ZSCORE_VARS = ['nersc_sar_primary', 'nersc_sar_secondary']
TEMP_PATTERNS = ('.part', '.tmp', '.filepart', '~')


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024:
            return f'{n:.1f}{unit}'
        n /= 1024
    return f'{n:.1f}PB'


def expected_scenes(list_paths):
    names = set()
    used = []
    for pattern in list_paths:
        for p in sorted(glob.glob(pattern)):
            with open(p) as f:
                names |= set(json.load(f))
            used.append(p)
    return names, used


def scan(data_dir):
    nc, temps = {}, []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            full = osp.join(root, fn)
            if fn.endswith('.nc'):
                try:
                    nc[fn] = (full, os.path.getsize(full))
                except OSError:
                    temps.append(full)
            elif fn.startswith('.') or fn.endswith(TEMP_PATTERNS) or '.nc.' in fn:
                temps.append(full)
    return nc, temps


def check_file(path):
    import h5py
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    out = {'ok': True, 'problems': []}
    try:
        with h5py.File(path, 'r') as h:
            shapes = {}
            for v in REQUIRED_VARS:
                if v not in h:
                    out['problems'].append(f'missing variable {v}')
                    continue
                shapes[v] = h[v].shape
            if len(set(shapes.values())) > 1:
                out['problems'].append(f'variables on different grids: {shapes}')
            if shapes:
                out['shape'] = list(next(iter(shapes.values())))
            for v in ZSCORE_VARS:
                if v in h:
                    a = h[v].attrs
                    if 'zscore_mean' not in a or 'zscore_std' not in a:
                        out['problems'].append(f'{v}: no zscore_mean/zscore_std attrs')
                    else:
                        out.setdefault('zscore', {})[v] = [float(a['zscore_mean']),
                                                           float(a['zscore_std'])]
    except Exception as e:  # truncated files raise here
        out['problems'].append(f'cannot open: {type(e).__name__}: {e}')
    out['ok'] = not out['problems']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--lists', nargs='+',
                    default=['datalists/train_417.json', 'datalists/val_417.json',
                             'datalists/cv5/fold*_train.json', 'datalists/cv5/fold*_val.json'])
    ap.add_argument('--open', type=int, default=0,
                    help='open N present files with h5py (-1 = all)')
    ap.add_argument('--watch-seconds', type=float, default=0,
                    help='look twice, this far apart, to find files still being written')
    ap.add_argument('--tile-index', default=os.environ.get('FLOE_TILE_INDEX', '/root/autodl-tmp/floe_tile_index_417.npz'))
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    if not osp.isdir(args.data_dir):
        sys.exit(f'not a directory (or no permission): {args.data_dir}')
    expected, used = expected_scenes(args.lists)
    print(f'lists: {len(used)} files, {len(expected)} distinct scenes')

    nc, temps = scan(args.data_dir)
    present = set(nc) & expected
    missing = sorted(expected - set(nc))
    extra = sorted(set(nc) - expected)
    sizes = sorted(s for _, s in nc.values())
    median = sizes[len(sizes) // 2] if sizes else 0
    small = sorted(n for n, (_, s) in nc.items() if s == 0 or (median and s < 0.2 * median))

    growing = []
    if args.watch_seconds > 0 and nc:
        time.sleep(args.watch_seconds)
        for n, (p, s) in nc.items():
            try:
                if os.path.getsize(p) != s:
                    growing.append(n)
            except OSError:
                growing.append(n)

    total = sum(sizes)
    print(f'present {len(present)}/{len(expected)} expected scenes | missing {len(missing)} | '
          f'unexpected .nc {len(extra)} | temp/partial files {len(temps)}')
    print(f'.nc total {human(total)}; median {human(median)}; '
          f'{len(small)} suspiciously small; {len(growing)} still growing')
    du = shutil.disk_usage(args.data_dir)
    print(f'filesystem free {human(du.free)} of {human(du.total)}')
    if missing and present:
        est = median * len(missing)
        print(f'estimated still to arrive: ~{human(est)}'
              + ('  ⚠️ more than the free space' if est > du.free else ''))

    for title, items in (('missing', missing), ('unexpected', extra), ('small', small),
                         ('growing', growing), ('temp', temps)):
        if items:
            print(f'\n{title} ({len(items)}):')
            for it in items[:15]:
                print('   ', it)
            if len(items) > 15:
                print(f'    ... {len(items) - 15} more')

    opened = {}
    if args.open:
        try:
            import h5py  # noqa: F401
        except ImportError:
            sys.exit('--open needs h5py')
        cand = sorted(n for n in present if n not in small and n not in growing)
        if args.open > 0:
            cand = cand[:args.open]
        bad = 0
        zs = set()
        for n in cand:
            r = check_file(nc[n][0])
            opened[n] = r
            if not r['ok']:
                bad += 1
                print(f'  ✗ {n}: {"; ".join(r["problems"])}')
            for v, ms in r.get('zscore', {}).items():
                zs.add((v, round(ms[0], 4), round(ms[1], 4)))
        print(f'\nopened {len(cand)} files: {len(cand) - bad} ok, {bad} with problems')
        if zs:
            print('z-score attributes seen (variable, mean, std):')
            for z in sorted(zs):
                print('   ', z)
            if len({z[0] for z in zs}) != len(zs):
                print('  ⚠️ different statistics across files: normalisation is not global')

    for label, path in (('tile index', args.tile_index),
                        ('zscore_stats.json', osp.join(args.data_dir, 'zscore_stats.json'))):
        print(f'{label}: {"found" if osp.exists(path) else "NOT found"} ({path})')

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump({'expected': len(expected), 'present': sorted(present), 'missing': missing,
                       'extra': extra, 'small': small, 'growing': growing, 'temps': temps,
                       'total_bytes': total, 'free_bytes': du.free, 'opened': opened}, f, indent=2)
        print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
