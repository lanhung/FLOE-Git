#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert AI4Arctic (AutoICE) ready-to-train scenes into FLOE scene files.

HH/HV/incidence angle go from AutoICE z-scores back to physical units (AutoICE toolkit
misc/global_meanstd.npy: HH -14.5083 / 5.6597 dB, HV -24.7012 / 4.7468 dB, incidence angle
33.9980 / 8.3223 deg), HH/HV are clipped to [-35, 5] dB and re-standardised with this
archive's statistics. With --stats autoice the AutoICE z-scores are kept as they are (no
clip), which is the space of the rebuilt archive (tools/build). FLOE classes 0-5 are kept;
6 (bergs) and invalid SAR become 255.
Output uses the 512x512 chunks, 2^-10 z-unit grid and codec of tools.data.recompress.

  python -m tools.data.convert_ai4arctic --src $A4A/rtt --dst $A4A/floe_format \
      --lists-out datalists/ai4arctic
"""
import argparse
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

from tools.data.recompress import codec_kwargs, quantise

A4A_STATS = {   # mean, std in physical units (AutoICE toolkit misc/global_meanstd.npy)
    'nersc_sar_primary': (-14.508255155222800, 5.659745919381868),
    'nersc_sar_secondary': (-24.701205147144236, 4.746759305258515),
    'sar_incidenceangle': (33.998017884565360, 8.322267212079046),
}
FLOE_STATS = {  # this archive (NC attributes zscore_mean / zscore_std)
    'nersc_sar_primary': (-13.90176742, 8.33489168),
    'nersc_sar_secondary': (-27.0105, 9.4454),
    'sar_incidenceangle': (34.2622, 7.7648),
}
DB_CLIP = (-35.0, 5.0)
VERSION = 'tools.data.convert_ai4arctic v1 (2026-11-10)'


def harmonise(z, var, native=False):
    if native:
        return z.astype(np.float32)
    ma, sa = A4A_STATS[var]
    mo, so = FLOE_STATS[var]
    phys = z.astype(np.float64) * sa + ma
    if var in ('nersc_sar_primary', 'nersc_sar_secondary'):
        phys = np.clip(phys, *DB_CLIP)
    return ((phys - mo) / so).astype(np.float32)


def convert_one(job):
    prep, ref, dst, q, codec, native = job
    stats = A4A_STATS if native else FLOE_STATS
    t0 = time.time()
    tmp = dst + '.tmp'
    with h5py.File(prep, 'r') as s:
        hh, hv, ia = (s[v][...] for v in ('nersc_sar_primary', 'nersc_sar_secondary',
                                          'sar_incidenceangle'))
        valid = (hh != 0) & (hv != 0) & (ia != 0) & np.isfinite(hh) & np.isfinite(hv) & np.isfinite(ia)
        if ref is not None:
            with h5py.File(ref, 'r') as r:
                floe = r['FLOE'][...]
        else:
            floe = s['FLOE'][...]
        root_attrs = {k: s.attrs[k] for k in ('scene_id', 'original_id', 'ice_service',
                                              'pixel_spacing') if k in s.attrs}
    floe = floe.astype(np.uint8)
    floe[floe == 6] = 255
    floe[~valid] = 255
    out = {}
    for var, z in (('nersc_sar_primary', hh), ('nersc_sar_secondary', hv), ('sar_incidenceangle', ia)):
        h = harmonise(z, var, native)
        h[~valid] = 0
        hq = quantise(h, q)
        hq[~valid] = 0
        out[var] = hq
    H, W = hh.shape
    kw = codec_kwargs(codec)
    with h5py.File(tmp, 'w', libver='earliest') as d:
        for k, v in root_attrs.items():
            d.attrs[k] = v
        d.attrs['source'] = 'AI4Arctic ready-to-train (doi 10.11583/DTU.21316608 / 21762830)'
        d.attrs['converted_by'] = VERSION
        d.attrs['harmonised_to'] = ('AutoICE global z-scores (unchanged)' if native else
                                    'FLOE archive z-scores (HH/HV clipped to [-35, 5] dB)')
        d.attrs['recompress_quant_step_z'] = q
        d.attrs['floe_label_mapping'] = 'AutoICE 0-5 unchanged; 6 (bergs) -> 255; invalid SAR -> 255'
        for name, n in (('y', H), ('x', W)):
            ds = d.create_dataset(name, data=np.arange(n, dtype=np.float32))
            ds.make_scale(name)
        for var, arr in out.items():
            dv = d.create_dataset(var, data=arr, chunks=(512, 512), **kw)
            dv.attrs['zscore_mean'] = np.array([stats[var][0]])
            dv.attrs['zscore_std'] = np.array([stats[var][1]])
            dv.attrs['autoice_mean'] = np.array([A4A_STATS[var][0]])
            dv.attrs['autoice_std'] = np.array([A4A_STATS[var][1]])
            dv.attrs['quantization_step_zscore'] = q
            dv.dims[0].attach_scale(d['y']); dv.dims[1].attach_scale(d['x'])
        for var, arr in (('global_valid_mask', valid.astype(np.uint8)), ('FLOE', floe)):
            dv = d.create_dataset(var, data=arr, chunks=(512, 512), **kw)
            dv.dims[0].attach_scale(d['y']); dv.dims[1].attach_scale(d['x'])
        d['FLOE'].attrs['chart_fill_value'] = np.array([255], dtype=np.uint8)
    # verify before renaming into place
    with h5py.File(tmp, 'r') as d:
        problems = []
        if not np.array_equal(d['FLOE'][...], floe):
            problems.append('FLOE not exact')
        if not np.array_equal(d['global_valid_mask'][...] > 0, valid):
            problems.append('mask not exact')
        for var, arr in out.items():
            if not np.array_equal(d[var][...], arr):
                problems.append(f'{var} not exact')
    if problems:
        os.remove(tmp)
        return prep, False, problems, time.time() - t0
    os.replace(tmp, dst)
    hist = np.bincount(floe[floe != 255], minlength=6)[:6]
    return prep, True, hist.tolist(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='extracted ready-to-train dir with train/ and test/')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--q-exp', type=int, default=-10)
    ap.add_argument('--codec', choices=['gzip', 'zstd'], default='zstd')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='convert only the first N scenes (testing)')
    ap.add_argument('--lists-out', default=None, help='write train/val/test datalists here')
    ap.add_argument('--val-frac', type=float, default=0.1, help='fraction of training DAYS held out')
    ap.add_argument('--stats', choices=['floe', 'autoice'], default='floe',
                    help='floe: original archive z-scores (v0); autoice: keep AutoICE z-scores (v2)')
    args = ap.parse_args()
    q = 2.0 ** args.q_exp
    # One flat output directory (like the FLOE archive); splits are datalists.
    os.makedirs(args.dst, exist_ok=True)
    jobs, splits = [], {'train': [], 'test': []}
    for split in ('train', 'test'):
        sd = osp.join(args.src, split)
        if not osp.isdir(sd):
            continue
        for f in sorted(os.listdir(sd)):
            if not f.endswith('_prep.nc'):
                continue
            ref = osp.join(sd, f.replace('_prep.nc', '_prep_reference.nc'))
            ref = ref if (split == 'test' and osp.exists(ref)) else None
            if split == 'test' and ref is None:
                print(f'skip {f}: no reference chart', flush=True)
                continue
            splits[split].append(f)
            dst = osp.join(args.dst, f)
            if not osp.exists(dst):
                jobs.append((osp.join(sd, f), ref, dst, q, args.codec, args.stats == 'autoice'))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f'{len(jobs)} scenes to convert -> {args.dst}', flush=True)
    tot = np.zeros(6, dtype=np.int64)
    bad = 0
    with Pool(args.workers) as pool:
        for prep, ok, info, dt in pool.imap_unordered(convert_one, jobs):
            if ok:
                tot += np.array(info)
                print(f'ok  {osp.basename(prep)} ({dt:.0f}s)', flush=True)
            else:
                bad += 1
                print(f'BAD {osp.basename(prep)}: {info}', flush=True)
    if tot.sum():
        print('FLOE pixel shares (converted scenes): ' +
              ' '.join(f'{c}={v / tot.sum():.4f}' for c, v in enumerate(tot)), flush=True)
    if args.lists_out and not args.limit:
        write_lists(splits, args.lists_out, args.val_frac)
    sys.exit(1 if bad else 0)


def write_lists(splits, out_dir, val_frac):
    """Write train/val/test datalists. test = the official test scenes; val holds out whole
    acquisition days (stable hash of the date), so frames of one day never straddle splits."""
    import hashlib, json
    os.makedirs(out_dir, exist_ok=True)
    tr = sorted(splits['train'])
    day = lambda f: f[:8]
    days = sorted({day(f) for f in tr})
    held = {d for d in days
            if int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_frac}
    val = [f for f in tr if day(f) in held]
    train = [f for f in tr if day(f) not in held]
    for name, lst in (('train', train), ('val', val), ('test', sorted(splits['test']))):
        with open(osp.join(out_dir, f'{name}.json'), 'w') as fh:
            json.dump(lst, fh, indent=1)
    print(f'datalists -> {out_dir}: train {len(train)} / val {len(val)} '
          f'({len(held)} of {len(days)} days) / test {len(splits["test"])}', flush=True)


if __name__ == '__main__':
    main()
