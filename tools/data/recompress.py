#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-encode FLOE scenes into a compact copy, verified against the source.

Continuous channels are rounded to a grid of q = 2**q_exp z-units (default 2^-10, max
error q/2, i.e. 0.004 dB on HH) with exact zeros (invalid pixels) preserved. Labels and
global_valid_mask stay bit-exact and the 512x512 chunking is kept, so the tile index stays
valid. zstd output needs `import hdf5plugin` in the reader (floe/loaders.py does this).

  python -m tools.data.recompress --src $FLOE_DATA_DIR --dst DIR [--codec gzip|zstd] [--verify-only]
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  (registers Blosc/zstd filters)
except ImportError:      # gzip mode works without it
    hdf5plugin = None

VERSION = 'tools.data.recompress v1 (2026-11-10)'
CONTINUOUS = ['nersc_sar_primary', 'nersc_sar_secondary', 'sar_incidenceangle',
              'glcm_sigma0_hh_contrast', 'glcm_sigma0_hh_dissimilarity',
              'glcm_sigma0_hh_homogeneity']
EXACT = ['FLOE', 'SIC', 'SOD', 'global_valid_mask']   # categorical: copied bit-exact
DIMS = ['y', 'x']
SKIP_ATTRS = {'DIMENSION_LIST', 'REFERENCE_LIST', 'CLASS', 'NAME'}


def codec_kwargs(codec):
    if codec == 'gzip':
        return dict(compression='gzip', compression_opts=4, shuffle=True)
    if codec == 'zstd':
        if hdf5plugin is None:
            raise SystemExit('--codec zstd needs hdf5plugin')
        # clevel 5: 18x faster to encode than 9 for ~3% larger output (measured).
        return dict(**hdf5plugin.Blosc(cname='zstd', clevel=5,
                                       shuffle=hdf5plugin.Blosc.BITSHUFFLE))
    raise ValueError(codec)


def quantise(blk, q):
    """Round to the grid k*q (q a power of two, so the scaling is exact in float32).
    Valid pixels that would round to 0 keep their value: 0 marks invalid pixels."""
    out = np.multiply(blk, np.float32(1.0 / q), dtype=np.float32)
    np.rint(out, out=out)
    out *= np.float32(q)
    tiny = (out == 0) & (blk != 0)
    out[tiny] = blk[tiny]
    return out


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        if k not in SKIP_ATTRS:
            dst.attrs[k] = v


def convert_one(job):
    src_path, dst_path, q, codec = job
    tmp = dst_path + '.tmp'
    t0 = time.time()
    kw = codec_kwargs(codec)
    with h5py.File(src_path, 'r') as s, h5py.File(tmp, 'w', libver='earliest') as d:
        copy_attrs(s, d)
        d.attrs['recompressed_by'] = VERSION
        d.attrs['recompress_codec'] = codec
        d.attrs['recompress_quant_step_z'] = q
        for name in DIMS:
            ds = d.create_dataset(name, data=s[name][...])
            copy_attrs(s[name], ds)
            ds.make_scale(s[name].attrs['NAME'].decode()
                          if isinstance(s[name].attrs['NAME'], bytes) else s[name].attrs['NAME'])
        for name in s:
            if name in DIMS:
                continue
            sv = s[name]
            if sv.ndim != 2:
                raise ValueError(f'{src_path}:{name} unexpected ndim {sv.ndim}')
            if name not in CONTINUOUS and name not in EXACT:
                raise ValueError(f'{src_path}: unknown variable {name}; refusing to guess')
            dv = d.create_dataset(name, shape=sv.shape, dtype=sv.dtype,
                                  chunks=sv.chunks or (512, 512), **kw)
            copy_attrs(sv, dv)
            if name in CONTINUOUS:
                dv.attrs['quantization_step_zscore'] = q
                dv.attrs['quantization_max_abs_error_zscore'] = q / 2
            step = dv.chunks[0]
            for r0 in range(0, sv.shape[0], step):
                blk = sv[r0:r0 + step]
                if name in CONTINUOUS:
                    blk = quantise(blk, q).astype(sv.dtype, copy=False)
                dv[r0:r0 + step] = blk
            dv.dims[0].attach_scale(d['y'])
            dv.dims[1].attach_scale(d['x'])
    problems = verify_pair(src_path, tmp, q)
    if problems:
        os.remove(tmp)
        return src_path, False, problems, time.time() - t0
    os.replace(tmp, dst_path)
    return src_path, True, [], time.time() - t0


def verify_pair(src_path, dst_path, q):
    problems = []
    with h5py.File(src_path, 'r') as s, h5py.File(dst_path, 'r') as d:
        if sorted(s) != sorted(d):
            problems.append(f'variables differ: {sorted(s)} vs {sorted(d)}')
            return problems
        for name in s:
            sv, dv = s[name], d[name]
            if sv.shape != dv.shape or sv.dtype != dv.dtype:
                problems.append(f'{name}: shape/dtype {sv.shape}/{sv.dtype} vs {dv.shape}/{dv.dtype}')
                continue
            if sv.ndim == 2 and dv.chunks != (sv.chunks or (512, 512)):
                problems.append(f'{name}: chunks {dv.chunks} != {sv.chunks}')
            for k, v in sv.attrs.items():
                if k in SKIP_ATTRS:
                    continue
                if k not in dv.attrs or not np.array_equal(np.asarray(v), np.asarray(dv.attrs[k])):
                    problems.append(f'{name}: attribute {k} differs')
            if name in DIMS:
                if not np.array_equal(sv[...], dv[...]):
                    problems.append(f'{name}: coordinate values differ')
                continue
            step = sv.chunks[0] if sv.chunks else 512
            worst = 0.0
            for r0 in range(0, sv.shape[0], step):
                a, b = sv[r0:r0 + step], dv[r0:r0 + step]
                if name in EXACT:
                    if not np.array_equal(a, b):
                        problems.append(f'{name}: not bit-exact')
                        break
                else:
                    worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b))))
                    if not np.array_equal(a == 0, b == 0):
                        problems.append(f'{name}: zero pattern changed')
                        break
            if name in CONTINUOUS and worst > q / 2 * (1 + 1e-6) + 1e-7:
                problems.append(f'{name}: max error {worst:.3g} > q/2 = {q / 2:.3g}')
        for k, v in s.attrs.items():
            if k not in d.attrs:
                problems.append(f'root attribute {k} missing')
    try:
        import xarray as xr
        with xr.open_dataset(dst_path, engine='h5netcdf', mask_and_scale=False) as ds:
            if tuple(ds['FLOE'].dims) != ('y', 'x'):
                problems.append(f'xarray dims {ds["FLOE"].dims}')
    except Exception as e:
        problems.append(f'xarray cannot open: {type(e).__name__}: {e}')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--q-exp', type=int, default=-10, help='quantisation step = 2**q_exp z-units')
    ap.add_argument('--codec', choices=['gzip', 'zstd'], default='zstd')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lists', nargs='*', default=None,
                    help='only scenes in these datalists (default: every .nc in --src)')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    q = 2.0 ** args.q_exp
    os.makedirs(args.dst, exist_ok=True)

    on_disk = {f for f in os.listdir(args.src) if f.endswith('.nc') and not f.startswith('.')}
    partial = {f for f in os.listdir(args.src) if f.startswith('.')}
    if args.lists:
        wanted = set()
        for p in args.lists:
            for g in glob.glob(p):
                wanted |= set(json.load(open(g)))
        names = sorted(n for n in wanted if n in on_disk)
    else:
        names = sorted(on_disk)
    names = [n for n in names if not any(p.startswith('.' + n) for p in partial)]

    if args.verify_only:
        bad = 0
        for n in names:
            dp = osp.join(args.dst, n)
            if not osp.exists(dp):
                print('missing', n)
                bad += 1
                continue
            pr = verify_pair(osp.join(args.src, n), dp, q)
            if pr:
                bad += 1
                print('BAD', n, pr)
        print(f'verified {len(names)}: {len(names) - bad} ok, {bad} bad')
        sys.exit(1 if bad else 0)

    todo = [(osp.join(args.src, n), osp.join(args.dst, n), q, args.codec)
            for n in names if not osp.exists(osp.join(args.dst, n))]
    print(f'{len(names)} scenes complete in src, {len(names) - len(todo)} already converted, '
          f'{len(todo)} to do | q=2^{args.q_exp} codec={args.codec} workers={args.workers}',
          flush=True)
    s_in = s_out = 0
    n_bad = 0
    with Pool(args.workers) as pool:
        for src_path, ok, problems, dt in pool.imap_unordered(convert_one, todo):
            n = osp.basename(src_path)
            if ok:
                a, b = osp.getsize(src_path), osp.getsize(osp.join(args.dst, n))
                s_in += a
                s_out += b
                print(f'ok  {n}  {a / 1e6:.0f} -> {b / 1e6:.0f} MB  ({dt:.0f}s)', flush=True)
            else:
                n_bad += 1
                print(f'BAD {n}: {problems}', flush=True)
    if s_in:
        print(f'done: {s_in / 1e9:.1f} GB -> {s_out / 1e9:.1f} GB ({s_in / max(s_out, 1):.2f}x), '
              f'{n_bad} failed', flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
