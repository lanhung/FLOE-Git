"""Write the public release of the rebuilt archive (dataset paper, data availability section).

Every scene of the dataset (the 3224 scenes of tools.build.polygon_table; the 3 scenes of the
two excluded charts are left out) is rewritten as a netCDF4 file that standard readers open
without plugins or phony dimensions:

  - the same arrays, bit for bit, and all attributes of the working copy;
  - standard deflate compression (level 4, shuffle) instead of zstd, same 512x512 chunks;
  - named dimensions: y, x for the rasters, polygon (and floe_class) for the per-polygon
    table, grid_row, grid_col for the georeferencing tie-point grid;
  - global attributes: title, licence (CC BY 4.0), attribution of the charts (as agreed with
    DMI, November 2026) and of Copernicus Sentinel data, version, creation date;
  - a long_name for every variable.

Output (under --out): data/<year>/<product>.nc, scenes.csv (one row per scene: product, time,
mission, charts, largest chart-frame gap, strict-subset flag, benchmark role, file size, MD5),
splits/ (the benchmark fold lists), MD5SUMS, README.md, LICENSE.txt.

    python -m tools.build.release --data-dir $FLOE_V2_DIR --table $FLOE_LOG_DIR/polygons_v2_full.npz \
        --log $FLOE_V2_DIR/archive_log.jsonl --split datalists/v2_bench --out $FLOE_ROOT/release_v1 --workers 4

Files are written to <name>.part and renamed when complete, so an interrupted run resumes by
skipping finished files. --check re-reads every released file and compares it with its source.
"""
import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import shutil
from datetime import datetime, timezone
from multiprocessing import Pool

import h5py
import hdf5plugin  # noqa: F401  (reads the zstd working copies)
import netCDF4
import numpy as np

VERSION = '1.0'
LICENSE = 'CC-BY-4.0'
TITLE = 'FLOE: Sentinel-1 EW scenes with DMI ice-chart floe-size labels around Greenland, 2022-2026'
CHART_ATTRIBUTION = 'The Greenland Ice Service, Danish Meteorological Institute'
S1_ATTRIBUTION = 'Contains modified Copernicus Sentinel data 2022-2026'

LONG_NAMES = {
    'nersc_sar_primary': 'HH backscatter, noise-corrected, standardised (z-score)',
    'nersc_sar_secondary': 'HV backscatter, noise-corrected, standardised (z-score)',
    'sar_incidenceangle': 'incidence angle, standardised (z-score)',
    'global_valid_mask': 'valid SAR pixel (1) or not (0)',
    'FLOE': 'floe-size class of the chart polygon (AutoICE rules, dominance 0.65)',
    'SIC': 'sea-ice concentration of the chart polygon in tenths (AutoICE rules)',
    'SOD': 'stage of development of the chart polygon (AutoICE rules)',
    'polygon_id': 'index of the chart polygon covering the pixel (into the poly_* table)',
    'chart_edge_distance': 'distance to the nearest edge of a chart polygon',
    'poly_chart': 'chart of the polygon (index into the global attribute charts)',
    'poly_codes': 'SIGRID-3 egg code of the polygon (fields in poly_codes_fields)',
    'poly_composition': 'floe-size composition of the polygon from its egg code',
    'poly_floe': 'floe-size class of the polygon',
    'poly_sic': 'concentration class of the polygon',
    'poly_sod': 'stage-of-development class of the polygon',
    'poly_unknown': 'share of the ice of the polygon with unknown floe size',
    'sar_grid_row': 'row of the georeferencing tie points',
    'sar_grid_col': 'column of the georeferencing tie points',
    'sar_grid_lat': 'latitude of the georeferencing tie points',
    'sar_grid_lon': 'longitude of the georeferencing tie points',
    'sar_grid_height': 'height of the georeferencing tie points',
    'sar_grid_inc': 'incidence angle at the georeferencing tie points',
    'x': 'column in the SAR image geometry',
    'y': 'row in the SAR image geometry',
}
SKIP_ATTRS = {'DIMENSION_LIST', 'REFERENCE_LIST', 'CLASS', 'NAME', '_Netcdf4Dimid', '_Netcdf4Coordinates'}


def dims_of(name, shape, ny, nx, n_poly, grid):
    if name in ('x',):
        return ('x',)
    if name in ('y',):
        return ('y',)
    if name.startswith('poly_'):
        return ('polygon', 'floe_class')[:len(shape)]
    if name == 'sar_grid_row':
        return ('grid_row',)
    if name == 'sar_grid_col':
        return ('grid_col',)
    if name.startswith('sar_grid_'):
        return ('grid_row', 'grid_col')
    if shape == (ny, nx):
        return ('y', 'x')
    raise ValueError(f'no dimensions known for {name} {shape}')


def attr_value(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind == 'S':
        return [s.decode() for s in v]
    return v


def write_one(job):
    src, dst = job
    if osp.exists(dst):
        return dst, 'exists'
    tmp = dst + '.part'
    os.makedirs(osp.dirname(dst), exist_ok=True)
    with h5py.File(src, 'r') as a:
        ny, nx = a['FLOE'].shape
        n_poly = a['poly_floe'].shape[0]
        with netCDF4.Dataset(tmp, 'w', format='NETCDF4') as b:
            b.createDimension('y', ny)
            b.createDimension('x', nx)
            b.createDimension('polygon', n_poly)
            b.createDimension('floe_class', a['poly_composition'].shape[1])
            b.createDimension('grid_row', a['sar_grid_row'].shape[0])
            b.createDimension('grid_col', a['sar_grid_col'].shape[0])
            for k, v in a.attrs.items():
                if k not in SKIP_ATTRS:
                    b.setncattr(k, attr_value(v))
            b.setncattr('title', TITLE)
            b.setncattr('version', VERSION)
            b.setncattr('license', LICENSE)
            b.setncattr('ice_chart_attribution', CHART_ATTRIBUTION)
            b.setncattr('sar_attribution', S1_ATTRIBUTION)
            b.setncattr('date_created', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            for name in sorted(a.keys()):
                d = a[name]
                data = d[()]
                dims = dims_of(name, d.shape, ny, nx, n_poly, None)
                if d.dtype.kind == 'S':
                    v = b.createVariable(name, str, dims)
                    v[:] = np.array([s.decode() for s in data], dtype=object)
                else:
                    kw = {}
                    if d.chunks and len(d.shape) == 2:
                        kw = dict(zlib=True, complevel=4, shuffle=True, chunksizes=d.chunks)
                    v = b.createVariable(name, d.dtype, dims, **kw)
                    v.set_auto_maskandscale(False)
                    v[...] = data
                for k, val in d.attrs.items():
                    if k not in SKIP_ATTRS:
                        v.setncattr(k, attr_value(val))
                if name in LONG_NAMES:
                    v.setncattr('long_name', LONG_NAMES[name])
    os.replace(tmp, dst)
    return dst, 'written'


def check_one(job):
    src, dst = job
    with h5py.File(src, 'r') as a, netCDF4.Dataset(dst, 'r') as b:
        for name in a.keys():
            x = a[name][()]
            v = b.variables[name]
            v.set_auto_maskandscale(False)
            y = v[...]
            if x.dtype.kind == 'S':
                x = np.array([s.decode() for s in x], dtype=object)
                y = np.asarray(y, dtype=object)
            same = (np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True) if np.asarray(x).dtype.kind == 'f'
                    else np.array_equal(np.asarray(x), np.asarray(y)))
            if not same:   # NaN (undefined compositions) must count as equal to NaN
                return dst, f'MISMATCH in {name}'
        for k in a.attrs:
            if k not in SKIP_ATTRS and k not in b.ncattrs():
                return dst, f'missing attribute {k}'
    return dst, 'ok'


def md5(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--table', required=True, help='tools.build.polygon_table output (defines the scenes)')
    ap.add_argument('--log', required=True, help='archive_log.jsonl of the build')
    ap.add_argument('--split', required=True, help='benchmark split directory (fold lists, test.json, excluded.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None, help='first N scenes only (testing)')
    ap.add_argument('--check', action='store_true', help='compare every released file with its source')
    args = ap.parse_args()

    names = sorted(str(s) for s in np.load(args.table)['scenes'])
    if args.limit:
        names = names[:args.limit]
    jobs = [(osp.join(args.data_dir, n + '.nc'), osp.join(args.out, 'data', n[17:21], n + '.nc')) for n in names]
    missing = [s for s, _ in jobs if not osp.exists(s)]
    if missing:
        raise SystemExit(f'{len(missing)} source files missing, e.g. {missing[:2]}')

    with Pool(args.workers) as pool:
        if args.check:
            bad = [r for r in pool.imap_unordered(check_one, jobs, chunksize=4) if r[1] != 'ok']
            print(f'checked {len(jobs)} files: {len(bad)} problems', *bad[:10], sep='\n')
            raise SystemExit(1 if bad else 0)
        done = 0
        for dst, status in pool.imap_unordered(write_one, jobs, chunksize=2):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {done}/{len(jobs)}', flush=True)

    # scene table, checksums, splits
    log = {}
    for line in open(args.log):
        r = json.loads(line)
        log[r['product']] = r
    role = {}
    for k in range(5):
        for n in json.load(open(osp.join(args.split, f'fold{k}_val.json'))):
            role[n[:-3]] = f'fold{k}'
    for n in json.load(open(osp.join(args.split, 'test.json'))):
        role[n[:-3]] = 'test_2026'
    for n, why in json.load(open(osp.join(args.split, 'excluded.json'))).items():
        role[n] = 'excluded_buffer' if 'within' in why else 'excluded_chart_gap'
    rows = []
    with Pool(args.workers) as pool:
        sums = dict(zip([d for _, d in jobs], pool.map(md5, [d for _, d in jobs], chunksize=8)))
    for n, (_, dst) in zip(names, jobs):
        r = log[n]
        gaps = r['dt_minutes'] if isinstance(r['dt_minutes'], list) else [r['dt_minutes']]
        gap = max(abs(g) for g in gaps)
        rows.append({'product': n, 'file': osp.relpath(dst, args.out), 'year': n[17:21],
                     'start': n[17:32], 'mission': n[:3], 'charts': ';'.join(r['charts']),
                     'max_chart_gap_min': f'{gap:.2f}', 'strict_subset': int(gap <= 5.0),
                     'benchmark_role': role.get(n, 'none'), 'bytes': osp.getsize(dst), 'md5': sums[dst]})
    with open(osp.join(args.out, 'scenes.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(osp.join(args.out, 'MD5SUMS'), 'w') as f:
        for row in rows:
            f.write(f"{row['md5']}  {row['file']}\n")
    sd = osp.join(args.out, 'splits')
    os.makedirs(sd, exist_ok=True)
    for f in os.listdir(args.split):
        if f.endswith('.json') or f.endswith('.md'):
            shutil.copy(osp.join(args.split, f), sd)
    total = sum(r['bytes'] for r in rows)
    print(f'{len(rows)} scenes, {total / 1e9:.1f} GB; roles:',
          {k: sum(r["benchmark_role"] == k for r in rows) for k in sorted({r["benchmark_role"] for r in rows})})


if __name__ == '__main__':
    main()
