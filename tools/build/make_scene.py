"""Build training scenes from zipped Sentinel-1 EW GRDM products and DMI ice charts.

    python -m tools.build.make_scene --zips $FLOE_ROOT/raw/s1 --charts $FLOE_ROOT/raw/dmi \
        --out $FLOE_ROOT/data_v2 --chart-out $FLOE_ROOT/charts_v2 --workers 6

For every product: sigma0 HH/HV at 80 m in SAR geometry (tools.build.s1), the DMI charts
within --max-minutes of the acquisition that overlap the scene, polygons burnt into the
image (tools.build.rasterise; the closest chart in time wins where charts overlap), and
SIC / SOD / FLOE labels plus per-polygon floe-size compositions (tools.build.charts).

Output <out>/<product>.nc (HDF5/netCDF, the layout the loaders read):
  nersc_sar_primary, nersc_sar_secondary, sar_incidenceangle  z-scores with the AutoICE
      statistics (so AutoICE ready-to-train data share the space), 0 where invalid,
      quantised to 2^-10 z and zstd-compressed as tools.data.recompress
  global_valid_mask, SIC (tenths), SOD (AutoICE classes 0-5), FLOE (0-6; 6 = bergs),
      polygon_id (-1 none), chart_edge_distance (80 m pixels to the nearest polygon edge)
  sar_grid_* tie points (row/col in 80 m pixels, lat, lon, height, incidence) for georeferencing
  poly_* table (one row per polygon: chart, raw egg code string, labels, composition)
and <chart-out>/<product>.chart.h5 for composition supervision (floe/composition.py).
Scenes with no matching chart are skipped and listed in <out>/build_log.jsonl.
"""
import argparse
import json
import os
import os.path as osp
import time
import traceback
from datetime import datetime
from multiprocessing import Pool

import h5py
import numpy as np

from tools.build import charts as ch
from tools.build import rasterise as rz
from tools.build import s1
from tools.data.recompress import codec_kwargs, quantise

VERSION = 'tools.build.make_scene v1 (2026-11-10)'
# AutoICE global statistics (toolkit misc/global_meanstd.npy), physical units
ZSTATS = {'nersc_sar_primary': (-14.508255155222800, 5.659745919381868),
          'nersc_sar_secondary': (-24.701205147144236, 4.746759305258515),
          'sar_incidenceangle': (33.998017884565360, 8.322267212079046)}
FIELDS = ('CT', 'CA', 'SA', 'FA', 'CB', 'SB', 'FB', 'CC', 'SC', 'FC', 'CN', 'CD', 'CF', 'POLY_TYPE')
Q = 2.0 ** -10


def _parse_time(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).replace(tzinfo=None)


def build_one(zip_path, chart_root, index, out_dir, chart_out, max_minutes):
    t0 = time.time()
    sc = s1.process(zip_path)
    from datetime import timezone
    start = _parse_time(sc.meta['start']).replace(tzinfo=timezone.utc)
    stop = _parse_time(sc.meta['stop']).replace(tzinfo=timezone.utc)
    fp = rz.footprint(sc.geo)
    matched = ch.match(index, start, stop, fp, max_minutes)
    if not matched:
        return {'product': sc.name, 'status': 'no_chart'}
    matched.sort(key=lambda r: -r['dt_minutes'])       # closest last: it wins overlaps

    geoms, ids, table, chart_names = [], [], [], []
    for k, rec in enumerate(matched):
        g = ch.read_chart(osp.join(chart_root, rec['path']))
        chart_names.append(rec['name'])
        for _, row in g.iterrows():
            pid = len(table)
            geoms.append(row.geometry)
            ids.append(pid)
            raw = ';'.join(str(ch.code(row.get(f))) if f != 'POLY_TYPE' else str(row.get(f, ''))
                           for f in FIELDS)
            table.append((k, raw, row['sic'], row['sod'], row['floe'], row['comp'], row['unknown']))
    shape = sc.valid.shape
    poly, rms = rz.burn(geoms, ids, sc.geo, shape)
    poly[~sc.valid] = -1

    P = len(table)
    sic_t = np.array([t[2] for t in table], np.uint8)
    sod_t = np.array([t[3] for t in table], np.uint8)
    floe_t = np.array([t[4] for t in table], np.uint8)
    comp_t = np.array([t[5] if t[5] is not None else np.full(6, np.nan) for t in table], np.float32)
    unk_t = np.array([t[6] for t in table], np.float32)
    has = poly >= 0
    lab = {}
    for name, tab in (('SIC', sic_t), ('SOD', sod_t), ('FLOE', floe_t)):
        a = np.full(shape, 255, np.uint8)
        a[has] = tab[poly[has]]
        lab[name] = a
    edge = rz.edge_distance(poly)

    used = np.unique(poly[has])
    coverage = float(has.sum() / max(sc.valid.sum(), 1))
    floe_hist = np.bincount(lab['FLOE'][lab['FLOE'] < 7], minlength=7).tolist()

    kw = codec_kwargs('zstd')
    dst = osp.join(out_dir, sc.name + '.nc')
    tmp = dst + '.tmp'
    with h5py.File(tmp, 'w', libver='earliest') as f:
        H, W = shape
        for nm, n in (('y', H), ('x', W)):
            d = f.create_dataset(nm, data=np.arange(n, dtype=np.float32))
            d.make_scale(nm)

        def put(name, arr, **attrs):
            d = f.create_dataset(name, data=arr, chunks=(512, 512), **kw)
            d.dims[0].attach_scale(f['y']); d.dims[1].attach_scale(f['x'])
            for k, v in attrs.items():
                d.attrs[k] = v

        for var, src in (('nersc_sar_primary', sc.sigma0_db['hh']), ('nersc_sar_secondary', sc.sigma0_db['hv']),
                         ('sar_incidenceangle', sc.incidence)):
            m, s = ZSTATS[var]
            z = np.where(sc.valid, (src.astype(np.float64) - m) / s, 0.0).astype(np.float32)
            z = quantise(z, Q)
            z[~sc.valid] = 0
            put(var, z, zscore_mean=np.array([m]), zscore_std=np.array([s]),
                units='dB' if var != 'sar_incidenceangle' else 'degrees',
                quantization_step_zscore=Q)
        put('global_valid_mask', sc.valid.astype(np.uint8))
        for name in ('SIC', 'SOD', 'FLOE'):
            put(name, lab[name], chart_fill_value=np.array([255], np.uint8))
        put('polygon_id', poly)
        put('chart_edge_distance', edge, units='80 m pixels, capped at 255')
        for k in ('line', 'pixel'):
            f.create_dataset(f'sar_grid_{"row" if k == "line" else "col"}', data=sc.geo[k])
        for k in ('lat', 'lon', 'height', 'inc'):
            f.create_dataset(f'sar_grid_{k}', data=sc.geo[k])
        f.create_dataset('poly_chart', data=np.array([t[0] for t in table], np.int16))
        f.create_dataset('poly_codes', data=np.array([t[1] for t in table], dtype='S96'))
        f.create_dataset('poly_sic', data=sic_t)
        f.create_dataset('poly_sod', data=sod_t)
        f.create_dataset('poly_floe', data=floe_t)
        f.create_dataset('poly_composition', data=comp_t)
        f.create_dataset('poly_unknown', data=unk_t)
        f.attrs['poly_codes_fields'] = ';'.join(FIELDS)
        f.attrs['product'] = sc.name
        f.attrs['mission'] = sc.meta['mission']
        f.attrs['ipf'] = sc.meta['ipf']
        f.attrs['pass'] = sc.meta['pass']
        f.attrs['start'] = sc.meta['start']
        f.attrs['stop'] = sc.meta['stop']
        f.attrs['charts'] = json.dumps([{'name': r['name'], 'dt_minutes': r['dt_minutes']} for r in matched])
        f.attrs['geometry'] = 'Sentinel-1 ground range (no map projection); 2x2 mean of 40 m pixels'
        f.attrs['denoising'] = 'ESA annotated noise (range x azimuth), negatives kept through averaging'
        f.attrs['label_rules'] = ('AutoICE: CA:=CT for single type, same-class partials summed, '
                                  'dominance >= 0.65 of CT, FLOE masked if any partial is fast ice, '
                                  'water polygons 0; range codes -> midpoint; DMI pancake (0) masked')
        f.attrs['chart_fit_rms_px'] = rms
        f.attrs['nonpositive_replaced'] = json.dumps(sc.meta['nonpositive_replaced'])
        f.attrs['built_by'] = VERSION
    os.replace(tmp, dst)

    # composition sidecar: local ids 0..n-1 over the polygons that occur in the image
    remap = np.full(P, -1, np.int64)
    remap[used] = np.arange(len(used))
    local = np.where(has, remap[np.maximum(poly, 0)], -1).astype(np.int32)
    comp = comp_t[used]
    unk = unk_t[used]
    masked = ~np.isfinite(comp).all(1)
    comp[masked] = 0
    unk[masked] = 1.0                                   # unknown everything: no supervision
    side = osp.join(chart_out, sc.name + '.chart.h5')
    with h5py.File(side + '.tmp', 'w') as f:
        f.create_dataset('polygon_id', data=local, chunks=(512, 512), **kw)
        f.create_dataset('composition', data=comp)
        f.create_dataset('unknown', data=unk)
        f.attrs['source'] = osp.basename(dst)
    os.replace(side + '.tmp', side)
    return {'product': sc.name, 'status': 'ok', 'charts': chart_names,
            'dt_minutes': [r['dt_minutes'] for r in matched], 'shape': list(shape),
            'coverage': coverage, 'polygons': int(len(used)), 'fit_rms_px': rms,
            'floe_hist': floe_hist, 'nonpositive_replaced': sc.meta['nonpositive_replaced'],
            'ipf': sc.meta['ipf'], 'seconds': round(time.time() - t0, 1)}


def _job(args):
    zp, chart_root, index, out_dir, chart_out, max_minutes = args
    try:
        return build_one(zp, chart_root, index, out_dir, chart_out, max_minutes)
    except Exception as e:
        return {'product': osp.basename(zp)[:-4], 'status': 'error', 'error': repr(e),
                'trace': traceback.format_exc()[-2000:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--zips', required=True, help='directory of <product>.zip or one .zip')
    ap.add_argument('--charts', required=True, help='DMI chart root with index.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chart-out', required=True)
    ap.add_argument('--max-minutes', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.chart_out, exist_ok=True)
    index = json.load(open(osp.join(args.charts, 'index.json')))
    zips = [args.zips] if args.zips.endswith('.zip') else \
        sorted(osp.join(args.zips, f) for f in os.listdir(args.zips) if f.endswith('.zip'))
    todo = [z for z in zips if args.overwrite or
            not osp.exists(osp.join(args.out, osp.basename(z)[:-4] + '.nc'))]
    print(f'{len(todo)} of {len(zips)} products to build', flush=True)
    log = open(osp.join(args.out, 'build_log.jsonl'), 'a')
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_job, [(z, args.charts, index, args.out, args.chart_out,
                                             args.max_minutes) for z in todo]):
            log.write(json.dumps(r) + '\n'); log.flush()
            print(f"{r['status']:8s} {r['product']} {r.get('charts', '')} "
                  f"cov={r.get('coverage', 0):.2f} rms={r.get('fit_rms_px', 0):.2f}px "
                  f"{r.get('seconds', '')}s {r.get('error', '')}", flush=True)


if __name__ == '__main__':
    main()
