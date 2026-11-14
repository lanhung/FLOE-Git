"""One table with every chart polygon of the archive, for statistics and figures.

A scene file stores what the chart says about each polygon (egg codes, classes,
composition); how many pixels the polygon covers is only in the rasters. This script
reads the rasters once and writes a flat table (one row per polygon and scene), so that
class statistics by region, month or year, polygon sizes and the label-purity numbers
can be computed without opening the scenes again.

    python -m tools.build.polygon_table --data-dir $FLOE_V2_DIR --out $FLOE_LOG_DIR/polygons_v2

Output (.npz): scenes, and per row: scene (index into scenes), polygon, region (DMI
chart region), pixels (all pixels of the polygon), labelled (pixels with a FLOE class),
floe, sic, sod, composition (6 floe classes), unknown, codes (raw egg code).
"""
import argparse
import json
import os.path as osp
from multiprocessing import Pool

import h5py
import hdf5plugin  # noqa: F401  zstd filter
import numpy as np

from tools.build.split import load_scenes


def one(path):
    with h5py.File(path, 'r') as f:
        poly = f['polygon_id'][...]
        lab = f['FLOE'][...]
        charts = [c['name'].split('_')[1] for c in json.loads(f.attrs['charts'])]
        cols = {k: f['poly_' + k][...] for k in ('chart', 'floe', 'sic', 'sod', 'composition', 'unknown', 'codes')}
    n_poly = len(cols['floe'])
    inside = poly >= 0
    pixels = np.bincount(poly[inside], minlength=n_poly)
    labelled = np.bincount(poly[inside & (lab < 6)], minlength=n_poly)
    keep = np.nonzero(pixels)[0]
    out = {k: v[keep] for k, v in cols.items() if k != 'chart'}
    out.update(polygon=keep, pixels=pixels[keep], labelled=labelled[keep],
               region=np.array([charts[c] for c in cols['chart'][keep]]))
    return osp.basename(path)[:-3], out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--lists', nargs='*', help='datalists to restrict to (default: every scene with an ok record)')
    ap.add_argument('--out', required=True, help='output prefix (.npz)')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    if args.lists:
        names = sorted({n[:-3] if n.endswith('.nc') else n for p in args.lists for n in json.load(open(p))})
    else:
        names = sorted(load_scenes(osp.join(args.data_dir, 'archive_log.jsonl'), args.data_dir))
    rows = {}
    with Pool(args.workers) as pool:
        for k, (name, out) in enumerate(pool.imap_unordered(one, [osp.join(args.data_dir, n + '.nc') for n in names])):
            rows[name] = out
            if (k + 1) % 200 == 0:
                print(f'{k + 1}/{len(names)}', flush=True)
    table = {key: np.concatenate([rows[n][key] for n in names]) for key in rows[names[0]]}
    table['scene'] = np.concatenate([np.full(len(rows[n]['polygon']), i) for i, n in enumerate(names)])
    np.savez_compressed(args.out + '.npz', scenes=np.array(names), **table)
    print(f'{len(names)} scenes, {len(table["scene"])} polygons, {table["pixels"].sum() / 1e6:.0f} Mpx '
          f'({table["labelled"].sum() / 1e6:.0f} Mpx with a floe class) -> {args.out}.npz')


if __name__ == '__main__':
    main()
