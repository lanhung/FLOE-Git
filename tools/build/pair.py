"""Pair DMI ice charts with the Sentinel-1 EW frames they were drawn on.

A DMI regional chart is a snapshot of one Sentinel-1 pass, named with the image time.
For every chart day the ASF catalogue is queried once over Greenland, and each chart is
matched to the EW GRDM HH+HV frames that start within --max-minutes of its time and
overlap its bounding box. Charts drawn on other satellites (RCM, RADARSAT-2) find no
frame and are dropped, which is the point: the label must come from the same image.

    python -m tools.build.pair --charts $FLOE_ROOT/raw/dmi --out $FLOE_ROOT/raw/pairs.json

Output: {"frames": {product: {url, bytes, md5, start, charts: [...], dt_minutes}},
         "charts": {chart: [products]}, "unpaired_charts": [...]}
"""
import argparse
import json
import os.path as osp
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
import shapely
from shapely.geometry import box, shape

SEARCH = 'https://api.daac.asf.alaska.edu/services/search/param'
GREENLAND = 'POLYGON((-80 55,15 55,15 85,-80 85,-80 55))'


def day_products(day):
    """All EW GRDM HH+HV frames over Greenland on one UTC day (start time inside it)."""
    params = {'platform': 'Sentinel-1', 'processingLevel': 'GRD_MD', 'beamMode': 'EW',
              'polarization': 'HH+HV', 'intersectsWith': GREENLAND, 'output': 'geojson',
              'start': (day - timedelta(minutes=45)).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'end': (day + timedelta(days=1, minutes=45)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    for attempt in range(6):
        try:
            r = requests.get(SEARCH, params=params, timeout=180)
            if r.status_code == 200:
                return r.json().get('features', [])
        except requests.RequestException:
            pass
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f'ASF search failed for {day:%Y-%m-%d}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--charts', required=True, help='DMI chart root with index.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-minutes', type=float, default=30.0)
    ap.add_argument('--years', type=int, nargs='*', default=None)
    args = ap.parse_args()

    index = json.load(open(osp.join(args.charts, 'index.json')))
    if args.years:
        index = [r for r in index if int(r['time'][:4]) in args.years]
    by_day = defaultdict(list)
    for r in index:
        t = datetime.fromisoformat(r['time'])
        by_day[t.date()].append((t, r))

    frames, charts, unpaired = {}, {}, []
    for k, (d, recs) in enumerate(sorted(by_day.items())):
        day = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        feats = day_products(day)
        prods = []
        for f in feats:
            p = f['properties']
            start = datetime.fromisoformat(p['startTime'].replace('Z', '+00:00'))
            stop = datetime.fromisoformat(p['stopTime'].replace('Z', '+00:00'))
            prods.append((p['sceneName'], start, stop, shape(f['geometry']), p))
        for t, r in recs:
            bb = box(*r['bbox'])
            hits = []
            for name, start, stop, geom, p in prods:
                dt = 0.0 if start <= t <= stop else min(abs((t - start).total_seconds()),
                                                        abs((t - stop).total_seconds())) / 60
                if dt <= args.max_minutes and geom.intersects(bb):
                    hits.append(name)
                    fr = frames.setdefault(name, {'url': p['url'], 'bytes': int(p.get('bytes') or 0),
                                                  'md5': (p.get('md5sum') or '').lower(),
                                                  'start': p['startTime'], 'charts': [], 'dt_minutes': []})
                    fr['charts'].append(r['name'])
                    fr['dt_minutes'].append(round(dt, 2))
            if hits:
                charts[r['name']] = hits
            else:
                unpaired.append(r['name'])
        if k % 50 == 0:
            print(f'{d}: {len(charts)} charts paired, {len(unpaired)} unpaired, {len(frames)} frames', flush=True)
        time.sleep(0.5)

    with open(args.out, 'w') as fh:
        json.dump({'frames': frames, 'charts': charts, 'unpaired_charts': unpaired}, fh)
    gb = sum(f['bytes'] for f in frames.values()) / 1e9
    print(f'{len(charts)} of {len(index)} charts paired with {len(frames)} frames ({gb:.0f} GB); '
          f'{len(unpaired)} charts without a Sentinel-1 frame -> {args.out}')


if __name__ == '__main__':
    main()
