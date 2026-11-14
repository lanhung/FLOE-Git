"""DMI ice charts (SIGRID-3): download, index, match to Sentinel-1 scenes, decode egg codes.

Archive (open, no account): https://download.dmi.dk/public/ICESERVICE/SIGRID3/{YYYY}/
one folder per chart, {YYYYMMDDhhmm}_{Region}_RIC/{same}.{shp,shx,dbf,prj,cpg}, WGS84.
A regional chart is drawn on one Sentinel-1 image and its name carries that image's
time in UTC, so charts are matched to scenes by time (minutes) and footprint overlap.

    python -m tools.build.charts fetch --years 2022 2023 2024 2025 2026 --out $FLOE_ROOT/raw/dmi
    python -m tools.build.charts index --dir $FLOE_ROOT/raw/dmi

Labels follow the AutoICE converter (Stokholm et al. 2024; convert_raw_icechart.py):
CT copied into CA for single-type polygons, partials mapping to the same class summed,
a SOD/FLOE class only where it holds >= 65 % of CT, FLOE masked when any partial is
fast ice, water polygons 0 in all charts. Two departures, both deliberate: range codes
(e.g. 46 = 4-6/10) are decoded to their midpoint, and DMI's pancake code 0 is masked
instead of being read as water.
"""
import argparse
import json
import os
import os.path as osp
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

BASE = 'https://download.dmi.dk/public/ICESERVICE/SIGRID3'
EXTS = ('shp', 'shx', 'dbf', 'prj')
NAME_RE = re.compile(r'^(\d{12})_([A-Za-z]+)_RIC$')

MASK = 255
DOMINANCE = 0.65
# SIGRID-3 stage of development -> AutoICE SOD class (0 water, 1 new, 2 young,
# 3 thin FYI, 4 thick FYI, 5 old); 98 glacier ice and 99 unknown are masked.
SOD_CLASS = {80: 0, 81: 1, 82: 1, 83: 2, 84: 2, 85: 2, 86: 4, 87: 3, 88: 3, 89: 3,
             91: 4, 93: 4, 95: 5, 96: 5, 97: 5}
# form of ice (egg digits, as written in DMI/CIS/NIC files) -> AutoICE FLOE class
# (0 water, 1 cake, 2 small, 3 medium, 4 big, 5 vast/giant, 6 bergs). Pancake (0, 22),
# brash (1), fast ice (8), strips (11-20), level ice (21), 99 have no class.
FLOE_CLASS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5, 9: 6, 10: 6}
FAST_ICE = 8

# Charts with a polygon far outside their region, found by screen_extents() and checked by
# hand (11 Nov 2026). In both the oversized polygon covers none of the chart's other
# polygons and shares the map frame's corner (117.5 W, about 55 N), which looks like a ring
# whose orientation was corrected wrongly when the shapefile is read. It would label the
# open sea outside the chart, or most of the Arctic, with one code. Pairing (match), the
# scene lists (tools.build.split, tools.build.polygon_table) and the chart comparisons skip
# them. Two CentralWest charts of 23 and 25 May 2025 also pass the screen; their large
# polygon is open water over Davis Strait, copied from one chart to the next, and is kept.
EXCLUDED_CHARTS = {
    '202206211050_NorthWest_RIC': 'open-water polygon of 1.6e6 km2 reaching 117.5 W and 87 N',
    '202408170800_North_RIC': 'ice polygon (CT 10/10) over most of the Arctic, 117.5 W to 41 E',
}


# ---------------------------------------------------------------- archive

def list_charts(year, session=None):
    s = session or requests
    r = s.get(f'{BASE}/{year}/', timeout=120)
    r.raise_for_status()
    names = sorted(set(re.findall(r'href="(\d{12}_[A-Za-z]+_RIC)/"', r.text)))
    return names


def _fetch_chart(year, name, out, sess):
    d = osp.join(out, str(year), name)
    if all(osp.exists(osp.join(d, f'{name}.{e}')) for e in ('shp', 'shx', 'dbf')):
        return False
    os.makedirs(d, exist_ok=True)
    for e in EXTS:
        url = f'{BASE}/{year}/{name}/{name}.{e}'
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=120)
                if r.status_code == 404:          # some folders lack a file; nothing to retry
                    break
                r.raise_for_status()
                with open(osp.join(d, f'{name}.{e}.part'), 'wb') as f:
                    f.write(r.content)
                os.replace(osp.join(d, f'{name}.{e}.part'), osp.join(d, f'{name}.{e}'))
                break
            except requests.RequestException as ex:
                print(f'  {name}.{e}: {ex}; retry', flush=True)
                time.sleep(10 * (attempt + 1))
    return True


def fetch(years, out, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    sess = requests.Session()
    sess.mount('https://', requests.adapters.HTTPAdapter(pool_maxsize=workers))
    n_new = 0
    for y in years:
        names = list_charts(y, sess)
        print(f'{y}: {len(names)} charts', flush=True)
        with ThreadPoolExecutor(workers) as pool:
            n_new += sum(pool.map(lambda n: _fetch_chart(y, n, out, sess), names))
        print(f'{y}: done ({n_new} new so far)', flush=True)
    print(f'{n_new} charts downloaded to {out}')


def chart_time(name):
    m = NAME_RE.match(name)
    return datetime.strptime(m.group(1), '%Y%m%d%H%M').replace(tzinfo=timezone.utc), m.group(2)


def build_index(root):
    """One record per chart: name, time, region, bounding box (lon/lat), polygon count."""
    import pyogrio
    recs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith('.shp'):
                continue
            name = f[:-4]
            if not NAME_RE.match(name) or not all(
                    osp.exists(osp.join(dirpath, f'{name}.{e}')) for e in ('shx', 'dbf')):
                continue
            info = pyogrio.read_info(osp.join(dirpath, f))
            t, region = chart_time(name)
            recs.append({'name': name, 'path': osp.relpath(osp.join(dirpath, f), root),
                         'time': t.isoformat(), 'region': region.lower(),
                         'bbox': [float(v) for v in info['total_bounds']],
                         'n_polygons': int(info['features'])})
    recs.sort(key=lambda r: r['time'])
    with open(osp.join(root, 'index.json'), 'w') as fh:
        json.dump(recs, fh, indent=0)
    print(f'{len(recs)} charts indexed -> {osp.join(root, "index.json")}')
    for name, ratio, bbox in screen_extents(recs):
        flag = 'excluded' if name in EXCLUDED_CHARTS else 'CHECK BY HAND'
        print(f'  extent {ratio:.1f}x the median of its region: {name} {bbox} [{flag}]')
    return recs


def screen_extents(index, factor=3.0):
    """Charts whose bounding box (in square degrees) is more than `factor` times the
    median of their region: candidates for EXCLUDED_CHARTS, to be looked at by hand."""
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])
    by = {}
    for r in index:
        by.setdefault(r['region'], []).append(r)
    out = []
    for recs in by.values():
        med = float(np.median([area(r['bbox']) for r in recs]))
        out += [(r['name'], area(r['bbox']) / med, [round(v, 1) for v in r['bbox']])
                for r in recs if area(r['bbox']) > factor * med]
    return sorted(out, key=lambda o: -o[1])


def match(index, start, stop, footprint, max_minutes=30):
    """Charts whose time is within max_minutes of [start, stop] and whose bbox
    intersects the scene footprint (a shapely geometry in lon/lat)."""
    from shapely.geometry import box
    out = []
    for r in index:
        if r['name'] in EXCLUDED_CHARTS:
            continue
        t = datetime.fromisoformat(r['time'])
        dt = 0.0 if start <= t <= stop else min(abs((t - start).total_seconds()),
                                                abs((t - stop).total_seconds())) / 60
        if dt <= max_minutes and box(*r['bbox']).intersects(footprint):
            out.append({**r, 'dt_minutes': dt})
    return out


# ---------------------------------------------------------------- egg codes

def code(v):
    """SIGRID-3 text field -> int; blank/None/garbage -> -9."""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return -9


def concentration_tenths(c):
    """CT/CA/CB/CC code -> tenths (float), or NaN if unknown/not filled."""
    if c in (0, 2, 55, 98):       # ice free (00, 98 in v3.1, 55 in AI4Arctic), bergy water
        return 0.0
    if c == 1:                    # open water, < 1/10
        return 0.5
    if c in (91, 92):             # 9+/10, 10/10
        return 10.0
    if 10 <= c <= 90 and c % 10 == 0:
        return c / 10.0
    if c == 81:                   # 8-10/10
        return 9.0
    lo, hi = divmod(c, 10)
    if 1 <= lo < hi <= 9:         # ranges such as 13, 46, 89
        return (lo + hi) / 2.0
    return float('nan')


def decode(row):
    """One chart polygon -> dict with poly_type, ct (tenths) and partials
    [(tenths, stage, form), ...]. Missing CA for a single ice type becomes CT."""
    pt = str(row.get('POLY_TYPE', '') or '').strip().upper()[:1]
    ct = concentration_tenths(code(row.get('CT')))
    parts = []
    for c, s, f in (('CA', 'SA', 'FA'), ('CB', 'SB', 'FB'), ('CC', 'SC', 'FC')):
        cc, ss, ff = code(row.get(c)), code(row.get(s)), code(row.get(f))
        if ss == -9 and ff == -9 and cc == -9:
            continue
        parts.append((concentration_tenths(cc) if cc != -9 else float('nan'), ss, ff))
    if parts and not np.isfinite(parts[0][0]) and np.isfinite(ct) and ct > 0 and \
            all(not np.isfinite(p[0]) for p in parts):
        parts[0] = (ct, parts[0][1], parts[0][2])
    return {'poly_type': pt, 'ct': ct, 'parts': parts}


def _dominant(parts, ct, table):
    sums = {}
    for c, s, f in parts:
        k = table(s, f)
        if k is None or not np.isfinite(c):
            continue
        sums[k] = sums.get(k, 0.0) + c
    if not sums or not ct > 0:
        return MASK
    k, v = max(sums.items(), key=lambda kv: kv[1])
    return k if v / ct >= DOMINANCE else MASK


def labels(d):
    """Dominant SIC (tenths 0-10), SOD (0-5) and FLOE (0-6) of one decoded polygon,
    255 where masked."""
    pt, ct, parts = d['poly_type'], d['ct'], d['parts']
    if pt == 'W' or (pt == 'I' and ct < 1):      # ice free, bergy water, open water (< 1/10)
        return 0, 0, 0
    if pt != 'I' or not np.isfinite(ct):
        return MASK, MASK, MASK
    sic = int(round(ct))
    sod = _dominant(parts, ct, lambda s, f: SOD_CLASS.get(s))
    if any(f == FAST_ICE for _, _, f in parts):
        floe = MASK
    else:
        floe = _dominant(parts, ct, lambda s, f: FLOE_CLASS.get(f))
    return sic, sod, floe


def floe_composition(d):
    """6-class FLOE composition (water, cake, small, medium, big, vast) and the
    unknown ice fraction; see floe.composition. Bergs, pancake, brash, fast ice and
    undetermined forms count as unknown ice. Returns (q[6], unknown) or None if masked."""
    pt, ct, parts = d['poly_type'], d['ct'], d['parts']
    if pt == 'W' or (pt == 'I' and ct == 0):
        q = np.zeros(6); q[0] = 1.0
        return q, 0.0
    if pt != 'I' or not np.isfinite(ct):
        return None
    q = np.zeros(6)
    ctf = min(ct, 10.0) / 10.0
    q[0] = 1.0 - ctf
    known = 0.0
    for c, s, f in parts:
        k = FLOE_CLASS.get(f)
        if k is None or k == 6 or not np.isfinite(c) or c <= 0:
            continue
        q[k] += c / 10.0
        known += c / 10.0
    if known > ctf > 0:               # partials that overshoot CT (seen in charts) are rescaled
        q[1:] *= ctf / known
        known = ctf
    unknown = ctf - known
    return q, unknown if unknown > 1e-6 else 0.0


def read_chart(path):
    """Chart polygons as a GeoDataFrame in EPSG:4326 with decoded columns
    poly_type, ct, sic, sod, floe, comp (6-vector or None), unknown."""
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    dec = [decode(r) for r in g.drop(columns='geometry').to_dict('records')]
    g['poly_type'] = [d['poly_type'] for d in dec]
    g['ct'] = [d['ct'] for d in dec]
    lab = [labels(d) for d in dec]
    g['sic'], g['sod'], g['floe'] = zip(*lab) if lab else ([], [], [])
    comp = [floe_composition(d) for d in dec]
    g['comp'] = [c[0] if c else None for c in comp]
    g['unknown'] = [c[1] if c else np.nan for c in comp]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--years', type=int, nargs='+', required=True)
    f.add_argument('--out', required=True)
    i = sub.add_parser('index')
    i.add_argument('--dir', required=True)
    args = ap.parse_args()
    if args.cmd == 'fetch':
        fetch(args.years, args.out)
    else:
        build_index(args.dir)


if __name__ == '__main__':
    sys.exit(main())
