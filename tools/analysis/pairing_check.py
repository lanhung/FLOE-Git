"""How far the charts of our scenes are from their frames, next to the pairing of the
AutoICE organisers.

The DMI chart files do not name the image they were drawn on, so tools.build.pair pairs
by time (frames within --max-minutes of the chart time) and place. The ready-to-train
AutoICE files name both the Sentinel-1 product and the chart of every scene in their
`original_id` attribute, e.g.

  S1A_EW_GRDM_1SDH_20180108T184332_20180108T184437_..._icechart_dmi_201801081845_CentralEast_RIC.nc

which gives the gap between chart time and frame in a pairing made by the organisers.
Gap = 0 if the chart time falls inside the frame, else the distance to its start or
stop. For our scenes the largest gap over the charts of a scene is used (archive_log
key dt_minutes), as in Sect. 5.4 of the data paper.

    python -m tools.analysis.pairing_check --log $FLOE_ROOT/data_v2/archive_log.jsonl \
        --scenes $FLOE_ROOT/logs/polygons_v2.npz --a4a $FLOE_ROOT/ai4arctic/rtt
"""
import argparse
import glob
import json
import os.path as osp
import re
from datetime import datetime

import numpy as np

OID = re.compile(r'S1[ABCD]_EW_GRDM_1SD[HV]_(\d{8}T\d{6})_(\d{8}T\d{6})_.*?_icechart_dmi_(\d{12})_')


def ours(log, scenes=None):
    keep = None
    if scenes:
        keep = set(str(s) for s in np.load(scenes, allow_pickle=False)['scenes'])
    gaps = {}
    for line in open(log):
        r = json.loads(line)
        if r.get('status') == 'ok' and (keep is None or r['product'] in keep):
            gaps[r['product']] = max(r['dt_minutes'])
    return np.array(list(gaps.values()))


def autoice(root):
    import h5py
    seen, gaps = set(), []
    for p in sorted(glob.glob(osp.join(root, '**', '*.nc'), recursive=True)):
        if p.endswith('_reference.nc'):       # the test scenes come twice
            continue
        with h5py.File(p, 'r') as f:
            oid = f.attrs.get('original_id')
        oid = oid.decode() if isinstance(oid, bytes) else str(oid)
        m = OID.match(oid)
        if not m or oid in seen:              # Canadian charts or a repeated scene
            continue
        seen.add(oid)
        a, b = (datetime.strptime(m.group(i), '%Y%m%dT%H%M%S') for i in (1, 2))
        t = datetime.strptime(m.group(3), '%Y%m%d%H%M')
        gaps.append(0.0 if a <= t <= b else min(abs((t - a).total_seconds()), abs((t - b).total_seconds())) / 60)
    return np.array(gaps)


def show(name, g):
    print(f'{name}: {len(g)} scenes; gap median {np.median(g):.1f}, 95th percentile {np.percentile(g, 95):.1f}, '
          f'largest {g.max():.1f} min')
    print('   above 1 / 5 / 10 / 15 / 19 min: ' +
          ' / '.join(f'{int((g > t).sum())} ({(g > t).mean():.1%})' for t in (1, 5, 10, 15, 19)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--log', required=True, help='archive_log.jsonl of tools.build.run_archive')
    ap.add_argument('--scenes', default=None, help='polygon table (.npz with key scenes) to restrict the scenes')
    ap.add_argument('--a4a', required=True, help='directory of the ready-to-train AutoICE files')
    args = ap.parse_args()
    show('ours (largest gap of a scene)', ours(args.log, args.scenes))
    show('AutoICE, DMI scenes', autoice(args.a4a))


if __name__ == '__main__':
    main()
