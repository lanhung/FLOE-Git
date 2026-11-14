"""Scenes shared by ASID-v2 and the AI4Arctic challenge data, from the file names.

Both name a scene by the start time of the Sentinel-1 product. The ASID-v2 names come from
the public record of the dataset, saved with

    curl -o asid_v2.json https://api.figshare.com/v2/articles/13011134

    python -m tools.analysis.public_overlap --asid asid_v2.json \
        --ai4arctic datalists/ai4arctic/train.json datalists/ai4arctic/val.json datalists/ai4arctic/test.json
"""
import argparse
import json
from collections import Counter
from datetime import datetime

FMT = '%Y%m%dT%H%M%S'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--asid', required=True, help='figshare record of ASID-v2 (JSON)')
    ap.add_argument('--ai4arctic', nargs='+', required=True, help='scene lists of the challenge data (JSON)')
    ap.add_argument('--seconds', type=float, default=2, help='start times this close count as one scene')
    args = ap.parse_args()

    asid = sorted(f['name'] for f in json.load(open(args.asid))['files'] if f['name'].endswith('.nc'))
    asid_t = [datetime.strptime(n[:15], FMT) for n in asid]
    a4a = sorted(n for f in args.ai4arctic for n in json.load(open(f)))
    a4a_t = [datetime.strptime(n[:15], FMT) for n in a4a]
    service = Counter(n.split('_')[1] for n in a4a)

    shared = [(n, m) for n, t in zip(asid, asid_t) for m, u in zip(a4a, a4a_t)
              if abs((t - u).total_seconds()) <= args.seconds]
    shared_service = Counter(m.split('_')[1] for _, m in shared)
    inside = sum(min(asid_t) <= u <= max(asid_t) and n.split('_')[1] == 'dmi' for n, u in zip(a4a, a4a_t))

    print(f'ASID-v2: {len(asid)} scenes, {min(asid_t):%Y-%m-%d} to {max(asid_t):%Y-%m-%d}')
    print(f'challenge data: {len(a4a)} scenes ({dict(service)}), {min(a4a_t):%Y-%m-%d} to {max(a4a_t):%Y-%m-%d}')
    print(f'shared: {len(shared)} ({dict(shared_service)}); '
          f'{inside} DMI scenes of the challenge data lie in the ASID-v2 period')
    print(f'together: {len(asid) + len(a4a) - len(shared)} scenes, '
          f'{len(asid) + service["dmi"] - shared_service["dmi"]} of them with DMI charts')


if __name__ == '__main__':
    main()
