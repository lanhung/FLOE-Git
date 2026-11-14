"""Time between each test scene and the nearest training scene, from the file names.

For the AutoICE ready-to-train lists (<start time>_<service>_prep.nc) this shows how
close the 20 test scenes are to the 512 training scenes. Time only: whether two scenes
close in time also overlap on the ground is not checked here.

The same gaps are given for the training scenes among themselves (each against all
other training scenes of its service). That is the base rate: a test set drawn from the
days of the training set is expected to be as close to it as that.

    python -m tools.analysis.test_distance --train datalists/ai4arctic/train.json \
        datalists/ai4arctic/val.json --test datalists/ai4arctic/test.json
"""
import argparse
import json
from datetime import datetime

import numpy as np


def parse(name):
    stamp, service = name.split('_')[:2]
    return datetime.strptime(stamp, '%Y%m%dT%H%M%S'), service


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--test', required=True)
    args = ap.parse_args()

    train = [parse(n) for f in args.train for n in json.load(open(f))]
    test = sorted(json.load(open(args.test)))
    same, anyone = [], []
    for name in test:
        t, service = parse(name)
        gaps = np.array([abs((t - u).total_seconds()) / 3600 for u, _ in train])
        mine = np.array([s == service for _, s in train])
        same.append(gaps[mine].min())
        anyone.append(gaps.min())
        print(f'{name:32s} nearest training scene: same service {same[-1]:8.2f} h, any {anyone[-1]:8.2f} h')
    same, anyone = np.array(same), np.array(anyone)
    services = np.array([parse(n)[1] for n in test])

    # Base rate: every training scene against the other training scenes of its service.
    hours_train = np.array([(t - train[0][0]).total_seconds() / 3600 for t, _ in train])
    service_train = np.array([s for _, s in train])
    among = np.full(len(train), np.inf)
    for s in set(service_train):
        idx = np.flatnonzero(service_train == s)
        gap = np.abs(hours_train[idx][:, None] - hours_train[idx][None, :])
        np.fill_diagonal(gap, np.inf)
        among[idx] = gap.min(1)

    print(f'\n{len(test)} test scenes, {len(train)} training scenes')
    print('  nearest scene of the same service:   test    training (base rate)    [test, any service]')
    for label, hours in (('3 min', 0.05), ('1 h', 1), ('24 h', 24), ('72 h', 72), ('7 d', 168)):
        # the base rate in the service mix of the test scenes
        expected = sum((among[service_train == s] <= hours).mean() * (services == s).sum() for s in set(services))
        print(f'  within {label:5s}: {(same <= hours).sum():2d} of {len(test)} ({(same <= hours).mean():4.0%})'
              f'   {(among <= hours).mean():4.0%} of training scenes, {expected:4.1f} of {len(test)} expected'
              f'   [{(anyone <= hours).sum():2d}]')
    for s in sorted(set(service_train)):
        a, b = among[service_train == s], same[services == s]
        print(f'  {s}: {len(b)} test and {len(a)} training scenes; within 3 min / 24 h / 72 h: '
              f'test {(b <= 0.05).sum()} / {(b <= 24).sum()} / {(b <= 72).sum()}, '
              f'training {(a <= 0.05).mean():.0%} / {(a <= 24).mean():.0%} / {(a <= 72).mean():.0%}')
    print(f'  median gap: test {np.median(same):.1f} h, training {np.median(among):.1f} h '
          f'(test, any service: {np.median(anyone):.1f} h)')


if __name__ == '__main__':
    main()
