#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check that the training data stream is identical across model variants.

The ablations need every config to see the same tiles, crops, flips and cutmix boxes in
the same order under one seed. Each config is built via tools.train.build_seeded_model and
the first N train batches are hashed; PASS if all match. --legacy uses the old seeding (no
re-seed after the model is built), which is expected to FAIL across configs with different
modules. Needs data: the train list may only contain scenes on disk.

  python -m tools.checks.seed_stream CFG_A CFG_B [...] [--batches 3] [--workers 2] [--legacy]
"""
import argparse
import copy
import hashlib
import sys

import torch
from mmcv import Config

from floe.functions import create_train_validation_and_test_scene_list
from floe.loaders import get_variable_options, limit_torch_threads
from tools.train import build_seeded_model, create_dataloaders


def stream_hashes(cfg_path, n_batches, workers, seed, legacy, device, charts=None):
    opts = get_variable_options(copy.deepcopy(Config.fromfile(cfg_path).train_options))
    opts['_resolved_seed'] = seed
    opts['num_workers'] = workers
    opts['compile_model'] = False
    opts['_legacy_rng'] = legacy
    build_seeded_model(opts, device, reseed=not legacy)
    create_train_validation_and_test_scene_list(opts)
    dl_train, _ = create_dataloaders(opts)
    out = []
    for i, (bx, by) in enumerate(dl_train):
        h = hashlib.sha256()
        h.update(bx.contiguous().numpy().tobytes())
        for c in sorted(by):
            if charts is None or c in charts:   # --charts: compare only targets all configs share
                h.update(by[c].contiguous().numpy().tobytes())
        out.append(h.hexdigest()[:16])
        if i + 1 >= n_batches:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--batches', type=int, default=3)
    ap.add_argument('--charts', nargs='*', default=None,
                    help='hash only these label charts besides the inputs (e.g. FLOE when the configs differ in '
                         'auxiliary targets such as SIC/SOD or composition bags)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--legacy', action='store_true')
    args = ap.parse_args()
    assert len(args.configs) >= 2, 'need at least two configs to compare'
    limit_torch_threads()
    device = torch.device('cpu')   # model is only built, never run

    res = {}
    for c in args.configs:
        res[c] = stream_hashes(c, args.batches, args.workers, args.seed,
                               args.legacy, device, charts=args.charts)
        print(f'{c}: {" ".join(res[c])}', flush=True)
    ref = res[args.configs[0]]
    same = all(v == ref for v in res.values())
    mode = 'old behaviour (--legacy)' if args.legacy else 'fixed protocol'
    print(f'\n{"PASS" if same else "FAIL"}: data stream {"identical" if same else "DIFFERS"} '
          f'across {len(res)} configs ({mode}, seed {args.seed}, {args.batches} batches)')
    sys.exit(0 if same else 1)


if __name__ == '__main__':
    main()
