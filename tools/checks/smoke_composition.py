#!/usr/bin/env python
"""CPU checks for chart-composition supervision (floe/composition.py). No real data.

  [A] SIGRID-3 concentration codes and egg code -> composition.
  [B] CompositionLoss: gradient descent on free logits drives every polygon mean to its
      chart composition; pure_only masks pixels outside pure polygons.
  [C] polygon histograms and composition metrics on a hand-made case.
  [D] training loader on a synthetic scene whose polygon id equals the label: after
      flips, rot90, affine and cutmix, '_bag' still lines up with the FLOE labels.
"""
import copy
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

from floe.composition import (CompositionLoss, composition_metrics, egg_to_composition,
                              polygon_histograms, sigrid_concentration)


def check_codes():
    assert sigrid_concentration(92) == 1.0 and sigrid_concentration(91) == 0.95
    assert sigrid_concentration(40) == 0.4 and sigrid_concentration(13) == 0.2
    assert sigrid_concentration(0) == 0.0 and np.isnan(sigrid_concentration(99))
    # 9/10 total: 6/10 medium (4), 3/10 small (3)
    q, u = egg_to_composition(0.9, [(0.6, 4), (0.3, 3)])
    assert np.allclose(q, [0.1, 0, 0.3, 0.6, 0, 0]) and u == 0
    # single type, partial not given -> CT
    q, u = egg_to_composition(1.0, [(np.nan, 6)])
    assert np.allclose(q, [0, 0, 0, 0, 0, 1]) and u == 0
    # undetermined / pancake forms become unknown mass
    q, u = egg_to_composition(0.8, [(0.5, 3), (0.3, 22)])
    assert np.allclose(q, [0.2, 0, 0.5, 0, 0, 0]) and np.isclose(u, 0.3)
    print('[A] egg codes OK')


def check_loss():
    torch.manual_seed(0)
    Q = np.array([[0.1, 0, 0.3, 0.6, 0, 0],      # mixed
                  [0, 0, 0, 0, 0.05, 0.95],      # pure vast
                  [0.2, 0, 0.5, 0, 0, 0]], np.float32)
    U = np.array([0, 0, 0.3], np.float32)
    loss_fn = CompositionLoss(Q, U, min_pixels=1)
    bag = torch.full((2, 16, 16), -1, dtype=torch.long)
    bag[0, :, :8], bag[0, :, 8:], bag[1] = 0, 1, 2
    logits = torch.zeros(2, 6, 16, 16, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        loss, _ = loss_fn(logits, bag)
        loss.backward()
        opt.step()
    p = torch.softmax(logits.detach(), 1)
    m0 = p[0, :, :, :8].mean((1, 2))
    m1 = p[0, :, :, 8:].mean((1, 2))
    m2 = p[1].mean((1, 2))
    assert torch.allclose(m0, torch.tensor(Q[0]), atol=0.03), m0
    assert torch.allclose(m1, torch.tensor(Q[1]), atol=0.03), m1
    assert abs(float(m2[0]) - 0.2) < 0.03 and float(m2[2]) > 0.45, m2   # unknown goes to ice
    y = torch.full((2, 16, 16), 3, dtype=torch.long)
    y_pure = loss_fn.pure_only(y, bag)
    assert (y_pure[0, :, 8:] == 3).all() and (y_pure[0, :, :8] == 255).all() and (y_pure[1] == 255).all()
    # ice_only: the floe-size mix of the ice is matched, the water share is left free
    loss_ice = CompositionLoss(Q, U, min_pixels=1, ice_only=True)
    logits = torch.zeros(2, 6, 16, 16)
    logits[:, 0] = 3.0                                     # mostly water
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        loss, _ = loss_ice(logits, bag)
        loss.backward()
        opt.step()
    p = torch.softmax(logits.detach(), 1)
    for m, q in ((p[0, :, :, :8].mean((1, 2)), Q[0]), (p[0, :, :, 8:].mean((1, 2)), Q[1])):
        q = torch.tensor(q)
        if q[1:].sum() > 0:
            assert torch.allclose(m[1:] / m[1:].sum(), q[1:] / q[1:].sum(), atol=0.03), m
    print('[B] composition loss OK')


def check_metrics():
    poly = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    pred = np.array([[3, 2, 5, 5], [3, 3, 5, 4]])
    ids, hist = polygon_histograms(pred, poly)
    assert ids.tolist() == [0, 1] and hist[0].tolist() == [0, 0, 1, 3, 0, 0]
    q = np.array([[0, 0, 0.25, 0.75, 0, 0], [0, 0, 0, 0, 0, 1.0]])
    m = composition_metrics(hist, q, np.zeros(2), min_pixels=1)
    assert np.isclose(m['tv_poly'], 0.125) and m['dominant_agree_pure'] == 1.0
    assert m['dominant_agree_mixed'] == 1.0
    print('[C] polygon metrics OK')


def check_loader():
    from mmcv import Config
    from floe.loaders import AI4ArcticChallengeDataset, get_variable_options

    tmp = tempfile.mkdtemp(dir=os.environ.get('FLOE_LOG_DIR'))   # stay inside the project area
    data, charts = osp.join(tmp, 'data'), osp.join(tmp, 'charts')
    os.makedirs(data); os.makedirs(charts)
    H, W, name = 1024, 1536, 'S1A_EW_GRDM_1SDH_20230101T000000_synthetic.nc'
    g = np.random.default_rng(0)
    floe = np.zeros((H, W), np.uint8)
    for r in range(0, H, 128):                      # blocks of classes 0..5
        for c in range(0, W, 128):
            floe[r:r + 128, c:c + 128] = (r // 128 + c // 128) % 6
    with h5py.File(osp.join(data, name), 'w') as f:
        for v in ('nersc_sar_primary', 'nersc_sar_secondary', 'sar_incidenceangle'):
            f.create_dataset(v, data=g.normal(size=(H, W)).astype(np.float32), chunks=(512, 512))
        f.create_dataset('global_valid_mask', data=np.ones((H, W), np.uint8), chunks=(512, 512))
        f.create_dataset('FLOE', data=floe, chunks=(512, 512))
    # polygon id == label, so after any geometric op the two channels must agree
    with h5py.File(osp.join(charts, name[:-3] + '.chart.h5'), 'w') as f:
        f.create_dataset('polygon_id', data=floe.astype(np.int32), chunks=(512, 512))
        f.create_dataset('composition', data=np.eye(6, dtype=np.float32))
        f.create_dataset('unknown', data=np.zeros(6, np.float32))
    index = osp.join(tmp, 'index.npz')
    subprocess.run([sys.executable, '-m', 'tools.data.scan_tiles', '--data-dir', data, '--out', index,
                    '--patch-size', '256', '--workers', '1'], check=True, stdout=subprocess.DEVNULL)

    cfg = Config.fromfile('configs/FLOE/res80_noglcm.py').train_options
    opts = get_variable_options(copy.deepcopy(cfg))
    opts.update(path_to_train_data=data + '/', tile_index_path=index, batch_size=8, num_workers=0,
                composition_supervision=dict(enabled=True, chart_dir=charts),
                floe_class_merge=None)
    da = opts['data_augmentations']
    da.update(Random_h_flip=0.5, Random_v_flip=0.5, Random_rot90_prob=0.5, Random_rotation=20,
              Random_rotation_prob=0.5, Cutmix_prob=1.0)
    ds = AI4ArcticChallengeDataset(options=opts, files=[name], do_transform=True)
    checked = 0
    for _ in range(10):
        x, y = ds[0]
        lab, bag = y['FLOE'], y['_bag']
        both = (lab != 255) & (bag >= 0)
        assert both.sum() > 0
        assert torch.equal(lab[both], bag[both]), 'bag ids are misaligned with labels'
        checked += int(both.sum())
    print(f'[D] loader alignment OK ({checked} labelled pixels checked over 10 batches)')
    shutil.rmtree(tmp)


if __name__ == '__main__':
    check_codes()
    check_loss()
    check_metrics()
    check_loader()
    print('PASS')
