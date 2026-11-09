#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU check of floe.loaders.degrade_resolution_ (resolution experiment). No data.

  [A] k=1 is a no-op.
  [B] every valid pixel equals the masked mean of its k x k block (brute force);
      invalid pixels, the mask channel and nothing else change.
  [C] chunk-wise application on a numpy view (on-demand validation path, 512-px
      chunks) is bit-identical to whole-scene application (whole-scene path).
  [D] odd sizes (ceil blocks at the edges) work.
"""
import numpy as np
import torch

from floe.loaders import degrade_resolution_

V = ['nersc_sar_primary', 'nersc_sar_secondary', 'sar_incidenceangle', 'global_valid_mask']


def scene(H, W, seed=0):
    g = np.random.default_rng(seed)
    x = g.normal(size=(4, H, W)).astype(np.float32)
    m = (g.random((H, W)) > 0.2).astype(np.float32)
    m[:, :37] = 0                                   # a black border
    x[3] = m
    x[:3][:, m == 0] = 0                            # invalid pixels are 0 in the files
    return x


def brute(x, k):
    out = x.copy()
    H, W = x.shape[1:]
    m = x[3] > 0.5
    for i in range(0, H, k):
        for j in range(0, W, k):
            mb = m[i:i + k, j:j + k]
            if not mb.any():
                continue
            for c in range(3):
                blk = x[c, i:i + k, j:j + k]
                val = blk[mb].astype(np.float64).mean()
                o = out[c, i:i + k, j:j + k]
                o[mb] = val
    return out


def main():
    x = scene(64, 80)
    a = x.copy()
    degrade_resolution_(a, V, 1)
    assert np.array_equal(a, x)
    print('[A] PASS k=1 no-op')

    for k in (2, 4):
        b = x.copy()
        degrade_resolution_(b, V, k)
        ref = brute(x, k)
        assert np.allclose(b, ref, atol=1e-5), np.abs(b - ref).max()
        assert np.array_equal(b[3], x[3]), 'mask changed'
        inval = x[3] == 0
        assert np.array_equal(b[:3][:, inval], x[:3][:, inval]), 'invalid pixels changed'
    print('[B] PASS masked block means exact; mask and invalid pixels untouched (k=2,4)')

    H, W, C = 1300, 1100, 512
    big = scene(H, W, seed=1)
    whole = torch.from_numpy(big.copy()).unsqueeze(0)
    degrade_resolution_(whole, V, 2)
    chunked = big.copy()
    for y0 in range(0, H, C):
        for x0 in range(0, W, C):
            degrade_resolution_(chunked[:, y0:y0 + C, x0:x0 + C], V, 2)
    assert np.array_equal(whole[0].numpy(), chunked), 'chunked != whole-scene'
    print('[C] PASS chunk-wise (on-demand path) bit-identical to whole-scene path')

    odd = scene(33, 47, seed=2)
    o2 = odd.copy()
    degrade_resolution_(o2, V, 2)
    assert np.allclose(o2, brute(odd, 2), atol=1e-5)
    print('[D] PASS odd sizes')
    print('ALL PASS')


if __name__ == '__main__':
    main()
