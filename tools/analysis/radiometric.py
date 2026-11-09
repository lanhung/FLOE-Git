#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""辐射扰动探针：在已有 checkpoint 上扰动 val 的 SAR 辐射水平，看 hard-ice 退化多少（无需训练）。

z 空间（默认 VARIANTS 或 --variants）：有效像元内 HH/HV ← a·x + b（z-score 单位；b 为定标偏置，
a 为对比度）。HH 与 HV 的 σ 不同（8.33 / 9.45 dB），b=0.30 ≈ +2.50 dB(HH) / +2.83 dB(HV)。
dB 空间（--db-shifts，论文口径）：HH 与 HV 同加 d dB，x' = clip(z·σ+μ+d, −35, 5)，再按 μ/σ 归一化；
μ/σ 取自 NC 属性 zscore_mean/zscore_std 并逐景核对（或 --zscore-stats）。
两种模式都从扰动后的 HH/HV 重算 pol_ratio；入射角、GLCM、mask 不动，结果是捷径依赖的下界。
推理走 fast_tiled_val_predict（对齐 tile + 门控 + 按需读）；--dump-cm 每个变体一个文件，格式同
tools.evaluate --dump-cm。

  python -m tools.analysis.radiometric --self-test
  python -m tools.analysis.radiometric --config <cfg> --checkpoint <ckpt> \
      --val-list datalists/cv5/fold0_val.json --db-shifts=-4,-2,-1,1,2,4 --dump-cm <out>.npz
"""
import argparse
import copy
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from mmcv import Config

from floe.loaders import (get_variable_options, val_scoped_options, AI4ArcticChallengeTestDataset,
                     limit_torch_threads, val_ondemand_ok)
from floe.functions import get_model, fast_tiled_val_predict

# (标签, a 乘性增益=对比度, b 加性偏置=定标漂移[z-score 单位])
VARIANTS = [
    ('id',      1.00,  0.00),   # 恒等参照
    ('b+0.15',  1.00,  0.15),   # ≈ +1.25 dB(HH) / +1.42 dB(HV)
    ('b-0.15',  1.00, -0.15),
    ('b+0.30',  1.00,  0.30),   # ≈ +2.50 dB(HH) / +2.83 dB(HV)
    ('b-0.30',  1.00, -0.30),
    ('a0.90',   0.90,  0.00),   # 对比度 −10%
    ('a1.10',   1.10,  0.00),   # 对比度 +10%
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--checkpoint')
    p.add_argument('--val-list')
    p.add_argument('--dump-cm', default=None, help='逐景 CM 输出 .npz 前缀（每变体一个文件）')
    p.add_argument('--self-test', action='store_true', help='只跑扰动逻辑自检后退出')
    # 如 "id:1:0,b-0.70:1:-0.70"；省略时用 VARIANTS。--self-test 始终用完整 VARIANTS。
    p.add_argument('--variants', default=None,
                   help='覆盖扰动清单："tag:a:b" 逗号分隔；须含 id:1:0')
    p.add_argument('--db-shifts', default=None,
                   help='dB 空间共模偏移清单（dB），如 "-4,-2,2,4"；给出时替代 z 空间清单，'
                        '自动加 id 参照')
    p.add_argument('--zscore-stats', default=None,
                   help='zscore_stats.json（全局 mean/std）；缺省时从各景 NC 变量属性 '
                        'zscore_mean/zscore_std 读取（417 景归档每个文件都带）')
    p.add_argument('--check-nc-attrs', action='store_true',
                   help='给了 --zscore-stats 时，与首景 NC 属性对拍')
    return p.parse_args()


DB_CLIP = (-35.0, 5.0)   # 归一化前的 dB 截断区间（DATASET_HANDBOOK §8）


def load_zscore_stats(path, var_hh='nersc_sar_primary', var_hv='nersc_sar_secondary'):
    """读 zscore_stats.json，返回 {'hh': (mean, std), 'hv': (mean, std)}。
    兼容两种常见结构：{var: {mean, std}} 与 {'mean': {var: ..}, 'std': {var: ..}}。"""
    d = json.load(open(path))

    def get(var):
        if var in d and isinstance(d[var], dict):
            e = d[var]
            return float(e.get('mean', e.get('zscore_mean'))), float(e.get('std', e.get('zscore_std')))
        if 'mean' in d and 'std' in d:
            return float(d['mean'][var]), float(d['std'][var])
        raise KeyError(f'{path}: 找不到 {var} 的 mean/std，结构未知：{list(d)[:8]}')
    return {'hh': get(var_hh), 'hv': get(var_hv)}


def nc_zscore_stats(nc_path, var_hh='nersc_sar_primary', var_hv='nersc_sar_secondary'):
    """从 NC 变量属性读 {'hh': (mean, std), 'hv': (mean, std)}。"""
    import h5py
    out = {}
    with h5py.File(nc_path, 'r') as h:
        for key, var in (('hh', var_hh), ('hv', var_hv)):
            a = h[var].attrs
            if 'zscore_mean' not in a or 'zscore_std' not in a:
                raise KeyError(f'{nc_path}:{var} 无 zscore_mean/zscore_std 属性')
            out[key] = (float(np.asarray(a['zscore_mean']).reshape(-1)[0]),
                        float(np.asarray(a['zscore_std']).reshape(-1)[0]))
    return out


def check_stats_against_nc(stats, nc_path, var_hh='nersc_sar_primary',
                           var_hv='nersc_sar_secondary', rtol=1e-4):
    nc = nc_zscore_stats(nc_path, var_hh, var_hv)
    for key, var in (('hh', var_hh), ('hv', var_hv)):
        m, sd = nc[key]
        if not (np.isclose(m, stats[key][0], rtol=rtol) and np.isclose(sd, stats[key][1], rtol=rtol)):
            raise ValueError(f'统计量不一致 {var}: json=({stats[key][0]}, {stats[key][1]}) '
                             f'nc=({m}, {sd})')


def parse_db_shifts(spec):
    out = [('id', 1.0, 0.0)]
    for item in spec.split(','):
        d = float(item)
        if d == 0:
            continue
        out.append(('dB%+.1f' % d, 'db', d))
    return out


def apply_db_shift(x, hh0, hv0, valid, d, stats, pol_idx):
    """有效像元内：HH/HV 同加 d dB（先反归一化、再按 [-35,5] dB 截断、再归一化），
    并从扰动后的两通道重算 pol_ratio。返回 (HH 截断比例, HV 截断比例)。"""
    lo, hi = DB_CLIP
    clip_frac = []
    for ch, orig, key in ((0, hh0, 'hh'), (1, hv0, 'hv')):
        mu, sd = stats[key]
        db = orig * sd + mu + d
        clipped = ((db < lo) | (db > hi)) & valid
        n_valid = max(int(valid.sum()), 1)
        clip_frac.append(float(clipped.sum()) / n_valid)
        z = (db.clamp(lo, hi) - mu) / sd
        x[:, ch:ch + 1] = torch.where(valid, z.to(x.dtype), orig)
    if pol_idx is not None:
        x[:, pol_idx:pol_idx + 1] = x[:, 0:1] - x[:, 1:2]
    return tuple(clip_frac)


def parse_variants(spec):
    """解析 "tag:a:b,..."。另认两个逐景自适应 tag（a/b 填 0，运行时按该景 valid 区统计现算）：
      center: 逐景居中，b_ch = −mean_valid(ch)，a=1
      zscene: 逐景标准化，a_ch = 1/std_valid(ch)，b_ch = −mean/std"""
    out = []
    for item in spec.split(','):
        tag, a, b = item.split(':')
        out.append((tag, float(a), float(b)))
    assert any(t == 'id' and a == 1.0 and b == 0.0 for t, a, b in out), \
        '--variants 必须包含参照项 id:1:0'
    return out


def channel_layout(opts):
    """返回 (hh_idx, hv_idx, mask_idx, pol_idx, n_ch)。pol_idx 为 None 表示无该通道。

    通道顺序由 loaders._augment_x 定死：train_variables → [month×2] → [pol_ratio] → [texture]。
    """
    tv = list(opts['train_variables'])
    assert tv[0] == 'nersc_sar_primary' and tv[1] == 'nersc_sar_secondary', \
        f'前两个通道应为 nersc SAR primary/secondary，实际 {tv[:2]}'
    assert 'global_valid_mask' in tv, '需要 global_valid_mask 来把扰动限制在有效像元内'
    assert not opts.get('texture_channels', False), \
        'texture_channels=True 时纹理通道是从 HH 派生的，本探针未实现其同步重算 —— 拒绝运行'
    n_ch = len(tv) + (2 if opts.get('month_encoding', False) else 0) \
                   + (1 if opts.get('pol_ratio_channel', False) else 0)
    pol_idx = (len(tv) + (2 if opts.get('month_encoding', False) else 0)) \
        if opts.get('pol_ratio_channel', False) else None
    return 0, 1, tv.index('global_valid_mask'), pol_idx, n_ch


def apply_radiometric(x, hh0, hv0, valid, a, b, pol_idx, a2=None, b2=None):
    """就地把 x 的 HH/HV 置为 a·orig + b（仅 valid 处），并从扰动后的两通道重算 pol_ratio。

    hh0/hv0 是原始通道的备份，每个变体都从原始出发，不跨变体累积浮点误差。
    a2/b2 是 HV 的独立系数，None 时与 HH 共用。
    """
    a2 = a if a2 is None else a2
    b2 = b if b2 is None else b2
    x[:, 0:1] = torch.where(valid, hh0 * a + b, hh0)
    x[:, 1:2] = torch.where(valid, hv0 * a2 + b2, hv0)
    if pol_idx is not None:
        x[:, pol_idx:pol_idx + 1] = x[:, 0:1] - x[:, 1:2]


def self_test():
    """扰动逻辑自检（不碰数据/GPU）：无效区不动、有效区精确仿射、pol_ratio 自洽、
    每个变体都从原始出发（无跨变体漂移）。"""
    torch.manual_seed(0)
    C = 8
    x = torch.randn(1, C, 16, 16)
    x[:, 6] = (torch.rand(1, 16, 16) > 0.3).float()      # ch6 = 0/1 mask
    x[:, 7] = x[:, 0] - x[:, 1]                          # ch7 = pol_ratio
    x_orig = x.clone()
    valid = x[:, 6:7] > 0.5
    hh0, hv0 = x[:, 0:1].clone(), x[:, 1:2].clone()

    for tag, a, b in VARIANTS + VARIANTS:   # 跑两轮，验证无跨变体漂移
        apply_radiometric(x, hh0, hv0, valid, a, b, 7)
        v = valid.squeeze(1)
        # 有效区：精确仿射
        assert torch.allclose(x[:, 0][v], x_orig[:, 0][v] * a + b, atol=1e-6), tag
        assert torch.allclose(x[:, 1][v], x_orig[:, 1][v] * a + b, atol=1e-6), tag
        # 无效区：逐位不动
        assert torch.equal(x[:, 0][~v], x_orig[:, 0][~v]), tag
        assert torch.equal(x[:, 1][~v], x_orig[:, 1][~v]), tag
        # pol_ratio 自洽：有效区 = a·原值（b 消掉），无效区 = 原值
        assert torch.allclose(x[:, 7][v], x_orig[:, 7][v] * a, atol=1e-5), tag
        assert torch.allclose(x[:, 7][~v], x_orig[:, 7][~v], atol=1e-6), tag
        # 未涉及的通道逐位不动
        for c in (2, 3, 4, 5, 6):
            assert torch.equal(x[:, c], x_orig[:, c]), f'{tag} ch{c} 被改动了'
    # id 变体必须把 x 完全还原
    apply_radiometric(x, hh0, hv0, valid, 1.0, 0.0, 7)
    assert torch.allclose(x, x_orig, atol=1e-6), 'id 变体未还原原始输入'
    print('[self-test] PASS 扰动逻辑：有效区精确仿射、无效区与其余通道逐位不动、'
          'pol_ratio 自洽、变体间无浮点漂移、id 可还原', flush=True)

    # dB 模式：已知统计量下的精确换算 + 截断 + pol 重算 + 无效区不动
    stats = {'hh': (-13.9018, 8.3349), 'hv': (-27.0105, 9.4454)}
    x = x_orig.clone()
    hh_db = x_orig[:, 0:1] * 8.3349 - 13.9018
    hv_db = x_orig[:, 1:2] * 9.4454 - 27.0105
    for d in (2.0, -4.0, 0.0):
        apply_db_shift(x, hh0, hv0, valid, d, stats, 7)
        v = valid.squeeze(1)
        exp_hh = ((hh_db + d).clamp(*DB_CLIP) + 13.9018) / 8.3349
        exp_hv = ((hv_db + d).clamp(*DB_CLIP) + 27.0105) / 9.4454
        assert torch.allclose(x[:, 0][v], exp_hh[:, 0][v], atol=1e-5), d
        assert torch.allclose(x[:, 1][v], exp_hv[:, 0][v], atol=1e-5), d
        assert torch.equal(x[:, 0][~v], x_orig[:, 0][~v]), d
        assert torch.allclose(x[:, 7], x[:, 0] - x[:, 1], atol=1e-6), d
        for c in (2, 3, 4, 5, 6):
            assert torch.equal(x[:, c], x_orig[:, c]), f'dB{d} ch{c} 被改动了'
    print('[self-test] PASS dB 模式：反归一化-偏移-截断-归一化精确、pol 重算、其余通道不动',
          flush=True)


def infer_pred(x, net, opts, labels, valid_maps):
    """一次整景推理，返回 (H,W) uint8 CPU 预测图（与训练验证同一条 fast_tiled_val_predict 路径）。

    被门控跳过的 tile 预测为 0，但这些像素打分时被 valid_maps 排除，不影响混淆矩阵。"""
    with torch.no_grad(), torch.cuda.amp.autocast():
        preds, _ = fast_tiled_val_predict(x, net, opts, labels=labels,
                                          valid_maps=valid_maps, gate=True)
    return preds['FLOE']


def cm_iou(cm, absent=0.0):
    """absent：GT 与预测都不含该类时的 IoU 取值。全局汇总用 0（与论文表一致，缺失会告警），
    逐景用 NaN。"""
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), absent)


def main():
    args = parse_args()
    limit_torch_threads()
    if args.self_test:
        self_test()
        return
    for req in ('config', 'checkpoint', 'val_list'):
        assert getattr(args, req), f'--{req.replace("_", "-")} 必填（或用 --self-test）'
    self_test()          # 真跑之前也过一遍，几乎零成本
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opts = get_variable_options(copy.deepcopy(Config.fromfile(args.config).train_options))
    assert opts['down_sample_scale'] == 1, '需原生分辨率验证口径'
    assert opts['model_selection'] not in ('swin', 'PA2'), \
        'swin/PA2 走重叠滑窗（batched_slide_inference），与本探针的对齐 tile 路径 + 按需读' \
        '不兼容 —— 拒绝运行，免得静默读到 0'
    n_cls = opts['n_classes']['FLOE']
    db_stats = None
    if args.db_shifts:
        assert not args.variants, '--db-shifts 与 --variants 互斥'
        if args.zscore_stats:
            db_stats = load_zscore_stats(args.zscore_stats)
        else:
            _first = json.load(open(args.val_list))[0]
            db_stats = nc_zscore_stats(osp.join(opts['path_to_train_data'], _first))
            print(f'[dB] 统计量取自 {_first} 的 NC 属性（每景还会逐一核对）', flush=True)
        variants = parse_db_shifts(args.db_shifts)
        (mh, sh), (mv, sv) = db_stats['hh'], db_stats['hv']
        print(f'[dB] stats HH μ={mh:.4f} σ={sh:.4f} dB | HV μ={mv:.4f} σ={sv:.4f} dB；'
              f'1 dB = {1/sh:.4f} z(HH) / {1/sv:.4f} z(HV)；'
              f'pol 通道每 dB 变化 {1/sh - 1/sv:+.4f}', flush=True)
        print(f'[variants] dB 清单：{variants}', flush=True)
    else:
        variants = parse_variants(args.variants) if args.variants else VARIANTS
        if args.variants:
            print(f'[variants] 使用覆盖清单：{variants}', flush=True)
    hh_i, hv_i, mask_i, pol_i, n_ch = channel_layout(opts)
    print(f'[layout] n_ch={n_ch} | HH=ch{hh_i} HV=ch{hv_i} mask=ch{mask_i} '
          f'pol_ratio={"ch%d" % pol_i if pol_i is not None else "无"}；'
          f'扰动只作用于 HH/HV/pol_ratio 且仅在 mask==1 内', flush=True)

    net = get_model(opts, device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    # 模型按完整 charts 建好并 strict load（多任务 checkpoint 含辅头）；此后读取、前向与指标
    # 只跑 target chart：val 景可能没有辅任务变量，且辅任务处处有效会让 tile 门控失效。
    opts = val_scoped_options(opts)

    files = json.load(open(args.val_list))
    if db_stats is not None and args.check_nc_attrs:
        check_stats_against_nc(db_stats, osp.join(opts['path_to_train_data'], files[0]))
    # 本探针不改 tile 网格、只在门控后的对齐 tile 上前向，满足按需读的前提；
    # 不满足时（或 config 关闭）退回整景读并打印原因。
    ok, why = val_ondemand_ok(opts, 1)
    ondemand = bool(opts.get('val_ondemand_read', True)) and ok
    print(f'[ondemand] {"启用" if ondemand else "退回整景读"}：{why}', flush=True)
    ds = AI4ArcticChallengeTestDataset(options=opts, files=files, mode='train',
                                       ondemand_read=ondemand)
    dl = torch.utils.data.DataLoader(ds, batch_size=None,
                                     num_workers=opts.get('num_workers_val', 0), shuffle=False)

    tags = [v[0] for v in variants]
    global_cm = {t: np.zeros((n_cls, n_cls), dtype=np.float64) for t in tags}
    scene_cms = {t: [] for t in tags}
    scene_ids = []

    for inf_x, inf_y, cfv_masks, tfv_mask, name, original_size in dl:
        sid = osp.splitext(str(name))[0]
        x = inf_x if inf_x.dim() == 4 else inf_x.unsqueeze(0)     # (1,C,H,W) CPU
        assert x.shape[1] == n_ch, f'{sid}: 通道数 {x.shape[1]} != 预期 {n_ch}'
        H, W = x.shape[-2:]
        gt = np.squeeze(np.asarray(inf_y['FLOE'])).astype(np.int64)
        m = gt != 255

        if db_stats is not None and not args.zscore_stats:
            # 归一化必须是全局的：逐景核对属性，任何一景不同即中止（否则 dB 换算失真）
            check_stats_against_nc(db_stats, osp.join(opts['path_to_train_data'], str(name)))
        valid = x[:, mask_i:mask_i + 1] > 0.5
        hh0, hv0 = x[:, hh_i:hh_i + 1].clone(), x[:, hv_i:hv_i + 1].clone()
        # 门控/打分掩膜同训练验证：valid = ~cfv_masks
        valid_maps = {c: ~cfv_masks[c] for c in opts['charts']}
        line = []
        for tag, a, b in variants:
            if tag in ('center', 'zscene'):
                # 逐景自适应：从该景 valid 区的原始统计现算（hh0/hv0 是原始备份）
                m_hh = float(hh0[valid].mean()); m_hv = float(hv0[valid].mean())
                if tag == 'center':
                    apply_radiometric(x, hh0, hv0, valid, 1.0, -m_hh, pol_i,
                                      a2=1.0, b2=-m_hv)
                else:
                    s_hh = max(float(hh0[valid].std()), 1e-6)
                    s_hv = max(float(hv0[valid].std()), 1e-6)
                    apply_radiometric(x, hh0, hv0, valid,
                                      1.0 / s_hh, -m_hh / s_hh, pol_i,
                                      a2=1.0 / s_hv, b2=-m_hv / s_hv)
            elif a == 'db':
                fh, fv = apply_db_shift(x, hh0, hv0, valid, b, db_stats, pol_i)
                if max(fh, fv) > 0.01:
                    line.append('[clip %s HH%.1f%% HV%.1f%%]' % (tag, 100 * fh, 100 * fv))
            else:
                apply_radiometric(x, hh0, hv0, valid, a, b, pol_i)
            pred = infer_pred(x, net, opts, inf_y, valid_maps).numpy().astype(np.int64)
            cm = np.bincount(gt[m] * n_cls + pred[m], minlength=n_cls * n_cls
                             ).reshape(n_cls, n_cls).astype(np.float64)
            global_cm[tag] += cm
            scene_cms[tag].append(cm)
            iou = cm_iou(cm, absent=np.nan)
            # 景内不存在（GT 与预测均无）的类记 NaN，否则逐景 hard-ice 只是在数该景有几类。
            hi_ = [iou[2], iou[3]]
            line.append('%s=%s' % (tag, '%.3f' % np.nanmean(hi_) if not np.all(np.isnan(hi_)) else 'nan'))
            torch.cuda.empty_cache()
        del hh0, hv0, valid, x
        scene_ids.append(sid)
        print('%-44s hard-ice %s  (%dx%d)' % (sid[:44], ' '.join(line), H, W), flush=True)

    # 汇总
    order = np.argsort(scene_ids)
    sorted_ids = np.array(scene_ids)[order]
    base_iou = cm_iou(global_cm['id'])
    _support = global_cm['id'].sum(1)
    if (_support == 0).any():
        print(f'[WARN] 评估集中无 GT 像元的类：{np.where(_support == 0)[0].tolist()}；'
              f'其 IoU 记 0 并计入 macro', flush=True)
    base_hard = float(np.mean([base_iou[2], base_iou[3]]))
    print('\n' + '=' * 96, flush=True)
    print('%-9s %8s %8s %9s | %s' % ('variant', 'macro', 'hard-ice', 'Δhard',
                                     ' '.join('%6s' % c for c in
                                              ('water', 'cake', 'small', 'medium', 'big', 'vast'))),
          flush=True)
    print('=' * 96, flush=True)
    for tag, a, b in variants:
        iou = cm_iou(global_cm[tag])
        hard = float(np.mean([iou[2], iou[3]]))
        print('%-9s %8.4f %8.4f %+9.4f | %s' %
              (tag, iou.mean(), hard, hard - base_hard,
               ' '.join('%6.3f' % v for v in iou)), flush=True)
        if args.dump_cm:
            root, ext = osp.splitext(args.dump_cm)
            os.makedirs(osp.dirname(args.dump_cm) or '.', exist_ok=True)
            np.savez('%s_%s%s' % (root, tag, ext or '.npz'),
                     scenes=sorted_ids, cm=np.stack(scene_cms[tag])[order])
    print('=' * 96, flush=True)
    print('参照 id: macro=%.4f hard-ice=%.4f（%d 景）' %
          (base_iou.mean(), base_hard, len(scene_ids)), flush=True)
    print('读法：Δhard 明显负且随 |b| 单调加深 ⇒ 模型依赖绝对辐射水平（H1 成立，不变性线有靶子）；'
          'Δhard ≈ 0 ⇒ 捷径不存在（H1 否掉）。参考噪声带：baseline 两 seed 的 hard-ice 差 0.005。',
          flush=True)
    print('RADIO_EVAL_DONE', flush=True)


if __name__ == '__main__':
    main()
    sys.exit(0)
