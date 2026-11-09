#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冒烟测试：在启动训练之前验证模型 config 能端到端跑通。有 GPU 用 GPU，否则用 CPU（只验证正确性）。

检查 config 合并、模型实例化、真实 tile batch 的通道数、前向 + 反向 + AMP（与训练循环一致）
的 loss 有限、输出为 {'FLOE': (B, 6, H, W)}、验证推理与混淆矩阵累加，以及 tile 门控 /
confmat 模式与全量前向路径的等价性（[7a]-[7d]）。

用法：
    python -m tools.checks.smoke_pipeline [CFG ...]     # 不给参数则测 DEFAULT_CONFIGS
"""
import contextlib
import copy
import json
import math
import sys
import time
import warnings

import torch
from mmcv import Config

from floe.loaders import (get_variable_options, val_scoped_options, AI4ArcticChallengeDataset,
                     limit_torch_threads)
from floe.functions import (get_model, get_optimizer, get_loss, fast_tiled_val_inference,
                       fast_tiled_val_predict, compute_valid_tile_mask,
                       batched_slide_inference, class_decider,
                       new_confusion_matrix, accumulate_confusion_matrix)

DEFAULT_CONFIGS = ['configs/FLOE/sota_segformer_in1k_dlr_417.py', 'configs/FLOE/pgsg.py']


def smoke_one(cfg_path, device, cuda):
    print('=' * 72, flush=True)
    print('SMOKE:', cfg_path, flush=True)
    print('=' * 72, flush=True)

    opts = get_variable_options(Config.fromfile(cfg_path).train_options)
    opts['diag_log_every_batches'] = 0
    # num_workers_val 也要置 0：swin 的 batched_slide_inference 内部会起 DataLoader。
    opts['num_workers'] = 0
    opts['num_workers_val'] = 0

    # 辅助任务（aux_charts）只用于训练，验证只跑 target chart；无 aux_charts 时 opts_val 就是 opts。
    opts_val = val_scoped_options(opts)

    target = opts.get('target_chart', 'FLOE')
    n_cls = opts['n_classes'][target]
    n_tex = 0
    if opts.get('texture_channels', False):
        from floe.loaders import N_TEXTURE_CHANNELS  # 训练 _augment_x 追加 N 个纹理通道
        n_tex = N_TEXTURE_CHANNELS
    exp_in_ch = (len(opts['train_variables'])
                 + (2 if opts.get('month_encoding', False) else 0)
                 + (1 if opts.get('pol_ratio_channel', False) else 0)
                 + n_tex)
    print(f'model_selection = {opts["model_selection"]} | target_chart = {target} | '
          f'n_classes = {n_cls} | expected in_channels = {exp_in_ch} '
          f'(pol_ratio={opts.get("pol_ratio_channel", False)}, '
          f'texture={opts.get("texture_channels", False)})', flush=True)

    # [2] 建模型
    net = get_model(opts, device)
    n_param = sum(p.numel() for p in net.parameters())
    print(f'[2] model built OK | params = {n_param / 1e6:.2f} M', flush=True)

    # [3-5] 真实 batch 前向/反向
    net.train()
    optimizer = get_optimizer(opts, net)
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    loss_fn = {c: get_loss(opts['chart_loss'][c]['type'], chart=c, **opts['chart_loss'][c]).to(device)
               for c in opts['charts']}
    grad_clip = opts.get('grad_clip_max_norm', 5.0)

    train_list = json.load(open(opts['path_to_env'] + opts['train_list_path']))
    files = [f if f.endswith('.nc') else f + '.nc' for f in train_list]
    ds = AI4ArcticChallengeDataset(opts, files, do_transform=True)

    autocast = (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext())
    for step in range(2):
        bx, by = ds[step]
        assert bx.shape[1] == exp_in_ch, \
            f'通道数不符：batch 有 {bx.shape[1]} 通道，模型期望 {exp_in_ch}'
        bx = bx.to(device)
        with autocast:
            out = net(bx)
            assert isinstance(out, dict) and target in out, \
                f'模型输出必须是含 {target!r} 键的 dict，实际得到 {type(out)}'
            # 多任务（aux_charts）时每个 chart 都要有自己的头，逐个查形状与类数。
            for c in opts['charts']:
                assert c in out, f'模型输出缺少 chart {c!r}（charts={opts["charts"]}）'
                assert out[c].shape[1] == opts['n_classes'][c], \
                    f'{c} 输出类别数 {out[c].shape[1]} != n_classes {opts["n_classes"][c]}'
                assert out[c].shape[-2:] == bx.shape[-2:], \
                    f'{c} 输出空间尺寸 {tuple(out[c].shape[-2:])} != 输入 {tuple(bx.shape[-2:])}'
            loss = sum(w * loss_fn[c](out[c], by[c].to(device))
                       for c, w in zip(opts['charts'], opts['task_weights']))
        assert torch.isfinite(loss), f'loss 非有限值：{loss.item()}'
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()
    print(f'[3-5] train step OK | batch_x {tuple(bx.shape)} -> out[{target}] '
          f'{tuple(out[target].shape)} | loss = {loss.item():.4f} (finite)', flush=True)

    # [6] 验证推理路径（合成小场景，避免在 CPU 上跑整景），分派同训练循环：
    # 'swin' 用 batched_slide_inference（滑窗），其余用 fast_tiled_val_inference。
    # 700×900 的合成景测不出最大验证景上的显存峰值。
    net.eval()
    H, W = 700, 900   # 会被 pad 到 256 的倍数
    # 验证输入用基础通道数：纹理通道由 fast_tiled_val_inference/predict 逐 tile 现算补上，
    # 传 exp_in_ch 会被重复追加。
    val_in_ch = exp_in_ch - n_tex
    fake_x = torch.randn(1, val_in_ch, H, W, device=device)
    fake_true = torch.randint(0, n_cls, (H, W), device=device)
    cm = new_confusion_matrix(n_cls, device)
    with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
        if opts['model_selection'] == 'swin':
            vout = batched_slide_inference(fake_x, net, opts_val, 'val')
            val_path = 'batched_slide_inference'
        else:
            vout = fast_tiled_val_inference(fake_x, net, opts_val)
            val_path = 'fast_tiled_val_inference'
        assert vout[target].shape == (1, n_cls, H, W), \
            f'验证输出形状 {tuple(vout[target].shape)} != (1,{n_cls},{H},{W})'
        pred = class_decider(vout[target], opts_val, target)
    accumulate_confusion_matrix(cm, fake_true.reshape(-1), pred.reshape(-1), n_cls)
    assert int(cm.sum().item()) == H * W, f'混淆矩阵像素数 {int(cm.sum())} != {H * W}'
    print(f'[6] val inference OK | {val_path} -> {tuple(vout[target].shape)} | '
          f'confmat total = {int(cm.sum())}', flush=True)

    # [7] tile 门控等价性：构造大片 255 的标签使部分 tile 全无效，对拍
    #   旧：fast_tiled_val_inference（全量前向）+ class_decider
    #   新：fast_tiled_val_predict（门控 + 流式 argmax）
    # 7a) gate=False：tile 顺序与分批同旧路径，必须全图逐像素相同，否则是索引 bug。
    # 7b) gate=True：跳过 tile 会改变批组成，cuDNN 可能选到不同算法，fp16 下末位差 1 ULP
    #     会让 argmax 平局翻转。只要求差异比例极小、差异像素的 top1-top2 间隔接近 0
    #     （随机噪声输入上实测翻 0.03%、间隔 max 1.2e-4）。
    # swin 走滑窗、不经过 tile 路径，跳过本项。
    if opts['model_selection'] != 'swin':
        ps = opts['patch_size']
        lab = torch.full((H, W), 255, dtype=torch.long)
        # 左上一块整 tile + 右下角一块（落在右、下都不足 patch_size 的残缺 tile 里，覆盖边缘
        # 补齐分支）。H=700/W=900、ps=256 时 tile 网格 3×4，留下 5 个有效、7 个全无效 tile。
        lab[0:500, 0:400] = torch.randint(0, n_cls, (500, 400))
        lab[600:700, 800:900] = torch.randint(0, n_cls, (100, 100))
        valid = lab != 255
        tile_mask = compute_valid_tile_mask(valid, ps)
        n_tiles = int(tile_mask.numel())
        n_kept = int(tile_mask.sum())
        assert 0 < n_kept < n_tiles, \
            f'门控没起作用：kept={n_kept}/{n_tiles}（构造的标签应当留下部分全无效 tile）'

        with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
            ref_logits = fast_tiled_val_inference(fake_x, net, opts_val)[target]
            ref_pred = class_decider(ref_logits, opts_val, target).reshape(H, W).cpu()
            ref_loss = loss_fn[target](ref_logits, lab.unsqueeze(0).to(device)).item()
            ungated, _ = fast_tiled_val_predict(
                fake_x, net, opts_val, labels={target: lab}, valid_maps={target: valid},
                gate=False)
            gated, gated_loss = fast_tiled_val_predict(
                fake_x, net, opts_val, labels={target: lab}, valid_maps={target: valid},
                loss_fns=loss_fn, task_weights=opts_val['task_weights'])

        # 7a) 分批完全一致 ⇒ 必须逐像素相同（全图，含被标签排除的区域）
        n_diff_ungated = int((ungated[target].long() != ref_pred).sum())
        assert n_diff_ungated == 0, \
            f'gate=False 时预测就与旧路径不同（{n_diff_ungated}/{H*W} 像素）——取 tile 或写回索引有 bug'

        # 7b) 门控后允许极小比例的 argmax 平局翻转，但必须确认「确实是平局」
        n_valid_px = int(valid.sum())
        diff = gated[target][valid].long() != ref_pred[valid]
        n_diff = int(diff.sum())
        margin = 0.0
        if n_diff:
            lg = ref_logits[0].float().cpu().permute(1, 2, 0)[valid]   # (n_valid, C)
            top2 = lg[diff].topk(2, dim=1).values
            margin = float((top2[:, 0] - top2[:, 1]).max())
        # fp16 噪声可能刚好超过阈值（UPerNet+SwinV2 实测 margin 1.24e-2），此时转 CPU/fp32
        # 仲裁：fp32 下门控只改变批组成，不应产生任何差异。
        try:
            assert n_diff <= 0.005 * n_valid_px, \
                f'门控前后预测差异过大：{n_diff}/{n_valid_px}（>0.5%，不像平局翻转）'
            assert margin <= 1e-2, \
                f'门控前后差异像素的 top1-top2 间隔达 {margin:.3e}，不是平局翻转，是真的算错了'
        except AssertionError as _e:
            if not cuda:
                raise      # CPU 本来就是 fp32，没有仲裁余地
            print(f'[7b] fp16 断言未通过（{_e}）→ 转 CPU/fp32 仲裁', flush=True)
            # 仲裁用独立的 fp32 CPU 副本，不改 net 本身的 device/dtype/状态。
            net32 = copy.deepcopy(net).float().cpu().eval()
            x32 = fake_x.detach().float().cpu()
            with torch.no_grad():        # 刻意不进 autocast：全程 fp32
                ref_logits32 = fast_tiled_val_inference(x32, net32, opts_val)[target]
                ref_pred32 = class_decider(ref_logits32, opts_val, target).reshape(H, W).cpu()
                gated32, _ = fast_tiled_val_predict(
                    x32, net32, opts_val, labels={target: lab},
                    valid_maps={target: valid}, gate=True)
            n_diff32 = int((gated32[target][valid].long() != ref_pred32[valid]).sum())
            del net32, x32, ref_logits32, ref_pred32, gated32
            if n_diff32 != 0:
                raise AssertionError(
                    f'CPU/fp32 仲裁【未通过】：fp32 下门控前后仍有 {n_diff32}/{n_valid_px} '
                    f'像素不同（fp32 只改批组成、理论上不该有差异）——大概率是索引 bug。'
                    f'⚠️ 但 CPU fp32 下不同批尺寸也可能让 MKL 选到不同分块、产生极小数值差；'
                    f'若差异只有个位数像素，需人工看一眼这些像素的 top1-top2 间隔再定性，'
                    f'不必然是索引 bug。原 fp16 断言：{_e}')
            # fp32 通过只说明不是索引 bug；fp16 下翻转超过 5% 有效像元时验证指标本身已被噪声污染。
            if n_diff > 0.05 * n_valid_px:
                raise AssertionError(
                    f'CPU/fp32 仲裁通过（fp32 下逐像素相同 ⇒ 不是索引 bug），但 fp16 下门控'
                    f'翻转了 {n_diff}/{n_valid_px}（{n_diff / max(n_valid_px, 1) * 100:.1f}% '
                    f'> 5%）——fp16 数值不稳到会污染验证指标的程度，拒绝放行。')
            print(f'[7b] CPU/fp32 仲裁通过：fp16 数值噪声（原 diff {n_diff}/{n_valid_px}、'
                  f'margin {margin:.3e}），放行', flush=True)
        if math.isnan(gated_loss):
            # 不可按批聚合的 loss（Dice/Lovász/带类权重 CE）按设计记 nan（见
            # functions._loss_is_mean_over_valid），无法对拍；[7c]/[7d] 断言两条路径的 nan 判定一致。
            print('[7b] val loss 为 nan（该 loss 不满足「Σ逐像素/有效像元数」形式，'
                  '按设计不做按批聚合）→ 跳过 loss 对拍', flush=True)
        else:
            assert abs(gated_loss - ref_loss) <= 2e-3 * max(1.0, abs(ref_loss)), \
                f'门控前后 val loss 不一致：{gated_loss:.6f} vs {ref_loss:.6f}'
        print(f'[7] tile gate OK | tiles {n_kept}/{n_tiles} (跳过 '
              f'{(1 - n_kept / n_tiles) * 100:.0f}%) | ungated diff 0 | '
              f'gated diff {n_diff}/{n_valid_px} (max margin {margin:.1e}) | '
              f'loss {gated_loss:.4f} vs {ref_loss:.4f}', flush=True)

        # 7c) confmat 模式（训练验证循环的默认路径）：同 gate 下批组成与 preds 模式相同，GPU 逐批
        #     累加的混淆矩阵应与「preds + valid 掩膜筛选 + accumulate_confusion_matrix」整数级相等。
        #     两次前向间若有非确定性算子，退一步要求行和精确相等、翻转像素 ≤ 0.5%。
        #     另在 sync_debug_mode('warn') 下统计 CUDA 同步次数，只打印不断言（模型前向内部
        #     可能有架构相关的合法同步）。
        _sync_n = None
        _has_sync_dbg = cuda and hasattr(torch.cuda, 'set_sync_debug_mode')
        if _has_sync_dbg:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode('warn')
        try:
            with warnings.catch_warnings(record=True) as _sync_w:
                warnings.simplefilter('always')
                with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
                    cms, cm_loss = fast_tiled_val_predict(
                        fake_x, net, opts_val, labels={target: lab}, valid_maps={target: valid},
                        loss_fns=loss_fn, task_weights=opts_val['task_weights'], out='confmat')
            if _has_sync_dbg:
                # 只数同步警告：同一 catch 里可能混进别的警告（如 val loss 不可按批聚合的
                # RuntimeWarning）。
                _sync_n = len([w for w in _sync_w
                               if 'synchron' in str(w.message).lower()])
        finally:
            if _has_sync_dbg:
                torch.cuda.set_sync_debug_mode('default')
        ref_cm = new_confusion_matrix(n_cls, device)
        accumulate_confusion_matrix(
            ref_cm, lab[valid].to(device), gated[target][valid].to(device).long(), n_cls)
        cm_new = cms[target].long().cpu()
        cm_ref = ref_cm.long().cpu()
        assert int(cm_new.sum()) == n_valid_px, \
            f'confmat 统计的像素数 {int(cm_new.sum())} != 有效像素数 {n_valid_px}——掩膜口径错了'
        cm_flip = 0
        if not torch.equal(cm_new, cm_ref):
            row_ok = bool(torch.equal(cm_new.sum(dim=1), cm_ref.sum(dim=1)))
            cm_flip = int((cm_new - cm_ref).abs().sum()) // 2
            assert row_ok and cm_flip <= 0.005 * n_valid_px, \
                (f'confmat 模式与 preds 模式不等价：行和一致={row_ok}、差异像素 '
                 f'{cm_flip}/{n_valid_px}——超出两次前向间数值噪声的量级，是真 bug')
        # 不可按批聚合的 loss 两边都记 nan，是设计行为。
        if math.isnan(cm_loss) or math.isnan(gated_loss):
            assert math.isnan(cm_loss) and math.isnan(gated_loss), \
                (f'nan 不对称：confmat {cm_loss} vs preds {gated_loss}——两条路径对'
                 f'「该 loss 是否可按批聚合」的判定不一致')
        else:
            assert abs(cm_loss - gated_loss) <= 2e-3 * max(1.0, abs(gated_loss)), \
                f'confmat 模式 loss 与 preds 模式不一致：{cm_loss:.6f} vs {gated_loss:.6f}'
        print(f'[7c] confmat mode OK | CM {"整数级相等" if cm_flip == 0 else f"平局翻转 {cm_flip} px（行和精确相等）"} | '
              f'loss {cm_loss:.4f} vs {gated_loss:.4f}'
              + (f' | CUDA syncs {_sync_n}（预期为个位数：仅函数末尾取回 loss）' if _sync_n is not None else ''),
              flush=True)

        # 7d) n_valid==0 守卫：val_batch_size=1 + gate=False 时 7 个全 255 的 tile 各自成批，
        #     逐一命中「整批无有效像元」分支；守卫失效时 loss 会被 nan 污染且不报错。
        #     CM 与 ungated preds 模式对拍（批组成不同，允许平局翻转，但行和必须精确相等）。
        opts_val_b1 = dict(opts_val)
        opts_val_b1['val_batch_size'] = 1
        with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
            cms0, loss0 = fast_tiled_val_predict(
                fake_x, net, opts_val_b1, labels={target: lab}, valid_maps={target: valid},
                loss_fns=loss_fn, task_weights=opts_val['task_weights'], gate=False,
                out='confmat')
        if math.isnan(gated_loss):
            assert math.isnan(loss0), \
                'preds 模式 loss 为 nan（不可按批聚合）时 confmat 模式也应为 nan'
        else:
            assert math.isfinite(loss0), \
                f'gate=False/vbs=1 下 confmat loss = {loss0}——n_valid==0 批的 where 守卫失效'
            assert abs(loss0 - ref_loss) <= 2e-3 * max(1.0, abs(ref_loss)), \
                (f'gate=False confmat loss 与整景 loss 不一致：{loss0:.6f} vs {ref_loss:.6f}'
                 f'——vbs=1 的批组成与整景差异极大，fp16 噪声大的架构（如 SwinV2）若在此'
                 f'翻车，先按 [7b] 先例做 CPU/fp32 仲裁判定是否数值噪声，别直接当等价性 bug')
        cm0 = cms0[target].long().cpu()
        assert int(cm0.sum()) == n_valid_px, \
            f'[7d] confmat 统计的像素数 {int(cm0.sum())} != {n_valid_px}'
        ref_cm0 = new_confusion_matrix(n_cls, device)
        accumulate_confusion_matrix(
            ref_cm0, lab[valid].to(device), ungated[target][valid].to(device).long(), n_cls)
        cm0_ref = ref_cm0.long().cpu()
        cm0_flip = 0
        if not torch.equal(cm0, cm0_ref):
            row_ok0 = bool(torch.equal(cm0.sum(dim=1), cm0_ref.sum(dim=1)))
            cm0_flip = int((cm0 - cm0_ref).abs().sum()) // 2
            assert row_ok0 and cm0_flip <= 0.005 * n_valid_px, \
                (f'[7d] vbs=1 confmat 与 ungated preds 不等价：行和一致={row_ok0}、'
                 f'差异像素 {cm0_flip}/{n_valid_px}')
        print(f'[7d] n_valid==0 guard OK | loss {loss0:.4f} vs ref {ref_loss:.4f} | '
              f'CM flip {cm0_flip}', flush=True)

    if cuda:
        print(f'    peak GPU mem = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB', flush=True)
        torch.cuda.reset_peak_memory_stats()

    print(f'PASS  {cfg_path}\n', flush=True)


def main():
    limit_torch_threads()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'device = {device}  (无卡模式下用 CPU 只验证正确性，不代表真实速度)\n', flush=True)

    cfgs = sys.argv[1:] or DEFAULT_CONFIGS
    results = {}
    t0 = time.time()
    for c in cfgs:
        try:
            smoke_one(c, device, cuda)
            results[c] = 'PASS'
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[c] = f'FAIL: {type(e).__name__}: {e}'
            print(f'FAIL  {c}\n', flush=True)

    print('=' * 72, flush=True)
    print(f'SUMMARY ({time.time() - t0:.0f}s, device={device}):', flush=True)
    for c, r in results.items():
        print(f'  {"OK " if r == "PASS" else "XX "} {c:48s} {r}', flush=True)
    print('=' * 72, flush=True)
    sys.exit(0 if all(r == 'PASS' for r in results.values()) else 1)


if __name__ == '__main__':
    main()
