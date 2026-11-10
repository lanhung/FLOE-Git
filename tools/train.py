"""Train a segmentation model from an mmcv config.

    python -m tools.train CONFIG --wandb-project P [--fold K] [--seed S] [--work-dir DIR]

--fold K trains one fold of the grouped CV split (datalists/cv5 by default). With
sequential_cv=True all folds found in cv_fold_dir are trained in one process.
"""
import argparse
import copy
import csv
import json
import random
import os
import os.path as osp
import re
import shutil
from icecream import ic
import pathlib
import warnings
import torchmetrics

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm

import wandb
from floe.functions import save_best_model, load_model, \
    batched_slide_inference, fast_tiled_val_predict, class_decider, \
    create_train_validation_and_test_scene_list, compute_combined_score, \
    get_scheduler, get_optimizer, get_loss, get_model, supcon_region_loss, ema_update, \
    new_confusion_matrix, accumulate_confusion_matrix, cm_weighted_f1, cm_classwise_f1, \
    cm_overall_accuracy, cm_classwise_iou, cm_miou, cm_precision_recall, cm_normalized, \
    cm_class_proportions, cm_ordinal_cost
from floe.loaders import (get_variable_options, val_scoped_options, AI4ArcticChallengeDataset,
                     AI4ArcticChallengeTestDataset, preload_scene_cache, limit_torch_threads,
                     val_ondemand_ok)
from floe.utils import colour_str
from floe.composition import CompositionLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FLOE segmentation model')

    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--wandb-name', default='MyDS', help='Name of wandb run (default: MyDS)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', default=None,
                        help='the seed to use, if not provided, seed from config file will be taken')
    parser.add_argument('--fold', type=int, default=None,
                        help='train a single fold of the frozen grouped CV split: reads '
                             '<cv_fold_dir>/fold<K>_train.json and fold<K>_val.json '
                             '(cv_fold_dir defaults to datalists/cv5)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume-from', type=pathlib.Path, default=None,
                       help='Resume Training from checkpoint, it will use the \
                        optimizer and schduler defined on checkpoint')
    group.add_argument('--finetune-from', type=pathlib.Path, default=None,
                       help='Start new tranining using the weights from checkpoitn')

    args = parser.parse_args()

    return args


def save_epoch_sod_distribution(save_path, epoch, class_counts, mask_count, chart='FLOE'):
    """Append one row of epoch-level label pixel distribution to CSV."""
    prefix = chart.lower()
    fieldnames = ['epoch'] + [f'{prefix}_{c}' for c in range(len(class_counts))] + [f'{prefix}_mask']
    write_header = not osp.exists(save_path)

    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {'epoch': epoch, f'{prefix}_mask': int(mask_count)}
        for c, count in enumerate(class_counts):
            row[f'{prefix}_{c}'] = int(count)
        writer.writerow(row)


# 尺寸回归辅助任务的类别刻度：类 1..5 的 log10(几何均值浮冰尺寸 / m)，分界见 utils.FLOE_LOOKUP。
#   cake <20m → 10m | small 20-100m → 45m | medium 100-500m → 224m
#   big 0.5-2km → 1000m | vast 2-10km → 4472m
# 水（类 0）与 255 不监督。改动此表后结果与已有 run 不可比。
_SCALE_REG_MU = (1.0, 1.65, 2.35, 3.0, 3.65)


def scale_reg_aux_loss(pred, labels, mu=_SCALE_REG_MU, beta=0.5, return_stats=False):
    """Smooth-L1 regression of the log10 floe size per stride-4 cell.

    pred is the model's '_scale_reg' output [B, 1, h, w]; labels are target-chart labels
    [B, H, W] (0 = water, 1..5 = ice, 255 = ignore). The target of a cell is the
    ice-class-fraction-weighted mean of `mu`, and the pixel weight is the ice fraction,
    so water and 255 drop out. A batch without ice gives exactly 0 (not nan).

    With return_stats=True also returns {'cls_sum', 'cls_cnt'}: per-class sums of the
    prediction over cells whose dominant ice class covers more than half the cell. They
    stay on the GPU (no per-batch sync); the caller reads them once per epoch.
    """
    pred = pred.float()
    y = labels.long()
    n_ice = len(mu)
    B, _, h, w = pred.shape
    assert labels.shape[0] == B and labels.dim() == 3, \
        f'labels 形状应为 [B,H,W]，实际 {tuple(labels.shape)}（pred {tuple(pred.shape)}）'
    H, W = y.shape[-2:]
    assert H % h == 0 and W % w == 0 and H // h == W // w, \
        f'标签 {H}×{W} 与预测网格 {h}×{w} 的下采样倍数不一致'
    pool = H // h
    # clamp 只为让 one_hot 不越界，255 随后由 valid 掩膜清零。
    valid = (y >= 1) & (y <= n_ice)
    onehot = F.one_hot(y.clamp(0, n_ice), n_ice + 1)[..., 1:].to(pred.dtype)  # [B,H,W,5]
    onehot = onehot * valid.unsqueeze(-1).to(pred.dtype)
    f = F.avg_pool2d(onehot.permute(0, 3, 1, 2), pool)          # [B,5,h,w] 各冰类占比
    ice_frac = f.sum(dim=1, keepdim=True)                        # [B,1,h,w]
    mu_vec = torch.as_tensor(mu, dtype=pred.dtype, device=pred.device).view(1, -1, 1, 1)
    target = (f * mu_vec).sum(dim=1, keepdim=True) / ice_frac.clamp_min(1e-6)
    weight = ice_frac.squeeze(1)                                 # [B,h,w]
    per_cell = F.smooth_l1_loss(pred, target, beta=beta, reduction='none').squeeze(1)
    # 写成 Σ(l·w)/Σw 而不用布尔掩膜索引（后者每 batch 触发一次 D2H 同步）：无冰 cell 的
    # w 恰为 0，结果与掩膜后求均值逐位等价；整批无冰时 loss 与梯度恰为 0。
    loss = (per_cell * weight).sum() / weight.sum().clamp_min(1e-6)
    if not return_stats:
        return loss
    with torch.no_grad():
        # 只统计主导类自身占满半格（max_c f_c > 0.5）的 cell。"冰占比 > 0.5"不够：
        # 0.45 small + 0.45 medium 的 cell 没有明确主导类，argmax 平局时会全记给 small。
        # 用 one-hot 求和代替布尔索引，同样是为了避免同步。
        dom_frac, dom = f.max(dim=1)                             # [B,h,w]，0..4 → 类 1..5
        pure = (dom_frac > 0.5).to(pred.dtype).unsqueeze(-1)     # [B,h,w,1]
        sel = F.one_hot(dom, n_ice).to(pred.dtype) * pure        # [B,h,w,5]
        cls_cnt = sel.sum(dim=(0, 1, 2))
        cls_sum = (sel * pred.squeeze(1).unsqueeze(-1)).sum(dim=(0, 1, 2))
    return loss, {'cls_sum': cls_sum, 'cls_cnt': cls_cnt}


def dann_lambda_schedule(epoch, weight, warmup):
    """DANN adversarial strength: linear ramp from 0 at epoch 0 to `weight` at `warmup`.

    The discriminator warms up before the reversed gradient reaches the encoder.
    warmup <= 0 means full strength from the first epoch.
    """
    if warmup <= 0:
        return float(weight)
    return float(weight) * min(1.0, epoch / float(warmup))


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0,
          resume_state=None):
    '''
    Trains the model; returns the path of the best checkpoint.

    resume_state: state saved with the best checkpoint by functions.save_best_model
    (best_primary_miou, val_hist_tail, early_stop_counter, val_smoothing_window), so a
    resumed run continues best selection, the smoothing window and early stopping from
    that validation. Not bit-exact: the best epoch is re-run. Only used with
    model_save_criterion == 'miou'; None keeps the old behaviour.
    '''
    best_primary_miou = -np.Inf
    best_ordinal_score = -np.Inf
    target_chart = train_options.get('target_chart', 'FLOE')
    model_save_criterion = train_options.get('model_save_criterion', 'miou')  # 'miou' | 'ordinal_score'

    early_stop_patience = train_options.get('early_stop_patience', 10)
    early_stop_counter = 0

    # Per-epoch val mIoU is noisy (std ~0.02), so a single-epoch max picks lucky spikes
    # and an early spike can trigger early stopping. With val_smoothing_window > 1 the
    # mean of the last k validations drives best selection and early stopping (1 = off).
    val_smoothing_window = train_options.get('val_smoothing_window', 1)
    # window=0 would make _val_hist[-0:] the whole history.
    assert val_smoothing_window >= 1, \
        f'val_smoothing_window 必须 >=1，得到 {val_smoothing_window}'
    _val_hist = []  # primary_miou of each validation, in order

    # Restore best score / smoothing window / early-stop counter. Checkpoints without
    # resume_state start from best=-inf, so the first validation overwrites best_model.
    _expected_best_path = osp.join(cfg.work_dir,
                                   f'best_model_{osp.basename(cfg.work_dir)}.pth')
    model_path = None   # returned if a resumed run never improves on its best
    if resume_state is not None:
        assert model_save_criterion == 'miou', (
            f'resume_state 仅支持 model_save_criterion=miou（当前 {model_save_criterion}）')
        _rs_w = resume_state.get('val_smoothing_window')
        assert _rs_w == val_smoothing_window, (
            f'resume_state 的 val_smoothing_window={_rs_w} 与当前配置 '
            f'{val_smoothing_window} 不一致——平滑口径不同禁止续跑')
        best_primary_miou = float(resume_state['best_primary_miou'])
        assert np.isfinite(best_primary_miou), (
            f'resume_state.best_primary_miou={best_primary_miou} 非有限值——checkpoint 来源可疑')
        _val_hist = [float(v) for v in resume_state.get('val_hist_tail', [])]
        early_stop_counter = int(resume_state.get('early_stop_counter', 0))
        if osp.exists(_expected_best_path):
            model_path = _expected_best_path
        if resume_state.get('ema'):
            print('\033[93m[Resume][WARN] 该 best 由 EMA 轨迹产生，而 EMA 影子权重不进 '
                  'checkpoint——续跑无法复现该轨迹（欠平均若干步），结果解读要打折\033[0m')
        print(f"\033[92m[Resume] 断点状态已还原：best_smoothed={best_primary_miou:.6f}  "
              f"val_hist_tail={[f'{v:.4f}' for v in _val_hist]}  "
              f"early_stop_counter={early_stop_counter}/{early_stop_patience}\033[0m")
    elif start_epoch > 0:
        print('\033[91m[Resume][WARN] checkpoint 无 resume_state（旧格式）：best 记录/平滑窗/'
              '早停计数从零开始，恢复后第一次验证将覆盖 best_model——如需保留旧 best 先自行备份\033[0m')

    # 验证时使用原始（未编译）模型，避免动态形状触发 recompile 警告
    net_val = net._orig_mod if train_options.get('compile_model') and hasattr(net, '_orig_mod') else net

    # 权重 EMA：验证、best 选择与保存都用 EMA 权重。EMA 影子不进 checkpoint，
    # --resume-from 后从 checkpoint 权重重新初始化，前 ~1/(1-decay) 步欠平均。
    ema_enabled = bool(train_options.get('ema', False))
    ema_decay = float(train_options.get('ema_decay', 0.999))
    ema_net = None
    ema_updates = 0  # 全局 update 计数，驱动 decay warmup ramp
    if ema_enabled:
        assert 0.0 < ema_decay < 1.0, f'ema_decay 应在 (0,1)，得到 {ema_decay}'
        assert not train_options.get('compile_model'), \
            'ema 与 compile_model 同开未适配（checkpoint 键名前缀不一致）'
        ema_net = copy.deepcopy(net_val)
        ema_net.eval()
        for p in ema_net.parameters():
            p.requires_grad_(False)
        print(f'[EMA] enabled: decay={ema_decay}（时间尺度 ~{1.0/(1.0-ema_decay):.0f} 步 '
              f'≈ {1.0/(1.0-ema_decay)/max(train_options["epoch_len"],1):.1f} epoch），'
              'warmup ramp d_t=min(decay,(1+t)/(10+t))；验证与 best 保存均使用 EMA 权重')

    # 验证只跑 target chart（辅任务 train-only，见 loaders.val_scoped_options）。
    val_options = val_scoped_options(train_options)
    val_charts = val_options['charts']
    # 有辅任务时 train loss 含 aux 分量而 val loss 不含，另记一条 target 分量曲线供 train/val 对比。
    aux_present = len(train_options['charts']) > 1

    primary_idx = train_options['charts'].index(target_chart)
    compute_train_primary_iou = (train_options['task_weights'][primary_idx] != 0)
    primary_n_classes = train_options['n_classes'][target_chart]
    if compute_train_primary_iou:
        train_iou_metric = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=primary_n_classes, average='none', ignore_index=255).to(device)

    loss_ce_functions = {chart: get_loss(train_options['chart_loss'][chart]['type'], chart=chart, **train_options['chart_loss'][chart]).to(device)
                         for chart in train_options['charts']}

    # Chart-composition supervision (floe/composition.py): polygon-level proportion and
    # ordinal losses on the target chart; hard_ce='pure' keeps pixel CE only inside
    # polygons dominated by one class, 'all' keeps the ordinary CE, 'none' drops it.
    comp_cfg = train_options.get('composition_supervision') or {}
    comp_loss = None
    if comp_cfg.get('enabled', False):
        _ds = dataloader_train.dataset
        comp_loss = CompositionLoss(
            _ds.bag_Q, _ds.bag_U, pure_threshold=comp_cfg.get('pure_threshold', 0.9),
            w_prop=comp_cfg.get('w_prop', 1.0), w_ord=comp_cfg.get('w_ord', 1.0),
            min_pixels=comp_cfg.get('min_bag_pixels', 64),
            max_unknown=comp_cfg.get('max_unknown', 0.5),
            ice_only=comp_cfg.get('ice_only', False)).to(device)
        comp_weight = float(comp_cfg.get('weight', 1.0))
        comp_hard = comp_cfg.get('hard_ce', 'pure')
        assert comp_hard in ('pure', 'all', 'none'), comp_hard
        print(f'[Composition] weight={comp_weight} hard_ce={comp_hard} ice_only={comp_loss.ice_only} '
              f'pure polygons {int((comp_loss.pure_class >= 0).sum())}/{len(_ds.bag_Q)}')

    # 区域级 SupCon 辅助损失：segformer_supcon 时 forward 在训练态多返回 '_supcon'
    # （stride-4 投影特征），以 λ 加权进总 loss。验证 loss 不含本项。
    supcon_enabled = bool(train_options.get('segformer_supcon', False))
    supcon_weight = float(train_options.get('segformer_supcon_weight', 0.1))
    supcon_temp = float(train_options.get('segformer_supcon_temp', 0.1))
    supcon_min_px = int(train_options.get('segformer_supcon_min_px', 16))
    if supcon_enabled:
        print(f'[SupCon] enabled: weight={supcon_weight} temp={supcon_temp} '
              f'min_px={supcon_min_px}（区域嵌入取自 encoder stride-4 投影特征）')

    # log10 尺寸回归辅助损失：segformer_scale_reg 时 forward 在训练态多返回 '_scale_reg'
    # （stride-4 的 1 通道回归图），以 λ 加权进总 loss。验证 loss 不含本项。
    scale_reg_enabled = bool(train_options.get('segformer_scale_reg', False))
    scale_reg_weight = float(train_options.get('segformer_scale_reg_weight', 0.1))
    if scale_reg_enabled:
        assert primary_n_classes == len(_SCALE_REG_MU) + 1, \
            (f'segformer_scale_reg 假定 6 类（水 + 5 个冰尺寸类），但 '
             f"n_classes[{target_chart}]={primary_n_classes}；μ 刻度表需同步修订")
        print(f'[ScaleReg] enabled: weight={scale_reg_weight} '
              f'μ(log10 m, 类1..5)={_SCALE_REG_MU}（目标=cell 内冰类占比加权均值，'
              '水/255 不监督）')

    # DANN 场景对抗：segformer_dann 时 forward 在训练态多返回 '_dann'（判别器对 patch
    # 所属场景的 logits，已过梯度反转层），loader 在 y 里给出 '_domain'（LongTensor [B]）。
    # 验证 loss 不含本项。判别器损失按权重 1.0 进总 loss，对抗强度 λ 只作用在 GRL 反传给
    # encoder 的梯度上；乘 λ 会同时削弱判别器本身。
    dann_enabled = bool(train_options.get('segformer_dann', False))
    dann_weight = float(train_options.get('segformer_dann_weight', 0.1))   # = λ 的上限
    dann_warmup = int(train_options.get('segformer_dann_warmup', 10))      # 线性升温 epoch 数
    if dann_enabled:
        # λ 是每 epoch 变的 python float，dynamo 会把它当 guard 常量，每个 epoch 整图 recompile。
        assert not train_options.get('compile_model'), \
            'segformer_dann 与 compile_model 同开未适配：dann_lambda 是 dynamo guard 的 ' \
            'float，每 epoch 改一次会触发整图 recompile。'
        n_domains = int(train_options['segformer_dann_domains'])
        # 判别器输出维度必须等于实际 train 景数，否则域序号可能越界，触发 CUDA device-side assert。
        assert len(train_options['train_list']) == n_domains, \
            (f'segformer_dann_domains={n_domains} 与实际 train 景数 '
             f'{len(train_options["train_list"])} 不一致（train_list 会剔除出现在 '
             f'validate_list 里的景，别直接抄 datalist 的长度）。')
        print(f'[DANN] enabled: λ_max={dann_weight} warmup={dann_warmup} epochs '
              f'n_domains={n_domains}（判别器吃 encoder stride-4 特征的 GAP；'
              f'域 acc 的随机基线 = {1.0 / n_domains:.4f}，越低越说明特征去场景化）')

    # GradScaler.step() skips the optimizer update on inf/nan gradients, so one overflowing
    # fp16 batch cannot corrupt the AdamW moments.
    scaler = torch.cuda.amp.GradScaler()
    grad_clip_max_norm = train_options.get('grad_clip_max_norm', 5.0)

    print('Training...')
    for epoch in tqdm(iterable=range(start_epoch, train_options['epochs'])):
        train_loss_sum = torch.tensor([0.])
        cross_entropy_loss_sum = torch.tensor([0.])
        primary_ce_loss_sum = torch.tensor([0.])  # target chart 的 CE 分量（仅 aux run 记录）
        supcon_loss_sum = torch.tensor([0.])  # SupCon 分量（未乘 λ）
        supcon_active = 0            # 产生了有效损失（存在同类跨样本对）的 batch 数
        supcon_same_sum = 0.0        # 同类跨样本平均余弦（仅 active 且非 nan 的 batch）
        supcon_diff_sum = 0.0
        supcon_cos_batches = 0
        supcon_cls_regions = np.zeros(primary_n_classes, dtype=np.float64)  # 各类区域嵌入数
        # 尺寸回归：loss（未乘 λ）与按主导冰类分组的预测均值（看回归输出是否按尺寸排序）。
        scale_reg_loss_sum = torch.tensor([0.])
        scale_reg_cls_sum = torch.zeros(len(_SCALE_REG_MU), device=device)
        scale_reg_cls_cnt = torch.zeros(len(_SCALE_REG_MU), device=device)
        # 裁剪前梯度范数与裁剪命中率（所有 run 都记），累加留 GPU、epoch 末同步一次。
        # 只累加有限的范数：GradScaler 标定 scale 时会周期性溢出，那些 batch 的范数是 inf；
        # 非有限占比单独记一条。
        gn_sum = torch.zeros((), device=device)
        gn_fin_cnt = torch.zeros((), device=device)
        clip_hits = torch.zeros((), device=device)
        # DANN：判别器损失与域判别准确率，域 acc 留在 GPU 上累加。
        dann_loss_sum = torch.tensor([0.])
        dann_correct = torch.zeros((), device=device)
        dann_seen = torch.zeros((), device=device)
        comp_sums = {'loss': 0.0, 'l_prop': 0.0, 'l_ord': 0.0, 'tv': 0.0, 'bags': 0, 'batches': 0}
        if dann_enabled:
            # 写在 net_val 上：compile_model 时 net 是 OptimizedModule 包装。
            # resume 时 epoch 从 checkpoint 续，ramp 自然接上。
            net_val.dann_lambda = dann_lambda_schedule(epoch, dann_weight, dann_warmup)
        val_loss_sum = torch.tensor([0.])
        val_cross_entropy_loss_sum = torch.tensor([0.])
        # 实际参与 backward/step 的 batch 数（跳过的坏 batch 不计入平均 loss 的分母）。
        processed_batches = 0

        # Target-chart label distribution over all training pixels in this epoch.
        primary_num_classes = train_options['n_classes'][target_chart]
        primary_epoch_class_counts = np.zeros(primary_num_classes, dtype=np.int64)
        primary_epoch_mask_count = 0
        # Online confusion matrix for train-set diagnostics.
        train_primary_confmat = torch.zeros((primary_num_classes, primary_num_classes), dtype=torch.float32, device=device)

        net.train()

        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'],
                                                    colour='red')):
            train_loss_batch = torch.tensor([0.]).to(device)
            cross_entropy_loss = torch.tensor([0.]).to(device)
            batch_x = batch_x.to(device, non_blocking=True)

            # Label statistics are gathered here in the main process, so they also
            # work with num_workers > 0.
            if target_chart in batch_y:
                primary_batch = batch_y[target_chart]
                for c in range(primary_num_classes):
                    primary_epoch_class_counts[c] += int((primary_batch == c).sum().item())
                primary_epoch_mask_count += int((primary_batch == 255).sum().item())

            bag_dev = batch_y['_bag'].to(device, non_blocking=True) if comp_loss is not None else None

            with torch.cuda.amp.autocast():
                output = net(batch_x)

                # tile 采样只保证 target chart 的标签覆盖，aux chart 可能整批全是 255，
                # CE(ignore_index=255) 会返回 nan，下面的 isfinite 守卫就会丢掉整个 batch。
                # 所以 aux chart 先查有效像元，没有就跳过该项（常态，不打日志）。
                for chart, weight in zip(train_options['charts'], train_options['task_weights']):
                    y_chart = batch_y[chart].to(device)
                    if chart != target_chart:
                        # aux chart 的值域校验：越界类别码在 CUDA 上会变成无 traceback 的
                        # device-side assert。放在这里而不是 loaders 的 tile 路径，因为
                        # _read_tile 抛异常会无限重试。下界也要查：负值约定为无效（-9），
                        # NaN 转 long 后也是极大负数。
                        _valid_c = (y_chart != 255)
                        _n_cls_c = train_options['n_classes'][chart]
                        _oob = _valid_c & ((y_chart < 0) | (y_chart >= _n_cls_c))
                        # 两个标志一次同步拿齐。
                        _has_valid, _has_oob = torch.stack(
                            [_valid_c.any(), _oob.any()]).tolist()
                        if _has_oob:
                            raise RuntimeError(
                                f'aux chart {chart!r} 的标签越界：n_classes[{chart}]='
                                f'{_n_cls_c}，但 epoch {epoch} batch {i} 里出现了类别码 '
                                f'{y_chart[_oob].unique()[:10].tolist()}（只列前 10 个；'
                                f'255=ignore 已排除）。检查 .nc 里该变量的取值口径 '
                                f'或 n_classes 配置。')
                        if not _has_valid:
                            continue
                    elif comp_loss is not None and comp_hard != 'all':
                        if comp_hard == 'none':
                            primary_ce_batch = torch.zeros((), device=device)
                            continue
                        y_chart = comp_loss.pure_only(y_chart, bag_dev)
                        if not bool((y_chart != 255).any()):
                            primary_ce_batch = torch.zeros((), device=device)
                            continue
                    _ce_chart = loss_ce_functions[chart](output[chart], y_chart)
                    cross_entropy_loss += weight * _ce_chart
                    if chart == target_chart:
                        primary_ce_batch = _ce_chart.detach()
            train_loss_batch = cross_entropy_loss

            # The auxiliary losses below run outside autocast in fp32 (normalize/logsumexp
            # and small regression targets are fragile in fp16). A missing output key
            # raises instead of silently training plain CE.
            supcon_loss, supcon_stats = None, None
            if supcon_enabled:
                if '_supcon' not in output:
                    raise RuntimeError(
                        "segformer_supcon=True 但模型输出没有 '_supcon' 键——"
                        f"model_selection={train_options.get('model_selection')!r} "
                        "是否为 sota_segformer？（只有它实现了 SupCon 投影头）")
                supcon_loss, supcon_stats = supcon_region_loss(
                    output['_supcon'], batch_y[target_chart].to(device),
                    n_classes=primary_n_classes,
                    temperature=supcon_temp, min_pixels=supcon_min_px,
                    return_stats=True)
                train_loss_batch = train_loss_batch + supcon_weight * supcon_loss

            scale_reg_loss, scale_reg_stats = None, None
            if scale_reg_enabled:
                if '_scale_reg' not in output:
                    raise RuntimeError(
                        "segformer_scale_reg=True 但模型输出没有 '_scale_reg' 键——"
                        f"model_selection={train_options.get('model_selection')!r} "
                        "是否为 sota_segformer？（只有它实现了尺寸回归辅助头）")
                scale_reg_loss, scale_reg_stats = scale_reg_aux_loss(
                    output['_scale_reg'], batch_y[target_chart].to(device),
                    return_stats=True)
                train_loss_batch = train_loss_batch + scale_reg_weight * scale_reg_loss

            loss_dann = None
            if dann_enabled:
                if '_dann' not in output:
                    raise RuntimeError(
                        "segformer_dann=True 但模型输出没有 '_dann' 键——"
                        f"model_selection={train_options.get('model_selection')!r} "
                        "是否为 sota_segformer？（只有它实现了 DANN 判别器）")
                if '_domain' not in batch_y:
                    raise RuntimeError(
                        "segformer_dann=True 但 batch 里没有 '_domain' 键——"
                        'dataset 是否走的 tile 采样路径（loaders 只在那条路上记账域序号）？')
                _dom = batch_y['_domain'].to(device, non_blocking=True)
                _dlogits = output['_dann'].float()
                # 权重 1.0，不乘 dann_weight：对抗强度由 GRL 里的 λ 承担。
                loss_dann = F.cross_entropy(_dlogits, _dom)
                train_loss_batch = train_loss_batch + loss_dann
                with torch.no_grad():
                    # 累加放在 isfinite 跳批守卫之后，与 dann_loss_sum 用同一批样本。
                    dann_hits_b = (_dlogits.argmax(dim=1) == _dom).sum()
                    dann_seen_b = _dom.numel()

            comp_loss_b, comp_stats = None, None
            if comp_loss is not None:
                comp_loss_b, comp_stats = comp_loss(output[target_chart], bag_dev)
                train_loss_batch = train_loss_batch + comp_weight * comp_loss_b

            optimizer.zero_grad()

            if not torch.isfinite(train_loss_batch):
                # 非有限 loss：跳过本次更新，避免污染 AdamW 的动量估计。
                print(f'[WARN] epoch {epoch} batch {i}: non-finite train loss '
                      f'({train_loss_batch.item()}), skipping this batch.')
                continue

            scaler.scale(train_loss_batch).backward()
            scaler.unscale_(optimizer)
            # error_if_nonfinite=False：inf/nan 梯度交给 scaler.step 跳过，而不是抛异常。
            # 返回值是裁剪前的全局梯度范数，累加留在 GPU 上、epoch 末同步一次。
            _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_max_norm,
                                                 error_if_nonfinite=False).detach()
            _gn_fin = torch.isfinite(_gn)
            gn_sum += torch.where(_gn_fin, _gn, torch.zeros_like(_gn))
            gn_fin_cnt += _gn_fin
            clip_hits += _gn_fin & (_gn > grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            # ReduceLROnPlateau steps once per validation, other schedulers per batch.
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # EMA decay warmup ramp (timm convention): without it the EMA still holds ~74%
            # of the initial weights after epoch 0 (0.999^300).
            if ema_net is not None:
                ema_updates += 1
                _d = min(ema_decay, (1.0 + ema_updates) / (10.0 + ema_updates))
                ema_update(ema_net, net_val, _d)

            processed_batches += 1
            train_loss_sum += train_loss_batch.detach().item()
            cross_entropy_loss_sum += cross_entropy_loss.detach().item()
            if aux_present:
                primary_ce_loss_sum += primary_ce_batch.item()
            if supcon_loss is not None:
                supcon_loss_sum += supcon_loss.detach().item()
                supcon_cls_regions += np.asarray(supcon_stats['regions_per_class'])
                if supcon_stats['active']:
                    supcon_active += 1
                    if (np.isfinite(supcon_stats['same_cos'])
                            and np.isfinite(supcon_stats['diff_cos'])):
                        supcon_same_sum += supcon_stats['same_cos']
                        supcon_diff_sum += supcon_stats['diff_cos']
                        supcon_cos_batches += 1
            if scale_reg_loss is not None:
                scale_reg_loss_sum += scale_reg_loss.detach().item()
                scale_reg_cls_sum += scale_reg_stats['cls_sum']
                scale_reg_cls_cnt += scale_reg_stats['cls_cnt']
            if comp_loss_b is not None:
                comp_sums['loss'] += float(comp_loss_b.detach())
                comp_sums['batches'] += 1
                for k in ('l_prop', 'l_ord', 'tv', 'bags'):
                    comp_sums[k] += comp_stats.get(k, 0)
            if loss_dann is not None:
                dann_loss_sum += loss_dann.detach().item()
                dann_correct += dann_hits_b
                dann_seen += dann_seen_b

            if compute_train_primary_iou:
                with torch.no_grad():
                    primary_pred_train = output[target_chart].detach().float().argmax(dim=1)
                    primary_true_train = batch_y[target_chart].to(device).long()
                    train_iou_metric.update(primary_pred_train, primary_true_train)

                    # Update confusion matrix incrementally to avoid storing full-epoch pixels.
                    valid = ((primary_true_train >= 0) & (primary_true_train < primary_num_classes) &
                             (primary_pred_train >= 0) & (primary_pred_train < primary_num_classes))
                    if valid.any():
                        t = primary_true_train[valid].view(-1)
                        p = primary_pred_train[valid].view(-1)
                        flat_idx = t * primary_num_classes + p
                        binc = torch.bincount(flat_idx, minlength=primary_num_classes * primary_num_classes).float()
                        train_primary_confmat += binc.view(primary_num_classes, primary_num_classes)

        # 平均 loss 的分母用实际参与更新的 batch 数，而不是 i+1。
        if processed_batches < i + 1:
            print(f'[WARN] epoch {epoch}: skipped {i + 1 - processed_batches}/{i + 1} '
                  f'batches due to non-finite loss.')
        train_loss_epoch = torch.true_divide(train_loss_sum, max(processed_batches, 1)).detach().item()
        cross_entropy_epoch = torch.true_divide(cross_entropy_loss_sum, max(processed_batches, 1)).detach().item()
        supcon_epoch = torch.true_divide(supcon_loss_sum, max(processed_batches, 1)).detach().item()
        # 以下读数字典只在开了对应机制时并进 wandb.log。
        # SupCon：Cos Gap = 同类与异类余弦之差；某类 regions ≈ 0 说明 min_px 把它挡在了门外。
        supcon_log = {}
        if supcon_enabled:
            _nb = max(processed_batches, 1)
            _ncb = max(supcon_cos_batches, 1)
            supcon_log = {
                "Train SupCon Epoch Loss": supcon_epoch,
                "SupCon Active Batch Frac": supcon_active / _nb,
                "SupCon Same-class Cos": supcon_same_sum / _ncb,
                "SupCon Diff-class Cos": supcon_diff_sum / _ncb,
                "SupCon Cos Gap": (supcon_same_sum - supcon_diff_sum) / _ncb,
            }
            for c in range(primary_n_classes):
                supcon_log[f"SupCon Regions Class {c}"] = supcon_cls_regions[c] / _nb

        scale_reg_log = {}
        if scale_reg_enabled:
            _cnt = scale_reg_cls_cnt.clamp_min(1.0)
            _mean = (scale_reg_cls_sum / _cnt).tolist()          # epoch 一次同步
            _seen = scale_reg_cls_cnt.tolist()
            scale_reg_log["ScaleReg Loss"] = torch.true_divide(
                scale_reg_loss_sum, max(processed_batches, 1)).detach().item()
            for c in range(len(_SCALE_REG_MU)):
                # 类 c+1 占满半格的 cell 上的预测均值，与 _SCALE_REG_MU[c] 对比。
                scale_reg_log[f"ScaleReg Pred cls{c + 1}"] = (
                    _mean[c] if _seen[c] > 0 else float('nan'))

        # 梯度范数读数（所有 run 都记，便于跨 run 对比）。走到裁剪那一步的 batch 数恰为
        # processed_batches；均值与命中率只在范数有限的 batch 上算。
        _nb_gn = max(processed_batches, 1)
        _nfin = gn_fin_cnt.clamp_min(1.0)
        gn_log = {
            "Grad Norm": (gn_sum / _nfin).item(),
            "Clip Hit Frac": (clip_hits / _nfin).item(),
            "Grad Nonfinite Frac": ((_nb_gn - gn_fin_cnt) / _nb_gn).item(),
        }

        # DANN：域判别准确率降向 1/N 说明特征去场景化。
        dann_log = {}
        if dann_enabled:
            _dseen = float(dann_seen.item())
            dann_log = {
                "DANN Loss": torch.true_divide(
                    dann_loss_sum, max(processed_batches, 1)).detach().item(),
                "DANN Domain Acc": (float(dann_correct.item()) / _dseen) if _dseen > 0
                                   else float('nan'),
                "DANN Lambda": float(getattr(net_val, 'dann_lambda', 0.0)),
            }

        comp_log = {}
        if comp_loss is not None:
            _cb = max(comp_sums['batches'], 1)
            comp_log = {'Composition Loss': comp_sums['loss'] / _cb,
                        'Composition Prop Loss': comp_sums['l_prop'] / _cb,
                        'Composition Ord Loss': comp_sums['l_ord'] / _cb,
                        'Composition TV': comp_sums['tv'] / _cb,
                        'Composition Bags per Batch': comp_sums['bags'] / _cb}
            print('[Composition] ' + ' | '.join(f'{k.split(" ", 1)[1]} {v:.4f}' for k, v in comp_log.items()))

        # 模型自带的训练前向统计（epoch 均值）。pop_sk_stats 会清零累加器，所以每个 epoch
        # 恰好取一次，放在 do_validate 分支之前，两条 wandb.log 路径共用。
        sk_log = {}
        if hasattr(net_val, 'pop_sk_stats'):
            _sk = net_val.pop_sk_stats()
            if _sk:
                sk_log = _sk

        # 'Train {target} Loss' 只含 target 分量，与 val loss 同量纲（仅 aux run 记录）。
        aux_log = {}
        if aux_present:
            aux_log[f"Train {target_chart} Loss"] = torch.true_divide(
                primary_ce_loss_sum, max(processed_batches, 1)).detach().item()

        patch_log_path = osp.join(cfg.work_dir, 'patch_sampling_log.csv')
        save_epoch_sod_distribution(
            patch_log_path,
            epoch,
            primary_epoch_class_counts,
            primary_epoch_mask_count,
            chart=target_chart)

        # Dynamic FocalLoss weight update (inverse frequency from actual pixel distribution)
        _dyn_enabled = train_options.get('dynamic_loss_weight', False)
        _dyn_warmup  = train_options.get('dynamic_weight_warmup', 5)
        if (_dyn_enabled
                and epoch >= _dyn_warmup
                and primary_epoch_class_counts.sum() > 0
                and target_chart in loss_ce_functions):
            _counts = primary_epoch_class_counts.astype(np.float64) + 1.0  # +1 avoid /0
            _freq   = _counts / _counts.sum()
            _raw_w  = 1.0 / _freq
            _raw_w  = _raw_w / _raw_w.mean()          # normalise to mean=1
            _max_w  = float(train_options.get('dynamic_weight_max', 5.0))
            _raw_w  = np.clip(_raw_w, 1.0 / _max_w, _max_w)
            _raw_w  = _raw_w / _raw_w.mean()          # re-normalise after clip
            _ema    = float(train_options.get('dynamic_weight_ema', 0.7))
            _loss_fn = loss_ce_functions[target_chart]
            if hasattr(_loss_fn, 'weight') and _loss_fn.weight is not None:
                _old_w = _loss_fn.weight.detach().cpu().numpy().astype(np.float64)
                _new_w = _ema * _old_w + (1.0 - _ema) * _raw_w
                _loss_fn.weight.data.copy_(torch.FloatTensor(_new_w).to(device))
                print(f"  [DynW] {target_chart} loss weights: {[f'{w:.3f}' for w in _new_w]}")
                wandb.log(
                    {f"{target_chart}/DynLossWeight_cls{c}": float(w)
                     for c, w in enumerate(_new_w)},
                    step=epoch)
            else:
                print(f"  [DynW] {target_chart} loss ({type(_loss_fn).__name__}) has no per-class weight, skipping.")

        if compute_train_primary_iou:
            train_iou_per_class = train_iou_metric.compute()  # shape: (n_classes,)
            train_miou = train_iou_per_class.mean()
            train_iou_metric.reset()

            # Derive train-set precision/recall/proportions/confusion-matrix from online CM.
            train_tp = torch.diag(train_primary_confmat)
            train_pred_sum = train_primary_confmat.sum(dim=0)
            train_true_sum = train_primary_confmat.sum(dim=1)
            train_precision_per_class = torch.where(
                train_pred_sum > 0, train_tp / train_pred_sum, torch.zeros_like(train_tp))
            train_recall_per_class = torch.where(
                train_true_sum > 0, train_tp / train_true_sum, torch.zeros_like(train_tp))
            train_row_sum = train_primary_confmat.sum(dim=1, keepdim=True)
            train_norm_confmat = torch.where(
                train_row_sum > 0, train_primary_confmat / train_row_sum, torch.zeros_like(train_primary_confmat))
            train_total = train_true_sum.sum().clamp_min(1.0)
            train_true_props = train_true_sum / train_total
            train_pred_props = train_pred_sum / train_total

            print(f"Train {target_chart} mIoU: {train_miou:.4f}")
            print(f"Train {target_chart} IoU per class: {train_iou_per_class}")
            print(f"Train {target_chart} Precision per class: {train_precision_per_class}")
            print(f"Train {target_chart} Recall per class: {train_recall_per_class}")
            print(f"Train {target_chart} True class proportion: {train_true_props}")
            print(f"Train {target_chart} Pred class proportion: {train_pred_props}")
            print(f"Train {target_chart} Normalized Confusion Matrix (row=true, col=pred):")
            print(train_norm_confmat)

            wandb.log({f"Train {target_chart} mIoU": train_miou.item()}, step=epoch)
            for c, iou_c in enumerate(train_iou_per_class):
                wandb.log({f"Train {target_chart}/IoU Class {c}": iou_c.item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Precision Class {c}": train_precision_per_class[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Recall Class {c}": train_recall_per_class[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/True Proportion Class {c}": train_true_props[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Pred Proportion Class {c}": train_pred_props[c].item()}, step=epoch)
            for r in range(train_norm_confmat.shape[0]):
                for c in range(train_norm_confmat.shape[1]):
                    wandb.log({f"Train {target_chart}/ConfMatNorm r{r}c{c}": train_norm_confmat[r, c].item()}, step=epoch)
        else:
            train_iou_per_class = None

        # -- Validation -- #
        val_freq = train_options.get('val_freq', 1)
        do_validate = (epoch % val_freq == 0) or (epoch == train_options['epochs'] - 1)

        if not do_validate:
            wandb.log({"Train Epoch Loss": train_loss_epoch,
                       "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                       "Learning Rate": optimizer.param_groups[0]["lr"],
                       **supcon_log, **scale_reg_log, **sk_log, **aux_log,
                       **gn_log, **dann_log, **comp_log}, step=epoch)
            print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
            print(f"Grad Norm (pre-clip mean): {gn_log['Grad Norm']:.3f} | "
                  f"Clip Hit Frac: {gn_log['Clip Hit Frac']:.3f} | Nonfinite Frac: "
                  f"{gn_log['Grad Nonfinite Frac']:.3f} (max_norm={grad_clip_max_norm})")
            if supcon_enabled:
                print(f"Train SupCon Epoch Loss: {supcon_epoch:.3f}")
            if scale_reg_enabled:
                print(f"Train ScaleReg Epoch Loss: {scale_reg_log['ScaleReg Loss']:.4f}")
            if dann_enabled:
                print(f"Train DANN Loss: {dann_log['DANN Loss']:.4f} | Domain Acc: "
                      f"{dann_log['DANN Domain Acc']:.4f} | λ={dann_log['DANN Lambda']:.4f}")
            continue

        # 逐景累加 (n_classes × n_classes) 混淆矩阵，而不是把所有景的像素拼成扁平张量：
        # 原生分辨率下验证集约 8 亿有效像素，拼接会 OOM。所有验证指标都是混淆矩阵的函数。
        torch.cuda.empty_cache()  # 释放训练残留的缓存显存，为滑窗推理腾出空间
        val_confmats = {chart: new_confusion_matrix(train_options['n_classes'][chart], device)
                        for chart in val_charts}
        net.eval()
        eval_net = ema_net if ema_net is not None else net_val
        if ema_net is not None:
            # ‖ema−raw‖/‖raw‖：恒为 0 说明 EMA 影子没有在更新。
            with torch.no_grad():
                _num, _den = 0.0, 0.0
                _msd = net_val.state_dict()
                for k, v in ema_net.state_dict().items():
                    if v.dtype.is_floating_point:
                        _num += (v - _msd[k]).float().pow(2).sum().item()
                        _den += _msd[k].float().pow(2).sum().item()
            _ema_lag = (_num ** 0.5) / max(_den ** 0.5, 1e-12)
            print(f'[EMA] Δ‖ema−raw‖/‖raw‖ = {_ema_lag:.5f}')
            wandb.log({"EMA Lag": _ema_lag}, step=epoch)
        print('Validating...')
        for i, (inf_x, inf_y, cfv_masks, tfv_mask, name, original_size) in enumerate(tqdm(iterable=dataloader_val,
                                                                            total=len(train_options['validate_list']),
                                                                            colour='green')):
            torch.cuda.empty_cache()
            # 有效像元掩膜（label != 255）。最大景约 2.5 亿像素，算一次给 tile 门控和
            # 混淆矩阵筛选共用。
            valid_masks = {c: ~cfv_masks[c] for c in val_charts}

            with torch.no_grad(), torch.cuda.amp.autocast():
                # 大景走 fast_tiled_val_predict：只前向含有效标签的 tile（全 255 的 tile 对
                # 混淆矩阵和 loss 的贡献为 0），整景留在 CPU，不开整景 logits 缓冲。
                # Swin 走批量滑窗。
                use_fast_tiled = (train_options['model_selection'] != 'swin' and
                                  (inf_x.shape[2] > train_options['patch_size'] or
                                   inf_x.shape[3] > train_options['patch_size']))
                if use_fast_tiled:
                    # out='confmat' 在 GPU 上逐批累加混淆矩阵，结果与 preds 路径整数级相等；
                    # val_confmat_batched=False 回退到 preds 路径。
                    _batched_cm = train_options.get('val_confmat_batched', True)
                    pred_maps = None
                    if _batched_cm:
                        scene_cms, val_ce = fast_tiled_val_predict(
                            inf_x, eval_net, val_options,
                            labels=inf_y,
                            valid_maps=valid_masks,
                            loss_fns=loss_ce_functions,
                            task_weights=val_options['task_weights'],
                            gate=train_options.get('val_tile_gate', True),
                            out='confmat')
                    else:
                        pred_maps, val_ce = fast_tiled_val_predict(
                            inf_x, eval_net, val_options,
                            labels=inf_y,
                            valid_maps=valid_masks,
                            # 函数内部只按 options['charts'] = val_charts 取用 loss_fns。
                            loss_fns=loss_ce_functions,
                            task_weights=val_options['task_weights'],
                            gate=train_options.get('val_tile_gate', True))
                else:
                    inf_x = inf_x.to(device, non_blocking=True)
                    if train_options['model_selection'] == 'swin':
                        output = batched_slide_inference(inf_x, eval_net, val_options, 'val')
                    else:
                        output = eval_net(inf_x)

                    val_ce = 0.0
                    for chart, weight in zip(val_charts, val_options['task_weights']):
                        val_ce += weight * loss_ce_functions[chart](
                            output[chart],
                            inf_y[chart].unsqueeze(0).long().to(device)).item()
                    pred_maps = {c: class_decider(output[c], val_options, c)
                                 for c in val_charts}
                    del output

            # Accumulate valid pixels into the confusion matrices.
            if use_fast_tiled and pred_maps is None:
                for chart in val_charts:
                    val_confmats[chart] += scene_cms[chart]
                del scene_cms
            else:
                for chart in val_charts:
                    _sel = valid_masks[chart]
                    _p = pred_maps[chart]
                    # fast 路径的预测图在 CPU（uint8）：先在 CPU 上筛选，只把有效像素送上 GPU。
                    _pred = (_p[_sel] if _p.device.type == 'cpu' else _p[_sel.to(_p.device)])
                    _pred = _pred.to(device, non_blocking=True)
                    _true = inf_y[chart][_sel].to(device, non_blocking=True)
                    accumulate_confusion_matrix(val_confmats[chart], _true, _pred,
                                                train_options['n_classes'][chart])
                    del _pred, _true
            val_loss_sum += val_ce
            val_cross_entropy_loss_sum += val_ce

        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()
        val_cross_entropy_epoch = torch.true_divide(val_cross_entropy_loss_sum, i + 1).detach().item()

        # All validation metrics are derived from val_confmats.
        print('Computing Metrics on Val dataset')
        scores = {}
        for chart in val_charts:
            _func_name = train_options['chart_metric'][chart]['func'].__name__
            if _func_name != 'f1_metric':
                raise NotImplementedError(
                    f"chart_metric[{chart}]['func'] = {_func_name}；混淆矩阵版指标目前只实现了 "
                    f"f1_metric（加权 F1）。SIC 的 r2_metric 是连续值指标，无法从混淆矩阵导出。")
            scores[chart] = torch.round(cm_weighted_f1(val_confmats[chart]) * 100, decimals=3)
        combined_score = compute_combined_score(scores=scores, charts=val_charts,
                                                metrics=train_options['chart_metric'])

        # target chart 的 mIoU 是 best 选择与早停的准则。
        miou_scores = {c: cm_miou(val_confmats[c]) for c in val_charts}
        primary_miou = miou_scores[target_chart]

        # 有序代价指标：错误按类别距离计代价（cost_per_distance）。
        _ordinal_cfg = train_options.get('ordinal_metric', {})
        if _ordinal_cfg.get('enabled', False):
            _cpd = _ordinal_cfg.get('cost_per_distance', None)
            ordinal_raw_costs, ordinal_norm_scores = {}, {}
            for chart in val_charts:
                ordinal_raw_costs[chart], ordinal_norm_scores[chart] = cm_ordinal_cost(
                    val_confmats[chart], cost_per_distance=_cpd)
        else:
            ordinal_raw_costs, ordinal_norm_scores = None, None

        classwise_iou_scores = {c: cm_classwise_iou(val_confmats[c]) for c in val_charts}

        classwise_precision_scores, classwise_recall_scores = {}, {}
        normalized_conf_mats, true_class_props, pred_class_props = {}, {}, {}
        for chart in val_charts:
            classwise_precision_scores[chart], classwise_recall_scores[chart] = \
                cm_precision_recall(val_confmats[chart])
            normalized_conf_mats[chart] = cm_normalized(val_confmats[chart])
            true_class_props[chart], pred_class_props[chart] = cm_class_proportions(val_confmats[chart])

        if train_options['compute_classwise_f1score']:
            classwise_scores = {c: cm_classwise_f1(val_confmats[c]) for c in val_charts}
            oa_scores = {c: cm_overall_accuracy(val_confmats[c]) for c in val_charts}

        del val_confmats
        import gc; gc.collect()
        torch.cuda.empty_cache()

        print("")
        print(f"Epoch {epoch} score:")

        for chart in val_charts:
            if train_options['chart_metric'][chart]['weight'] == 0:
                continue

            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            if train_options['compute_classwise_f1score']:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])

                print(f"{chart} OA:", oa_scores[chart])
                wandb.log({f"{chart}/OA": oa_scores[chart]}, step=epoch)

                print(f"{chart} mIoU:", miou_scores[chart])
                wandb.log({f"{chart}/mIoU": miou_scores[chart]}, step=epoch)

                print(f"{chart} IoU per class:", classwise_iou_scores[chart])
                for c, iou_c in enumerate(classwise_iou_scores[chart]):
                    wandb.log({f"{chart}/IoU Class {c}": iou_c.item()}, step=epoch)

            if ordinal_raw_costs is not None and chart in ordinal_raw_costs:
                print(f"{chart} Ordinal Mean Cost (lower=better): {ordinal_raw_costs[chart]:.4f}")
                print(f"{chart} Ordinal Score (higher=better):    {ordinal_norm_scores[chart]:.4f}")
                wandb.log({f"{chart}/OrdinalMeanCost":  ordinal_raw_costs[chart].item(),
                           f"{chart}/OrdinalScore":     ordinal_norm_scores[chart].item()}, step=epoch)

            print(f"{chart} Precision per class:", classwise_precision_scores[chart])
            for c, prec_c in enumerate(classwise_precision_scores[chart]):
                wandb.log({f"{chart}/Precision Class {c}": prec_c.item()}, step=epoch)

            print(f"{chart} Recall per class:", classwise_recall_scores[chart])
            for c, rec_c in enumerate(classwise_recall_scores[chart]):
                wandb.log({f"{chart}/Recall Class {c}": rec_c.item()}, step=epoch)

            print(f"{chart} True class proportion:", true_class_props[chart])
            print(f"{chart} Pred class proportion:", pred_class_props[chart])
            for c, p_true in enumerate(true_class_props[chart]):
                wandb.log({f"{chart}/True Proportion Class {c}": p_true.item()}, step=epoch)
            for c, p_pred in enumerate(pred_class_props[chart]):
                wandb.log({f"{chart}/Pred Proportion Class {c}": p_pred.item()}, step=epoch)

            cm_norm = normalized_conf_mats[chart]
            print(f"{chart} Normalized Confusion Matrix (row=true, col=pred):")
            print(cm_norm)
            for r in range(cm_norm.shape[0]):
                for c in range(cm_norm.shape[1]):
                    wandb.log({f"{chart}/ConfMatNorm r{r}c{c}": cm_norm[r, c].item()}, step=epoch)

        if compute_train_primary_iou and train_iou_per_class is not None and target_chart in classwise_iou_scores:
            train_val_iou_gap = train_iou_per_class - classwise_iou_scores[target_chart]
            print(f"{target_chart} Train-Val IoU gap per class:", train_val_iou_gap)
            for c, gap_c in enumerate(train_val_iou_gap):
                wandb.log({f"{target_chart}/Train-Val IoU Gap Class {c}": gap_c.item()}, step=epoch)

        # ReduceLROnPlateau 由 target mIoU 驱动，与 best 选择准则一致。
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(primary_miou)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Train Cross Entropy Epoch Loss: {cross_entropy_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"Validation Cross Entropy Epoch Loss: {val_cross_entropy_epoch:.3f}")
        print(f"Validation {target_chart} mIoU: {primary_miou.item():.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        wandb.log({"Combined score": combined_score,
                   f"Validation {target_chart} mIoU": primary_miou.item(),
                   "Train Epoch Loss": train_loss_epoch,
                   "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                   "Validation Epoch Loss": val_loss_epoch,
                   "Validation Cross Entropy Epoch Loss": val_cross_entropy_epoch,
                   "Learning Rate": optimizer.param_groups[0]["lr"],
                   **supcon_log, **scale_reg_log, **sk_log, **aux_log,
                   **gn_log, **dann_log, **comp_log}, step=epoch)
        print(f"Grad Norm (pre-clip mean): {gn_log['Grad Norm']:.3f} | "
              f"Clip Hit Frac: {gn_log['Clip Hit Frac']:.3f} | Nonfinite Frac: "
              f"{gn_log['Grad Nonfinite Frac']:.3f} (max_norm={grad_clip_max_norm})")
        if supcon_enabled:
            print(f"Train SupCon Epoch Loss: {supcon_epoch:.3f}")
        if scale_reg_enabled:
            print(f"Train ScaleReg Epoch Loss: {scale_reg_log['ScaleReg Loss']:.4f} "
                  f"| Pred per class 1..5: "
                  + ', '.join(f"{scale_reg_log[f'ScaleReg Pred cls{c}']:.2f}"
                              for c in range(1, len(_SCALE_REG_MU) + 1)))
        if dann_enabled:
            print(f"Train DANN Loss: {dann_log['DANN Loss']:.4f} | Domain Acc: "
                  f"{dann_log['DANN Domain Acc']:.4f}（随机基线 "
                  f"{1.0 / int(train_options['segformer_dann_domains']):.4f}）"
                  f" | λ={dann_log['DANN Lambda']:.4f}")
        if aux_present:
            print(f"Train {target_chart} Loss (target 分量): "
                  f"{aux_log[f'Train {target_chart} Loss']:.3f}")

        # Swin 的注意力残差 alpha（零初始化）：始终为 0 说明残差链路被截断。
        if (train_options['model_selection'] == 'swin'
                and hasattr(net_val, 'layers')
                and len(net_val.layers) > 2):
            stage2_alphas = torch.stack([
                blk.attn.alpha.detach().cpu()
                for blk in net_val.layers[2].blocks
            ])  # [num_blocks=6, num_heads=12]
            alpha_mean = stage2_alphas.mean().item()
            alpha_max  = stage2_alphas.max().item()
            # 每个 block 在所有 head 上的均值（偶数=W-MSA，奇数=SW-MSA）
            alpha_per_block = stage2_alphas.mean(dim=1).tolist()
            print(f"[AttnResidual] Stage2 alpha | "
                  f"mean={alpha_mean:.4f}  max={alpha_max:.4f}  "
                  f"per_block(W/SW alternating)={[f'{v:.4f}' for v in alpha_per_block]}")
            wandb.log({
                "AttnResidual/alpha_mean": alpha_mean,
                "AttnResidual/alpha_max":  alpha_max,
                **{f"AttnResidual/block{i}_alpha_mean": v
                   for i, v in enumerate(alpha_per_block)},
            }, step=epoch)

        if model_save_criterion == 'ordinal_score' and ordinal_norm_scores is not None:
            _current_criterion = ordinal_norm_scores.get(target_chart, torch.tensor(-np.Inf))
            _current_criterion = _current_criterion.item() if hasattr(_current_criterion, 'item') else float(_current_criterion)
            _is_best = _current_criterion > best_ordinal_score
            if _is_best:
                best_ordinal_score = _current_criterion
        else:
            # 最近 val_smoothing_window 次验证 mIoU 的均值决定是否最佳；保存的是当前 epoch 的权重。
            _val_hist.append(primary_miou.item())
            _current_criterion = sum(_val_hist[-val_smoothing_window:]) / len(_val_hist[-val_smoothing_window:])
            if val_smoothing_window > 1:
                print(f"Smoothed val mIoU (w={val_smoothing_window}): {_current_criterion:.4f} "
                      f"(raw {primary_miou.item():.4f})")
            _is_best = _current_criterion > best_primary_miou
            if _is_best:
                best_primary_miou = _current_criterion

        if _is_best:
            early_stop_counter = 0

            wandb.run.summary[f"While training/Best {target_chart} mIoU"] = best_primary_miou
            for chart in val_charts:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary[f"While training/Validation {target_chart} mIoU"] = primary_miou.item()
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch

            # Saved locally only, not uploaded to wandb. With EMA the EMA weights are saved
            # (they are what validation selected); optimizer/scheduler state is from the raw
            # weights. resume_state only makes sense for the miou criterion, so it is None
            # otherwise and a resumed run prints the old-format warning.
            _snap = None
            if model_save_criterion == 'miou':
                _snap = {
                    'best_primary_miou': best_primary_miou,
                    'val_hist_tail': list(_val_hist[-val_smoothing_window:]),
                    'early_stop_counter': early_stop_counter,
                    'val_smoothing_window': val_smoothing_window,
                    'ema': bool(ema_enabled),
                }
            model_path = save_best_model(cfg, train_options,
                                         ema_net if ema_net is not None else net,
                                         optimizer, scheduler, epoch,
                                         resume_state=_snap)
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} mIoU: {best_primary_miou:.4f}")
                break

    return model_path


def create_dataloaders(train_options):
    '''
    Create the train and validation dataloaders from train_list / validate_list.

    The train DataLoader gets its own generator seeded with `_resolved_seed`, so sampler
    order and worker base seeds do not depend on how much global RNG the model consumed.
    '''
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options, do_transform=True)

    prefetch = train_options.get('prefetch_factor', 2) if train_options['num_workers'] > 0 else None
    seed = train_options.get('_resolved_seed', None)
    # '_legacy_rng' is a test-only switch (tools.checks.seed_stream --legacy) that
    # reproduces the old shared-RNG behaviour; never set it in a config.
    use_gen = seed is not None and not train_options.get('_legacy_rng', False)
    gen = torch.Generator().manual_seed(int(seed)) if use_gen else None
    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'],
        pin_memory=True, prefetch_factor=prefetch, generator=gen)

    # 验证按需读是显式 opt-in 的（见 loaders.val_ondemand_ok），判定结果与原因打进日志。
    # 验证 dataset 只读 target chart 的标签与掩膜（辅任务 train-only）。
    val_options = val_scoped_options(train_options)
    _od_ok, _od_why = val_ondemand_ok(val_options)
    print(f"[Val] 按需读 = {'ON' if _od_ok else 'OFF'}（{_od_why}）")
    dataset_val = AI4ArcticChallengeTestDataset(
        options=val_options, files=train_options['validate_list'], mode='train',
        ondemand_read=_od_ok)

    # prefetch_factor=1：一个验证样本是一整景（最大约 8 GB），默认的 2 会让
    # num_workers×2 个整景同时在内存里。
    _nw_val = train_options['num_workers_val']
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=_nw_val, shuffle=False,
        **({'prefetch_factor': 1} if _nw_val > 0 else {}))

    return dataloader_train, dataloader_val


def _set_global_seed(seed):
    """Seed python, numpy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_device(train_options):
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))

        # torch.compile speed-ups need compute capability >= 7.0.
        if train_options['compile_model']:
            gpu_ok = False
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                gpu_ok = True

            if not gpu_ok:
                warnings.warn(
                    colour_str("GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.", 'red')
                )

        device = torch.device(f"cuda:{train_options['gpu_id']}")
    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')
    return device


def _collect_cv_folds(train_options):
    """List the fold<K>_train.json / fold<K>_val.json pairs in cv_fold_dir, sorted by K."""
    fold_dir = train_options.get('cv_fold_dir', None)
    if not fold_dir:
        return []

    fold_dir_rel = fold_dir.replace('\\', '/').rstrip('/')
    fold_dir_abs = osp.join(train_options['path_to_env'], fold_dir_rel)
    if not osp.isdir(fold_dir_abs):
        raise FileNotFoundError(f'cv_fold_dir not found: {fold_dir_abs}')

    train_pat = re.compile(r'^fold(\d+)_train\.json$')
    fold_ids = []
    for name in os.listdir(fold_dir_abs):
        m = train_pat.match(name)
        if m is not None:
            fold_ids.append(int(m.group(1)))

    fold_ids = sorted(set(fold_ids))
    if not fold_ids:
        return []

    requested_num = int(train_options.get('cv_num_folds', 0) or 0)
    if requested_num > 0:
        fold_ids = fold_ids[:requested_num]

    folds = []
    for fold_id in fold_ids:
        train_rel = f'{fold_dir_rel}/fold{fold_id}_train.json'
        val_rel = f'{fold_dir_rel}/fold{fold_id}_val.json'
        train_abs = osp.join(train_options['path_to_env'], train_rel)
        val_abs = osp.join(train_options['path_to_env'], val_rel)
        if not osp.exists(train_abs):
            raise FileNotFoundError(f'Missing fold train list: {train_abs}')
        if not osp.exists(val_abs):
            raise FileNotFoundError(f'Missing fold val list: {val_abs}')
        folds.append({'fold': fold_id, 'train_list_path': train_rel, 'val_path': val_rel})

    return folds


def _read_json_list(path):
    with open(path) as f:
        return json.loads(f.read())


def build_seeded_model(train_options, device, reseed=True):
    """Build model, optimizer and scheduler, re-seeding before and after construction.

    Variants that build extra modules consume more global RNG. Seeding before keeps the
    shared weights identical across variants and folds; seeding again afterwards makes
    the data stream independent of the architecture. reseed=False (old behaviour) is
    for tools.checks.seed_stream only.
    """
    seed = train_options.get('_resolved_seed', None)
    if seed is not None:
        _set_global_seed(seed)

    net = get_model(train_options, device)
    if train_options['compile_model']:
        net = torch.compile(net)
    optimizer = get_optimizer(train_options, net)
    scheduler = get_scheduler(train_options, optimizer)

    if seed is not None and reseed:
        _set_global_seed(seed)
        print(f'[Seed] re-seeded to {seed} after model construction '
              '(data stream independent of architecture)')
    return net, optimizer, scheduler


def _run_training_job(cfg, args, train_options, device, fold_tag=None):
    net, optimizer, scheduler = build_seeded_model(train_options, device)

    epoch_start = 0
    # 顺序 CV 模式下不做 resume/finetune（每折都是全新训练）
    resume_state = None
    if fold_tag is None:
        if args.resume_from is not None:
            print(f"\033[91m Resuming work from {args.resume_from}\033[0m")
            epoch_start, resume_state = load_model(net, args.resume_from, optimizer, scheduler,
                                                   return_resume_state=True)
        elif args.finetune_from is not None:
            print(f"\033[91m Finetune model from {args.finetune_from}\033[0m")
            _ = load_model(net, args.finetune_from)

    run_name = args.wandb_name if fold_tag is None else f"{args.wandb_name}_fold{fold_tag}"
    if not train_options['cross_val_run']:
        wandb.init(name=run_name, project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options)
    else:
        wandb.init(name=run_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options)

    # Per-epoch curves; keep them out of the run summary.
    wandb.define_metric("Train Epoch Loss", summary="none")
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Train SupCon Epoch Loss", summary="none")
    for _k in ("SupCon Active Batch Frac", "SupCon Same-class Cos",
               "SupCon Diff-class Cos", "SupCon Cos Gap"):
        wandb.define_metric(_k, summary="none")
    wandb.define_metric("EMA Lag", summary="none")
    if len(train_options['charts']) > 1:
        wandb.define_metric(f"Train {train_options.get('target_chart', 'FLOE')} Loss", summary="none")
    wandb.define_metric("Train Water Consistency Epoch Loss", summary="none")
    wandb.define_metric("Validation Epoch Loss", summary="none")
    wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Validation Water Consistency Epoch Loss", summary="none")
    wandb.define_metric("Combined score", summary="none")
    wandb.define_metric("SIC r2_metric", summary="none")
    wandb.define_metric("SOD f1_metric", summary="none")
    wandb.define_metric("FLOE f1_metric", summary="none")
    wandb.define_metric(f"Validation {train_options.get('target_chart', 'FLOE')} mIoU", summary="none")
    wandb.define_metric("Water Consistency Accuarcy", summary="none")
    wandb.define_metric("Learning Rate", summary="none")
    # Absolute path + explicit base path: a relative config path outside the repo
    # (e.g. ../tmp/x.py) would otherwise make wandb refuse to save it.
    _cfg_abs = osp.abspath(str(args.config))
    wandb.save(_cfg_abs, base_path=osp.dirname(_cfg_abs))
    print(colour_str('Save Config File', 'green'))

    create_train_validation_and_test_scene_list(train_options)
    dataloader_train, dataloader_val = create_dataloaders(train_options)

    wandb.config['validate_list'] = train_options['validate_list']
    print('Data setup complete.')

    print('-----------------------------------')
    print('Starting Training')
    print('-----------------------------------')
    if args.resume_from is not None and fold_tag is None:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler, epoch_start, resume_state=resume_state)
    else:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler)

    print('-----------------------------------')
    print('Training Complete')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Staring Validation with best model')
    print('-----------------------------------')

    del dataloader_train, dataloader_val
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # The best checkpoint is evaluated separately with tools.evaluate.
    print('-----------------------------------')
    print('Completed validation')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Starting testing with best model')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Completed testing')
    print('-----------------------------------')

    wandb.finish()

    # Fold-level cleanup; global scene cache stays alive for next fold.
    del net, optimizer, scheduler
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # 必须在任何 torch CPU 算子之前调用：容器里 nproc 报宿主机核数而 cgroup 配额小得多，
    # 线程池超订会让 CPU 侧算子慢几百倍。详见 loaders.limit_torch_threads。
    limit_torch_threads()

    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = copy.deepcopy(cfg.train_options)
    train_options = get_variable_options(train_options)
    id = wandb.util.generate_id()

    if train_options['seed'] != -1 or args.seed is not None:
        if args.seed is not None:
            seed = int(args.seed)
        else:
            seed = train_options['seed']
        _set_global_seed(seed)
        train_options['_resolved_seed'] = int(seed)
        print(f"Seed: {seed}")
    else:
        train_options['_resolved_seed'] = None
        print("Random Seed Chosen")

    # --fold K: one fold of the frozen grouped CV split.
    if args.fold is not None:
        if train_options.get('sequential_cv', False):
            raise ValueError('--fold and sequential_cv are mutually exclusive')
        fold_dir = train_options.get('cv_fold_dir') or 'datalists/cv5'
        fold_dir = fold_dir.replace('\\', '/').rstrip('/')
        fold_train = f'{fold_dir}/fold{args.fold}_train.json'
        fold_val = f'{fold_dir}/fold{args.fold}_val.json'
        for rel in (fold_train, fold_val):
            if not osp.isfile(osp.join(train_options['path_to_env'], rel)):
                raise FileNotFoundError(f'--fold {args.fold}: missing {rel}')
        train_options['train_list_path'] = fold_train
        train_options['val_path'] = fold_val
        # No independent test set: test_path is a placeholder, keep it on the fold's val list.
        train_options['test_path'] = fold_val
        print(f'[CV] fold {args.fold}: train={fold_train} val={fold_val}')
    elif (not train_options.get('sequential_cv', False)
          and 'train_417' in str(train_options.get('train_list_path', ''))):
        print(colour_str('[WARN] train_417/val_417 split: 14 of 42 val scenes are adjacent '
                         'frames of a train datatake (leaky). Not for paper results; '
                         'use --fold K.', 'red'))

    # work_dir priority: CLI > config file > config filename
    if args.work_dir is not None:
        base_work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        if not train_options['cross_val_run']:
            base_work_dir = osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0])
        else:
            run_name = id
            base_work_dir = osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0], run_name)
    else:
        base_work_dir = cfg.work_dir

    ic(base_work_dir)
    mkdir_or_exist(osp.abspath(base_work_dir))
    shutil.copy(args.config, osp.join(base_work_dir, osp.basename(args.config)))

    device = _setup_device(train_options)

    # Sequential CV: one process, one-time preload, loop over folds.
    if train_options.get('sequential_cv', False):
        if args.resume_from is not None or args.finetune_from is not None:
            raise ValueError('sequential_cv mode does not support --resume-from/--finetune-from')

        folds = _collect_cv_folds(train_options)
        if not folds:
            raise ValueError('No folds discovered. Please set train_options["cv_fold_dir"] correctly.')

        print(f'Sequential CV mode enabled. Total folds: {len(folds)}')

        # Optional one-time preload for all fold train scenes.
        if train_options.get('preload_all_cv_scenes', True):
            all_train_files = []
            for fold in folds:
                train_json_abs = osp.join(train_options['path_to_env'], fold['train_list_path'])
                all_train_files.extend(_read_json_list(train_json_abs))
            print(f'Preloading unique train scenes across folds: {len(set(all_train_files))}')
            preload_scene_cache(train_options, all_train_files)

        for fold in folds:
            fold_id = fold['fold']
            print('===================================')
            print(f'Starting fold {fold_id}')
            print('===================================')

            fold_cfg = cfg.deepcopy()
            fold_cfg.work_dir = osp.join(base_work_dir, f'fold{fold_id}')
            mkdir_or_exist(osp.abspath(fold_cfg.work_dir))
            shutil.copy(args.config, osp.join(fold_cfg.work_dir, osp.basename(args.config)))

            fold_options = copy.deepcopy(train_options)
            fold_options['train_list_path'] = fold['train_list_path']
            fold_options['val_path'] = fold['val_path']

            _run_training_job(fold_cfg, args, fold_options, device, fold_tag=fold_id)

            print('===================================')
            print(f'Completed fold {fold_id}')
            print('===================================')
    else:
        cfg.work_dir = base_work_dir
        _run_training_job(cfg, args, train_options, device)


if __name__ == '__main__':
    main()
