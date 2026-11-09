#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training, inference and metric helpers used by tools/train.py, tools/evaluate.py
and floe/evaluation.py. Derived from the AutoICE challenge toolkit."""

# -- File info -- #
__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

# -- Built-in modules -- #
import importlib
import os
import json
import warnings
# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from torchmetrics.functional import r2_score, f1_score, jaccard_index
from torchmetrics.functional.classification import multiclass_confusion_matrix
import segmentation_models_pytorch as smp
from tqdm import tqdm
# -- Proprietary modules -- #
from floe.utils import ICE_STRINGS, GROUP_NAMES


def chart_cbar(ax, n_classes, chart, cmap='vridis'):
    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    n_labels = len(GROUP_NAMES[chart])  # number of labelled (non-mask) classes
    arranged = np.arange(0, n_labels + 1)
    cmap = plt.get_cmap(cmap, n_labels)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the boundary sentinel.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.set_label(label=ICE_STRINGS[chart])
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))


def compute_metrics(true, pred, charts, metrics, num_classes):
    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = torch.round(metrics[chart]['func'](
                true=true[chart], pred=pred[chart], num_classes=num_classes[chart]) * 100, decimals=3)

        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} \
                and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")

    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores


def r2_metric(true, pred, num_classes=None):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.

    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(preds=pred, target=true)

    return r2


def f1_metric(true, pred, num_classes):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.

    """
    f1 = f1_score(target=true, preds=pred, average='weighted', task='multiclass', num_classes=num_classes)

    return f1


def water_edge_metric(outputs, options):
    # 跨任务水体一致性需要 3 个 chart；单任务模式返回 0。
    if len(options['charts']) < 3:
        return torch.tensor(0.0)

    # Convert ouput into water and not water
    for chart in options['charts']:
        outputs[chart] = torch.where(outputs[chart] > 0.0, 1.0, 0.0)

    water_edge_accuracy = 1 - torch.mean(torch.abs(outputs[options['charts'][0]]-outputs[options['charts'][1]])
                                         + torch.abs(outputs[options['charts'][1]]-outputs[options['charts'][2]])
                                         + torch.abs(outputs[options['charts'][2]]-outputs[options['charts'][0]]))
    return water_edge_accuracy


def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.

    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']

    return torch.round(combined_metric / sum_weight, decimals=3)


# -- functions to save models -- #
def save_best_model(cfg, train_options: dict, net, optimizer, scheduler, epoch: int,
                    resume_state: dict = None):
    """Save model, optimizer and scheduler state to <work_dir>/best_model_<name>.pth.

    resume_state carries what --resume-from needs to continue seamlessly (best smoothed
    mIoU, the tail of the smoothing window, the early-stopping counter).
    """
    print('saving model....')
    config_file_name = os.path.basename(cfg.work_dir)
    torch.save(obj={'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    # load_model 据此校验：SequentialLR（warmup）与纯 CosineAnnealingLR
                    # 的 state_dict 结构不同，交叉加载会静默错调度。
                    'scheduler_type': type(scheduler).__name__,
                    'epoch': epoch,
                    'train_options': train_options,
                    'resume_state': resume_state,
                    },
               f=os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth'))
    print(f"model saved successfully at {os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')}")

    return os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')


def load_model(net, checkpoint_path, optimizer=None, scheduler=None,
               return_resume_state=False):
    """Load state from a checkpoint into net (and optimizer/scheduler if given).

    Returns the stored epoch, or (epoch, resume_state) if return_resume_state; resume_state
    is None for checkpoints written before it was saved.
    """
    # 存的是完整 dict（含 train_options），需要 weights_only=False；旧版 PyTorch 没有该参数。
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        # scheduler 类型不一致时 load_state_dict 一个方向 KeyError、另一个方向静默错调度。
        # 没有 scheduler_type 的旧 checkpoint 跳过校验（它们都是纯 cosine）。
        ck_type = checkpoint.get('scheduler_type')
        cur_type = type(scheduler).__name__
        if ck_type is not None and ck_type != cur_type:
            raise ValueError(
                f"--resume-from scheduler 类型不匹配：checkpoint 是 {ck_type}，当前配置是 "
                f"{cur_type}。warmup 配置（SequentialLR）与纯 cosine 的 state_dict 结构不同，"
                f"必须用相同 scheduler.warmup_epochs 的 config 恢复。")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    if return_resume_state:
        return epoch, checkpoint.get('resume_state')
    return epoch

def rand_bbox(size, lam):
    '''
    Given the 4D dimensions of a batch (size), and the ratio
    of the spatial dimension (lam) to be cut, returns a bounding box coordinates
    used for cutmix

    Parameters
    ----------
    size : 4D shape of the batch (N, C, H, W)
    lam : Ratio (portion) of the input to be cutmix'd

    Returns
    ----------
    Bounding box (x1, y1, x2, y2)
    '''
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform
    cx = np.random.randint(H)
    cy = np.random.randint(W)

    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)

    return bbx1, bby1, bbx2, bby2


def slide_inference(img, net, options, mode):
    """Sliding-window inference with overlap; mode ('val' or 'test') picks the stride.

    img is (N, C, H, W); returns dict[chart] -> (N, n_classes, H, W) averaged logits
    on img.device.
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise ValueError('Unrecognized mode')

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    batch_size, _, h_img, w_img = img.size()
    device = img.device
    charts = options['charts']
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # 累积张量放 CPU，避免大场景下 GPU OOM；每个 crop 的 forward 仍在 GPU 上执行
    preds = {chart: torch.zeros((batch_size, options['n_classes'][chart], h_img, w_img),
                                dtype=img.dtype) for chart in charts}
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), dtype=img.dtype)

    total_patches = h_grids * w_grids
    patch_bar = tqdm(total=total_patches, desc='slide patches', leave=False,
                     disable=total_patches < 20)
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            patch_bar.update(1)
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_img_size = crop_img.size()
            crop_height_pad = max(options['patch_size'] - crop_img_size[2], 0)
            crop_width_pad  = max(options['patch_size'] - crop_img_size[3], 0)

            if crop_height_pad > 0 or crop_width_pad > 0:
                crop_img = torch.nn.functional.pad(
                    crop_img, (0, crop_width_pad, 0, crop_height_pad), mode='constant', value=0)

            crop_seg_logit = net(crop_img)

            for chart in charts:
                logit = crop_seg_logit[chart]
                if crop_height_pad > 0:
                    logit = logit[:, :, :-crop_height_pad, :]
                if crop_width_pad > 0:
                    logit = logit[:, :, :, :-crop_width_pad]
                preds[chart][:, :, y1:y2, x1:x2] += logit.cpu()

            count_mat[:, :, y1:y2, x1:x2] += 1

    patch_bar.close()
    assert (count_mat == 0).sum() == 0

    return {chart: (preds[chart] / count_mat).to(device) for chart in charts}


class Slide_patches_index(data.Dataset):
    def __init__(self, h_img, w_img, h_crop, w_crop, h_stride, w_stride):
        super(Slide_patches_index, self).__init__()

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        self.patches_list = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                self.patches_list.append((y1, y2, x1, x2))

    def __getitem__(self, index):
        return self.patches_list[index]

    def __len__(self):
        return len(self.patches_list)


class Take_crops(data.Dataset):
    def __init__(self, img, patches):
        super(Take_crops, self).__init__()

        self.img = img
        self.patches = patches

    def __getitem__(self, index):
        y1, y2, x1, x2 = self.patches[index]

        return self.img[:, y1:y2, x1:x2]

    def __len__(self):
        return len(self.patches)


def batched_slide_inference(img, net, options, mode):
    """Sliding-window inference that batches the crops through a DataLoader.

    Only the first image of the batch is used. Returns dict[chart] -> (1, n_classes, H, W).
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise ValueError('Unrecognized mode')

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    # Pad so that the crop grid covers the image exactly.
    _, _, h_img, w_img = img.size()
    height_pad = h_crop - h_img if h_img - h_crop < 0 else \
        (h_stride - (h_img - h_crop) % h_stride) % h_stride
    width_pad = w_crop - w_img if w_img - w_crop < 0 else \
        (w_stride - (w_img - w_crop) % w_stride) % w_stride
    if height_pad > 0 or width_pad > 0:
        img = torch.nn.functional.pad(
            img, (0, width_pad, 0, height_pad), mode='constant', value=0)

    _, _, h_img, w_img = img.size()
    indexes = Slide_patches_index(h_img, w_img, h_crop, w_crop, h_stride, w_stride)
    samples = Take_crops(img.detach().cpu().numpy()[0], indexes.patches_list)
    samples_dataloader = data.DataLoader(dataset=samples, batch_size=options['batch_size']*4,
                                         shuffle=False, num_workers=options['num_workers_val'])

    n_batches = len(samples_dataloader)
    data_iterator = iter(samples_dataloader)
    idx_iterator = iter(indexes)

    charts = options['charts']
    preds = {chart: img.new_zeros((options['n_classes'][chart], h_img, w_img))
             for chart in charts}
    count_mat = img.new_zeros((h_img, w_img))

    for i in range(n_batches):
        crop_imgs = next(data_iterator)
        crop_imgs = crop_imgs.to(img.device)

        crop_seg_logit = net(crop_imgs)

        for j in range(crop_imgs.shape[0]):
            y1, y2, x1, x2 = next(idx_iterator)

            for chart in charts:
                preds[chart][:, y1:y2, x1:x2] += crop_seg_logit[chart][j, :, 0:(y2-y1), 0:(x2-x1)]

            count_mat[y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0

    for chart in charts:
        preds[chart] = preds[chart] / count_mat

        # Remove the pad; guarded because slicing with :-0 gives an empty dimension.
        if height_pad > 0:
            preds[chart] = preds[chart][:, :-height_pad, :]
        if width_pad > 0:
            preds[chart] = preds[chart][:, :, :-width_pad]

        preds[chart] = preds[chart].unsqueeze(0)

    return preds


def compute_valid_tile_mask(valid_map, patch_size):
    """把像素级有效掩膜聚合成 tile 级：(H, W) bool → (n_h, n_w) bool。

    True = 该 tile 内至少有一个参与打分的像元。验证指标只统计 `~cfv_masks` 的像素，
    loss 是 ignore_index=255 的 masked mean，所以 False 的 tile 对混淆矩阵和 loss 的
    贡献严格为 0，跳过它的前向不改变任何指标（验证集上约 2/3 的 tile 属于此类）。

    tile 网格与 fast_tiled_val_inference 一致：右/下 pad 到 patch_size 的整数倍，
    pad 区没有标签，计为无效。
    """
    if not torch.is_tensor(valid_map):
        valid_map = torch.from_numpy(np.asarray(valid_map))
    valid_map = valid_map.detach().to('cpu').bool().reshape(
        valid_map.shape[-2], valid_map.shape[-1])
    H, W = valid_map.shape
    n_h = -(-H // patch_size)      # ceil
    n_w = -(-W // patch_size)
    if n_h * patch_size != H or n_w * patch_size != W:
        buf = torch.zeros((n_h * patch_size, n_w * patch_size), dtype=torch.bool)
        buf[:H, :W] = valid_map
        valid_map = buf
    return valid_map.reshape(n_h, patch_size, n_w, patch_size).any(dim=3).any(dim=1)


def _val_tile_ids(valid_tile_mask, n_h, n_w):
    """tile 门控掩膜 → 需要前向的 tile 编号（行优先）。"""
    if valid_tile_mask is None:
        return torch.arange(n_h * n_w)
    if tuple(valid_tile_mask.shape) != (n_h, n_w):
        raise ValueError(
            f'valid_tile_mask 形状 {tuple(valid_tile_mask.shape)} 与 tile 网格 '
            f'({n_h}, {n_w}) 不符——多半是标签与输入尺寸不一致，或 patch_size 传错。')
    return torch.nonzero(valid_tile_mask.reshape(-1), as_tuple=False).squeeze(1)


def _gather_val_tiles(src, ids, n_w, patch_size, fill, pinned=False):
    """按 tile 编号从 (C, H, W) 或 (H, W) 取出一批 tile，右/下不足处用 fill 补齐。

    刻意不预先 pad 整景：最大的验证景 8ch×6767×37468 约 8.1GB，pad 会再复制一份，
    正好叠在验证的内存峰值上。这里每批只实体化 len(ids) 个 tile。

    pinned=True 时批分配在页锁定内存上，`.to(device, non_blocking=True)` 才是真正
    异步的 H2D；可分页内存的 H2D 会阻塞宿主，「CPU 取下一批」与「GPU 算当前批」
    就退化成串行。pinned 块由 CachingHostAllocator 管理，挂起的异步拷贝完成前不会
    被复用，无需手动双缓冲。仅在 CUDA 可用时传 True。
    """
    ps = patch_size
    H, W = src.shape[-2], src.shape[-1]
    shape = ((len(ids), ps, ps) if src.dim() == 2
             else (len(ids), src.shape[0], ps, ps))
    if pinned:
        # src 若在 GPU，会静默退化成每个 tile 一次 D2H。
        assert src.device.type == 'cpu', \
            f'_gather_val_tiles(pinned=True) 要求 src 在 CPU 上，得到 {src.device}'
        out = torch.full(shape, fill, dtype=src.dtype, pin_memory=True)
    else:
        out = src.new_full(shape, fill)
    for k, t in enumerate(ids.tolist()):
        i, j = divmod(int(t), n_w)
        y0, x0 = i * ps, j * ps
        y1, x1 = min(y0 + ps, H), min(x0 + ps, W)
        out[k, ..., :y1 - y0, :x1 - x0] = src[..., y0:y1, x0:x1]
    return out


# 「Σ逐像素 loss / 有效像元数」形式的 loss 类名。只有这类 loss 能按每批的有效像元数
# 加权聚合出与整景一次算完相同的值：
#   - losses.FocalLoss / losses.GCELoss: `loss.sum() / valid_mask.sum()`，带类别权重也成立；
#   - losses.OrderedCrossEntropyLoss: 先取出有效像元再 .mean()，同上；
#   - torch.nn.CrossEntropyLoss: reduction='mean' 的分母是 Σ w_{y_i}，带类别权重时不是
#     像素数，单独判定（见 _loss_is_mean_over_valid）；
#   - smp 的 Dice / Lovasz / Tversky / MCC: 集合级指标，不成立。
_VAL_LOSS_MEAN_OVER_VALID = ('FocalLoss', 'GCELoss', 'OrderedCrossEntropyLoss')
_VAL_LOSS_WARNED = set()


def _loss_is_mean_over_valid(fn):
    """该 loss 是否等于「Σ逐像素 loss / 有效像元数」（见 _VAL_LOSS_MEAN_OVER_VALID）。"""
    name = type(fn).__name__
    if name in _VAL_LOSS_MEAN_OVER_VALID:
        return True
    if name == 'CrossEntropyLoss':
        return (getattr(fn, 'weight', None) is None
                and getattr(fn, 'reduction', 'mean') == 'mean')
    return False


def fast_tiled_val_predict(img, net, options, labels, valid_maps=None,
                           loss_fns=None, task_weights=None, gate=True,
                           out='preds'):
    """验证用的流式 tile 推理：只前向含有效标签的 tile，且不实体化整景 logits。

    整景留在 CPU，每批 tile 上 GPU 后就地 argmax，只保留 (H, W) uint8 类别图。
    标签全为 ignore 的 tile 不前向：它们对混淆矩阵和 loss 的贡献严格为 0。

    与 fast_tiled_val_inference 的关系：tile 互不重叠，同一批组成下逐 tile logits
    逐位相同。门控会改变每批的组成，cuDNN/cuBLAS 可能选到不同算法，logits 有 ULP
    级差异，argmax 近乎平局的像素可能翻转（全量验证集上约 5e-7 的有效像元，
    ΔmIoU 约 2e-7，与改 val_batch_size 的抖动同量级）。gate=False 时批组成不变，
    预测逐像素相同（tools/checks/smoke_pipeline.py 第 [7a] 步）。

    Parameters
    ----------
    img : Tensor (1, C, H, W)，CPU（推荐）或 GPU。
    labels : dict[chart] -> (H, W) 整型标签（已完成 255 重映射）。
    valid_maps : dict[chart] -> (H, W) bool，True = 参与打分。缺省时用
        `labels != options['class_fill_values'][chart]`；可直接传 `~cfv_masks[chart]`。
    loss_fns : dict[chart] -> loss module，给了才算 loss。
    task_weights : 与 options['charts'] 对齐的权重，缺省全 1。
    gate : False 关闭 tile 门控（调试对拍用）。
    out : 'preds' 或 'confmat'。
        'confmat' 在 GPU 上逐批累加混淆矩阵（index_add_）和 float64 loss，整景只在
        返回前同步一次，CPU 可以在 GPU 前向期间准备后续批次。preds 模式每批有
        `.item()` 和 `.cpu()` 两次同步。同一 gate 下两种模式批组成相同，混淆矩阵与
        「preds + accumulate_confusion_matrix」整数级相等，loss 逐位一致
        （smoke_pipeline.py 第 [7c]/[7d] 步）。

    Returns
    -------
    preds : out='preds' 时为 dict[chart] -> (H, W) uint8 CPU 张量。被门控跳过的 tile
        为 0（打分时本就被排除；做可视化请传 gate=False）。
        out='confmat' 时为 dict[chart] -> (n_classes, n_classes) float64 GPU 张量，
        行 = true、列 = pred，只统计 valid_maps 为 True 的像素，可直接 `+=` 进
        new_confusion_matrix 的累加器。
    loss : float，`Σ(loss_b · n_valid_b) / Σ n_valid_b`。只有「Σ逐像素 loss / 有效
        像元数」形式的 loss 才与整景一次算完相等（见 _loss_is_mean_over_valid）；其余
        loss（带类别权重的 CrossEntropyLoss、Dice/Lovasz 等）warn 一次并返回 nan，
        而不是一个看似正常的错数。val loss 只用于日志，best model 按 mIoU 选。
    """
    if out not in ('preds', 'confmat'):
        raise ValueError(f"out 必须是 'preds' 或 'confmat'，得到 {out!r}")
    confmat_mode = (out == 'confmat')
    patch_size = options['patch_size']
    charts = options['charts']
    val_batch_size = options.get('val_batch_size', options['batch_size'] * 4)
    if task_weights is None:
        task_weights = [1] * len(charts)

    device = next(net.parameters()).device
    img_cpu = img.detach()
    if img_cpu.dim() == 4:
        img_cpu = img_cpu[0]
    img_cpu = img_cpu.to('cpu', copy=False)
    C, H, W = img_cpu.shape
    n_h = -(-H // patch_size)
    n_w = -(-W // patch_size)

    lab_t, valid_t = {}, {}
    for chart in charts:
        lab = labels[chart]
        if not torch.is_tensor(lab):
            lab = torch.from_numpy(np.asarray(lab))
        lab = lab.detach().to('cpu').reshape(H, W)
        lab_t[chart] = lab
        if valid_maps is not None and valid_maps.get(chart) is not None:
            v = valid_maps[chart]
            if not torch.is_tensor(v):
                v = torch.from_numpy(np.asarray(v))
            valid_t[chart] = v.detach().to('cpu').bool().reshape(H, W)
        else:
            valid_t[chart] = lab != options['class_fill_values'][chart]

    # 任一 chart 有有效像元的 tile 都要前向。
    if gate:
        tile_mask = None
        for chart in charts:
            m = compute_valid_tile_mask(valid_t[chart], patch_size)
            tile_mask = m if tile_mask is None else (tile_mask | m)
    else:
        tile_mask = None
    ids_all = _val_tile_ids(tile_mask, n_h, n_w)

    if confmat_mode:
        preds = None
        # 展平的 int64 累加器：n² 个合法 bin + 1 个哨兵 bin，整数加法精确。返回前切掉
        # 哨兵并转成 (n, n) float64，与 new_confusion_matrix / cm_* 的约定一致。
        cms = {chart: torch.zeros(options['n_classes'][chart] ** 2 + 1,
                                  dtype=torch.int64, device=device)
               for chart in charts}
        # index_add_ 的全 1 source，按实际最大批尺寸预分配一次（门控后 tile 数可能远小于
        # val_batch_size）。
        ones_flat = torch.ones(
            min(val_batch_size, max(len(ids_all), 1)) * patch_size * patch_size,
            dtype=torch.int64, device=device)
        # float64 累加，与 preds 模式的 python float 累加同序同操作数，结果逐位一致。
        loss_num_t = {chart: torch.zeros((), dtype=torch.float64, device=device)
                      for chart in charts}
        loss_den_t = {chart: torch.zeros((), dtype=torch.float64, device=device)
                      for chart in charts}
    else:
        preds = {chart: torch.zeros((H, W), dtype=torch.uint8) for chart in charts}
        for chart in charts:
            if options['n_classes'][chart] > 255:
                raise NotImplementedError(
                    f"n_classes[{chart}]={options['n_classes'][chart]} > 255，预测图存不进 uint8。")

    # 逐 chart 分别累加「loss × 有效像元数」与「有效像元数」，最后各自求均值再加权求和，
    # 与整景一次算完的 Σ_c w_c · mean_c 对应（各 chart 的有效像元集合不同，不能混算）。
    loss_num = {chart: 0.0 for chart in charts}
    loss_den = {chart: 0 for chart in charts}
    loss_ok = {}
    for chart in charts:
        if loss_fns is None or chart not in loss_fns:
            continue
        loss_ok[chart] = _loss_is_mean_over_valid(loss_fns[chart])
        if not loss_ok[chart] and chart not in _VAL_LOSS_WARNED:
            _VAL_LOSS_WARNED.add(chart)
            warnings.warn(
                f"[fast_tiled_val_predict] {chart} 的 loss "
                f"{type(loss_fns[chart]).__name__} 不是「Σ逐像素 loss / 有效像元数」形式"
                f"（带类别权重的 CrossEntropyLoss、Dice/Lovasz 等），无法按批聚合出与整景"
                f"一致的值 → 本次验证的 val loss 记为 nan。指标（mIoU/F1）不受影响。",
                RuntimeWarning)
    texture_on = options.get('texture_channels', False)
    # 只在 confmat 模式用 pinned + non_blocking：preds 模式每批本来就有 .cpu()/.item() 同步。
    pinned = confmat_mode and device.type == 'cuda'
    for start in range(0, len(ids_all), val_batch_size):
        ids = ids_all[start:start + val_batch_size]
        batch = _gather_val_tiles(img_cpu, ids, n_w, patch_size, 0,
                                  pinned=pinned).to(device, non_blocking=pinned)
        # 纹理通道按 tile 在 GPU 上计算，与训练时 _augment_x 的 patch 同口径；整景 loader
        # 只读原始通道，保持按需读取和低内存。
        if texture_on:
            from floe.loaders import _texture_channels  # floe.loaders 导入本模块，避免循环导入
            batch = torch.cat([batch, _texture_channels(batch)], dim=1)
        batch_out = net(batch)

        for chart, weight in zip(charts, task_weights):
            logits = batch_out[chart]
            if logits.dim() != 4 or logits.shape[1] != options['n_classes'][chart]:
                raise NotImplementedError(
                    f'fast_tiled_val_predict 只支持 (B, n_classes, H, W) 的分割输出，'
                    f'{chart} 得到 {tuple(logits.shape)}（回归头请走 '
                    f'fast_tiled_val_inference）。')

            if confmat_mode:
                # 本分支没有同步点：标签/掩膜 tile 上 GPU，loss 与 CM 都在 GPU 上累加。
                fill = options['class_fill_values'][chart]
                lab_b = _gather_val_tiles(lab_t[chart], ids, n_w, patch_size, fill,
                                          pinned=pinned)
                lab_b = lab_b.to(device, non_blocking=pinned).long()
                # 打分像素严格按 valid_t 取；pad 区填 False，自然被排除。
                vmap_b = _gather_val_tiles(valid_t[chart], ids, n_w, patch_size, False,
                                           pinned=pinned)
                vmap_b = vmap_b.to(device, non_blocking=pinned)

                if loss_ok.get(chart, False):
                    # loss 掩膜与 preds 模式相同（label != fill）。detach：调用方若忘了
                    # no_grad，累加器会挂住整景所有批的计算图。
                    n_valid = (lab_b != fill).sum()
                    contrib = (loss_fns[chart](logits, lab_b).detach().double()
                               * n_valid.double())
                    # 整批无有效像元时 loss 是 0/0 = nan，nan*0 仍是 nan，用 where 归零
                    # （等价于 preds 模式的 `if n_valid > 0`）。
                    loss_num_t[chart] += torch.where(
                        n_valid > 0, contrib, torch.zeros_like(contrib))
                    loss_den_t[chart] += n_valid.double()

                n_c = options['n_classes'][chart]
                p = class_decider(logits, options, chart)
                p = p.reshape(-1, patch_size, patch_size).long()
                # 输出空间尺寸 ≠ patch_size 时 reshape 的第 0 维不再是批大小，下游 index_add_
                # 只会报含糊的 numel mismatch。
                assert p.shape[0] == len(ids), \
                    f'预测 tile 数 {p.shape[0]} != 批内 tile 数 {len(ids)}——模型输出的空间尺寸与 patch_size 不符？'
                # 与 accumulate_confusion_matrix 同口径：除 valid 掩膜外还丢弃越界标签与越界
                # 预测（class_decider 回归分支的 clamp 上界是 n_classes 而不是 n_c-1，p == n_c
                # 会错记进下一行的 bin），被丢弃的像素指到第 n² 个哨兵 bin。
                # 不用 torch.bincount：它在 CUDA 上要做 min/max 归约并取回 CPU，每批同步两次。
                # index_add_ 是原子整数加，无同步、与顺序无关、结果精确。
                keep = vmap_b & (lab_b >= 0) & (lab_b < n_c) & (p < n_c)
                idx = torch.where(keep, lab_b * n_c + p,
                                  torch.full_like(lab_b, n_c * n_c))
                cms[chart].index_add_(0, idx.reshape(-1), ones_flat[:idx.numel()])
                continue

            if loss_ok.get(chart, False):
                fill = options['class_fill_values'][chart]
                lab_b = _gather_val_tiles(lab_t[chart], ids, n_w, patch_size, fill)
                lab_b = lab_b.to(device, non_blocking=True).long()
                n_valid = int((lab_b != fill).sum().item())
                if n_valid > 0:
                    loss_num[chart] += loss_fns[chart](logits, lab_b).item() * n_valid
                    loss_den[chart] += n_valid

            p = class_decider(logits, options, chart).reshape(-1, patch_size, patch_size)
            # 空间尺寸不符时 p[k] 会静默取错行。
            assert p.shape[0] == len(ids), \
                f'预测 tile 数 {p.shape[0]} != 批内 tile 数 {len(ids)}——模型输出的空间尺寸与 patch_size 不符？'
            p = p.to(torch.uint8).cpu()
            for k, t in enumerate(ids.tolist()):
                i, j = divmod(int(t), n_w)
                y0, x0 = i * patch_size, j * patch_size
                y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
                preds[chart][y0:y1, x0:x1] = p[k, :y1 - y0, :x1 - x0]

    if confmat_mode:
        # 整景唯一的同步点。CM 留在 GPU，调用方直接 += 进全局累加器。
        for chart in charts:
            loss_num[chart] = float(loss_num_t[chart].item())
            loss_den[chart] = float(loss_den_t[chart].item())

    if loss_ok and not all(loss_ok.values()):
        loss = float('nan')
    else:
        loss = 0.0
        for chart, weight in zip(charts, task_weights):
            if loss_den[chart] > 0:
                loss += weight * (loss_num[chart] / loss_den[chart])

    if confmat_mode:
        out_cms = {}
        for chart in charts:
            n_c = options['n_classes'][chart]
            out_cms[chart] = cms[chart][:n_c * n_c].reshape(n_c, n_c).to(torch.float64)
        return out_cms, loss
    return preds, loss


def fast_tiled_val_inference(img, net, options, valid_tile_mask=None):
    """
    Fast validation inference using non-overlapping tiles with batched GPU forward passes.

    Splits the scene into non-overlapping patch_size×patch_size tiles, batches them
    together, and runs a single net() call per batch. Dramatically faster than
    sliding-window inference for full-resolution (scale=1) scenes:
      e.g. 2560×2560, patch_size=256, val_batch_size=256 → 1 forward pass/scene
           vs. 484 forward passes with stride-128 sliding window.

    Caller is responsible for wrapping in torch.no_grad() / autocast().

    Parameters
    ----------
    img : Tensor (1, C, H, W)  on the target device
    net : nn.Module
    options : dict
        Uses 'patch_size', 'charts', 'n_classes', 'batch_size',
        and optionally 'val_batch_size' (defaults to batch_size * 4).
    valid_tile_mask : Tensor (n_h, n_w) bool, optional
        Output of `compute_valid_tile_mask`. Tiles marked False are skipped and keep
        zero logits (argmax = class 0). Metrics are unchanged because those pixels are
        masked out anyway, but do not pass it when the output is visualised or exported.

    Returns
    -------
    dict[str, Tensor (1, n_classes[chart], H, W)]  on the same device as img.
    """
    patch_size = options['patch_size']
    charts = options['charts']
    val_batch_size = options.get('val_batch_size', options['batch_size'] * 4)
    device = img.device

    _, C, h_img, w_img = img.size()

    # Pad H and W to exact multiples of patch_size
    h_pad = (-h_img) % patch_size
    w_pad = (-w_img) % patch_size
    if h_pad > 0 or w_pad > 0:
        img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad), mode='constant', value=0)
    _, _, h_padded, w_padded = img.size()

    n_h = h_padded // patch_size
    n_w = w_padded // patch_size

    # (C, h_padded, w_padded) → (n_h, n_w, C, ps, ps) 的零拷贝视图，每批再按需实体化。
    # 不要写成 `.reshape(n_tiles, C, ps, ps)`：permute 后张量非连续，reshape 会把整景
    # 再复制一份（最大验证景约 4.9GB），正好叠在验证的内存峰值上。
    img_cpu = img[0].cpu()
    tiles_view = (img_cpu
                  .reshape(C, n_h, patch_size, n_w, patch_size)
                  .permute(1, 3, 0, 2, 4))          # (n_h, n_w, C, ps, ps)，视图

    # Prediction buffers on CPU; only each batch touches the GPU
    preds = {chart: torch.zeros(options['n_classes'][chart], h_padded, w_padded)
             for chart in charts}

    # 未传掩膜时前向全部 tile。
    tile_ids = _val_tile_ids(valid_tile_mask, n_h, n_w)

    for start in range(0, len(tile_ids), val_batch_size):
        end = min(start + val_batch_size, len(tile_ids))
        idx = tile_ids[start:end]       # 行优先：tile_idx = h_idx * n_w + w_idx
        h_idx = torch.div(idx, n_w, rounding_mode='floor')
        w_idx = idx % n_w
        # 高级索引只实体化这一批 tile，不是整景。
        batch = tiles_view[h_idx, w_idx].contiguous().to(device)
        # 纹理通道按 tile 计算，与 fast_tiled_val_predict 和训练同口径。
        if options.get('texture_channels', False):
            from floe.loaders import _texture_channels  # floe.loaders 导入本模块，避免循环导入
            batch = torch.cat([batch, _texture_channels(batch)], dim=1)
        batch_out = net(batch)

        for local_j in range(end - start):
            y1 = int(h_idx[local_j]) * patch_size
            x1 = int(w_idx[local_j]) * patch_size
            for chart in charts:
                preds[chart][:, y1:y1 + patch_size, x1:x1 + patch_size] = \
                    batch_out[chart][local_j].detach().cpu()

    # Trim padding, restore batch dim, return on original device
    return {chart: preds[chart][:, :h_img, :w_img].unsqueeze(0).to(device)
            for chart in charts}


def class_decider(output, train_options, chart):
    """Logits -> class map. A trailing size-1 dim marks a regression head (round + clamp)."""
    if (train_options['binary_water_classifier'] == False):
        if output.size(3) == 1:
            output = torch.round(output.squeeze())
            output = torch.clamp(output, min=0, max=train_options
                                 ['n_classes'][chart])
        else:
            output = torch.argmax(output, dim=1).squeeze()
        return output

    # Binary water classifier: water vs. the summed ice classes, then argmax over ice.
    else:
        probability = torch.nn.Softmax(dim=1)(output)
        water = probability[:, 0, :, :]
        not_water = torch.sum(probability, dim=1) - water
        class_output = water <= not_water
        without_water = probability[:, 1:, :, :]
        class_output_without_water = torch.argmax(without_water, dim=1) + 1
        class_output = class_output_without_water * class_output

        return class_output.squeeze()


def compute_classwise_f1score(true, pred, charts, num_classes):
    """ This function computes the classwise evaluation score for each task and stores them in a dic

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns score_dictionary
    """
    score = {}
    for chart in charts:
        score[chart] = f1_score(target=true[chart], preds=pred[chart], average='none',
                                task='multiclass', num_classes=num_classes[chart])
    return score


def compute_overall_accuracy(true, pred, charts):
    """ Computes Overall Accuracy (OA) for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts

    Returns:
        dictionary: returns oa_score_dictionary
    """
    scores = {}
    for chart in charts:
        if true[chart].numel() == 0:
            scores[chart] = 0.0
        else:
            correct = (true[chart] == pred[chart]).sum()
            total = true[chart].numel()
            scores[chart] = (correct.float() / total).item()
    return scores


def compute_mIoU(true, pred, charts, num_classes):
    """ Computes Mean Intersection over Union (mIoU) for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns mIoU_score_dictionary
    """
    scores = {}
    for chart in charts:
        # average 默认为 'macro'，即 mIoU。
        scores[chart] = jaccard_index(target=true[chart], preds=pred[chart], task='multiclass', num_classes=num_classes[chart])
    return scores


def compute_ordinal_cost_metric(true, pred, charts, num_classes, cost_per_distance=None):
    """Ordinal Cost Metric for ordered classification (e.g. SOD thickness stages).

    Per-pixel cost is looked up from cost_per_distance[|pred - true|].
    Default is linear: cost = |pred - true|, which is consistent with EMDLoss.

    Parameters
    ----------
    cost_per_distance : list of numbers, length >= n_classes.
        cost_per_distance[d] is the penalty for a prediction that is d classes away.
        Must start with 0 (correct prediction costs nothing).
        Example (linear):  [0, 1, 2, 3, 4]
        Example (custom):  [0, 1, 2, 3, 3]  (cap at 3 for very distant errors)

    Returns
    -------
    raw_costs : dict[chart -> Tensor scalar]  mean cost per pixel (lower = better)
    norm_scores : dict[chart -> Tensor scalar]
        1 - mean_cost / max_cost, normalised to [0, 1] (higher = better).
    """
    raw_costs = {}
    norm_scores = {}
    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            raw_costs[chart] = torch.tensor(0.0)
            norm_scores[chart] = torch.tensor(1.0)
            continue

        # Build cost lookup table: index = distance, value = penalty
        if cost_per_distance is None:
            # Linear default: cost = |pred - true|, aligned with EMDLoss
            cost_table = torch.arange(n_classes, dtype=torch.float32, device=true_c.device)
        else:
            cost_table = torch.tensor(cost_per_distance, dtype=torch.float32, device=true_c.device)

        diff = torch.abs(true_valid - pred_valid)          # [N], integer distances
        cost = cost_table[diff]                            # lookup per-pixel cost

        max_cost = cost_table.max().clamp(min=1e-6)
        mean_cost = cost.mean()
        raw_costs[chart] = mean_cost
        norm_scores[chart] = 1.0 - mean_cost / max_cost

    return raw_costs, norm_scores


def compute_classwise_IoU(true, pred, charts, num_classes):
    """Computes per-class IoU for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: per-class IoU tensor for each chart
    """
    scores = {}
    for chart in charts:
        scores[chart] = jaccard_index(
            target=true[chart], preds=pred[chart],
            task='multiclass', num_classes=num_classes[chart],
            average='none'
        )
    return scores


def compute_classwise_precision_recall(true, pred, charts, num_classes):
    """Computes per-class precision and recall for each chart.

    Invalid labels (outside [0, num_classes-1]) are ignored.
    """
    precision_scores = {}
    recall_scores = {}

    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            precision_scores[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            recall_scores[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            continue

        cm = multiclass_confusion_matrix(
            preds=pred_valid,
            target=true_valid,
            num_classes=n_classes,
        ).float()

        tp = torch.diag(cm)
        pred_sum = cm.sum(dim=0)
        true_sum = cm.sum(dim=1)

        precision_scores[chart] = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))
        recall_scores[chart] = torch.where(true_sum > 0, tp / true_sum, torch.zeros_like(tp))

    return precision_scores, recall_scores


def compute_normalized_confusion_matrix(true, pred, charts, num_classes):
    """Computes row-normalized confusion matrix for each chart.

    Normalization is done by true-class row sums.
    """
    norm_cms = {}
    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            norm_cms[chart] = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=true_c.device)
            continue

        cm = multiclass_confusion_matrix(
            preds=pred_valid,
            target=true_valid,
            num_classes=n_classes,
        ).float()

        row_sum = cm.sum(dim=1, keepdim=True)
        norm_cms[chart] = torch.where(row_sum > 0, cm / row_sum, torch.zeros_like(cm))

    return norm_cms


def compute_class_proportions(true, pred, charts, num_classes):
    """Computes per-class true/pred pixel proportions for each chart."""
    true_props = {}
    pred_props = {}

    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            true_props[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            pred_props[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            continue

        true_hist = torch.bincount(true_valid, minlength=n_classes).float()
        pred_hist = torch.bincount(pred_valid, minlength=n_classes).float()
        denom = true_hist.sum().clamp_min(1.0)

        true_props[chart] = true_hist / denom
        pred_props[chart] = pred_hist / denom

    return true_props, pred_props


# Confusion-matrix based metrics.
# 上面的 compute_* 都是 (n, n) 混淆矩阵的纯函数。验证时逐景累加混淆矩阵再算指标，
# 显存与像素总数无关：原生分辨率下把全部有效像素 cat 成扁平张量再调用 compute_*，
# 加上 torchmetrics 内部的 int64 中间量，在 32GB 卡上会 OOM。
# 约定：cm[t, p] = 真实类别 t 被预测为 p 的像素数（行 = true，列 = pred）；
# 调用方需先剔除 255（验证循环里由 ~cfv_masks 完成）。

def new_confusion_matrix(num_classes, device):
    """float64 累加器：像素数可达 1e9，float32 的尾数只有 24 bit，累加会丢精度。"""
    return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)


def accumulate_confusion_matrix(cm, true, pred, num_classes, chunk=1 << 24):
    """把一景的 (true, pred) 累加进混淆矩阵。就地修改并返回 cm。

    分块 bincount：`true * num_classes + pred` 是 int64，单景有效像素可达 5000 万，
    一次性构造要 400MB。按 16M 元素分块把峰值压到 128MB 以内。
    """
    true = true.reshape(-1).long()
    pred = pred.reshape(-1).long()
    if true.numel() == 0:
        return cm

    valid = (true >= 0) & (true < num_classes) & (pred >= 0) & (pred < num_classes)
    if not bool(valid.all()):
        true = true[valid]
        pred = pred[valid]

    for i in range(0, true.numel(), chunk):
        t = true[i:i + chunk]
        p = pred[i:i + chunk]
        binned = torch.bincount(t * num_classes + p, minlength=num_classes ** 2)
        cm += binned.reshape(num_classes, num_classes).to(cm.dtype)
    return cm


def _cm_parts(cm):
    """返回 (tp, pred_sum, true_sum, total)，均为 float64 一维/标量张量。"""
    tp = torch.diag(cm)
    return tp, cm.sum(dim=0), cm.sum(dim=1), cm.sum()


def cm_classwise_f1(cm):
    """等价于 f1_score(average='none', task='multiclass')。分母为 0 的类记 0。"""
    tp, pred_sum, true_sum, _ = _cm_parts(cm)
    denom = pred_sum + true_sum
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return f1.float()


def cm_weighted_f1(cm):
    """等价于 f1_score(average='weighted')：按 true 支持度加权的 per-class F1。"""
    _, _, true_sum, total = _cm_parts(cm)
    if total <= 0:
        return torch.tensor(0.0)
    f1 = cm_classwise_f1(cm).double()
    return (f1 * true_sum / total).sum().float()


def cm_overall_accuracy(cm):
    total = cm.sum()
    if total <= 0:
        return 0.0
    return (torch.diag(cm).sum() / total).item()


def cm_classwise_iou(cm):
    """等价于 jaccard_index(average='none')。TP+FP+FN==0 的类（数据集中不存在）记 0。"""
    tp, pred_sum, true_sum, _ = _cm_parts(cm)
    denom = pred_sum + true_sum - tp
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return iou.float()


def cm_miou(cm):
    """等价于 jaccard_index(average='macro')：对所有类取均值，数据中不存在的类记 0 并计入。"""
    return cm_classwise_iou(cm).mean()


def cm_precision_recall(cm):
    tp, pred_sum, true_sum, _ = _cm_parts(cm)
    precision = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))
    recall = torch.where(true_sum > 0, tp / true_sum, torch.zeros_like(tp))
    return precision.float(), recall.float()


def cm_normalized(cm):
    """按真实类别行归一化。"""
    row_sum = cm.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, cm / row_sum, torch.zeros_like(cm)).float()


def cm_class_proportions(cm):
    """true / pred 各类占比，分母统一为 true 像素总数（与 compute_class_proportions 一致）。"""
    _, pred_sum, true_sum, total = _cm_parts(cm)
    denom = total.clamp_min(1.0)
    return (true_sum / denom).float(), (pred_sum / denom).float()


def cm_ordinal_cost(cm, cost_per_distance=None):
    """有序代价指标。mean_cost = Σ_{t,p} cm[t,p]·cost[|t-p|] / Σ cm。"""
    n = cm.shape[0]
    total = cm.sum()
    if total <= 0:
        return torch.tensor(0.0), torch.tensor(1.0)

    if cost_per_distance is None:
        cost_table = torch.arange(n, dtype=torch.float64, device=cm.device)
    else:
        cost_table = torch.tensor(cost_per_distance, dtype=torch.float64, device=cm.device)

    idx = torch.arange(n, device=cm.device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()
    cost_matrix = cost_table[dist]

    mean_cost = (cm * cost_matrix).sum() / total
    max_cost = cost_table.max().clamp(min=1e-6)
    return mean_cost.float(), (1.0 - mean_cost / max_cost).float()


def create_train_validation_and_test_scene_list(train_options):
    """Read the train/val/test scene lists (JSON lists of .nc filenames) into train_options.

    With cross_val_run the validation scenes are drawn with np.random, i.e. from the
    global NumPy RNG.
    """
    with open(train_options['path_to_env'] + train_options['train_list_path']) as file:
        train_options['train_list'] = json.loads(file.read())
    # Pooled training: scenes of other archives are appended to every fold's train list.
    for extra in train_options.get('extra_train_lists', ()):
        with open(train_options['path_to_env'] + extra) as file:
            train_options['train_list'] += json.loads(file.read())

    if train_options['cross_val_run']:
        train_options['validate_list'] = list(np.random.choice(np.array(
            train_options['train_list']), size=train_options['p-out'], replace=False))
    else:
        with open(train_options['path_to_env'] + train_options['val_path']) as file:
            train_options['validate_list'] = json.loads(file.read())

    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                   if scene not in train_options['validate_list']]

    with open(train_options['path_to_env'] + train_options['test_path']) as file:
        train_options['test_list'] = json.loads(file.read())
    # Labels live in the same .nc files, so the reference list is the test list.
    train_options['test_list_reference'] = train_options['test_list']

    print('Options initialised')


def get_scheduler(train_options, optimizer):
    if train_options['scheduler']['type'] == 'CosineAnnealingLR':
        T_max_epochs = train_options['scheduler'].get('T_max_epochs', train_options['epochs'])
        T_max = T_max_epochs * train_options['epoch_len']
        # scheduler 逐 batch 步进，所以 warmup 长度 = warmup_epochs × epoch_len 个 step；
        # 所有参数组按同一比例从 warmup_start_factor 爬到各自的基准 LR，cosine 的 T_max
        # 扣除 warmup 段，总长度不变。SequentialLR 的 state_dict 结构与纯 cosine 不同，
        # --resume-from 需用相同 warmup 配置的 checkpoint。
        warmup_epochs = train_options['scheduler'].get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_iters = warmup_epochs * train_options['epoch_len']
            assert warmup_iters < T_max, 'warmup_epochs 不能 >= 总调度长度'
            start_factor = train_options['scheduler'].get('warmup_start_factor', 0.01)
            linear = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max - warmup_iters, eta_min=train_options['scheduler']['lr_min'])
            print(f'[Scheduler] linear warmup {warmup_epochs} epochs '
                  f'({warmup_iters} steps, start_factor={start_factor}) -> cosine '
                  f'T_max={T_max - warmup_iters} steps', flush=True)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[linear, cosine], milestones=[warmup_iters])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                               eta_min=train_options['scheduler']['lr_min'])
    elif train_options['scheduler']['type'] == 'CosineAnnealingWarmRestartsLR':
        T_0 = train_options['scheduler']['EpochsPerRestart']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0,
                                                                         T_mult=train_options['scheduler']['RestartMult'],
                                                                         eta_min=train_options['scheduler']['lr_min'],
                                                                         last_epoch=-1)
    elif train_options['scheduler']['type'] == 'ReduceLROnPlateau':
        # 每个验证 epoch 后以 scheduler.step(val_score) 步进；监控的分数越大越好。
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=train_options['scheduler'].get('factor', 0.5),
            patience=train_options['scheduler'].get('patience', 5),
            min_lr=train_options['scheduler'].get('min_lr', 1e-5),
            threshold=train_options['scheduler'].get('threshold', 1e-4),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5, last_epoch=-1)
    return scheduler


def _optimizer_param_groups(train_options, net):
    """按 optimizer.encoder_lr_scale 把 encoder 参数单独分组并缩放 LR（判别式 LR）。

    用于预训练 encoder + 随机初始化 head（mmseg 的 SegFormer 配方是 backbone 1x /
    head 10x）。默认 1.0 为单组。参数名以 'encoder.' 开头或含 '.encoder.' 即归入
    encoder 组（SegFormer 的 self.encoder、SMP 系模型的 model.encoder 都命中）。
    optimizer 的 state_dict 与分组结构绑定，--resume-from 需用相同的 encoder_lr_scale。
    """
    base_lr = train_options['optimizer']['lr']
    scale = train_options['optimizer'].get('encoder_lr_scale', 1.0)
    if scale == 1.0:
        return [{'params': list(net.parameters())}]
    enc_params, rest_params = [], []
    for name, p in net.named_parameters():
        if name.startswith('encoder.') or '.encoder.' in name:
            enc_params.append(p)
        else:
            rest_params.append(p)
    if not enc_params:
        raise ValueError(
            "optimizer.encoder_lr_scale != 1.0 但没有任何参数名匹配 'encoder.'——"
            "该模型不适用此选项，检查 model_selection 或参数命名。")
    n_enc = sum(p.numel() for p in enc_params)
    n_rest = sum(p.numel() for p in rest_params)
    print(f"[Optimizer] encoder_lr_scale={scale}: encoder {n_enc/1e6:.1f}M params "
          f"lr={base_lr * scale:.2e} | rest {n_rest/1e6:.1f}M params lr={base_lr:.2e}")
    # base LR 组放首位：训练日志和 wandb 记录的是 param_groups[0]['lr']。
    return [{'params': rest_params},
            {'params': enc_params, 'lr': base_lr * scale}]


def get_optimizer(train_options, net):
    param_groups = _optimizer_param_groups(train_options, net)
    if train_options['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, lr=train_options['optimizer']['lr'],
                                     betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                     weight_decay=train_options['optimizer']['weight_decay'])

    elif train_options['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=train_options['optimizer']['lr'],
                                      betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                      weight_decay=train_options['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(param_groups, lr=train_options['optimizer']['lr'],
                                    momentum=train_options['optimizer']['momentum'],
                                    dampening=train_options['optimizer']['dampening'],
                                    weight_decay=train_options['optimizer']['weight_decay'],
                                    nesterov=train_options['optimizer']['nesterov'])
    return optimizer


def get_loss(loss, chart=None, **kwargs):
    """Build the loss named by a chart_loss 'type'; the other config keys go to its constructor."""
    if loss == 'DiceLoss':
        kwargs.pop('type')
        loss = smp.losses.DiceLoss(**kwargs)
    elif loss == 'FocalLoss':
        from floe.losses import FocalLoss as _FocalLoss
        kwargs.pop('type')
        loss = _FocalLoss(**kwargs)
    elif loss == 'JaccardLoss':
        raise NotImplementedError
    elif loss == 'LovaszLoss':
        kwargs.pop('type')
        loss = smp.losses.LovaszLoss(**kwargs)
    elif loss == 'MCCLoss':
        kwargs.pop('type')
        loss = smp.losses.MCCLoss(**kwargs)
    elif loss == 'SoftBCEWithLogitsLoss':
        raise NotImplementedError
    elif loss == 'SoftCrossEntropyLoss':
        raise NotImplementedError
    elif loss == 'TverskyLoss':
        kwargs.pop('type')
        loss = smp.losses.TverskyLoss(**kwargs)
    elif loss == 'CrossEntropyLoss':
        kwargs.pop('type')
        if 'weight' in kwargs and isinstance(kwargs['weight'], (list, tuple)):
            kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        loss = torch.nn.CrossEntropyLoss(**kwargs)
    elif loss == 'BinaryCrossEntropyLoss':
        raise NotImplementedError
    elif loss == 'OrderedCrossEntropyLoss':
        from floe.losses import OrderedCrossEntropyLoss
        kwargs.pop('type')
        loss = OrderedCrossEntropyLoss(**kwargs)
    elif loss == 'GCELoss':
        from floe.losses import GCELoss
        kwargs.pop('type')
        if 'weight' in kwargs and isinstance(kwargs['weight'], (list, tuple)):
            kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        loss = GCELoss(**kwargs)
    elif loss == 'EMDLoss':
        from floe.losses import EMDLoss
        kwargs.pop('type')
        loss = EMDLoss(**kwargs)
    elif loss == 'MSELossFromLogits':
        from floe.losses import MSELossFromLogits
        kwargs.pop('type')
        loss = MSELossFromLogits(chart=chart, **kwargs)
    elif loss == 'MSELoss':
        kwargs.pop('type')
        loss = torch.nn.MSELoss(**kwargs)
    elif loss == 'MSELossWithIgnoreIndex':
        from floe.losses import MSELossWithIgnoreIndex
        kwargs.pop('type')
        loss = MSELossWithIgnoreIndex(**kwargs)
    else:
        raise ValueError(f'The given loss \'{loss}\' is unrecognized or Not implemented')

    return loss


# model_selection -> (module, class). Every class is built as cls(options=train_options).
_MODELS = {
    'unet': ('floe.models.legacy.unet', 'UNet'),
    'swin': ('floe.models.legacy.swin_transformer', 'SwinTransformer'),
    'Separate_decoder': ('floe.models.legacy.unet', 'Sep_feat_dif_stages'),
    'UNet_regression': ('floe.models.legacy.unet', 'UNet_regression'),
    'unet_regression': ('floe.models.legacy.unet', 'UNet_regression'),
    'UNet_regression_all': ('floe.models.legacy.unet', 'UNet_regression_all'),
    'UNet_sep_dec_regression': ('floe.models.legacy.unet', 'UNet_sep_dec_regression'),
    'unet_sep_dec_regression': ('floe.models.legacy.unet', 'UNet_sep_dec_regression'),
    'UNet_sep_dec_mse': ('floe.models.legacy.unet', 'UNet_sep_dec_mse'),
    # Comparison models.
    'sota_unet': ('floe.models.sota.unet', 'UNet'),
    'sota_segnet': ('floe.models.sota.segnet', 'SegNet'),
    'sota_resnet': ('floe.models.sota.resnet', 'ResNetFPN'),
    'sota_densenet': ('floe.models.sota.densenet', 'DenseNetFPN'),
    'sota_pspnet': ('floe.models.sota.pspnet', 'PSPNet'),
    'sota_deeplabv3': ('floe.models.sota.deeplabv3', 'DeepLabV3Plus'),
    'sota_segformer': ('floe.models.sota.segformer', 'SegFormer'),
    'sota_poolformer': ('floe.models.sota.poolformer', 'PoolFormerFPN'),
    'sota_segnext': ('floe.models.sota.segnext', 'SegNeXt'),
}


def get_model(train_options, device):
    name = train_options['model_selection']
    if name not in _MODELS:
        raise ValueError(f"unknown model_selection {name!r}")
    module, cls_name = _MODELS[name]
    cls = getattr(importlib.import_module(module), cls_name)
    return cls(options=train_options).to(device)


def ema_update(ema_net, net, decay):
    """EMA 影子更新：浮点张量 v ← d·v + (1-d)·m，整型 buffer（如 BN num_batches_tracked）
    直接拷贝。在 state_dict 上操作，参数与 BN running stats 共用同一 EMA（同 timm ModelEmaV2）。
    """
    with torch.no_grad():
        msd = net.state_dict()
        for k, v in ema_net.state_dict().items():
            m = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(m, alpha=1.0 - decay)
            else:
                v.copy_(m)


def supcon_region_loss(feat, labels, n_classes, temperature=0.1, min_pixels=16,
                       return_stats=False):
    """区域级监督对比辅助损失（SupCon, Khosla et al. 2020）。

    把 batch 中每个 (样本, 类别) 的投影特征做掩膜均值池化成一个区域嵌入，同类跨样本
    拉近、异类推远。每个样本每类至多一个嵌入，所以正样本对必然来自不同 patch。
    只在训练路径使用，验证 loss 不含此项。

    参数：
        feat: [B, D, h, w] 投影特征（SegFormer 输出的 '_supcon'，autocast 下为 fp16）；
              内部全程 fp32。
        labels: [B, H, W]，255 = ignore；内部 nearest 下采样到 (h, w)。
        n_classes: 类别数。
        temperature: SupCon 温度 τ。
        min_pixels: (样本, 类别) 在 (h, w) 网格上的最少像素数，不足则不产出嵌入。
              钳到 >= 1，否则空区域除 0 得 NaN。
        return_stats: True 时返回 (loss, stats)。stats 为 python float/list（含 .item()
              同步）：active（本 batch 是否产生了有效损失）、regions、anchors、
              same_cos / diff_cos（同类跨样本 / 异类平均余弦，inactive 时为 nan）、
              regions_per_class（各类产出的嵌入数，长期为 0 说明被 min_pixels 挡掉）。
    返回：标量 fp32 loss，或 (loss, stats)。batch 内没有同类跨样本对时 loss 为 0
        （无梯度贡献）。
    """
    min_pixels = max(1, int(min_pixels))

    def _ret(loss, stats):
        return (loss, stats) if return_stats else loss

    def _empty_stats(per_class=None):
        return {'active': 0.0, 'regions': 0.0, 'anchors': 0.0,
                'same_cos': float('nan'), 'diff_cos': float('nan'),
                'regions_per_class': per_class if per_class is not None
                                     else [0.0] * int(n_classes)}

    B, D, h, w = feat.shape
    f = feat.float().flatten(2)                                   # [B, D, hw]
    with torch.no_grad():
        y = torch.nn.functional.interpolate(
            labels.float().unsqueeze(1), size=(h, w),
            mode='nearest').squeeze(1).long()                     # [B, h, w]，255 保持
        yf = y.flatten(1)                                         # [B, hw]
        valid = (yf >= 0) & (yf < n_classes)
        onehot = torch.zeros(B, n_classes, yf.shape[1],
                             device=feat.device, dtype=torch.float32)
        onehot.scatter_(1, yf.clamp(0, n_classes - 1).unsqueeze(1), 1.0)
        onehot = onehot * valid.unsqueeze(1).float()              # 255/越界 清零
        counts = onehot.sum(-1)                                   # [B, C]
        keep = counts >= float(min_pixels)                        # [B, C]
        per_class = keep.sum(dim=0).float().tolist()              # [C] 各类区域数
    if int(keep.sum().item()) < 2:
        return _ret(torch.zeros((), device=feat.device), _empty_stats(per_class))
    sums = torch.bmm(onehot, f.transpose(1, 2))                   # [B, C, D]
    emb = sums[keep] / counts[keep].unsqueeze(-1)                 # [N, D]
    cls = keep.nonzero(as_tuple=False)[:, 1]                      # [N] 类别 id
    z = torch.nn.functional.normalize(emb, dim=1)                 # fp32 归一化
    n = z.shape[0]
    sim = z @ z.t() / float(temperature)                          # [N, N]
    eye = torch.eye(n, dtype=torch.bool, device=feat.device)
    pos_mask = (cls.unsqueeze(0) == cls.unsqueeze(1)) & ~eye
    has_pos = pos_mask.any(dim=1)
    if not bool(has_pos.any()):
        return _ret(torch.zeros((), device=feat.device), _empty_stats(per_class))
    # 分母排除自身：masked_fill(-inf) + logsumexp（数值稳定）。sim_ip 恒是分母求和项之一，
    # 所以 log_prob ≤ 0、loss ≥ 0，极小 τ 下也成立（exp + clamp_min 地板的写法做不到）。
    log_prob = sim - torch.logsumexp(
        sim.masked_fill(eye, float('-inf')), dim=1, keepdim=True)
    mean_log_prob_pos = ((pos_mask.float() * log_prob).sum(dim=1)
                         / pos_mask.sum(dim=1).clamp_min(1).float())
    loss = -mean_log_prob_pos[has_pos].mean()
    # 对齐度读数：同类跨样本 vs 异类的平均余弦（cos = sim·τ）。间隔≈0 表示表征没有对齐。
    # return_stats=True 时每 batch 都算；否则只在下面打印快照的那几次调用里算。
    same_m = diff_m = float('nan')
    if return_stats or getattr(supcon_region_loss, '_calls', 0) + 1 in (
            1, 100, 1000, 10000, 100000):
        with torch.no_grad():
            cos = sim * float(temperature)
            same = pos_mask
            diff = (~eye) & ~pos_mask
            same_m = cos[same].mean().item() if bool(same.any()) else float('nan')
            diff_m = cos[diff].mean().item() if bool(diff.any()) else float('nan')
    supcon_region_loss._calls = getattr(supcon_region_loss, '_calls', 0) + 1
    if supcon_region_loss._calls in (1, 100, 1000, 10000, 100000):
        print(f'[SupCon] call #{supcon_region_loss._calls}: regions={n} '
              f'(anchors={int(has_pos.sum().item())}) | 同类跨样本 cos {same_m:.3f} '
              f'vs 异类 {diff_m:.3f}（间隔≈0 = 没对齐）| loss {loss.item():.3f} | '
              f'per-class regions {[int(v) for v in per_class]}', flush=True)
    stats = {'active': 1.0, 'regions': float(n),
             'anchors': float(int(has_pos.sum().item())),
             'same_cos': same_m, 'diff_cos': diff_m,
             'regions_per_class': per_class}
    return _ret(loss, stats)
