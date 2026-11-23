"""Datasets for FLOE scenes (HDF5/NetCDF, 80 m grid).

AI4ArcticChallengeDataset yields whole training batches of random patches, drawn from
chunk-aligned tiles weighted to a target pixel class mix (tile index built by
tools/data/scan_tiles.py). AI4ArcticChallengeTestDataset yields whole validation/test
scenes, optionally reading only the chunks that contain valid labels.
"""
# -- Built-in modules -- #
import os
import math
from collections import OrderedDict
from tqdm import tqdm

# -- Third-party modules -- #
import h5py
# hdf5plugin registers the Blosc/zstd filters on import; only scenes re-encoded with
# `tools.data.recompress --codec zstd` need it.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, get_worker_info
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_erosion as _scipy_binary_erosion

# -- Proprietary modules -- #
from floe.functions import rand_bbox
from floe.composition import BAG_OFFSET


# Byte-bounded LRU cache of decoded scenes for the legacy (non-tile) sampler,
# keyed by _scene_cache_key; values are CPU tensors.
_SCENE_CACHE = OrderedDict()
_SCENE_CACHE_BYTES = 0
# 跨所有 DataLoader worker 的总预算（见 _effective_scene_cache_max_bytes），
# 可在 config 的 scene_cache_max_bytes 里按容器内存覆盖。
_DEFAULT_SCENE_CACHE_MAX_BYTES = 40 * 1024 ** 3


# Tile 采样：.nc 底层 HDF5 的二维变量按 chunks=(512,512)+gzip 存储，读一个 chunk 对齐的
# 512×512 tile 时每个变量只解压一个 chunk（8 个变量约 27ms），而 xarray 整景解码约 9s/景、
# 输出约 3GB float32。不含任何有效标签像元的 tile 已被 scan_tiles.py 从索引中剔除，其余
# tile 按类别直方图加权抽取（_solve_tile_weights），因此不需要拒绝采样。

# 每个进程独立的 HDF5 句柄 LRU。句柄只占几 KB，缓存它是为了省掉重复 open 的约 10ms
# 以及 chunk cache 的重建。
_H5_HANDLES = OrderedDict()
_H5_HANDLES_PID = None
_H5_MAX_OPEN = 16
# HDF5 的 chunk cache 默认只有 1MB，装不下一个 512×512 float32 chunk，同一 tile 的多个
# 变量会反复解压同一个 chunk。64MB 足够放下一个 tile 的全部 8 个 chunk。
_H5_RDCC_NBYTES = 64 * 1024 ** 2


def _get_h5_handle(path):
    """按进程缓存打开的 HDF5 句柄（LRU，最多 _H5_MAX_OPEN 个）。

    fork 之后沿用继承来的 HDF5 句柄是未定义行为。文件只在 DataLoader worker 里打开，
    pid 校验只是安全网：pid 与建表时不同就丢弃整张表重新打开。只 clear() 不 close()：
    fork 后子进程的 fd 是独立副本，关闭它不影响父进程。
    """
    global _H5_HANDLES_PID
    pid = os.getpid()
    if _H5_HANDLES_PID != pid:
        _H5_HANDLES.clear()
        _H5_HANDLES_PID = pid

    handle = _H5_HANDLES.get(path)
    if handle is not None:
        _H5_HANDLES.move_to_end(path)
        return handle

    handle = h5py.File(path, 'r', rdcc_nbytes=_H5_RDCC_NBYTES, rdcc_nslots=5003)
    _H5_HANDLES[path] = handle
    while len(_H5_HANDLES) > _H5_MAX_OPEN:
        _, evicted = _H5_HANDLES.popitem(last=False)
        try:
            evicted.close()
        except Exception:
            pass
    return handle


def cpu_quota():
    """容器实际可用的 CPU 核数（cgroup 配额），而不是宿主机核数。

    容器里 `nproc` / `os.cpu_count()` 报的是宿主机核数（例如配额 25 核时报 208），
    真正的限制在 cgroup 的 cpu.max（v2）或 cpu.cfs_quota_us（v1）里。
    """
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:          # cgroup v2
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:   # cgroup v1
            quota = int(f.read())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except Exception:
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


_THREADS_LIMITED = False


def limit_torch_threads():
    """把 torch 的 intraop 线程数收到与 cgroup CPU 配额相称的水平。

    torch 按宿主机核数建 intraop 线程池；配额远小于此时每个并行算子都在做超订的线程
    同步，CPU 侧算子会慢几百倍（一个 batch 的组装从 0.55s 变成 205s）。DataLoader worker
    已是进程级并行，每个 worker 设 1 线程；主进程最多 8 线程，留给 collate、验证时在 CPU
    上累计预测等。
    """
    global _THREADS_LIMITED
    if _THREADS_LIMITED:
        return
    torch.set_num_threads(1 if get_worker_info() is not None else max(1, min(8, cpu_quota())))
    _THREADS_LIMITED = True


def _solve_tile_weights(hist, target, iters=400, eta=0.5):
    """求每个 tile 的采样概率，使训练实际看到的像素类别分布逼近 target。

    权重族取 w_i ∝ Σ_c a_c · frac[i, c]（frac 是 tile 内各类的归一化占比），对 a 做乘法
    更新（IPF 式）：每轮算出当前期望像素分布 p，把 a_c 乘以 (target_c / p_c)^eta。
    target 落在 {frac[i]} 的凸包内时收敛。

    期望像素分布按像素数加权，p = Σ_i w_i·hist[i] / Σ_i Σ_c w_i·hist[i, c]，而不是各 tile
    占比的平均 Σ_i w_i·frac[i]：tile 的有效像元数差异很大，后者会偏出约 ±0.02。
    hist 应传 scan_tiles.py 里按裁剪覆盖核加权的 hist_w：patch 在 tile 内随机裁，tile
    中心的像素比边缘更容易被采到，用原始直方图会再偏约 ±0.015。

    不直接按类别稀有度的倒数配权重：big/vast 的像素总量与 small/medium 相当，但只分布在
    约 1/3 的 tile 上（大浮冰成片连续），tile 级的均匀加权会系统性压低它们的像素份额。

    Parameters
    ----------
    hist : (N, C) float64  每个 tile 各类别的（核加权）像素数，不含 255
    target : (C,) float64  目标像素占比，和为 1；为 0 的类不参与

    Returns
    -------
    w : (N,) float64  归一化的 tile 采样概率
    achieved : (C,) float64  该权重下的期望像素分布
    """
    frac = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-12)
    active = target > 0
    a = np.zeros(hist.shape[1], dtype=np.float64)
    a[active] = 1.0

    def _pixel_mix(a_vec):
        s = frac @ a_vec
        total = s.sum()
        if not np.isfinite(total) or total <= 0:
            return None, None
        w_vec = s / total
        px = w_vec @ hist            # 期望各类像素数（按像素加权，不是占比平均）
        px_total = px.sum()
        if not np.isfinite(px_total) or px_total <= 0:
            return None, None
        return w_vec, px / px_total

    for _ in range(iters):
        w, p = _pixel_mix(a)
        if w is None:
            break
        ratio = np.ones_like(a)
        ok = active & (p > 0)
        ratio[ok] = target[ok] / p[ok]
        a = a * np.power(ratio, eta)
        a[~active] = 0.0
        amax = a.max()
        if amax > 0:
            a /= amax

    w, p = _pixel_mix(a)
    if w is None:
        raise RuntimeError('tile 采样权重求解失败：所有 tile 权重为 0，请检查 target_pixel_mix')
    return w, p


def _scene_cache_key(options, file):
    return (
        options['path_to_train_data'],
        file,
        options['down_sample_scale'],
        options['loader_downsampling'],
        options['patch_size'],
        tuple(options['full_variables']),
        tuple(options['charts']),
        options.get('cls2_filter_mask_dir', None),
    )


def _load_downsampled_scene(options, file, n_charts_init, cls2_mask_dir):
    """Load one scene, downsample/pad once, and return tensors ready for random crops."""
    # preprocessed_data_dir 下若有该景的 .npz（原生分辨率，labels=uint8, feats=float16），
    # 直接读它，跳过约 9s/景的 NetCDF 解码；否则解析原始 .nc。
    preproc_dir = options.get('preprocessed_data_dir', None)
    preproc_path = (os.path.join(preproc_dir, file.replace('.nc', '.npz'))
                     if preproc_dir is not None else None)

    if preproc_path is not None and os.path.exists(preproc_path):
        with np.load(preproc_path) as npz:
            labels_np = npz['labels']  # uint8, (n_charts_init, H, W)
            feats_np = npz['feats']    # float16, (n_features, H, W)
        temp_scene = torch.from_numpy(
            np.concatenate([labels_np.astype(np.float32), feats_np.astype(np.float32)], axis=0)
        ).unsqueeze(0)
    else:
        scene_path = os.path.join(options['path_to_train_data'], file)
        with xr.open_dataset(scene_path, engine='h5netcdf', mask_and_scale=False) as scene:
            temp_scene = scene[options['full_variables']].to_array()
            temp_scene = torch.from_numpy(np.expand_dims(temp_scene, 0))

    # scale=1 时跳过 interpolate：它会额外分配一份满尺寸临时张量（原生分辨率下单景可达
    # 150M+ 像素），这份瞬时峰值是多 worker 并发解码时 OOM 的主要来源之一。
    if options['down_sample_scale'] != 1:
        temp_scene = torch.nn.functional.interpolate(
            temp_scene,
            size=(temp_scene.size(2) // options['down_sample_scale'],
                  temp_scene.size(3) // options['down_sample_scale']),
            mode=options['loader_downsampling'])

    # Capture downsampled size before any padding (used for mask interpolation).
    h_down = temp_scene.size(2)
    w_down = temp_scene.size(3)

    if temp_scene.size(2) < options['patch_size']:
        height_pad = options['patch_size'] - temp_scene.size(2) + 1
    else:
        height_pad = 0

    if temp_scene.size(3) < options['patch_size']:
        width_pad = options['patch_size'] - temp_scene.size(3) + 1
    else:
        width_pad = 0

    if height_pad > 0 or width_pad > 0:
        temp_scene_y = torch.nn.functional.pad(
            temp_scene[:, :n_charts_init], (0, width_pad, 0, height_pad), mode='constant', value=255)
        temp_scene_x = torch.nn.functional.pad(
            temp_scene[:, n_charts_init:], (0, width_pad, 0, height_pad), mode='constant', value=0)
        temp_scene = torch.cat((temp_scene_y, temp_scene_x), dim=1)

    temp_scene = torch.squeeze(temp_scene)
    scene_labels = temp_scene[:n_charts_init].to(torch.uint8).contiguous()
    scene_feats = temp_scene[n_charts_init:].to(torch.float16).contiguous()

    cls2_mask = None
    if cls2_mask_dir is not None:
        mask_name = os.path.basename(file).replace('.nc', '_cls2_mask.npy')
        mask_path = os.path.join(cls2_mask_dir, mask_name)
        if os.path.exists(mask_path):
            try:
                mask_np = np.load(mask_path).astype(np.float32)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
                mask_ds = torch.nn.functional.interpolate(
                    mask_t, size=(h_down, w_down), mode='nearest'
                ).squeeze().numpy() > 0.5
                if height_pad > 0 or width_pad > 0:
                    mask_ds = np.pad(mask_ds,
                                     ((0, height_pad), (0, width_pad)),
                                     constant_values=False)
                cls2_mask = mask_ds
            except Exception as e:
                print(f'[WARN] Failed to load cls2 mask {mask_path}: {e}. Skipping mask for this scene.')

    return {
        'scene_labels': scene_labels,
        'scene_feats': scene_feats,
        'cls2_mask': cls2_mask,
    }


def _payload_nbytes(payload):
    """Approximate resident size of a cached scene payload, in bytes."""
    total = 0
    for k in ('scene_labels', 'scene_feats'):
        t = payload.get(k)
        if t is not None:
            total += t.element_size() * t.nelement()
    mask = payload.get('cls2_mask')
    if mask is not None:
        total += mask.nbytes
    return total


def _effective_scene_cache_max_bytes(options):
    """本 worker 的缓存预算：scene_cache_max_bytes 是跨所有 DataLoader worker 的总预算。

    每个 worker 是 fork 出的独立进程、各有一份 _SCENE_CACHE，若把配置值当作每个 worker
    的预算，实际峰值会是 num_workers 倍。
    """
    max_bytes = options.get('scene_cache_max_bytes', _DEFAULT_SCENE_CACHE_MAX_BYTES)
    worker_info = get_worker_info()
    if worker_info is not None and worker_info.num_workers > 0:
        max_bytes = max_bytes // worker_info.num_workers
    return max_bytes


def _get_or_create_cached_scene(options, file, n_charts_init, cls2_mask_dir, enable_cache=True):
    global _SCENE_CACHE_BYTES
    key = _scene_cache_key(options, file)
    if enable_cache and key in _SCENE_CACHE:
        _SCENE_CACHE.move_to_end(key)  # mark as most-recently-used
        return _SCENE_CACHE[key], True

    payload = _load_downsampled_scene(options, file, n_charts_init, cls2_mask_dir)
    if enable_cache:
        payload_bytes = _payload_nbytes(payload)
        max_bytes = _effective_scene_cache_max_bytes(options)
        # 按字节预算淘汰最久未用的场景：原生分辨率下单景可达 1~2GB，不设上限会超出容器内存。
        while _SCENE_CACHE and (_SCENE_CACHE_BYTES + payload_bytes) > max_bytes:
            _, evicted = _SCENE_CACHE.popitem(last=False)
            _SCENE_CACHE_BYTES -= _payload_nbytes(evicted)
        _SCENE_CACHE[key] = payload
        _SCENE_CACHE_BYTES += payload_bytes
    return payload, False


def preload_scene_cache(options, files):
    """Preload a file list into the global scene cache (called once before the CV loop)."""
    unique_files = sorted(set(files))
    if len(unique_files) == 0:
        return

    n_charts_init = len(options['charts'])
    cls2_mask_dir = options.get('cls2_filter_mask_dir', None)
    cache_enabled = bool(options.get('enable_scene_cache', True))

    cache_hits = 0
    cache_misses = 0
    for file in tqdm(unique_files, desc='Preloading scenes', colour='blue'):
        _, was_hit = _get_or_create_cached_scene(
            options, file, n_charts_init, cls2_mask_dir, enable_cache=cache_enabled)
        if was_hit:
            cache_hits += 1
        else:
            cache_misses += 1

    print(f'[SceneCache] Preload complete: {len(unique_files)} files | '
          f'hits={cache_hits}, misses={cache_misses}, cache_size={len(_SCENE_CACHE)}')


FLOE_CLASS_NAMES = ['water', 'cake', 'small', 'medium', 'big', 'vast']


def remap_floe_labels(arr, merge_map=None, ignore_val=255):
    """FLOE 标签重映射（就地），训练与验证的所有标签读取都走这里。

    1) 冰山(6) → 255（本数据集中该类为空，保留作安全网，见 DATASET_HANDBOOK §4）。
    2) 按 merge_map 合并类别，例如 {3: 2, 4: 3, 5: 4} 把中浮冰并入小浮冰、大/巨各降一位
       (0=水 1=饼状 2=小+中 3=大 4=巨，n_classes=5)。

    所有掩膜先从原数组算出再统一写回：逐项就地映射会串链（5→4 之后又被 4→3 改成 3）。
    """
    arr[arr == 6] = ignore_val
    if not merge_map:
        return arr
    masks = [(arr == int(src), int(dst)) for src, dst in merge_map.items()]
    for m, dst in masks:
        arr[m] = dst
    return arr


def val_ondemand_ok(options, n_models=1):
    """验证路径能否安全启用按需读。返回 (bool, 原因字符串)。

    按需读只解码含有效标签的 tile 所覆盖的 chunk，其余留 0，因此推理必须只在与 patch 网格
    对齐、不重叠的 tile 上做前向（functions.fast_tiled_val_predict / fast_tiled_val_inference
    + tile 门控）。读到别处像素的用法都会静默读到 0：重叠滑窗（batched_slide_inference，
    窗口由 swin_hp.val_stride 决定，base 默认 [56, 56]）、翻转或偏移 TTA、整景可视化与预测
    导出、不同 patch_size 的模型集成（need 网格只按第一个模型算）。所以按需读由调用方显式
    开启（AI4ArcticChallengeTestDataset(..., ondemand_read=)），config 的 val_ondemand_read
    只是全局开关。
    """
    if not options.get('val_ondemand_read', True):
        return False, 'config 的 val_ondemand_read=False（全局关闭）'
    if n_models != 1 and options.get('val_tile_gate', True):
        return False, f'集成推理（{n_models} 个模型），need 网格只按第一个模型的 patch_size 算'
    # without the tile gate every chunk is read (prep_scene_ondemand), so an ensemble is safe
    if options.get('down_sample_scale', 1) != 1 or options.get('val_downsample_scale', 1) != 1:
        return False, '降采样口径下整景 interpolate 需要全部像素'
    if 'global_valid_mask' not in list(options.get('train_variables', [])):
        return False, 'train_variables 里没有 global_valid_mask，tfv_mask 会退化成"未读即黑边"'
    if options.get('model_selection') in ('swin', 'PA2'):
        stride = tuple(options.get('swin_hp', {}).get('val_stride', ()))
        ps = options['patch_size']
        if stride != (ps, ps):
            return False, (f'滑窗推理 val_stride={stride} != ({ps}, {ps})，'
                           f'重叠窗口会读到未解码的区域')
    # texture_channels 与按需读兼容：纹理在 functions 里对每个 tile 单独计算，窗口在 tile
    # 内截断（与训练 patch 同口径），不需要 tile 之外的像素。
    return True, 'ok'


# 纹理通道（texture_channels）：从 HH（通道 0）在线算的密集纹理特征。3 个多尺度局部标准差
# 是变异函数的密集代理（小碎冰场的方差在很细的尺度就饱和，中浮冰场随尺度慢慢累积）；
# 1 个局部边界密度反映冰水边界的破碎程度。改通道数时要同步改模型 in_channels 的推导
# （models/sota/segformer.py 等）和本文件两处 _n_channels 预分配。
N_TEXTURE_CHANNELS = 4
_TEXTURE_STD_WINDOWS = (3, 7, 15)   # 窗口(px)，80m 像元下 = 240 / 560 / 1200 m
_TEXTURE_EDGE_WINDOW = 7            # 边界密度窗口(px) ≈ 560 m


def _texture_channels(x):
    """从 x [1, C, H, W] 的 HH（通道 0）算 N_TEXTURE_CHANNELS 个纹理通道，返回 [1, 4, H, W]。

    - 3 个多尺度局部标准差 σ(w) = sqrt(E[HH²]_w - E[HH]²_w)，盒式滤波；
    - 1 个局部边界密度：L1 梯度幅值的局部均值。
    count_include_pad=False 使边界像素按窗口内实际像元数平均（无零偏）。输入是 z-score
    后的 HH；新通道量级约 0.1~1，不再单独归一化，交由首层 conv/BN 吸收。
    """
    import torch.nn.functional as F
    hh = x[:, 0:1, :, :].float()
    outs = []
    for w in _TEXTURE_STD_WINDOWS:
        pad = w // 2
        mean = F.avg_pool2d(hh, w, stride=1, padding=pad, count_include_pad=False)
        meansq = F.avg_pool2d(hh * hh, w, stride=1, padding=pad, count_include_pad=False)
        outs.append((meansq - mean * mean).clamp_min(0).sqrt())
    gx = F.pad(hh[:, :, :, 1:] - hh[:, :, :, :-1], (0, 1, 0, 0))   # 水平差分，右侧补 0
    gy = F.pad(hh[:, :, 1:, :] - hh[:, :, :-1, :], (0, 0, 0, 1))   # 垂直差分，底部补 0
    gmag = gx.abs() + gy.abs()                                     # L1 梯度幅值
    ew = _TEXTURE_EDGE_WINDOW
    outs.append(F.avg_pool2d(gmag, ew, stride=1, padding=ew // 2, count_include_pad=False))
    return torch.cat(outs, dim=1)


def _need_chunk_grid(y, fill_values, charts, H, W, patch_size, chunk):
    """验证按需读：哪些 HDF5 chunk 必须读。返回 (cg_h, cg_w) 的 bool 网格。

    先在 patch_size 的 tile 网格（与 functions.compute_valid_tile_mask 相同，即真正喂进
    网络的 tile）上找出至少含一个有效标签像元的 tile，再把每个有效 tile 覆盖到的所有
    chunk 标为必读。不能简化成「chunk 内有有效像元就读」：一个有效 tile 可能跨多个 chunk，
    其中某个 chunk 自己没有有效标签，却承载着这个 tile 的输入像素。chunk 恰好是 patch 的
    整数倍时两者等价，但这里的逻辑不依赖这一点。
    """
    valid = None
    for chart in charts:
        v = y[chart] != fill_values[chart]
        valid = v if valid is None else (valid | v)

    n_h, n_w = -(-H // patch_size), -(-W // patch_size)
    pad_h, pad_w = n_h * patch_size - H, n_w * patch_size - W
    if pad_h or pad_w:
        valid = np.pad(valid, ((0, pad_h), (0, pad_w)))
    tile_ok = valid.reshape(n_h, patch_size, n_w, patch_size).any(axis=(1, 3))

    cg_h, cg_w = -(-H // chunk), -(-W // chunk)
    need = np.zeros((cg_h, cg_w), dtype=bool)
    for i, j in np.argwhere(tile_ok):
        y0, x0 = int(i) * patch_size, int(j) * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        need[y0 // chunk:(y1 - 1) // chunk + 1, x0 // chunk:(x1 - 1) // chunk + 1] = True
    return need


def _merge_hist_columns(hist, merge_map, n_new):
    """把 tile 索引的类别直方图按 merge_map 合并列。

    索引里的 hist/hist_w 总是按原始类别 0..5 存储，而 target_pixel_mix 的下标是合并后的
    类别；不对齐时 _solve_tile_weights 会静默解出错误的采样分布。
    """
    out = np.zeros((hist.shape[0], n_new), dtype=hist.dtype)
    for c_old in range(hist.shape[1]):
        c_new = int(merge_map.get(c_old, c_old))
        if c_new < n_new:
            out[:, c_new] += hist[:, c_old]
    return out


def merged_class_names(merge_map, n_cls):
    """合并后的类别名，仅用于日志打印。"""
    if not merge_map:
        return FLOE_CLASS_NAMES[:n_cls]
    groups = [[] for _ in range(n_cls)]
    for c_old, nm in enumerate(FLOE_CLASS_NAMES):
        c_new = int(merge_map.get(c_old, c_old))
        if c_new < n_cls:
            groups[c_new].append(nm)
    return ['+'.join(g) if g else '?' for g in groups]


def _erode_sod_boundaries(sod_patch, erosion_iters, ignore_val=255):
    """将SOD标签各类别的边界像素设为 ignore_val，只保留类别核心区域参与训练。

    CIS冰蛋图polygon与Sentinel-1 SAR存在时空错位，交界处像素标签最不可信。
    对每个类别的区域向内腐蚀 erosion_iters 像素，被腐蚀掉的边界部分设为 ignore_val。

    Parameters
    ----------
    sod_patch : ndarray, shape (H, W)
        SOD标签数组，ignore_val 表示无效像素。
    erosion_iters : int
        向内腐蚀的像素数，对应CIS对齐误差估计（建议3-7）。
    ignore_val : int
        无效像素的标记值，默认255。

    Returns
    -------
    result : ndarray, shape (H, W)
        边界像素已替换为 ignore_val 的标签数组。
    """
    result = sod_patch.copy()
    valid_mask = sod_patch != ignore_val
    for cls_id in np.unique(sod_patch[valid_mask]):
        cls_mask = (sod_patch == cls_id)
        eroded = _scipy_binary_erosion(cls_mask, iterations=erosion_iters)
        boundary = cls_mask & ~eroded
        result[boundary] = ignore_val
    return result


def _month_sin_cos(filename):
    """Extract month from scene filename and return (sin, cos) cyclic encoding.

    Filename format: S1A_EW_GRDM_1SDH_YYYYMMDDTHHMMSS_...nc
    Month digits are at filename[21:23].
    """
    month = int(filename[21:23])
    angle = 2 * math.pi * month / 12
    return math.sin(angle), math.cos(angle)


def degrade_resolution_(x, variables, k):
    """Emulate a coarser sensor grid in place (option input_degrade_factor).

    Every continuous input channel is replaced by its masked k x k block mean,
    broadcast back to the native grid. Blocks are aligned to the array origin, which
    is the global grid for whole scenes and, when k divides 512, for 512-px chunks.
    Invalid pixels (global_valid_mask == 0) are excluded from the means and left
    unchanged; the mask itself, the labels and the evaluation grid stay native.
    Derived channels (month, pol ratio, texture) must be computed after this.
    k <= 1 is a no-op.

    x: torch tensor (1, C, H, W) or (C, H, W), or a numpy view of the same, with
       channels in the order of `variables`. Returns x.
    """
    k = int(k or 1)
    if k <= 1:
        return x
    t = x if torch.is_tensor(x) else torch.from_numpy(x)
    t4 = t.unsqueeze(0) if t.dim() == 3 else t
    variables = list(variables)
    mi = variables.index('global_valid_mask')
    cont = [i for i, v in enumerate(variables) if v != 'global_valid_mask']
    H, W = t4.shape[-2], t4.shape[-1]
    m = (t4[:, mi:mi + 1] > 0.5).to(torch.float32)
    xc = t4[:, cont].to(torch.float32)
    num = torch.nn.functional.avg_pool2d(xc * m, k, stride=k, ceil_mode=True)
    den = torch.nn.functional.avg_pool2d(m, k, stride=k, ceil_mode=True)
    blk = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
    up = blk.repeat_interleave(k, dim=-2).repeat_interleave(k, dim=-1)[..., :H, :W]
    t4[:, cont] = torch.where(m > 0, up, xc).to(t4.dtype)
    return x


class AI4ArcticChallengeDataset(Dataset):
    """Training dataset. Each item is a whole batch of patches (use DataLoader(batch_size=None)).

    Patches come from the tile sampler when `tile_index_path` is set, otherwise from the
    legacy sampler (random scene, random crop, rejection filters).
    """

    def __init__(self, options, files, do_transform=False):
        self.options = options
        self.files = files
        self.do_transform = do_transform

        self._cls2_mask_dir = self.options.get('cls2_filter_mask_dir', None)

        # Legacy sampler: scenes are loaded lazily into the byte-bounded LRU cache
        # (_get_or_create_cached_scene), never all at once; a native-resolution scene can
        # take 1-2 GB.
        self._n_charts_init = len(self.options['charts'])
        self._cache_enabled = bool(self.options.get('enable_scene_cache', True))

        # Index of global_valid_mask in the full patch (charts first) and among the
        # feature channels only.
        if 'global_valid_mask' in self.options['train_variables']:
            self.global_valid_mask_idx = (len(self.options['charts']) +
                                          list(self.options['train_variables']).index('global_valid_mask'))
            self.global_valid_mask_feat_idx = list(self.options['train_variables']).index('global_valid_mask')
        else:
            self.global_valid_mask_idx = None
            self.global_valid_mask_feat_idx = None

        self._target_chart = self.options.get('target_chart', 'FLOE')
        self._target_chart_idx = self.options['charts'].index(self._target_chart)

        # 采样诊断计数（每个 worker 独立，不跨进程汇总），每 _diag_log_every 个 batch 打印并清零。
        self._diag_counts = {
            'attempts': 0, 'accepted': 0, 'cache_miss': 0,
            'reject_invalid_patch': 0,  # 黑边覆盖不足 或 patch 几乎全是 mask/255
            'reject_label_cov': 0, 'reject_water': 0,
            'reject_cls4': 0, 'reject_rare': 0,
        }
        self._diag_batches_done = 0
        # <=0 表示不打印。
        self._diag_log_every = max(0, self.options.get('diag_log_every_batches', 5))
        self._last_scene_was_cache_hit = None

        # 辐射域增强（radaug）：以概率 Radiometric_prob 给 HH/HV 的有效像元加常数偏移 b
        # （可选再做对比度缩放 a），让模型少依赖绝对辐射水平：单景 HH/HV 整体平移约
        # ±0.7 个标准差就能让 medium IoU 在 0.01 与 0.99 之间变化。扰动本体在 _augment_x。
        # data_augmentations 整键可能缺省（SOD 系 config），所以用 .get 链取。
        _da = self.options.get('data_augmentations') or {}
        self._radaug_p = float(_da.get('Radiometric_prob', 0) or 0)
        self._radaug_B = float(_da.get('Radiometric_b', 0.30))
        # 对比度项 a ~ U(1-A, 1+A)，默认 A=0 关闭；关闭时不抽 a，只配 b 的 config 随机流不变。
        self._radaug_A = float(_da.get('Radiometric_a', 0) or 0)
        # 逐通道抖动固定为 B/3：HH 与 HV 的景间变化大部分是共模的（corr=0.595，HH-HV 的
        # 景间 std 0.384，单通道 0.427），所以共模分量给满 B、通道差分只给少量抖动，
        # 不淹没 pol_ratio (HH-HV) 的真实信号。
        self._radaug_jit = self._radaug_B / 3.0
        self._radaug_gen = None        # 独立 Generator，惰性构造（见 _augment_x）
        self._radaug_mask_idx = None
        if self._radaug_p > 0:
            _tv = list(self.options['train_variables'])
            if 'global_valid_mask' not in _tv:
                raise ValueError(
                    'Radiometric_prob>0 需要 train_variables 里含 global_valid_mask：'
                    '辐射扰动只作用于有效像元，黑边与几何 fill 区必须被排除在外。'
                    f'当前 train_variables={_tv}')
            # 扰动作用于硬编码的通道 0/1，顺序不符时报错而不是静默扰动错通道。
            assert _tv[0] == 'nersc_sar_primary' and _tv[1] == 'nersc_sar_secondary', \
                f'radaug 假定通道 0/1 为 nersc SAR primary/secondary，实际 {_tv[:2]}'
            # 负的 A 会被静默忽略；A >= 1 时 a 可能 <= 0（对比度翻转或压平），都没有意义。
            assert 0.0 <= self._radaug_A < 1.0, \
                f'Radiometric_a 必须落在 [0, 1)，得到 {self._radaug_A}'
            # x_patch 只含 train_variables 通道（chart 通道已切掉），所以用不带 chart 偏移的下标。
            self._radaug_mask_idx = _tv.index('global_valid_mask')
            # dataset 在主进程构造，这行只打印一次，便于从日志确认 radaug 确实生效。
            print(f'[Radaug] p={self._radaug_p} B={self._radaug_B} A={self._radaug_A} '
                  f'jit={self._radaug_jit:.3f} mask_ch={self._radaug_mask_idx}', flush=True)

        # 配置了 tile_index_path 才启用 tile 采样，否则走 legacy 路径（未建索引的数据集，如 SOD）。
        self._tile_sampling = False
        if self.options.get('tile_index_path', None):
            self._init_tile_sampling()

        # segformer_dann：每个 patch 额外携带来源场景的域序号（self.files 即 train_list 的
        # 下标），经 prep_dataset 放进 y['_domain']。关闭时不加键、不消耗随机数。
        self._dann_on = bool(self.options.get('segformer_dann', False))
        if self._dann_on and not self._tile_sampling:
            raise ValueError(
                'segformer_dann=True 目前只支持 tile 采样路径（_getitem_tile）：legacy '
                '路径会在场景级重试/放弃后换景，"这个 patch 来自哪一景"的记账没有实现。'
                '请给 config 配 tile_index_path（FLOE 一律走 tile 采样），或关掉 '
                'segformer_dann。')

        # Chart-composition supervision: polygon ids travel with the labels as one extra
        # channel (global id + BAG_OFFSET), so flips, rotations and cutmix move them
        # together with the labels. See floe/composition.py.
        self._comp_on = bool((self.options.get('composition_supervision') or {}).get('enabled', False))
        self._n_extra = 1 if self._comp_on else 0
        if self._comp_on:
            if not self._tile_sampling:
                raise ValueError('composition_supervision needs tile sampling (tile_index_path).')
            self._init_composition()

    def _init_composition(self):
        cfg = self.options['composition_supervision']
        chart_dir = cfg['chart_dir']
        Q, U, base = [], [], []
        n = 0
        for f in self.files:
            path = os.path.join(chart_dir, os.path.splitext(f)[0] + '.chart.h5')
            with h5py.File(path, 'r') as h:
                q = h['composition'][...].astype(np.float32)
                u = h['unknown'][...].astype(np.float32)
            Q.append(q); U.append(u); base.append(n)
            n += len(q)
        if n + BAG_OFFSET >= 2 ** 24:
            raise ValueError(f'{n} polygons do not fit in a float32 label channel')
        self.bag_Q = np.concatenate(Q) if Q else np.zeros((0, 6), np.float32)
        self.bag_U = np.concatenate(U) if U else np.zeros(0, np.float32)
        self._bag_base = np.asarray(base, dtype=np.int64)
        self._chart_dir = chart_dir
        print(f'[Composition] {n} polygons over {len(self.files)} scenes from {chart_dir}', flush=True)

    def _read_bag_tile(self, scene_id, r, c, T):
        path = os.path.join(self._chart_dir, os.path.splitext(self.files[scene_id])[0] + '.chart.h5')
        pid = _get_h5_handle(path)['polygon_id'][r:r + T, c:c + T].astype(np.int64)
        return np.where(pid >= 0, pid + self._bag_base[scene_id] + BAG_OFFSET, 0).astype(np.float32)

    def _init_tile_sampling(self):
        opts = self.options
        ps = opts['patch_size']

        if opts['down_sample_scale'] != 1:
            raise ValueError(
                'tile 采样直接读原生分辨率的 chunk 对齐 tile，不支持 down_sample_scale != 1；'
                f'当前为 {opts["down_sample_scale"]}。要降采样请走旧路径（不要设 tile_index_path）。')

        index = np.load(opts['tile_index_path'], allow_pickle=False)
        tile_size = int(index['tile_size'])
        if tile_size < ps:
            raise ValueError(f'tile_size={tile_size} 小于 patch_size={ps}，无法在 tile 内裁 patch。')
        self._tile_size = tile_size

        # hist_w 是按 patch_size 的裁剪覆盖核加权的直方图（见 scan_tiles.crop_coverage_kernel）；
        # patch_size 变了核就变了，必须重扫，否则求解出的权重对不上实际采样分布。
        if 'hist_w' not in index.files:
            raise ValueError(
                f'{opts["tile_index_path"]} 是旧版索引（缺 hist_w）。请重跑 scan_tiles.py。')
        index_ps = int(index['patch_size'])
        if index_ps != ps:
            raise ValueError(
                f'tile 索引是按 patch_size={index_ps} 建的，但 config 的 patch_size={ps}。'
                f'裁剪覆盖核依赖 patch_size，请用 --patch-size {ps} 重跑 scan_tiles.py。')

        # 索引里的 scene_id 是相对于扫描时的全量文件表的；这里重映射到本 dataset 的
        # self.files 下标（train/val 用不同的 datalist，各自只取自己那部分 tile）。
        index_files = [str(f) for f in index['files']]
        file_to_index_id = {name: i for i, name in enumerate(index_files)}
        missing = [f for f in self.files if f not in file_to_index_id]
        if missing:
            raise ValueError(
                f'{len(missing)} 个场景不在 tile 索引里（如 {missing[:3]}）。'
                f'datalist 变了就要重跑 scan_tiles.py 生成新索引。')

        # index_id -> 本 dataset 的 scene 下标
        remap = np.full(len(index_files), -1, dtype=np.int32)
        for i, name in enumerate(self.files):
            remap[file_to_index_id[name]] = i

        scene_id = index['scene_id'].astype(np.int64)
        keep = remap[scene_id] >= 0

        cfg = opts.get('tile_sampling', {}) or {}

        # tile 级的黑边粗筛。patch 级还会再筛一次（在 tile 内存里做，零成本）：
        # 一个 valid_ratio=0.4 的 tile 里完全可能存在全 valid 的 256 patch。
        vmt = opts.get('valid_mask_threshold', 0.5)
        keep &= index['valid_ratio'] >= vmt

        # 标签有效率下限。valid_ratio 只排除 SAR 黑边（global_valid_mask==1 的比例），
        # label_valid_ratio 还排除陆地和模糊多边形（FLOE != 255 的比例）；陆地 tile 的
        # global_valid_mask 全是 1，标签却几乎全是 255。不筛掉的话：(1) 裁出的 patch 几乎都
        # 过不了标签覆盖检查，白耗重试；(2) frac 按有效标签像元归一化，只有几十个有效像元
        # 且恰好全是巨浮冰的陆地边缘 tile 会得到 frac[vast]=1，被求解器赋予极高的采样概率。
        mlvr = float(cfg.get('min_label_valid_ratio', 0.5))
        keep &= index['label_valid_ratio'] >= mlvr

        if not keep.any():
            raise ValueError(f'tile 索引在 valid_ratio>={vmt} 且 label_valid_ratio>={mlvr} '
                             f'且属于本 datalist 的条件下为空。')

        hist = index['hist'][keep].astype(np.float64)      # 原始像素数，仅用于统计/报告
        hist_w = index['hist_w'][keep].astype(np.float64)  # 裁剪覆盖核加权，用于求解权重
        self._tile_scene = remap[scene_id[keep]].astype(np.int32)
        self._tile_row = index['row'][keep].astype(np.int32)
        self._tile_col = index['col'][keep].astype(np.int32)

        # n_classes 是按 chart 索引的字典（见 configs/_base_/base.py），不是整数
        n_cls = opts['n_classes'][self._target_chart]

        # 索引直方图按原始 6 类存储；合并类别时先合并列，与 target_pixel_mix 的下标对齐。
        _merge = opts.get('floe_class_merge') if self._target_chart == 'FLOE' else None
        if _merge:
            hist = _merge_hist_columns(hist, _merge, n_cls)
            hist_w = _merge_hist_columns(hist_w, _merge, n_cls)

        target = np.zeros(n_cls, dtype=np.float64)
        for cls, share in (cfg.get('target_pixel_mix') or {}).items():
            target[int(cls)] = float(share)
        if target.sum() <= 0:
            raise ValueError('tile_sampling.target_pixel_mix 未配置或全为 0。')
        # 本 datalist 中没有像素的类采不到；给它非零目标会让乘法更新把 a_c 推向无穷，
        # 所以剔除并告警。
        absent = (hist.sum(axis=0) == 0) & (target > 0)
        if absent.any():
            print(f'[TileSampling] 类别 {np.where(absent)[0].tolist()} 在本 datalist 中没有任何像素，'
                  f'已从 target_pixel_mix 中剔除。')
            target[absent] = 0.0
        target /= target.sum()

        weights, achieved = _solve_tile_weights(hist_w, target)
        # searchsorted 采样：cumsum 末位钉成 1.0，否则浮点误差下 rand() 的极端取值会越界。
        self._tile_cum = np.cumsum(weights)
        self._tile_cum[-1] = 1.0

        self._patches_per_tile = max(1, cfg.get('patches_per_tile', 2))
        self._tile_sampling = True

        # 类 0 是水，其余都是冰（合并后的类别空间同样如此）。
        n_ice = int((hist[:, 1:].sum(axis=1) > 0).sum())
        names = merged_class_names(_merge, n_cls)
        print(f'[TileSampling] {len(self._tile_row)} tiles over {len(self.files)} scenes '
              f'(tile={tile_size}, patch={ps}, patches_per_tile={self._patches_per_tile}, '
              f'valid_ratio>={vmt}, label_valid_ratio>={mlvr}); {n_ice} tiles contain ice.')
        print('[TileSampling] expected batch pixel mix  : '
              + '  '.join(f'{n}={v:.3f}' for n, v in zip(names, achieved)))
        print('[TileSampling] requested target pixel mix: '
              + '  '.join(f'{n}={v:.3f}' for n, v in zip(names, target)))
        max_dev = float(np.abs(achieved - target).max())
        if max_dev > 0.02:
            print(f'[TileSampling][WARN] 期望分布与目标最大偏差 {max_dev:.3f}——目标可能落在 '
                  f'tile 类别分布的凸包之外（某类没有足够"纯"的 tile 把它的占比拉到目标值）。')

        # 采样集中度诊断：ESS = 1/Σw² 是有效 tile 数（等权时等于 tile 总数，权重全压在 k 个
        # tile 上时约为 k）。ESS 远小于 tile 总数说明为命中 target_pixel_mix 在反复抽同一小撮
        # tile。逐类再报承载 tile 数与这些 tile 的采样权重和：稀有类（如只在少数景出现的
        # ice cake）若由很少的 tile 承载了很大的权重，应下调它的目标份额。
        ess = 1.0 / float((weights ** 2).sum())
        print(f'[TileSampling] effective tiles (ESS=1/Σw²): {ess:,.0f} / {len(weights)}')
        for c, nm in enumerate(names):
            has_c = hist[:, c] > 0
            if has_c.any():
                print(f'[TileSampling]   {nm:<8s} tiles={int(has_c.sum()):6d}  '
                      f'采样权重和={float(weights[has_c].sum()):.3f}')

    def __len__(self):
        return self.options['epoch_len']

    def _augment_x(self, x_patch, scene_id):
        """依次做分辨率退化、辐射扰动（radaug，仅训练），再追加 month / pol_ratio / texture 派生通道。

        顺序有意义：派生通道必须从已退化、已扰动的 HH/HV 计算，否则 pol_ratio 会成为未扰动
        的干净捷径，train 与 val 的通道间关系也会不一致。tile 路径与 legacy 路径都在
        transform() 之前调用本函数。
        """
        x_patch = degrade_resolution_(x_patch, self.options['train_variables'],
                                      self.options.get('input_degrade_factor', 1))
        # radaug 关闭时不建 Generator、不抽任何随机数。
        if self.do_transform and self._radaug_p > 0:
            if self._radaug_gen is None:
                # 独立 Generator，与主随机流隔离：radaug run 与对照 run 看到逐位相同的 patch
                # 序列和几何增强。不借用 np.random，因为 numpy 流被 tile 选择和裁剪位置消费，
                # 借用会让两者的 patch 序列分叉。种子从 torch.initial_seed() 派生（worker 内为
                # base_seed + worker_id，base_seed 每 epoch 由主进程重新抽取），所以扰动序列
                # 逐 epoch 变化；常量种子会因 worker 每 epoch 重建而重放同一批 b。
                info = get_worker_info()
                wid = info.id if info is not None else 0
                self._radaug_gen = torch.Generator()
                self._radaug_gen.manual_seed(
                    (torch.initial_seed() * 6364136223846793005 + 7919 * wid + 17)
                    % (2 ** 63))
            g = self._radaug_gen
            if torch.rand(1, generator=g).item() < self._radaug_p:
                B = self._radaug_B
                b_common = (torch.rand(1, generator=g).item() * 2 - 1) * B
                # 抽取顺序固定为：触发判定、b_common、a（仅 A > 0 时）、逐通道抖动；改动顺序会
                # 改变已有 run 的扰动序列。a 由两个通道共享：对比度是整景性质，不随极化变化。
                a = 1.0
                if self._radaug_A > 0:
                    a = 1.0 + (torch.rand(1, generator=g).item() * 2 - 1) * self._radaug_A
                # 只扰动有效像元；affine 的 fill 区在之后的 transform 里才产生，同样不受影响。
                mask = x_patch[:, self._radaug_mask_idx] > 0.5      # (1, H, W)
                for ch in (0, 1):                                   # 0=HH, 1=HV
                    b = b_common + (torch.rand(1, generator=g).item() * 2 - 1) * self._radaug_jit
                    # 原地写不会回写 tile 或场景缓冲：tile 路径的 x_patch 是 _crop_from_tile 里
                    # np.ascontiguousarray 的副本（前提是 patch_size < tile_size，等尺寸时不复制），
                    # legacy 路径的 x_patch 是 .type(torch.float) 新建的张量。
                    xc = x_patch[:, ch]
                    if a != 1.0:
                        # 围绕本 patch、本通道有效像元均值的纯对比度：xc ← m + a·(xc - m) + b。
                        # a 不改变 patch 均值，因此与加性的 b 解耦。m_ch 逐 patch 现算，不是全局
                        # 统计量。mask 全 False 时 m_ch=nan，但赋值目标也为空，不会写入 nan。
                        m_ch = float(xc[mask].mean())
                        xc[mask] = m_ch + a * (xc[mask] - m_ch) + b
                    else:
                        xc[mask] = xc[mask] + b
        if self.options.get('month_encoding', False):
            s, c = _month_sin_cos(self.files[scene_id])
            H, W = x_patch.shape[-2], x_patch.shape[-1]
            month_ch = torch.tensor([s, c], dtype=torch.float).view(1, 2, 1, 1).expand(1, 2, H, W)
            x_patch = torch.cat([x_patch, month_ch], dim=1)
        if self.options.get('pol_ratio_channel', False):
            pol_ratio = x_patch[:, 0:1, :, :] - x_patch[:, 1:2, :, :]
            x_patch = torch.cat([x_patch, pol_ratio], dim=1)
        if self.options.get('texture_channels', False):
            x_patch = torch.cat([x_patch, _texture_channels(x_patch)], dim=1)
        return x_patch

    def random_crop_downsample(self, idx):
        """
        Perform random cropping on a pre-downsampled scene, fetched lazily via
        the bounded LRU cache (_get_or_create_cached_scene).

        Parameters
        ----------
        idx :
            Index from self.files to parse.

        Returns
        -------
        x_patch, y_patch :
            Torch tensors, or (None, None) if patch is invalid.
        """

        patch = np.zeros((len(self.options['full_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        payload, _was_hit = _get_or_create_cached_scene(
            self.options, self.files[idx], self._n_charts_init, self._cls2_mask_dir,
            enable_cache=self._cache_enabled)
        self._last_scene_was_cache_hit = _was_hit  # 供 _getitem_legacy 的诊断计数
        _label = payload['scene_labels']
        _feats = payload['scene_feats']
        _cls2_mask = payload['cls2_mask']
        _n_charts = len(self.options['charts'])
        ps = self.options['patch_size']

        # Get random index to crop from.
        row_rand = np.random.randint(low=0, high=_feats.shape[1] - ps)
        col_rand = np.random.randint(low=0, high=_feats.shape[2] - ps)

        # Discard patches with too many black border pixels using global_valid_mask.
        if self.global_valid_mask_feat_idx is not None:
            valid_mask_patch = _feats[
                self.global_valid_mask_feat_idx,
                row_rand: row_rand + ps,
                col_rand: col_rand + ps].numpy()
            valid_ratio = float(np.mean(valid_mask_patch))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        # Discard patches with too many label fill (masked) pixels.
        if np.sum(_label[self._target_chart_idx,
                         row_rand: row_rand + ps,
                         col_rand: col_rand + ps].numpy()
                  != self.options['class_fill_values'][self._target_chart]) > 1:

            # NumPy auto-upcasts uint8/float16 slices into the float64 `patch` buffer.
            patch[:_n_charts, :, :] = _label[:, row_rand:row_rand + ps,
                                             col_rand:col_rand + ps].numpy()
            patch[_n_charts:, :, :] = _feats[:, row_rand:row_rand + ps,
                                             col_rand:col_rand + ps].numpy()

            # SOD-specific: remap labels in-place: merge old class 2 (young ice) into
            # class 1 (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
            if self._target_chart == 'SOD':
                sod_ch = patch[self._target_chart_idx]
                valid = sod_ch != 255
                sod_ch[valid & (sod_ch >= 2)] -= 1

            # FLOE: 冰山(6)→255，并按 floe_class_merge 合并类别（见 remap_floe_labels）
            elif self._target_chart == 'FLOE':
                remap_floe_labels(patch[self._target_chart_idx],
                                  self.options.get('floe_class_merge'))

            # Boundary erosion: mask out class-boundary pixels that are most likely
            # mislabeled due to spatial misalignment between CIS ice charts and SAR.
            _erode_iters = self.options.get('boundary_erosion_iters', 0)
            if _erode_iters > 0:
                patch[self._target_chart_idx] = _erode_sod_boundaries(
                    patch[self._target_chart_idx], erosion_iters=_erode_iters)

            # cls2 filter mask: apply pre-loaded downsampled boolean mask.
            if (self._target_chart == 'SOD'
                    and _cls2_mask is not None):
                _mask_crop = _cls2_mask[
                    row_rand:row_rand + self.options['patch_size'],
                    col_rand:col_rand + self.options['patch_size']
                ]
                patch[self._target_chart_idx][_mask_crop] = 255

            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)
            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def _read_tile(self, tile_i):
        """读一个 chunk 对齐的 tile。返回 (len(full_variables), T, T) 的 float32 数组。

        tile 的 row/col 都是 tile_size 的整数倍且 tile_size 等于 HDF5 chunk 边长，所以每个
        变量恰好只解压 1 个 chunk（约 27ms/tile）；原点不对齐的读取会碰到 2×2 个 chunk，
        慢约 1.7 倍。
        """
        scene_id = int(self._tile_scene[tile_i])
        r = int(self._tile_row[tile_i])
        c = int(self._tile_col[tile_i])
        T = self._tile_size

        path = os.path.join(self.options['path_to_train_data'], self.files[scene_id])
        handle = _get_h5_handle(path)

        full_variables = self.options['full_variables']
        tile = np.empty((len(full_variables) + self._n_extra, T, T), dtype=np.float32)
        for j, var in enumerate(full_variables):
            tile[j] = handle[var][r:r + T, c:c + T]
        if self._comp_on:
            tile[-1] = self._read_bag_tile(scene_id, r, c, T)
        return tile, scene_id

    def _crop_from_tile(self, tile):
        """在 tile 内随机裁一个 patch_size 的 patch。返回 (x_patch, y_patch) 或 (None, None)。

        裁剪原点限制在 [0, T-ps]，patch 完全落在对齐 tile 内部（不跨 chunk）。跨 tile 边界
        的浮冰在单个 patch 里只能看到一部分，但相邻 tile 会从另一侧覆盖到它。
        """
        ps = self.options['patch_size']
        T = self._tile_size
        n_charts = len(self.options['charts'])

        row_rand = np.random.randint(low=0, high=T - ps + 1)
        col_rand = np.random.randint(low=0, high=T - ps + 1)
        # 全程 float32（标签 0..255 在 float32 中精确可表示），省掉 legacy 路径里那次
        # float64 → float32 的 .type() 拷贝。
        patch = np.ascontiguousarray(tile[:, row_rand:row_rand + ps, col_rand:col_rand + ps])

        # 黑边过滤（tile 级已粗筛过 valid_ratio，这里是 patch 级精筛，纯内存操作）
        if self.global_valid_mask_idx is not None:
            valid_ratio = float(np.mean(patch[self.global_valid_mask_idx]))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        # 标签覆盖：patch 里至少要有 2 个能参与 loss 的像元
        target_ch = patch[self._target_chart_idx]
        if np.sum(target_ch != self.options['class_fill_values'][self._target_chart]) <= 1:
            return None, None

        # FLOE: 冰山(6)→255，并按 floe_class_merge 合并类别（见 remap_floe_labels）
        if self._target_chart == 'FLOE':
            remap_floe_labels(target_ch, self.options.get('floe_class_merge'))

        _erode_iters = self.options.get('boundary_erosion_iters', 0)
        if _erode_iters > 0:
            patch[self._target_chart_idx] = _erode_sod_boundaries(
                patch[self._target_chart_idx], erosion_iters=_erode_iters)

        # patch 已是 float32，from_numpy 零拷贝。
        if self._n_extra:   # the bag channel is the last tile channel; it belongs to y
            x_patch = torch.from_numpy(patch[n_charts:-self._n_extra]).unsqueeze(0)
            y_patch = torch.from_numpy(np.concatenate([patch[:n_charts], patch[-self._n_extra:]])).unsqueeze(0)
            return x_patch, y_patch
        x_patch = torch.from_numpy(patch[n_charts:]).unsqueeze(0)
        y_patch = torch.from_numpy(patch[:n_charts]).unsqueeze(0)
        return x_patch, y_patch

    def _getitem_tile(self):
        """Tile 采样路径：组装一个 batch，返回 (x_patches, y_patches, domains)。

        tile 已按类别直方图加权抽取，目标分布由 tile_sampling.target_pixel_mix 指定，所以
        legacy 路径的概率拒绝（water_rejection / rare_samplers / cls4_rejection）在这里都不
        生效。patch 级只检查黑边和标签覆盖，失败时在同一个 tile 内换位置重裁。
        """
        _n_channels = (len(self.options['train_variables'])
                       + (2 if self.options.get('month_encoding', False) else 0)
                       + (1 if self.options.get('pol_ratio_channel', False) else 0)
                       + (N_TEXTURE_CHANNELS if self.options.get('texture_channels', False) else 0))
        batch_size = self.options['batch_size']
        ps = self.options['patch_size']
        x_patches = torch.zeros((batch_size, _n_channels, ps, ps))
        y_patches = torch.zeros((batch_size, len(self.options['charts']) + self._n_extra, ps, ps))
        # segformer_dann 的逐 patch 域序号（self.files 下标）；关闭时为 None。
        domains = torch.zeros(batch_size, dtype=torch.long) if self._dann_on else None

        max_crop_attempts = max(1, self.options.get('max_crop_attempts_per_scene', 20))
        sample_n = 0

        while sample_n < batch_size:
            tile_i = int(np.searchsorted(self._tile_cum, np.random.rand()))
            tile_i = min(tile_i, len(self._tile_cum) - 1)
            try:
                tile, scene_id = self._read_tile(tile_i)
            except Exception as e:
                print(f'[TileSampling] 读取 tile {tile_i} '
                      f'({self.files[int(self._tile_scene[tile_i])]}) 失败: {e}. 换一个 tile。',
                      flush=True)
                continue
            self._diag_counts['cache_miss'] += 1  # tile 路径下计的是读入的 tile 数

            for _ in range(self._patches_per_tile):
                if sample_n >= batch_size:
                    break
                for _attempt in range(max_crop_attempts):
                    self._diag_counts['attempts'] += 1
                    x_patch, y_patch = self._crop_from_tile(tile)
                    if x_patch is None:
                        self._diag_counts['reject_invalid_patch'] += 1
                        continue
                    x_patch = self._augment_x(x_patch, scene_id)
                    if self.do_transform:
                        x_patch, y_patch = self.transform(x_patch, y_patch)
                    x_patches[sample_n] = x_patch
                    y_patches[sample_n] = y_patch
                    if domains is not None:
                        domains[sample_n] = scene_id
                    sample_n += 1
                    self._diag_counts['accepted'] += 1
                    break

        return x_patches, y_patches, domains

    def prep_dataset(self, x_patches, y_patches, domains=None):
        """
        Cast a batch of patches and split the labels into a per-chart dict.

        Parameters
        ----------
        x_patches : Tensor [PATCH, CHANNEL, H, W]
            Input channels (train_variables plus derived channels).
        y_patches : Tensor [PATCH, len(charts), H, W]
            Label channels.
        domains : LongTensor [PATCH] or None
            segformer_dann 的逐 patch 域序号（self.files 下标）。None 时 y 中没有 '_domain' 键。

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        x = x_patches.type(torch.float)

        y = {}
        for idx, chart in enumerate(self.options['charts']):
            y[chart] = y_patches[:, idx].type(torch.long)

        if getattr(self, '_comp_on', False):
            v = y_patches[:, len(self.options['charts'])].type(torch.long)
            y['_bag'] = torch.where(v >= BAG_OFFSET, v - BAG_OFFSET, torch.full_like(v, -1))

        # 下划线前缀的保留键：训练循环只按 charts 遍历 y，'_domain' 只由 DANN 分支取用。
        if domains is not None:
            y['_domain'] = domains.long()

        return x, y

    def transform(self, x_patch, y_patch):
        data_aug_options = self.options['data_augmentations']
        if torch.rand(1) < data_aug_options['Random_h_flip']:
            x_patch = TF.hflip(x_patch)
            y_patch = TF.hflip(y_patch)

        if torch.rand(1) < data_aug_options['Random_v_flip']:
            x_patch = TF.vflip(x_patch)
            y_patch = TF.vflip(y_patch)

        # 无损 90° 旋转（不重采样、无 fill 区），与 h/v flip 一起覆盖完整的 D4 群；默认关闭。
        # patch 是方形，形状不变。
        if torch.rand(1) < data_aug_options.get('Random_rot90_prob', 0):
            k = int(np.random.randint(1, 4))   # {1,2,3} → 90/180/270°
            x_patch = torch.rot90(x_patch, k, dims=[-2, -1])
            y_patch = torch.rot90(y_patch, k, dims=[-2, -1])

        assert (data_aug_options['Random_rotation'] <= 180)
        if data_aug_options['Random_rotation'] != 0 and \
                torch.rand(1) < data_aug_options['Random_rotation_prob']:
            random_degree = np.random.randint(-data_aug_options['Random_rotation'],
                                                data_aug_options['Random_rotation']
                                                )
        else:
            random_degree = 0

        scale_diff = data_aug_options['Random_scale'][1] - \
            data_aug_options['Random_scale'][0]
        assert (scale_diff >= 0)
        if scale_diff != 0 and torch.rand(1) < data_aug_options['Random_scale_prob']:
            random_scale = np.random.rand()*(data_aug_options['Random_scale'][1] -
                                                data_aug_options['Random_scale'][0]) +\
                data_aug_options['Random_scale'][0]
        else:
            random_scale = 1.0

        # 恒等变换时跳过 affine：NEAREST 插值下恒等 affine 无损，跳过只省时间、不改变数值。
        if random_degree != 0 or random_scale != 1.0:
            x_patch = TF.affine(x_patch, angle=random_degree, translate=(0, 0),
                                shear=0, scale=random_scale, fill=0)
            y_patch = TF.affine(y_patch, angle=random_degree, translate=(0, 0),
                                shear=0, scale=random_scale, fill=255)

        return x_patch, y_patch

    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        limit_torch_threads()

        if self._tile_sampling:
            x_patches, y_patches, domains = self._getitem_tile()
        else:
            x_patches, y_patches, domains = self._getitem_legacy()

        if self.do_transform and torch.rand(1) < self.options['data_augmentations']['Cutmix_prob']:
            lam = np.random.beta(self.options['data_augmentations']['Cutmix_beta'],
                                  self.options['data_augmentations']['Cutmix_beta'])
            rand_index = torch.randperm(x_patches.size(0))
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_patches.size(), lam)
            x_patches[:, :, bbx1:bbx2, bby1:bby2] = x_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            y_patches[:, :, bbx1:bbx2, bby1:bby2] = y_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            # domains 刻意不随 cutmix 混合：混入区成为判别器的少量标签噪声。DANN 判别器只看
            # 整个 patch 的全局向量，按面积给软标签需要改损失形式。
        x, y = self.prep_dataset(x_patches, y_patches, domains)

        # 每 _diag_log_every 个 batch 打印一次采样诊断并清零（每个 worker 各自打印）。tile 路径
        # 下 cache_miss 是读入的 tile 数，reject_* 中只有 invalid_patch（黑边或标签覆盖不足）计数。
        self._diag_batches_done += 1
        if self._diag_log_every > 0 and self._diag_batches_done % self._diag_log_every == 0:
            c = self._diag_counts
            worker_info = get_worker_info()
            wid = worker_info.id if worker_info is not None else 'main'
            print(f"[SamplingDiag worker={wid}] last {self._diag_log_every} batches: "
                  f"attempts={c['attempts']} accepted={c['accepted']} cache_miss={c['cache_miss']} | "
                  f"reject: invalid_patch={c['reject_invalid_patch']} label_cov={c['reject_label_cov']} "
                  f"water={c['reject_water']} cls4={c['reject_cls4']} rare={c['reject_rare']}", flush=True)
            for k in c:
                c[k] = 0

        return x, y

    def _getitem_legacy(self):
        """Legacy 路径：随机选场景，整景解码（或命中 LRU 缓存），随机裁，再经概率拒绝过滤。

        仅在未配置 tile_index_path 时使用（未建 tile 索引的数据集，如 SOD）。FLOE 走
        _getitem_tile：整景解码约 9s/景，这条路径在原生分辨率下太慢。
        """
        _n_channels = (len(self.options['train_variables'])
                       + (2 if self.options.get('month_encoding', False) else 0)
                       + (1 if self.options.get('pol_ratio_channel', False) else 0)
                       + (N_TEXTURE_CHANNELS if self.options.get('texture_channels', False) else 0))
        x_patches = torch.zeros((self.options['batch_size'], _n_channels,
                                 self.options['patch_size'], self.options['patch_size']))
        y_patches = torch.zeros((self.options['batch_size'], len(self.options['charts']),
                                 self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # 每个场景解码（或取缓存）一次后产出最多 patches_per_scene 个 patch，每个 patch 最多
        # 在同一场景内重裁 max_crop_attempts 次，把解码成本分摊到多个 patch 上。两者下限
        # 为 1：配成 0 会让内层循环不执行，外层 while 无声空转。
        max_crop_attempts = max(1, self.options.get('max_crop_attempts_per_scene', 20))
        patches_per_scene = max(1, self.options.get('patches_per_scene', 6))

        while sample_n < self.options['batch_size']:
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()
            for _patch_idx in range(patches_per_scene):
                if sample_n >= self.options['batch_size']:
                    break
                got_patch = False  # 本次(同场景内)是否成功产出一个 patch
                fatal_error = False  # 场景解码/裁剪本身出错，整个场景作废
                for _attempt in range(max_crop_attempts):
                    self._diag_counts['attempts'] += 1
                    # 每 20 次尝试打印一次心跳，第一个 batch 组装很慢时也能看到进度。
                    if self._diag_counts['attempts'] % 20 == 0:
                        worker_info = get_worker_info()
                        wid = worker_info.id if worker_info is not None else 'main'
                        print(f"[SamplingDiag worker={wid}] heartbeat: sample_n={sample_n}/"
                              f"{self.options['batch_size']} attempts={self._diag_counts['attempts']} "
                              f"cache_miss={self._diag_counts['cache_miss']}", flush=True)
                    try:
                        self._last_scene_was_cache_hit = None
                        x_patch, y_patch = self.random_crop_downsample(scene_id)
                        if self._last_scene_was_cache_hit is False:
                            self._diag_counts['cache_miss'] += 1
                    except FileNotFoundError:
                        print(f"File {self.files[scene_id]} not found. Skipping scene.")
                        fatal_error = True
                        break
                    except Exception:
                        print(f"Cropping in {self.files[scene_id]} failed. Skipping.")
                        fatal_error = True
                        break

                    if x_patch is None:
                        self._diag_counts['reject_invalid_patch'] += 1
                        continue  # 同一场景换个裁剪位置重试

                    x_patch = self._augment_x(x_patch, scene_id)
                    target_flat = y_patch[0, self._target_chart_idx].numpy().flatten()

                    # Filter 1: label coverage.
                    sod_invalid_max = self.options.get('sod_invalid_max_ratio', 1.0)
                    if sod_invalid_max < 1.0:
                        if float((target_flat == 255).sum()) / len(target_flat) > sod_invalid_max:
                            self._diag_counts['reject_label_cov'] += 1
                            continue

                    # Filter 2: water patch rejection.
                    water_max_ratio = self.options.get('water_patch_max_ratio', 1.0)
                    water_reject_prob = self.options.get('water_rejection_prob', 0.0)
                    if water_reject_prob > 0.0 and water_max_ratio < 1.0:
                        valid_mask = target_flat != 255
                        valid_count = valid_mask.sum()
                        if valid_count > 0:
                            water_ratio = float((target_flat[valid_mask] == 0).sum()) / valid_count
                            if water_ratio > water_max_ratio and np.random.rand() < water_reject_prob:
                                self._diag_counts['reject_water'] += 1
                                continue

                    # Filter 2b: cls4 (MYI) patch rejection, same scheme as water.
                    cls4_max_ratio = self.options.get('cls4_patch_max_ratio', 1.0)
                    cls4_reject_prob = self.options.get('cls4_rejection_prob', 0.0)
                    if cls4_reject_prob > 0.0 and cls4_max_ratio < 1.0:
                        valid_mask = target_flat != 255
                        valid_count = valid_mask.sum()
                        if valid_count > 0:
                            cls4_ratio = float((target_flat[valid_mask] == 4).sum()) / valid_count
                            if cls4_ratio > cls4_max_ratio and np.random.rand() < cls4_reject_prob:
                                self._diag_counts['reject_cls4'] += 1
                                continue

                    # Filter 3: rare-class weighted sampling.
                    # Config: rare_samplers = [{'classes': [...], 'alpha': float}, ...], or the
                    # older single-sampler keys rare_sampling_classes / rare_sampling_alpha
                    # (rare_samplers takes precedence). With several samplers the acceptance
                    # probability is the max over samplers, so classes can be tuned independently.
                    _samplers = self.options.get('rare_samplers', None)
                    if _samplers is None:
                        # Fall back to legacy single-sampler config.
                        _cls = self.options.get('rare_sampling_classes', [])
                        _alpha = self.options.get('rare_sampling_alpha', 0.0)
                        _samplers = [{'classes': _cls, 'alpha': _alpha}] if _cls and _alpha > 0.0 else []
                    if _samplers:
                        _valid_target = target_flat[target_flat != 255]
                        if len(_valid_target) > 0:
                            _accept_prob = 0.0
                            for _s in _samplers:
                                _alpha_s = _s.get('alpha', 0.0)
                                _classes_s = _s.get('classes', [])
                                if not _classes_s or _alpha_s <= 0.0:
                                    continue
                                _rare_count = float(sum((_valid_target == c).sum() for c in _classes_s))
                                _rare_frac = _rare_count / len(_valid_target)
                                _p = (1.0 - _alpha_s) + _alpha_s * _rare_frac
                                if _p > _accept_prob:
                                    _accept_prob = _p
                            if _accept_prob > 0.0 and np.random.rand() > _accept_prob:
                                self._diag_counts['reject_rare'] += 1
                                continue

                    if self.do_transform:
                        x_patch, y_patch = self.transform(x_patch, y_patch)

                    x_patches[sample_n, :, :, :] = x_patch
                    y_patches[sample_n, :, :, :] = y_patch
                    sample_n += 1
                    self._diag_counts['accepted'] += 1
                    got_patch = True
                    break  # 跳出 attempt 循环，尝试从同一场景再产出下一个 patch

                if fatal_error or not got_patch:
                    # 解码出错，或 max_crop_attempts 次内都没通过过滤器：放弃这个场景。
                    break

        # legacy 路径不支持 DANN（__init__ 已拒绝），domains 恒为 None。
        return x_patches, y_patches, None


class AI4ArcticChallengeTestDataset(Dataset):
    """Validation / test dataset: each item is one whole scene (see __getitem__)."""

    def __init__(self, options, files, files_reference=None, mode='test',
                 ondemand_read=False):
        self.options = options
        self.files = files
        self.files_reference = files_reference
        # 按需读默认关闭，由调用方显式开启：多数消费者（可视化、推理导出等）不满足
        # 「只在对齐的 tile 上前向」的前提。判定见 val_ondemand_ok()。
        self.ondemand_read = ondemand_read

        if mode not in ["train", "test", "test_no_gt"]:
            raise ValueError("String variable must be one of 'train', 'test', or 'test_no_gt'")
        self.mode = mode

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

    def prep_scene(self, scene, scene_y=None, filename=None):
        """
        Load a full scene for inference/validation. For MyDS, all variables are at
        the same 80m resolution, so no upsampling is needed.

        Parameters
        ----------
        scene :
            Xarray dataset for input features.
        scene_y :
            Optional separate dataset for labels (targets). If None, uses scene.

        Returns
        -------
        x :
            4D torch tensor, ready inference data.
        y :
            Dict with 2D numpy arrays for each chart. None if mode is 'test_no_gt'.
        """
        # All MyDS variables are at the same 80m resolution - load all at once.
        x = torch.from_numpy(
            scene[self.options['train_variables']].to_array().values).unsqueeze(0).float()

        # 验证/测试时可用 val_downsample_scale 单独控制分辨率，避免大场景 GPU OOM
        if self.mode in ('train', 'test', 'test_no_gt') and self.options.get('val_downsample_scale', 1) != 1:
            effective_scale = self.options['val_downsample_scale']
        else:
            effective_scale = self.options['down_sample_scale']

        if effective_scale != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=1/effective_scale,
                mode=self.options['loader_downsampling'])

        if self.mode != 'test_no_gt':
            scene_for_y = scene_y if scene_y is not None else scene
            y_charts = torch.from_numpy(
                scene_for_y[self.options['charts']].isel().to_array().values).unsqueeze(0)
            if effective_scale != 1:
                y_charts = torch.nn.functional.interpolate(
                    y_charts, scale_factor=1/effective_scale, mode='nearest')

            y = {}
            for idx, chart in enumerate(self.options['charts']):
                y[chart] = y_charts[:, idx].squeeze().numpy()

            y = self._finalize_labels(y, filename, effective_scale)
        else:
            y = None

        # 与训练相同：分辨率退化在派生通道之前（整景从原点对齐，即全局网格）。
        x = degrade_resolution_(x, self.options['train_variables'],
                                self.options.get('input_degrade_factor', 1))

        # Append sin/cos month encoding as 2 constant spatial channels.
        if self.options.get('month_encoding', False) and filename is not None:
            s, c = _month_sin_cos(filename)
            H, W = x.shape[-2], x.shape[-1]
            month_ch = torch.tensor([s, c], dtype=torch.float).view(1, 2, 1, 1).expand(1, 2, H, W)
            x = torch.cat([x, month_ch], dim=1)

        # Append HH-HV polarization ratio channel (dB difference).
        # nersc_sar_primary (HH) is channel 0, nersc_sar_secondary (HV) is channel 1.
        if self.options.get('pol_ratio_channel', False):
            pol_ratio = x[:, 0:1, :, :] - x[:, 1:2, :, :]
            x = torch.cat([x, pol_ratio], dim=1)

        # 纹理通道不在这里加：functions.fast_tiled_val_predict 对每个 tile 单独计算（与训练
        # patch 同口径），避免整景多出 4 个通道的内存，也保留按需读。
        return x.float(), y

    def _finalize_labels(self, y, filename, effective_scale):
        """标签后处理：类别重映射 / 边界腐蚀 / cls2 污染掩膜。

        整景读取和按需读取都走这里，两条路径的标签才一致。
        """
        target_chart = self.options.get('target_chart', 'FLOE')

        # SOD-specific: remap labels: merge old class 2 (young ice) into class 1
        # (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
        if target_chart == 'SOD' and 'SOD' in y:
            sod = y['SOD'].copy()
            valid = sod != 255
            sod[valid & (sod >= 2)] -= 1
            y['SOD'] = sod

        # FLOE: 冰山(6)→255，并按与训练相同的 floe_class_merge 合并类别（见 remap_floe_labels）
        elif target_chart == 'FLOE' and 'FLOE' in y:
            floe = y['FLOE'].copy()
            remap_floe_labels(floe, self.options.get('floe_class_merge'))
            y['FLOE'] = floe

        # Boundary erosion on validation labels: mask out class-boundary pixels that
        # are most likely mislabeled due to spatial misalignment between CIS ice
        # charts and SAR. Controlled by 'boundary_erosion_iters' (default 0 = off).
        _erode_iters = self.options.get('boundary_erosion_iters', 0)
        if _erode_iters > 0 and target_chart in y:
            y[target_chart] = _erode_sod_boundaries(
                y[target_chart], erosion_iters=_erode_iters)

        # cls2 filter mask: apply pre-generated pollution mask to validation labels.
        # Polluted cls2 pixels are set to 255 (ignore), consistent with training.
        _cls2_mask_dir = self.options.get('cls2_filter_mask_dir', None)
        if (_cls2_mask_dir is not None
                and target_chart == 'SOD' and 'SOD' in y
                and filename is not None):
            _mn = os.path.basename(filename).replace('.nc', '_cls2_mask.npy')
            _mp = os.path.join(_cls2_mask_dir, _mn)
            if os.path.exists(_mp):
                _full_mask = np.load(_mp)           # bool (H_full, W_full)
                if effective_scale != 1:
                    _mask_t = torch.from_numpy(_full_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    _h_ds, _w_ds = y['SOD'].shape[0], y['SOD'].shape[1]
                    _mask_ds = torch.nn.functional.interpolate(
                        _mask_t, size=(_h_ds, _w_ds), mode='nearest'
                    ).squeeze().numpy() > 0.5
                else:
                    _mask_ds = _full_mask
                y['SOD'][_mask_ds] = 255
        return y

    def _ondemand_enabled(self):
        """本样本是否走按需读（只读含有效标签的 chunk）。

        原生分辨率下验证的瓶颈是数据加载而不是 GPU 前向，而验证集只有约 39% 的 chunk 含
        有效标签；按需读约 5.4s/景，整景读约 10.1s/景。以下情形退回整景读取：调用方没有
        传 ondemand_read=True（判定见 val_ondemand_ok）；没有标签（test_no_gt），无从判断
        需要哪些 chunk；降采样（整景 interpolate 需要全部像素）；config 关闭 val_ondemand_read。
        """
        return (self.ondemand_read
                and self.mode in ('train', 'test')
                and self.options.get('val_ondemand_read', True)
                and self.options.get('down_sample_scale', 1) == 1
                and self.options.get('val_downsample_scale', 1) == 1)

    def prep_scene_ondemand(self, path, filename=None):
        """用 h5py 只读含有效标签的 chunk，其余位置留 0。返回值与 prep_scene 相同。

        未读区域对应的 tile 会被 functions.compute_valid_tile_mask 挡掉、不进网络，指标不变
        （tools/checks/ondemand_read.py 与整景路径逐位对拍）。

        整景读取的例外：标签变量（用来决定读哪些 chunk，uint8 很便宜）；global_valid_mask
        （tfv_mask 由它推出，按变量名强制整读，不靠 dtype 判断）；其余单字节变量（便宜）。

        内存：所有通道（含 month / pol_ratio）一次性分配在同一个数组里，派生通道只在必读
        chunk 上计算。不要改成 torch.cat：它会新分配并写满一个整景张量（最大景约 8GB），
        未读页面不占物理内存的前提就失效了。
        """
        charts = self.options['charts']
        variables = list(self.options['train_variables'])
        patch_size = self.options['patch_size']
        use_month = bool(self.options.get('month_encoding', False)) and filename is not None
        use_pol = bool(self.options.get('pol_ratio_channel', False))

        with h5py.File(path, 'r', rdcc_nbytes=64 * 1024 ** 2, rdcc_nslots=4001) as f:
            # 1. 标签整景读取，后处理与整景路径相同
            y = {chart: f[chart][:] for chart in charts}
            y = self._finalize_labels(y, filename, effective_scale=1)

            # 2. 由最终标签算出需要哪些 chunk
            first = f[variables[0]]
            H, W = first.shape[-2], first.shape[-1]
            chunk = (first.chunks[-1] if first.chunks else patch_size)
            need = _need_chunk_grid(y, self.options['class_fill_values'], charts,
                                    H, W, patch_size, chunk)
            if not self.options.get('val_tile_gate', True):
                # Without the tile gate every tile is run, so every chunk must be read (the
                # *_evalfull configs; the xarray whole-scene path cannot open the v2 files).
                need = np.ones_like(need, dtype=bool)

            # 3. 一次性分配所有通道，顺序与 prep_scene 一致：变量、month、pol
            n_var = len(variables)
            c_month = n_var if use_month else None
            c_pol = (n_var + (2 if use_month else 0)) if use_pol else None
            n_ch = n_var + (2 if use_month else 0) + (1 if use_pol else 0)
            # dtype 固定 float32，与 prep_scene 的 .float() 一致；按 result_type 推导的话，任一
            # 变量是 float64 就会让整景数组大小翻倍。
            x_np = np.zeros((n_ch, H, W), dtype=np.float32)

            dsets = [f[v] for v in variables]
            full_read = [(v == 'global_valid_mask' or d.dtype.itemsize == 1)
                         for v, d in zip(variables, dsets)]
            for k, d in enumerate(dsets):
                if full_read[k]:
                    x_np[k] = d[:]

            month_s, month_c = _month_sin_cos(filename) if use_month else (0.0, 0.0)
            for ci, cj in np.argwhere(need):
                y0, x0 = int(ci) * chunk, int(cj) * chunk
                y1, x1 = min(y0 + chunk, H), min(x0 + chunk, W)
                for k, d in enumerate(dsets):
                    if not full_read[k]:
                        x_np[k, y0:y1, x0:x1] = d[y0:y1, x0:x1]
                # chunk 原点是 512 的倍数，k 整除 512 时 k×k 块与整景路径的网格一致
                degrade_resolution_(x_np[:n_var, y0:y1, x0:x1], variables,
                                    self.options.get('input_degrade_factor', 1))
                # 派生通道只在必读 chunk 上算：写满整景会毁掉稀疏性（见 docstring）
                if c_month is not None:
                    x_np[c_month, y0:y1, x0:x1] = month_s
                    x_np[c_month + 1, y0:y1, x0:x1] = month_c
                if c_pol is not None:
                    x_np[c_pol, y0:y1, x0:x1] = (x_np[0, y0:y1, x0:x1]
                                                 - x_np[1, y0:y1, x0:x1])

        return torch.from_numpy(x_np).unsqueeze(0), y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 2D numpy arrays for each chart. None if mode is 'test_no_gt'.
        cfv_masks :
            Dict of boolean masks for class fill values (label mask). None if 'test_no_gt'.
        tfv_mask :
            2D boolean mask for train fill value (input invalid region / black border).
        name : str
            Filename of scene.
        original_size : tuple
            (H, W) of the scene before any downsampling.
        """
        root = (self.options['path_to_train_data'] if self.mode == 'train'
                else self.options['path_to_test_data'])
        path = os.path.join(root, self.files[idx])

        if self._ondemand_enabled():
            # 只读含有效标签的 chunk，其余留 0（见 prep_scene_ondemand）。
            x, y = self.prep_scene_ondemand(path, filename=self.files[idx])
            with h5py.File(path, 'r') as _f:
                original_size = tuple(int(v) for v in _f['nersc_sar_primary'].shape)
        else:
            scene = xr.open_dataset(path, engine='h5netcdf', mask_and_scale=False)
            # For MyDS, labels are embedded in the same file as features.
            scene_y = scene if self.mode == 'test' else None
            x, y = self.prep_scene(scene, scene_y, filename=self.files[idx])
            original_size = scene['nersc_sar_primary'].values.shape
        name = self.files[idx]

        if self.mode != 'test_no_gt':
            cfv_masks = {}
            for chart in self.options['charts']:
                cfv_masks[chart] = (
                    y[chart] == self.options['class_fill_values'][chart]).squeeze()
        else:
            cfv_masks = None

        # Use global_valid_mask as the train fill value mask (more accurate than checking SAR=0).
        if 'global_valid_mask' in self.options['train_variables']:
            mask_idx = list(self.options['train_variables']).index('global_valid_mask')
            tfv_mask = (x.squeeze()[mask_idx, :, :] == 0).squeeze()
        else:
            tfv_mask = (x.squeeze()[0, :, :] == self.options['train_fill_value']).squeeze()

        return x, y, cfv_masks, tfv_mask, name, original_size


def val_scoped_options(train_options: dict):
    """返回验证/评估侧使用的 options：charts 坍缩为 val_charts = [target_chart]。

    aux_charts 只在训练期生效（理由见 get_variable_options；另外验证景未必有辅任务变量）。
    没配 aux_charts 时直接返回 train_options 本身，不是副本。凡是拿 checkpoint 在验证集上
    做前向的脚本，都应在 get_variable_options 之后调用本函数。
    """
    val_charts = train_options.get('val_charts', train_options['charts'])
    if val_charts == train_options['charts']:
        return train_options            # 单任务：返回同一对象
    val_options = dict(train_options)   # 浅拷贝足够：只覆盖两个顶层键
    val_options['charts'] = val_charts
    val_options['task_weights'] = [1] * len(val_charts)
    return val_options


def get_variable_options(train_options: dict):
    """
    Set up variable category lists for MyDS dataset.

    In MyDS all input variables share the same 80m pixel spacing as the SAR imagery,
    so no upsampling is required. All train_variables are treated as full-resolution.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.

    Returns
    -------
    train_options: dict
        Updated with variable category lists.
    """
    # Derive active charts and task weights from target_chart.
    # Set 'target_chart' in config to 'SOD' or 'FLOE' to switch the detection target.
    target_chart = train_options.get('target_chart', 'FLOE')
    # 辅助任务（aux_charts）只在训练期生效：它们进 train_options['charts']，loss 与 dataloader
    # 带上其标签，但验证/评估只跑 target chart（val_charts）。原因：SIC 这类辅任务几乎处处有
    # 标签，进了验证会让 tile 门控的跳过率从约 66% 降到接近 0（验证耗时约 3 倍，按需读也失效）；
    # 验证 loss 也会变成 target 与辅任务的加权和，与只含 target 的 val loss 不可比。
    aux_charts = [c for c in train_options.get('aux_charts', []) if c != target_chart]
    train_options['charts'] = [target_chart] + aux_charts
    aux_w = train_options.get('aux_task_weights', [0.2] * len(aux_charts))
    if len(aux_w) != len(aux_charts):
        raise ValueError(
            f"aux_task_weights 长度 {len(aux_w)} != aux_charts 长度 {len(aux_charts)}"
            f"（aux_charts={aux_charts}，已剔除与 target_chart 重复的项；"
            f"aux_task_weights={list(aux_w)}）")
    train_options['task_weights'] = [1] + [float(w) for w in aux_w]
    # 验证恒为 target 单任务（见 val_scoped_options）。
    train_options['val_charts'] = [target_chart]
    if aux_charts:   # 只在配了辅任务时打印
        print(f"[Charts] train={train_options['charts']} "
              f"task_weights={train_options['task_weights']} | "
              f"val={train_options['val_charts']}（辅任务 train-only）")

    # Update chart_metric weights: 1 for target, 0 for others.
    for chart in train_options.get('chart_metric', {}):
        train_options['chart_metric'][chart]['weight'] = 1 if chart == target_chart else 0

    # All train variables in MyDS are full-resolution (80m) - no separate AMSR/env grids.
    train_options['sar_variables'] = list(train_options['train_variables'])
    train_options['full_variables'] = list(train_options['charts']) + train_options['sar_variables']
    train_options['amsrenv_variables'] = []
    train_options['auxiliary_variables'] = []

    return train_options
