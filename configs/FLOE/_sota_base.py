# 对比模型共享的训练协议。configs/FLOE/sota_*.py 只设 model_selection 及模型自身选项，
# 其余超参全部从这里继承，使 benchmark 中唯一的变量是网络架构。不含 model_selection，
# 不能直接训练。数据路径、charts、task_weights、chart_loss 继承自 ../_base_/base.py。
#
# 输入：pol_ratio_channel=True → 8 通道（7 个 train_variables + HH-HV 极化比）；
# 所有 sota_* 模型按 len(train_variables)+pol_ratio 推导 in_channels。

import os

_base_ = ['../_base_/base.py']

train_options = {
    # ---- 数据划分：417 景，train 375 / val 42 ----
    # val 冰区以 big/vast 为主（89.4%），像素加权的总体指标偏向这两类；逐类 IoU/F1 不受影响。
    'train_list_path': 'datalists/train_417.json',
    'val_path': 'datalists/val_417.json',
    # 无独立测试集：test_path 只在启动时读入，训练循环不使用，占位即可。
    'test_path': 'datalists/val_417.json',

    'pol_ratio_channel': True,

    # ---- 训练规模 ----
    'seed': 42,
    'epochs': 200,
    'epoch_len': 300,
    'patch_size': 256,
    'batch_size': 16,
    'down_sample_scale': 1,   # 原生分辨率：小/中浮冰只有 1~6 px，不能下采样

    # ---- tile 采样（见 docs/tile_sampling.md）----
    # 换数据目录或 patch_size 后必须重扫索引（loaders 启动时会校验）：
    #   python -m tools.data.scan_tiles --data-dir /root/autodl-tmp/FLOE_datanc \
    #       --out /root/autodl-tmp/floe_tile_index_417.npz --patch-size 256 --workers 16
    # Override with FLOE_TILE_INDEX (same variable as tools/figures/common.py).
    'tile_index_path': os.environ.get('FLOE_TILE_INDEX', '/root/autodl-tmp/floe_tile_index_417.npz'),
    'tile_sampling': dict(
        # cake 只占有效像元约 0.5%，0.10 已是 ~20 倍过采样；再高会让采样集中到极少数 tile。
        # 其余四个冰类均分剩余份额，水 0.15。启动日志的 expected batch pixel mix
        # （偏差 >0.02 会 WARN）和 effective tiles 用来核对目标是否可达。
        target_pixel_mix={0: 0.15, 1: 0.10, 2: 0.1875, 3: 0.1875, 4: 0.1875, 5: 0.1875},
        patches_per_tile=2,
        min_label_valid_ratio=0.5,
    ),
    'enable_scene_cache': False,

    # ---- 验证 / 早停 ----
    # 验证走增量混淆矩阵，val mIoU 与 val_batch_size 无关，它只影响验证峰值显存。
    'val_batch_size': 32,
    # early_stop_patience 和 val_smoothing_window 都以「验证次数」计，改 val_freq 会同时
    # 改变两者覆盖的 epoch 跨度，破坏与已有 run 的可比性。
    'val_freq': 2,
    'early_stop_patience': 20,   # val_freq=2 下即 40 个 epoch 无提升
    # best checkpoint 与早停都用最近 3 次验证 mIoU 的滑动均值：单次 val mIoU 噪声 std~0.02，
    # 取单次最大值会选中偶然尖峰（换 seed 复跑的 run 间方差 0.033，平滑后 0.007）。
    'val_smoothing_window': 3,

    'compile_model': False,

    # ---- 优化器 / 调度器 ----
    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.05,
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'lr_min': 1e-6,
    },

    # ---- DataLoader（瓶颈是 gzip 解压，CPU-bound）----
    'num_workers': 12,
    # 每个验证 worker 解码一整景原生分辨率场景（最大 6369×23975），是验证期内存峰值来源。
    # 实测 42 景（按需读 + tile 门控）：w=1 278s / 25.2GB，w=2 183s / 26.7GB，w=3 159s / 33.5GB。
    # 内存要看 cgroup 总量（anon + shmem）：多 worker 的 payload 走 /dev/shm，只看 anon 会低估。
    # 验证期若被 SIGKILL（rc=137），调回 1。
    'num_workers_val': 2,

    # ---- 验证提速开关（均不改变结果，设 False 可做 A/B）----
    # tile 门控：只前向标签非全 255 的 tile（val_417 上跳过 66.1%）。
    'val_tile_gate': True,
    # 按需读：只解码含有效 tile 的 chunk（val_417 上 38.6%），其余留 0，对应 tile 被门控挡掉。
    # 这只是全局开关：按需读由调用方显式 opt-in（AI4ArcticChallengeTestDataset(..., ondemand_read=)，
    # 判定见 loaders.val_ondemand_ok），因为 TTA / 可视化 / 推理导出不满足「只在对齐 tile 上前向」
    # 的前提。stride≠patch_size、集成、降采样或缺 global_valid_mask 时自动退回整景读。
    # 等价性由 tools/checks/ondemand_read.py 验证。
    'val_ondemand_read': True,
    # 逐批在 GPU 上累加混淆矩阵（fast_tiled_val_predict(out='confmat')），整景只同步一次；
    # 与旧的 preds + 掩膜筛选路径整数级相等。
    'val_confmat_batched': True,

    'prefetch_factor': 4,

    # ---- 指标 ----
    'compute_classwise_f1score': True,
    'plot_confusion_matrix': True,

    # ---- patch 级过滤（tile 采样下只剩黑边与标签覆盖两项内存检查）----
    'max_crop_attempts_per_scene': 20,
    'diag_log_every_batches': 5,

    # ---- 数据增强 ----
    'data_augmentations': {
        'Random_h_flip': 0.5,
        'Random_v_flip': 0.5,
        'Random_rotation_prob': 0.5,
        'Random_rotation': 90,
        'Random_scale_prob': 0.5,
        'Random_scale': (0.9, 1.1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0.5,
    },

    # swin_hp 只在 model_selection=='swin' 时被 batched_slide_inference 读取，sota_* 模型不需要。
}
