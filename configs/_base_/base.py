import os
from floe.functions import f1_metric, r2_metric

# Data location: override with FLOE_DATA_DIR (same variable as tools/figures/common.py).
FLOE_DATA_DIR = os.path.join(os.environ.get('FLOE_DATA_DIR', '/root/autodl-tmp/FLOE_datanc'), '')
import torch.nn as nn

CHARTS = ['FLOE']


SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- GLCM texture features (from HH channel) -- #
    'glcm_sigma0_hh_contrast',
    'glcm_sigma0_hh_dissimilarity',
    'glcm_sigma0_hh_homogeneity',

    # -- Valid mask (0=invalid/black border, 1=valid) -- #
    'global_valid_mask',
]

# Sea Ice Concentration (SIC) code to class conversion lookup table.
SIC_LOOKUP = {
    'polygon_idx': 0,  # Index of polygon number.
    'total_sic_idx': 1,  # Total Sea Ice Concentration Index, CT.
    'sic_partial_idx': [2, 5, 8],  # Partial SIC polygon code index. CA, CB, CC.
    0: 0,
    1: 0,
    2: 0,
    55: 0,
    10: 1,  # 10 %
    20: 2,  # 20 %
    30: 3,  # 30 %
    40: 4,  # 40 %
    50: 5,  # 50 %
    60: 6,  # 60 %
    70: 7,  # 70 %
    80: 8,  # 80 %
    90: 9,  # 90 %
    91: 10,  # 100 %
    92: 10,  # Fast ice
    'mask': 255,
    'n_classes': 12
}

# Stage of Development code to class conversion lookup table.
SOD_LOOKUP = {
    'sod_partial_idx': [3, 6, 9],  # Partial SIC polygon code index. SA, SB, SC.
    'threshold': 0.7,  # < 1. Minimum partial percentage SIC of total SIC to select SOD. Otherwise ambiguous polygon.
    'invalid': -9,  # Value for polygons where the SOD is ambiguous or not filled.
    'water': 0,
    0: 0,
    80: 0,  # No stage of development
    81: 1,  # New ice
    82: 1,  # Nilas, ring ice
    83: 1,  # Young ice       → 合并入新冰/幼冰类(1)
    84: 1,  # Grey ice        → 合并入新冰/幼冰类(1)
    85: 1,  # White ice       → 合并入新冰/幼冰类(1)
    86: 3,  # First-year ice, overall category
    87: 2,  # Thin first-year ice
    88: 2,  # Thin first-year ice, stage 1
    89: 2,  # Thin first-year ice, stage 2
    91: 3,  # Medium first-year ice
    93: 3,  # Thick first-year ice
    95: 4,  # Old ice        → 多年冰类别
    96: 4,  # Second year ice → 多年冰类别
    97: 4,  # Multi-year ice  → 多年冰类别
    98: 255,  # Glacier ice     → 忽略
    99: 255,
    'mask': 255,
    'n_classes': 5  # 0=水体, 1=新冰/幼冰(合并), 2=薄一年冰, 3=厚一年冰, 4=多年冰
}

# Ice floe/form code to class conversion lookup table.
FLOE_LOOKUP = {
    'floe_partial_idx': [4, 7, 10],  # Partial SIC polygon code index. FA, FB, FC.
    'threshold': 0.5,  # < 1. Minimum partial concentration to select floe. Otherwise polygon may be ambiguous.
    'invalid': -9,  # Value for polygons where the floe is ambiguous or not filled.
    'water': 0,
    0: 0,
    22: 255,  # Pancake ice
    1: 255,  # Shuga / small ice cake
    2: 1,  # Ice cake
    3: 2,  # Small floe
    4: 3,  # Medium floe
    5: 4,  # Big floe
    6: 5,  # Vast floe
    7: 5,  # Giant floe
    8: 255,  # Fast ice
    9: 255,  # Growlers, floebergs or floebits → 忽略（样本极少）
    10: 255,  # Icebergs → 忽略（样本极少）
    21: 255,  # Level ice
    'fastice_class': 255,
    'mask': 255,
    # 类别方案（见 docs/DATASET_HANDBOOK.md §4）：
    # 0=水体, 1=饼状冰(ice cake), 2=小浮冰, 3=中浮冰, 4=大浮冰, 5=巨浮冰
    'berg_class': 6,  # .nc 中的冰山标签值（本数据集无），loader 读取时置为 mask(255)
    'n_classes': 6
}


train_options = {
    # -- Random Seed -- #
    'seed': -1,
    # -- Training options -- #
    # FLOE 数据集（已 z-score 归一化，见 docs/DATASET_HANDBOOK.md §8）
    'path_to_train_data': FLOE_DATA_DIR,
    'path_to_test_data': FLOE_DATA_DIR,

    # 无独立测试集：test_path 只在启动时被读入，训练循环不使用 test_list。
    'train_list_path': 'datalists/train_417.json',
    'val_path': 'datalists/val_417.json',
    'test_path': 'datalists/val_417.json',

    'path_to_env': './',

    'epochs': 100,  # Number of epochs before training stop.
    'epoch_len': 500,  # Number of batches for each epoch.
    # Size of patches sampled. Used for both Width and Height.
    'patch_size': 256,
    'batch_size': 16,  # Number of patches for each batch.
    # How to upscale low resolution variables to high resolution.
    'loader_upsampling': 'nearest',
    # How to down scale low resolution variables to low resolution.
    'loader_downsampling': 'nearest',
    # Down Sampling scale (If it is by 2 the image will get downsample by 2)
    'down_sample_scale': 1,
    'task_weights': [1],  # 与 charts 对齐，仅 FLOE 单任务

    # -- 目标任务与选模标准 -- #
    'target_chart': 'FLOE',
    'model_save_criterion': 'miou',  # best model 按 val mIoU 保存

    # p leave out cross val run
    'cross_val_run': False,
    'p-out': 20,
    'compute_classwise_f1score': False,
    'plot_confusion_matrix': False,

    # torch.compile the model (PyTorch 2).
    'compile_model': False,

    # -- loss options -- #
    'chart_loss': {  # Loss for the task
        'SIC': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
        'SOD': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
        'FLOE': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,

        },
    },
    # > 0 adds a loss penalising inconsistent water logits across the SIC/SOD/FLOE outputs.
    "edge_consistency_loss": 0,
    # Other loss 'type's and their parameters: floe.functions.get_loss, floe/losses.py and
    # https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/losses

    # If true accuracy will be calculated using binary classification between water and ice first
    "binary_water_classifier": False,

    # AMP 训练下梯度裁剪阈值（配合 GradScaler 使用，防止单个异常 batch 的梯度范数过大冲击 AdamW 状态）
    'grad_clip_max_norm': 5.0,

    'optimizer': {
        'type': 'Adam',
        'lr': 0.0001,  # Optimizer learning rate.
        'b1': 0.9,
        'b2': 0.999,
        'weight_decay': 0.0
    },
    # Also supported: 'AdamW', 'SGD' (floe.functions.get_optimizer).

    # Also supported: 'CosineAnnealingLR', 'CosineAnnealingWarmRestartsLR', 'ReduceLROnPlateau'
    # (floe.functions.get_scheduler).
    'scheduler': {'type': None},

    # -- Data preparation lookups and metrics.
    # Contains the relevant variables in the scenes.
    'train_variables': SCENE_VARIABLES,
    'charts': CHARTS,  # Charts to train on.
    'n_classes': {  # number of total classes in the reference charts, including the mask.
        'SIC': SIC_LOOKUP['n_classes'],
        'SOD': SOD_LOOKUP['n_classes'],
        'FLOE': FLOE_LOOKUP['n_classes']
    },
    # Training patches with a smaller fraction of valid (non-black-border) pixels are discarded.
    'valid_mask_threshold': 0.5,

    # Patch sampling log mode (requires num_workers=0):
    #   'per_epoch' : 每epoch一行，记录该epoch所有patch的SOD像素总计（推荐）
    #   'per_patch' : 每patch一行，记录单个patch的SOD像素分布及来源文件名
    'patch_log_mode': 'per_epoch',

    # SAR pixel spacing in metres.
    'pixel_spacing': 80,
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },

    # -- Validation options -- #
    'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        'SIC': {
            'func': r2_metric,
            'weight': 0,
        },
        'SOD': {
            'func': f1_metric,
            'weight': 0,
        },
        'FLOE': {
            'func': f1_metric,
            'weight': 1,
        },
    },
    # Number of scenes randomly sampled from train_list to use in validation.
    'num_val_scenes': 10,

    # -- GPU/cuda options -- #
    'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
    'num_workers': 0,  # Number of parallel processes to fetch data.
    'num_workers_val': 0,  # Number of parallel processes during validation.
    'prefetch_factor': 2,  # Batches to prefetch per worker (requires num_workers > 0).

    # 以下两阶段采样 key 没有任何代码读取（类别均衡由 tile_sampling 负责，
    # 见 configs/FLOE/_sota_base.py），仅为兼容旧 config 保留。
    'two_phase_sampling': True,
    'base_ratio': 0.6,
    'target_class_ratio': [0.2, 0.0, 0.2, 0.2, 0.2, 0.2],
    'class_bank_stride_factor': 0.5,
    'phase2_max_retries': 20,

    # -- U-Net Options -- #
    'unet_dropout': 0.0,  # Dropout probability applied after each DoubleConv block (0.0 = disabled).
    'unet_conv_filters': [16, 32, 64, 64],  # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.

    # -- Swin Transformer Options -- #
    'swin_hp': {'patch_size': 4,  # (int | tuple(int)): Patch size. Default: 4
                'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
                'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer.
                'num_heads': [3, 6, 12, 24],  # (tuple(int)): Number of attention heads in different layers.
                'window_size': 8,  # (int): Window size. Default: 8
                'mlp_ratio': 4.,  # (float): Ratio of mlp hidden dim to embedding dim. Default: 4
                'qkv_bias': True,  # (bool): If True, add a learnable bias to query, key, value. Default: True
                'qk_scale': None,  # (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
                'drop_rate': 0.,  # (float): Dropout rate. Default: 0
                'attn_drop_rate': 0.,  # (float): Attention dropout rate. Default: 0
                'drop_path_rate': 0.1,  # (float): Stochastic depth rate. Default: 0.1
                'norm_layer': nn.LayerNorm,  # (nn.Module): Normalization layer. Default: nn.LayerNorm.
                'ape': False,  # (bool): If True, add absolute position embedding to the patch embedding. Default: False
                'patch_norm': True,  # (bool): If True, add normalization after patch embedding. Default: True
                'use_checkpoint': False,  # (bool): Whether to use checkpointing to save memory. Default: False
                # (tuple(int)): Stride for the sliding inference method during validation. Default 56
                'val_stride': [56, 56],
                # (tuple(int)): Stride for the sliding inference method during testing. Default 4
                'test_stride': [4, 4],
                },

    # -- Separate_decoder options -- #
    'common_features_last_layer': 6,
    'backbone': 'unet',

    # -- Model selection -- #
    'model_selection': 'unet',  # all options: floe.functions.get_model

    # -- Data Augmentation Options -- #
    # Random_h_flip (float, [0, 1]) - Chance of doing random horizontal flip
    # Random_v_flip (float, [0, 1]) - Chance of doing random vertical flip
    # Random_rotation_prob (float, [0, 1]) - Chance of doing random rotation
    # Random_rotation (int, [0, 180]) - Maximum random degree rotation between 0 and Random_rotation
    # Random_scale_prob (float, [0, 1]) - Chance of doing random scaling
    # Random_scale (float_tuple(a, b)) - Scales image randomly to a scale between a and b. b > a.
    # Cutmix_beta (float, (0, inf)) - Parameter for random beta distribution
    # Cutmix_prob (float, [0, 1]) - Chance of doing random cutmix
    'data_augmentations': {
        'Random_h_flip': 0,
        'Random_v_flip': 0,
        'Random_rotation_prob': 0,
        'Random_rotation': 0,
        'Random_scale_prob': 0,
        'Random_scale': (1, 1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0,
    }
}
