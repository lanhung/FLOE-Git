# SegFormer-B2，MiT-B2 encoder 用 ImageNet-1k 预训练，统一 LR。
#
# encoder 经 smp.FPN(encoder_name='mit_b2') 构建，segformer_encoder_weights 透传给
# encoder_weights（'imagenet' 是 SMP 为 mit_b0~b5 提供的唯一预训练；首层 3 通道 patch
# embedding 由 SMP 平铺到 8 通道）。all-MLP head（~3M 参数）从零训练。
# 首次启动需联网下载 mit_b2 权重（~100MB）。
#
# 用法：
# python -m tools.train configs/FLOE/sota_segformer_in1k_417.py --wandb-project MyDS-FLOE \
#     --wandb-name floe_sota_segformer_in1k_417 --work-dir work_dir/floe_sota_segformer_in1k_417

_base_ = ['./_sota_base.py']

train_options = {
    'model_selection': 'sota_segformer',
    'segformer_encoder_weights': 'imagenet',
}
