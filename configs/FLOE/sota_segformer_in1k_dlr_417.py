# SegFormer-B2 基线：MiT ImageNet 预训练 + 判别式 LR。
#
# 与 sota_segformer_in1k_417 唯一差异是 optimizer.encoder_lr_scale=0.1（encoder 1e-5 /
# head 1e-4，同 mmseg 官方 SegFormer 的 backbone 1x / head 10x）。动机：统一 LR 下随机初始化
# head 的早期梯度会反传损伤预训练特征。启动日志应出现
#   [Optimizer] encoder_lr_scale=0.1: encoder ~24.2M params lr=1.00e-05 | rest ...
# --resume-from 必须用与原 run 相同的 encoder_lr_scale（optimizer state 与参数分组绑定）；
# CosineAnnealing 的 eta_min 是绝对值、两组共用，末期两组 LR 收敛到同一下限。
#
# 用法：
# python -m tools.train configs/FLOE/sota_segformer_in1k_dlr_417.py --wandb-project MyDS-FLOE \
#     --wandb-name floe_sota_segformer_in1k_dlr_417 --work-dir work_dir/floe_sota_segformer_in1k_dlr_417

_base_ = ['./sota_segformer_in1k_417.py']

train_options = {
    # mmcv Config 对 dict 递归合并：只写差异键，type/lr/b1/b2/weight_decay 继承 base。
    'optimizer': {
        'encoder_lr_scale': 0.1,
    },
}
