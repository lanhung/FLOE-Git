# Comparison model: DeepLabV3+ (ResNet-50, ImageNet init) under the same protocol as
# the SegFormer-B2 baseline, including encoder LR scale 0.1. SMP tiles the 3-channel
# ImageNet stem across the 8 input channels.
_base_ = ['./_sota_base.py']

train_options = {
    'model_selection': 'sota_deeplabv3',
    'deeplab_encoder_weights': 'imagenet',
    'optimizer': {
        'encoder_lr_scale': 0.1,
    },
}
