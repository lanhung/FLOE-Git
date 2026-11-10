# Pooled training: the v2 fold-K train scenes (configs/V2/baseline.py) plus all AutoICE
# train scenes in AutoICE z-scores (ai4arctic/floe_format_v2), validated on the v2
# fold-K val and, like every v2 run, tested on the 20 AutoICE test scenes. Shows what
# the rebuilt archive adds to AutoICE (compare with an AutoICE-only run, a4a_v2.py).
import os

_ROOT = os.environ.get('FLOE_ROOT', '/root/autodl-tmp/WannianLi/FLOE')
_POOL = os.path.join(os.environ.get('FLOE_V2_POOL_DIR', _ROOT + '/pooled_v2'), '')

_base_ = ['./baseline.py']

train_options = {
    'path_to_train_data': _POOL,
    'path_to_test_data': _POOL,
    'tile_index_path': os.environ.get('FLOE_V2_POOL_TILE_INDEX', _ROOT + '/index/v2_pooled_tile_index.npz'),
    'extra_train_lists': ['datalists/ai4arctic/train.json'],
}
