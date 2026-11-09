# Pooled training: this archive's fold-K train scenes plus all AutoICE train scenes,
# validated on this archive's fold-K val (--fold K). Same model and inputs as
# res80_noglcm.py, which is the matching single-archive reference. AutoICE adds big
# and vast floe under DMI charts, which this archive only has in Canadian scenes.
import os

_ROOT = os.environ.get('FLOE_ROOT', '/root/autodl-tmp/WannianLi/FLOE')
_POOL_DIR = os.path.join(os.environ.get('FLOE_POOL_DIR', _ROOT + '/pooled'), '')

_base_ = ['../FLOE/res80_noglcm.py']

train_options = {
    'path_to_train_data': _POOL_DIR,
    'path_to_test_data': _POOL_DIR,
    'tile_index_path': os.environ.get('FLOE_POOL_TILE_INDEX', _ROOT + '/pooled_tile_index.npz'),
    'extra_train_lists': ['datalists/ai4arctic/train.json'],
}
