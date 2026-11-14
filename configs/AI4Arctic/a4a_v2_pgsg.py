# External test of the PGSG model trained on the rebuilt archive (configs/V2/pgsg.py) on
# the AutoICE scenes in the AutoICE z-score space (as in a4a_v2.py). The model must be
# built from the config it was trained with, so this file only swaps the data.
import os

_A4A_V2_DIR = os.path.join(os.environ.get('FLOE_A4A_V2_DIR',
                                          '/root/autodl-tmp/WannianLi/FLOE/ai4arctic/floe_format_v2'), '')

_base_ = ['../V2/pgsg.py']

train_options = {
    'path_to_train_data': _A4A_V2_DIR,
    'path_to_test_data': _A4A_V2_DIR,
    'train_list_path': 'datalists/ai4arctic/train.json',
    'val_path': 'datalists/ai4arctic/val.json',
    'test_path': 'datalists/ai4arctic/test.json',
    'tile_index_path': os.environ.get('FLOE_A4A_TILE_INDEX',
                                      '/root/autodl-tmp/WannianLi/FLOE/ai4arctic/floe_tile_index_a4a.npz'),
}
