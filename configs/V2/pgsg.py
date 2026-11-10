# PGSG (configs/FLOE/pgsg.py: evidence-conditioned scale gate + scale-axis transformer)
# on the rebuilt archive, with the inputs and data settings of configs/V2/baseline.py
# (repeated here: mmcv cannot merge two bases that both set train_options).
import os

_ROOT = os.environ.get('FLOE_ROOT', '/root/autodl-tmp/WannianLi/FLOE')
_V2_DIR = os.path.join(os.environ.get('FLOE_V2_DIR', _ROOT + '/data_v2'), '')
_SPLIT = os.environ.get('FLOE_V2_SPLIT', 'datalists/v2_2023')

_base_ = ['../FLOE/pgsg.py']

train_options = {
    'path_to_train_data': _V2_DIR,
    'path_to_test_data': _V2_DIR,
    'cv_fold_dir': _SPLIT,
    'train_list_path': _SPLIT + '/fold0_train.json',
    'val_path': _SPLIT + '/fold0_val.json',
    'test_path': _SPLIT + '/fold0_val.json',
    'tile_index_path': os.environ.get('FLOE_V2_TILE_INDEX', _ROOT + '/index/v2_2023_tile_index.npz'),
    'train_variables': ['nersc_sar_primary', 'nersc_sar_secondary',
                        'sar_incidenceangle', 'global_valid_mask'],
    'num_workers': 8,
}
