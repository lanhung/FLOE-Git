# AutoICE benchmark: full PGSG (pgsg.py) under the same protocol.
# SAR-only inputs (HH, HV, incidence angle, valid mask + pol channel; AI4Arctic has
# no GLCM bands). Scenes come from tools.data.convert_ai4arctic (FLOE file format,
# harmonised to this archive's z-scores). val = held-out acquisition days (model
# selection only); test = the 20 official AutoICE test scenes with reference charts.
import os

_A4A_DIR = os.path.join(os.environ.get('FLOE_A4A_DIR', '/root/autodl-tmp/WannianLi/FLOE/ai4arctic/floe_format'), '')

_base_ = ['../FLOE/pgsg.py']

train_options = {
    'path_to_train_data': _A4A_DIR,
    'path_to_test_data': _A4A_DIR,
    'train_list_path': 'datalists/ai4arctic/train.json',
    'val_path': 'datalists/ai4arctic/val.json',
    'test_path': 'datalists/ai4arctic/test.json',
    'tile_index_path': os.environ.get('FLOE_A4A_TILE_INDEX',
                                      '/root/autodl-tmp/WannianLi/FLOE/ai4arctic/floe_tile_index_a4a.npz'),
    'train_variables': ['nersc_sar_primary', 'nersc_sar_secondary',
                        'sar_incidenceangle', 'global_valid_mask'],
}
