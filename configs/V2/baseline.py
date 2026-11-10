# Rebuilt archive (tools/build, docs/DATASET_HANDBOOK.md section 12): SegFormer-B2 at
# 80 m in SAR geometry, inputs HH, HV, incidence angle, valid mask + HH-HV, the same
# model and protocol as res80_noglcm.py on the original archive. Values are z-scores
# in the AutoICE global statistics, so AutoICE scenes standardised the same way can be
# pooled or used as an external test. Folds: tools/build/split.py (grouped by pass,
# chart and region x ISO week); train with --fold K.
import os

_ROOT = os.environ.get('FLOE_ROOT', '/root/autodl-tmp/WannianLi/FLOE')
_V2_DIR = os.path.join(os.environ.get('FLOE_V2_DIR', _ROOT + '/data_v2'), '')
_SPLIT = os.environ.get('FLOE_V2_SPLIT', 'datalists/v2_2023')

_base_ = ['../FLOE/res80_noglcm.py']

train_options = {
    'path_to_train_data': _V2_DIR,
    'path_to_test_data': _V2_DIR,
    'cv_fold_dir': _SPLIT,
    'train_list_path': _SPLIT + '/fold0_train.json',
    'val_path': _SPLIT + '/fold0_val.json',
    'test_path': _SPLIT + '/fold0_val.json',
    'tile_index_path': os.environ.get('FLOE_V2_TILE_INDEX', _ROOT + '/index/v2_2023_tile_index.npz'),
    # zstd tiles decode faster than the original gzip ones; fewer workers leave memory
    # for the archive build running on the same machine. Keep fixed across v2 runs.
    'num_workers': 8,
}
