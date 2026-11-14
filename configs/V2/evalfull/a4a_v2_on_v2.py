# The AutoICE model of configs/AI4Arctic/a4a_v2.py (trained on AutoICE in AutoICE z-scores)
# scored on scenes of the rebuilt archive, which are in the same z-scores (Table 7, row 4).
# Tile gate and on-demand reading off, as in configs/V2/baseline_evalfull.py.
import os

_V2_DIR = os.path.join(os.environ.get('FLOE_V2_DIR', '/root/autodl-tmp/WannianLi/FLOE/data_v2'), '')

_base_ = ['../../AI4Arctic/a4a_v2.py']

train_options = {
    'path_to_test_data': _V2_DIR,
    'val_tile_gate': False,
    # on-demand reading stays on: with val_tile_gate False it reads every chunk
    # (floe.loaders.prep_scene_ondemand); the xarray path cannot open the v2 files
    'num_workers_val': 1,
}
