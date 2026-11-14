# Evaluation-only variant of configs/V2/comp_aux.py: every tile run and every chunk read, so that
# polygon dumps (composition score) are not biased by the tile gate; see
# configs/V2/baseline_evalfull.py for the reason.
_base_ = ['../comp_aux.py']

train_options = {
    'val_tile_gate': False,
    # on-demand reading stays on: with val_tile_gate False it reads every chunk
    # (floe.loaders.prep_scene_ondemand); the xarray path cannot open the v2 files
    'num_workers_val': 1,
}
