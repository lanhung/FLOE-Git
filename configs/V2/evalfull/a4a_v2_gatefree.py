# Evaluation-only variant of configs/AI4Arctic/a4a_v2.py for ensembles on the AutoICE test scenes:
# tile gate off, so that the logits of several models are averaged over every tile and the
# on-demand reader reads every chunk (see configs/V2/baseline_evalfull.py).
_base_ = ['../../AI4Arctic/a4a_v2.py']

train_options = {
    'val_tile_gate': False,
    'num_workers_val': 1,
}
