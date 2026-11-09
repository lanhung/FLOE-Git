# Resolution experiment, degraded arm: res80_noglcm.py with every continuous input
# channel replaced by its masked 2x2 block mean (160 m effective grid,
# floe.loaders.degrade_resolution_), in training and validation. Labels, mask and
# the evaluation grid stay at 80 m.
_base_ = ['./res80_noglcm.py']

train_options = {
    'input_degrade_factor': 2,
}
