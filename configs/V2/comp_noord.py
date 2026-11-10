# comp.py without the ordinal (cumulative distribution) term: proportion loss only.
_base_ = ['./comp.py']

train_options = {'composition_supervision': dict(w_ord=0.0)}
