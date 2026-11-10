# Composition as an auxiliary loss: the baseline's dominant-class CE on every labelled
# pixel, plus the proportion and ordinal losses on the floe-size mix of the ice in each
# polygon. The open-water share (1 - CT) is not fitted (ice_only), because comp.py
# showed that fitting it puts water inside pack ice.
_base_ = ['./comp.py']

train_options = {'composition_supervision': dict(hard_ce='all', ice_only=True)}
