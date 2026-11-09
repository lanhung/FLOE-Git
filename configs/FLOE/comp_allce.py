# comp.py, but keeping the ordinary dominant-class CE on every labelled pixel
# (composition losses are added on top instead of replacing CE in mixed polygons).
_base_ = ['./comp.py']

train_options = {'composition_supervision': dict(hard_ce='all')}
