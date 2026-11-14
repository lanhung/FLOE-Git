# Multi-task training on the rebuilt archive. Floe size stays the target; the SIC and SOD
# rasters of the same chart polygons are fitted by two more heads during training only
# (validation, model selection and every score are floe size alone, as in baseline.py).
# The v2 files keep the AutoICE class tables: SIC in tenths (0-10) and six SOD classes,
# without the merge of young into new ice that base.py applies to the original archive.
# The two auxiliary losses together weigh as much as the floe-size loss; fixed before
# the first run and not tuned.
_base_ = ['./baseline.py']

train_options = {
    'aux_charts': ['SIC', 'SOD'],
    'aux_task_weights': [0.5, 0.5],
    'n_classes': {'SOD': 6},
}
