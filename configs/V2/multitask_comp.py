# Multi-task heads (SIC, SOD; configs/V2/multitask.py) combined with the chart-composition
# loss as an auxiliary (configs/V2/comp_aux.py: dominant-class CE on every labelled pixel plus
# proportion and ordinal losses on the floe-size mix of the ice in each polygon). Both use more
# of the chart than its dominant class. Weights as in the two parent configs, fixed before the
# first run (paper_method/DECISIONS.md, 14 Nov 2026) and not tuned.
_base_ = ['./comp_aux.py']

train_options = {
    'aux_charts': ['SIC', 'SOD'],
    'aux_task_weights': [0.5, 0.5],
    'n_classes': {'SOD': 6},
}
