# AutoICE scenes kept in the AutoICE z-score space (tools.data.convert_ai4arctic
# --stats autoice), which is the space of the rebuilt archive. Used as the external
# test of models trained with configs/V2 (same model and inputs) and for pooling.
# Labels and valid masks equal those of floe_format, so its tile index is reused.
import os

_A4A_V2_DIR = os.path.join(os.environ.get('FLOE_A4A_V2_DIR',
                                          '/root/autodl-tmp/WannianLi/FLOE/ai4arctic/floe_format_v2'), '')

_base_ = ['./a4a_baseline.py']

train_options = {
    'path_to_train_data': _A4A_V2_DIR,
    'path_to_test_data': _A4A_V2_DIR,
}
