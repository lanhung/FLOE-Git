# Chart-composition supervision on top of the baseline (a): polygon-level proportion
# loss plus an ordinal loss on the cumulative floe-size distribution, pixel CE only in
# pure polygons (one class >= 0.9). Needs $FLOE_CHART_DIR/<scene>.chart.h5 for every
# training scene (tools/data/build_composition.py). See floe/composition.py.
import os

_base_ = ['./sota_segformer_in1k_dlr_417.py']

train_options = {
    'composition_supervision': dict(
        enabled=True,
        chart_dir=os.environ.get('FLOE_CHART_DIR', '/root/autodl-tmp/WannianLi/FLOE/charts'),
        hard_ce='pure',
        pure_threshold=0.9,
        weight=1.0,
        w_prop=1.0,
        w_ord=1.0,
        min_bag_pixels=64,
        max_unknown=0.5,
    ),
}
