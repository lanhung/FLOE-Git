# Rebuilt archive + chart-composition supervision (floe/composition.py): polygon-level
# proportion loss and ordinal loss on the cumulative floe-size distribution, pixel CE
# only inside pure polygons (one class >= 0.9 of the ice). Sidecars come from
# tools/build/make_scene.py ($FLOE_V2_CHART_DIR/<scene>.chart.h5).
import os

_ROOT = os.environ.get('FLOE_ROOT', '/root/autodl-tmp/WannianLi/FLOE')

_base_ = ['./baseline.py']

train_options = {
    'composition_supervision': dict(
        enabled=True,
        chart_dir=os.environ.get('FLOE_V2_CHART_DIR', _ROOT + '/charts_v2'),
        hard_ce='pure',
        pure_threshold=0.9,
        weight=1.0,
        w_prop=1.0,
        w_ord=1.0,
        min_bag_pixels=64,
        max_unknown=0.5,
    ),
}
