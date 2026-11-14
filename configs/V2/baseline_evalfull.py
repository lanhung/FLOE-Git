# Evaluation-only variant of configs/V2/baseline.py (same model and inputs; load its
# checkpoints with this config). Every tile of a scene is run and every chunk read:
# with the tile gate (val_tile_gate) tiles whose dominant-class labels are all masked are
# not run and predict class 0, which is harmless for the confusion matrix but makes the
# masked (mixed) chart polygons look like water in the polygon dumps of
# `tools.evaluate --dump-polygons`, i.e. in the composition score. Use this config
# whenever polygons are dumped (tools/system/eval_bench.sh). Confusion matrices agree with
# the gated path up to near-tie argmax flips (floe.functions.fast_tiled_val_predict).
_base_ = ['./baseline.py']

train_options = {
    'val_tile_gate': False,
    # on-demand reading stays on: with val_tile_gate False it reads every chunk
    # (floe.loaders.prep_scene_ondemand); the xarray path cannot open the v2 files
    # whole scenes are decoded without on-demand reading: one worker keeps the memory
    # peak of up to four concurrent evaluations within the container limit
    'num_workers_val': 1,
}
