# Resolution experiment, reference arm: SegFormer-B2 baseline at the native 80 m grid
# without the three GLCM bands (computed upstream from the 80 m image, they would
# leak sub-160 m texture into the degraded arm). Inputs: HH, HV, incidence angle,
# valid mask + pol channel = 5 channels, the same set AI4Arctic provides.
_base_ = ['./sota_segformer_in1k_dlr_417.py']

train_options = {
    'train_variables': ['nersc_sar_primary', 'nersc_sar_secondary',
                        'sar_incidenceangle', 'global_valid_mask'],
}
