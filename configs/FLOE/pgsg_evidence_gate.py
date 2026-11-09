# PGSG ablation (b): evidence-conditioned softmax scale gate, no scale-axis transformer.
_base_ = ['./sota_segformer_in1k_dlr_417.py']

train_options = {
    'segformer_scale_gate': True,
    'segformer_gate_mode': 'softmax',
    'segformer_gate_evidence': True,
    'segformer_scale_xattn': False,
    'segformer_dann': False,
    'segformer_pba': False,
}