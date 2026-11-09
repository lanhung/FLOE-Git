# PGSG (c): evidence-conditioned scale gate + scale-axis transformer.
_base_ = ['./pgsg_evidence_gate.py']

train_options = {
    'segformer_scale_xattn': True,
}