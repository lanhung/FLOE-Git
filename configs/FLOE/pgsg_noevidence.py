# Control (c'): pgsg.py without evidence. segformer_gate_evidence=False removes the
# evidence streams, the gate's evidence input and the evidence token of the
# scale-axis transformer (all tied to this one option in SegFormer.__init__).
_base_ = ['./pgsg.py']

train_options = {
    'segformer_gate_evidence': False,
}
