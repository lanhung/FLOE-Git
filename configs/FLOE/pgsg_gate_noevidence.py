# Control (b'): pgsg_evidence_gate.py without evidence. The gate sees only the
# 4x256 = 1024-channel feature concat (no variogram / polarimetric streams).
_base_ = ['./pgsg_evidence_gate.py']

train_options = {
    'segformer_gate_evidence': False,
}
