# online-qml

`online-qml` is a PyTorch based package for quantum extreme learning machines (QELM) simulations. It is with online shadow training readouts.

The distribution name is `online-qml`; the import name is `online_qml`.

## Current focus

- generate Haar pure states and random Naimark POVMs;
- train OST and prior-frame readout layers;
- train dense pseudoinverse/ridge baselines;
- evaluate Haar bias and variance;
- study state-frame and measurement-frame distances.

## Minimal example

```python
import torch
from online_qml.quantum import sample_dm, sample_povm, shots_outcome
from online_qml.estimators import ShadowReadoutEstimator

states = sample_dm(1000, d=2, dtype=torch.cdouble)
povm = sample_povm(16, d=2, dtype=torch.cdouble)
outcomes = shots_outcome(povm, states, shots=1)
obs = sample_dm(1, d=2, dtype=torch.cdouble).T

est = ShadowReadoutEstimator(
    n_out=16,
    d=2,
    dtype=torch.float64,
    methods=("ost", "state_prior_ost"),
)
est.update_single_shot(outcomes[:, 0], states)
layers = est.layers(obs)
W_ost = layers["ost"]
W_state_prior = layers["state_prior_ost"]
```
