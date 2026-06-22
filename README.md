# online-qml

`online-qml` is a PyTorch based package for quantum extreme learning machines (QELM) simulations. It is with online shadow training readouts.

The distribution name is `online-qml`; the import name is `online_qml`.

## Installation

### Clone and sync with uv

Clone the repository, then sync the project environment:

```bash
git clone https://github.com/alex180500/online-qml.git
cd online-qml
uv sync
```

The default `uv sync` installs:

- package dependencies;
- development tools from the `dev` dependency group;
- the CPU PyTorch build from `https://download.pytorch.org/whl/cpu`.

To install the default CPU environment without development tools:

```bash
uv sync --no-dev
```

Supported PyTorch backends for local `uv` sync are:

| Backend | Command |
| --- | --- |
| CPU | `uv sync` |
| CPU, no dev tools | `uv sync --no-dev` |
| CUDA 12.6 | `uv sync --no-group cpu --extra cu126` |
| CUDA 12.8 | `uv sync --no-group cpu --extra cu128` |
| CUDA 13.0 | `uv sync --no-group cpu --extra cu130` |

The `cpu` group is enabled by default. Disable it when selecting a CUDA extra,
otherwise `uv` will reject the sync because two Torch backends were requested.

### Install as a dependency with pip

When `online-qml` is installed with `pip`, only the package dependencies are
installed. The local `uv` dependency groups, including `dev` and `cpu`, are not
installed by downstream projects.

Install the desired PyTorch build first, then install `online-qml`:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.11"
python -m pip install online-qml
```

For CUDA, use the matching PyTorch wheel index:

| Backend | PyTorch install command |
| --- | --- |
| CPU | `python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.11"` |
| CUDA 12.6 | `python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.11"` |
| CUDA 12.8 | `python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.11"` |
| CUDA 13.0 | `python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch>=2.11"` |

Then install `online-qml`:

```bash
python -m pip install online-qml
```

Do not rely on `pip install "online-qml[cu130]"` by itself to select the CUDA
wheel index. Package extras can request `torch`, but pip package metadata cannot
carry the `uv` source-index routing from this repository.

## Current focus

- generate Haar pure states and random Naimark POVMs;
- train OST and prior-frame readout layers;
- train dense pseudoinverse/ridge baselines;
- evaluate Haar bias and variance;
- study state-frame and measurement-frame distances.

## Minimal example

```python
import torch
from online_qml.estimators import ShadowReadoutEstimator
from online_qml.quantum import (
    sample_dm,
    sample_observable,
    sample_povm,
    shots_outcome,
)

states = sample_dm(1000, d=2, dtype=torch.cdouble)
povm = sample_povm(16, d=2, dtype=torch.cdouble)
outcomes = shots_outcome(povm, states, shots=1)
obs = sample_observable(1, d=2, kind="proj", dtype=torch.cdouble)

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
