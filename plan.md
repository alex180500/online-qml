# Online Shadow Training — Experiment and Script Plan

## 0. Article-level goal

The article should demonstrate that standard QELM readout training fails in the low-shot regime because it learns from a noisy empirical probability matrix, while Online Shadow Training learns the effective reservoir POVM and constructs the readout through the associated dual measurement frame.

The central numerical message should be:

> In the many-training-state, few-shot regime, OST gives a better statistical object to learn from than the pseudoinverse of empirical probabilities.

The main scripts should therefore answer four questions:

1. How does the error scale with the number of training states?
2. How does the error scale with the number of shots per training state?
3. How does the method scale with Hilbert-space dimension?
4. What happens when the training ensemble is local/product rather than global Haar?

Supplementary scripts should then validate the theoretical model and explain the frame geometry.

---

# 1. Naming convention for all scripts

Use the new naming scheme everywhere.

| Paper name | Code name | State frame | POVM frame | Role |
|---|---:|---:|---:|---|
| Online Shadow Training | `ost` | empirical `F_rho_hat` | empirical `F_mu_hat` | main proposed method |
| state-prior OST | `state_prior_ost` | prior `F_rho_0` | empirical `F_mu_hat` | practical prior-state variant |
| POVM-prior OST | `povm_prior_ost` | empirical `F_rho_hat` | prior `F_mu_0` | tests POVM-prior bias |
| prior OST | `prior_ost` | prior `F_rho_0` | prior `F_mu_0` | fully prior-frame baseline |
| pseudoinverse | `pinv` | none | none | standard probability-matrix baseline |
| ridge | `ridge` | none | none | regularized probability-matrix baseline |

Default method list for most scripts:

```python
methods = (
    "ost",
    "state_prior_ost",
    "povm_prior_ost",
    "prior_ost",
    "pinv",
)
````

Optional:

```python
methods += ("ridge",)
```

For main-text figures, ridge can be omitted unless it gives a useful comparison.

---

# 2. Common script structure

All experiment scripts should follow the same simple structure:

```text
scripts/
  ntrain_sweep.py
  shot_sweep.py
  dim_sweep.py
  local_training_sweep.py

  beta_fit_sweep.py
  frame_distance_sweep.py
  prediction_geometry.py
```

Each script should:

1. parse command-line arguments;
    
2. call `torch_setup`;
    
3. create an output folder under `./data`;
    
4. create or load `metadata.json`;
    
5. loop over indexed seeds `seed_0`, `seed_1`, ...;
    
6. generate/load raw data;
    
7. generate/load layers;
    
8. evaluate/load metrics;
    
9. save one `metrics_long.csv`.
    

Use simple file-exists caching:

```text
if raw file exists: reuse it
if layer file exists: reuse it
if metric file exists: reuse it
```

Do not add complicated validation yet. Metadata consistency can be checked manually or added later.

---

# 3. Shared output layout

Use the same folder style for every experiment.

For example:

```text
./data/ntrain_sweep/shots_1/
  metadata.json
  raw/
    seed_0.pt
    seed_1.pt
    ...
  layers/
    seed_0.pt
    seed_1.pt
    ...
  metrics/
    seed_0.pt
    seed_1.pt
    ...
    metrics_long.csv
```

For other scripts:

```text
./data/shot_sweep/ntrain_1000/
./data/dim_sweep/shots_1/
./data/local_training_sweep/global_target/
./data/beta_fit_sweep/
./data/frame_distance_sweep/
./data/prediction_geometry/
```

The metadata should include at least:

```json
{
  "experiment": "...",
  "dim": 2,
  "n_out": 16,
  "shots": 1,
  "nseeds": 100,
  "methods": ["ost", "state_prior_ost", "povm_prior_ost", "prior_ost", "pinv"],
  "seeds": [
    {
      "index": 0,
      "simulation_seed": 123,
      "observable_seed": 456
    }
  ]
}
```

For dimension and local experiments, metadata should also include:

```json
{
  "d_grid": [2, 3, 4, 5, 6, 8],
  "n_out_rule": "d**3",
  "n_train_rule": "100*d**2",
  "training_ensemble": "global_haar",
  "target_type": "global_projector"
}
```

---

# 4. Script 1 — `ntrain_sweep.py`

## Purpose

This is the first main numerical experiment.

It tests the central low-shot learning claim:

> At fixed number of shots per state, OST should improve as `1 / n_train`, while pseudoinverse training is unstable or inefficient in the low-shot regime.

This script should produce the main Figure 2.

## Parameters

Default local-debug values:

```text
d = 2
n_out = 16
shots = 1
train_start = 10
max_ntrain = 50000
train_step = 30
nseeds = 3
methods = ost state_prior_ost povm_prior_ost prior_ost pinv ridge
```

Article-scale values:

```text
d = 2
n_out = 16
shots ∈ {1, 10, 100}
train_start = 10
max_ntrain = 1_000_000
train_step = 100
nseeds = 100
methods = ost state_prior_ost povm_prior_ost prior_ost pinv
```

Run one shot value per invocation:

```bash
uv run python scripts/ntrain_sweep.py --shots 1
uv run python scripts/ntrain_sweep.py --shots 10
uv run python scripts/ntrain_sweep.py --shots 100
```

## Data generation

For each seed index:

1. sample global Haar pure training states:
    

```python
states = sample_dm(max_ntrain, d=d, ...)
```

2. sample a random Naimark/Stiefel POVM:
    

```python
povm = sample_povm(n_out, d=d, ...)
```

3. sample raw outcomes:
    

```python
outcomes = shots_outcome(povm, states, shots)
```

4. sample one target observable, initially a Haar random pure-state projector:
    

```python
observable = sample_dm(1, d=d, ...).T
```

## Layer generation

Use:

```python
make_layers_ntrain_grid(
    data,
    observable,
    train_grid,
    n_shots=shots,
    methods=methods,
)
```

The `train_grid` is:

```python
train_grid = logspace_int(train_start, max_ntrain, train_step)
```

## Evaluation

Use:

```python
evaluate_layer_result_haar(result, data.povm)
```

Primary metric:

```text
mse_exact_probs
```

Equivalent diagnostic metric:

```text
bias2
```

Also save:

```text
variance
```

because later finite-test-shot MSE can be reconstructed as:

```text
mse_test = bias2 + variance / N_test
```

## Plot

For each shot value:

```text
x-axis: n_train
y-axis: mse_exact_probs or bias2
scale: log-log
curves: ost, state_prior_ost, povm_prior_ost, prior_ost, pinv
aggregation: mean over seeds
shading: standard error or interquartile range
```

Recommended main figure:

```text
Panel A: N = 1
Panel B: N = 10
Panel C: N = 100
```

## Expected behavior

For `ost`:

```text
MSE should decrease approximately as 1 / n_train.
```

For `state_prior_ost`:

```text
Likely close to OST when the analytic Haar state prior is appropriate.
Potentially more stable at small n_train because it avoids empirical state-frame inversion.
```

For `povm_prior_ost` and `prior_ost`:

```text
Possible asymptotic error floor due to mismatch between finite POVM frame and the Naimark average prior.
```

For `pinv`:

```text
Worst low-shot behavior, especially for N = 1.
Should improve as N increases.
```

## Analysis to save

In addition to raw metrics, save a fitted summary:

```text
fit y(n) = c0 + c1 / n
```

For each method and shot value.

Expected:

```text
ost: c0 ≈ 0
state_prior_ost: c0 ≈ 0 or small
povm_prior_ost: c0 > 0
prior_ost: c0 > 0
pinv: may not fit cleanly at low shots
```

---

# 5. Script 2 — `shot_sweep.py`

## Purpose

This is the second main numerical experiment.

It tests how each method uses additional shots per training state.

It should show:

> OST already works in the low-shot regime, while pseudoinverse training requires many shots before its probability matrix becomes reliable.

## Parameters

Default local-debug values:

```text
d = 2
n_out = 16
n_train = 1000
shot_start = 1
shot_max = 10000
shot_step = 40
nseeds = 3
```

Article-scale values:

```text
d = 2
n_out = 16
n_train = 1000
shot_start = 1
shot_max = 100000
shot_step = 100
nseeds = 100
methods = ost state_prior_ost povm_prior_ost prior_ost pinv
```

## Data generation

For each seed index, generate:

```text
states shape: (d^2, n_train)
outcomes shape: (n_train, shot_max)
povm shape: (n_out, d^2)
```

Use one fixed observable per seed.

## Layer generation

Use:

```python
make_layers_shot_grid(
    data,
    observable,
    shot_grid,
    n_train=n_train,
    methods=methods,
)
```

The `shot_grid` is:

```python
shot_grid = logspace_int(shot_start, shot_max, shot_step)
```

This script should rely on cumulative shot statistics, not regenerate probabilities from scratch for every shot value.

## Evaluation

Use:

```python
evaluate_layer_result_haar(result, data.povm)
```

Primary metric:

```text
mse_exact_probs
```

## Plot

```text
x-axis: shots N
y-axis: mse_exact_probs
scale: log-log or semilog
curves: ost, state_prior_ost, povm_prior_ost, prior_ost, pinv
```

## Expected behavior

The theoretical form at fixed `n_train` is:

```text
MSE(N) = a + b / N
```

where:

```text
a ≈ beta0 + beta2 / n_train
b ≈ beta1 / n_train
```

Expected qualitative results:

```text
ost: good already at N = 1, then saturates
state_prior_ost: similar or sometimes better at small n_train
povm_prior_ost/prior_ost: saturate to a larger floor
pinv: improves strongly with N
```

## Analysis to save

Fit:

```text
y(N) = a + b / N
```

For each method.

Save:

```text
a
b
estimated_beta1 = b * n_train
fit_residual
```

---

# 6. Script 3 — `dim_sweep.py`

## Purpose

This is the new main-text dimension-scaling study.

It asks:

> Is the low-shot advantage of OST only a qubit effect, or does it persist as `d` increases?

This should become a main article figure because it gives the work a stronger scaling story.

## Main design choice

Use:

```text
n_out = d**3
```

This gives:

```text
alpha = n_out / d^2 = d
```

So the POVM becomes increasingly overcomplete as dimension grows.

This is useful because it reduces the risk that failure at larger `d` is merely caused by an insufficient POVM. The experiment focuses more cleanly on the training method.

## Parameters

First version:

```text
d_grid = 2 3 4 5 6 8
n_out = d**3
n_train = 100 * d**2
shots = 1
nseeds = 20
methods = ost state_prior_ost pinv
```

Second version:

```text
d_grid = 2 3 4 5 6 8
n_out = d**3
n_train = 100 * d**2
shots = 100 or 1000
nseeds = 20
methods = ost state_prior_ost pinv
```

Possible full version:

```text
shots_grid = [1, 100]
d_grid = [2, 3, 4, 5, 6, 8]
n_train_factor = 100
n_out_rule = d**3
```

## Data generation

For each dimension `d` and seed index:

```python
n_out = d**3
n_train = n_train_factor * d**2
```

Then generate:

```python
states = sample_dm(n_train, d=d, ...)
povm = sample_povm(n_out, d=d, ...)
outcomes = shots_outcome(povm, states, shots)
observable = sample_dm(1, d=d, ...).T
```

## Layer generation

Use `make_layers_ntrain_grid` with a single-point grid:

```python
train_grid = torch.tensor([n_train])
```

Then:

```python
make_layers_ntrain_grid(
    data,
    observable,
    train_grid,
    n_shots=shots,
    methods=methods,
)
```

## Evaluation

Use Haar exact-probability evaluation:

```python
evaluate_layer_result_haar(result, povm)
```

## Plot

Recommended main plot:

```text
Panel A: shots = 1
Panel B: shots = 100 or 1000

x-axis: d
y-axis: mse_exact_probs
curves: ost, state_prior_ost, pinv
scale: log-log or semilog-y
```

Alternative x-axis:

```text
d^2
```

Useful additional plot:

```text
MSE * n_train vs d
```

This checks whether the leading coefficient grows with dimension after accounting for the number of training states.

## Expected behavior

At `shots = 1`:

```text
OST should show the strongest advantage over pinv.
```

At high shots:

```text
pinv should improve, because the empirical probability matrix becomes more accurate.
```

Potentially:

```text
state_prior_ost may outperform full ost for small training budgets because the empirical state-frame inverse can be noisy when n_train is only moderately above d^2.
```

## Important interpretation

This experiment does not prove asymptotic scalability. It tests finite-dimensional scaling in the relevant simulation regime.

The careful statement should be:

> For fixed training budget proportional to the operator-space dimension and an increasingly overcomplete POVM, OST retains a low-shot advantage over probability-matrix pseudoinverse training.

---

# 7. Script 4 — `local_training_sweep.py`

## Purpose

This is the fourth main article experiment, but it should be treated carefully.

It tests:

> What happens when the training ensemble is local/product rather than global Haar?

This is important because it shows the geometric limitation of the method:

```text
OST is not magic; the training-state ensemble must be compatible with the target observable class.
```

## First version: local training for a global target

This is the cleanest boundary test.

### Parameters

```text
n_qubits_grid = 2 3 4
d = 2**n_qubits
n_out = d**3
n_train = 100 * d**2
shots = 1
nseeds = 20
target = global Haar projector
training ensembles = global_haar, product_haar
methods = ost pinv
```

Optionally include:

```text
state_prior_ost
```

but only if the correct product-Haar state prior is passed when using product-Haar states.

### Data generation

For `global_haar`:

```python
states = sample_dm(n_train, d=d, ...)
state_prior_frame = haar_state_frame(d, ...)
```

For `product_haar`:

```python
states = sample_product_dm(n_train, n_sites=n_qubits, local_dim=2, ...)
state_prior_frame = product_haar_state_frame(n_sites=n_qubits, local_dim=2, ...)
```

Use the same type of POVM for both:

```python
povm = sample_povm(n_out, d=d, ...)
```

Use a global target:

```python
observable = sample_dm(1, d=d, ...).T
```

## Layer generation

For global Haar:

```python
make_layers_ntrain_grid(..., state_prior_frame=None)
```

For product Haar:

```python
make_layers_ntrain_grid(..., state_prior_frame=product_haar_state_frame(...))
```

## Evaluation

Use the same Haar test evaluator first:

```python
evaluate_layer_result_haar(result, povm)
```

This evaluates global generalization.

## Plot

```text
x-axis: n_qubits or d
y-axis: mse_exact_probs
curves:
  global_haar + ost
  product_haar + ost
  global_haar + pinv
  product_haar + pinv
```

Recommended layout:

```text
Panel A: shots = 1
Panel B: shots = 100
```

## Expected behavior

For a global Haar target:

```text
global-Haar training should outperform product-Haar training.
```

The likely message:

> Product training states are not geometrically matched to generic global observables.

This is a good boundary/negative result.

## Second version: local training for a local target

This can be a follow-up or supplementary extension.

Use a `k`-local observable, e.g. a one-site or two-site Pauli observable.

Parameters:

```text
target = k-local observable
training ensembles = global_haar, product_haar
methods = ost state_prior_ost pinv
```

Expected behavior:

```text
product-Haar training should become much more competitive for local observables.
```

This would support the broader claim:

> The relevant training ensemble should match the target operator class.

## Library functions needed

Already added:

```python
sample_product_dm
product_haar_state_frame
```

Useful later:

```python
sample_pauli_observable(n_qubits, locality)
sample_local_observable(n_qubits, locality)
```

Do not add these until the first product-Haar/global-target experiment works.

---

# 8. Supplementary Script A — `beta_fit_sweep.py`

## Purpose

This script validates the analytical scaling model:

```text
MSE(N, n_train) = beta0 + beta1 / (N n_train) + beta2 / n_train
```

This should go in the supplementary material because the main article should focus on the method and its numerical consequences.

## Parameters

First version:

```text
d = 2
n_out = 16
shots_grid = 1 2 5 10 20 50 100 200 500 1000
train_grid = logspace_int(20, 100000, 50)
nseeds = 20
methods = ost state_prior_ost povm_prior_ost prior_ost pinv
```

Full version:

```text
nseeds = 50 or 100
```

## Data generation

For each seed:

```text
states shape: (d^2, max(train_grid))
outcomes shape: (max(train_grid), max(shots_grid))
```

## Layer generation

There are two possible strategies.

### Strategy 1: nested calls

For each shot value:

```python
make_layers_ntrain_grid(..., n_shots=shots)
```

This is simpler and consistent with `ntrain_sweep.py`.

### Strategy 2: specialized 2D layer grid

Add later only if necessary:

```python
make_layers_shot_ntrain_grid(...)
```

Do not add this now unless runtime becomes a problem.

## Evaluation

Evaluate all layers with:

```python
evaluate_layer_result_haar(...)
```

Then arrange MSE into tensors:

```text
shape: (n_shots, n_train)
```

for every method.

## Fit

For each method, fit:

```text
y = beta0 + beta1 / (N n_train) + beta2 / n_train
```

with global least squares.

Use:

```python
fit_beta_coefficients(...)
```

## Plot

Supplementary figure:

```text
Panel A: beta0 by method
Panel B: beta1 by method
Panel C: beta2 by method
Panel D: fit residual by method
```

Possible second figure:

```text
predicted MSE surface vs measured MSE surface
```

## Expected behavior

```text
ost: beta0 approximately zero or very small
state_prior_ost: beta0 approximately zero or small
povm_prior_ost: beta0 positive
prior_ost: beta0 positive
pinv: may not fit the OST beta model cleanly
```

Important caveat:

```text
The beta model is cleanest for fixed prior-frame estimators.
For empirical-frame methods, the inverse frames are data-dependent, so fitted betas should be interpreted phenomenologically.
```

---

# 9. Supplementary Script B — `frame_distance_sweep.py`

## Purpose

This script explains when the prior approximations are valid.

It should validate the frame-geometry statements:

```text
state-frame concentration needs n_train on the order of d^2
measurement-frame concentration needs n_out on the order of d^2
```

The natural metric is the relative whitened frame error, not raw Frobenius distance.

## Part 1: state-frame distance

Use:

```python
state_frame_distance_grid(...)
```

Parameters:

```text
d_grid = 2 4 8
train_grid = logspace_int(d**2, 1_000_000, 100)
nseeds = 20
```

Metrics:

```text
state_rel_op
state_rel_fro
state_lambda_min
state_lambda_max
state_condition
```

Plot:

```text
x-axis: n_train / d^2
y-axis: state_rel_op
curves: d = 2, 4, 8
```

Also plot:

```text
lambda_min and lambda_max vs n_train / d^2
```

Expected:

```text
relative error decreases approximately like d / sqrt(n_train)
```

## Part 2: measurement-frame distance

Use:

```python
measurement_frame_distance_grid(...)
```

Parameters:

```text
d_grid = 2 4 8
alpha_grid = 1 2 4 8 16 32 64
n_out = alpha * d**2
nseeds = 20
```

Metrics:

```text
measurement_rel_op
measurement_rel_fro
measurement_lambda_min
measurement_lambda_max
measurement_condition
```

Plot:

```text
x-axis: alpha = n_out / d^2
y-axis: measurement_rel_op
curves: d = 2, 4, 8
```

Expected:

```text
relative error decreases approximately like 1 / sqrt(alpha)
```

## Why this is supplementary

The main paper can state the geometric idea, while this script provides numerical evidence for when the prior-state and prior-POVM approximations become reliable.

---

# 10. Supplementary Script C — `prediction_geometry.py`

## Purpose

This script is optional.

It produces true-vs-predicted scatter plots and calibration diagnostics.

It helps show that:

```text
high correlation does not imply correct calibration
```

This is useful especially for `pinv`, `povm_prior_ost`, and `prior_ost`.

## Parameters

```text
d = 2
n_out = 16
shots = 1
n_train = 1000, 10000, or 100000
n_test = 10000
methods = ost state_prior_ost povm_prior_ost prior_ost pinv
nseeds = 5 or 10
```

## Procedure

1. Load or generate one trained layer file.
    
2. Sample test states:
    

```python
test_states = sample_dm(n_test, d=d, ...)
```

3. Compute true values:
    

```python
y_true = Tr(O rho)
```

4. Compute predicted values:
    

```python
y_pred = W p(rho)
```

5. Save:
    

```text
true
pred
slope
intercept
pearson
mse
```

Use existing helper:

```python
prediction_geometry(...)
```

## Plot

For each method:

```text
x-axis: true expectation
y-axis: predicted expectation
diagonal: y = x
```

Recommended layout:

```text
Panel A: ost
Panel B: state_prior_ost
Panel C: povm_prior_ost
Panel D: pinv
```

## Expected behavior

```text
ost: points close to diagonal
state_prior_ost: close to diagonal if state prior is appropriate
povm_prior_ost/prior_ost: tilted or biased cloud if POVM prior mismatches actual POVM
pinv: low-shot shrinkage or tilted line
```

This figure should only be included if it is visually clean.

---

# 11. Plotting scripts

Do not mix plotting heavily into the data-generation scripts.

Use separate plotting scripts or notebooks:

```text
scripts/plotting/plot_ntrain_sweep.py
scripts/plotting/plot_shot_sweep.py
scripts/plotting/plot_dim_sweep.py
scripts/plotting/plot_local_training_sweep.py
scripts/plotting/plot_beta_fit.py
scripts/plotting/plot_frame_distance.py
scripts/plotting/plot_prediction_geometry.py
```

Matplotlib should remain a dev dependency:

```bash
uv add --dev matplotlib
```

The data-generation scripts should not require matplotlib.

---

# 12. Shared helper functions to add later to the library

The scripts will stay cleaner if these are added after the first two scripts stabilize.

## 12.1 Simulation generation

Add:

```python
def make_simulation_data(
    d: int,
    n_out: int,
    n_states: int,
    shots: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    state_sampler: str = "global_haar",
    n_sites: int | None = None,
    local_dim: int = 2,
) -> SimulationData:
    ...
```

Purpose:

```text
Used by ntrain_sweep, shot_sweep, dim_sweep, local_training_sweep, beta_fit_sweep.
```

## 12.2 Artifact loaders

Add:

```python
def load_layer_result(path, device="cpu") -> LayerResult:
    ...
```

```python
def load_metric_result(path, device="cpu") -> MetricResult:
    ...
```

Purpose:

```text
Avoid reconstructing dataclasses manually in every script.
```

## 12.3 Simple load-or-make helper

Add:

```python
def load_or_make(path, make_fn, load_fn=load_pt, save_fn=save_pt):
    if Path(path).exists():
        return load_fn(path)
    obj = make_fn()
    save_fn(obj, path)
    return obj
```

Keep it simple. Do not add a complicated cache framework yet.

## 12.4 Metrics CSV writer

Add:

```python
def save_metrics_long_csv(
    metric_files: list[Path],
    csv_path: Path,
    extra_fields: dict | None = None,
) -> None:
    ...
```

Purpose:

```text
Used by all scripts.
```

The CSV should always have columns:

```text
seed_index
method
metric
value
```

plus experiment-specific columns such as:

```text
shots
n_train
d
n_out
n_qubits
training_ensemble
target_type
```

## 12.5 Fixed-budget layer helper

Optional:

```python
def make_layers_fixed_budget(
    data,
    observable,
    n_train,
    shots,
    methods,
    **kwargs,
):
    train_grid = torch.tensor([n_train], device=data.states.device)
    return make_layers_ntrain_grid(...)
```

This is mainly for `dim_sweep.py` and `local_training_sweep.py`.

---

# 13. Execution priority

Run scripts in this order.

## Step 1 — Finish and test `ntrain_sweep.py`

Local debug:

```bash
uv run python scripts/ntrain_sweep.py \
  --shots 1 \
  --max-ntrain 50000 \
  --train-step 30 \
  --nseeds 3
```

Then full:

```bash
uv run python scripts/ntrain_sweep.py --shots 1 --max-ntrain 1000000 --train-step 100 --nseeds 100
uv run python scripts/ntrain_sweep.py --shots 10 --max-ntrain 1000000 --train-step 100 --nseeds 100
uv run python scripts/ntrain_sweep.py --shots 100 --max-ntrain 1000000 --train-step 100 --nseeds 100
```

## Step 2 — Build `shot_sweep.py`

Local debug:

```bash
uv run python scripts/shot_sweep.py \
  --n-train 1000 \
  --shot-max 10000 \
  --shot-step 40 \
  --nseeds 3
```

Full:

```bash
uv run python scripts/shot_sweep.py \
  --n-train 1000 \
  --shot-max 100000 \
  --shot-step 100 \
  --nseeds 100
```

## Step 3 — Build `dim_sweep.py`

Start small:

```bash
uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 \
  --shots 1 \
  --n-train-factor 50 \
  --nseeds 5
```

Then full:

```bash
uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 5 6 8 \
  --shots 1 \
  --n-train-factor 100 \
  --nseeds 20
```

Repeat with:

```bash
--shots 100
```

or:

```bash
--shots 1000
```

## Step 4 — Build `local_training_sweep.py`

Start with:

```bash
uv run python scripts/local_training_sweep.py \
  --n-qubits-grid 2 3 \
  --shots 1 \
  --n-train-factor 50 \
  --nseeds 5
```

Then extend to:

```bash
--n-qubits-grid 2 3 4
--n-train-factor 100
--nseeds 20
```

## Step 5 — Supplementary scripts

After the main four scripts work:

```text
beta_fit_sweep.py
frame_distance_sweep.py
prediction_geometry.py
```

---

# 14. Main article figure list

## Figure 1 — Method schematic

Already planned.

Show:

```text
standard QELM:
  rho_i -> reservoir -> shots -> empirical P -> pseudoinverse -> W

OST:
  (rho_i, b_i) -> operator update -> mu_hat -> F_mu_hat^+ -> W
```

## Figure 2 — MSE vs number of training states

Generated by:

```text
ntrain_sweep.py
```

Panels:

```text
N = 1
N = 10
N = 100
```

Methods:

```text
ost
state_prior_ost
povm_prior_ost
prior_ost
pinv
```

## Figure 3 — MSE vs number of shots

Generated by:

```text
shot_sweep.py
```

Fixed:

```text
n_train = 1000
d = 2
n_out = 16
```

## Figure 4 — Dimension scaling

Generated by:

```text
dim_sweep.py
```

Panels:

```text
N = 1
N = 100 or 1000
```

Methods:

```text
ost
state_prior_ost
pinv
```

## Figure 5 — Local vs global training

Generated by:

```text
local_training_sweep.py
```

Compare:

```text
global Haar training
product Haar training
```

for:

```text
global target observable
```

Optional second panel:

```text
local target observable
```

---

# 15. Supplementary figure list

## Supplementary Figure S1 — Beta fit

Generated by:

```text
beta_fit_sweep.py
```

Show:

```text
beta0
beta1
beta2
fit residual
```

## Supplementary Figure S2 — Frame distances

Generated by:

```text
frame_distance_sweep.py
```

Show:

```text
state-frame relative error vs n_train / d^2
POVM-frame relative error vs n_out / d^2
```

## Supplementary Figure S3 — Prediction geometry

Generated by:

```text
prediction_geometry.py
```

Show:

```text
true vs predicted expectation values
slope
intercept
Pearson correlation
MSE
```

---

# 16. Final recommended storyline

The numerical section should tell this story:

1. `ntrain_sweep.py`:  
    OST improves with many low-shot training states; pseudoinverse struggles.
    
2. `shot_sweep.py`:  
    OST is already useful at one or few shots; pseudoinverse requires high shots to become competitive.
    
3. `dim_sweep.py`:  
    The advantage is not only a qubit artifact; the comparison remains meaningful as `d` increases when `n_out = d^3` and `n_train ∝ d^2`.
    
4. `local_training_sweep.py`:  
    The training ensemble matters. Product/local training may fail for generic global targets, but this suggests a path toward locality-adapted OST for local observables.
    
5. Supplementary:  
    Beta fits and frame distances explain why the curves behave this way.