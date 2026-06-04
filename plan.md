# Online Shadow Training — Experiment and Script Plan

## 0. Article-level goal

The article should demonstrate that standard QELM readout training fails in the low-shot regime because it learns from a noisy empirical probability matrix, while Online Shadow Training learns the effective reservoir POVM and constructs the readout through the associated dual POVM frame.

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

Current library default for sweep scripts:

```python
training_methods = [
    "ost",
    "state_prior_ost",
    "povm_prior_ost",
    "prior_ost",
    "pinv",
    "ridge",
]
```

For main-text figures, ridge can be omitted unless it gives a useful comparison.

---

# 2. Common script structure

## Implementation rule for coding assistants

When helping build or edit these scripts:

```text
Do not run the experiment scripts.
Do not launch local debug runs.
Do not execute full sweeps.
Do not run plotting scripts unless explicitly asked.
```

The user will run the code on their own machine or HPC setup.
Assistant work should focus on reading the existing code, editing scripts,
keeping interfaces consistent, and doing static checks by inspection.

Scripts should be thin wrappers around the library. Reuse existing helpers such as:

```python
torch_setup
seed_run
sample_data
ntrain_layers
shot_layers
haar_metrics
save_metrics
save_json
save_pt
logspace_int
training_methods
```

Do not duplicate library behavior inside scripts. In particular:

```text
do not create script-specific CSV writers when save_metrics works
do not create complicated rule parsers for simple paper choices
do not add extra seeding machinery unless it is scientifically needed
do not add cache frameworks or validation frameworks yet
```

Prefer explicit paper choices in the script, for example:

```python
n_out = d**3
n_train = 100_000
```

over parsing generic rules such as `"alpha*d**2"`.

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
    
3. create an output folder under the chosen data folder;
    
4. overwrite `metadata.json` with `save_json`;
    
5. loop over indexed seeds `seed_0`, `seed_1`, ...;
    
6. generate the simulation data;
    
7. generate layers through library helpers;
    
8. evaluate metrics through library helpers;
    
9. save summary CSVs with `save_metrics`.
    

For now, prefer recomputation over reuse. Add caching only after an explicit
request. Do not add complicated validation yet. Metadata consistency can be
checked manually or added later.

---

# 3. Shared output layout

Use the same folder style for every experiment.

For example:

```text
./data/ntrain_sweep/shots_1/
  metadata.json
  seed_0/
    layers.pt
    metrics.pt
  seed_1/
    layers.pt
    metrics.pt
  ...
  metrics/
    bias2.csv
    variance.csv
```

For other scripts:

```text
./data/shot_sweep/ntrain_1000/
./data/dim_sweep/ntrain_100000/shots_1/
./data/local_training_sweep/global_target/
./data/beta_fit_sweep/
./data/frame_distance_sweep/
./data/prediction_geometry/
```

Metadata should be run-level metadata only, written once with `save_json`.
The current seed contract is simple:

```text
metadata["seeds"] is a list of integer RNG seeds.
seed_i uses metadata["seeds"][i].
seed_run(out_dir, i, seed) creates seed_i/ and seeds Torch.
```

Do not store separate `simulation_seed` and `observable_seed` objects unless a
new experiment truly needs multiple independent RNG streams.

For `ntrain_sweep.py`, metadata should look like:

```json
{
  "script": "ntrain_sweep.py",
  "dim": 2,
  "n_out": 16,
  "shots": 1,
  "train_max": 1000000,
  "train_start": 1,
  "train_step": 100,
  "nseeds": 3,
  "methods": ["ost", "state_prior_ost", "povm_prior_ost", "prior_ost", "pinv", "ridge"],
  "precision": "float64",
  "seeds": [123, 456, 789]
}
```

For `shot_sweep.py`, metadata should look like:

```json
{
  "script": "shot_sweep.py",
  "dim": 2,
  "n_out": 16,
  "n_train": 1000,
  "shot_start": 1,
  "shot_max": 100000,
  "shot_step": 100,
  "nseeds": 3,
  "methods": ["ost", "state_prior_ost", "povm_prior_ost", "prior_ost", "pinv", "ridge"],
  "precision": "float64",
  "seeds": [123, 456, 789]
}
```

For `dim_sweep.py`, metadata should look like:

```json
{
  "script": "dim_sweep.py",
  "experiment": "dim_sweep",
  "shots": 1,
  "d_grid": [2, 3, 4, 5, 6, 8],
  "n_out_grid": [8, 27, 64, 125, 216, 512],
  "n_train": 100000,
  "nseeds": 3,
  "methods": ["ost", "state_prior_ost", "povm_prior_ost", "prior_ost", "pinv", "ridge"],
  "precision": "float64",
  "seeds": [123, 456, 789]
}
```

For local experiments, add experiment-specific fields such as:

```json
{
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

Use the library helper:

```python
data = sample_data(
    train_max,
    d=dim,
    n_out=n_out,
    shots=shots,
    seed=seed,
    device=device,
    dtype=cdtype,
)
```

Then sample one target observable, initially a Haar random pure-state projector:
    

```python
observable = sample_dm(1, d=dim, device=device, dtype=cdtype).T
```

## Layer generation

Use:

```python
result = ntrain_layers(
    data,
    observable,
    train_grid=train_grid,
    n_shots=shots,
    methods=methods,
    pinv_tol=pinv_tol,
    ridge_alpha=ridge_alpha,
    dtype=rdtype,
)
```

The `train_grid` is:

```python
train_grid = logspace_int(train_start, max_ntrain, train_step)
```

## Evaluation

Use:

```python
metrics = haar_metrics(result, data.povm)
```

Primary metric:

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
y-axis: bias2
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
result = shot_layers(
    data,
    observable,
    shot_grid=shot_grid,
    n_train=n_train,
    methods=methods,
    pinv_tol=pinv_tol,
    ridge_alpha=ridge_alpha,
    dtype=rdtype,
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
metrics = haar_metrics(result, data.povm)
```

Primary metric:

```text
bias2
```

## Plot

```text
x-axis: shots N
y-axis: bias2
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

Current version:

```text
d_grid = 2 3 4 5 6 8
n_out = d**3
n_train = 100000
shots = 1
nseeds = 20
methods = training_methods
```

Repeat separate invocations for:

```text
shots = 1 10 100 1000
```

Keep this simple for now:

```text
do not parse n_out rules
do not use n_train_factor
do not reseed dimensions separately
```

## Data generation

For each dimension `d` and seed index:

```python
n_out = d**3
n_train = args.ntrain
```

Use:

```python
data = sample_data(
    n_train,
    d=d,
    n_out=n_out,
    shots=shots,
    seed=seed,
    device=device,
    dtype=cdtype,
)
observable = sample_dm(1, d=d, device=device, dtype=cdtype).T
```

## Layer generation

Use `ntrain_layers` with a single-point training grid:

```python
train_grid = torch.tensor([n_train])
```

Then:

```python
result = ntrain_layers(
    data,
    observable,
    train_grid=train_grid,
    n_shots=shots,
    methods=methods,
    pinv_tol=pinv_tol,
    ridge_alpha=ridge_alpha,
    dtype=rdtype,
)
```

Save each dimension layer as:

```text
seed_i/d_<d>_layers.pt
```

Collect the scalar metric for each dimension into one per-seed `MetricResult`:

```text
seed_i/metrics.pt
```

Use `stack_metric_results(...)` with coordinate metadata:

```python
seed_metric_result = stack_metric_results(
    metric_results,
    grid_name="d",
    grid_values=d_grid,
    extra_coords={
        "n_out": n_out_grid,
        "n_train": ntrain,
        "shots": shots,
    },
)
```

Then `save_metrics(..., grid_column="d")` writes plot-ready CSVs using
`MetricResult.coords`.

## Evaluation

Use Haar exact-probability evaluation:

```python
metrics = haar_metrics(result, data.povm)
```

## Plot

Recommended main plot:

```text
Panel A: shots = 1
Panel B: shots = 10
Panel C: shots = 100
Panel D: shots = 1000

x-axis: d
y-axis: bias2
curves: methods in training_methods, or a selected subset for the figure
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
ntrain_layers(..., state_prior_frame=None)
```

For product Haar:

```python
ntrain_layers(..., state_prior_frame=product_haar_state_frame(...))
```

## Evaluation

Use the same Haar test evaluator first:

```python
haar_metrics(result, povm)
```

This evaluates global generalization.

## Plot

```text
x-axis: n_qubits or d
y-axis: bias2
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
ntrain_layers(..., n_shots=shots)
```

This is simpler and consistent with `ntrain_sweep.py`.

### Strategy 2: specialized 2D layer grid

Add later only if necessary:

```python
shot_ntrain_layers(...)
```

Do not add this now unless runtime becomes a problem.

## Evaluation

Evaluate all layers with:

```python
haar_metrics(...)
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
POVM-frame concentration needs n_out on the order of d^2
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
gamma_grid = logspace_int(1, 10_000, 30)
n_train = gamma * d**2
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
x-axis: gamma = n_train / d^2
y-axis: state_rel_op
curves: d = 2, 4, 8
```

Also plot:

```text
lambda_min and lambda_max vs gamma = n_train / d^2
```

Expected:

```text
relative error decreases approximately like d / sqrt(n_train)
```

## Part 2: POVM-frame distance

Use:

```python
povm_frame_distance_grid(...)
```

Parameters:

```text
d_grid = 2 4 8
alpha_grid = logspace_int(1, 64, 30)
n_out = alpha * d**2
nseeds = 20
```

Metrics:

```text
povm_rel_op
povm_rel_fro
povm_lambda_min
povm_lambda_max
povm_condition
```

Plot:

```text
x-axis: alpha = n_out / d^2
y-axis: povm_rel_op
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

# 12. Shared library helpers

The scripts should stay cleaner by reusing shared library helpers.
Before adding any new helper, first check whether the current library already
provides the needed operation.

Currently useful top-level imports from `online_qml` include:

```python
training_methods
torch_setup
random_seed
seed_run
timed
logspace_int
save_pt
load_pt
save_json
save_metrics
sample_dm
sample_product_dm
product_haar_state_frame
sample_data
ntrain_layers
shot_layers
haar_metrics
stack_metric_results
MetricResult
```

The current default method list is:

```python
training_methods = [
    "ost",
    "state_prior_ost",
    "povm_prior_ost",
    "prior_ost",
    "pinv",
    "ridge",
]
```

## 12.1 Simulation generation

Already available:

```python
sample_data
```

Purpose:

```text
Used by ntrain_sweep, shot_sweep, dim_sweep, local_training_sweep, beta_fit_sweep.
```

## 12.2 Artifact loaders

Only add typed artifact loaders if scripts start reconstructing dataclasses
manually in several places. Until then, use:

```python
save_pt
load_pt
```

and keep save/load logic simple.

## 12.3 Cache helpers

Do not add a cache framework yet.
The current preference is to regenerate data and overwrite outputs unless the
user explicitly asks for file reuse.

## 12.4 Metrics CSV writer

Already available:

```python
save_metrics
```

Use this for n-train, shot, and dimension sweeps. Do not add custom pandas
summary writers in individual scripts unless `save_metrics` truly cannot
represent the output.

`MetricResult` stores named coordinates in `coords`. `save_metrics` should read
the sweep axis from `coords[grid_column]` when available, and include other
scalar or same-length coordinates such as `shots`, `n_train`, and `n_out` in the
CSV.

Current plot-friendly output is one CSV per metric, for example:

```text
bias2.csv
variance.csv
```

with columns like:

```text
n_train,shots,ost,ost_q30,ost_q70,pinv,pinv_q30,pinv_q70,...
```

## 12.5 Fixed-budget layer helper

Optional later only if repeated fixed-budget code becomes annoying:

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
    return ntrain_layers(...)
```

This is mainly for `dim_sweep.py` and `local_training_sweep.py`.

---

# 13. Build priority and user-run commands

Build scripts in this order.

Important:

```text
The commands below are for the user to run.
Coding assistants should not run these commands unless explicitly asked.
```

## Step 1 — Finish and test `ntrain_sweep.py`

User local debug:

```bash
uv run python scripts/ntrain_sweep.py \
  --shots 1 \
  --max-ntrain 50000 \
  --train-step 30 \
  --nseeds 3
```

User full runs:

```bash
uv run python scripts/ntrain_sweep.py --shots 1 --max-ntrain 1000000 --train-step 100 --nseeds 100
uv run python scripts/ntrain_sweep.py --shots 10 --max-ntrain 1000000 --train-step 100 --nseeds 100
uv run python scripts/ntrain_sweep.py --shots 100 --max-ntrain 1000000 --train-step 100 --nseeds 100
```

## Step 2 — Build `shot_sweep.py`

User local debug:

```bash
uv run python scripts/shot_sweep.py \
  --n-train 1000 \
  --shot-max 10000 \
  --shot-step 40 \
  --nseeds 3
```

User full run:

```bash
uv run python scripts/shot_sweep.py \
  --n-train 1000 \
  --shot-max 100000 \
  --shot-step 100 \
  --nseeds 100
```

## Step 3 — Build `dim_sweep.py`

This script should use simple hard-coded paper rules:

```python
n_out = d**3
n_train = 100_000
```

Do not parse dimension rules such as `"alpha*d**2"` unless that becomes a
real need later.

User small run:

```bash
uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 \
  --shots 1 \
  --ntrain 100000 \
  --nseeds 5
```

User full runs:

```bash
uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 5 6 8 \
  --shots 1 \
  --ntrain 100000 \
  --nseeds 20

uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 5 6 8 \
  --shots 10 \
  --ntrain 100000 \
  --nseeds 20

uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 5 6 8 \
  --shots 100 \
  --ntrain 100000 \
  --nseeds 20

uv run python scripts/dim_sweep.py \
  --d-grid 2 3 4 5 6 8 \
  --shots 1000 \
  --ntrain 100000 \
  --nseeds 20
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
