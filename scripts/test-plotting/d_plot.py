import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("physics")

METRIC = "bias2"
N = 10_000_000
RUNS = [
    # (r"my-data/d_sweep/a_4_g_200_s_100000", r"$\alpha=4$", "-", "s"),
    # (r"my-data/d_sweep/a_8_g_200_s_100000", r"$\alpha=8$", "--", "o"),
    # (r"my-data/d_sweep/a_16_g_200_s_100000", r"$\alpha=16$", ":", "^"),
    # (r"my-data/d_sweep/a_16_g_200_s_100000", r"$\gamma=200$", "-", "s"),
    # (r"my-data/d_sweep/a_16_g_20_s_100000", r"$\gamma=20$", "--", "o"),
    # (r"my-data/d_sweep/a_16_g_200_s_100000", r"$N=10^5$", "-", "s"),
    # (r"my-data/d_sweep/a_16_g_200_s_1000000", r"$N=10^6$", "--", "o"),
    # (r"my-data/d_sweep_finals/a_16_g_200_s_1000000", r"$\alpha=16, \gamma=200, N=10^6$", "-", "s"),
    (r"my-data/d_sweep_finals/a_16_g_200_s_10000000", r"$\alpha=16, \gamma=200, N=10^7$", "-", "s"),
]
METHODS = (
    ("ost", "OST", "#4477AA"),
    # ("state_prior_ost", "State-prior OST", "#EE6677"),
    ("pinv", "Pseudo-inverse", "#228833"),
)


def plot_run(ax, data_file, label, style, marker):
    data = pd.read_csv(data_file)
    x = data["d"]
    n_train = data["gamma"] * x**2
    normalization = N * n_train

    if not data["shots"].eq(N).all():
        raise ValueError(f"Expected all runs to have N={N} shots.")

    for method, label, color in METHODS:
        mse = normalization * data[method]
        ax.loglog(
            x, mse, label=label, color=color, linestyle=style, marker=marker
        )
        ax.fill_between(
            x,
            normalization * data[f"{method}_q30"],
            normalization * data[f"{method}_q70"],
            color=color,
            alpha=0.3,
            linewidth=0,
        )


fig = plt.figure(figsize=(3.75, 2.75))

for run_dir, run_label, style, marker in RUNS:
    data_file = f"{run_dir}/metrics/{METRIC}.csv"
    plot_run(fig.gca(), data_file, run_label, style=style, marker=marker)

d_theory = range(2, 11)
plt.loglog(
    d_theory,
    [d * (d + 2) * (d - 1) / (d + 1) for d in d_theory],
    color="black",
    linestyle="--",
    linewidth=1.0,
    label=r"$\frac{d(d+2)(d-1)}{d+1}$",
)

plt.xlabel(r"Input dimension $d$")
plt.ylabel(r"$N n_{\mathrm{tr}}\,\mathrm{MSE}^{(\infty)}$")
plt.legend(fontsize=9)
plt.show()
