import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use(["science", "no-latex"])
SHOTS = (1, 10, 100, 1000)
METRIC = "bias2"
data_dir = "my-data/ntrain_sweep"
out_dir = "my-tests/plots/"

METHODS = (
    ("ost", "OST", "#4477AA", "-"),
    ("state_prior_ost", "State-prior OST", "#EE6677", "-"),
    ("povm_prior_ost", "POVM-prior OST", "#AA3377", ":"),
    ("prior_ost", "Prior OST", "#CCBB44", ":"),
    ("ridge", "Ridge regression", "#66CCEE", "--"),
    ("pinv", "Pseudo-inverse", "#228833", "--"),
)


def plot_panel(ax, data, title):
    x = data["n_train"]
    for method, label, color, ls in METHODS:
        ax.loglog(x, data[method], lw=1.8, label=label, color=color, linestyle=ls)
        ax.fill_between(
            x,
            data[f"{method}_q30"],
            data[f"{method}_q70"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax.set_title(title)
    ax.set_xlim(1, 1e5)
    ax.set_ylim(1e-8, 1e1)
    ax.grid(True, which="both", alpha=0.25)


datasets = {
    shots: pd.read_csv(f"{data_dir}/shots_{shots}/metrics/{METRIC}.csv")
    for shots in SHOTS
}

fig = plt.figure(figsize=(6, 4), dpi=300)
grid = fig.add_gridspec(2, 1, height_ratios=(1.3, 1))
top_grid = grid[0].subgridspec(1, 3, width_ratios=(1, 1, 0.7))
bottom_grid = grid[1].subgridspec(1, 3)
axes = (
    fig.add_subplot(top_grid[0, :2]),
    fig.add_subplot(bottom_grid[0, 0]),
    fig.add_subplot(bottom_grid[0, 1]),
    fig.add_subplot(bottom_grid[0, 2]),
)
legend_ax = fig.add_subplot(top_grid[0, 2])
legend_ax.axis("off")
titles = (
    "N = 1",
    "N = 10",
    "N = 100",
    "N = 1000",
)

for ax, shots, title in zip(axes, SHOTS, titles, strict=True):
    plot_panel(ax, datasets[shots], title)
for ax in axes[2:]:
    ax.set_ylabel("")
    ax.tick_params(axis="y", left=False, labelleft=False)
axes[0].set_ylim(1e-5, 1)

handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc="right", fontsize=9)

fig.supylabel("Bias$^2$")
fig.supxlabel("Number of Training States")
fig.tight_layout(pad=0.5)
# plt.show()
plt.savefig(out_dir + f"train_grid.pdf", bbox_inches="tight")
