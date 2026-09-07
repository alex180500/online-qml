import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("physics")

DIST_METRIC = "rel_fro"
COND_METRIC = "condition"
STATE_DATA_DIR = "my-data/state-distances/"
POVM_DATA_DIR = "my-data/povm-distances/"
OUT_DIR = "my-tests/plots/"
state_metadata = pd.read_json(STATE_DATA_DIR + "metadata.json", typ="series")
povm_metadata = pd.read_json(POVM_DATA_DIR + "metadata.json", typ="series")
COLORS = ("#4477AA", "#EE6677", "#228833")

fig = plt.figure(figsize=(5.5, 4), dpi=300)
grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.7))

state_error_ax = fig.add_subplot(grid[0, 0])
state_condition_ax = fig.add_subplot(grid[1, 0], sharex=state_error_ax)
povm_error_ax = fig.add_subplot(grid[0, 1])
povm_condition_ax = fig.add_subplot(grid[1, 1], sharex=povm_error_ax)
s_ax = (state_error_ax, state_condition_ax)
p_ax = (povm_error_ax, povm_condition_ax)

error_legend_ax = fig.add_subplot(grid[0, 2])
condition_legend_ax = fig.add_subplot(grid[1, 2])
error_legend_ax.axis("off")
condition_legend_ax.axis("off")


def plot_frame_column(
    data_dir,
    metadata,
    frame,
    x_column,
    error_ax,
    condition_ax,
    add_labels=False,
):
    for index, d in enumerate(metadata["d_grid"]):
        color = COLORS[index]

        data = pd.read_csv(data_dir + f"metrics/{frame}/d_{d}/{DIST_METRIC}.csv")
        x = data[x_column]
        y = data[frame]

        error_ax.loglog(
            x,
            y,
            lw=1.2,
            color=color,
            label=rf"$d={d}$" if add_labels else None,
        )
        error_ax.fill_between(
            x,
            data[f"{frame}_q30"],
            data[f"{frame}_q70"],
            color=color,
            alpha=0.3,
            linewidth=0,
        )

        m, q = np.polyfit(np.log10(x), np.log10(y), 1)
        print(f"{frame} d={d}: slope={m:.2f}, intercept={q:.2f}")

        if index == 2:
            error_ax.loglog(
                x,
                x ** (-0.5),
                ls="-.",
                lw=0.8,
                color="black",
                label=(
                    r"$\gamma^{-1/2}$" + "\n" + r"$\alpha^{-1/2}$"
                    if add_labels
                    else None
                ),
            )

        condition_data = pd.read_csv(
            data_dir + f"metrics/{frame}/d_{d}/{COND_METRIC}.csv"
        )
        x_cond = condition_data[x_column]

        condition_ax.semilogx(
            x_cond,
            condition_data[frame],
            color=color,
            lw=1.2,
            label=rf"$\kappa,\, d={d}$" if add_labels else None,
        )
        condition_ax.fill_between(
            x_cond,
            condition_data[f"{frame}_q30"],
            condition_data[f"{frame}_q70"],
            color=color,
            alpha=0.3,
            linewidth=0,
        )


plot_frame_column(
    STATE_DATA_DIR,
    state_metadata,
    "state",
    "gamma",
    s_ax[0],
    s_ax[1],
    add_labels=True,
)
plot_frame_column(
    POVM_DATA_DIR,
    povm_metadata,
    "povm",
    "alpha",
    p_ax[0],
    p_ax[1],
)

s_ax[0].set_title(r"State frame $\mathcal{F}_\rho$")
p_ax[0].set_title(r"POVM frame $\mathcal{F}_\mu$")

s_ax[0].set_ylabel(r"Relative Frobenius error")
s_ax[1].set_ylabel(r"Condition number")

s_ax[1].set_xlabel(r"$\gamma = n_{\mathrm{tr}} / d^2$")
p_ax[1].set_xlabel(r"$\alpha = n_{\mathrm{out}} / d^2$")

s_ax[0].set_xlim(1, 1e4)
s_ax[1].set_xlim(1, 1e4)
p_ax[0].set_xlim(1, 64)
p_ax[1].set_xlim(1, 64)
s_ax[0].set_ylim(1e-3, 1)
s_ax[1].set_ylim(1, 10)
p_ax[1].set_ylim(1, 10)

# for ax in (s_ax[1], p_ax[1]):
    # ax.axhline(1.0, color="black", lw=0.8, ls="-.", zorder=0)

handles, labels = s_ax[0].get_legend_handles_labels()
error_legend_ax.legend(handles, labels, loc="center left")
handles, labels = s_ax[1].get_legend_handles_labels()
condition_legend_ax.legend(handles, labels, loc="center left")

plt.show()
# plt.savefig(OUT_DIR + "frame_cond.pdf", bbox_inches="tight")
# plt.savefig(OUT_DIR + "frame_cond.png", bbox_inches="tight")
