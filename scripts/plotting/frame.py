import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("physics")

DIST_METRIC = "rel_fro"
STATE_DATA_DIR = "my-data/state-distances/"
POVM_DATA_DIR = "my-data/povm-distances/"
OUT_DIR = "my-tests/plots/"
state_metadata = pd.read_json(STATE_DATA_DIR + "metadata.json", typ="series")
povm_metadata = pd.read_json(POVM_DATA_DIR + "metadata.json", typ="series")
COLORS = ("#4477AA", "#EE6677", "#228833")

fig = plt.figure(figsize=(5.5, 4), dpi=300)
grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.7))

state_error_ax = fig.add_subplot(grid[0, 0])
state_spectrum_ax = fig.add_subplot(grid[1, 0], sharex=state_error_ax)
povm_error_ax = fig.add_subplot(grid[0, 1])
povm_spectrum_ax = fig.add_subplot(grid[1, 1], sharex=povm_error_ax)
s_ax = (state_error_ax, state_spectrum_ax)
p_ax = (povm_error_ax, povm_spectrum_ax)

error_legend_ax = fig.add_subplot(grid[0, 2])
spectrum_legend_ax = fig.add_subplot(grid[1, 2])
error_legend_ax.axis("off")
spectrum_legend_ax.axis("off")


def plot_frame_column(
    data_dir,
    metadata,
    frame,
    x_column,
    error_ax,
    spectrum_ax,
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

        lambda_min_data = pd.read_csv(
            data_dir + f"metrics/{frame}/d_{d}/lambda_min.csv"
        )
        lambda_max_data = pd.read_csv(
            data_dir + f"metrics/{frame}/d_{d}/lambda_max.csv"
        )
        x_lam = lambda_min_data[x_column]

        spectrum_ax.semilogx(
            x_lam,
            lambda_min_data[frame],
            color=color,
            lw=1.2,
            label=rf"$\lambda_{{\min}},\, d={d}$" if add_labels else None,
        )
        spectrum_ax.fill_between(
            x_lam,
            lambda_min_data[f"{frame}_q30"],
            lambda_min_data[f"{frame}_q70"],
            color=color,
            alpha=0.3,
            linewidth=0,
        )

        spectrum_ax.semilogx(
            x_lam,
            lambda_max_data[frame],
            lw=1.2,
            color=color,
            ls="--",
            label=rf"$\lambda_{{\max}},\, d={d}$" if add_labels else None,
        )
        spectrum_ax.fill_between(
            x_lam,
            lambda_max_data[f"{frame}_q30"],
            lambda_max_data[f"{frame}_q70"],
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
s_ax[1].set_ylabel(r"Eigenvalues")

s_ax[1].set_xlabel(r"$\gamma = n_{\mathrm{tr}} / d^2$")
p_ax[1].set_xlabel(r"$\alpha = n_{\mathrm{out}} / d^2$")

s_ax[0].set_xlim(1, 1e4)
s_ax[1].set_xlim(1, 1e4)
p_ax[0].set_xlim(1, 64)
p_ax[1].set_xlim(1, 64)

for ax in (s_ax[1], p_ax[1]):
    ax.axhline(1.0, color="black", lw=0.8, ls="-.", zorder=0)

handles, labels = s_ax[0].get_legend_handles_labels()
error_legend_ax.legend(handles, labels, loc="center left")
handles, labels = s_ax[1].get_legend_handles_labels()
spectrum_legend_ax.legend(handles, labels, loc="center left")

# plt.show()
plt.savefig(OUT_DIR + "frame.pdf", bbox_inches="tight")
# plt.savefig(OUT_DIR + "frame.png", bbox_inches="tight")
