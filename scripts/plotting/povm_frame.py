import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("physics")

DIST_METRIC = "rel_fro"
POVM_DATA_DIR = "my-data/povm-distances/"
OUT_DIR = "my-tests/plots/"
povm_metadata = pd.read_json(POVM_DATA_DIR + "metadata.json", typ="series")
COLORS = ("#4477AA", "#EE6677", "#228833")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=300)

for index, d in enumerate(povm_metadata["d_grid"]):
    color = COLORS[index]

    # --- ax1: relative Frobenius distance ---

    data_file = POVM_DATA_DIR + f"metrics/povm/d_{d}/{DIST_METRIC}.csv"
    data = pd.read_csv(data_file)

    x = data["alpha"]
    y = data["povm"]

    ax1.loglog(x, y, lw=1.2, label=rf"$d={d}$", color=color)
    ax1.fill_between(
        x,
        data["povm_q30"],
        data["povm_q70"],
        color=color,
        alpha=0.3,
        linewidth=0,
    )

    m, q = np.polyfit(np.log10(x), np.log10(y), 1)
    print(f"d={d}: slope={m:.2f}, intercept={q:.2f}")

    if index == 2:
        ax1.loglog(
            x, x ** (-0.5), ls="-.", lw=0.8, color="black", label=r"$\alpha^{-1/2}$"
        )

    # --- ax2: eigenvalue convergence ---

    lambda_min_data = pd.read_csv(
        POVM_DATA_DIR + f"metrics/povm/d_{d}/lambda_min.csv"
    )
    lambda_max_data = pd.read_csv(
        POVM_DATA_DIR + f"metrics/povm/d_{d}/lambda_max.csv"
    )
    x_lam = lambda_min_data["alpha"]
    lam_min = lambda_min_data["povm"]
    lam_max = lambda_max_data["povm"]

    ax2.semilogx(
        x_lam,
        lam_min,
        color=color,
        lw=1.2,
        label=rf"$\lambda_{{\min}},\, d={d}$",
    )
    ax2.fill_between(
        x_lam,
        lambda_min_data["povm_q30"],
        lambda_min_data["povm_q70"],
        color=color,
        alpha=0.3,
        linewidth=0,
    )

    ax2.semilogx(
        x_lam,
        lam_max,
        lw=1.2,
        color=color,
        ls="--",
        label=rf"$\lambda_{{\max}},\, d={d}$",
    )
    ax2.fill_between(
        x_lam,
        lambda_max_data["povm_q30"],
        lambda_max_data["povm_q70"],
        color=color,
        alpha=0.3,
        linewidth=0,
    )

ax1.set_title(r"Error of $\mathcal{F}_\mu$")
ax1.set_xlabel(r"$\alpha = n_{\mathrm{out}} / d^2$")
ax1.set_ylabel(r"Relative Frobenius error")
ax1.set_xlim(povm_metadata["alpha_start"], povm_metadata["alpha_max"])
ax1.legend(fontsize=7)

ax2.axhline(1.0, color="black", lw=0.8, ls="-.", zorder=0)
ax2.set_title(r"Spectrum of $\mathcal{F}_\mu$")
ax2.set_xlabel(r"$\alpha = n_{\mathrm{out}} / d^2$")
ax2.set_ylabel(r"Eigenvalues")
ax2.set_xlim(povm_metadata["alpha_start"], povm_metadata["alpha_max"])
ax2.legend(fontsize=7)

plt.tight_layout()

# plt.show()
plt.savefig(OUT_DIR + "povm_frame.pdf", bbox_inches="tight")
# plt.savefig(OUT_DIR + "povm_frame.png", bbox_inches="tight")
