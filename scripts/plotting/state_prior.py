import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("physics")

STATE_DATA_DIR = "my-data/sp_gamma500/"
OUT_DIR = "my-tests/plots/"
SHOTS = 10000
METRIC = "bias2"
DIMS = (2, 4, 8)

COLORS = ("#4477AA", "#EE6677", "#228833")

fig, ax = plt.subplots(figsize=(3.5, 3), dpi=300)

for index, d in enumerate(DIMS):
    data_file = f"{STATE_DATA_DIR}d{d}/shots_{SHOTS}/metrics/{METRIC}.csv"
    data = pd.read_csv(data_file)

    color = COLORS[index]
    gamma = data["n_train"] / d**2

    delta = data["state_prior_ost"] - data["ost"]
    delta_q30 = data["state_prior_ost_q30"] - data["ost_q70"]
    delta_q70 = data["state_prior_ost_q70"] - data["ost_q30"]

    ax.loglog(gamma, delta, lw=1.8, label=rf"$d={d}$", color=color)
    ax.fill_between(
        gamma,
        delta_q30,
        delta_q70,
        color=color,
        alpha=0.15,
        linewidth=0,
    )

    theory_delta = 2 * (d - 1) / (d**3 * gamma)
    ax.loglog(
        gamma,
        theory_delta,
        color=color,
        lw=1.0,
        ls="--",
        alpha=0.6
    )

    mask = gamma >= 10
    m, q = np.polyfit(np.log10(gamma[mask]), np.log10(delta[mask]), 1)
    print(f"d={d}: slope={m}, intercept={q}")


# ref_gamma = np.array([20.0, 1000.0])
# raw_ref = 0.03 * (ref_gamma / ref_gamma[0]) ** (-1)
# scaled_ref = 0.1 * (ref_gamma / ref_gamma[0]) ** (-1)

# ax.loglog(ref_gamma, raw_ref, color="black", lw=1.0, ls="--", alpha=0.6)
# ax.text(28, 0.018, r"$\gamma^{-1}$", fontsize=8)

ax.set_title(r"State-prior excess")
ax.set_xlabel(r"$\gamma = n_{\mathrm{tr}} / d^2$")
ax.set_ylabel(
    r"$\Delta_{\mathrm{sp}} = \mathrm{MSE}_{\mathrm{sp}} - \mathrm{MSE}_{\mathrm{OST}}$"
)
ax.set_xlim(gamma.min(), gamma.max())
ax.plot([], [], color="black", lw=1.0, ls="--", alpha=0.6, label=r"$\frac{2(d-1)}{d^3} \gamma^{-1}$")
ax.legend()
# ax.legend(fontsize=8)

plt.show()
# plt.savefig(OUT_DIR + "state_prior_validity_raw.pdf", bbox_inches="tight")
# plt.savefig(OUT_DIR + "state_prior_validity_raw.png", bbox_inches="tight")
