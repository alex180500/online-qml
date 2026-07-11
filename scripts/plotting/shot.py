import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use(["science", "no-latex"])
NTRAIN = 1000
METRIC = "bias2"
data_file = f"my-data/shot_sweep/ntrain_{NTRAIN}/metrics/{METRIC}.csv"
out_dir = "my-tests/plots/"

METHODS = (
    ("ost", "OST", "#4477AA", "-"),
    ("state_prior_ost", "State-prior OST", "#EE6677", "-"),
    ("povm_prior_ost", "POVM-prior OST", "#AA3377", ":"),
    ("prior_ost", "Prior OST", "#CCBB44", ":"),
    ("ridge", "Ridge regression", "#66CCEE", "--"),
    ("pinv", "Pseudo-inverse", "#228833", "--"),
)

data = pd.read_csv(data_file)
x = data["shots"]

fig = plt.figure(figsize=(4, 3), dpi=300)
for method, label, color, ls in METHODS:
    if method not in data:
        continue
    plt.loglog(x, data[method], lw=1.8, label=label, color=color, linestyle=ls)
    plt.fill_between(
        x,
        data[f"{method}_q30"],
        data[f"{method}_q70"],
        color=color,
        alpha=0.15,
        linewidth=0,
    )

plt.xlabel("Number of Shots")
plt.ylabel("Bias$^2$")
plt.xlim(1, 1e5)

plt.grid(True, alpha=0.25)
plt.legend(fontsize=9)
fig.tight_layout(pad=0.5)
# plt.show()
plt.savefig(out_dir + f"shot_sweep.pdf", bbox_inches="tight")
