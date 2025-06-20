import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 24,             # base font size
    'axes.titlesize': 24,        # title font size for each subplot
    'axes.labelsize': 22,        # x/y axis label font size
    'xtick.labelsize': 22,       # x tick label font size
    'ytick.labelsize': 21,       # y tick label font size
    'legend.fontsize': 20,       # legend font size
})

# 95 % cut-off values and lambdas
upper_bounds = [1.06, 1.12, 1.2, 1.5, 2.0, 3.0, 4.0]
p = 0.95
lambdas = [-np.log(1 - p) / (u - 1.0) for u in upper_bounds]

x = np.linspace(1, 5, 800)

fig, ax = plt.subplots(figsize=(10, 4.2))

# Sort by decreasing upper bound so smaller ones plot later (on top)
for upper, lam, color in zip(
    *zip(*sorted(zip(upper_bounds, lambdas, plt.cm.tab10.colors)))
):
    cdf = 1 - np.exp(-lam * (x - 1))
    cdf[x < 1] = 0
    mask = x <= upper

    lam_dict = {lam: val for lam,val in zip(lambdas,[1,1.5,3,6,15,24,48])}

    ax.plot(x, cdf, color=color, lw=2,
            label=rf"$\lambda=${lam_dict[lam]:<3}")
    ax.fill_between(x[mask], 0, cdf[mask], color=color, alpha=0.15)

# Axis labels and legend
ax.set_xlabel(r"Training-time $\omega$")
ax.set_ylabel("CDF")
ax.set_xlim(1, 5)
ax.set_yticks([0,0.5,1])
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=16)
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),   # X=1.02 (just outside), Y=0.5 (vertical center)
    frameon=True,
    fontsize=19,
    edgecolor='gray',
    facecolor='white'
)
plt.tight_layout()
plt.savefig("./tables/cdf_95mass_2d.png", dpi=300)
plt.show()
