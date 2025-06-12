import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib as mpl

# Set Times New Roman globally with appropriate font sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 15,             # base font size
    'axes.titlesize': 17,        # title font size
    'axes.labelsize': 15,        # axis label font size
    'xtick.labelsize': 14,       # x-tick label size
    'ytick.labelsize': 14,       # y-tick label size
    'legend.fontsize': 14,       # legend font size
    'figure.titlesize': 18       # overall figure title font size
})


# Define the data
guidance = np.array([
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0
])

art_dog = np.array([
    23.36, 20.64271112, 19.10490305, 18.42764452, 17.68276682, 17.2610323,
    17.45292, 17.6912202, 18.71477625, 18.4938064, 19.39596725, 20.11708508,
    20.62347763, 21.73222594, 22.83895795, 24.2526, 25.46244773, 26.5477468,
    27.70409488, 28.74102158, 30.85407232
])
art_dgf = np.array([
    23.36, 20.70983197, 18.74597628, 18.01384357, 16.84800355, 16.73247574,
    16.83501738, 17.81226611, 18.55197749, 20.17300358, 21.77990726,
    24.65308996, 26.23709141, 30.22777633, 31.22678358, 34.8767639,
    40.58203858, 40.41980935, 51.08742026, 51.36849519, 59.06135442
])

food_dog = np.array([
    16.75, 14.8233533, 13.35101575, 12.27037437, 11.36177606, 10.84938082,
    10.62821246, 10.61583298, 10.73041693, 11.04104528, 11.45124, 11.92891976,
    12.45027169, 13.02274304, 13.57179448, 14.17553112, 14.88756493,
    15.43108207, 16.15558749, 16.6897402, 17.26383781
])
food_dgf = np.array([
    16.75, 14.60125227, 13.12185394, 12.16860384, 11.36623507, 10.73994504,
    10.50626251, 10.450695, 11.245352, 12.41427296, 13.44731361,
    14.31681603, 15.66683464, 17.26066138, 19.42113156, 21.27789529,
    23.87244796, 26.11097034, 28.88691718, 31.4009277, 35.24407647
])

# Define smoothing
def smooth(x, y, num=1000):
    spline = make_interp_spline(x, y, k=3)
    x_smooth = np.linspace(x.min(), x.max(), num)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Smooth curves
x_art_dog, y_art_dog = smooth(guidance, art_dog)
x_art_dgf, y_art_dgf = smooth(guidance, art_dgf)
x_food_dog, y_food_dog = smooth(guidance, food_dog)
x_food_dgf, y_food_dgf = smooth(guidance, food_dgf)

# Setup plot
plt.figure(figsize=(7, 4.5))

# Colors
art_color = "#1f77b4"  # blue
#food_color = "#2ca02c"  # green
food_color = "#d62728"  # rich red

# Plot lines
plt.plot(x_art_dog, y_art_dog, label="Artbench-10 (DoG)", color=art_color, alpha=0.4, linewidth=2)
plt.plot(x_art_dgf, y_art_dgf, label=r"Artbench-10 (DGF)", color=art_color, alpha=1.0, linewidth=2)
plt.plot(x_food_dog, y_food_dog, label="Food-101 (DoG)", color=food_color, alpha=0.4, linewidth=2)
plt.plot(x_food_dgf, y_food_dgf, label=r"Food-101 (DGF)", color=food_color, alpha=1.0, linewidth=2)

# Minima
min_art_dog = np.argmin(art_dog)
min_art_dgf = np.argmin(art_dgf)
min_food_dog = np.argmin(food_dog)
min_food_dgf = np.argmin(food_dgf)

def annotate_min(x, y, label, color, offset=0, alpha=1, bold=False):
    plt.plot(x, y, "o", color=color, alpha=alpha)
    plt.text(
        x, y + offset, f"{y:.2f}",
        ha="center", va="bottom",
        color=color,
        alpha=alpha,
        fontweight="bold" if bold else "normal"
    )


annotate_min(guidance[min_art_dog], art_dog[min_art_dog], "Artbench-10 (DoG)", art_color, offset=3.5, alpha=0.6)
annotate_min(guidance[min_art_dgf], art_dgf[min_art_dgf], "Artbench-10 (DGF)", art_color, offset=1, bold=True)
annotate_min(guidance[min_food_dog], food_dog[min_food_dog], "Food-101 (DoG)", food_color, offset=3.5, alpha=0.6)
annotate_min(guidance[min_food_dgf], food_dgf[min_food_dgf], "Food-101 (DGF)", food_color, offset=1, bold=True)

# Labels and legend
plt.xlabel(r"$\omega$")
plt.ylabel("FID (↓)")
plt.ylim([8, 35])

line_art_dog, = plt.plot(x_art_dog, y_art_dog, label="ArtBench DoG", color=art_color, alpha=0.4, linewidth=2)
line_art_dgf, = plt.plot(x_art_dgf, y_art_dgf, label="ArtBench DGF (Ours)", color=art_color, alpha=1.0, linewidth=2)
line_food_dog, = plt.plot(x_food_dog, y_food_dog, label="Food DoG", color=food_color, alpha=0.4, linewidth=2)
line_food_dgf, = plt.plot(x_food_dgf, y_food_dgf, label="Food DGF (Ours)", color=food_color, alpha=1.0, linewidth=2)


# Add legend outside plot at the bottom with two rows
handles = [line_art_dgf, line_food_dgf, line_art_dog, line_food_dog]
labels = [h.get_label() for h in handles]

legend = plt.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.45),  # increase negative value
    ncol=2,
    frameon=True,
    handletextpad=0.6,
    columnspacing=1.5,
    fancybox=True,             # <-- optional: rounded corners
    edgecolor='gray',         # <-- box border color
    facecolor='white',         # <-- background color
)

# Bold DGF entries
for text in legend.get_texts():
    if "DGF" in text.get_text():
        text.set_fontweight("bold")

# Add extra bottom margin for the legend
plt.subplots_adjust(bottom=0.3)

plt.xticks(np.arange(1.0, 3.1, 0.5))

plt.tight_layout()
plt.grid(True)

plt.savefig("tables/plot_guidance_strength_ablation.png", bbox_inches='tight')
