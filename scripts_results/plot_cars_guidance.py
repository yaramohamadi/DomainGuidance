import os
import random
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 24,             # base font size
    'axes.titlesize': 35,        # title font size for each subplot
    'axes.labelsize': 22,        # x/y axis label font size
    'xtick.labelsize': 22,       # x tick label font size
    'ytick.labelsize': 21,       # y tick label font size
    'legend.fontsize': 20,       # legend font size
})

# === CONFIGURATION ===
ROOT_DIR = pathlib.Path("/export/datasets/public/diffusion_datasets/tmp_weights/stanford-cars_processed/DiT_inception_ours/control_normalizing_exponential_cutofflatestart/50in1to1.5/dogfinetune_LATE_START_ITER7000_MG1_W_TRAIN_DOG1.5_control1_W_MIN1_W_MAX3")
OUT_FILE = pathlib.Path("./grids_wdgft1.5.png")
SEED = 3
N_IMAGES = 16
GRID_SIZE = (4, 4)
IMAGE_SIZE = (256, 256)

# === COLLECT FOLDERS ===
def collect_paths(root: pathlib.Path):
    out = {}
    for d in root.iterdir():
        if d.is_dir() and "w_dgft" in d.name and d.name[-1].isdigit():
            k = int(d.name[-1])
            subdir = d / "0024000"
            if subdir.exists():
                out[k] = subdir
    return dict(sorted(out.items()))

folders = collect_paths(ROOT_DIR)
folders = {k: v for k, v in folders.items() if 1 <= k <= 5}
if len(folders) != 5:
    raise RuntimeError(f"Expected 4 folders w_dgft1 to w_dgft4, got {len(folders)}")

# === GET COMMON FILENAMES ===
common_filenames = set.intersection(*[set(p.name for p in folder.iterdir() if p.is_file()) for folder in folders.values()])
common_filenames = sorted(list(common_filenames))
if len(common_filenames) < N_IMAGES:
    raise ValueError("Not enough common images.")
random.seed(SEED)
picked = random.sample(common_filenames, N_IMAGES)

# === PLOT ===
fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw=dict(wspace=0.02))
for col, (k, folder) in enumerate(folders.items()):
    axes[col].set_xticks([])
    axes[col].set_yticks([])
    axes[col].set_title(f"$\omega_{{\\text{{DGFT}}}}={k}$")
    for i, name in enumerate(picked):
        r, c = divmod(i, GRID_SIZE[1])
        x0, y0 = c / GRID_SIZE[1], 1 - (r + 1) / GRID_SIZE[0]
        sub_ax = axes[col].inset_axes([x0, y0, 1 / GRID_SIZE[1], 1 / GRID_SIZE[0]])
        img = Image.open(folder / name).convert("RGB").resize(IMAGE_SIZE)
        sub_ax.imshow(np.array(img))
        sub_ax.axis("off")

# === SAVE ===
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved to: {OUT_FILE}")
