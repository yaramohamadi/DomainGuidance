import os
import glob
import re
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Keep even MG numbers and also 0.1 (which we treat as 0)
ALLOWED_MG = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CAPTION_OVERRIDE = {0.1: "0000"}

def extract_mg_value(folder_name):
    match = re.search(r'_MG([0-9.]+)_', folder_name)
    if match:
        val = float(match.group(1))
        return 0.0 if val == 0.1 else val  # treat 0.1 as 0
    return None

def load_paired_images(base_dir, image_index):
    pattern = os.path.join(base_dir, "dogfinetune_LATE_START_ITER12000_MG*_W_TRAIN_DOG1.5")
    all_folders = glob.glob(pattern)

    folder_info = []
    for folder in all_folders:
        mg_val = extract_mg_value(os.path.basename(folder))
        if mg_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            folder_info.append((mg_val, folder))

    # Sort by MG value
    folder_info.sort()

    image_paths = []
    labels = []
    for mg_val, folder in folder_info:
        image_path = os.path.join(folder, "samples", "0024000", f"{image_index:06d}.png")
        if os.path.exists(image_path):
            image_paths.append(image_path)
            label = CAPTION_OVERRIDE.get(mg_val if mg_val == 0.0 else mg_val, str(mg_val))
            labels.append(label)
        else:
            print(f"Warning: Image {image_index:06d}.png not found in {folder}")

    return image_paths, labels

def plot_and_save_grid(image_paths, labels, save_path):
    images = [Image.open(path) for path in image_paths]
    num_images = len(images)

    fig, axarr = plt.subplots(1, num_images, figsize=(2*num_images, 2.5))
    if num_images == 1:
        axarr = [axarr]

    for ax, img, label in zip(axarr, images, labels):
        ax.imshow(img)
        ax.set_title(f"Iter {label}", fontsize=10, pad=2)
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved image grid to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image grid from LATE_START_ITER12000 + MG settings.")
    parser.add_argument("--image_index", type=int, required=True, help="Image index (e.g. 0 to 9999)")
    parser.add_argument("--base_dir", type=str, default="/home/ymbahram/scratch/results/DoG/artbench-10_processed/ablation_best_latestart_mg/", help="Base directory path")
    parser.add_argument("--out", type=str, default="tables/output_grid.png", help="Output image path")

    args = parser.parse_args()

    image_paths, labels = load_paired_images(args.base_dir, args.image_index)
    if image_paths:
        plot_and_save_grid(image_paths, labels, args.out)
