import os
import glob
import re
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Only keep these specific iterations
ALLOWED_ITERS = [1000, 4000, 8000, 12000, 16000, 20000]
CAPTION_OVERRIDE = {1000: "0000"}

def extract_iter_number(folder_name):
    match = re.search(r'LATE_START_ITER(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def load_paired_images(base_dir, image_index):
    pattern = os.path.join(base_dir, "dogfinetune_LATE_START_ITER*")
    all_folders = glob.glob(pattern)

    folder_info = []
    for folder in all_folders:
        iter_num = extract_iter_number(os.path.basename(folder))
        if iter_num in ALLOWED_ITERS:
            folder_info.append((iter_num, folder))

    # Sort by iteration value
    folder_info.sort()

    image_paths = []
    iter_labels = []
    for iter_num, folder in folder_info:
        image_path = os.path.join(folder, "samples", "0024000", f"{image_index:06d}.png")
        if os.path.exists(image_path):
            image_paths.append(image_path)
            label = CAPTION_OVERRIDE.get(iter_num, str(iter_num))
            iter_labels.append(label)
        else:
            print(f"Warning: Image {image_index:06d}.png not found in {folder}")

    return image_paths, iter_labels

def plot_and_save_grid(image_paths, labels, save_path):
    images = [Image.open(path) for path in image_paths]
    num_images = len(images)

    fig, axarr = plt.subplots(1, num_images, figsize=(2*num_images, 2.5))
    if num_images == 1:
        axarr = [axarr]

    for ax, img, label in zip(axarr, images, labels):
        ax.imshow(img)
        ax.set_title(f"Iter {label}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved image grid to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image pairs from specific late-start experiments with custom captions.")
    parser.add_argument("--image_index", type=int, required=True, help="Image index (e.g. 0 to 9999)")
    parser.add_argument("--base_dir", type=str, default="/home/ens/AT74470/results/DoG/artbench-10_processed/ablation_latestart/", help="Base directory path")
    parser.add_argument("--out", type=str, default="tables/output_grid.png", help="Output image path")

    args = parser.parse_args()

    image_paths, labels = load_paired_images(args.base_dir, args.image_index)
    if image_paths:
        plot_and_save_grid(image_paths, labels, args.out)
