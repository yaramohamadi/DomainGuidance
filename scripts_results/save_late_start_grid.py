import os
import glob
import re
import argparse
from PIL import Image, ImageDraw, ImageFont
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

    folder_info.sort()

    image_paths = []
    iter_labels = []
    for iter_num, folder in folder_info:
        image_path = os.path.join(folder, "samples", "0024000", f"{image_index:06d}.png")
        if os.path.exists(image_path):
            # Format label as 0K, 4K, ..., 20K
            if iter_num == 1000:
                label = "0K"
            else:
                label = f"{iter_num // 1000}K"
            image_paths.append(image_path)
            iter_labels.append(label)
        else:
            print(f"Warning: Image {image_index:06d}.png not found in {folder}")

    return image_paths, iter_labels


def overlay_text(img, text, position=(5, 5), font_size=40):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, fill="white", font=font)
    return img


def plot_and_save_grid(image_paths, labels, save_path):
    images = []
    for path, label in zip(image_paths, labels):
        img = Image.open(path).convert("RGB")
        img = overlay_text(img, label)
        images.append(img)

    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    grid_image = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        grid_image.paste(img, (x_offset, 0))
        x_offset += img.width

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid_image.save(save_path)
    print(f"Saved image grid to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image pairs from specific late-start experiments with custom captions inside image.")
    parser.add_argument("--image_index", type=int, required=True, help="Image index (e.g. 0 to 9999)")
    parser.add_argument("--base_dir", type=str, default="/home/ens/AT74470/results/DoG/artbench-10_processed/ablation_latestart/", help="Base directory path")
    parser.add_argument("--out", type=str, default="tables/output_grid.png", help="Output image path")

    args = parser.parse_args()

    image_paths, labels = load_paired_images(args.base_dir, args.image_index)
    if image_paths:
        plot_and_save_grid(image_paths, labels, args.out)
