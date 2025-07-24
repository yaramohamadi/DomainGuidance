import os
import glob
import re
import argparse
from PIL import Image, ImageDraw, ImageFont

# Allowed MG values
ALLOWED_MG = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CAPTION_OVERRIDE = {0.1: "0000"}

def extract_mg_value(folder_name):
    match = re.search(r'_MG([0-9.]+)_', folder_name)
    if match:
        val = float(match.group(1))
        return 0.0 if val == 0.1 else val
    return None

def load_paired_images(base_dir, image_index):
    pattern = os.path.join(base_dir, "dogfinetune_LATE_START_ITER0_MG*_W_TRAIN_DOG1.5")
    all_folders = glob.glob(pattern)

    folder_info = []
    for folder in all_folders:
        mg_val = extract_mg_value(os.path.basename(folder))
        if mg_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            folder_info.append((mg_val, folder))

    folder_info.sort()

    image_paths = []
    labels = []
    for mg_val, folder in folder_info:
        image_path = os.path.join(folder, "samples", "0024000", f"{image_index:06d}.png")
        if os.path.exists(image_path):
            label = CAPTION_OVERRIDE.get(mg_val, str(mg_val))
            image_paths.append(image_path)
            labels.append(label)
        else:
            print(f"Warning: Image {image_index:06d}.png not found in {folder}")

    return image_paths, labels

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
    parser = argparse.ArgumentParser(description="Visualize MG-guided image grid with inline text annotations.")
    parser.add_argument("--image_index", type=int, required=True, help="Image index (e.g. 0 to 9999)")
    parser.add_argument("--base_dir", type=str, default="/home/ymbahram/scratch/results/DoG/artbench-10_processed/ablation_mghigh/", help="Base directory path")
    parser.add_argument("--out", type=str, default="tables/output_grid.png", help="Output image path")

    args = parser.parse_args()

    image_paths, labels = load_paired_images(args.base_dir, args.image_index)
    if image_paths:
        plot_and_save_grid(image_paths, labels, args.out)
