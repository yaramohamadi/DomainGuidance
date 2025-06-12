import os
from PIL import Image
import math

# === Config ===
input_root = "./scripts_results/images_samples"  # Source folders
output_dir = "./scripts_results/grids"           # Flat folder for all grid images
os.makedirs(output_dir, exist_ok=True)

max_images = 50
image_size = (128, 128)
padding = 4  # Padding between images

def create_grid_image(image_paths, grid_size, image_size, padding):
    grid_width = grid_size[1] * (image_size[0] + padding) - padding
    grid_height = grid_size[0] * (image_size[1] + padding) - padding
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(image_size)
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        x = col * (image_size[0] + padding)
        y = row * (image_size[1] + padding)
        grid_image.paste(img, (x, y))

    return grid_image

# === Traverse method/dataset folders ===
for method in sorted(os.listdir(input_root)):
    method_path = os.path.join(input_root, method)
    if not os.path.isdir(method_path):
        continue

    for dataset in sorted(os.listdir(method_path)):
        dataset_path = os.path.join(method_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        image_files = sorted([
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])[:max_images]

        if not image_files:
            print(f"❌ No images found in {dataset_path}")
            continue

        grid_rows = math.ceil(len(image_files) / 10)
        grid_cols = min(len(image_files), 10)

        grid_img = create_grid_image(image_files, (grid_rows, grid_cols), image_size, padding)

        # Create flat file name: e.g., DoG_Caltech.png
        file_name = f"{dataset}_{method}.png".replace(" ", "_")
        save_path = os.path.join(output_dir, file_name)
        grid_img.save(save_path)
        print(f"✅ Saved grid: {save_path}")
