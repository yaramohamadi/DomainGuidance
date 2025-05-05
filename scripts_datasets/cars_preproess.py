import os
import scipy.io
import shutil
import re
def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip().replace(' ', '_'))

    
# Set your actual paths
annos_path = "/export/datasets/public/diffusion_datasets/stanford-cars/car_devkit/devkit/cars_train_annos.mat"
meta_path = "/export/datasets/public/diffusion_datasets/stanford-cars/car_devkit/devkit/cars_meta.mat"
image_root = "/export/datasets/public/diffusion_datasets/stanford-cars/cars_train/cars_train/"
output_root = "/export/datasets/public/diffusion_datasets/stanford-cars/cars_processed/"

# Load MATLAB files
annos = scipy.io.loadmat(annos_path)['annotations']
class_names = scipy.io.loadmat(meta_path)['class_names'][0]
class_names = [sanitize(c[0]) for c in class_names]

# Create output folders
os.makedirs(output_root, exist_ok=True)
for name in class_names:
    class_folder = os.path.join(output_root, name)
    os.makedirs(class_folder, exist_ok=True)

# Copy images into class folders
for i in range(annos.shape[1]):
    anno = annos[0][i]
    class_id = int(anno[4][0][0]) - 1  # 1-based to 0-based
    fname = str(anno['fname'][0])
    src = os.path.join(image_root, fname)
    print(class_id)
    print(class_names[class_id])
    dst = os.path.join(output_root, class_names[class_id], fname)

    if os.path.exists(src):
        shutil.copy2(src, dst)
