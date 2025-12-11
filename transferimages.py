import os
import shutil

source_root = 'dataset/train'
destination_root = 'dataset/test'
interval = 5   # move every 5th image

# Loop over each class folder in train/
for cls in os.listdir(source_root):
    # Skip hidden files like .DS_Store
    if cls.startswith('.'):
        continue

    src_folder = os.path.join(source_root, cls)
    if not os.path.isdir(src_folder):
        continue  # skip anything that is not a folder

    dst_folder = os.path.join(destination_root, cls)
    os.makedirs(dst_folder, exist_ok=True)

    # List images in this class
    image_files = sorted(os.listdir(src_folder))
    count = 0

    for filename in image_files:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        if count % interval == 0:
            source_path = os.path.join(src_folder, filename)
            destination_path = os.path.join(dst_folder, filename)
            shutil.move(source_path, destination_path)
        count += 1

print("Images Transferred Successfully...")