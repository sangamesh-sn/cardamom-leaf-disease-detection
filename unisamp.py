import os
import shutil
import math

def move_evenly_spread_images(src_folder, dst_folder, num_to_move):
    os.makedirs(dst_folder, exist_ok=True)
    
    all_images = sorted(os.listdir(src_folder))  # sorting for predictable order
    total_images = len(all_images)
    
    interval = total_images / num_to_move
    index = 0.0
    moved = 0

    while moved < num_to_move and int(index) < total_images:
        img = all_images[int(index)]
        src = os.path.join(src_folder, img)
        dst = os.path.join(dst_folder, img)
        shutil.move(src, dst)
        moved += 1
        index += interval
    
    print(f"Moved {moved} images from {src_folder} to {dst_folder}")

# Configuration
train_dir = 'ddd_ds/train'
val_dir = 'ddd_ds/validation'

move_plan = {
    'CloseEye': 8404,
    'OpenEye': 8576
}

for class_name, count in move_plan.items():
    src_path = os.path.join(train_dir, class_name)
    dst_path = os.path.join(val_dir, class_name)
    move_evenly_spread_images(src_path, dst_path, count)
