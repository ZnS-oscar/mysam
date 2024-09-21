import os
import cv2
import numpy as np
image_folder = 'data_zoo/pcbhbb_slice_coco/images'
depth_folder = 'data_zoo/pcbcoco/depth'
output_folder = 'data_zoo/pcbhbb_slice_coco/depth'
# Function to split image into patches with overlap
def split_image_into_patches(image, patch_size=1024, overlap=204):
    patches = []
    step = patch_size - overlap
    h, w = image.shape[:2]
    
    # Adjust patches to not exceed image boundaries
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Adjust the patch size to fit within the image boundaries
            x_end = min(x + patch_size, w)
            y_end = min(y + patch_size, h)
            x_start = max(0, x_end - patch_size)
            y_start = max(0, y_end - patch_size)
            
            patch = image[y_start:y_end, x_start:x_end]
            left_upper = (x_start, y_start)
            right_lower = (x_end, y_end)
            patches.append((patch, left_upper, right_lower))
    
    return patches

# Step 1: Build the dictionary from image/ folder

image_dict = {}

for image_name in os.listdir(image_folder):
    if image_name.endswith('.jpg'):
        parts = image_name.split('_')
        key = '_'.join(parts[:4])  # First 4 elements
        value = parts[4]  # 5th element
        image_dict[key] = value

# Step 2: Split the depth images into patches

os.makedirs(output_folder, exist_ok=True)

for depth_image_name in os.listdir(depth_folder):
    if depth_image_name.endswith('.jpg'):
        parts = depth_image_name.split('_')
        depth_key = '_'.join(parts[:4])  # First 4 elements
        
        # Load the depth image
        depth_image_path = os.path.join(depth_folder, depth_image_name)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        if depth_key in image_dict:
            # Step 3: Split the depth image into patches
            patches = split_image_into_patches(depth_image)

            for idx, (patch, left_upper, right_lower) in enumerate(patches):
                left_x, left_y = left_upper
                right_x, right_y = right_lower
                patch_name = f"{depth_key}_{image_dict[depth_key]}_{left_x}_{left_y}_{right_x}_{right_y}.jpg"
                patch_path = os.path.join(output_folder, patch_name)
                
                # Save the patch
                cv2.imwrite(patch_path, patch)

print("Patching complete!")
