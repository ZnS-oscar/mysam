import os
import cv2
import numpy as np

def normalize_image(image):
    # Find the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Normalize the image to 0 - 255 range
    normalized_img = (image - min_val) * (255.0 / (max_val - min_val))
    return normalized_img.astype(np.uint8)

def process_images_in_folder(folder_path,out_dir):
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            # Read the depth image (assume it's grayscale, 1 channel)
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue
            
            # Normalize the image to 0-255
            normalized_img = normalize_image(image)
            cv2.imwrite(os.path.join(out_dir,file_name),normalized_img)
            # Update pixel statistics
            pixel_sum += np.sum(normalized_img)
            pixel_squared_sum += np.sum(normalized_img ** 2)
            num_pixels += normalized_img.size

    # Calculate mean and standard deviation
    pixel_mean = pixel_sum / num_pixels
    pixel_std = np.sqrt((pixel_squared_sum / num_pixels) - (pixel_mean ** 2))

    return pixel_mean, pixel_std

# Folder path containing the depth images
depth_image_folder = '../data_zoo/pcbhbb_slice_coco/depth'
out_dir="../data_zoo/pcbhbb_slice_coco/norm_depth"
# Process the images and compute pixel_mean and pixel_std
pixel_mean, pixel_std = process_images_in_folder(depth_image_folder,out_dir)

# Output the results
print(f'Pixel mean after normalization: {pixel_mean}')
print(f'Pixel std after normalization: {pixel_std}')
