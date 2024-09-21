import json
import cv2
import numpy as np
from pycocotools import mask as coco_mask

def smooth_segmentation_mask(mask, kernel_size=2):
    """
    Smooths a binary segmentation mask using morphological operations.

    Parameters:
    - mask (numpy array): Binary segmentation mask (values should be 0 and 255).
    - kernel_size (int): Kernel size for morphological operations. Default is 5.

    Returns:
    - smooth_mask (numpy array): Smoothed binary mask.
    """
    # Create a structuring element (kernel) for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological closing to remove small holes and smooth the mask
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply morphological opening to remove small noise and isolated pixels
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    return opened_mask

def mask_to_rle(binary_mask):
    """
    Convert binary mask to RLE format (used in COCO).
    """
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8') 
    return rle

def process_coco_json(input_json_path, output_json_path, kernel_size=5):
    """
    Read masks from a COCO JSON, smooth the masks, and save them in a new COCO JSON file.

    Parameters:
    - input_json_path (str): Path to the input COCO JSON file.
    - output_json_path (str): Path to save the modified COCO JSON file.
    - kernel_size (int): Kernel size for smoothing. Default is 5.
    """
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data['annotations']
    
    # Process each annotation in the COCO JSON
    for ann in annotations:
        if 'segmentation' in ann:
            # Get the binary mask from RLE or polygon
            if isinstance(ann['segmentation'], list):
                # If it's a polygon, convert to binary mask
                height = coco_data['images'][ann['image_id']]['height']
                width = coco_data['images'][ann['image_id']]['width']
                binary_mask = coco_mask.frPyObjects(ann['segmentation'], height, width)
                binary_mask = coco_mask.decode(binary_mask)
            else:
                # It's already in RLE format, decode to binary mask
                binary_mask = coco_mask.decode(ann['segmentation'])

            # Smooth the mask
            smoothed_mask = smooth_segmentation_mask(binary_mask, kernel_size)

            # Convert the smoothed mask back to RLE format
            rle_smoothed_mask = mask_to_rle(smoothed_mask)
            rle_smoothed_mask['counts']=str(rle_smoothed_mask['counts'])

            # Update the annotation with the smoothed mask
            ann['segmentation'] = rle_smoothed_mask

    # Save the modified data into a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

# Example usage:
input_json_path = 'data_zoo/imagenet_anno/imagenet_val_fixsize384_tau0.15_N2.json'  # Path to your original COCO JSON
output_json_path = 'data_zoo/imagenet_anno/smoothed_imagenet_val_fixsize384_tau0.15_N2.json'  # Path to save the new COCO JSON with smoothed masks
process_coco_json(input_json_path, output_json_path, kernel_size=5)
