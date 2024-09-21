import json
import os
import cv2
import numpy as np
from pycocotools import mask as coco_mask

def draw_mask_and_bbox(image, mask, bbox, color=(0, 255, 0), alpha=0.4):
    """
    Draw a binary mask and bounding box on the image.
    
    Parameters:
    - image (numpy array): Image to draw on.
    - mask (numpy array): Binary mask of the object.
    - bbox (list): Bounding box coordinates [x, y, width, height].
    - color (tuple): Color for the mask and bounding box (default green).
    - alpha (float): Transparency for the mask overlay.
    
    Returns:
    - image with drawn mask and bounding box.
    """
    # Create a colored mask
    if mask is not None:
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask == 1] = color

        # Blend the colored mask with the image
        image = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)

    # Draw the bounding box
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    return image

def rle_to_mask(rle, height, width):
    """
    Decode RLE to a binary mask.
    
    Parameters:
    - rle (dict): Run-length encoding of the mask.
    - height (int): Height of the image.
    - width (int): Width of the image.
    
    Returns:
    - Binary mask as a numpy array.
    """
    return coco_mask.decode(rle).reshape((height, width))

def process_coco_json(coco_json_path, images_dir, output_dir):
    """
    Process the COCO JSON and draw masks and bounding boxes on the images.
    
    Parameters:
    - coco_json_path (str): Path to the COCO JSON file.
    - images_dir (str): Directory where the images are stored.
    - output_dir (str): Directory to save the output images.
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
        image_path = os.path.join(images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        
        # Get mask and bounding box
        if 'segmentation' in ann:
            rle = ann['segmentation']
            mask=None
            if rle is not None and len(rle)!=0:
                mask = rle_to_mask(rle,image_info['height'],image_info['width'])
            bbox = ann['bbox']
            
            # Draw mask and bounding box
            image_with_overlay = draw_mask_and_bbox(image, mask, bbox)
            
            # Save the result
            os.makedirs(os.path.join(output_dir,image_info['file_name'].split('/')[0]),exist_ok=True) #train
            output_image_path = os.path.join(output_dir, image_info['file_name'].split('/')[0],f"{str(ann['id'])}{image_info['file_name'].split('/')[1]}") #train
            # output_image_path=os.path.join(output_dir,f"{str(ann['id'])}{image_info['file_name']}")#val set
            cv2.imwrite(output_image_path, image_with_overlay)
            print(f"Saved: {output_image_path}")

# Paths
coco_json_path = "/home/lhx/mylightning-sam/imagenet_train_SAM50k.json"
images_dir = "data_zoo/imagenet1k"

# coco_json_path = "/home/lhx/mylightning-sam/imagenet_val_SAM500.json"
# images_dir = "data_zoo/imagenet1k_val"

output_dir = "data_zoo/viscoco"

# Process and draw masks and bounding boxes
process_coco_json(coco_json_path, images_dir, output_dir)
