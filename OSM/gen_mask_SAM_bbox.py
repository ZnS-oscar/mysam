import os
import json
import torch
import numpy as np
from torch.multiprocessing import Pool, set_start_method
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as mask_utils
from tqdm import tqdm
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.strategies import DDPStrategy
import lightning as L
from .model import Model
from torch.utils.data import DataLoader
from .dataset import load_datasets,load_test_datasets
# Initialize SAM model on a specific device (GPU)
cfg=Box({
        "num_gpus":1,
        "points_per_side":8,
        "points_per_batch":64,
        "mask_number":2,
        "min_mask_region_area":50,
    })
def load_sam_model(checkpoint_path, device):
    model_type = "vit_h"  # or "vit_l" depending on your model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    return sam

# Generate masks on a specific GPU
def generate_masks_on_gpu(args):
    image_info, sam_checkpoint, num_masks, device_id = args
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # Initialize SAM model on the current GPU
    sam = load_sam_model(sam_checkpoint, device)
    
    # Load image
    image_id, image_path = image_info
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=cfg.points_per_side,points_per_batch=cfg.points_per_batch,min_mask_region_area=cfg.min_mask_region_area)
    masks = mask_generator.generate(image_np)
    image_height, image_width = image_np.shape[:2]
    for mask in masks:
        mask['height'] = image_height
        mask['width'] = image_width

    return image_id, masks

def select_relevant_masks(masks, image_size, num_masks=2):
    img_center = np.array(image_size) / 2
    def relevance_score(mask):
        def gaussian_peak(x, sigma=20):
            return np.exp(-((x - 0.4) ** 2) / (2 * sigma ** 2))
        binary_mask = mask['segmentation']
        mask_area = np.sum(binary_mask)
        y_coords, x_coords = np.where(binary_mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return float('inf')  # Skip empty masks
        centroid = np.array([np.mean(x_coords), np.mean(y_coords)])  # Calculate the centroid
        distance_to_center = np.linalg.norm(img_center - centroid)
        half_diag=np.linalg.norm(image_size)
        # print(distance_to_center/half_diag,gaussian_peak(mask_area/img_center[0]/img_center[1]))
        return distance_to_center/half_diag-100*gaussian_peak(mask_area/image_size[0]/image_size[1])

    masks = sorted(masks, key=lambda x: relevance_score(x))
    for mask in masks:
        a=mask['segmentation']
        print(len(a))


    return masks[:num_masks]

def convert_mask_to_coco_format(mask, image_id, category_id, mask_id):
    # Binary mask from SAM
    binary_mask = np.array(mask['segmentation'], dtype=np.uint8)  # Ensure it's in uint8 format
    
    # Convert the binary mask to RLE (Run Length Encoding) format
    rle = mask_utils.encode(np.asfortranarray(binary_mask))  # Encoding requires fortran order array
    rle['counts'] = rle['counts'].decode('utf-8')  # RLE count must be stored as strings in COCO
    
    coco_annotation = {
        "id": mask_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,  # Use RLE format for the segmentation field
        "area": mask_utils.area(rle).item(),  # Calculate the area from RLE
        "bbox": mask_utils.toBbox(rle).tolist(),  # Bounding box from RLE
        "iscrowd": 0
    }
    
    return coco_annotation
def save_coco_json(annotations, output_json_path):
    with open(output_json_path, 'w') as f:
        json.dump(annotations, f, indent=4)

def process_images_in_parallel(image_info_list, coco_data, sam_checkpoint, output_json_path, num_gpus=cfg.num_gpus):
    annotations = {
        "images": coco_data["images"],
        "annotations": [],
        "categories": coco_data["categories"]
    }

    pool_inputs = [(image_info, sam_checkpoint, cfg.mask_number, i % num_gpus) for i, image_info in enumerate(image_info_list)]

    with Pool(processes=num_gpus) as pool:
        with tqdm(total=len(image_info_list), desc="Processing images") as pbar:
            for result in pool.imap(generate_masks_on_gpu, pool_inputs):
                image_id, masks = result
                img_info = next(img for img in coco_data["images"] if img["id"] == image_id)
                relevant_masks = select_relevant_masks(masks, (img_info['width'], img_info['height']), num_masks=2)
                
                for i, mask in enumerate(relevant_masks):
                    coco_annotation = convert_mask_to_coco_format(mask, image_id, 1, len(annotations['annotations']) + 1)
                    annotations['annotations'].append(coco_annotation)
                
                # Update progress bar after processing each image
                pbar.update(1)

    save_coco_json(annotations, output_json_path)


# Main function to start the process
if __name__ == "__main__":
    try:
        set_start_method('spawn')  # For multiprocessing on GPUs
    except RuntimeError:
        pass
   
    # Paths
    coco_json_path = 'imagenet1k_val_cocobox.json'
    images_folder = 'data_zoo/imagenet1k_val'
    output_json_path = 'imagenet_val_SAM500.json'
    checkpoint_path = 'weight/sam_hq_vit_h.pth'
    
    # Load COCO JSON
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Prepare image info list
    image_info_list = [(img_info["id"], os.path.join(images_folder, img_info["file_name"])) for img_info in coco_data["images"]]

    # Run processing on multiple GPUs
    process_images_in_parallel(image_info_list, coco_data, checkpoint_path, output_json_path, num_gpus=2)

    print("COCO JSON with relevant masks has been saved!")
