import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
from abl import ABL

IMGMEAN=torch.tensor([0.485, 0.456, 0.406])
IMGSTD=torch.tensor([0.229, 0.224, 0.225])
#imagenet
DEPMEAN=torch.tensor([0.476])
DEPSTD=torch.tensor([0.278])
#pcb
# DEPMEAN=torch.tensor([0.337])
# DEPSTD=torch.tensor([0.248])
NOT_GOODDATA_CLSLIST=[]
class COCODataset(Dataset):

    def __init__(self, root_dir, depth_root_dir,annotation_file, transform=None):
        self.root_dir = root_dir
        self.depth_root_dir=depth_root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]
                        #   and self.coco.loadImgs(image_id)[0]['file_name'].split('/')[0] in NOT_GOODDATA_CLSLIST ]
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_path=os.path.join(self.depth_root_dir,image_info['file_name'].split('.')[0]+"_depth.jpg")
        depth_image=cv2.imread(depth_path,0)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        
        if len(anns)>50:
            partial_dataset=random.sample(range(0,len(anns)), 50)
            for annid,ann in enumerate(anns):
                if annid not in partial_dataset:
                    continue
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
                mask = self.coco.annToMask(ann)
                masks.append(mask)
        else:
            for annid,ann in enumerate(anns):
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
                mask = self.coco.annToMask(ann)
                masks.append(mask)
        if self.transform:
            image, depth_image,masks, bboxes = self.transform(image, depth_image,masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = torch.tensor(np.stack(masks, axis=0)).float()
        rgbd=torch.concat([image,depth_image],dim=0)
        abl=ABL(ignore_label=255)
        gt_boundary = abl.gt2boundary(masks, )
        dist_maps = abl.get_dist_maps(gt_boundary) # <-- it will slow down the training, you can put it to dataloader.


        return rgbd, torch.tensor(bboxes), masks.float(),dist_maps, image_info['file_name']


def collate_fn(batch):
    images, bboxes, masks,dist_maps,image_name = zip(*batch)
    images = torch.stack(images)
    dist_maps=torch.stack(dist_maps)

    return images, bboxes, masks,dist_maps,image_name


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes

class JustResize:
    def __init__(self, target_size):
        self.target_size = target_size
        
        self.to_tensor = transforms.ToTensor()
        self.normalizeRGB=transforms.Normalize(
                mean=IMGMEAN,
                std=IMGSTD)
        self.normalizeD=transforms.Normalize(
                mean=DEPMEAN,
                std=DEPSTD)

    def __call__(self, image, depth_image,masks, bboxes):
        
        """
        Resize image and adjust bounding boxes.
        
        Parameters:
            image (numpy.ndarray): The input image.
            depth_image (numpy.ndarray): the input depth image
            bboxes (list of tuples): List of bounding boxes as (x_min, y_min, x_max, y_max).
            target_shape (tuple): The target shape (width, height).

        Returns:
            resized_image (numpy.ndarray): Resized image.
            new_bboxes (list of tuples): Adjusted bounding boxes.
        """
        # Get current dimensions
        h, w = image.shape[:2]
        new_w, new_h = self.target_size,self.target_size

        # Calculate resize ratios
        x_ratio = new_w / w
        y_ratio = new_h / h

        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h))
        resized_image= self.to_tensor(resized_image)
        resized_image=self.normalizeRGB(resized_image)

        resized_depth_image = cv2.resize(depth_image, (new_w, new_h))
        #imagenet
        resized_depth_image= self.to_tensor(resized_depth_image)
        #pcb
        # resized_depth_image=torch.from_numpy(resized_depth_image).contiguous()
        # resized_depth_image=resized_depth_image.unsqueeze(0).to(dtype=torch.float32)
        resized_depth_image=self.normalizeD(resized_depth_image)


        resized_masks=[cv2.resize(mask,(new_w, new_h)) for mask in masks]

        # Adjust bounding boxes
        new_bboxes = []
        for bbox in bboxes:
            x, y, w, h= bbox
            new_x = int(x * x_ratio)
            new_y = int(y * y_ratio)
            new_w = int(w * x_ratio)
            new_h = int(h * y_ratio)
            new_bboxes.append((new_x, new_y, new_w, new_h))
        return resized_image, resized_depth_image,resized_masks, new_bboxes

def load_datasets(cfg, img_size):
    # transform = ResizeAndPad(img_size)
    transform= JustResize(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        depth_root_dir=cfg.dataset.train.depth_root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      depth_root_dir=cfg.dataset.val.depth_root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=1,# distill dataset
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader

def load_test_datasets(cfg, img_size):
    # transform = ResizeAndPad(img_size)
    transform= JustResize(img_size)

    test = COCODataset(root_dir=cfg.dataset.test.root_dir,
                      depth_root_dir=cfg.dataset.test.depth_root_dir,
                      annotation_file=cfg.dataset.test.annotation_file,
                      transform=transform)

    test_dataloader = DataLoader(test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return test_dataloader
