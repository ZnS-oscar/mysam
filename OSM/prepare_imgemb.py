import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from losses import BoundaryLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchstat import stat
from thop import profile
from imutils import perspective
import datetime
from pathlib import Path
import math
from lightning.fabric.strategies import DDPStrategy
from lora import LoRA_sam
from safetensors.torch import save_file
from abl import ABL
from train import draw_imgs
torch.set_float32_matmul_precision('high')
IMGMEAN=np.array([0.485, 0.456, 0.406])
IMGSTD=np.array([0.229, 0.224, 0.225])
def xywhangle_to_rext(x, y, w, h, angle):
    rect_xy = []
    rect_wh = []
    rect = []
    rect_xy.append(x)
    rect_xy.append(y)
    rect_wh.append(w)
    rect_wh.append(h)
    rect_xy = tuple(rect_xy)
    rect_wh = tuple(rect_wh)
    rect.append(rect_xy)
    rect.append(rect_wh)
    rect.append(angle)
    rect = tuple(rect)

    return rect

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    mask = mask.cpu().numpy()
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    box = box[0].cpu().numpy().reshape(-1) / 3
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_prediction(image, masks, boxes):
    plt.figure()
    image = image.cpu().numpy()
    plt.imshow(image)
    show_box(boxes, plt.gca())
    show_mask(masks, plt.gca())
    plt.axis('off')
    plt.show()





def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps * 2 / cfg.batch_size:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0] * 2 / cfg.batch_size:
            return 1.0
        elif step < cfg.opt.steps[1] * 2 / cfg.batch_size:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def get_max_memory_allocated():
    return torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert bytes to megabytes


def infer_emb(model,emb,images,bboxes,img_info):
        _, _, H, W = images.shape
        image_embeddings = model.image_encoder(images)  # (Bsize, 256, 64, 64)
        if not (image_embeddings==emb).all():
            raise RuntimeError(f"emb not same {img_info}")
        # pred_masks = []
        # ious = []
        # for embedding, bbox in zip(image_embeddings, bboxes):
        #     sparse_embeddings, dense_embeddings = model.prompt_encoder(
        #         points=None,
        #         boxes=bbox,
        #         masks=None,
        #     )  # sparse_embeddings: (1, 3, 256)  dense_embeddings: (1, 256, 64, 64)

        #     low_res_masks, iou_predictions = model.mask_decoder(
        #         image_embeddings=embedding.unsqueeze(0),  # image_embeddings  (1, 256, 64, 64)
        #         image_pe=model.prompt_encoder.get_dense_pe(),  # positional encoding  (1, 256, 64, 64)
        #         sparse_prompt_embeddings=sparse_embeddings,  # embeddings of the points and boxes  (1, 3, 256)
        #         dense_prompt_embeddings=dense_embeddings,  # embeddings of the mask inputs  (1, 256, 64, 64)
        #         multimask_output=False,
        #     )

        #     masks = F.interpolate(
        #         low_res_masks,
        #         (H, W),
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        #     pred_masks.append(masks.squeeze(1))
        #     ious.append(iou_predictions)

        # return pred_masks, ious

def save_imgemb(fabric: L.Fabric, model: Model, sam_lora: LoRA_sam,val_dataloader: DataLoader,save_path:str):
    model.eval()
    
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks,img_info = data
            num_images = images.size(0)
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            imgembs=model.model.image_encoder(images)
            for iminfo,imgemb in zip(img_info,imgembs):
                if '/' in iminfo:
                    os.makedirs(os.path.join(save_path,iminfo.split('/')[0]),exist_ok=True)
                imgemb_path=os.path.join(save_path,iminfo.replace(".JPEG",".npy"))
                np.save(imgemb_path, imgemb.cpu().numpy())# .astype(np.float16))
    
            if iter%100==0:
                # pred_masks,_=infer_emb(model.model,imgembs,images,bboxes,img_info)
                infer_emb(model.model,imgembs,images,bboxes,img_info)
                # draw_imgs(images=images,bboxes=bboxes,pred_mask=pred_masks,gt_mask=gt_masks,epoch=-233,upiter=0,iter=0,save_path='runs/val/visemb/',)

                

   
        

def main(cfg: Box) -> None:
    print(f"------------Now {cfg.mode}--------------")
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy=DDPStrategy(find_unused_parameters=True),
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    # with fabric.device:
    model = Model(cfg)
    # put sam into lora
    sam_lora=LoRA_sam(model.model,cfg.model.lora.r)
    if cfg.model.lora.checkpoint is not None:
        sam_lora.load_lora_parameters(cfg.model.lora.checkpoint)
    model.model=sam_lora.sam 
    if cfg.model.checkpoint is not None:# pop the proj anyway
            with open(cfg.model.checkpoint, "rb") as f:
                state_dict = torch.load(f)
    if cfg.model.proj_checkpoint== "inSAM":# means proj is in sam's ckpt, no need another pth
        sam_lora.sam.load_state_dict(state_dict, strict=False)

    model.setup(fabric.device)
    # for pname,param in model.model.named_parameters():
    #     if param.requires_grad:
    #         if 'linear_a_' not in pname and 'linear_b_' not in pname and pname!='image_encoder.patch_embed.proj.weight':
    #             raise  ValueError(f'weight {pname} should be freeze')

    print("unfreeze params----------------------")
    for pname,param in model.model.named_parameters():
        if param.requires_grad:
            print(pname)
   # print(model)
    # imagergbd = torch.randn(1, 4, 1024, 1024).to(fabric.device)
    # boxes = torch.randn(1, 4).to(fabric.device)
    # boxes = [boxes]
    # boxes = tuple(boxes)
    # flops, params = profile(model, inputs=(imagergbd, boxes))
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    # optimizer, scheduler = configure_opt(cfg, model)
    # model, optimizer = fabric.setup(model, optimizer)
    model.eval()
    # if not os.path.exists('runs/val/log.txt'):
    #     path = Path('runs/val/log.txt')
    #     path.touch()
    # else:
    #     with open('runs/val/log.txt', 'w') as file:
    #         file.truncate(0)

    torch.cuda.reset_max_memory_allocated()
    train_imgemb_path='/home/lhx/mysam/data_zoo/imagenet1krgbd_imgemb/train'
    val_imgemb_path='/home/lhx/mysam/data_zoo/imagenet1krgbd_imgemb/val'
    save_imgemb(fabric, model, sam_lora,train_data,train_imgemb_path)
    save_imgemb(fabric, model, sam_lora,val_data,val_imgemb_path)
    # if cfg.mode=='train':
    #     train_sam(cfg, fabric, model, sam_lora,optimizer, scheduler, train_data, val_data)
    #     validate(fabric, model, sam_lora,val_data, epoch=-2,upiter=0)

    # max_memory = get_max_memory_allocated()


if __name__ == "__main__":
    main(cfg)
