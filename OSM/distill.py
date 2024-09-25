import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
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
from torch.utils.data import Dataset, DataLoader
from distill_dataset import load_distill_datasets
from tensorboardX import SummaryWriter  
from torch import distributed as dist
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
def draw_imgs(images, bboxes, pred_mask, gt_mask, epoch, upiter,iter, save_path='runs/val/masks', show_image=None):
    for sub,img in enumerate(images):
        draw_mask(img, bboxes, pred_mask, gt_mask, epoch, upiter,iter, sub,save_path, show_image)


def draw_mask(image, bboxes, pred_mask, gt_mask, epoch, upiter,iter, sub,save_path='runs/val/masks', show_image=None):
    pred_mask = (pred_mask >= 0.5).float()
    h, w = pred_mask.shape[1], pred_mask.shape[2]
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    # fuse
    # a, b, c, d = np.max(pred_mask), np.min(pred_mask), np.max(gt_mask), np.min(gt_mask)  # 1, 0, 1, 0
    pred_mask_fuse, gt_mask_fuse = np.zeros((h, w)), np.zeros((h, w))
    for i in range(len(pred_mask)):
        pred_mask_fuse += pred_mask[i]
    for i in range(len(gt_mask)):
        gt_mask_fuse += gt_mask[i]
    pred_mask_fuse[pred_mask_fuse > 0] = 1
    gt_mask_fuse[gt_mask_fuse > 0] = 1

    if show_image == None:
        show_image = np.zeros((h, w, 3))

    # draw masks
    gt_mask_fuse[gt_mask_fuse > 0] = 255
    show_image[:, :, 2] = show_image[:, :, 2] + gt_mask_fuse  # B
    pred_mask_fuse[pred_mask_fuse > 0] = 255
    show_image[:, :, 0] = show_image[:, :, 0] + pred_mask_fuse  # R  (B + R = Pure)

    # draw boxes
    # 'OBB'
    # for box in bboxes[0]:
    #     box = box.cpu().numpy().reshape(-1)  # (N,)  x1, y1(0-1), x2, y2(0-1), angle(0-180)
    #     x_min, y_min, x_max, y_max,  = box[0], box[1], box[2], box[3], box[4]
    #     p1 = np.array([x_min, y_min]) * image.shape[2]
    #     p2 = np.array([x_max, y_min]) * image.shape[2]
    #     p3 = np.array([x_max, y_max]) * image.shape[2]
    #     p4 = np.array([x_min, y_max]) * image.shape[2]
    #     center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]) * image.shape[2]

    #     angle = angle / 180 * math.pi
    #     rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
    #                                 [np.sin(angle), np.cos(angle)]])

    #     p1 = np.dot(rotation_matrix, p1 - center) + center
    #     p2 = np.dot(rotation_matrix, p2 - center) + center
    #     p3 = np.dot(rotation_matrix, p3 - center) + center
    #     p4 = np.dot(rotation_matrix, p4 - center) + center

    #     cv2.line(show_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)  # G
    #     cv2.line(show_image, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 255, 0), 2)
    #     cv2.line(show_image, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 2)
    #     cv2.line(show_image, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

    'HBB'
    for box in bboxes[0]:
        box = box.cpu().numpy().reshape(-1)  # (N,)  x1, y1(0-1), x2, y2(0-1), angle(0-180)
        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
        p1 = np.array([x_min, y_min]) 
        p2 = np.array([x_max, y_min]) 
        p3 = np.array([x_max, y_max]) 
        p4 = np.array([x_min, y_max]) 
        center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]) 

        # angle=0
        # angle = angle / 180 * math.pi
        # rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
        #                             [np.sin(angle), np.cos(angle)]])

        # p1 = np.dot(rotation_matrix, p1 - center) + center
        # p2 = np.dot(rotation_matrix, p2 - center) + center
        # p3 = np.dot(rotation_matrix, p3 - center) + center
        # p4 = np.dot(rotation_matrix, p4 - center) + center

        cv2.line(show_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)  # G
        cv2.line(show_image, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 255, 0), 2)
        cv2.line(show_image, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 2)
        cv2.line(show_image, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

    save_path = save_path + '_epoch' + str(epoch) + '_iter'+ str(upiter)+'/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + str(iter) +'_'+str(sub)+ '.png'
    show_image = cv2.cvtColor(show_image.astype("uint8"), cv2.COLOR_RGB2BGR)

    'overlap'
    
    image = torch.squeeze(image, dim=0).cpu()
    image=image[:3,...]
    numpy_image = image.permute(1, 2, 0).numpy()
    numpy_image=numpy_image*IMGSTD+IMGMEAN
    restored_image = (numpy_image * 255).astype(np.uint8)
    restored_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("restored_debug.png", restored_image)

    # 2. overlap
    alpha = 0.6
    show_image = cv2.addWeighted(restored_image, 1 - alpha, show_image, alpha, 0)
    # show_image=cv2.resize(show_image, (4072, 3096))
    cv2.imwrite(save_path, show_image)
    # print('')

# def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
def validate(fabric: L.Fabric, model: Model, sam_lora: LoRA_sam,val_dataloader: DataLoader, epoch: int = 0,upiter:int=0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            if iter>50:
                break
            images, bboxes, gt_masks,img_info = data

            num_images = images.size(0)

            t1 = time.time()
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred_masks, _ = model(images, bboxes)
            t2 = time.time()

            
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_mask = F.sigmoid(pred_mask)
                pred_mask = torch.clamp(pred_mask, min=0, max=1)
                # draw masks, boxes 
                if (fabric.device==1 or fabric.global_rank==0) and iter>(10/val_dataloader.batch_size) \
                            and iter<(25/val_dataloader.batch_size) :
                #either only 1 gpu, or the 1st subprocess; only draw first 10 val img
                    draw_imgs(images, bboxes, pred_mask, gt_mask, epoch, upiter,iter)

                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")

                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)

    inference_time = (t2 - t1) * 1000  # ms
    print('model inference time: ', inference_time, 'ms.')
    with open("runs/val/log.txt", 'a') as f:
        f.write('Val: [' + str(epoch) + '] - ' + f'[ {str(upiter)} ] - ')
        f.write('Mean IoU: [' + str(torch.round(ious.avg, decimals=4)) + '] ')
        f.write('Inference Time: [' + str(round(inference_time, 3)) + 'ms]')
        f.write('\n')
        f.flush()
    f.close()

    fabric.print(f'Validation [{epoch},{upiter}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        # torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
        sam_lora.save_lora_parameters(os.path.join(cfg.out_dir, f"epoch_{epoch-1:02d}_iter-{upiter:05d}-f1{f1_scores.avg:.2f}-lora{sam_lora.rank}.safetensors"))
        # proj_weight={'image_encoder.patch_embed.proj.weight':model.model.image_encoder.patch_embed.proj.weight}
        # save_file(proj_weight, os.path.join(cfg.out_dir, f"epoch-{epoch-1:02d}-iter-{upiter:05d}-f1{f1_scores.avg:.2f}-lora{sam_lora.rank}-proj.safetensors"))
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch_{epoch-1:06d}_f1{f1_scores.avg:.2f}-ckpt.pth"))
        # print(model.model.image_encoder.patch_embed.proj.weight.shape)
    model.train()

def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()



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
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
def distill(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
):  
    mse_loss_meter=AverageMeter()
    eval_interval_iter=int((len(train_loader.dataset)+1)/train_loader.batch_size*cfg.eval_interval_iter_percent/cfg.num_devices)-1
    if fabric.global_rank == 0:
        writer = SummaryWriter('runs/tensorboardwriter')
    # 运行 tensorboard
    # tensorboard --logdir "./runs/tensorboardwriter"
    total_iters=0
    for epoch in range(1, cfg.num_epochs+1):
        # new epoch
        if fabric.global_rank == 0:
            print("------start epoch {}------".format(epoch))
        # train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, data in enumerate(train_loader):
            rgbd, _bboxes,_masks,imgemb,image_filename=data
            total_iters+=1                
            # imgs, target_feats = imgs.cuda(args.local_rank), target_feats.cuda(args.local_rank)
            optimizer.zero_grad()
            predemb=model.model.image_encoder(rgbd)
            loss = customized_mseloss(predemb, imgemb)
            fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(),max_norm=2.0)
            optimizer.step()
            scheduler.step()
            
            mse_loss_meter.update(loss,len(rgbd))
            # if is master process
            if fabric.global_rank == 0:
                # print training info
                if (batch_idx+1) %int(eval_interval_iter*0.05) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                        epoch, batch_idx * len(rgbd) * dist.get_world_size(), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    writer.add_scalar("mse_loss", loss.item(), total_iters)
                
                # save model
                if total_iters % eval_interval_iter == 0:
                    if fabric.global_rank == 0:
                        save_path=os.path.join(cfg.out_dir, f"epoch_{epoch-1:03d}_iter-{total_iters:06d}-mse{mse_loss_meter.avg:.2f}-distillvittiny.pth")
                        print("save model to {}".format(save_path))
                        torch.save(model.model.state_dict(), save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        # save final model
    if fabric.global_rank == 0:
        torch.save(model.model.state_dict(), os.path.join(cfg.out_dir, "iter_final.pth"))
        writer.close()

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

    train_data, val_data = load_distill_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)
    # model.mark_forward_method('get_imgemb')

    # if not os.path.exists('runs/val/log.txt'):
    #     path = Path('runs/val/log.txt')
    #     path.touch()
    # else:
    #     with open('runs/val/log.txt', 'w') as file:
    #         file.truncate(0)

    distill(cfg, fabric, model,optimizer, scheduler, train_data, val_data)
    # torch.cuda.reset_max_memory_allocated()
        
    
    
    # if cfg.mode=='train':
    #     train_sam(cfg, fabric, model, sam_lora,optimizer, scheduler, train_data, val_data)
    #     validate(fabric, model, sam_lora,val_data, epoch=-2,upiter=0)

    # max_memory = get_max_memory_allocated()


if __name__ == "__main__":
    main(cfg)
