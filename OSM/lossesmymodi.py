import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
ALPHA = 0.8
GAMMA = 2
from abl import ABL
# def manual_bce_with_logits(inputs, targets, epsilon=1e-7):

    
#     # Apply sigmoid to convert logits to probabilities
#     pred_probs = torch.sigmoid(inputs)
#     pred_probs = torch.clamp(pred_probs, min=epsilon, max=1-epsilon)
    
#     # Clamp probabilities to avoid log(0) and log(1)

#     # Compute binary cross-entropy manually
#     loss = - (targets * torch.log(pred_probs) + (1 - targets) * torch.log(1 - pred_probs))

#     return loss


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, epsilon=1e-8):
        # Apply sigmoid to the inputs to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Clamp the inputs to prevent probabilities from being exactly 0 or 1
        inputs = torch.clamp(inputs, min=epsilon, max=1.0 - epsilon)

        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply the focal loss formula
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * ((1 - BCE_EXP) ** gamma) * BCE
        
        # Return the mean focal loss
        focal_loss = focal_loss.mean()

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = F.sigmoid(inputs)
        inputs=torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-8, max=1 - 1e-8)
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class BoundaryLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()
    def mask_to_boundary(self,mask, dilation_ratio=0.02):
        """
        Convert binary mask to boundary mask.
        :param mask (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary mask (numpy array)
        """
        _, h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary.
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        # G_d intersects G in the paper.
        return mask - mask_erode

    def forward(self,dt,gt, dilation_ratio=0.02):
        """
        Compute boundary iou between two binary masks.
        :param gt (numpy array, uint8): binary mask
        :param dt (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary iou (float)
        """
        gt_boundary = self.mask_to_boundary(gt, dilation_ratio)
        dt_boundary = self.mask_to_boundary(dt, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        boundary_iou = intersection / union
        return boundary_iou

class BinaryBoundaryLoss(ABL):
    def __init__(self,isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        ABL.__init__(self,isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.)
    def forward(self, logits, target,dist_maps):
        input_logits=torch.cat([logits,-logits],dim=1)
        return super(BinaryBoundaryLoss, self).forward(input_logits, target,dist_maps) 