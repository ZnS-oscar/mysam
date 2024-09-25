import torch.nn as nn
import torch.nn.functional as F
from sam import sam_model_registry
from sam import SamPredictor


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint,proj_checkpoint=cfg.model.proj_checkpoint)

    def setup(self, device):
        # self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.model.to(device)
        self.model.train()
        

        '''image encoders already freeze in lora.py __init__(), normally do not uncomment these codes'''
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        # if not self.cfg.model.freeze.lora:
        #     for pname,param in zip(self.model.image_encoder.state_dict(),self.model.image_encoder.parameters()):
        #         if 'linear_a_' in pname or 'linear_b_' in pname:
        #             param.requires_grad = True
        
                           

        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        
        for pname,param in self.model.image_encoder.named_parameters():
            if 'blocks.' in pname or 'layers.' in pname:
                if self.cfg.model.freeze.unfreeze_image_encoder_layer is not None:
                    if int(pname.split('.')[1]) in self.cfg.model.freeze.unfreeze_image_encoder_layer:
                        param.requires_grad=True
                if self.cfg.model.freeze.unfreeze_image_encoder_norm is not None:
                    if int(pname.split('.')[1]) in self.cfg.model.freeze.unfreeze_image_encoder_norm and ('.norm' in pname or '.bn' in pname):
                        param.requires_grad=True
            if not self.cfg.model.freeze.preprocess_layers:
                if ('blocks.' not in pname and 'layers.' not in pname) and 'linear_a_' not in pname and  'linear_b_' not in pname:
                    param.requires_grad=True
            if not self.cfg.model.freeze.lora:
                if 'linear_a_' in pname or 'linear_b_' in pname:
                    param.requires_grad = True
             


    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)  # (Bsize, 256, 64, 64)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )  # sparse_embeddings: (1, 3, 256)  dense_embeddings: (1, 256, 64, 64)

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),  # image_embeddings  (1, 256, 64, 64)
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # positional encoding  (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # embeddings of the points and boxes  (1, 3, 256)
                dense_prompt_embeddings=dense_embeddings,  # embeddings of the mask inputs  (1, 256, 64, 64)
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
    # def get_imgemb(self,images):
    #     image_embeddings = self.model.image_encoder(images)
    #     return image_embeddings

