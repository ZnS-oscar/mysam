from box import Box

config = {
    'mode':'train',
    "num_devices": 2,
    "batch_size": 1,
    "num_workers": 8,
    "num_epochs":4,
    "eval_interval": 1,# deprecated
    "eval_interval_iter_percent": 0.25,
    "out_dir": "./runs/train",
    "opt": {
        "learning_rate": 1e-4,  # 1e-3
        "weight_decay": 1e-4,
        "decay_factor": 5,
        "steps":[184000,192000],#"num_epochs":4
        # "steps":[22220,23190],#numepoch1
        "warmup_steps": 5000,
    },
    "model": {
        "type": 'vit_l',# vit_h, vit_l, vit_b, vit_tiny
        "stu_ckpt":"09232014rightdistilll/train/epoch_003_iter-018735-mse1.41-distillvittiny.pth",
        # "checkpoint":"09221510goodrgbdfreezevit/train/epoch_000000_iter-24996-f10.93-ckpt.pth",
        "checkpoint": "weight/sam_hq_vit_l.pth",
        # "checkpoint":"09241200unfreez/emoreandbl/train/epoch_000000_iter-06249-f10.86-ckpt.pth",
        # "checkpoint":"09221510goodrgbdfreezevit/train/epoch_000000_iter-24996-f10.93-ckpt.pth",
        # "checkpoint":"09232014rightdistilll/train/epoch_003_iter-018735-mse1.41-distillvittiny.pth",
        # "proj_checkpoint":'09181127/train/epoch-03-iter-24156-f10.92-lora32-proj.safetensors',
        "proj_checkpoint":None,
        # "proj_checkpoint":"inSAM",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
            "unfreeze_image_encoder_layer":[0,1],
            "unfreeze_image_encoder_norm":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],#[0],
            "preprocess_layers":False,#False,
            "lora":False # the lora should always be False
            
        },
        "lora":{
            'use_lora':True,
            'r':32,
            'lora_alpha':16,
            # 'target_modules':["query", "value"],
            'lora_dropout':0.1,
            'bias':'none',
            # 'modules_to_save':["classifier"],
            # 'checkpoint':'09241200unfreezemoreandbl/train/epoch_00_iter-06249-f10.86-lora32.safetensors',
            'checkpoint':None,
        }
    },
    "dataset": {
       "train": {
            "root_dir": "data_zoo/imagenet1k",
            "depth_root_dir":"data_zoo/depth/train",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_train_SAM50k.json",
            "imgemb_dir":"data_zoo/imagenet1krgbd_imgemb/train",
        },
        # "train": {
        #     "root_dir": "data_zoo/imagenet1k_val",
        #     "depth_root_dir":"data_zoo/depth/val",
        #     "annotation_file": "data_zoo/imagenet_anno/imagenet_val_SAM500.json",
        # },
        "val": {
            "root_dir": "data_zoo/imagenet1k_val",
            "depth_root_dir":"data_zoo/depth/val",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_val_SAM500.json",
            "imgemb_dir":"data_zoo/imagenet1krgbd_imgemb/val",
        },
        # "val": {
        #     "root_dir": "data_zoo/pcbhbb_slice_coco/images",
        #     "depth_root_dir":"data_zoo/pcbhbb_slice_coco/depth",
        #     "annotation_file": "data_zoo/pcbhbb_slice_coco/images.json",
        # },
        # "test": {
        #     "root_dir": "data_zoo/imagenet1k_val",
        #     "depth_root_dir":"data_zoo/depth/val",
        #     "annotation_file": "data_zoo/imagenet_anno/imagenet_val_SAM500.json",
        # },
    }
}

cfg = Box(config)
# RGBD training 50000 val 500
# 6h/epoch