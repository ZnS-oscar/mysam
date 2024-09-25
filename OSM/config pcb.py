from box import Box

config = {
    'mode':'train',
    "num_devices": 2,
    "batch_size": 1,
    "num_workers": 8,
    "num_epochs":30,
    "eval_interval": 1,# deprecated
    "eval_interval_iter_percent": 1,
    "out_dir": "./runs/train",
    "opt": {
        "learning_rate": 1e-4,  # 1e-3
        "weight_decay": 1e-4,
        "decay_factor": 5,
        # "steps": [177800 , 185550],
        # "steps":[66680,69580], #numepoch4
        # "steps":[177810,185540],#numepoch5
        # "steps":[12690,13530],#numepoch1
        "warmup_steps":1000,
    },
    "model": {
        "type": 'vit_l',# vit_h, vit_l, vit_b, vit_tiny
        # "checkpoint": "rgbdbless/train/epoch_000000_iter-24999-f10.93-ckpt.pth",
        # "checkpoint":"09212110goodRGBD/train/epoch_000000_f10.89-ckpt.pth",
        "checkpoint":"weight/sam_hq_vit_l.pth",
        "proj_checkpoint":None,
        # "proj_checkpoint":"inSAM",
        #"proj_checkpoint":"inSAM" means the proj is in sam checkpoint, no need to load
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
            "unfreeze_image_encoder_layer":[0,1],
            "unfreeze_image_encoder_norm":[1,2,3,4],
            "preprocess_layers":False,
            "lora":True # the lora should always be False except inference
        },
        "lora":{
            'r':32,
            'lora_alpha':16,
            # 'target_modules':["query", "value"],
            'lora_dropout':0.1,
            'bias':'none',
            # 'modules_to_save':["classifier"],
            # 'checkpoint':'09212110goodRGBD/train/epoch_-2_iter-00000-f10.83-lora32.safetensors'
            # 'checkpoint':'09212110goodRGBD/train/epoch_01_iter-18747-f10.90-lora32.safetensors'
            'checkpoint':'rgbdbless/train/epoch_00_iter-24999-f10.93-lora32.safetensors'
        }
    },
    "dataset": {
    #     "train": {
    #         "root_dir": "data_zoo/pcbhbb_slice_coco/images",
    #         "depth_root_dir":"data_zoo/pcbhbb_slice_coco/depth",
    #         "annotation_file": "data_zoo/pcbhbb_slice_coco/images.json",
    #     },
       "val": {
            "root_dir": "data_zoo/pcbhbb_slice_coco/images",
            "depth_root_dir":"data_zoo/pcbhbb_slice_coco/depth",
            "annotation_file": "data_zoo/pcbhbb_slice_coco/images.json",
        },
    #     # "test": {
    #     #     "root_dir": "data_zoo/pcbcoco/images/train",
    #     #     "depth_root_dir":"data_zoo/pcbcoco/depth",
    #     #     "annotation_file": "data_zoo/pcbcoco/pcbhbb_coco.json",
    #     # }
    #     "test": {
    #         "root_dir": "data_zoo/pcbhbb_slice_coco/images",
    #         "depth_root_dir":"data_zoo/pcbhbb_slice_coco/depth",
    #         "annotation_file": "data_zoo/pcbhbb_slice_coco/images.json",
    #     },
        "train": {
            "root_dir": "data_zoo/imagenet1k",
            "depth_root_dir":"data_zoo/depth/train",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_train_SAM50k.json",
        },
        # "val": {
        #     "root_dir": "data_zoo/imagenet1k_val",
        #     "depth_root_dir":"data_zoo/depth/val",
        #     "annotation_file": "data_zoo/imagenet_anno/imagenet_val_SAM500.json",
        # },
    }
}

cfg = Box(config)
# RGBD training 48319 val 478
# 6h/epoch