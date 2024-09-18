from box import Box

config = {
    'mode':'train',
    "num_devices": 1 ,
    "batch_size": 1,
    "num_workers": 8,
    "num_epochs":5,
    "eval_interval": 1,# deprecated
    "eval_interval_iter_percent": 0.25,
    "out_dir": "./runs/train",
    "opt": {
        "learning_rate": 1e-4,  # 1e-3
        "weight_decay": 1e-4,
        "decay_factor": 5,
        # "steps": [177800 , 185550],
        # "steps":[66680,69580], #numepoch4
        "steps":[177810,185540],#numepoch5
        "warmup_steps": 1000,
    },
    "model": {
        "type": 'vit_l',# vit_h, vit_l, vit_b, vit_tiny
        "checkpoint": "weight/sam_hq_vit_l.pth",
        "proj_checkpoint":'09181127/train/epoch-03-iter-24156-f10.92-lora32-proj.safetensors',
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
            "lora":False # the lora should always be False
        },
        "lora":{
            'r':32,
            'lora_alpha':16,
            # 'target_modules':["query", "value"],
            'lora_dropout':0.1,
            'bias':'none',
            # 'modules_to_save':["classifier"],
            'checkpoint':'09181127/train/epoch-03-iter-24156-f10.92-lora32.safetensors'
        }
    },
    "dataset": {
       "train": {
            "root_dir": "data_zoo/imagenet1k",
            "depth_root_dir":"data_zoo/depth/train",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_train_fixsize384_tau0.15_N2.json",

        },
        "val": {
            "root_dir": "data_zoo/imagenet1k_val",
            "depth_root_dir":"data_zoo/depth/val",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_val_fixsize384_tau0.15_N2.json",

        }
    }
}

cfg = Box(config)
# RGBD training 48319 val 478
# 6h/epoch