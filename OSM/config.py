from box import Box

config = {
    "num_devices": 1 ,
    "batch_size": 1,
    "num_workers": 1,
    "num_epochs": 10,
    "eval_interval": 0.25,
    "eval_interval_iter_percent": 0.01,
    "out_dir": "./runs/train",
    "opt": {
        "learning_rate": 1e-3,  # 1e-3
        "weight_decay": 1e-4,
        "decay_factor": 5,
        "steps": [2000, 4000],
        "warmup_steps": 400,
    },
    "model": {
        "type": 'vit_h',# vit_h, vit_l, vit_b, vit_tiny
        "checkpoint": "weight/sam_hq_vit_h.pth",
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
        }
    },
    "dataset": {
       "train": {
            "root_dir": "data_zoo/imagenet1k_val",
            "depth_root_dir":"data_zoo/depth/val",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_val_fixsize384_tau0.15_N2.json",

        },
        "val": {
            "root_dir": "data_zoo/imagenet1k_val",
            "depth_root_dir":"data_zoo/depth/val",
            "annotation_file": "data_zoo/imagenet_anno/imagenet_val_fixsize384_tau0.15_N2.json",

        }
    }
}

cfg = Box(config)
