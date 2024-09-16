from box import Box

config = {
    "num_devices": 1 ,
    "batch_size": 1,
    "num_workers": 4,
    "num_epochs": 7,
    "eval_interval": 1,
    "out_dir": "./runs/train",
    "opt": {
        "learning_rate": 1e-3,  # 1e-3
        "weight_decay": 1e-4,
        "decay_factor": 10,
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
            "root_dir": "data_zoo/tiny_coco_dataset/tiny_coco/train2017",
            "annotation_file": "data_zoo/tiny_coco_dataset/tiny_coco/annotations/instances_train2017.json"
        },
        "val": {
            "root_dir": "data_zoo/tiny_coco_dataset/tiny_coco/val2017",
            "annotation_file": "data_zoo/tiny_coco_dataset/tiny_coco/annotations/instances_val2017.json"
        }
    }
}

cfg = Box(config)
