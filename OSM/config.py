from box import Box

config = {
    "num_devices": 2 ,
    "batch_size": 4,
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
        "type": 'vit_tiny',# vit_h, vit_l, vit_b, vit_tiny
        "checkpoint": "weight/sam_hq_vit_tiny.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
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
