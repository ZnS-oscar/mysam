import time
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from OSM.dataset import COCODataset,ResizeAndPad,collate_fn
import multiprocessing
from box import Box

config = {
    "num_devices": 2 ,
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
transform = ResizeAndPad(1024)
# Example dataset
train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
# train_dataloader = DataLoader(train,
#                                   batch_size=cfg.batch_size,
#                                   shuffle=True,
#                                   num_workers=cfg.num_workers,
#                                   collate_fn=collate_fn)


for num_workers in range(2, multiprocessing.cpu_count(), 2):
    start_time = time.time()
    dataloader = DataLoader(train, batch_size=1, num_workers=num_workers,shuffle=False,collate_fn=collate_fn)
    for data in dataloader:
        pass  # Simulate data loading
    print(f"num_workers: {num_workers}, Time: {time.time() - start_time:.2f} seconds")

#bs=2
# num_workers: 2, Time: 1.74 seconds
# num_workers: 4, Time: 1.08 seconds
# num_workers: 6, Time: 0.82 seconds
# num_workers: 8, Time: 0.72 seconds
# num_workers: 10, Time: 0.92 seconds
# num_workers: 12, Time: 0.81 seconds
# num_workers: 14, Time: 0.81 seconds
# num_workers: 16, Time: 0.82 seconds
# num_workers: 18, Time: 0.78 seconds
# num_workers: 20, Time: 0.86 seconds
# num_workers: 22, Time: 0.83 seconds
# num_workers: 24, Time: 0.90 seconds
# num_workers: 26, Time: 0.89 seconds
# num_workers: 28, Time: 0.86 seconds
# num_workers: 30, Time: 0.87 seconds

#bs=1 12,14,16
# num_workers: 2, Time: 1.76 seconds
# num_workers: 4, Time: 1.25 seconds
# num_workers: 6, Time: 1.00 seconds
# num_workers: 8, Time: 0.85 seconds
# num_workers: 10, Time: 0.77 seconds
# num_workers: 12, Time: 0.74 seconds
# num_workers: 14, Time: 0.73 seconds
# num_workers: 16, Time: 0.76 seconds
# num_workers: 18, Time: 0.89 seconds
# num_workers: 20, Time: 0.79 seconds
# num_workers: 22, Time: 0.77 seconds
# num_workers: 24, Time: 0.77 seconds
# num_workers: 26, Time: 0.84 seconds
# num_workers: 28, Time: 0.88 seconds
# num_workers: 30, Time: 0.85 seconds


# Epoch: [2][1/25] | Time [(0.293s)] | Data [(0.430s)] | valTime [(4.592)s] | Focal Loss [(0.0050)] | Dice Loss [(0.0793)] | IoU Loss [(0.0362)] | Total Loss [(0.2162)]
# Epoch: [2][2/25] | Time [(5.189s)] | Data [(0.474s)] | valTime [(0.000)s] | Focal Loss [(0.0045)] | Dice Loss [(0.0989)] | IoU Loss [(0.0313)] | Total Loss [(0.2196)]
# Epoch: [2][3/25] | Time [(5.529s)] | Data [(0.559s)] | valTime [(0.000)s] | Focal Loss [(0.0060)] | Dice Loss [(0.0908)] | IoU Loss [(0.0366)] | Total Loss [(0.2482)]
# Epoch: [2][4/25] | Time [(5.912s)] | Data [(0.623s)] | valTime [(0.000)s] | Focal Loss [(0.0085)] | Dice Loss [(0.1438)] | IoU Loss [(0.0432)] | Total Loss [(0.3574)]
# Epoch: [2][5/25] | Time [(6.214s)] | Data [(0.668s)] | valTime [(0.000)s] | Focal Loss [(0.0087)] | Dice Loss [(0.1438)] | IoU Loss [(0.0441)] | Total Loss [(0.3625)]
# Epoch: [2][6/25] | Time [(6.512s)] | Data [(0.708s)] | valTime [(0.000)s] | Focal Loss [(0.0098)] | Dice Loss [(0.1379)] | IoU Loss [(0.0396)] | Total Loss [(0.3733)]
# Epoch: [2][7/25] | Time [(6.831s)] | Data [(0.755s)] | valTime [(0.000)s] | Focal Loss [(0.0091)] | Dice Loss [(0.1268)] | IoU Loss [(0.0353)] | Total Loss [(0.3437)]
# Epoch: [2][8/25] | Time [(7.133s)] | Data [(0.797s)] | valTime [(0.000)s] | Focal Loss [(0.0138)] | Dice Loss [(0.1200)] | IoU Loss [(0.0323)] | Total Loss [(0.4286)]
# Epoch: [2][9/25] | Time [(7.448s)] | Data [(0.839s)] | valTime [(0.000)s] | Focal Loss [(0.0144)] | Dice Loss [(0.1401)] | IoU Loss [(0.0334)] | Total Loss [(0.4614)]
# Epoch: [2][10/25] | Time [(7.747s)] | Data [(0.879s)] | valTime [(0.000)s] | Focal Loss [(0.0140)] | Dice Loss [(0.1305)] | IoU Loss [(0.0335)] | Total Loss [(0.4437)]
# Epoch: [2][11/25] | Time [(8.041s)] | Data [(0.919s)] | valTime [(0.000)s] | Focal Loss [(0.0139)] | Dice Loss [(0.1263)] | IoU Loss [(0.0318)] | Total Loss [(0.4365)]
# Epoch: [2][12/25] | Time [(8.371s)] | Data [(0.966s)] | valTime [(0.000)s] | Focal Loss [(0.0143)] | Dice Loss [(0.1261)] | IoU Loss [(0.0318)] | Total Loss [(0.4449)]
# Epoch: [2][13/25] | Time [(8.679s)] | Data [(1.019s)] | valTime [(0.000)s] | Focal Loss [(0.0179)] | Dice Loss [(0.1204)] | IoU Loss [(0.0296)] | Total Loss [(0.5070)]
# Epoch: [2][14/25] | Time [(8.992s)] | Data [(1.078s)] | valTime [(0.000)s] | Focal Loss [(0.0167)] | Dice Loss [(0.1132)] | IoU Loss [(0.0275)] | Total Loss [(0.4738)]
# Epoch: [2][15/25] | Time [(9.355s)] | Data [(1.158s)] | valTime [(0.000)s] | Focal Loss [(0.0170)] | Dice Loss [(0.1136)] | IoU Loss [(0.0304)] | Total Loss [(0.4847)]
# Epoch: [2][16/25] | Time [(9.669s)] | Data [(1.207s)] | valTime [(0.000)s] | Focal Loss [(0.0164)] | Dice Loss [(0.1154)] | IoU Loss [(0.0313)] | Total Loss [(0.4753)]
# Epoch: [2][17/25] | Time [(9.993s)] | Data [(1.256s)] | valTime [(0.000)s] | Focal Loss [(0.0183)] | Dice Loss [(0.1235)] | IoU Loss [(0.0310)] | Total Loss [(0.5213)]
# Epoch: [2][18/25] | Time [(10.294s)] | Data [(1.304s)] | valTime [(0.000)s] | Focal Loss [(0.0186)] | Dice Loss [(0.1232)] | IoU Loss [(0.0301)] | Total Loss [(0.5253)]
# Epoch: [2][19/25] | Time [(10.676s)] | Data [(1.408s)] | valTime [(0.000)s] | Focal Loss [(0.0178)] | Dice Loss [(0.1202)] | IoU Loss [(0.0300)] | Total Loss [(0.5070)]
# Epoch: [2][20/25] | Time [(10.969s)] | Data [(1.448s)] | valTime [(0.000)s] | Focal Loss [(0.0183)] | Dice Loss [(0.1152)] | IoU Loss [(0.0285)] | Total Loss [(0.5095)]
# Epoch: [2][21/25] | Time [(11.324s)] | Data [(1.542s)] | valTime [(0.000)s] | Focal Loss [(0.0181)] | Dice Loss [(0.1175)] | IoU Loss [(0.0283)] | Total Loss [(0.5080)]
# Epoch: [2][22/25] | Time [(11.619s)] | Data [(1.585s)] | valTime [(0.000)s] | Focal Loss [(0.0178)] | Dice Loss [(0.1136)] | IoU Loss [(0.0272)] | Total Loss [(0.4978)]
# Epoch: [2][23/25] | Time [(11.944s)] | Data [(1.646s)] | valTime [(0.000)s] | Focal Loss [(0.0181)] | Dice Loss [(0.1120)] | IoU Loss [(0.0285)] | Total Loss [(0.5016)]
# Epoch: [2][24/25] | Time [(12.235s)] | Data [(1.684s)] | valTime [(0.000)s] | Focal Loss [(0.0179)] | Dice Loss [(0.1123)] | IoU Loss [(0.0288)] | Total Loss [(0.4998)]
# Epoch: [2][25/25] | Time [(12.576s)] | Data [(1.741s)] | valTime [(0.000)s] | Focal Loss [(0.0174)] | Dice Loss [(0.1107)] | IoU Loss [(0.0292)] | Total Loss [(0.4871)]

'''
50img/batch 12.6s=data 1.8s + val50draw10 4.6s + 6.2 train50
'''