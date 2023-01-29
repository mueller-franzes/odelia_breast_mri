from odelia.data.datasets import DUKE_Dataset3D


ds = DUKE_Dataset3D(
    path_root='/mnt/hdd/datasets/breast/DUKE/dataset_256x256x32_lr_2'
)

print(len(ds))

item = ds[0]
img = item['source']
label = item['target']
