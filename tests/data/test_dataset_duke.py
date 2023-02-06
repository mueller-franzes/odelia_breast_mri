from odelia.data.datasets import DUKE_Dataset3D
import torch 
from pathlib import Path 
from torchvision.utils import save_image

def tensor2image(tensor, batch=0):
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])


ds = DUKE_Dataset3D(
    norm="min-max",
    path_root='/home/gustav/Documents/datasets/DUKE/dataset_256x256x32_lr_2/'
)

print("Dataset Length", len(ds))

item = ds[0]
uid = item["uid"]
img = item['source']
label = item['target']

print("UID", uid, "Image Shape", list(img.shape), "Label", label)

path_out = Path.cwd()/'results/test'
path_out.mkdir(parents=True, exist_ok=True)
img = tensor2image(img[None])
save_image(img, path_out/'test.png', normalize=True)
