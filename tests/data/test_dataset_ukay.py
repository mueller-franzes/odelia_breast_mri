from odelia.data.datasets import UKA_Dataset3D
import torch 
from pathlib import Path 
from torchvision.utils import save_image

def tensor2image(tensor, batch=0):
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])


ds = UKA_Dataset3D(mode="val", random_center=False)

print("Dataset Length", len(ds))

item = ds[10]
uid = item["uid"]
img = item['source']
label = item['target']

print("UID", uid, "Image Shape", list(img.shape), "Label", label)

path_out = Path.cwd()/'results/test'
path_out.mkdir(parents=True, exist_ok=True)
img = tensor2image(img[None])
save_image(img, path_out/'test.png', normalize=True)
