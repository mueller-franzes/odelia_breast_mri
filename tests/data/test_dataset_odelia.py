from odelia.data.datasets import ODELIA_Dataset3D
import torch 
from pathlib import Path 
from torchvision.utils import save_image

def tensor2image(tensor, batch=0):
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])

all_institutions = ODELIA_Dataset3D.ALL_INSTITUTIONS
for institution in ['UKA'] :
    ds = ODELIA_Dataset3D(
        institutions=institution,
        # random_flip=True,
        # random_rotate=True, 
        # random_inverse=True,
        # noise=True
    )

    print("Dataset Length", len(ds))

    item = ds[2]
    uid = item["uid"]
    img = item['source']
    label = item['target']

    print("UID", uid, "Image Shape", list(img.shape), "Label", label)

    path_out = Path.cwd()/'results/test'
    path_out.mkdir(parents=True, exist_ok=True)
    img = tensor2image(img[None])
    save_image(img, path_out/f'test_{institution}.png', normalize=True)
