from odelia.data.datasets import ODELIA_Dataset3D
import torch 
from pathlib import Path 
from torchvision.utils import save_image

def tensor2image(tensor, batch=0):
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])

all_institutions = ODELIA_Dataset3D.ALL_INSTITUTIONS
for institution in ['UMCU']:
    ds = ODELIA_Dataset3D(
        institutions=institution,
        # random_flip=True,
        # random_rotate=True, 
        # random_inverse=True,
        # noise=True
    )

    print(f"Dataset {institution} Length", len(ds))
    df = ds.df 
    num_patients = df['PatientID'].nunique()
    # num_cancer = df.groupby('PatientID')[ds.LABEL].apply(lambda x: (x == 2).any()).sum()
    num_cancer = df.groupby('PatientID')[ds.LABEL].apply(lambda x: (x == 1).any()).sum()
    print("Cancer ", num_cancer, " No Cancer:", num_patients-num_cancer )
    print("Cancer ", df[ds.LABEL].sum() )

    item = ds[20]
    uid = item["uid"]
    img = item['source']
    label = item['target']

    print("UID", uid, "Image Shape", list(img.shape), "Label", label)

    path_out = Path.cwd()/'results/test'
    path_out.mkdir(parents=True, exist_ok=True)
    img = tensor2image(img[None])
    save_image(img, path_out/f'test_{institution}.png', normalize=True)
