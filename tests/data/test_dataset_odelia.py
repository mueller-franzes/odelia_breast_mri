from odelia.data.datasets import ODELIA_Dataset3D
import torch 
from pathlib import Path 
from torchvision.utils import save_image

def tensor2image(tensor, batch=0):
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])

all_institutions = ODELIA_Dataset3D.ALL_INSTITUTIONS
for institution in all_institutions:
    ds = ODELIA_Dataset3D(
        institutions=institution,
        random_flip=True,
        random_rotate=True, 
        # random_inverse=True,
        # noise=True
        binary=False,
        config='unilateral',
    )

    print(f" ------------- Dataset {institution} ------------" )
    df = ds.df 
    print("Number of exams: ", len(df)) 
    print("Number of patients: ", df['PatientID'].nunique())
    
    for label in ds.labels:
        print(f"Label {label}")
        print(df[label].value_counts())


    # ----------------- Print some examples ----------------
    item = ds[20]
    uid = item["uid"]
    img = item['source']
    label = item['target']

    print("UID", uid, "Image Shape", list(img.shape), "Label", label)

    path_out = Path.cwd()/'results/test'
    path_out.mkdir(parents=True, exist_ok=True)
    img = tensor2image(img[None])
    save_image(img, path_out/f'test_{institution}.png', normalize=True)
