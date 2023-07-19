#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
from glob import glob
import os
from enum import Enum
import torch
from monai.config import print_config
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    DivisiblePadd,
    Lambdad,
    LoadImaged,
    Resized,
    Rotate90d,
    ScaleIntensityd,
)
from monai.networks.utils import eval_mode
from contextlib import nullcontext
from matplotlib import pyplot as plt
from matplotlib import colors
from monai.data import Dataset, DataLoader
from monai.networks.nets import DenseNet121
from monai.data.utils import pad_list_data_collate
from random import shuffle
import numpy as np
from tqdm.notebook import tqdm, trange
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from monai.visualize import (
    GradCAMpp,
    OcclusionSensitivity,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad,
)
from monai.utils import set_determinism
from monai.apps import download_and_extract
from urllib.request import urlretrieve
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(0)

print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
data_path = os.path.join(root_dir, "CatsAndDogs")
# check folder exists and contains 25,000 jpgs total
if len(glob(os.path.join(data_path, "**", "**", "*.jpg"))) < 25000:
    url = (
        "https://download.microsoft.com/download/3/E/1/"
        + "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    )
    md5 = "e137a4507370d942469b6d267a24ea04"
    download_and_extract(url, output_dir=data_path, hash_val=md5)

class Animals(Enum):
    cat = 0
    dog = 1


def remove_non_rgb(data, max_num=None):
    """Some images are grayscale or rgba. For simplicity, remove them."""
    loader = LoadImaged("image")
    out = []
    for i in data:
        if os.path.getsize(i["image"]) > 100:
            im = loader(i)["image"]
            if im.ndim == 3 and im.shape[-1] == 3:
                out.append(i)
        if max_num is not None and len(out) == max_num:
            return out
    return out


def get_data(animal, max_num=None):
    files = glob(os.path.join(data_path, "PetImages", animal.name.capitalize(), "*.jpg"))
    data = [{"image": i, "label": animal.value} for i in files]
    shuffle(data)
    data = remove_non_rgb(data, max_num)
    return data


# 500 of each class as this is sufficient
cats, dogs = [get_data(i, max_num=500) for i in Animals]
all_data = cats + dogs
shuffle(all_data)

print(f"Num im cats: {len(cats)}")
print(f"Num im dogs: {len(dogs)}")
print(f"Num images to be used: {len(all_data)}")

batch_size = 20
divisible_factor = 20
transforms = Compose(
    [
        LoadImaged("image"),
        EnsureChannelFirstd("image"),
        ScaleIntensityd("image"),
        Rotate90d("image", k=3),
        DivisiblePadd("image", k=divisible_factor),
    ]
)

ds = Dataset(all_data, transforms)
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    collate_fn=pad_list_data_collate,
    drop_last=True,
)

def imshow(data):
    nims = len(data)
    if nims < 6:
        shape = (1, nims)
    else:
        shape = int(np.floor(np.sqrt(nims))), int(np.ceil(np.sqrt(nims)))
    fig, axes = plt.subplots(*shape, figsize=(20, 20))
    axes = np.asarray(axes) if nims == 1 else axes
    for d, ax in zip(data, axes.ravel()):
        # channel last for matplotlib
        im = np.moveaxis(d["image"].detach().cpu().numpy(), 0, -1)
        ax.imshow(im, cmap="gray")
        ax.set_title(Animals(d["label"]).name, fontsize=25)
        ax.axis("off")
    plt.show()

# Random images
rand_idxs = np.random.choice(len(ds), size=12, replace=False)
imshow([ds[i] for i in rand_idxs])

model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
use_amp = True
label_dtype = torch.float16 if use_amp else torch.float32
scaler = torch.cuda.amp.GradScaler() if use_amp else None


def criterion(y_pred, y):
    return torch.nn.functional.cross_entropy(y_pred, y, reduction="sum")


def get_num_correct(y_pred, y):
    return (y_pred.argmax(dim=1) == y).sum().item()

max_epochs = 2
for epoch in trange(max_epochs, desc="Epoch"):
    loss, acc = 0, 0
    for data in dl:
        inputs, labels = data["image"].to(device), data["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast() if use_amp else nullcontext():
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            acc += get_num_correct(outputs, labels)
        if use_amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss += train_loss.item()
    loss /= len(dl) * batch_size
    acc /= len(dl) * batch_size
    print(f"Epoch {epoch+1}, loss: {loss:.3f}, acc: {acc:.4f}")

model.eval()
with eval_mode(model):
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)

    for data in tqdm(dl):
        images, labels = data["image"].to(device), data["label"].to(device)
        with torch.cuda.amp.autocast() if use_amp else nullcontext():
            outputs = model(images).detach()
        y_pred = torch.cat([y_pred, outputs], dim=0)
        y = torch.cat([y, labels], dim=0)

    y_pred = y_pred.argmax(dim=1)

    cm = confusion_matrix(
        y.cpu().numpy(),
        y_pred.cpu().numpy(),
        normalize="true",
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[a.name for a in Animals],
    )
    _ = disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])