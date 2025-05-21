import torch 
from tqdm import tqdm

from odelia.models import MST, MSTRegression
from odelia.models import ResNet, ResNetRegression
from odelia.data.datasets import ODELIA_Dataset3D
from odelia.data.datamodules import DataModule 



config = "unilateral" # original or unilateral
task = "ordinal" # binary or ordinal
model = "MST" # ResNet or MST
label = None 

binary = task == "binary"
ds_train = ODELIA_Dataset3D(split='train', institutions='ODELIA', binary=binary, config=config, labels=label)


device=torch.device(f'cuda:5')

loss_kwargs = {}
out_ch = len(ds_train.labels)
if task== "ordinal":
    out_ch = sum(ds_train.class_labels_num)  
    loss_kwargs={'class_labels_num': ds_train.class_labels_num} 


if label is not None:
    class_counts = ds_train.df[label].value_counts()
    class_weights = 1 / class_counts / len(class_counts)
    weights = ds_train.df[label].map(lambda x: class_weights[x]).values

model_map = {
    'ResNet': ResNet if binary else ResNetRegression,
    'MST': MST if binary else MSTRegression
}
MODEL = model_map.get(model, None)
model = MODEL(
    in_ch=1, 
    out_ch=out_ch,
    loss_kwargs=loss_kwargs
)


model.to(device)
model.eval()

dm = DataModule(ds_train=ds_train, batch_size=3, num_workers=0)
dl = dm.train_dataloader() 


for idx, batch in tqdm(enumerate(iter(dl))):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)
    print("loss", loss)

