
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data.dataset import Subset
from pytorch_lightning.loggers import WandbLogger

from odelia.data.datasets import ODELIA_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--institution', default='All', type=str)
    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / args.institution / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    # ------------ Load Data ----------------
    ds_train = ODELIA_Dataset3D(institutions=args.institution, split='train', random_flip=True, random_rotate=True, random_inverse=False, noise=True)
    ds_val = ODELIA_Dataset3D(institutions=args.institution, split='val' )
    
    samples = len(ds_train) + len(ds_val)
    batch_size = 2
    accumulate_grad_batches = 1 
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    class_counts = ds_train.df[ds_train.LABEL].value_counts()
    class_weights = 0.5 / class_counts
    weights = ds_train.df[ds_train.LABEL].map(lambda x: class_weights[x]).values

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=batch_size, 
        pin_memory=True,
        weights=None, # weights,
        num_workers=24,
    )

    # ------------ Initialize Model ------------
    model = ResNet(
        in_ch=1, 
        model=18,
        # out_ch=3,
        # loss=torch.nn.CrossEntropyLoss,
        # aucroc_kwargs={"task":"multiclass", "num_classes":3},
        # acc_kwargs={"task":"multiclass", "num_classes":3}
    )

    # Load pretrained model 
    # model = ResNet.load_from_checkpoint('runs/DUKE/2024_11_14_132823/epoch=41-step=17514.ckpt')


    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  
    min_max = "max"
    log_every_n_steps = 50
    logger = WandbLogger(project='ODELIA', name=f"ResNet_{args.institution}", log_model=False)

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=25, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        # every_n_train_steps=log_every_n_steps,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        max_epochs=1000,
        num_sanity_val_steps=2,
        logger=logger
    )
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)


