
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data.dataset import Subset
from pytorch_lightning.loggers import WandbLogger

from odelia.data.datasets import DUKE_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet



if __name__ == "__main__":

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D(
        flip=True, 
        path_root = '/mnt/hdd/datasets/breast/DUKE/dataset_256x256x32_lr_2'
    )

    # WARNING: Very simple split approach
    train_size = int(0.64 * len(ds))
    val_size = int(0.16 * len(ds))
    test_size = len(ds) - train_size - val_size
    ds_train = Subset(ds, list(range(train_size)))
    ds_val = Subset(ds, list(range(train_size, train_size+val_size)))
    ds_test = Subset(ds, list(range(train_size+val_size, len(ds))))
    
    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        ds_test = ds_test,
        batch_size=1, 
        # num_workers=0,
        pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = ResNet(in_ch=1, out_ch=1, spatial_dims=3)


    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  
    min_max = "max"
    log_every_n_steps = 1
    logger = WandbLogger(project='ODELIA', name="ResNet", log_model=False)

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=5, # number of checks with no improvement
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
        # devices=[0],
        precision=16,
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping], 
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps, 
        auto_lr_find=False,
        # limit_train_batches=1,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=20,
        max_epochs=1001,
        num_sanity_val_steps=2,
        logger=logger
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)


