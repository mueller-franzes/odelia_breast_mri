
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset
from pytorch_lightning.loggers import WandbLogger

from odelia.data.datasets.dataset_3d_uka import UKA_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet


if __name__ == "__main__":

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    # ------------ Load Data ----------------
    ds_train = UKA_Dataset3D(mode="train", flip=True, noise=True, random_center=True)
    ds_val = UKA_Dataset3D(mode="val")

    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        batch_size=2, 
        # num_workers=0,
    ) 


    # ------------ Initialize Model ------------
    model = ResNet(in_ch=1, out_ch=1, spatial_dims=3)

    
    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  
    min_max = "max"
    log_every_n_steps = 1
    logger = WandbLogger(project='UKA_Classification', name="ResNet", log_model=False)

    # early_stopping = EarlyStopping(
    #     monitor=to_monitor,
    #     min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
    #     patience=5, # number of checks with no improvement
    #     mode=min_max
    # )
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
        callbacks=[checkpointing, ], # early_stopping
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps, 
        # limit_train_batches=1,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        # min_epochs=20,
        # max_epochs=1001,
        num_sanity_val_steps=2,
        logger=logger
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)


