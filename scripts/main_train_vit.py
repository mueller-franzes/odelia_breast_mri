
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset

from odelia.data.datasets import DUKE_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7
import os
import argparse

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run_dir', default=None, help='Path to the run directory')
    parser.add_argument('--device_num', default=None, help='')
    parser.add_argument('--resnet', default=None, help='')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num  # Set to the desired GPU index

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if args.path_run_dir:
        path_run_dir = Path(args.path_run_dir)
    else:
        path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D(
        flip=True, 
        path_root = '/home/jeff/dataset/duke_3d/train_val')

    # WARNING: Very simple split approach
    train_size = int(0.8 * len(ds))
    val_size = int(0.2 * len(ds))
    ds_train = Subset(ds, list(range(train_size)))
    ds_val = Subset(ds, list(range(train_size, train_size+val_size)))
    print('train_size: ',train_size)
    print('val_size: ',val_size)
    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        batch_size=1,
        # num_workers=0,
        pin_memory=True,
    )


    # ------------ Initialize Model ------------
    model = EfficientNet3Db7(in_ch=1, out_ch=1, spatial_dims=3)
    #model = VisionTransformer(in_ch=1, out_ch=1, spatial_dims=3)

    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  
    min_max = "max"
    log_every_n_steps = 1

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
        accelerator='gpu', devices=1,
        # devices=[0],
        precision=16,
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing], #, early_stopping
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps, 
        auto_lr_find=False,
        # limit_train_batches=1,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=20,
        max_epochs=100,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=path_run_dir)
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


