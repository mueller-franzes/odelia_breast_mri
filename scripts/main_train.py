
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from odelia.data.datasets import ODELIA_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet, MST, ResNetRegression, MSTRegression
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--institution', default='ODELIA', type=str)
    parser.add_argument('--model', type=str, default='MST', choices=['ResNet', 'MST']) 
    parser.add_argument('--task', type=str, default="binary", choices=['binary', 'ordinal']) # binary: malignant lesion yes/no, ordinal: no lesion, benign, malignant
    parser.add_argument('--config', type=str, default="unilateral", choices=['original', 'unilateral'])
    args = parser.parse_args()
    binary = args.task == 'binary'

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f'{args.model}_{args.task}_{args.config}_{current_time}'
    path_run_dir = Path.cwd() / 'runs' / args.institution / run_name
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    # ------------ Load Data ----------------
    ds_train = ODELIA_Dataset3D(institutions=args.institution, split='train', binary=binary, config=args.config,
                                 random_flip=True, random_rotate=True, random_inverse=False, noise=True)
    ds_val = ODELIA_Dataset3D(institutions=args.institution, split='val', binary=binary, config=args.config)
    
    samples = len(ds_train) + len(ds_val)
    batch_size = 1
    accumulate_grad_batches = 1 
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    # class_counts = ds_train.df["Lesion"].value_counts()
    # class_weights = 1 / class_counts / len(class_counts)
    # weights = ds_train.df["Lesion"].map(lambda x: class_weights[x]).values

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=batch_size, 
        pin_memory=True,
        weights= None, #weights,
        num_workers=16,
    )

    # ------------ Initialize Model ------------
    loss_kwargs = {}
    out_ch = len(ds_train.labels)
    if not binary:
        out_ch = sum(ds_train.class_labels_num)  
        loss_kwargs={'class_labels_num': ds_train.class_labels_num}

    model_map = {
        'ResNet': ResNet if binary else  ResNetRegression,
        'MST': MST if binary else MSTRegression
    }
    MODEL = model_map.get(args.model, None)
    model = MODEL(
        in_ch = 1,
        out_ch=out_ch,
        loss_kwargs=loss_kwargs
    )


    # Load pretrained model 
    # model = ResNet.load_from_checkpoint('runs/DUKE/2024_11_14_132823/epoch=41-step=17514.ckpt')


    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  if binary else "val/MAE"
    min_max = "max" if binary else "min"
    log_every_n_steps = 50
    logger = WandbLogger(project='ODELIA', group=args.institution, name=run_name, log_model=False)

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


