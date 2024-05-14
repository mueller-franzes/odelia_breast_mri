import logging
from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset

from odelia.data.datasets import DUKE_Dataset3D
from odelia.data.datamodules import DataModule

import os  # !
import logging

from odelia.models import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F
#from swarmlearning.pyt import SwarmCallback

# ------------ Swarm Callback ------------ #!



class ResNet(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            block='basic',
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            feed_forward=True,
            loss=torch.nn.CrossEntropyLoss,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={},
            lr_scheduler=None,
            lr_scheduler_kwargs={}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs)
        self.model = nets.ResNet(block, layers, block_inplanes, spatial_dims, in_ch, 7, 1, False, 'B', 1.0, out_ch,
                                 feed_forward, True)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor
'''
    def training_step(self, batch, batch_idx):
        x, y = batch['source'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        swarmCallback.on_batch_end()
        return loss
'''

max_expochs = 100
if __name__ == "__main__":
    # ------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # path_run_dir = Path.cwd() / 'runs' / str(current_time)
    #path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using {accelerator} for training")

    # ------------ Load Data ----------------
    #dataDir = os.getenv('DATA_DIR', '/tmp/test/host1-partial-data')  # !
    #print current directory
    print("Current Directory " , os.getcwd())
    ds = DUKE_Dataset3D(
        flip=True,
        path_root="/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user-odelia-breast-mri-192.168.33.102/data-and-scratch/app-data/host2-partial-data"
        # path_root = '/mnt/sda1/swarm-learning/radiology-dataset/odelia_dataset_unilateral_256x256x32/'
    )
    print("++++++++++")
    print(len(ds))
    print(ds[0])

    # WARNING: Very simple split approach
    train_size = int(0.64 * len(ds))
    val_size = int(0.16 * len(ds))
    test_size = len(ds) - train_size - val_size
    ds_train = Subset(ds, list(range(train_size)))
    ds_val = Subset(ds, list(range(train_size, train_size + val_size)))
    ds_test = Subset(ds, list(range(train_size + val_size, len(ds))))
    print(train_size)
    print(val_size)
    print(ds_train)
    #print(ds_train[0])
    #print(ds_val[0])
    #print(ds_test[0])


    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=4,
        # num_workers=0,
        # pin_memory=True,
    )

    print('========1========')

    # ------------ Initialize Model ------------
    model = ResNet(in_ch=2, out_ch=2, spatial_dims=3)

    print('========2========')
    # -------------- Training Initialization ---------------
    to_monitor = "val/loss"
    min_max = "min"
    log_every_n_steps = 10

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
        patience=30,  # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),  # dirpath
        monitor=to_monitor,
        every_n_train_steps=log_every_n_steps,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    useCuda = torch.cuda.is_available()

    device = torch.device("cuda" if useCuda else "cpu")
    model = model.to(torch.device(device))
    print('========4========')
    for epoch in range(max_expochs):
        print('---------epoch: ', epoch, '---------')
        trainer = Trainer(
            accelerator=accelerator,
            # devices=[0],
            # precision=16,
            # gradient_clip_val=0.5,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing, early_stopping],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=log_every_n_steps,
            auto_lr_find=False,
            # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available
            max_epochs=1,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )
        trainer.fit(model, datamodule=dm)
    # ---------------- Execute Training ----------------

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    print('========6========')

# ? where place "swarmCallback.on_batch_end()" and "swarmCallback.on_epoch_end(epoch)"
