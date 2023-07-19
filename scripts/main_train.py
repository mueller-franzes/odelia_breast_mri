
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset

from odelia.data.datasets import DUKE_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet, UNet3D, ResNet2D
import os
import argparse

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run_dir', default=None, help='Path to the run directory')
    parser.add_argument('--device_num', default=None, help='')
    parser.add_argument('--network', default=None, help='')
    args = parser.parse_args()
    args.network = 'DenseNet121'
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num  # Set to the desired GPU index

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
        path_root = '/mnt/sda1/swarm-learning/radiology-dataset/divided_odelia_dataset/3d-cnn/train')

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
    if args.network == 'ResNet18':
        layers = [2, 2, 2, 2]
    elif args.network == 'ResNet34':
        layers = [3, 4, 6, 3]
    elif args.network == 'ResNet50':
        layers = [3, 4, 6, 3]
    elif args.network == 'ResNet101':
        layers = [3, 4, 23, 3]
    elif args.network == 'ResNet152':
        layers = [3, 8, 36, 3]
    else:
        layers = None
    print('layers: ',layers)
    if layers is not None:
        # ------------ Initialize Model ------------
        model = ResNet(in_ch=1, out_ch=1, spatial_dims=3, layers=layers)
    #print('model: ',model)
    if args.network == 'ResNet2D':
            model = ResNet2D(in_ch=1, out_ch=1)
    elif args.network in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
        model = EfficientNet(model_name=args.network, in_ch=1, out_ch=1, spatial_dims=3)
    elif args.network == 'EfficientNet3Db0':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25"]
    elif args.network == 'EfficientNet3Db4':
        blocks_args_str = [
            "r1_k3_s11_e1_i48_o24_se0.25",
            "r3_k3_s22_e6_i24_o32_se0.25",
            "r3_k5_s22_e6_i32_o56_se0.25",
            "r4_k3_s22_e6_i56_o112_se0.25",
            "r4_k5_s11_e6_i112_o160_se0.25",
            "r5_k5_s22_e6_i160_o272_se0.25",
            "r2_k3_s11_e6_i272_o448_se0.25"]
    elif args.network == 'EfficientNet3Db7':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o32_se0.25",
            "r4_k3_s22_e6_i32_o48_se0.25",
            "r4_k5_s22_e6_i48_o80_se0.25",
            "r4_k3_s22_e6_i80_o160_se0.25",
            "r6_k5_s11_e6_i160_o256_se0.25",
            "r6_k5_s22_e6_i256_o384_se0.25",
            "r3_k3_s11_e6_i384_o640_se0.25"]
    elif args.network in['DenseNet121','DenseNet169','DenseNet201','DenseNet264'] :
        model = DenseNet(in_ch=1, out_ch=1, spatial_dims=3, model_name=args.network)
    elif args.network == 'UNet3D':
        model = UNet3D(in_ch=1, out_ch=1, spatial_dims=3)
    elif model is None:
        raise Exception("Invalid network model specified")

    if args.network.startswith('EfficientNet3D'):
        model = EfficientNet3D(in_ch=1, out_ch=1, spatial_dims=3, blocks_args_str=blocks_args_str)

    #model = VisionTransformer(in_ch=1, out_ch=1, spatial_dims=3)

    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  
    min_max = "max"
    log_every_n_steps = 1

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=15, # number of checks with no improvement
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
        min_epochs=50,
        max_epochs=100,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=path_run_dir)
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


