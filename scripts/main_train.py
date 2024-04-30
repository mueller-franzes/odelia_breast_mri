
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset

from odelia.data.datamodules import DataModule
from odelia.models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet, UNet3D, ResNet2D
import os
import argparse
from collections import Counter

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run_dir', default=None, help='Path to the run directory')
    parser.add_argument('--device_num', default=None, help='')
    parser.add_argument('--network', default=None, help='')
    args = parser.parse_args()
    args.network = 'ResNet101'
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num  # Set to the desired GPU index
    sequence = 'pre'
    endpoint='malignancy'
    if endpoint == 'malignancy':
        from odelia.data.datasets import DUKE_Dataset3D
    elif endpoint == 'art':
        from odelia.data.datasets import DUKE_Dataset3D_art as DUKE_Dataset3D
    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if args.path_run_dir:
        path_run_dir = Path(args.path_run_dir)
    else:
        path_run_dir = Path('/mnt/sda1/swarm-learning/training_results')  / (str(current_time)+'_DUKE_internal_'+args.network+'_'+sequence+'_'+endpoint)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D(
        flip=True, 
        path_root = '/mnt/sda1/swarm-learning/radiology-dataset/divided_odelia_dataset/3d-cnn/train',
        sequence=sequence)

    labels = ds.get_labels()

    # Generate indices and perform stratified split
    indices = list(range(len(ds)))

    #align the labels with the indices , some of the dataset is exluded, but the labels are not
    indices = [i for i in indices if i < len(labels)]
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

    # Create training and validation subsets
    ds_train = Subset(ds, train_indices)
    ds_val = Subset(ds, val_indices)
    # Extract training labels using the train_indices
    train_labels = [labels[i] for i in train_indices]
    # Count the occurrences of each label in the training set
    label_counts = Counter(train_labels)

    # Calculate the total number of samples in the training set
    total_samples = len(train_labels)

    # Print the percentage of the training set for each label
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Label '{label}': {percentage:.2f}% of the training set, Exact count: {count}")

    # print the total number of different labels in the training set:
    print(f"Total number of different labels in the training set: {len(label_counts)}")

    adsValData = DataLoader(ds_val, batch_size=2, shuffle=False)
    # print adsValData type
    print('adsValData type: ', type(adsValData))

    train_size = len(ds_train)
    val_size = len(ds_val)
    print('train_size: ',train_size)
    print('val_size: ',val_size)

    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        batch_size=1,
        num_workers=32,
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
    print(f"Using model: { args.network}")


    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        # every_n_train_steps=log_every_n_steps,
        save_last=True,
        save_top_k=2,
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
        min_epochs=40,
        max_epochs=100,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=path_run_dir)
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


