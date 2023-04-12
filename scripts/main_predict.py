import os
from pathlib import Path
import logging
from tqdm import tqdm
import torch 
import torch.nn.functional as F
import numpy as np 
from torch.utils.data.dataset import Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

from odelia.data.datasets import DUKE_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet, UNet3D, ResNet2D
from odelia.utils.roc_curve import plot_roc_curve, cm2acc, cm2x

import argparse



if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default=None, help='Path to the run directory')
    parser.add_argument('--path_out', default=None, help='Path to the output directory')
    parser.add_argument('--network', default=None, help='')

    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    if args.path_run:
        path_run = Path(args.path_run)
        args.network = str(path_run).split('_')[-1]
        if len(args.network) == 2:
            args.network = 'efficientnet_' + args.network
        print(args.network)
    else:
        path_run = Path('/home/jeff/PycharmProjects/odelia_breast_mri/2023_04_06_084638_DUKE_ResNet50_swarm_learning')
        args.network = str(path_run).split('_')[-1]
        if len(args.network) == 2:
            args.network = 'efficientnet_' + args.network
        #args.network="ResNet50"
        print(args.network)
    if args.path_out:
        path_out = Path(args.path_out)
    else:
        path_out = Path().cwd()/'results'/path_run.name
        print(path_out)
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}


    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D(
        flip=False,
        path_root = '/home/jeff/dataset/duke_3d/test/'
    )

    # WARNING: Very simple split approach
    ds_test = ds
    
    dm = DataModule(
        ds_test = ds_test,
        batch_size=1,
        # num_workers=0,
        # pin_memory=True,
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
    print(layers)
    if layers is not None:
        # ------------ Initialize Model ------------
        model = ResNet.load_best_checkpoint(path_run, version=0, layers =layers)
        print('1212')
    elif args.network == 'ResNet2D':
        model = ResNet2D(in_ch=1, out_ch=1)
    elif args.network in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
        model = EfficientNet.load_best_checkpoint(path_run, version=0, model_name = args.network)
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
    elif args.network in ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264']:
        model = DenseNet.load_best_checkpoint(path_run, in_ch=1, out_ch=1, spatial_dims=3, model_name=args.network)
    elif args.network == 'UNet3D':
        model = UNet3D.load_best_checkpoint(path_run, in_ch=1, out_ch=1, spatial_dims=3)
    else:
        raise Exception("Invalid network model specified")

    if args.network.startswith('EfficientNet3D'):
        model = EfficientNet3D.load_best_checkpoint(path_run, version=0, blocks_args_str = blocks_args_str)
    #print(model)
    #print(args.network)
    model.to(device)
    model.eval()

    results = {'GT':[], 'NN':[], 'NN_pred':[]}
    for batch in tqdm(dm.test_dataloader()):
        source, target = batch['source'], batch['target']

        # Run Model 
        pred = model(source.to(device)).cpu()
        pred = torch.sigmoid(pred)
        pred_binary = torch.argmax(pred, dim=1)

        results['GT'].extend(target.tolist())
        results['NN'].extend(pred_binary.tolist())
        results['NN_pred'].extend(pred[:, 0].tolist())

    df = pd.DataFrame(results)
    df.to_csv(path_out/'results.csv')

    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(df['GT'], df['NN'])
    tn, fp, fn, tp = cm.ravel()
    n = len(df)
    logger.info("Confusion Matrix: TN {} ({:.2f}%), FP {} ({:.2f}%), FN {} ({:.2f}%), TP {} ({:.2f}%)".format(tn, tn/n*100, fp, fp/n*100, fn, fn/n*100, tp, tp/n*100 ))


    # ------------------------------- ROC-AUC ---------------------------------
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6)) 
    y_pred_lab = np.asarray(df['NN_pred'])
    y_true_lab = np.asarray(df['GT'])
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict, path_out = path_out)
    print('auc_val: ', auc_val)
    fig.tight_layout()
    fig.savefig(path_out/f'roc.png', dpi=300)


    #  -------------------------- Confusion Matrix -------------------------
    acc = cm2acc(cm)
    _,_, sens, spec = cm2x(cm)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    fig, axis = plt.subplots(1, 1, figsize=(4,4))
    sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
    axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]]
    with open(os.path.join(path_out, 'CM.txt'), 'w') as f:
        f.write(str(f'Confusion Matrix ACC={acc:.2f}'))
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'confusion_matrix.png', dpi=300)

    logger.info(f"Malign  Objects: {np.sum(y_true_lab)}")
    logger.info("Confusion Matrix {}".format(cm))
    logger.info("Sensitivity {:.2f}".format(sens))
    logger.info("Specificity {:.2f}".format(spec))

