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
from odelia.models import ResNet
from odelia.utils.roc_curve import plot_roc_curve, cm2acc, cm2x



if __name__ == "__main__":

    #------------ Settings/Defaults ----------------
    path_run = Path.cwd() / 'runs/2023_01_30_084906'
    path_out = Path().cwd()/'results'/path_run.name
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}

    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D(
        flip=False, 
        path_root = '/mnt/hdd/datasets/breast/DUKE/dataset_unilateral_256x256x32'
    )

    # WARNING: Very simple split approach
    train_size = int(0.64 * len(ds))
    val_size = int(0.16 * len(ds))
    test_size = len(ds) - train_size - val_size
    ds_test = Subset(ds, list(range(train_size+val_size, len(ds))))
    
    dm = DataModule(
        ds_test = ds_test,
        batch_size=1, 
        # num_workers=0,
        # pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = ResNet.load_best_checkpoint(path_run, version=0)
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
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'roc.png', dpi=300)


    #  -------------------------- Confusion Matrix -------------------------
    acc = cm2acc(cm)
    _,_, sens, spec = cm2x(cm)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    fig, axis = plt.subplots(1, 1, figsize=(4,4))
    sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True) 
    axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'confusion_matrix.png', dpi=300)

    logger.info(f"Malign  Objects: {np.sum(y_true_lab)}")
    logger.info("Confusion Matrix {}".format(cm))
    logger.info("Sensitivity {:.2f}".format(sens))
    logger.info("Specificity {:.2f}".format(spec))

