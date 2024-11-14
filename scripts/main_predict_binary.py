import argparse
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

from odelia.data.datasets import ODELIA_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import ResNet
from odelia.utils.roc_curve import plot_roc_curve, cm2acc, cm2x



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default='/home/gustav/code/odelia_breast_mri/runs/RSH/2024_11_14_155246_pretrained', type=str)
    parser.add_argument('--test_institution', default='', type=str) # Leave empty to test model on the test set of the training institution 
    args = parser.parse_args()

    labels = ["No Cancer", 'Cancer']

    #------------ Settings/Defaults ----------------
    path_run = Path(args.path_run) 
    train_institution = path_run.parts[-2]
    version = path_run.parts[-1]
    test_institution = train_institution if args.test_institution == "" else args.test_institution
    path_out = Path().cwd()/'results'/train_institution/version/test_institution
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}

    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # ------------ Load Data ----------------
    ds_test = ODELIA_Dataset3D(institutions=test_institution, split='test')


    dm = DataModule(
        ds_test = ds_test,
        batch_size=1, 
        # num_workers=0,
        # pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = ResNet.load_best_checkpoint(path_run)
    model.to(device)
    model.eval()

    results = {'GT':[], 'NN':[], 'NN_prob':[]}
    for batch in tqdm(dm.test_dataloader()):
        source, target = batch['source'], batch['target']

        # Run Model 
        pred = model(source.to(device)).cpu()
        pred_prob = torch.sigmoid(pred)[:, 0]
        pred = (pred_prob>0.5).type(torch.int)
        

        results['GT'].extend(target.tolist())
        results['NN_prob'].extend(pred_prob.tolist())
        results['NN'].extend(pred.tolist())

    df = pd.DataFrame(results)
    df.to_csv(path_out/'results.csv')

    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(df['GT'], df['NN'], labels=list(range(len(labels))))
    print(cm)

    
    # ------------------------------- ROC-AUC ---------------------------------
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6)) 

    y_pred_lab = np.asarray(df['NN_prob'])
    y_true_lab = np.asarray(df['GT'])

    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, color='r', name=f'{labels[1]}:', fontdict=fontdict)
    print("AUC", auc_val)
    axis.set_title(f'{test_institution}', fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'roc.png', dpi=300)


    #  -------------------------- Confusion Matrix -------------------------
    acc = cm2acc(cm)
    _,_, sens, spec = cm2x(cm)
    df_cm = pd.DataFrame(data=cm, columns=labels, index=labels)
    fig, axis = plt.subplots(1, 1, figsize=(3,3))
    sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True, cmap='Reds') 
    axis.set_title(f'Confusion Matrix', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_title(f'{test_institution} ACC={acc:.2f}', fontdict=fontdict)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=0)
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'confusion_matrix.png', dpi=300)



