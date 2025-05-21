import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import ast 
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, roc_auc_score, roc_curve

from odelia.data.datasets import ODELIA_Dataset3D
from odelia.data.datamodules import DataModule
from odelia.models import MST, ResNet, MSTRegression, ResNetRegression
from odelia.utils.roc_curve import cm2x, plot_roc_curve, sensitivity_at_fixed_specificity, specificity_at_fixed_sensitivity

def one_hot(y, num_classes):
    return np.eye(num_classes, dtype=int)[y]

def evaluate(gt, nn, nn_prob, label, label_vals, path_out):
    plt.rcParams.update({'font.size': 12})
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}  
    colors = ['b', 'g', 'r']
    y_prob = np.asarray(nn_prob)  
    y_pred = np.asarray(nn)
    y_true = np.asarray(gt)
    labels = list(range(len(label_vals)))

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    
    # ------------------------------- ROC-AUC ---------------------------------
    y_true_hot = one_hot(y_true, len(label_vals))
    y_prob = np.stack([1-y_prob, y_prob], axis=1) if binary else y_prob # Convert to one-hot 
    # fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
    axis = axes[0]
    results = {'AUC': [], 'Sensitivity': [], 'Specificity': []}
    for i in range(len(label_vals)):
        if binary and i == 0:
            continue
        y_true_i = y_true_hot[:,i]
        y_prob_i = y_prob[:, i]
        fprs, tprs, auc_val, thrs, opt_idx = plot_roc_curve(y_true_i, y_prob_i, axis, color=colors[i], name=f"AUC {label_vals[i]} {label} ", fontdict=fontdict)
        # fprs, tprs, thrs = roc_curve(y_true_hot[:,i], y_prob[:, i], drop_intermediate=False)
        sensitivity = sensitivity_at_fixed_specificity(y_true_i, y_prob_i, fprs, thrs, 0.9)
        specificity = specificity_at_fixed_sensitivity(y_true_i, y_prob_i, tprs, thrs, 0.9)
        print(f"{label_vals[i]} {label}: AUC {auc_val:.2f} Sensitivity {sensitivity:.2f} Specificity: {specificity:.2f}")
        results['AUC'].append(auc_val)
        results['Sensitivity'].append(sensitivity)
        results['Specificity'].append(specificity)
    print(f"{label}: AUC {np.mean(results['AUC']):.2f} Sensitivity {np.mean(results['Sensitivity']):.2f} Specificity: {np.mean(results['Specificity']):.2f}")
    # fig.tight_layout()
    # fig.savefig(path_out/f'roc_{label}.png', dpi=300)



    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    metrics = cm2x(cm, "macro")

    print(f"Accuracy: {acc:.2f}")
    print(f"Sensitivity: {metrics['TPR']:.2f}")
    print(f"Specificity {metrics['TNR']:.2f}")

    df_cm = pd.DataFrame(data=cm, columns=label_vals, index=label_vals)
    # fig, axis = plt.subplots(1, 1, figsize=(4,4))
    axis = axes[1]
    sns.heatmap(df_cm, ax=axis, cbar=False, cmap="Blues", fmt='d', annot=True) 
    axis.set_title(f'{label}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_xlabel('Neural Network' , fontdict=fontdict)
    axis.set_ylabel('Radiologist' , fontdict=fontdict)
    # fig.tight_layout()
    # fig.savefig(path_out/f'confusion_matrix_{label}.png', dpi=300)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(path_out/f'roc_conf_{label}.png', dpi=300)

    #  -------------------------- Agreement -------------------------
    # kappa = cohen_kappa_score(y_true, y_pred, weights="linear") 
    # print(label, "Kappa", kappa)


if __name__ == "__main__":
    #------------ Get Arguments ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default='runs/ODELIA/MST_binary_unilateral_2025_05_13_170027/epoch=22-step=188922.ckpt', type=str)
    parser.add_argument('--test_institution', default='ODELIA', type=str)
    args = parser.parse_args()
    batch_size = 4

    #------------ Settings/Defaults ----------------
    path_run = Path(args.path_run) 
    train_institution = path_run.parent.parent.name
    run_name = path_run.parent.name
    path_out = Path().cwd()/'results'/train_institution/run_name/args.test_institution
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    # ------------ Load Data ----------------
    split = None if args.test_institution == 'RUMC' else 'test' # Use all samples if testing on RUMC  
    binary = run_name.split('_')[1] == "binary"
    config = run_name.split('_')[2]
    ds_test = ODELIA_Dataset3D(split=split, institutions=args.test_institution, binary=binary, config=config)


    dm = DataModule(
        ds_test = ds_test,
        batch_size=batch_size, 
        num_workers=16,
        # pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = run_name.split('_')[0]
    model_map = {
        'ResNet': ResNet if binary else  ResNetRegression,
        'MST': MST if binary else MSTRegression
    }
    MODEL = model_map.get(model, None)
    model = MODEL.load_from_checkpoint(path_run)
    model.to(device)
    model.eval()


    # ------------ Predict ----------------
    results = []
    for batch in tqdm(dm.test_dataloader()):
        uid, source, target = batch['uid'], batch['source'], batch['target']

        with torch.no_grad():
            logits = model(source.to(device)).cpu()

        # Transfer logits to integer 
        pred_prob = model.logits2probabilities(logits)
        pred = model.logits2labels(logits)
        
        for b in range(pred.size(0)):
            results.append({
                'UID': uid[b],
                'GT': target[b].tolist(),
                'NN': pred[b].tolist(),
                'NN_prob': pred_prob[b].tolist(),
            })

    # ------------ Save Results ----------------
    df = pd.DataFrame(results)
    df.to_csv(path_out/'results.csv', index=False)


    # ------------ Evaluate ----------------
    df = pd.read_csv(path_out/'results.csv')
    df['GT'] = df['GT'].apply(ast.literal_eval)
    df['NN'] = df['NN'].apply(ast.literal_eval)
    df['NN_prob'] = df['NN_prob'].apply(ast.literal_eval)

    gt = np.stack(df['GT'].values)
    nn = np.stack(df['NN'].values)
    nn_prob = np.stack(df['NN_prob'].values) 
    labels = ODELIA_Dataset3D.CLASS_LABELS[config] # {'Malignant Lesion': ['No', 'Yes']} if binary else 
    for i in range(gt.shape[1]):
        label = list(labels.keys())[i]
        label_vals = labels[label]
        evaluate(gt[:, i], nn[:, i], nn_prob[:, i], label, label_vals, path_out)

    # If original(bilateral), evaluate for left and right together
    if config == 'original':
        gt = gt.reshape(-1, 1)
        nn = nn.reshape(-1, 1)
        nn_prob = nn_prob.reshape(-1, 1)
        labels = ODELIA_Dataset3D.CLASS_LABELS['unilateral']
        for i in range(gt.shape[1]):
            label = list(labels.keys())[i]
            label_vals = labels[label]
            evaluate(gt[:, i], nn[:, i], nn_prob[:,i], label, label_vals,  path_out)
