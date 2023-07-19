#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from odelia.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
out_path = '/home/swarm/PycharmProjects/odelia_breast_mri/scripts/out'
fontdict = {'fontsize': 10, 'fontweight': 'bold'}

result_file = '/home/swarm/Downloads/results.csv'
# load the csv file as df
result_df = pd.read_csv(result_file)
seg_csv = '/mnt/sda1/Oliver/segmentation_metadata_unilateral.csv'
seg_csv = pd.read_csv(seg_csv)
merged_df = pd.merge(result_df, seg_csv[['PATIENT', 'Lesion']], left_on='uid', right_on='PATIENT', how='left')
merged_df['Lesion'] = merged_df['Lesion'].map({0: False, 1: True})
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
print(merged_df)
df = merged_df
#  -------------------------- Confusion Matrix -------------------------
cm = confusion_matrix(df['Lesion'], df['NN'])
tn, fp, fn, tp = cm.ravel()
n = len(df)
logger.info(
    "Confusion Matrix: TN {} ({:.2f}%), FP {} ({:.2f}%), FN {} ({:.2f}%), TP {} ({:.2f}%)".format(tn, tn / n * 100, fp,
                                                                                                  fp / n * 100, fn,
                                                                                                  fn / n * 100, tp,
                                                                                                  tp / n * 100))

# ------------------------------- ROC-AUC ---------------------------------
fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
y_pred_lab = np.asarray(df['NN_pred'])
y_true_lab = np.asarray(df['Lesion'])
tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict,
                                                        path_out=out_path)
print('auc_val: ', auc_val)
fig.tight_layout()
fig.savefig(os.path.join(out_path, 'roc.png'), dpi=300)

#  -------------------------- Confusion Matrix -------------------------
acc = cm2acc(cm)
_, _, sens, spec = cm2x(cm)
df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
fig, axis = plt.subplots(1, 1, figsize=(4, 4))
sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict)  # CM =  [[TN, FP], [FN, TP]]
with open(os.path.join(out_path, 'CM.txt'), 'w') as f:
    f.write(str(f'Confusion Matrix ACC={acc:.2f}'))
axis.set_xlabel('Prediction', fontdict=fontdict)
axis.set_ylabel('True', fontdict=fontdict)
fig.tight_layout()
fig.savefig(os.path.join(out_path, 'confusion_matrix.png'), dpi=300)

logger.info(f"Malign  Objects: {np.sum(y_true_lab)}")
logger.info("Confusion Matrix {}".format(cm))
logger.info("Sensitivity {:.2f}".format(sens))
logger.info("Specificity {:.2f}".format(spec))