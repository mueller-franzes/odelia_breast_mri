import numpy as np 
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib

def auc_bootstrapping(y_true, y_score, bootstrapping=1000, drop_intermediate=False):
    tprs, aucs, thrs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    np.random.seed(0)
    rand_idxs = np.random.randint(0, len(y_true), size=(bootstrapping, len(y_true))) # Note: with replacement 
    for rand_idx in rand_idxs:
        y_true_set = y_true[rand_idx]
        y_score_set = y_score[rand_idx]
        fpr, tpr, thresholds = roc_curve(y_true_set, y_score_set, drop_intermediate=drop_intermediate)
        tpr_interp = np.interp(mean_fpr, fpr, tpr) # must be interpolated to gain constant/equal fpr positions
        tprs.append(tpr_interp) 
        aucs.append(auc(fpr, tpr))
        optimal_idx = np.argmax(tpr - fpr)
        thrs.append(thresholds[optimal_idx])
    return tprs, aucs, thrs, mean_fpr



def plot_roc_curve(y_true, y_score, axis, bootstrapping=1000, drop_intermediate=False, fontdict={}, name='ROC', color='b', show_wp=True):
    # ----------- Bootstrapping ------------
    tprs, aucs, thrs, mean_fpr = auc_bootstrapping(y_true, y_score, bootstrapping, drop_intermediate)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0        
    std_tpr = np.nanstd(tprs, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # ------ Averaged based on bootspraping ------
    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs, ddof=1)
  

    # --------- Specific Case -------------
    fprs, tprs, thrs = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
    auc_val = auc(fprs, tprs)
    opt_idx = np.argmax(tprs - fprs)
    opt_tpr = tprs[opt_idx]
    opt_fpr = fprs[opt_idx]

  
    # --------- Plotting -------------
    axis.plot(fprs, tprs, color=color, label=rf"{name} (AUC={auc_val:.2f}$\pm${std_auc:.2f})",
                lw=2, alpha=.8)
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    if show_wp:
        axis.hlines(y=opt_tpr, xmin=0.0, xmax=opt_fpr, color=color, linestyle='--')
        axis.vlines(x=opt_fpr, ymin=0.0, ymax=opt_tpr, color=color, linestyle='--')
    axis.plot(opt_fpr, opt_tpr, color=color, marker='o') 
    axis.plot([0, 1], [0, 1], linestyle='--', color='k')
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.0])
    
    axis.legend(loc='lower right')
    axis.set_xlabel('1 - Specificity', fontdict=fontdict)
    axis.set_ylabel('Sensitivity', fontdict=fontdict)
    
    axis.grid(color='#dddddd')
    axis.set_axisbelow(True)
    axis.tick_params(colors='#dddddd', which='both')
    for xtick in axis.get_xticklabels():
        xtick.set_color('k')
    for ytick in axis.get_yticklabels():
        ytick.set_color('k')
    for child in axis.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('#dddddd')
 
    return fprs, tprs, auc_val, thrs, opt_idx



def cm2acc(cm):
    # [[TN, FP], [FN, TP]] 
    tn, fp, fn, tp = cm.ravel()
    return (tn+tp)/(tn+tp+fn+fp)

def safe_div(x,y):
    if y == 0:
        return float('nan') 
    return x / y


def specificity_at_fixed_sensitivity(y_true, y_scores, tpr, thresholds, sensitivity_target=0.90):
    """ Compute specificity at a fixed sensitivity """    
    # Find the threshold where sensitivity (tpr) is closest to the target (e.g., 90%)
    idx = np.argmin(np.abs(tpr - sensitivity_target))
    chosen_threshold = thresholds[idx]

    # Compute specificity at that threshold
    y_pred = (y_scores >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)  # TN / (TN + FP)
    return specificity

def sensitivity_at_fixed_specificity(y_true, y_scores, fpr, thresholds, specificity_target=0.90):
    """ Compute sensitivity at a fixed specificity """    
    # Find the threshold where specificity is closest to the target (e.g., 90%)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - specificity_target))
    chosen_threshold = thresholds[idx]

    # Compute sensitivity at that threshold
    y_pred = (y_scores >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # TP / (TP + FN)
    return sensitivity

def cm2x(cm, average='macro', pos_label=1):
    """
    Compute sensitivity, specificity, PPV, and NPV for binary or multiclass classification.

    Parameters:
    cm (numpy.ndarray): Confusion matrix (C x C) where C is the number of classes.
    average (str): One of {'binary', 'micro', 'macro', 'weighted'}.
    pos_label (int): Relevant for 'binary' mode, the class considered as positive.

    Returns:
    dict: A dictionary containing PPV, NPV, TPR (Sensitivity), and TNR (Specificity).
    """
    num_classes = cm.shape[0]
    metrics_per_class = {}

    # Micro-Averaging (Global Calculation)
    if average == 'micro':
        TP = np.sum([cm[i, i] for i in range(num_classes)])
        FP = np.sum([cm[:, i].sum() - cm[i, i] for i in range(num_classes)])
        FN = np.sum([cm[i, :].sum() - cm[i, i] for i in range(num_classes)])
        TN = np.sum([cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i]) for i in range(num_classes)])

        return {
            "PPV": safe_div(TP, TP + FP),
            "NPV": safe_div(TN, TN + FN),
            "TPR": safe_div(TP, TP + FN),
            "TNR": safe_div(TN, TN + FP),
        }

    # Compute per-class metrics
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        ppv = safe_div(TP, TP + FP)  # Precision
        npv = safe_div(TN, TN + FN)  # Negative Predictive Value
        tpr = safe_div(TP, TP + FN)  # Sensitivity / Recall
        tnr = safe_div(TN, TN + FP)  # Specificity

        metrics_per_class[i] = {"PPV": ppv, "NPV": npv, "TPR": tpr, "TNR": tnr}

    # Binary Mode
    if average == 'binary':
        if pos_label not in metrics_per_class:
            raise ValueError(f"pos_label={pos_label} is not in the valid class labels: {list(metrics_per_class.keys())}")
        return metrics_per_class[pos_label]

    # Compute per-class metrics as a list
    ppv_vals = [metrics_per_class[i]["PPV"] for i in range(num_classes)]
    npv_vals = [metrics_per_class[i]["NPV"] for i in range(num_classes)]
    tpr_vals = [metrics_per_class[i]["TPR"] for i in range(num_classes)]
    tnr_vals = [metrics_per_class[i]["TNR"] for i in range(num_classes)]
    
    # Macro-Averaging (Unweighted Mean)
    if average == 'macro':
        return {
            "PPV": np.mean(ppv_vals),
            "NPV": np.mean(npv_vals),
            "TPR": np.mean(tpr_vals),
            "TNR": np.mean(tnr_vals),
        }

    # Weighted-Averaging (Weighted by Support)
    if average == 'weighted':
        support = cm.sum(axis=1)  # Number of true instances per class
        total_support = support.sum()
        weights = support / total_support  # Normalize class weights

        return {
            "PPV": np.sum(weights * np.array(ppv_vals)),
            "NPV": np.sum(weights * np.array(npv_vals)),
            "TPR": np.sum(weights * np.array(tpr_vals)),
            "TNR": np.sum(weights * np.array(tnr_vals)),
        }

    raise ValueError("Invalid average method. Choose from {'binary', 'micro', 'macro', 'weighted'}.")
