#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import glob

import pandas as pd
from sklearn.metrics import roc_auc_score

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy.stats import sem
from math import sqrt
# get all the
# assuming df1 and df2 are your dataframes
#cnn_folder = '/mnt/sda1/Duke Compare/rerun_internal/'
#mil_folder = '/mnt/sda1/Duke Compare/mil/trained_models/'
test_folder = '/mnt/sda1/Duke Compare/delongtest/resnet101_int'
# get all the files ends with results.csv
results_files = []
for folder in [test_folder]:
    files = glob.glob(folder+'/*result*.csv')
    results_files.extend(files)
# invert the list
results_files = results_files[::-1]
#f2 = pd.read_csv('/mnt/sda1/Duke Compare/ext_val_occlusion_sensitivity/2023_04_08_113058_DUKE_ResNet101_swarm_learning_bestperforming/results.csv')
# put the results.csv into non-overlapping pairs
pairs = []
for i in range(len(results_files)):
    for j in range(i+1, len(results_files)):
        pairs.append((results_files[i], results_files[j]))
for pair in pairs:
    print(pair)
    df1 = pd.read_csv(pair[0])
    df2 = pd.read_csv(pair[1])

    # add the 'NN' column
    df1['NN'] = df1['NN_pred'].astype(float) >= 0.5
    df2['NN'] = df2['NN_pred'].astype(float) >= 0.5

    # convert 'NN' column to integers (0s and 1s)
    df1['NN'] = df1['NN'].astype(int)
    df2['NN'] = df2['NN'].astype(int)

    # get the prediction scores and true labels
    scores1 = df1['NN_pred']
    labels1 = df1['GT']

    scores2 = df2['NN_pred']
    labels2 = df2['GT']

    # calculate the AUC for both sets of predictions
    auc1 = roc_auc_score(labels1, scores1)
    auc2 = roc_auc_score(labels2, scores2)


    pROC = importr('pROC')

    roc_test = pROC.roc_test

    # R's vector object
    r_labels1 = robjects.FloatVector(labels1)
    r_scores1 = robjects.FloatVector(scores1)

    r_labels2 = robjects.FloatVector(labels2)
    r_scores2 = robjects.FloatVector(scores2)

    # perform ROC analysis
    roc_obj1 = pROC.roc(r_labels1, r_scores1)
    roc_obj2 = pROC.roc(r_labels2, r_scores2)

    # perform DeLong test
    result = roc_test(roc_obj1, roc_obj2)


    print('AUC for model 1: ', auc1)
    print('AUC for model 2: ', auc2)
    '''
    print(result.names)
    result = dict(zip(result.names, list(result)))
    print('Z-statistic: ', result['statistic'])
    print('p-value: ', result['p.value'])
    
    print('====')
    '''
    #print(result)
    print((result.names))
    print('Z-statistic: ', result[4])
    #write the results to a file with the pair names
    with open(test_folder+ '/'+test_folder.split('/')[-1] +'_delong_test_results.txt', 'a') as f:
        name0 = str(pair[0])
        name1 = str(pair[1])
        name0 = name0.split('/')[-1]
        name1 = name1.split('/')[-1]
        # in name0 and name1 replace Host_Sentinal with 40%data, Host_100 with 10%data, Host_101 or Host_103 with 30%data, merged with centralized_model.
        name0 = name0.replace('Host_Sentinal', '40%data')
        name0 = name0.replace('Host_100', '10%data')
        name0 = name0.replace('Host_101', '30%data')
        name0 = name0.replace('Host_103', '30%data')
        name0 = name0.replace('merged', 'centralized_model')
        name1 = name1.replace('Host_Sentinal', '40%data')
        name1 = name1.replace('Host_100', '10%data')
        name1 = name1.replace('Host_101', '30%data')
        name1 = name1.replace('Host_103', '30%data')
        name1 = name1.replace('merged', 'centralized_model')


        f.write(name0 + ' vs ' + name1 + '\n')
        #f.write('AUC for model 1: ' + str(auc1) + '\n')
        #f.write('AUC for model 2: ' + str(auc2) + '\n')
        #f.write('Z-statistic: ' + str(result[4]) + '\n')
        f.write('p-value: ' + str(result[7]))
        #f.write('\n')