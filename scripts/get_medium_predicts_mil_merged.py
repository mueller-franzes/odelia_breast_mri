##!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import numpy as np
import pandas as pd
import glob

parent_folder = '/mnt/sda1/Duke Compare/mil/ext_val_results/merged/'
#for node in ['Host_100/', 'Host_101/', 'merged/','Host_Sentinal/']:
    #main_folder = parent_folder + node
model_list = ['transformer', 'timmViT', 'attmil']
for model in model_list:
    #folder = main_folder + model

    # get all the subfolders in the folder
    #model_folders = glob.glob(folder+'*')
    #print(model_folders)
    #train_type_list = ['_swarm_learning', '_local_compare']
#for train_type in train_type_list:
    # match folder ends with ResNet101_local_compare
    folders = glob.glob(parent_folder+''+model)
    print(folders)
    # get the results.csv
    csv_files = []

    for subfolder in folders:
        #print(subfolder)
        csv_file = glob.glob(subfolder+'/*/*patient-preds.csv')
        if len(csv_file) > 0:
            #print(csv_file)
            csv_files.append(csv_file[0])

    # create an empty dictionary to store DataFrames
    dfs = {}

    # loop through the list of CSV files
    for file in csv_files:
        #print(file)
        # read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # add the DataFrame to the dictionary, using the file name as the key
        dfs[file] = df

    print(len(dfs))
    if len(dfs) == 0:
        continue
    # use a DataFrame method to align all DataFrames by 'uid'
    dfs_aligned = pd.concat([df.set_index('PATIENT') for df in dfs.values()], axis=1, keys=dfs.keys())

    # calculate the median of the 'NN_pred' values for each row
    medians = dfs_aligned.xs('Malign_old_1', axis=1, level=1).median(axis=1)

    # get the first DataFrame (it doesn't matter which one since they're now aligned)
    df_first = list(dfs.values())[0]

    # replace the 'Malign_old_1' column with the median values
    df_first = df_first.set_index('PATIENT')
    df_first['Malign_old_1'] = medians

    # rename the columns
    df_first.rename(columns={
        'Malign_old': 'GT',
        'pred': 'NN',
        'Malign_old_1': 'NN_pred'
    }, inplace=True)

    # reset index and rename 'PATIENT' to 'uid'
    df_first = df_first.reset_index().rename(columns={'PATIENT': 'uid'})

    # make 'uid' the first column
    df_first = df_first[['uid', 'GT', 'NN', 'NN_pred']]

    # write the DataFrame to a new CSV file
    df_first.reset_index().rename(columns={'PATIENT': 'uid'}).to_csv(
        parent_folder + model.strip("/") + '_result.csv', index=False)

    # write the DataFrame to a new CSV file
    df_first.to_csv(parent_folder+model.strip("/")+'_result.csv', index=False)
