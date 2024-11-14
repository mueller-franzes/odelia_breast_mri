from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

path_root = Path('/home/gustav/Documents/datasets/ODELIA/')
path_root_dataset = path_root/'DUKE'
path_root_metadata = path_root_dataset/'metadata'


df = pd.read_excel(path_root_metadata/'Clinical_and_Other_Features.xlsx', header=[0, 1, 2])
df = df[df[df.columns[38]] != 'NC'] # check if cancer is bilateral=1, unilateral=0 or NC 
df = df[[df.columns[0], df.columns[36],  df.columns[38]]] # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
df.columns = ['PatientID', 'Location', 'Bilateral']  # Simplify columns as: Patient ID, Tumor Side
dfs = []
for side in ["left", 'right']:
    dfs.append(pd.DataFrame({
        'PatientID': df["PatientID"].str.split('_').str[2],
        'UID': df["PatientID"] + f"_{side}",
        'Class':df[["Location", "Bilateral"]].apply(lambda ds: int((ds[0] == side[0].upper()) | (ds[1]==1)) , axis=1) } ) )
df = pd.concat(dfs,  ignore_index=True)

df = df.reset_index(drop=True)
splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['UID'], df['Class'], groups=df['PatientID'])):
    df_split = df.copy()
    df_split['Fold'] = fold_i 
    df_trainval = df_split.loc[train_val_idx]
    train_idx, val_idx = list(sgkf2.split(df_trainval['UID'], df_trainval['Class'], groups=df_trainval['PatientID']))[0]
    train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
    df_split.loc[train_idx, 'Split'] = 'train' 
    df_split.loc[val_idx, 'Split'] = 'val' 
    df_split.loc[test_idx, 'Split'] = 'test' 
    splits.append(df_split)
df_splits = pd.concat(splits)


df_splits.to_csv(path_root_metadata/'split.csv', index=False)