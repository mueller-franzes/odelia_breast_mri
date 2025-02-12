from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

path_root = Path('/home/gustav/Documents/datasets/ODELIA/')
path_root_dataset = path_root/'UKA_all'
path_root_metadata = path_root_dataset/'metadata'


df = pd.read_excel(path_root_metadata/'annotation_regex.xlsx')
assert len(df[df.duplicated(subset='UID', keep=False)])==0, "Duplicates exist"

df['DCISoderKarzinom'] = df[df.columns[-1]] | df[df.columns[-2]]
print(f"Text available for {len(df)} cases")

# Include only examinations were MR image is available
uids = [path.name for path in (path_root_dataset/'data_unilateral').iterdir()]
print(f"Image available for {len(uids)} cases")

# Merge
df = df[df['UID'].isin(uids)].reset_index(drop=True)
print(f"Text and Image available for {len(df)} cases")

# label = df.columns[6]
for label in df.columns[6:]:
    print(label)
    df = df.reset_index(drop=True)
    splits = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['UID'], df[label], groups=df['PNR'])):
        df_split = df.copy()
        df_split['Fold'] = fold_i 
        df_trainval = df_split.loc[train_val_idx]
        train_idx, val_idx = list(sgkf2.split(df_trainval['UID'], df_trainval[label], groups=df_trainval['PNR']))[0]
        train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
        df_split.loc[train_idx, 'Split'] = 'train' 
        df_split.loc[val_idx, 'Split'] = 'val' 
        df_split.loc[test_idx, 'Split'] = 'test' 
        splits.append(df_split)
    df_splits = pd.concat(splits)


    df_splits.to_csv(path_root_metadata/f'split_regex_{label}.csv', index=False)