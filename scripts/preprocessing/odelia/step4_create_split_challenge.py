from pathlib import Path 
import pandas as pd 
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold



def create_split(df, uid_col='UID', label_col='Label', group_col='PatientID'):
    df = df.reset_index(drop=True)
    splits = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df[uid_col], df[label_col], groups=df[group_col])):
        df_split = df.copy()
        df_split['Fold'] = fold_i 
        df_trainval = df_split.iloc[train_val_idx]
        train_idx, val_idx = list(sgkf2.split(df_trainval[uid_col], df_trainval[label_col], groups=df_trainval[group_col]))[0]
        train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
        df_split.loc[train_idx, 'Split'] = 'train' 
        df_split.loc[val_idx, 'Split'] = 'val' 
        df_split.loc[test_idx, 'Split'] = 'test' 
        splits.append(df_split)
    df_splits = pd.concat(splits)
    return df_splits 



if __name__ == "__main__":
    for dataset in [ 'CAM']: # 'CAM', 'MHA', 'RSH', 'RUMC', 'UKA', 'UMCU'
        print(f"----------------- {dataset} ---------------")

        path_root = Path('/home/gustav/Documents/datasets/ODELIA/')/dataset
        path_root_metadata = path_root/'metadata'

        df = pd.read_excel(path_root_metadata/'annotation.xlsx', dtype={'ID':str})
        df = df.rename(columns={'ID': 'PatientID'})

        if dataset == 'CAM':
            df['PatientID'] = df['PatientID'].str.upper()
        if dataset == 'UMCU':
            df['PatientID'] = df['PatientID'].astype(str).str.zfill(10)



        # Define lesion severity order
        severity_order = {
            'No lesion': 0,
            'Benign lesion': 1,
            'Malignant lesion (DCIS)': 2,
            'Malignant lesion (unknown)': 3,
            'Malignant lesion (Invasive)': 4,
        }
        

        df_left = df[['PatientID',	'Left side']]
        df_left = df_left.rename(columns={'Left side': 'Lesion'})
        df_left.insert(1, 'Side', 'left')
        df_left.insert(0, 'UID', df_left['PatientID'].astype(str)+'_'+df_left['Side'])

        df_left = df_left.dropna(subset='Lesion').reset_index(drop=True)
        df_left['Severity'] = df_left['Lesion'].map(severity_order)
        df_left = df_left.loc[df_left.groupby('PatientID')['Severity'].idxmax()]
        df_left = df_left.drop(columns=['Severity'])

        df_right = df[['PatientID', 'Right side']]
        df_right = df_right.rename(columns={'Right side': 'Lesion'})
        df_right.insert(1, 'Side', 'right')
        df_right.insert(0, 'UID', df_right['PatientID'].astype(str)+'_'+df_right['Side'])

        df_right = df_right.dropna(subset='Lesion').reset_index(drop=True)
        df_right['Severity'] = df_right['Lesion'].map(severity_order)
        df_right = df_right.loc[df_right.groupby('PatientID')['Severity'].idxmax()]
        df_right = df_right.drop(columns=['Severity'])
        
        
        df = pd.concat([df_left, df_right]).reset_index(drop=True)
        print("Patients", df['PatientID'].nunique())
        assert len(df) == 2*df['PatientID'].nunique(), "Number of Lesions must be 2* Number of Patients"
        

        df['Class'] = df['Lesion'].map({'No lesion':0, 'Benign lesion':1, 'Malignant lesion (DCIS)': 2, 'Malignant lesion (Invasive)':2, 'Malignant lesion (unknown)':2})
        print(df['Class'].value_counts(dropna=False))

        df_splits = create_split(df, uid_col='UID', label_col='Class', group_col='PatientID')
        df_splits.to_csv(path_root_metadata/'split.csv', index=False)


        
    