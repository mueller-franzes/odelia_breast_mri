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
    for dataset in [ 'UKA']: # 'CAM', 'MHA', 'RSH', 'RUMC', 'UKA', 'UMCU'
        print(f"----------------- {dataset} ---------------")

        path_root = Path('/home/homesOnMaster/gfranzes/Documents/datasets/ODELIA/')/dataset
        path_root_metadata = path_root/'metadata'

        df = pd.read_excel(path_root_metadata/'ODELIA annotation scheme-2.0.xlsx', dtype={'Patient ID':str})
        df = df[11:].reset_index(drop=True) # Remove rows with annotation hints
        df = df.rename(columns={'Patient ID': 'PatientID', 'Type of Lesion':'Lesion'})
        assert not df[['PatientID', 'StudyInstanceUID', 'Lesion']].isna().any().any(), "Missing values detected"
        

        # Define class mapping
        class_mapping = {
            'No lesion': 0,
            'Benign lesion': 1,
            'DCIS': 2,
            'Proliferative with atypia': 2,
            'Invasive Cancer (no special type)': 2, # TODO should invasive cancer be separate class?
            'Invasive Cancer (lobular carcinoma)': 2,
            'Invasive Cancer (all other)':2,
            'not provided': pd.NA
        }

        df_left = df[df['Side'] == "left"] 
        df_left = df_left[['PatientID', 'StudyInstanceUID', 'Side', 'Lesion']]
        df_left.insert(0, 'UID', df_left['StudyInstanceUID'].astype(str)+'_'+df_left['Side'])

        df_left['Class'] = df_left['Lesion'].map(class_mapping)
        df_left = df_left.dropna(subset='Class').reset_index(drop=True) # TODO: Should the entire study be removed?
        df_left = df_left.loc[df_left.groupby('StudyInstanceUID')['Class'].idxmax()]


        df_right = df[df['Side'] == "right"] 
        df_right = df_right[['PatientID', 'StudyInstanceUID', 'Side', 'Lesion']]
        df_right.insert(0, 'UID', df_right['StudyInstanceUID'].astype(str)+'_'+df_right['Side'])

        df_right['Class'] = df_right['Lesion'].map(class_mapping)
        df_right = df_right.dropna(subset='Class').reset_index(drop=True) # TODO: Should the entire study be removed?
        df_right = df_right.loc[df_right.groupby('StudyInstanceUID')['Class'].idxmax()]
   
        
        # ------------------- Merge left and right ----------------------
        df = pd.concat([df_left, df_right]).reset_index(drop=True)
        df['Class'] = df['Class'].astype(int)

        print("Patients", df['PatientID'].nunique())
        print("Studies", df['StudyInstanceUID'].nunique())
        print("Breasts", df['UID'].nunique())
        for class_name, count in df['Class'].value_counts().sort_index().items():
            print(f"Lesion Type {class_name}: {count}")

        df_splits = create_split(df, uid_col='UID', label_col='Class', group_col='PatientID')
        df_splits.to_csv(path_root_metadata/'split.csv', index=False)


        
    