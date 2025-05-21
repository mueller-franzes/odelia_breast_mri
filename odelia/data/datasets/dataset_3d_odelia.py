from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  
import numpy as np 

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, ZNormalization, CropOrPad


class ODELIA_Dataset3D(data.Dataset):
    PATH_ROOT = Path('/home/homesOnMaster/gfranzes/Documents/datasets/ODELIA/')
    ALL_INSTITUTIONS = ['CAM', 'MHA', 'RSH', 'UKA', 'UMCU']  # exclude 'RUMC' to keep it for testing
    DATA_DIR = {
        "original": "data",
        "unilateral": "data_unilateral"
    }
    META_DIR = {
        "original": "metadata",
        "unilateral": "metadata_unilateral"
    }
    # CLASS_LABELS = {
    #     'original': {
    #         'Lesion_Left': ['No', 'Benign', 'Malignant'],
    #         'Lesion_Right': ['No', 'Benign', 'Malignant'],
    #     },
    #     'unilateral': {
    #         'Lesion': ['No', 'Benign', 'Malignant'],
    #     }
    # }

    _CLS = [
        # 'Fibroadenoma',
        # 'Adenosis',
        # 'LymphNode',
        # 'Cyst',
        # 'FatNecrosis',
        # # 'Hyperplasia', # To few  
        # 'DuctEctasia',
        # 'Papilloma',
        # # 'Hamartoma', # Too few 
        # 'Scar',
        # 'DCIS',
        'Carcinoma',
    ]
    CLASS_LABELS = {
        'original': {
            f'{cls_str}_{side}': ['Absence', 'Presence'] for cls_str in _CLS for side in ['left', 'right']
        },
        'unilateral': {
            f'{cls_str}': ['Absence', 'Presence'] for cls_str in _CLS
        },
    }

    def __init__(
            self,
            path_root=None,
            institutions=None,
            fold = 0,
            binary=True, # If True, binary labels otherwise ordinal 
            labels=None, # None = all labels or  list of labels 
            config="original", # original, unilateral 
            split= None,
            fraction=None,
            transform = None,
            random_flip = False,
            random_rotate=False,
            random_inverse=False,
            noise=False, 
            to_tensor = True,
            
        ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.split = split 
        self.binary = binary
        self.config = config 
        self.class_labels = self.CLASS_LABELS[config]
        self.meta_dir = self.META_DIR[config]
        self.data_dir = self.DATA_DIR[config]
        self.labels = list(self.class_labels.keys()) if labels is None else labels 
        self.class_labels_num = [len(self.class_labels[l]) for l in self.labels] # For CORN Loss -1 
        
        if (institutions is None) or (institutions == "ODELIA"):
            institutions = self.ALL_INSTITUTIONS
        elif isinstance(institutions, str):
            institutions = [institutions]
        self.institutions = institutions
        
        flip_axes = (0,1) if config == "original" else (0,1,2) # Do not flip horizontal axis 2, otherwise labels incorrect 
        if transform is None: 
            self.transform = tio.Compose([
                tio.ToCanonical() if config == "original" else tio.Lambda(lambda x: x),
                tio.Resample((0.7, 0.7, 3)) if config == "original" else tio.Lambda(lambda x: x),
                
                tio.Flip((1,0)), # Just for viewing, otherwise upside down
                CropOrPad((448, 448, 32), random_center=random_rotate) if config == "original" else CropOrPad((224, 224, 32), random_center=random_rotate),

                ZNormalization(per_channel=True, per_slice=False, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)), 

                tio.OneOf([
                    #tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                    tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                    tio.RandomFlip(flip_axes) if random_flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                ]),
                tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if random_inverse else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)             
            ])
        else:
            self.transform = transform
 

        # Get split  
        dfs = []
        for institution in self.institutions:
            path_metadata = self.path_root/institution/self.meta_dir 
            df = self.load_split(path_metadata/'split.csv', fold=fold, split=split, fraction=fraction)
            # df = self.load_split('scripts/playground/debora_split.csv', fold=fold, split=split, fraction=fraction)
            df['Institution'] = institution

            # Verify files exist
            # uids = self.run_item_crawler(self.path_root/institution/'data_unilateral')
            # df = df[df['UID'].isin(uids)]

            # Merge with annotations 
            df_anno = pd.read_csv(path_metadata/'annotation.csv', dtype={'UID':str, 'PatientID':str})
            df = df.merge(df_anno, on='UID', how='inner')

            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)



        self.item_pointers = df.index.tolist()
        self.df = df 

        


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]      
        uid = item['UID']
        institution = item['Institution']

        target = np.stack(item[self.labels].values)

        # Use only binary label (Cancer yes/no)
        # NOTE: 0=No Lesion, 1=Benign Lesion, 2=Malignant Lesion
        # NOTE: Duke already binary: 0=No or benign Lesion, 1=Malignant Lesion
        if (institution not in ["DUKE", "UKA_all_new"]) and self.binary:
            target = (target == 2).astype(int)
    
        path_folder = self.path_root/institution/self.data_dir/uid
        # img = self.load_img(path_folder/'Pre.nii.gz') 
        # img = self.load_img(path_folder/'T2.nii.gz') 
        # img = self.load_img(path_folder/'Sub_1.nii.gz') 
        # img = self.load_img([path_folder/f'{name}.nii.gz' for name in [ 'Pre', 'T2']]) 
        img = self.load_img([path_folder/f'{name}.nii.gz' for name in [ 'Pre', 'Sub_1']]) 
        # img = self.load_img([path_folder/f'{name}.nii.gz' for name in [ 'Pre', 'Post_1']]) 
        # img = self.load_img([path_folder/f'{name}.nii.gz' for name in [ 'Pre', 'Sub_1', 'T2']]) 
        # img = self.load_img([path_folder/f'{name}.nii.gz' for name in [ 'Pre', 'Sub_1', 'Sub_2', 'Sub_3', 'Sub_4', 'T2']]) 
        img = self.transform(img)


        return {'uid':uid, 'source': img, 'target':target}


    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        # WARNING: PatientID must be read as string otherwise leading zeros are cut off
        df = pd.read_csv(filepath_or_buffer, dtype={'UID':str})
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
    
    @classmethod
    def run_item_crawler(cls, path_root, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir() ]