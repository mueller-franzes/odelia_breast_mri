from pathlib import Path 
import pandas as pd 
from odelia.data.datasets import SimpleDataset3D
from tqdm import tqdm
import numpy as np 

class UKA_Dataset3D(SimpleDataset3D):
    def __init__(self, path_root=Path('/mnt/hdd/Share/datasets/breast/UKA/data_lr_256/'), mode='train', item_pointers=[], crawler_glob='*.nii.gz', transform=None, image_resize=None, flip=False, image_crop=(256, 256, 32), random_center=False, norm='znorm_clip', noise=False, to_tensor=True):
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, random_center, norm, noise, to_tensor)
        self.df = self.load_split(mode=mode)
        self.item_pointers = [n for n in self.df.index if n in self.item_pointers]
        # self.item_pointers = self.df.index


        # self.cached_items = {
        #     uid: self.load_item([self.path_root/uid/name for name in [ 'Sub.nii.gz']]) # 'Dyn_0.nii.gz',
        #     for uid in self.item_pointers}
        # for key,value in tqdm(self.cached_items.items(), desc="Loading to cache"):
        #     value.load()

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        path_item = [self.path_root/uid/name for name in [ 'Sub.nii.gz']] # 'Dyn_0.nii.gz',
        # img = self.cached_items[uid] 
        img = self.load_item(path_item)
        target = int(self.df.loc[uid]['Malign'])
        return {'uid':uid, 'source': self.transform(img), 'target':target}
    
    @classmethod
    def run_item_crawler(cls, path_root, crawler_ext, **kwargs):
        # return []
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir() ]
    
    def load_split(self, mode):
        df = pd.read_csv(Path.cwd()/'splits'/f'split_{mode}.csv')
        df['uid'] = df['AnforderungsNrE'].astype(str)+'_'+df['Side'].str.lower()
        df = df.set_index('uid', drop=True)
        return df 