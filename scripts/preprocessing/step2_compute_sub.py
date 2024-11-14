from pathlib import Path 
import SimpleITK as sitk
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm


def process(path_patient):
    # Compute subtraction image
    # Note: if dtype not specified, data is read as uint16 -> subtraction wrong
    dyn0_nii = sitk.ReadImage(str(path_patient/'Pre.nii.gz'), sitk.sitkInt16)
    dyn1_nii = sitk.ReadImage(str(path_patient/'Post_1.nii.gz'), sitk.sitkInt16)
    dyn0 = sitk.GetArrayFromImage(dyn0_nii)
    dyn1 = sitk.GetArrayFromImage(dyn1_nii)
    sub = dyn1-dyn0
    sub = sub-sub.min() # Note: negative values causes overflow when using uint 
    sub = sub.astype(np.uint16)
    sub_nii = sitk.GetImageFromArray(sub)
    sub_nii.CopyInformation(dyn0_nii)
    sitk.WriteImage(sub_nii, str(path_patient/'Sub.nii.gz'))





if __name__ == "__main__":
    path_root = Path('/home/gustav/Documents/datasets/ODELIA/')
    for dataset in ['DUKE', ]: # 'CAM', 'MHA', 'RSH', 'RUMC', 'UKA', 'UMCU', 'DUKE'
        path_data = path_root/dataset/'data'

        files = path_data.iterdir()  

        # Option 1: Multi-CPU 
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(process, files)):
                pass

        # Option 2: Single-CPU 
        # for path_dir in tqdm(files):
        #     process(path_dir)
        
    