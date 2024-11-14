from pathlib import Path 
import torchio as tio 
import torch
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm


def crop_breast_height(image, margin_top=10):
    "Crop height to 256 and try to cover breast based on intensity localization"
    # threshold = int(image.data.float().quantile(0.9))
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data>threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return  tio.Crop((0,0, bottom, top, 0, 0))


def preprocess(path_dir):
    # -------- Settings --------------
    ref_img = tio.ScalarImage(path_dir/'Pre.nii.gz')
    ref_img = tio.ToCanonical()(ref_img)

    # Spacing 
    target_spacing = (0.7, 0.7, 3) 
    ref_img = tio.Resample(target_spacing)(ref_img)

    # Crop 
    target_shape = (512, 512, 32)

    padding_constant = ref_img.data.min().item() # Ugly workaround: padding_mode='minimum' calculates the minimum per axis, not globally  
    transform = tio.Compose([
        tio.Resample(ref_img), # Resample to reference image to ensure that origin, direction, etc, fit
        tio.CropOrPad(target_shape, padding_mode=padding_constant),
    ])
    crop_height = crop_breast_height(transform(ref_img))     
    split_side = {
        'right': tio.Crop((256, 0, 0, 0, 0, 0)),
        'left': tio.Crop((0, 256, 0, 0, 0, 0)),
    }
    

    for n, path_img in enumerate(path_dir.glob('*.nii.gz')):
        # Read image 
        img = tio.ScalarImage(path_img)

        # Preprocess (eg. Crop/Pad)
        padding_constant = img.data.min().item()
        transform = tio.Compose([
            tio.Resample(ref_img), 
            tio.CropOrPad(target_shape, padding_mode=padding_constant),
        ])
        img = transform(img)

        # Crop bottom and top so that height is 256 and breast is preserved  
        img = crop_height(img)

        # Split left and right side 
        for side in ['left', 'right']:
            # Create output directory 
            path_out_dir = path_root_out_data/f"{path_dir.relative_to(path_root_in_data)}_{side}"
            path_out_dir.mkdir(exist_ok=True, parents=True)

            # Crop left/right side 
            img_side = split_side[side](img)

            # Save 
            img_side.save(path_out_dir/path_img.name)

if __name__ == "__main__":
    for dataset in ['DUKE', ]: # 'CAM', 'MHA', 'RSH', 'RUMC', 'UKA', 'UMCU', 'DUKE'

        path_root = Path('/home/gustav/Documents/datasets/ODELIA/')/dataset
        path_root_in_data = path_root/'data'
    
        path_root_out_data =  path_root/'data_unilateral'
        path_root_out_data.mkdir(parents=True, exist_ok=True)

        
        path_patients = list(path_root_in_data.iterdir())  # Convert the iterator to a list
        
        # Option 1: Multi-CPU 
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(preprocess, path_patients)):
                pass

        # Option 2: Single-CPU 
        # for path_dir in tqdm(path_patients):
        #     preprocess(path_dir)
        
    