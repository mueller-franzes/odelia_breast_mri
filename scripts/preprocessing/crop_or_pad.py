from pathlib import Path 
import torchio as tio 
import torch
from multiprocessing import Pool

from odelia.data.augmentation.augmentations_3d import CropOrPad

def crop_breast_height(image, margin_top=10):
    "Crop height to 256 and try to cover breast based on intensity localization"
    threshold = int(image.data.float().quantile(0.9))
    foreground = image.data>threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return  tio.Crop((0,0, bottom, top, 0, 0))


def preprocess(path_dir):
    # -------- Settings --------------
    target_spacing = (0.7, 0.7, 3) 
    target_shape = (512, 512, 32)
    ref_img = tio.Resample(target_spacing)(tio.ScalarImage(path_dir/'pre.nii.gz'))
    transform = tio.Compose([
        tio.Resample(ref_img), # Resample to reference image to ensure that origin, direction, etc, fit
        CropOrPad(target_shape, padding_mode='mean'),
        tio.ToCanonical(),
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
        img = transform(img)

        # Crop bottom and top so that height is 256 and breast is preserved  
        img = crop_height(img)

        # Split left and right side 
        for side in ['left', 'right']:
            # Create output directory 
            path_out_dir = path_out/f"{path_dir.relative_to(path_data)}_{side}"
            path_out_dir.mkdir(exist_ok=True, parents=True)

            # Crop left/right side 
            img_side = split_side[side](img)

            # Save 
            img_side.save(path_out_dir/path_img.name)

if __name__ == "__main__":
    path_data = Path('/mnt/sda1/swarm-learning/radiology-dataset/odelia_converted/')
    path_out = Path('/mnt/sda1/swarm-learning/radiology-dataset/odelia_dataset_unilateral_256x256x32/')
    path_out.mkdir(parents=True, exist_ok=True)
    pool = Pool()
    pool.map(preprocess, path_data.iterdir())
    # for path_dir in path_data.iterdir():
    #     preprocess(path_dir)

    