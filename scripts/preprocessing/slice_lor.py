#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os

import numpy as np
import SimpleITK as sitk
from typing import Iterable, Optional, Sequence, Union
import sys
import pandas as pd
from pathlib import Path
import cv2
import tqdm

def getslice(df: pd.DataFrame,
    odir: Union[str, Path], imtype: str):
    odir.mkdir(parents=True, exist_ok=True)
    imfile = Path(df[imtype])
    #maskfile = Path(df['MASK'])
    print('-----------------')
    #print('Processing image: ', maskfile)
    image = sitk.ReadImage(str(imfile))
    im_arr = sitk.GetArrayFromImage(image)
    #get the slice with more pixels corresponding to the tumor
    for n in range(int(np.shape(im_arr)[0])):
        plane_z = np.array(im_arr[n,:,:])
        final_filename= imfile.stem+'_{}.jpg'.format(n)
        (odir/imfile.parent.name).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(odir/imfile.parent.name/final_filename),plane_z)

odir = Path('/mnt/sda1/Oliver/data_slices_sub')
dir_path = Path('/mnt/sda1/Oliver/data')
# get all the files in the directory ends with .nii
#recursivly get all the files in the directory
#post_1_files = [Path(f) for f in dir_path.glob('**/*') if f.is_file() and Path(f).name == 'post_1.nii.gz']
#pre_files = [Path(f) for f in dir_path.glob('**/*') if f.is_file() and Path(f).name == 'pre.nii.gz']
#post_1_files = [Path(f) for f in os.listdir(dir_path) if f == 'post_1.nii.gz']
#pre_files = [Path(f) for f in os.listdir(dir_path) if f == 'pre.nii.gz']

#print(post_1_files)
#print(pre_files)
#files = post_1_files + pre_files
sub_files = [Path(f) for f in dir_path.glob('**/*') if f.is_file() and Path(f).name == 'Sub.nii.gz']
#print(sub_files)
for imfile in (sub_files):
    imfile = Path(imfile)
    #print(imfile)
    patient_id = str(imfile.parent).split('/')[-1]
    #print(patient_id)
    image = sitk.ReadImage(imfile)
    im_arr = sitk.GetArrayFromImage(image)
    full_patient_id = Path('Breast_MRI_' + patient_id)
    # get the slice with more pixels corresponding to the tumor
    for n in range(int(np.shape(im_arr)[0])):
        plane_z = np.array(im_arr[n, :, :])
        final_filename = imfile.stem.split('.')[0] + '_{}.jpg'.format(n)
        #print(odir / patient_id / final_filename)
        (odir / full_patient_id).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(odir / full_patient_id / final_filename), plane_z)
        
