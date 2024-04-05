# ODELIA - Breast MRI Classification 
## Description:
Code base for breast MRI classification

<br/>

## Step 0: Setup 
* Clone this repository 
* Make sure that your graphics card driver supports the current CUDA version 11.8. You can check this by running `nvidia-smi`. 
If not, change `pytorch-cuda=11.8` in the [environment.yaml](environment.yaml) file or update your driver.
* Run: `conda env create -f environment.yaml` (can take up to 30min)
* Run `conda activate odelia`

## Step 1: Download [DUKE](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) Dataset
* [Download](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) files form The Cancer Imaging Archive (TCIA)
* Folder should look like:
    ```bash
    ├── dataset_raw
    │   ├── Breast_MRI_001
    │   │   ├── ...
    │   ├── Breast_MRI_002
    │   |   ├── ...
    ├── Breast-Cancer-MRI-filepath_filename-mapping.xlsx
    ├── Clinical_and_Other_Features.xlsx
    ```

## Step 2: Prepare Data 
* Got to  [scripts/preprocessing/data_preparation.py](scripts/preprocessing/data_preparation.py)
* Specify path to downloaded files as `path_root=...` 
* Run script - It will store DICOM files as NIFTI files 
* Got to [scripts/preprocessing/crop_or_pad.py](scripts/preprocessing/crop_or_pad.py)
* Specify path to downloaded files as `path_root=...` 
* Run script - It will split breast into left and right side and resample to uniform shape

## Step 3: Run Training
* Run Script: [scripts/main_train.py](scripts/main_train.py)
* Make sure to set `path_root`

## Step 4: Predict & Evaluate Performance
* Run Script: [scripts/main_predict.py](scripts/main_predict.py)
* Set `path_run` to root directory of latest model 