# ODELIA - Breast MRI Classification 
## Description:
Code base for breast MRI classification

<br/>

## Step 0: Setup 
* Clone this repository 
* Make sure that your graphics card driver supports the current CUDA version 12.1. You can check this by running `nvidia-smi`. 
If not, change `pytorch-cuda=12.1` in the [environment.yaml](environment.yaml) file or update your driver.
* Run: `conda env create -f environment.yaml`
* Run `conda activate odelia`

## Step 1: Download [DUKE](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) Dataset
* Create a folder `DUKE` with a subfolder `data_raw`
* [Download](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) files form The Cancer Imaging Archive (TCIA) into  `data_raw`
* Make sure to download the dataset in the "classical" structure (PatientID - StudyInstanceUID - SeriesInstanceUID) 
* Place all tables in a folder "metadata" 
* The folder structure should look like:
    ```bash
    DUKE
    ├── data_raw
    │   ├── Breast_MRI_001
    │   │   ├── 1.3.6.1.4.1.14519
    |   |   |   ├── 1.3.6.1.4.1.14519.5.2.1.10
    |   |   |   ├── 1.3.6.1.4.1.14519.5.2.1.17
    │   ├── Breast_MRI_002
    │   |   ├── ...
    ├── metadata
    |   ├── Breast-Cancer-MRI-filepath_filename-mapping.xlsx
    |   ├── Clinical_and_Other_Features.xlsx
    ```

## Step 2: Prepare Data 
* Specify the path to the parent folder as `path_root=...` and `dataset=DUKE` in the following scripts
* Run  [scripts/preprocessing/duke/step1_dicom2nifti.py](scripts/preprocessing/duke/step1_dicom2nifti.py) - It will store DICOM files as NIFTI files in a new folder `data`
* Run [scripts/preprocessing/step2_compute_sub.py](scripts/preprocessing/step2_compute_sub.py) - computes the subtraction image
* Run [scripts/preprocessing/step3_unilateral.py](scripts/preprocessing/step3_unilateral.py) - splits breasts into left and right side and resamples to uniform shape. The result is stored in a new folder `data_unilateral`
* Run [scripts/preprocessing/duke/step4_create_split.py](scripts/preprocessing/duke/step4_create_split.py) - creates a stratified five-fold split and stores the result in `metadata/split.csv`

## Step 3: Run Training
* Specify path to downloaded folder as `PATH_ROOT=` in [dataset_3d_odelia.py](odelia/data/datasets/dataset_3d_odelia.py)
* Run Script: [scripts/main_train.py --institution DUKE](scripts/main_train.py)


## Step 4: Predict & Evaluate Performance
* Run Script: [scripts/main_predict.py](scripts/main_predict.py)
* Set `path_run` to root directory of latest model 

<br>

## Own Dataset
* Create a folder with the initials of your institution e.g. `ABC`
* Place your DICOM files in a subfolder `data_raw`
* Overwrite [scripts/preprocessing/odelia/step1_dicom2nifti.py](scripts/preprocessing/odelia/step1_dicom2nifti.py). It should create a subfolder `data` and subfolders labeled as $PatientID with files named as `Pre.nii.gz`, `Post_1.nii.gz`, `Post_2.nii.gz` etc 
* Create a folder `metadata` with your `annotation.xlsx` file inside 
* Follow instructions at `Step 2` but use the scripts from the `odelia` folder  
* The folder structure should look like:
    ```bash
    ABC
    ├── data_raw
    ├── data
    │   ├── ID_001
    │   │   ├── Pre.nii.gz
    |   |   ├── Post_1.nii.gz
    |   |   ├── Post_2.nii.gz
    │   ├── ID_002
    │   |   ├── ...
    ├── metadata
    |   ├── annotation.xlsx
    ```