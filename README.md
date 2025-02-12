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
* <b> Skip to Step 3 if you don't want to use the DUKE dataset </b>

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

## Step 2: Prepare Data ([DUKE](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/))
* Specify the path to the parent folder as `path_root=...` and `dataset=DUKE` in the following scripts
* Run  [scripts/preprocessing/duke/step1_dicom2nifti.py](scripts/preprocessing/duke/step1_dicom2nifti.py) - It will store DICOM files as NIFTI files in a new folder `data`
* Run [scripts/preprocessing/step2_compute_sub.py](scripts/preprocessing/step2_compute_sub.py) - computes the subtraction image
* Run [scripts/preprocessing/step3_unilateral.py](scripts/preprocessing/step3_unilateral.py) - splits breasts into left and right side and resamples to uniform shape. The result is stored in a new folder `data_unilateral`
* Run [scripts/preprocessing/duke/step4_create_split.py](scripts/preprocessing/duke/step4_create_split.py) - creates a stratified five-fold split and stores the result in `metadata/split.csv`


<br>

## Step 3: Prepare Data ([ODELIA](https://odelia.ai/))
* Create a folder with the initials of your institution e.g. `ABC`
* Place your DICOM files in a subfolder `data_raw`
* Create a folder `metadata` with the following file inside:
    * Challenge: `annotation.xlsx` 
    * Local Training: `ODELIA annotation scheme-2.0.xlsx` 
* Overwrite [scripts/preprocessing/odelia/step1_dicom2nifti.py](scripts/preprocessing/odelia/step1_dicom2nifti.py). It should create a subfolder `data` and subfolders with files named as `T2.nii.gz`, `Pre.nii.gz`, `Post_1.nii.gz`, `Post_2.nii.gz`, etc.
The subfolder should be labeled as follows:  
    * Challenge: Folders must have the same name as the entries in the `ID` column of the `annotation.xlsx` file.  
    * Local Training: Folders must have the same name as the entries in the `StudyInstanceUID` column of the `ODELIA annotation scheme-2.0.xlsx` file. 
* Run [scripts/preprocessing/step2_compute_sub.py](scripts/preprocessing/step2_compute_sub.py) - computes the subtraction image
* Run [scripts/preprocessing/step3_unilateral.py](scripts/preprocessing/step3_unilateral.py) - splits breasts into left and right side and resamples to uniform shape. The result is stored in a new folder `data_unilateral`
* To create a five-fold stratified split and store the result in `metadata/split.csv`, run the following script:
  * Challenge:  [scripts/preprocessing/odelia/step4_create_split_challenge.py](scripts/preprocessing/odelia/step4_create_split_challenge.py) 
  * Local Training: [scripts/preprocessing/odelia/step4_create_split.py](scripts/preprocessing/odelia/step4_create_split.py) 

* The final folder structure should look like:
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
    ├── data_unilateral
    │   ├── ID_001_left
    │   ├── ID_001_right
    ├── metadata
    |   ├── annotation.xlsx
    |   ├── split.csv
    ```

<br>

## Step 4: Run Training
* Specify path to downloaded folder as `PATH_ROOT=` in [dataset_3d_odelia.py](odelia/data/datasets/dataset_3d_odelia.py)
* Run Script: [scripts/main_train.py --institution DUKE](scripts/main_train.py)


## Step 5: Predict & Evaluate Performance
* Run Script: [scripts/main_predict.py](scripts/main_predict.py)
* Set `path_run` to root directory of latest model 
