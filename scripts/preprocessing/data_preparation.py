

from pathlib import Path 
import pydicom
import logging 
import sys 

from tqdm import tqdm
import SimpleITK as sitk 
import json 
import numpy as np 
import pandas as pd 


# Setting 
path_root = Path('/mnt/hdd/datasets/breast/DUKE/')

path_root_in = path_root/'dataset_raw'
path_root_out = path_root/'dataset'
path_root_out.mkdir(parents=True, exist_ok=True)


# Logging 
path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
s_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler(path_log_file, 'w')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[s_handler, f_handler])



# Extract these DICOM keys and save them as a separate JSON file.
metadatakeys = [ 
    'AcquisitionMatrix', 'AccessionNumber', 'AcquisitionDate', 'AcquisitionDuration', 'AcquisitionTime', 
    'Allergies', 'Columns', 'DeviceSerialNumber', 'EchoNumbers', 'EchoTime', 'EchoTrainLength', 'FlipAngle', 
    'ImagingFrequency', 'InPlanePhaseEncodingDirection', 'InstitutionName', 'MRAcquisitionType', 'MagneticFieldStrength', 
    'Manufacturer', 'ManufacturerModelName', 'Modality', 'NumberOfAverages', 'NumberOfPhaseEncodingSteps', 'NumberOfSlices', 
    'PatientAge', 'PatientBirthDate', 'PatientID', 'PatientName', 'PatientSex', 'PatientWeight', 'ParallelAcquisitionTechnique', 
    'PercentPhaseFieldOfView', 'PercentSampling', 'PerformingPhysicianName', 'PulseSequenceName', 'PregnancyStatus', 
    'ProtocolName', 'ReceiveCoilName', 'ReferringPhysicianName', 'RepetitionTime', 'RequestedProcedureID', 'Rows', 
    'PixelSpacing', 'SOPInstanceUID', 'SpacingBetweenSlices', 'ScanningSequence', 'SequenceName', 'SequenceVariant', 
    'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber', 'SliceThickness', 'StudyID', 'StudyInstanceUID'
]


# Get sequence names (warning: time intensive)
df_path2name = pd.read_excel(path_root/'Breast-Cancer-MRI-filepath_filename-mapping.xlsx')
seq_paths = df_path2name['original_path_and_filename'].str.split('/')
df_path2name['UID'] = seq_paths.apply(lambda x:int(x[1].rsplit('_', 1)[1]))
df_path2name['SequenceName'] = seq_paths.apply(lambda x:x[2])
df_path2name['SeriesInstanceUID'] = df_path2name['classic_path'].str.split('/').apply(lambda x:x[3]) # StudyInstanceUID/SeriesInstanceUID
df_path2name = df_path2name.drop_duplicates(subset=['UID', 'SequenceName'], keep='first')
df_path2name[['UID', 'SequenceName', 'SeriesInstanceUID' ]].to_csv(path_root/'filepath2sequence.csv', index=False)
# df_path2name = pd.read_csv(path_root/'filepath2sequence.csv')

reader = sitk.ImageSeriesReader()
for case_i, path_dir in enumerate(tqdm(list(path_root_in.iterdir()))):
    # Only proceed if path is pointing to a folder
    if not path_dir.is_dir():
        continue

    case_id = path_dir.name.rsplit('_', 1)[1]  
    logger.debug(f"Case ID: {case_id}, Number {case_i}")
    
    # Every case should have exactly one folder with several subfolder 
    sub_dirs = [ sub_dir for sub_dir in path_dir.iterdir() if sub_dir.is_dir()]
    if len(sub_dirs) != 1:
        raise "Expecting exactly one subdirectory per case !"
    path_dir_series = sub_dirs[0]

    # Create output folder 
    path_out_dir = path_root_out/case_id
    path_out_dir.mkdir(exist_ok=True)

    # Iterate over all sequences (post_1, post_2, ..., pre, T1)
    df_sequences = df_path2name[df_path2name['UID'] == int(case_id)]
    for path_seq_dir in path_dir_series.iterdir():
        # Get one DICOM file  
        path_dicom_file_1 = path_seq_dir/'1-001.dcm' 
        if not path_dicom_file_1.is_file():
            path_dicom_file_1 = path_seq_dir/'1-01.dcm'   

        # Get Meta-Data (DICOM Tags)
        ds = pydicom.dcmread(path_dicom_file_1)
        metadata = {key:str(getattr(ds, key, 'NaN'))  for key in metadatakeys}

        # Get Sequence Name 
        matches = df_sequences[df_sequences['SeriesInstanceUID'] == metadata['SeriesInstanceUID']]
        if len(matches) != 1:
            raise "There should be exactly one matching sequence!"
        seq_name = matches.iloc[0]['SequenceName']
        logger.debug(f"Write {seq_name} to disk")

        # -------------- Read+Write Image ---------------
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_seq_dir))
        reader.SetFileNames(dicom_names) 
        img_nii = reader.Execute()
        sitk.WriteImage(img_nii, str(path_out_dir/(seq_name+'.nii.gz')) )
        

        # --------------- Write Meta-Data (DIOCM Tags) -----------
        # Add number of slices and voxels size
        metadata.update({'_NumberOfSlices': img_nii.GetDepth()})
        metadata.update({'_VoxelSize':img_nii.GetSpacing()})

        path_out_file = path_out_dir/(seq_name+'.json')
        with open(path_out_file, 'w') as f:
            json.dump(metadata, f) 


  

    # Compute subtraction image
    logger.debug(f"Compute and write sub to disk")
    dyn0_nii = sitk.ReadImage(str(path_out_dir/'pre.nii.gz'), sitk.sitkInt16) # Note: if dtype not specified, data is read as uint16 -> subtraction wrong
    dyn1_nii = sitk.ReadImage(str(path_out_dir/'post_1.nii.gz'), sitk.sitkInt16)
    dyn0 = sitk.GetArrayFromImage(dyn0_nii)
    dyn1 = sitk.GetArrayFromImage(dyn1_nii)
    sub = dyn1-dyn0
    sub = sub-sub.min() # Note: negative values causes overflow when using uint 
    sub = sub.astype(np.uint16)
    sub_nii = sitk.GetImageFromArray(sub)
    sub_nii.CopyInformation(dyn0_nii)
    sitk.WriteImage(sub_nii, str(path_out_dir/'sub.nii.gz'))


    # Compute resampled T1-weighted image
    logger.debug(f"Compute and write resampled T1 to disk")
    t1_nii = sitk.ReadImage(str(path_out_dir/'T1.nii.gz'), sitk.sitkInt16)
    t1_resampled_nii = sitk.Resample(t1_nii, dyn0_nii, sitk.Transform(), sitk.sitkBSpline, 0, dyn0_nii.GetPixelID()) # Interpolation: sitk.sitkBSpline, sitk.sitkLinear
    sitk.WriteImage(t1_resampled_nii, str(path_dir/'T1_resampled.nii.gz'))

