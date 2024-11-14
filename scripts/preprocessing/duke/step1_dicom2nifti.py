

from pathlib import Path 
import logging  
import pandas as pd 
from multiprocessing import Pool

import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk 





# Logging 
# path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
# s_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler(path_log_file, 'w')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[s_handler, f_handler])



def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def series2nifti(series_info):
    seq_name, path_series = series_info 
    path_series = path_root_data/Path(path_series)
    if not path_series.is_dir():
        logger.warning(f"Directory not found: {path_series}:")
        return 
    
    try:
        # Read DICOM
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series))
        reader.SetFileNames(dicom_names) 
        img_nii = reader.Execute()

        # Read Metadata 
        ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
        metadata = dataset2dict(ds)
        
        # Create output folder 
        path_out_dir = path_root_out_data/path_series.parts[-3]
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write 
        filename = seq_name
        logger.info(f"Writing file: {filename}:")
        path_file = path_out_dir/f'{seq_name}.nii.gz'
        sitk.WriteImage(img_nii, path_file )

        metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        return metadata


    except Exception as e:
        logger.warning(f"Error in: {path_series}")
        logger.warning(str(e))






if __name__ == "__main__":
    # Setting 
    path_root = Path('/home/gustav/Documents/datasets/')
    path_root_dataset = path_root/'DUKE'

    path_root_data = path_root_dataset/'data_raw/'
    path_root_metadata = path_root_dataset/'metadata'

    path_root_out_data = path_root_dataset/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    # Init reader 
    reader = sitk.ImageSeriesReader()

    # Note: Contains path to every single dicom file 
    # WARNING: reading this .xlsx file takes some time 
    df_path2name = pd.read_excel(path_root_metadata/'Breast-Cancer-MRI-filepath_filename-mapping.xlsx') 
    df_path2name = df_path2name[df_path2name.columns[:4]].copy()
    seq_paths = df_path2name['original_path_and_filename'].str.split('/')
    df_path2name['PatientID'] = seq_paths.apply(lambda x:int(x[1].rsplit('_', 1)[1]))
    df_path2name['SequenceName'] = seq_paths.apply(lambda x:x[2])
    df_path2name['classic_path'] = df_path2name['classic_path'].str.rsplit('/', n=1).str[0] # remove xx.dcm 
    df_path2name['classic_path'] = df_path2name['classic_path'].str.split('/', n=1).str[1] # remove Duke-Breast-Cancer-MRI/
    df_path2name = df_path2name.drop_duplicates(subset=['PatientID', 'SequenceName'], keep='first')
    df_path2name['SequenceName'] = df_path2name['SequenceName'].str.capitalize() # Just convention 
    df_path2name.to_csv(path_root_metadata/'Breast-Cancer-MRI-filepath_filename-mapping.csv', index=False)


    df_path2name = pd.read_csv(path_root_metadata/'Breast-Cancer-MRI-filepath_filename-mapping.csv')
    series = list(zip(df_path2name['SequenceName'], df_path2name['classic_path'])) # NOTE: Only working with TCIA download strategy 'classic_path' 

    # Validate 
    print("Number Series: ", len(series), "of 5034 (5034+127=5161) ")


    # Option 1: Multi-CPU 
    metadata_list = []
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series2nifti, series), total=len(series)):
            metadata_list.append(meta)


    # Option 2: Single-CPU (if you need a coffee break)
    # metadata_list = []
    # for series_info in tqdm(series):
    #     meta = series2nifti(series_info)
    #     metadata_list.append(meta)

    
    df = pd.DataFrame(metadata_list)
    df.to_csv(path_root_metadata/'metadata.csv', index=False)

    # Check export 
    num_series = len([path for path in path_root_out_data.rglob('*.nii.gz')])
    print("Number Series: ", num_series, "of 5034 (5034+127=5161) ")




