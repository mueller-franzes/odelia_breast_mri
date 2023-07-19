#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import json
import logging
import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import fire


class DICOMConverter:
    def __init__(self, root_path_in, root_path_out, log_file_path, excel_path):
        """
        Initialize the DICOMConverter with user-defined paths and other necessary attributes.

        Args:
            root_path_in (str): The root input directory path.
            root_path_out (str): The root output directory path.
            log_file_path (str): The log file path.
            excel_path (str): The Excel file path.
        """
        self.path_root_in = Path(root_path_in)
        self.path_root_out = Path(root_path_out)
        self.path_log_file = Path(log_file_path)
        self.excel_path = Path(excel_path)

        # Create directories if not exist
        self.path_root_out.mkdir(parents=True, exist_ok=True)

        # Extract these DICOM keys and save them as a separate JSON file.
        self.metadatakeys = ['AcquisitionMatrix', 'AccessionNumber', 'AcquisitionDate', 'AcquisitionDuration',
                             'AcquisitionTime', 'Allergies', 'Columns', 'DeviceSerialNumber', 'EchoNumbers',
                             'EchoTime', 'EchoTrainLength', 'FlipAngle', 'ImagingFrequency',
                             'InPlanePhaseEncodingDirection', 'InstitutionName', 'MRAcquisitionType',
                             'MagneticFieldStrength', 'Manufacturer', 'ManufacturerModelName', 'Modality',
                             'NumberOfAverages', 'NumberOfPhaseEncodingSteps', 'NumberOfSlices', 'PatientAge',
                             'PatientBirthDate', 'PatientID', 'PatientName', 'PatientSex', 'PatientWeight',
                             'ParallelAcquisitionTechnique', 'PercentPhaseFieldOfView', 'PercentSampling',
                             'PerformingPhysicianName', 'PulseSequenceName', 'PregnancyStatus', 'ProtocolName',
                             'ReceiveCoilName', 'ReferringPhysicianName', 'RepetitionTime', 'RequestedProcedureID',
                             'Rows', 'PixelSpacing', 'SOPInstanceUID', 'SpacingBetweenSlices', 'ScanningSequence',
                             'SequenceName', 'SequenceVariant', 'SeriesDescription', 'SeriesInstanceUID',
                             'SeriesNumber', 'SliceThickness', 'StudyID', 'StudyInstanceUID']

        self.set_logger()
        self.df_path2name = self.read_excel()

        self.reader = sitk.ImageSeriesReader()

    def set_logger(self):
        """
        Set the logger for the script.
        """
        logger = logging.getLogger(__name__)
        s_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(self.path_log_file, 'w')
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[s_handler, f_handler])

    def read_excel(self):
        """
        Read the Excel file and return the data frame.
        """
        df_path2name = pd.read_excel(self.excel_path)
        seq_paths = df_path2name['original_path_and_filename'].str.split('/')
        df_path2name['UID'] = seq_paths.apply(lambda x: int(x[1].rsplit('_', 1)[1]))
        df_path2name['SequenceName'] = seq_paths.apply(lambda x: x[2])
        df_path2name['SeriesInstanceUID'] = df_path2name['classic_path'].str.split('/').apply(
            lambda x: x[3])  # StudyInstanceUID/SeriesInstanceUID
        df_path2name = df_path2name.drop_duplicates(subset=['UID', 'SequenceName'], keep='first')
        df_path2name[['UID', 'SequenceName', 'SeriesInstanceUID']].to_csv(self.path_root_in / 'filepath2sequence.csv',
                                                                          index=False)

        return df_path2name

    def process_case(self, path_dir):
        """
        Process a single case directory.

        Args:
            path_dir (Path): The case directory path.
        """
        pass  # Add your processing logic here.

    def process_all_cases(self):
        """
        Process all case directories in the root input directory.
        """
        for case_i, path_dir in enumerate(tqdm(list(self.path_root_in.iterdir()))):
            if path_dir.is_dir():
                self.process_case(path_dir)

    def main(self):
        """
        The main method of the script. It serves as the entry point.
        """
        self.process_all_cases()


if __name__ == "__main__":
    fire.Fire(DICOMConverter)
