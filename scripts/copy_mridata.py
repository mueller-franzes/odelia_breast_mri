#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
import shutil
import pandas as pd
# Define the source and destination directories
src_dir = "/mnt/sda1/Oliver/data"
dst_dir = "/mnt/sda1/Oliver/data_both"

merged_df = '/home/swarm/Downloads/merged_df.csv'
# Get the list of uids
merged_df = pd.read_csv(merged_df)
# set the uids if the NN is 1
merged_df = merged_df[merged_df['NN'] == 1]
uids = merged_df['uid'].tolist()

# For each uid...
for uid in uids:
    # Define the path to the source folder
    src_folder = os.path.join(src_dir, uid)

    # If the source folder exists...
    if os.path.exists(src_folder):
        # Define the path to the destination folder
        dst_folder = os.path.join(dst_dir, uid)

        # Copy the folder (and all of its contents) to the destination directory
        shutil.copytree(src_folder, dst_folder)
