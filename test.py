#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
import os
import fnmatch

target_string = "df = pd.DataFrame(data_list, columns=['Model', 'Category', 'AUC-ROC'])"
directory = "/home/swarm/PycharmProjects/"  # replace with the path to the directory you want to search

for root, dirnames, filenames in os.walk(directory):
    for filename in fnmatch.filter(filenames, "*.py"):
        filepath = os.path.join(root, filename)
        with open(filepath, "r") as f:
            contents = f.read()
            if target_string in contents:
                print(f"Found target string in {filepath}")
