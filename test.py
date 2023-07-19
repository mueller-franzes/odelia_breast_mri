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

import torch

# Assuming your lists contain PyTorch tensors
gradcam_result_list = [torch.randn(1, 1, 1, 256, 256) for _ in range(10)]
gradcampp_result_list = [torch.randn(1, 1, 1, 256, 256) for _ in range(10)]
occ_result_list = [torch.randn(1, 1, 1, 256, 256) for _ in range(10)]

# Stack tensors along the third dimension
gradcam_result_tensor = torch.stack(gradcam_result_list, dim=3).squeeze(0)
gradcampp_result_tensor = torch.stack(gradcampp_result_list, dim=3).squeeze(0)
occ_result_tensor = torch.stack(occ_result_list, dim=3).squeeze(0)
print(gradcam_result_tensor.shape)
