#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
import os
import fnmatch
import torch
from odelia.models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet, UNet3D, ResNet2D

# Load the model checkpoint
checkpoint_path = "/mnt/sda1/Duke Compare/trained_models/Host_Sentinal/DensNet121/2023_04_05_130511_DUKE_DenseNet121_swarm_learning/epoch=28-step=11600.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Create an instance of the DenseNet model
model = DenseNet(in_ch=1, out_ch=1, spatial_dims=3, model_name = 'DenseNet121')
# Load the model weights from the checkpoint
model.load_state_dict(checkpoint['state_dict'])

# Count the number of parameters
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the result in million
print(f"Number of parameters: {num_parameters/1e6:.2f}M")

# read out the model site and print in mb
model_size = os.path.getsize(checkpoint_path)/1e6
print(f"Model size: {model_size:.2f}M")



