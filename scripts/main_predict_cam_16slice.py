import math
import os
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import monai

from odelia.data.datasets import DUKE_Dataset3D_external, DUKE_Dataset3D_collab
from odelia.data.datamodules import DataModule
from odelia.models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet, UNet3D, \
    ResNet2D
from odelia.utils.roc_curve import plot_roc_curve, cm2acc, cm2x

import argparse
import matplotlib.pyplot as plt
import monai
import numpy as np
import torch


# model = ...
# in_t = ...
# model.zero_grad()
# in_t = in_t.detach()
# in_t.requires_grad = True
# out_t = model(in_t)
# out_t[10].backward()
# gradcam = (in_t.grad * in_t).abs()

data_path = '/mnt/sda1/Oliver/data_best_athens/fp_redo'
import matplotlib.gridspec as gridspec
def get_next_im(itera):
    test_data = next(itera)
    return test_data[0].to(device), test_data[1].unsqueeze(0).to(device)


def plot_occlusion_heatmap(im, heatmap):
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(im.cpu()))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default=None, help='Path to the run directory')
    parser.add_argument('--path_out', default=None, help='Path to the output directory')
    parser.add_argument('--network', default=None, help='')

    args = parser.parse_args()

    # ------------ Settings/Defaults ----------------
    if args.path_run:
        path_run = Path(args.path_run)
        args.network = str(path_run).split('_')[-1]
        if len(args.network) == 2:
            args.network = 'efficientnet_' + args.network
        print(args.network)
    else:
        path_run = Path(
            '/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user/data-and-scratch/scratch/2024_03_18_173324_multi_ext_ResNet101_swarm_learning')
        args.network = str(path_run).split('_')[-3]
        if len(args.network) == 2:
            args.network = 'efficientnet_' + args.network
        # args.network="ResNet50"
        print(args.network)
    if args.path_out:
        path_out = Path(args.path_out)
    else:
        path_out = Path().cwd() / 'results' / path_run.name
        print(path_out)
    path_out = Path(data_path)
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}

    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    # ./workspace/automate_scripts/launch_sl/run_swop.sh -w <workspace_name> -s <sentinel_ip_address>  -d <host_index>
    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D_collab(
        flip=False,
        path_root=data_path
    )

    # WARNING: Very simple split approach
    ds_test = ds

    dm = DataModule(
        ds_test=ds_test,
        batch_size=1,
        # num_workers=0,
        # pin_memory=True,
    )
    # pack = monai.utils.misc.first(dm.test_dataloader())
    # print(pack)
    # print(type(im), im.shape, label, label.shape)
    # args.network = 'ResNet101'

    if args.network == 'ResNet18':
        layers = [2, 2, 2, 2]
    elif args.network == 'ResNet34':
        layers = [3, 4, 6, 3]
    elif args.network == 'ResNet50':
        layers = [3, 4, 6, 3]
    elif args.network == 'ResNet101':
        layers = [3, 4, 23, 3]
    elif args.network == 'ResNet152':
        layers = [3, 8, 36, 3]
    else:
        layers = None
    # print(layers)
    if layers is not None:
        # ------------ Initialize Model ------------
        model = ResNet.load_best_checkpoint(path_run, version=0, layers=layers, out_ch=1)
        # print('1212')
    elif args.network == 'ResNet2D':
        model = ResNet2D(in_ch=1, out_ch=1)
    elif args.network in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
        model = EfficientNet.load_best_checkpoint(path_run, version=0, model_name=args.network)
    elif args.network == 'EfficientNet3Db0':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25"]
    elif args.network == 'EfficientNet3Db4':
        blocks_args_str = [
            "r1_k3_s11_e1_i48_o24_se0.25",
            "r3_k3_s22_e6_i24_o32_se0.25",
            "r3_k5_s22_e6_i32_o56_se0.25",
            "r4_k3_s22_e6_i56_o112_se0.25",
            "r4_k5_s11_e6_i112_o160_se0.25",
            "r5_k5_s22_e6_i160_o272_se0.25",
            "r2_k3_s11_e6_i272_o448_se0.25"]
    elif args.network == 'EfficientNet3Db7':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o32_se0.25",
            "r4_k3_s22_e6_i32_o48_se0.25",
            "r4_k5_s22_e6_i48_o80_se0.25",
            "r4_k3_s22_e6_i80_o160_se0.25",
            "r6_k5_s11_e6_i160_o256_se0.25",
            "r6_k5_s22_e6_i256_o384_se0.25",
            "r3_k3_s11_e6_i384_o640_se0.25"]
    elif args.network in ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264']:
        model = DenseNet.load_best_checkpoint(path_run, in_ch=1, out_ch=1, spatial_dims=3, model_name=args.network)
    elif args.network == 'UNet3D':
        model = UNet3D.load_best_checkpoint(path_run, in_ch=1, out_ch=1, spatial_dims=3)
    else:
        raise Exception("Invalid network model specified")

    if args.network.startswith('EfficientNet3D'):
        model = EfficientNet3D.load_best_checkpoint(path_run, version=0, blocks_args_str=blocks_args_str)
    # print(model)
    print(args.network)
    model.to(device)
    model.eval()

    itera = iter(dm.test_dataloader())


    def boolean_to_onehot(tensor):
        tensor = tensor.long()  # Convert boolean tensor to long type
        num_classes = 2  # The number of classes, True and False, hence 2
        one_hot = torch.nn.functional.one_hot(tensor, num_classes)  # Create a one-hot tensor
        return one_hot.float()  # Return the tensor as float


    def get_next_im():
        test_data = next(itera)
        # print(test_data)
        # encode test_data['target'].unsqueeze(0) into one-hot
        # print(test_data['target'].unsqueeze(0))
        target = test_data['target']
        one_hot_target = boolean_to_onehot(target)
        return test_data['source'].to(device), one_hot_target, test_data['uid']


    def plot_occlusion_heatmap(im, heatmap):
        plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(im.cpu()))
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.colorbar()
        plt.show()


    results = {'uid': [], 'GT': [], 'NN': [], 'NN_pred': []}
    # print(model)
    # print model layer names and shapes
    target_layers_list = []
    for name, param in model.named_parameters():
        print(name, param.shape)
        if 'layer' in name:
            if '3.0' in name:
                name = ('.').join(name.split('.')[:3])
                if name not in target_layers_list:
                    target_layers_list.append(name)
    for name in target_layers_list:
        # print(name, param.shape)
        target_layers = name
        gradcam = monai.visualize.GradCAM(nn_module=model, target_layers='model.layer3.1')
        # cam = monai.visualize.CAM(nn_module=model, target_layers=target_layers, fc_layers='model.fc')
        gradcampp = monai.visualize.GradCAMpp(nn_module=model, target_layers='model.layer3.0')
        occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=8, overlap=0.25, n_batch=1, mode='mean_img')
        for batch in tqdm(dm.test_dataloader()):

            ############################
            target = batch['target']
            one_hot_target = boolean_to_onehot(target)
            img, label, uid = batch['source'].to(device), one_hot_target, batch['uid']
            print(img.shape, label)
            # depth_slice = img.shape[2] // 2
            gradcam_result_list = []
            gradcampp_result_list = []
            occ_result_list = []
            start_slice = 0
            end_slice = 31
            img_orig = img

            start = 5
            end = 27
            num_slices = 16

            # Create an array with evenly spaced values within the specified range
            slice_indices_array = np.linspace(start, end, num_slices, dtype=int)
            print('slice_indices', slice_indices_array)
            # Convert the numpy array to a PyTorch tensor
            slice_indices_tensor = torch.tensor(slice_indices_array)
            print('slice_indices', slice_indices_tensor)
            # Select the slices from the tensor
            slice_indices = slice_indices_tensor.to(img.device)
            print('slice_indices', slice_indices)
            img = img.index_select(2, slice_indices)
            max_occ_result=0
            min_occ_result=1
            for depth_slice in slice_indices_array:
                print('depth_slice', depth_slice)
                # run CAM

                gradcam_result = gradcam(x=img_orig, class_idx=None)
                gradcampp_result = gradcampp(x=img_orig, class_idx=None)
                print('cam_result.shape', gradcam_result.shape)

                #print('original cam_result.shape', cam_result.shape)

                gradcam_result = gradcam_result[:, :, depth_slice, :, :]
                gradcampp_result = gradcampp_result[:, :, depth_slice, :, :]


                occ_sens_b_box = [depth_slice, depth_slice+1, -1, -1, -1, -1]
                occ_result, _ = occ_sens(x=img_orig, b_box=occ_sens_b_box, mode='mean_img')
                occ_result = occ_result[0, 0][None]
                #occ_result = 1 - (occ_result - occ_result.min()) / (occ_result.max() - occ_result.min())
                #gradcam_result = 1 - (gradcam_result - gradcam_result.min()) / (
                            #gradcam_result.max() - gradcam_result.min())
                #gradcampp_result = 1 - (gradcampp_result - gradcampp_result.min()) / (
                            #gradcampp_result.max() - gradcampp_result.min())
                #print('cam_result.shape after getting depth slice', cam_result.shape)
                occ_result = 1 - occ_result
                gradcam_result = -1 * gradcam_result
                gradcampp_result = -1 * gradcampp_result
                gradcam_result_list.append(gradcam_result)
                gradcampp_result_list.append(gradcampp_result)

                max_occ_result = occ_result.max() if max_occ_result < occ_result.max() else max_occ_result
                min_occ_result = occ_result.min() if min_occ_result > occ_result.min() else min_occ_result

                occ_result_list.append(occ_result)
            # calculate mean of cam_result_list
            #gradcam_result = torch.mean(torch.stack(gradcam_result_list), dim=0)
            #gradcampp_result = torch.mean(torch.stack(gradcampp_result_list), dim=0)
            #invert the color map for visualization,


            gradcam_result_tensor = torch.stack(gradcam_result_list, dim=2)
            gradcampp_result_tensor = torch.stack(gradcampp_result_list, dim=2)
            occ_result_tensor = torch.stack(occ_result_list, dim=2)

            #print('max_occ_result_tensor', max_occ_result_tensor)
            #print('min_occ_result_tensor', min_occ_result_tensor)

            print('gradcam_result_tensor.shape', gradcam_result_tensor.shape)
            print('gradcampp_result_tensor.shape', gradcampp_result_tensor.shape)
            print('occ_result_tensor.shape', occ_result_tensor.shape)
            # get mean of img
            #print('img.shape', img.shape)
            #img = torch.mean(img, dim=2, keepdim=True)
            #print('img.shape after taking mean', img.shape)
            #n_examples = 2

            # create a figure
            fig = plt.figure(figsize=(10, 47), facecolor="white")  # More square aspect ratio

            # define gridspec
            gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1])  # Equal height for each row

            #pick out slices in img from start to end with step size 2

            nrows = 4
            ncols = 4  # Adjust this to change how many images are displayed per row
            start_slice = 0
            end_slice = 16
            for row, (im, title) in enumerate(
                    zip(
                        [img, gradcam_result_tensor, gradcampp_result_tensor, occ_result_tensor],
                        [name, "GradCam", "GradCam++", "OCA"],
                    )
            ):
                cmap = "gray" if row == 0 else "jet"
                if row == 3:
                    vmin = min_occ_result
                    vmax = max_occ_result
                elif row != 0:
                    vmin = -1
                    vmax = 0
                else:
                    vmin = None
                    vmax = None

                if isinstance(im, torch.Tensor):
                    im = im.cpu().detach()

                print('im', im.shape)
                if row == 0:
                    # plot all the slices for img
                    title = 'original series'
                    # define a sub gridspec


                sub_gs = gs[row].subgridspec(nrows, ncols, wspace=0.01, hspace=0.01)  # reduced spacing
                for slice_index in range(start_slice, end_slice):
                    i = ((slice_index - start_slice)) // ncols
                    j = ((slice_index - start_slice))% ncols
                    ax = fig.add_subplot(sub_gs[i, j])
                    ax.imshow(im[0, 0, slice_index], cmap=cmap, vmin=vmin, vmax=vmax)
                    #ax.set_title(title, fontsize=25)
                    ax.axis("off")
            # save cam_result
            plt.savefig(f"{path_out}/{uid}_{depth_slice}_{target_layers}_visall.svg", bbox_inches="tight", pad_inches=0)
            print('saved cam_result to ', f"{path_out}/{uid}_16slice_eval_{target_layers}_visall.png")