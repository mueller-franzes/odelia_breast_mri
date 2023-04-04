# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from pathlib import Path
import random
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import RandFlip, EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, SplitDim, SplitChannel
from monai.metrics import ROCAUCMetric
from monai.apps import CrossValidation
import matplotlib.pyplot as plt

#from models import MILModel


def main():

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-exp','--exp', help='Experiment name', required=True)
    parser.add_argument('-gpu','--gpu', help='Index of the gpu that will be used for training, e.g. 0 or 1', default=0)
    parser.add_argument('-model','--model', help='Model architecture', required=True)
    parser.add_argument('-mil','--mil', help='Mil mode, i.e. mean, max, att, att_trans, att_trans_pyramid')
    parser.add_argument('-dim','--dim', help='Dimension of the input, i.e. 2d or 3d', default='3d')

    args = parser.parse_args()

    path_file = os.path.dirname(os.path.realpath(__file__))

    logs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'runs/{args.exp}/logs')
    checkpoints_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'runs/{args.exp}/checkpoints')

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)


    monai.config.print_config()
    logging.basicConfig(filename=os.path.join(logs_path, 'training.log'), 
                        level=logging.INFO)

    data_path = Path('/home/jeff/dataset/duke_3d/odelia_dataset_unilateral_256x256x32')

    patient_IDs =  list(set([x[0].split('/')[-1].split('_')[0] for x in os.walk(data_path)][1:]))

    # images = [[os.path.join(data_path, patient_ID + '_right', 'sub.nii.gz'),
    #           os.path.join(data_path, patient_ID + '_left', 'sub.nii.gz')]
    #           for patient_ID in patient_IDs] # contains a list of path to right and left breast image

    # images = [os.path.join(data_path, patient_ID + '_right', 'sub.nii.gz')
    #           for patient_ID in patient_IDs] # contains a list of path to right and left breast image

    # images = []
    # for patient_ID in sorted(patient_IDs):
    #     images.append(os.path.join(data_path, patient_ID + '_right', 'sub.nii.gz'))
    #     images.append(os.path.join(data_path, patient_ID + '_left', 'sub.nii.gz'))

    df = pd.read_excel(os.path.join(data_path.parent, 'Clinical_and_Other_Features.xlsx'), header=[0, 1, 2])
    df = df[df[df.columns[38]] == 0]  # Only use unilateral tumors, TODO: Include bilateral tumor
    df = df[[df.columns[0], df.columns[36]]]  # Only pick relevant columns: Patient ID, Tumor Side
    df.columns = ['PatientID', 'Location']  # Simplify columns as: Patient ID, Tumor Side

    df_temp = []
    df_temp = pd.DataFrame({
                'PatientID': df["PatientID"].str.split('_').str[2],
                'Malign_R': df["Location"].apply(lambda x: 1 if x == 'R' else 0),
                'Malign_L': df["Location"].apply(lambda x: 1 if x == 'L' else 0)})
    df = df_temp
    # 2 binary labels for gender classification: man and woman
    #labels = np.array(df['Malign'], dtype=np.int64)
    images = []
    labels = []
    for index, row in df.iterrows():
        #labels.append(np.array([row['Malign_R'], row['Malign_L']], dtype='int8')) # TODO change
        patient_ID = df['PatientID'][index]
        images.append(os.path.join(data_path, patient_ID + '_right', 'sub.nii.gz'))
        images.append(os.path.join(data_path, patient_ID + '_left', 'sub.nii.gz'))
        labels.append(row['Malign_R'])
        labels.append(row['Malign_L'])
    
    # shuffle
    #images_labels = list(zip(images, labels))
    #random.Random(135).shuffle(images_labels)
    #images, labels = zip(*images_labels)

    len_dataset = len(images)
    n_folds_cv = 5 # number of folds for cross-validation

    len_fold = len(images) // n_folds_cv

    end_idx_fold_1 = len_fold
    end_idx_fold_2 = len_fold * 2
    end_idx_fold_3 = len_fold * 3
    end_idx_fold_4 = len_fold * 4
    end_idx_fold_5 = len_fold * 5



    labels = np.expand_dims(np.array(labels, dtype='float32'), axis=1)

    # Define transforms
    if args.dim == '3d':
        train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandFlip(prob=0.5, spatial_axis=0), RandFlip(prob=0.5, spatial_axis=1)])
        val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])
    elif args.dim == '2d':
        train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandFlip(prob=0.5, spatial_axis=0), RandFlip(prob=0.5, spatial_axis=1)])
        val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

    # Define image dataset, data loader
    # check_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms)
    # check_loader = DataLoader(check_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())
    # im, label = monai.utils.misc.first(check_loader)
    # print(type(im), im.shape, label)

    train_idx_end = int(len(labels)*0.7)
    val_idx_end = int(len(labels)*0.85)
    print(train_idx_end, val_idx_end)

    # create a training data loader
    train_ds = ImageDataset(image_files=images[:train_idx_end], labels=labels[:train_idx_end], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = ImageDataset(image_files=images[train_idx_end:val_idx_end], labels=labels[train_idx_end:val_idx_end], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=1, pin_memory=torch.cuda.is_available())

    # create a test data loader
    test_ds = ImageDataset(image_files=images[val_idx_end:], labels=labels[val_idx_end:], transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=1, pin_memory=torch.cuda.is_available())


    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'training on {device}')
    if args.model == 'dense':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    elif args.model == 'mil':
        model = MILModel(num_classes=1, mil_mode=args.mil, pretrained=False).to(device)
    elif args.model == 'swin':
        model = monai.networks.nets.SwinUNETR(
                                            img_size=(256, 256, 32),
                                            in_channels=1,
                                            out_channels=1,
                                            feature_size=48,
                                            use_checkpoint=True).to(device)
        # model.decoder1 = torch.nn.Identity()
        # model.decoder2 = torch.nn.Identity()
        # model.decoder3 = torch.nn.Identity()
        # model.decoder4 = torch.nn.Identity()
        # model.decoder5 = torch.nn.Identity()
        # model.out = torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d(output_size=1),
        #                                 torch.nn.Flatten(start_dim=1, end_dim=-1),
        #                                 torch.nn.Linear(768, 1)) # only use encoder as classsifier
        # model.to(device)
        # model = torch.nn.Sequential(*(list(model.children())[:6]),
        #                             torch.nn.AdaptiveAvgPool3d(output_size=1),
        #                             torch.nn.Flatten(start_dim=1, end_dim=-1),
        #                             torch.nn.Linear(768, 1)).to(device) # only use encoder as classsifier

    # model = monai.networks.nets.ResNet(block='basic',
    #                                    layers=[3, 4, 6, 3], 
    #                                    block_inplanes=[64, 128, 256, 512], 
    #                                    spatial_dims=3, 
    #                                    n_input_channels=1, 
    #                                    conv1_t_size=7, 
    #                                    conv1_t_stride=1, 
    #                                    no_max_pool=False, 
    #                                    shortcut_type='B', 
    #                                    widen_factor=1.0, 
    #                                    num_classes=1,
    #                                    feed_forward=True, 
    #                                    bias_downsample=True)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    auc_metric = ROCAUCMetric()


    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    auc_best_val = -1
    auc_best_test = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(log_dir=logs_path)
    for epoch in range(200):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            shape_img_batch = batch_data[0].shape
            if args.dim == '3d': # TODO change dataloader list_data_collate
                #inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                inputs = torch.reshape(batch_data[0], 
                                       (shape_img_batch[0], shape_img_batch[1],
                                       shape_img_batch[4], shape_img_batch[3],
                                       shape_img_batch[2])).to(device)
            elif args.dim == '2d':
                inputs = torch.reshape(batch_data[0], 
                                       (shape_img_batch[0], shape_img_batch[-1],
                                       shape_img_batch[1], shape_img_batch[2],
                                       shape_img_batch[3])).to(device)

            labels = batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                val_all_outputs = torch.tensor([], dtype=torch.float32, device=device)
                val_all_labels = torch.tensor([], dtype=torch.float32, device=device)
                for val_data in val_loader:
                    shape_img_batch = val_data[0].shape
                    if args.dim == '3d':  # TODO change dataloader list_data_collate
                        # inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                        val_images = torch.reshape(val_data[0],
                                               (shape_img_batch[0], shape_img_batch[1],
                                                shape_img_batch[4], shape_img_batch[3],
                                                shape_img_batch[2])).to(device)
                    elif args.dim == '2d':
                        val_images = torch.reshape(val_data[0],
                                               (shape_img_batch[0], shape_img_batch[-1],
                                                shape_img_batch[1], shape_img_batch[2],
                                                shape_img_batch[3])).to(device)


                    val_labels = val_data[1].to(device)
                    val_outputs = torch.sigmoid(model(val_images)) # output value between 0 and 1
                    val_all_outputs = torch.cat([val_all_outputs, val_outputs], dim=0)
                    val_all_labels = torch.cat([val_all_labels, val_labels], dim=0)

                    value = torch.eq(torch.round(val_outputs), val_labels)
                    #value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()

                auc_metric(val_all_outputs, val_all_labels)
                auc = auc_metric.aggregate()
                auc_metric.reset()

                test_all_outputs = torch.tensor([], dtype=torch.float32, device=device)
                test_all_labels = torch.tensor([], dtype=torch.float32, device=device)
                for test_data in test_loader:
                    shape_img_batch = test_data[0].shape
                    if args.dim == '3d':  # TODO change dataloader list_data_collate
                        # inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                        test_images = torch.reshape(test_data[0],
                                               (shape_img_batch[0], shape_img_batch[1],
                                                shape_img_batch[4], shape_img_batch[3],
                                                shape_img_batch[2])).to(device)
                    elif args.dim == '2d':
                        test_images = torch.reshape(test_data[0],
                                               (shape_img_batch[0], shape_img_batch[-1],
                                                shape_img_batch[1], shape_img_batch[2],
                                                shape_img_batch[3])).to(device)
                    test_labels = test_data[1].to(device)
                    test_outputs = torch.sigmoid(model(test_images))
                    test_all_outputs = torch.cat([test_all_outputs, test_outputs], dim=0)
                    test_all_labels = torch.cat([test_all_labels, test_labels], dim=0)
                auc_metric(test_all_outputs, test_all_labels)
                auc_test = auc_metric.aggregate()
                auc_metric.reset()
                writer.add_scalar("test AUC", auc_test, epoch + 1)


                #del y_pred_act, y_onehot
                metric_values.append(auc)
                accuracy = num_correct / metric_count
                if auc > auc_best_val:
                    auc_best_val = auc
                    auc_best_test = auc_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(checkpoints_path, "best_model_{args.exp}.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best AUC on val: {:.4f} at epoch {}".format(
                        epoch + 1, auc, auc_best_val, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", accuracy, epoch + 1)
                writer.add_scalar("val_auc", auc, epoch + 1)


    print(f"train completed, best AUC on val: \n{auc_best_val:.4f}\non test:\n{auc_best_test:.4f}\n at epoch: {best_metric_epoch}")
    with open(os.path.join(logs_path, 'results.txt'), 'w') as f:
        f.write(f"train completed, best AUC on val: \n{auc_best_val:.4f}\non test:\n{auc_best_test:.4f}\n at epoch: {best_metric_epoch}")
    
    
    writer.close()

    best_model = torch.load(os.path.join(checkpoints_path, "best_model_{args.exp}.pth"))
    best_model.eval()
    with torch.no_grad():
        val_all_outputs = torch.tensor([], dtype=torch.float32, device=device)
        val_all_labels = torch.tensor([], dtype=torch.float32, device=device)
        for val_data in val_loader:
            shape_img_batch = val_data[0].shape
            val_images = torch.reshape(val_data[0], 
                                (shape_img_batch[0], shape_img_batch[-1],
                                shape_img_batch[1], shape_img_batch[2],
                                shape_img_batch[3])).to(device)

            val_labels = val_data[1].to(device)
            val_outputs = torch.sigmoid(best_model(val_images)) # output value between 0 and 1
            val_all_outputs = torch.cat([val_all_outputs, val_outputs], dim=0)
            val_all_labels = torch.cat([val_all_labels, val_labels], dim=0)
            # TODO resume here; add test set, save as csv, 
        


if __name__ == "__main__":
    main()
