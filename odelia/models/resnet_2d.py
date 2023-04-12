import torch
import torch.nn as nn
import torch.nn.functional as F
from odelia.models import BasicClassifier
from torchvision.models import resnet50


class ResNet2D(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            feed_forward=True,
            loss=torch.nn.BCEWithLogitsLoss,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-4},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={"task": "binary"},
            acc_kwargs={"task": "binary"}
    ):
        super().__init__(in_ch, out_ch, 2, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)

        # Create a 2D ResNet50 model
        self.model = resnet50()

        # Adjust the input and output layers to match the required number of channels
        self.model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_ch)

    def forward(self, x_in, **kwargs):
        # Assuming the input shape is (batch_size, in_ch, depth, height, width)
        # Extract 2D slices along the depth dimension
        slices = torch.unbind(x_in, dim=2)

        # Calculate the output for each 2D slice and store them in a list
        slice_outputs = []
        for slice_ in slices:
            slice_output = self.model(slice_)
            slice_outputs.append(slice_output)

        # Stack the outputs along the depth dimension
        pred_hor = torch.stack(slice_outputs, dim=2)

        return pred_hor
