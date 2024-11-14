from odelia.models import BasicClassifier
import monai.networks.nets as nets
import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange



def _get_resnet_monai(model):
    return {
        18: nets.resnet18, 34: nets.resnet34, 50: nets.resnet50, 101: nets.resnet101, 152: nets.resnet152
    }.get(model)

class ResNet(BasicClassifier):
    def __init__(
        self, 
        in_ch, 
        out_ch=1, 
        spatial_dims=3,
        model = 34,
        resnet_kwargs = {},
        loss=torch.nn.BCEWithLogitsLoss, 
        loss_kwargs={}, 
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4}, 
        lr_scheduler=None, 
        lr_scheduler_kwargs={},
        aucroc_kwargs={"task":"binary"},
        acc_kwargs={"task":"binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        Model = _get_resnet_monai(model)
        self.model = Model(n_input_channels=in_ch, spatial_dims=spatial_dims, num_classes=out_ch, **resnet_kwargs)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor
