from odelia.models import BasicClassifier, BasicRegression
import monai.networks.nets as nets
import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange



def _get_resnet_monai(model):
    return {
        18: nets.resnet18, 34: nets.resnet34, 50: nets.resnet50, 101: nets.resnet101, 152: nets.resnet152
    }.get(model)

class _ResNet(nn.Module):
    def __init__(
        self, 
        in_ch, 
        out_ch=1, 
        spatial_dims=3,
        model = 34,
    ):
        super().__init__()
        Model = _get_resnet_monai(model)
        self.model = Model(n_input_channels=in_ch, spatial_dims=spatial_dims, num_classes=out_ch)

    def forward(self, x):
        return self.model(x)



class ResNet(BasicClassifier):
    def __init__(
        self, 
        in_ch, 
        out_ch=1, 
        spatial_dims=3,
        model = 34,
        **kwargs
    ):
        super().__init__(in_ch, out_ch, spatial_dims, **kwargs)
        self.model = _ResNet(in_ch, out_ch, spatial_dims, model)

    def forward(self, x):
        return self.model(x)



class ResNetRegression(BasicRegression):
    def __init__(
        self, 
        in_ch,
        out_ch=1, 
        spatial_dims=3,
        model = 34,
        **kwargs
    ):
        super().__init__(in_ch, out_ch, spatial_dims, **kwargs)
        self.model = _ResNet(in_ch, out_ch, spatial_dims, model)

    def forward(self, x):
        return self.model(x)
