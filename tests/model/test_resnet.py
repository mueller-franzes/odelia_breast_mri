import torch 
from odelia.models import ResNet

input = torch.randn((1,1,32, 224,224))
model = ResNet(in_ch=1, out_ch=2, spatial_dims=3, model=18)


pred = model(input)
print(pred.shape)
print(pred)
