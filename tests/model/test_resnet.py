import torch 
from odelia.models import ResNet, ResNetRegression

input = torch.randn((1,1,32, 224,224))
model = ResNet(in_ch=1, out_ch=2, spatial_dims=3, model=18)
model = ResNetRegression(in_ch=1, out_ch=2+3, spatial_dims=3, loss_kwargs={"class_labels_num": [2,3]})

pred = model(input)
print(pred.shape)
print(pred)
