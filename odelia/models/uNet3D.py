from monai.networks.nets import UNet
from odelia.models import BasicClassifier
import torch

class UNet3D(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            init_features=64,
            layers=4,
            loss=torch.nn.BCEWithLogitsLoss,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-4},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={"task": "binary"},
            acc_kwargs={"task": "binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=[init_features * 2 ** i for i in range(layers)],
            strides=[2 ** i for i in range(layers)],
            num_res_units=2,
        )
        self.adaptive_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = torch.nn.Linear(out_ch, 1)

    def forward(self, x_in, **kwargs):
        x = self.model(x_in)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        pred_hor = self.fc(x)
        return pred_hor
