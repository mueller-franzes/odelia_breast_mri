from odelia.models import BasicClassifier
import monai.networks.nets as nets
import torch 
import torch.nn.functional as F


class ResNet(BasicClassifier):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        spatial_dims=3,
        block='basic',
        layers=[3, 8, 36, 3],
        block_inplanes=[64, 128, 256, 512],
        feed_forward=True,
        loss=torch.nn.BCEWithLogitsLoss, 
        loss_kwargs={}, 
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-3},
        lr_scheduler=None, 
        lr_scheduler_kwargs={},
        aucroc_kwargs={"task":"binary"},
        acc_kwargs={"task":"binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = nets.ResNet(block, layers, block_inplanes, spatial_dims, in_ch, 7, 1, False, 'B', 1.0, out_ch, feed_forward, True)
   
    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor

    #def training_step(self, batch, batch_idx):
        #source, target = batch['source'], batch['target']
        #y_hat = self(source)
        #loss = F.cross_entropy(y_hat, target)
        #return loss
