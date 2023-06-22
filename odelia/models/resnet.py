from odelia.models import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F
from torch.autograd import Function

class ResNet(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            block='basic',
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
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
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = nets.ResNet(block, layers, block_inplanes, spatial_dims, in_ch, 7, 1, False, 'B', 1.0, out_ch,
                                 feed_forward, True)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor

    # def training_step(self, batch, batch_idx):
    # source, target = batch['source'], batch['target']
    # y_hat = self(source)
    # loss = F.cross_entropy(y_hat, target)
    # return loss


class GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.model.eval()
        self.feature_grad = None
        self.feature_map = None
        self.hooks = []

        # Register hook on the selected layer
        self.hooks.append(self.feature_layer.register_forward_hook(self.save_feature_map))
        self.hooks.append(self.feature_layer.register_backward_hook(self.save_feature_grad))

    def save_feature_map(self, module, input, output):
        self.feature_map = output.detach()

    def save_feature_grad(self, module, grad_input, grad_output):
        self.feature_grad = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        """
        Args:
            x: input image tensor.
            class_idx (int): Index of target class. Default is the predicted class from the model.
        """
        self.model.zero_grad()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #x = x.to(device)
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax().item()

        target = output[:, class_idx]
        target.backward()

        grad_cam_map = (self.feature_grad * self.feature_map).mean(dim=[2, 3, 4],
                                                                   keepdim=True)  # Take the mean across the spatial dimensions
        grad_cam_map = F.relu(grad_cam_map)  # Apply ReLU to the grad_cam_map
        grad_cam_map = F.interpolate(grad_cam_map, size=x.shape[2:], mode="trilinear",
                                     align_corners=False)  # Upscale to the input size

        # Clear stored forward and backward hooks
        for hook in self.hooks:
            hook.remove()

        return grad_cam_map
