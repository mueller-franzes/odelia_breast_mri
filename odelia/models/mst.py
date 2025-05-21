
import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from x_transformers import Encoder

from .base_model import BasicClassifier, BasicRegression

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 

class TransformerEncoder(Encoder):
    def forward(self, 
            x, 
            mask=None, 
            src_key_padding_mask=None,
        ):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x=x, context=None, mask=src_key_padding_mask, context_mask=None, attn_mask=mask)


class _MST(nn.Module):
    def __init__(
        self, 
        out_ch=1, 
        backbone_type="dinov2",
        model_size = "s", # 34, 50, ... or 's', 'b', 'l'
        slice_fusion_type = "transformer", # transformer, linear, average, none 
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type == "resnet":
            Model = _get_resnet_torch(model_size)
            self.backbone = Model(weights="DEFAULT")
            emb_ch = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_type == "dinov2":
            self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            self.backbone.mask_token = None  # Remove - otherweise unused parameters error"
            emb_ch = self.backbone.num_features
        else:
            raise ValueError("Unknown backbone_type")


        self.emb_ch = emb_ch 
        if slice_fusion_type == "transformer":
            self.slice_fusion = TransformerEncoder(
                dim = emb_ch,
                heads = 12 if emb_ch%12 == 0 else 8,
                ff_mult = 1,
                attn_dropout=0.0,
                pre_norm = True,
                depth = 1,
                attn_flash = True,
                ff_no_bias = True, 
                rotary_pos_emb=True,
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        # elif slice_fusion_type == 'linear':
        #     self.slice_fusion = nn.Linear(num_slices, 1)
        elif slice_fusion_type == 'average':
            pass 
        elif slice_fusion_type == "none":
            pass 
        else:
            raise ValueError("Unknown slice_fusion_type")

        self.linear = nn.Linear(emb_ch, out_ch)



    def forward(self, x):
        B, *_ = x.shape
        x = rearrange(x, 'b c d h w -> (b c d) h w')
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB

        # -------------- Backbone --------------
        x = self.backbone(x)
        # x = checkpoint(self.backbone, x.requires_grad_())
        x = rearrange(x, '(b d) e -> b d e', b=B)
 
        # -------------- Slice Fusion --------------
        if self.slice_fusion_type == 'none':
            return x
        elif self.slice_fusion_type == 'transformer':
            x = torch.concat([x, self.cls_token.repeat(B, 1, 1)], dim=1) # [B, 1+D, E]
            x = self.slice_fusion(x) # [B, D+1, L]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b e d')
            x = self.slice_fusion(x) # ->  [B, E, 1]
            x = rearrange(x, 'b e d -> b d e') #  ->  [B, 1, E]
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=True) #  [B, D, E] ->  [B, 1, E]

        # -------------- Logits --------------
        x = self.linear(x[:, -1])
        return x
    

class MST(BasicClassifier):
    # MST - https://arxiv.org/abs/2411.15802 
    def __init__(
            self,
            in_ch=1, 
            out_ch=1, 
            spatial_dims=3,
            backbone_type="dinov2",
            model_size = "s", # 34, 50, ... or 's', 'b', 'l'
            slice_fusion_type = "transformer", # transformer, linear, average, none  
            optimizer_kwargs={'lr':1e-6}, 
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)  
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)
    
    def forward(self, x):
        return self.mst(x)


class MSTRegression(BasicRegression):
    def __init__(
            self,
            in_ch=1, 
            out_ch=1, 
            spatial_dims=3,
            backbone_type="dinov2",
            model_size = "s", # 34, 50, ... or 's', 'b', 'l'
            slice_fusion_type = "transformer", # transformer, linear, average, none  
            optimizer_kwargs={'lr':1e-6}, 
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)  
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)
    
    def forward(self, x):
        return self.mst(x)