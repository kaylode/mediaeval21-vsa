from typing import Dict, List, Any, Optional
import timm
import torch
import copy
import torch.nn as nn
from .embedding import FeatureEmbedding, SpatialEncoding
from .utils import init_xavier
from theseus.utilities.cuda import move_to

TIMM_MODELS = [
        "deit_tiny_distilled_patch16_224", 
        'deit_small_distilled_patch16_224', 
        'deit_base_distilled_patch16_224',
        'deit_base_distilled_patch16_384']

def get_clones(module, N):
    """
    "Produce N identical layers."
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


TIMM_MODELS = ["vit_base_patch16_384"]

def get_pretrained_encoder(model_name, **kwargs):
    assert model_name in TIMM_MODELS, "Timm Model not found"
    model = timm.create_model(model_name, pretrained=False, **kwargs)
    return model

class MetaEncoder(nn.Module):
    """
    Core encoder is a stack of N EncoderLayers
    :input:
        feat_dim:       feature dim
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, d_model):
        super().__init__()
        self.feat_embed = FeatureEmbedding(2048, d_model)
        self.face_embed = FeatureEmbedding(512, d_model)
        self.loc_embed = SpatialEncoding(d_model)
    def forward(self, src, spatial_src, facial_src):
        x = self.feat_embed(src)
        x_face = self.face_embed(facial_src)
        spatial_x = self.loc_embed(spatial_src)
        x += spatial_x
        x = torch.cat([x, x_face], dim=1)
        return x

class MetaVIT(nn.Module):
    """
    Pretrained Transformers Encoder from timm Vision Transformers
    :output:
        encoded embeddings shape [batch * (image_size/patch_size)**2 * model_dim]
    """
    def __init__(
        self, 
        model_name='vit_base_patch16_384', 
        num_classes: int = 1000,
        classnames: Optional[List] = None):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        vit = get_pretrained_encoder(model_name)
        self.cls_token = vit.cls_token
        self.embed_dim = vit.embed_dim 
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.pre_logits = vit.pre_logits
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.encoder = MetaEncoder(768)

        init_xavier(self)
        
    def forward(self, batch, device: torch.device):

        facial_feats = move_to(batch['facial_feats'], device)
        det_feats = move_to(batch['det_feats'], device)
        loc_feats = move_to(batch['loc_feats'], device)
        x = self.encoder(det_feats, loc_feats, facial_feats)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        return {
            'outputs': x,
        }

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.

        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.forward(adict, device)

        probs, outputs = torch.max(torch.softmax(outputs, dim=1), dim=1)

        probs = probs.cpu().detach().numpy()
        classids = outputs.cpu().detach().numpy()

        if self.classnames:
            classnames = [self.classnames[int(clsid)] for clsid in classids]
        else:
            classnames = []

        return {
            'labels': classids,
            'confidences': probs, 
            'names': classnames,
        }