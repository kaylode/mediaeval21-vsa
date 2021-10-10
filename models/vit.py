import torch
import torch.nn as nn

from vit_pytorch import ViT
from vit_pytorch.twins_svt import TwinsSVT

def get_model(name, num_classes):
    if name == 'twinssvt':
        model = TwinsSVT(
            num_classes = num_classes,       # number of output classes
            s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
            s1_patch_size = 4,        # stage 1 - patch size for patch embedding
            s1_local_patch_size = 7,  # stage 1 - patch size for local attention
            s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
            s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
            s2_emb_dim = 128,         # stage 2 (same as above)
            s2_patch_size = 2,
            s2_local_patch_size = 7,
            s2_global_k = 7,
            s2_depth = 1,
            s3_emb_dim = 256,         # stage 3 (same as above)
            s3_patch_size = 2,
            s3_local_patch_size = 7,
            s3_global_k = 7,
            s3_depth = 5,
            s4_emb_dim = 512,         # stage 4 (same as above)
            s4_patch_size = 2,
            s4_local_patch_size = 7,
            s4_global_k = 7,
            s4_depth = 4,
            peg_kernel_size = 3,      # positional encoding generator kernel size
            dropout = 0.              # dropout
        )
    
    if name == 'vit':
        model = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    return model


class TransformerBasedModel(nn.Module):
    def __init__(self, name, num_classes):
        model = get_model(name, num_classes)
        self.model = nn.DataParallel(model)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs = self.model(inputs)
        return outputs