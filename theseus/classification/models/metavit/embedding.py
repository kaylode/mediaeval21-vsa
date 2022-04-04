import torch.nn as nn

class FeatureEmbedding(nn.Module):
    """
    Projects image features into a space of
    dimensionality `embed_dim`.
    """

    def __init__(self, features_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(features_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class SpatialEncoding(nn.Module):
    """
    Encodes bounding box coordinates and relative sizes
    as vector of dimensionality `embed_dim`.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(5, embed_dim)

    def forward(self, x):
        return self.linear(x)