import torch
import numpy as np

def make_feature_batch(features,  pad_token=0):
    """
    List of features,
    each feature is [K, model_dim] where K is number of objects of each image
    This function pad max length to each feature and tensorize, also return the masks
    """

    # Find maximum length
    max_len=0
    for feat in features:
        feat_len = feat.shape[0]
        max_len = max(max_len, feat_len)
    
    # Init batch
    batch_size = len(features)
    batch = np.ones((batch_size, max_len, 512))
    batch *= pad_token
        
    # Copy data to batch
    for i, feat in enumerate(features):
        feat_len = feat.shape[0]
        batch[i, :feat_len] = feat

    batch = torch.from_numpy(batch).type(torch.float32)
    return batch  