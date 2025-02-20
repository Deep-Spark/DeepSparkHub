"""Buckle Embedding"""

import copy

import torch
from torch import nn

__all__ = ["BuckleEmbedding"]

class BuckleEmbedding(nn.Module):
    """Buckle multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing). Use nn.Embedding to deal with sparse wgrad
    before I fully customizing it.

    Args:
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        embedding_dim (int): the size of each embedding vector
        device (torch.device): where to create the embedding. Default "cuda"
    """
    def __init__(self, categorical_feature_sizes, embedding_dim, device="cuda"):
        super(BuckleEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        self.register_buffer("offsets", torch.tensor([0] + categorical_feature_sizes, device=device).cumsum(0))

        embedding_weight = torch.empty((self.offsets[-1].item(), embedding_dim), device=device)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False, sparse=True)

    # pylint:disable=missing-docstring
    def forward(self, categorical_inputs):
        # Check input has the right shape
        assert categorical_inputs.shape[1] == len(self.categorical_feature_sizes)

        embedding_out = self.embedding(categorical_inputs + self.offsets[:-1])

        return embedding_out

    def extra_repr(self):
        s = F"offsets={self.offsets.cpu().numpy()}"
        return s
    # pylint:enable=missing-docstring
