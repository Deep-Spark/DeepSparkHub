"""Customized embedding gather"""
import copy

import torch
from torch.autograd import Function
from torch import nn

from apex import amp

from dlrm import cuda_ext

__all__ = ["EmbeddingGatherFunction", "JointSparseEmbedding", "embedding_gather"]

class EmbeddingGatherFunction(Function):
    """Customized embedding gather with fused plain SGD"""
    @staticmethod
    def forward(ctx, embedding, indices):
        output = cuda_ext.gather_gpu_fwd(embedding, indices)
        ctx.save_for_backward(indices)
        ctx.num_features = embedding.size(0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]

        grad_embedding = cuda_ext.gather_gpu_bwd(grad_output, indices, ctx.num_features)

        return grad_embedding, None

class JointSparseEmbedding(nn.Module):
    """Joint multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing).

    Args:
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        embedding_dim (int): the size of each embedding vector
        device (torch.device): where to create the embedding. Default "cuda"
    """
    def __init__(self, categorical_feature_sizes, embedding_dim, device="cuda"):
        super(JointSparseEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        self.register_buffer("offsets", torch.tensor([0] + categorical_feature_sizes).cumsum(0).to(device))
        self.weights = torch.nn.Parameter(torch.rand((self.offsets[-1].item(), embedding_dim), device=device))

    def forward(self, categorical_inputs):
        # Check input has the right shape
        assert categorical_inputs.shape[1] == len(self.categorical_feature_sizes)

        embedding_out = embedding_gather(self.weights, categorical_inputs + self.offsets[:-1])

        return embedding_out

    def extra_repr(self):
        s = F"categorical_feature_sizes={self.categorical_feature_sizes}\n"
        s += F"offsets={self.offsets.cpu().numpy()}"
        return s

embedding_gather = amp.float_function(EmbeddingGatherFunction.apply)
