"""Tests for buckle embedding"""
from absl.testing import absltest

import torch
from torch import nn

from dlrm.nn import BuckleEmbedding

# pylint:disable=missing-docstring, no-self-use

class DistEmbeddingBagTest(absltest.TestCase):

    def test_smoke(self):
        test_buckle_embedding = BuckleEmbedding([3, 5, 7, 11], 3, device="cpu")
        test_buckle_embedding(torch.tensor([[1, 2, 3, 4], [2, 4, 6, 10]]))

    def test_2embeddings_batch1(self):
        test_sizes = [3, 5]

        test_buckle_embedding = BuckleEmbedding(test_sizes, 3, device="cpu")
        ref_embeddings = nn.ModuleList()
        for size in test_sizes:
            ref_embeddings.append(nn.Embedding(size, 3))

        test_buckle_embedding.embedding.weight.data = torch.cat(
            [embedding.weight for embedding in ref_embeddings]).clone()

        test_indices = torch.tensor([[1, 3]])
        embedding_out = test_buckle_embedding(test_indices)
        ref_out = []
        for embedding_id, embedding in enumerate(ref_embeddings):
            ref_out.append(embedding(test_indices[:, embedding_id]))
        ref_out = torch.cat(ref_out)
        assert (ref_out == embedding_out).all()

    def test_4embeddings_batch2(self):
        test_sizes = [3, 5, 11, 13]

        test_buckle_embedding = BuckleEmbedding(test_sizes, 3, device="cpu")
        ref_embeddings = nn.ModuleList()
        for size in test_sizes:
            ref_embeddings.append(nn.Embedding(size, 3))

        test_buckle_embedding.embedding.weight.data = torch.cat(
            [embedding.weight for embedding in ref_embeddings]).clone()

        test_indices = torch.tensor([[1, 3, 5, 7], [2, 4, 10, 12]])
        embedding_out = test_buckle_embedding(test_indices)
        ref_out = []
        for embedding_id, embedding in enumerate(ref_embeddings):
            ref_out.append(embedding(test_indices[:, embedding_id].unsqueeze(-1)))
        ref_out = torch.cat(ref_out, dim=1)
        assert (ref_out == embedding_out).all()


if __name__ == '__main__':
    absltest.main()
