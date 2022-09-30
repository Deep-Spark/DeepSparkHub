"""Tests for distributed model"""
from copy import copy
from absl.testing import absltest

import torch
from torch import nn

from dlrm import model
from dlrm import dist_model

# pylint:disable=missing-docstring, no-self-use

_DUMMY_BOTTOM_CONFIG = {
    "num_numerical_features" : 13,
    "categorical_feature_sizes" : [5, 7],
    "bottom_mlp_sizes" : [512, 256, 64, 5],
    "embedding_dim": 5,
}

_DUMMY_TOP_CONFIG = {
    "top_mlp_sizes" : [512, 256, 1],
    "num_interaction_inputs": 3,
    "embedding_dim": 5
}

class DlrmBottomTest(absltest.TestCase):

    def test_simple(self):
        # test creation
        test_model = dist_model.DlrmBottom(**_DUMMY_BOTTOM_CONFIG)

        # Test forward
        test_numerical_input = torch.randn(2, 13, device="cuda")
        test_sparse_inputs = torch.tensor([[1, 1], [2, 2]], device="cuda")  # pylint:disable=not-callable
        test_out = test_model(test_numerical_input, test_sparse_inputs)

    def test_empty_bottom_mlp(self):
        config = copy(_DUMMY_BOTTOM_CONFIG)
        config.pop('bottom_mlp_sizes')
        test_model = dist_model.DlrmBottom(**config)

        test_numerical_input = torch.randn(2, 13, device="cuda")
        test_sparse_inputs = torch.tensor([[1, 1], [2, 2]], device="cuda")  # pylint:disable=not-callable
        test_out = test_model(test_numerical_input, test_sparse_inputs)


class DlrmTopTest(absltest.TestCase):

    def test_simple(self):
        # test creation
        test_model = dist_model.DlrmTop(**_DUMMY_TOP_CONFIG).to("cuda")

        # Test forward
        test_bottom_output = torch.rand(2, 3, 5, device="cuda")
        test_model(test_bottom_output)

class DlrmBottomAndTopTest(absltest.TestCase):

    def test_against_base_model(self):
        model_config = copy(_DUMMY_BOTTOM_CONFIG)
        model_config.update(_DUMMY_TOP_CONFIG)
        model_config.pop('num_interaction_inputs')
        ref_model = model.DlrmJointEmbedding(**model_config)
        ref_model.to("cuda")

        test_model = dist_model.DistDlrm(**model_config)
        test_model.to("cuda")
        print(test_model)

        # Copy weight to make to models identical
        test_model.bottom_model.joint_embedding.embedding.weight.data.copy_(ref_model.embeddings[0].embedding.weight)
        for i in range(len(test_model.bottom_model.bottom_mlp)):
            if isinstance(ref_model.bottom_mlp[i], nn.Linear):
                test_model.bottom_model.bottom_mlp[i].weight.data.copy_(ref_model.bottom_mlp[i].weight)
                test_model.bottom_model.bottom_mlp[i].bias.data.copy_(ref_model.bottom_mlp[i].bias)
        for i in range(len(test_model.top_model.top_mlp)):
            if isinstance(ref_model.bottom_mlp[i], nn.Linear):
                test_model.top_model.top_mlp[i].weight.data.copy_(ref_model.top_mlp[i].weight)
                test_model.top_model.top_mlp[i].bias.data.copy_(ref_model.top_mlp[i].bias)

        test_numerical_input = torch.randn(2, 13, device="cuda")
        test_sparse_inputs = torch.tensor([[1, 1], [2, 2]], device="cuda")  # pylint:disable=not-callable
        test_top_out = test_model(test_numerical_input, test_sparse_inputs)
        ref_top_out = ref_model(test_numerical_input, test_sparse_inputs.t())
        assert (test_top_out == ref_top_out).all()

if __name__ == '__main__':
    absltest.main()
