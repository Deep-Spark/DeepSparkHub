"""Tests for model class"""
from absl.testing import absltest

import torch

from dlrm import model

# pylint:disable=missing-docstring, no-self-use

_DUMMY_CONFIG = {
    "num_numerical_features" : 13,
    "categorical_feature_sizes" : [5, 7],
    "bottom_mlp_sizes" : [512, 256, 64, 3],
    "top_mlp_sizes" : [512, 256, 1],
    "embedding_dim": 3,
}

class DlrmTest(absltest.TestCase):

    def test_simple(self):
        # test creation
        test_model = model.Dlrm(**_DUMMY_CONFIG)
        test_model.set_devices("cuda")

        # Test forward
        test_numerical_input = torch.randn(2, 13, device="cuda")
        test_sparse_inputs = torch.tensor([[1, 1], [2, 2]], device="cuda")  # pylint:disable=not-callable
        test_model(test_numerical_input, test_sparse_inputs)

    def test_kaggle_criteo(self):
        """Test a real configuration stored in json
        It is not tiny so will take a while to create all the embedding tables
        """
        with open("dlrm/config/criteo_kaggle_tiny.json", "r") as jsonf:
            dlrm_criteo_kaggle = model.Dlrm.from_json(jsonf.read())
        dlrm_criteo_kaggle.cuda()
        print(dlrm_criteo_kaggle)

    def test_interaction(self):
        """Test interaction ops
        TODO(haow): It probably deserves more tests, especially the dot interaction
        """
        test_model = model.Dlrm(
            num_numerical_features=13,
            categorical_feature_sizes=range(2, 28),
            bottom_mlp_sizes=[128, 32],
            top_mlp_sizes=[256, 1],)

        # 26 sparse features + 13 dense feature with embedding size 32, plus padding 1
        assert test_model.top_mlp[0].in_features == 383 + 1

    def test_hash(self):
        # test creation
        test_model = model.Dlrm(**_DUMMY_CONFIG, hash_indices=True)
        test_model.set_devices("cuda")

        # Test forward
        ref_numerical_input = torch.randn(2, 13, device="cuda")
        ref_sparse_inputs = torch.tensor([[1, 2], [2, 3]], device="cuda")  # pylint:disable=not-callable
        ref = test_model(ref_numerical_input, ref_sparse_inputs)

        # Test indices that will be hashed to the same value as ref
        test_sparse_inputs = torch.tensor([[1, 7], [9, 3]], device="cuda")  # pylint:disable=not-callable
        test_result = test_model(ref_numerical_input, test_sparse_inputs)

        assert (ref == test_result).all()


class DlrmJointEmbeddingTest(absltest.TestCase):

    def test_against_base(self):
        torch.set_printoptions(precision=4, sci_mode=False)
        ref_model = model.Dlrm(**_DUMMY_CONFIG)
        test_model = model.DlrmJointEmbedding(**_DUMMY_CONFIG)
        ref_model.set_devices("cuda")
        test_model.to("cuda")

        # Copy model weight from ref_model
        test_model.embeddings[0].embedding.weight.data = torch.cat(
            [embedding.weight for embedding in ref_model.embeddings]).clone()
        test_module_dict = dict(test_model.named_modules())
        for name, module in ref_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                test_module_dict[name].weight.data.copy_(module.weight)
                test_module_dict[name].bias.data.copy_(module.bias)

        test_numerical_input = torch.randn(3, 13, device="cuda")
        test_sparse_inputs = torch.randint(0, 3, (2, 3), device="cuda")  # pylint:disable=not-callable

        ref_out = ref_model(test_numerical_input, test_sparse_inputs)
        test_out = test_model(test_numerical_input, test_sparse_inputs)
        assert (ref_out == test_out).all()

    def test_hash(self):
        # test creation
        test_model = model.DlrmJointEmbedding(**_DUMMY_CONFIG, hash_indices=True)
        test_model.to("cuda")

        # Test forward
        ref_numerical_input = torch.randn(2, 13, device="cuda")
        ref_sparse_inputs = torch.tensor([[1, 2], [2, 3]], device="cuda")  # pylint:disable=not-callable
        ref = test_model(ref_numerical_input, ref_sparse_inputs)

        # Test indices that will be hashed to the same value as ref
        test_sparse_inputs = torch.tensor([[1, 7], [9, 3]], device="cuda")  # pylint:disable=not-callable
        test_result = test_model(ref_numerical_input, test_sparse_inputs)

        assert (ref == test_result).all()

if __name__ == '__main__':
    absltest.main()
