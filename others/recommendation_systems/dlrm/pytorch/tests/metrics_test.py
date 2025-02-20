"""Tests for metrics"""
from absl.testing import absltest
from sklearn.metrics import roc_auc_score

import numpy as np

import torch

from dlrm.utils import metrics

# pylint:disable=missing-docstring, no-self-use

class AucTest(absltest.TestCase):

    def test_against_sklearn_exact(self):
        for num_samples in [100, 1000, 10000, 100000, 1048576]:
            y = np.random.randint(0, 2, num_samples)
            scores = np.random.power(10, num_samples)
            ref_auc = roc_auc_score(y, scores)
            test_auc = metrics.ref_roc_auc_score(y, scores)
            assert ref_auc == test_auc

    def test_against_sklearn_almost_exact(self):
        for num_samples in [100, 1000, 10000, 100000, 1048576]:
            y = np.random.randint(0, 2, num_samples)
            scores = np.random.power(10, num_samples)
            ref_auc = roc_auc_score(y, scores)
            test_auc = metrics.ref_roc_auc_score(y, scores, exact=False)
            np.testing.assert_almost_equal(ref_auc, test_auc)

    def test_pytorch_against_sklearn(self):
        for num_samples in [100, 1000, 10000, 100000, 1048576]:
            y = np.random.randint(0, 2, num_samples).astype(np.float32)
            scores = np.random.power(10, num_samples).astype(np.float32)
            ref_auc = roc_auc_score(y, scores)

            test_auc = metrics.roc_auc_score(torch.from_numpy(y).cuda(), torch.from_numpy(scores).cuda())
            np.testing.assert_almost_equal(ref_auc, test_auc.cpu().numpy())

if __name__ == '__main__':
    absltest.main()
