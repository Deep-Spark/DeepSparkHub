"""Unit tests for custom dot operation"""
import numpy as np
import torch

from absl.testing import absltest
from dlrm.cuda_ext import dotBasedInteract

DECIMAL_MATRIX = 0
DECIMAL_LINEAR = 0
MAX_INT_VALUE = 1024  # clip integers larger than `MAX_INT_VALUE` (used in debugging only).
SEED = 12345
SCALE = 1  # Scale the random numbers to check different ranges.
PADDING_SIZE = 1
VERBOSE = False

np.random.seed(seed=SEED)


def log(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def elems_almost_equal(achived, reference, error_ratio):
    """Computes the relative error between to float values."""
    ratio = abs(achived - reference)
    if reference != 0:
        ratio = ratio / reference
    return ratio < error_ratio


def print_differences(arr, ref, error_ratio, verbose=False):
    """Prints values whose relative difference is larger than error_ratio."""
    if not verbose:
        return
    arr = arr.astype(float)
    ref = ref.astype(float)
    assert arr.shape == ref.shape
    if len(arr.shape) == 3:
        batch_size = arr.shape[0]
        num_rows = arr.shape[1]
        num_cols = arr.shape[2]
        for i in range(batch_size):
            for j in range(num_rows):
                for k in range(num_cols):
                    if not elems_almost_equal(arr[i][j][k], ref[i][j][k], error_ratio):
                        print(i, j, k, arr[i][j][k], ref[i][j][k])

    elif len(arr.shape) == 2:
        batch_size = arr.shape[0]
        num_cols = arr.shape[1]
        for i in range(batch_size):
            for k in range(num_cols):
                if not elems_almost_equal(arr[i][k], ref[i][k], error_ratio):
                    print(i, k, arr[i][k], ref[i][k])
    else:
        raise NotImplementedError


def dot_based_interact_test(num_rows,
                            num_cols,
                            batch_size,
                            synthesize_mode,
                            upstream_grad_synthesize_mode,
                            direction,
                            linear_output,
                            decimal,
                            max_value=MAX_INT_VALUE,
                            verbose=VERBOSE):
    """Computes the forward and backward for custom dot and checks the result."""
    # Input tensor configuration and initialization
    if synthesize_mode == 'seq':
        bottom_mlp_output_np = np.arange(batch_size * num_cols).reshape(batch_size, num_cols)
        bottom_mlp_output_np = bottom_mlp_output_np % max_value
        embedding_outputs_np = []
        for i in range(num_rows - 1):  # `num_rows` embedding and one MLP
            tmp = np.arange(batch_size * num_cols).reshape(batch_size, num_cols)
            tmp = tmp % max_value
            embedding_outputs_np.append(tmp)
    elif synthesize_mode == 'rand':
        bottom_mlp_output_np = np.random.randn(batch_size, num_cols)
        bottom_mlp_output_np = bottom_mlp_output_np * SCALE
        embedding_outputs_np = []
        for i in range(num_rows - 1):
            tmp = np.random.randn(batch_size, num_cols)
            tmp = tmp * SCALE
            embedding_outputs_np.append(tmp)
    elif synthesize_mode == 'ones':
        bottom_mlp_output_np = np.ones((batch_size, num_cols))
        embedding_outputs_np = []
        for i in range(num_rows - 1):
            tmp = np.ones((batch_size, num_cols))
            embedding_outputs_np.append(tmp)
    else:
        print('Invalid synthesize_mode {}'.format(synthesize_mode))
        raise NotImplementedError

    # Identical inputs for reference and test
    ref_bottom_mlp_output = torch.Tensor(bottom_mlp_output_np).half().cuda().requires_grad_()
    test_bottom_mlp_output = torch.Tensor(bottom_mlp_output_np).half().cuda().requires_grad_()

    ref_embedding_outputs = []
    test_embedding_outputs = []
    for elem in embedding_outputs_np:
        ref_embedding_outputs.append(torch.Tensor(elem).half().cuda().requires_grad_())
        test_embedding_outputs.append(torch.Tensor(elem).half().cuda().requires_grad_())

    assert ref_bottom_mlp_output.shape == test_bottom_mlp_output.shape
    assert ref_bottom_mlp_output.shape[0] == batch_size
    assert ref_bottom_mlp_output.shape[1] == num_cols

    assert ref_embedding_outputs[0].shape == test_embedding_outputs[0].shape
    assert len(ref_embedding_outputs) == len(test_embedding_outputs)
    assert len(ref_embedding_outputs) == num_rows - 1
    assert ref_embedding_outputs[0].shape[0] == batch_size
    assert ref_embedding_outputs[0].shape[1] == num_cols

    reference_input = torch.cat([ref_bottom_mlp_output] + ref_embedding_outputs, dim=1)
    test_input = torch.cat([test_bottom_mlp_output] + test_embedding_outputs, dim=1)

    reference_input = reference_input.view((batch_size, -1, num_cols))
    test_input = test_input.view((batch_size, -1, num_cols))

    assert reference_input.shape == test_input.shape
    assert reference_input.shape[0] == batch_size
    assert reference_input.shape[1] == num_rows
    assert reference_input.shape[2] == num_cols

    ref_pad = torch.zeros(batch_size, 1, dtype=ref_bottom_mlp_output.dtype, device=ref_bottom_mlp_output.device)

    # FWD path in reference
    interaction = torch.bmm(reference_input, torch.transpose(reference_input, 1, 2))
    tril_indices_row = [i for i in range(interaction.shape[1]) for j in range(i)]
    tril_indices_col = [j for i in range(interaction.shape[2]) for j in range(i)]
    interaction_flat = interaction[:, tril_indices_row, tril_indices_col]
    reference_output = torch.cat((ref_bottom_mlp_output, interaction_flat, ref_pad), dim=1)

    num_output_elems = (num_rows * (num_rows - 1) >> 1) + num_cols + PADDING_SIZE
    assert reference_output.shape[0] == batch_size
    assert reference_output.shape[1] == num_output_elems

    if linear_output:
        reference_output = torch.sum(reference_output, dim=1)
        reference_output = torch.sum(reference_output, dim=0)

    # New FWD path
    test_output = dotBasedInteract(test_input, test_bottom_mlp_output)

    if linear_output:
        test_output = torch.sum(test_output, dim=1)
        test_output = torch.sum(test_output, dim=0)

    assert test_output.shape == reference_output.shape
    # FWD path test
    if direction in ['fwd', "both"]:
        log(verbose, 'Starting FWD Test ...')
        print_differences(test_output.detach().cpu().numpy(), reference_output.detach().cpu().numpy(), decimal)
        np.testing.assert_almost_equal(test_output.detach().cpu().numpy(),
                                       desired=reference_output.detach().cpu().numpy(),
                                       decimal=decimal)
        log(verbose, 'FWD test ended successfully.')
    if direction == 'fwd':
        return

    # BWD path
    test_input.retain_grad()
    reference_input.retain_grad()
    if linear_output:
        reference_output.backward()
        test_output.backward()
    else:
        # Synthesize upstream gradient
        if upstream_grad_synthesize_mode == 'ones':
            upstream_grad = np.ones(reference_output.shape)
        elif upstream_grad_synthesize_mode == 'seq':
            upstream_grad = np.arange(reference_output.numel()).reshape(reference_output.shape)
            upstream_grad = upstream_grad % max_value
        elif upstream_grad_synthesize_mode == 'rand':
            upstream_grad = np.random.randn(reference_output.numel()).reshape(reference_output.shape)
            upstream_grad = upstream_grad * SCALE
        else:
            print('Invalid upstream_grad_synthesize_mode {}'.format(synthesize_mode))
            raise NotImplementedError

        reference_upstream_grad = torch.Tensor(upstream_grad).half().cuda()
        test_upstream_grad = torch.Tensor(upstream_grad).half().cuda()
        reference_output.backward(reference_upstream_grad)
        test_output.backward(test_upstream_grad)

        log(verbose, 'Starting BWD Test ...')
        print_differences(test_input.grad.detach().cpu().numpy(), reference_input.grad.detach().cpu().numpy(), decimal)
        print_differences(test_bottom_mlp_output.grad.detach().cpu().numpy(),
                          ref_bottom_mlp_output.grad.detach().cpu().numpy(), decimal)
        np.testing.assert_almost_equal(test_input.grad.detach().cpu().numpy(),
                                       desired=reference_input.grad.detach().cpu().numpy(),
                                       decimal=decimal)
        np.testing.assert_almost_equal(test_bottom_mlp_output.grad.detach().cpu().numpy(),
                                       desired=ref_bottom_mlp_output.grad.detach().cpu().numpy(),
                                       decimal=decimal)
        log(verbose, 'BWD test ended successfully.')


class AccuracyWithoutSelfInteraction(absltest.TestCase):
    """Unit tests for testing forward and backward precision of custom dot-based interact"""

    def test_dlrm_specific_matrix_output(self):
        "Test for matrix output."
        for num_rows in [27]:
            for num_cols in [128]:
                for batch_size in [1024, 2048, 8192, 16384]:
                    dot_based_interact_test(num_rows=num_rows,
                                            num_cols=num_cols,
                                            batch_size=batch_size,
                                            synthesize_mode='rand',
                                            upstream_grad_synthesize_mode='rand',
                                            direction='both',
                                            linear_output=False,
                                            decimal=DECIMAL_MATRIX)

    def test_dlrm_agnostic_matrix_output(self):
        "Test for matrix output."
        for num_rows in [6, 11, 22, 32]:
            for num_cols in [16, 17, 64, 120]:
                for batch_size in [2048]:
                    dot_based_interact_test(num_rows=num_rows,
                                            num_cols=num_cols,
                                            batch_size=batch_size,
                                            synthesize_mode='rand',
                                            upstream_grad_synthesize_mode='rand',
                                            direction='both',
                                            linear_output=False,
                                            decimal=DECIMAL_MATRIX)

    def test_dlrm_specific_linear_output(self):
        "Test for linear output."
        for num_rows in [27]:
            for num_cols in [128]:
                for batch_size in [2048]:
                    dot_based_interact_test(num_rows=num_rows,
                                            num_cols=num_cols,
                                            batch_size=batch_size,
                                            synthesize_mode='rand',
                                            upstream_grad_synthesize_mode='rand',
                                            direction='both',
                                            linear_output=True,
                                            decimal=DECIMAL_LINEAR)


if __name__ == '__main__':
    absltest.main()
