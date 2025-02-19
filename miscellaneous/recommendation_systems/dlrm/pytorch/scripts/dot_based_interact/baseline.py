"""Baseline benchmark for custom dot operation"""
import sys

import numpy as np
import torch
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16384, "Batch Size")

PADDING_SIZE = 1


def dot_based_interact_benchmark(num_rows, num_cols, batch_size, num_iterations=50):
    """Computes the fwd and bwd for custom dot and measures the time."""
    # Bottom MLP Output
    bottom_mlp_output_np = np.random.randn(batch_size, num_cols)
    ref_bottom_mlp_output = torch.Tensor(bottom_mlp_output_np).half().cuda().requires_grad_()

    # Embedding
    embedding_outputs_np = []
    for i in range(num_rows - 1):
        tmp = np.random.randn(batch_size, num_cols)
        embedding_outputs_np.append(tmp)

    ref_embedding_outputs = []
    for elem in embedding_outputs_np:
        ref_embedding_outputs.append(torch.Tensor(elem).half().cuda().requires_grad_())

    reference_input = torch.cat([ref_bottom_mlp_output] + ref_embedding_outputs, dim=1)
    reference_input = reference_input.view((batch_size, -1, num_cols))
    output_size = int(num_rows * (num_rows - 1) / 2) + num_cols + PADDING_SIZE
    reference_output = torch.Tensor(np.zeros((1, output_size))).half().cuda()

    # Create timer components
    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    # FWD path in reference
    torch.cuda.synchronize()
    start_fwd.record()
    ref_pad = torch.zeros(batch_size, 1, dtype=ref_bottom_mlp_output.dtype, device=ref_bottom_mlp_output.device)
    for count in range(num_iterations):
        reference_input = reference_input + count
        interaction = torch.bmm(reference_input, torch.transpose(reference_input, 1, 2))
        tril_indices_row = [i for i in range(interaction.shape[1]) for j in range(i)]
        tril_indices_col = [j for i in range(interaction.shape[2]) for j in range(i)]
        interaction_flat = interaction[:, tril_indices_row, tril_indices_col]
        reference_output = reference_output + torch.cat((ref_bottom_mlp_output, interaction_flat, ref_pad), dim=1)
    end_fwd.record()
    torch.cuda.synchronize()
    fwd_elapsed_time = (start_fwd.elapsed_time(end_fwd)) / num_iterations

    # Synthesize the upstream grad
    upstream_grad = np.random.randn(reference_output.numel()).reshape(reference_output.shape)
    upstream_grad = torch.Tensor(upstream_grad).cuda()

    torch.cuda.synchronize()
    start_bwd.record()
    reference_output.backward(upstream_grad)
    end_bwd.record()
    torch.cuda.synchronize()
    bwd_elapsed_time = (start_bwd.elapsed_time(end_bwd)) / num_iterations
    fwd_elapsed_time = round(fwd_elapsed_time, 2)
    bwd_elapsed_time = round(bwd_elapsed_time, 2)
    print('Baseline - batch size: {}, forward: {} ms, backward {} ms.'.format(batch_size, fwd_elapsed_time,
                                                                              bwd_elapsed_time))


def main(argv):
    batch_size = FLAGS.batch_size
    dot_based_interact_benchmark(num_rows=27, num_cols=128, batch_size=batch_size)


if __name__ == '__main__':
    app.run(main)
