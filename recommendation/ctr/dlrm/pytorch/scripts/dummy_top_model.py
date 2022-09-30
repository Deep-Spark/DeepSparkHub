from time import time

from absl import app
from absl import flags

import torch
from torch import nn

from apex import amp
from apex import parallel

from dlrm.cuda_ext import dotBasedInteract
from dlrm.utils import distributed as dist

import utils

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16384, "")
flags.DEFINE_integer("num_iters", "10", "Number of iterations to time")
flags.DEFINE_boolean("fp16", True, "Use fp16")
flags.DEFINE_boolean("bias_relu", True, "If False, skip bias and ReLU")
flags.DEFINE_list("mlp_sizes", [480, 1024, 1024, 512, 256, 1], "MLP sizes")

EMBED_DIM = 128

def create_top_mlp():
    mlp_sizes = [int(size) for size in FLAGS.mlp_sizes]
    top_mlp_layers = []
    input_dims = mlp_sizes[0]
    for output_dims in mlp_sizes[1:-1]:
        top_mlp_layers.append(nn.Linear(input_dims, output_dims, bias=FLAGS.bias_relu))
        if FLAGS.bias_relu:
            top_mlp_layers.append(nn.ReLU(inplace=True))
        input_dims = output_dims
    # last Linear layer uses sigmoid
    top_mlp_layers.append(nn.Linear(input_dims, mlp_sizes[-1]))
    top_mlp_layers.append(nn.Sigmoid())

    return nn.Sequential(*top_mlp_layers)

def dot_interaction(bottom_mlp_output, embedding_outputs, batch_size):
    concat = torch.cat([bottom_mlp_output] + embedding_outputs, dim=1)
    concat = concat.view((batch_size, -1, EMBED_DIM))
    if FLAGS.fp16:
        interaction_output = dotBasedInteract(concat, bottom_mlp_output)
    else:
        interaction = torch.bmm(concat, torch.transpose(concat, 1, 2))
        tril_indices_row, tril_indices_col = torch.tril_indices(
            interaction.shape[1], interaction.shape[2], offset=-1)
        interaction_flat = interaction[:, tril_indices_row, tril_indices_col]

        padding = torch.empty(FLAGS.batch_size, 1, device="cuda")
        # concatenate dense features and interactions
        interaction_output = torch.cat([bottom_mlp_output] + [interaction_flat, padding], dim=1)


    return interaction_output

def main(argv):
    rank, world_size, gpu = dist.init_distributed_mode()

    top_mlp = create_top_mlp().to("cuda")
    print(top_mlp)

    optimizer = torch.optim.SGD(top_mlp.parameters(), lr=1.)

    if FLAGS.fp16:
        top_mlp, optimizer = amp.initialize(top_mlp, optimizer, opt_level="O1", loss_scale=1)

    if world_size > 1:
        top_mlp = parallel.DistributedDataParallel(top_mlp)
        model_without_ddp = top_mlp.module

    dummy_bottom_mlp_output = torch.rand(FLAGS.batch_size, EMBED_DIM, device="cuda")
    dummy_embedding_output = torch.rand(FLAGS.batch_size, 26 * EMBED_DIM, device="cuda")
    dummy_target = torch.ones(FLAGS.batch_size, device="cuda")

    if FLAGS.fp16:
        dummy_bottom_mlp_output = dummy_bottom_mlp_output.to(torch.half)
        dummy_embedding_output = dummy_embedding_output.to(torch.half)

    # warm up GPU
    for _ in range(100):
        interaction_out = dot_interaction(dummy_bottom_mlp_output, [dummy_embedding_output], FLAGS.batch_size)
        output = top_mlp(interaction_out)

    start_time = utils.timer_start()
    for _ in range(FLAGS.num_iters):
        interaction_out = dot_interaction(dummy_bottom_mlp_output, [dummy_embedding_output], FLAGS.batch_size)
        output = top_mlp(interaction_out).squeeze()
        dummy_loss = output.mean()
        optimizer.zero_grad()
        if FLAGS.fp16:
            with amp.scale_loss(dummy_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            dummy_loss.backward()
        optimizer.step()
    stop_time = utils.timer_stop()

    elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3
    print(F"Average step time: {elapsed_time:.4f} ms.")

if __name__ == '__main__':
    app.run(main)
