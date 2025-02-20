"""Time different component of the model
A lot of codes are copied from train.py, but breaking them out is still easier for timing.
"""
import os
from time import time

from absl import app
from absl import flags
from absl import logging

from apex import amp

import torch

from dlrm.data import ref_terabyte_loader
from dlrm.model import Dlrm, DlrmJointEmbedding

from utils import timer_start, timer_stop

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16384, "")
flags.DEFINE_string("model_config", None, "json file of model configuration")
flags.DEFINE_boolean("joint_embedding", False, "Combine embedding tables to single one")
flags.DEFINE_boolean("fp16", True, "Use fp16")
flags.DEFINE_string(
    "dataset", None,
    "Full path to reference binary dataset. Must have filename, train_data.bin ... day_fea_count.npz")

flags.DEFINE_string("base_device", "cuda", "Device to run majority the model.")
flags.DEFINE_list("mem_donor_devices", None, "Devices to place the 5 largest embedding tables of Criteo.")

flags.DEFINE_integer("num_iters", "100", "Number of iterations to time")
flags.DEFINE_boolean("bottom_mlp", True, "Time bottom mlp")
flags.DEFINE_boolean("top_mlp", True, "Time top mlp")
flags.DEFINE_boolean("interaction", True, "Time interaction")
flags.DEFINE_boolean("embeddings", True, "Time embeddings")
flags.DEFINE_boolean("misc", True, "Time BCE loss and optimizer and maybe others")

flags.mark_flags_as_required(["dataset", "model_config"])


def time_mlp(mlp, input):
    """
    Args:
        mlp (ModuleList):
        input (Tensor):

    Returns:
        fwd_elapsed_time (float): FWD time in ms
        bwd_elapsed_time (float): BWD time in ms
    """
    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        mlp_output = mlp(input)
    stop_time = timer_stop()

    fwd_elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3

    grad = torch.rand_like(mlp_output)
    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        mlp_output.backward(grad, retain_graph=True)
    stop_time = timer_stop()

    bwd_elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3
    return fwd_elapsed_time, bwd_elapsed_time

def time_embeddings(model, input):
    """

    Args:
        model (Dlrm):
        input (Tensor): with shape [num_categorical_features, batch_size]

    Returns:
        fwd_elapsed_time (float): FWD time in ms
        bwd_elapsed_time (float): BWD time in ms
    """
    # Put indices on the same device as corresponding embedding
    device_indices = []
    if not FLAGS.joint_embedding:
        for embedding_id, embedding in enumerate(model.embeddings):
            device_indices.append(input[embedding_id].to(model._embedding_device_map[embedding_id]))
    else:
        device_indices.append(input.t())

    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        for embedding_id, embedding in enumerate(model.embeddings):
            embedding(device_indices[embedding_id]).to(FLAGS.base_device)
    stop_time = timer_stop()
    fwd_elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3

    # Run a untimed path to collect output of embeddings
    model.zero_grad()
    embedding_outputs = []
    for embedding_id, embedding in enumerate(model.embeddings):
        embedding_outputs.append(embedding(device_indices[embedding_id]).to(FLAGS.base_device))

    concat_output = torch.cat(embedding_outputs)
    grad = torch.rand_like(concat_output)

    logging.info("Backward of embedding seems to be pure memcpyD2D.")
    bwd_elapsed_time = 0
    for _ in range(FLAGS.num_iters):
        start_time = timer_start()
        concat_output.backward(grad, retain_graph=True)
        stop_time = timer_stop()
        model.zero_grad()  # Sparse gradient will keep aggregating if not cleared
        bwd_elapsed_time += (stop_time - start_time) * 1e3
    bwd_elapsed_time /= FLAGS.num_iters

    return fwd_elapsed_time, bwd_elapsed_time

def time_interaction(interaction, bottom_mlp_output, embedding_outputs, batch_size):
    """
    Args:
        interaction (function):
        bottom_mlp_output (Tensor):
        embedding_outputs (list): Sequence of tensors
        batch_size (int):

    Returns:
        fwd_elapsed_time (float): FWD time in ms
        bwd_elapsed_time (float): BWD time in ms
    """
    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        interaction_output = interaction(bottom_mlp_output, embedding_outputs, batch_size)
    stop_time = timer_stop()
    fwd_elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3

    dummy_grad = torch.rand_like(interaction_output)
    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        interaction_output.backward(dummy_grad, retain_graph=True)
    stop_time = timer_stop()
    bwd_elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3

    return fwd_elapsed_time, bwd_elapsed_time

def time_loss(loss_fn):
    dummy_out = torch.rand(FLAGS.batch_size, device=FLAGS.base_device, requires_grad=True)
    dummy_label = torch.rand(FLAGS.batch_size, device=FLAGS.base_device)
    if FLAGS.fp16:
        dummy_out = dummy_out.half()
        dummy_label = dummy_label.half()

    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        loss = loss_fn(dummy_out, dummy_label)
        loss.backward()
    stop_time = timer_stop()

    elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3
    return elapsed_time

def time_optimizer(optimizer):
    start_time = timer_start()
    for _ in range(FLAGS.num_iters):
        optimizer.step()
    stop_time = timer_stop()
    elapsed_time = (stop_time - start_time) / FLAGS.num_iters * 1e3
    return elapsed_time

def main(argv):
    print("Creating model")
    with open(FLAGS.model_config, "r") as f:
        if not FLAGS.joint_embedding:
            model = Dlrm.from_json(f.read(), base_device=FLAGS.base_device, mem_donor_devices=FLAGS.mem_donor_devices)
            model.set_devices(FLAGS.base_device, FLAGS.mem_donor_devices)
        else:
            model = DlrmJointEmbedding.from_json(f.read())
            model.to(FLAGS.base_device)
    print(model)

    # Read one batch of data
    dataset_path = os.path.join(FLAGS.dataset, "train_data.bin")
    dataset = ref_terabyte_loader.CriteoBinDataset(
        data_file=dataset_path, batch_size=FLAGS.batch_size)
    numerical_features, categorical_features, click = ref_terabyte_loader.data_collate_fn(dataset[0], FLAGS.base_device)

    loss_fn = torch.nn.BCELoss()
    # Time it before initializing AMP
    if FLAGS.misc:
        bce_time = time_loss(loss_fn)

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    dummy_top_mlp_input = torch.rand(FLAGS.batch_size, model.top_mlp[0].in_features, device=FLAGS.base_device)
    if FLAGS.fp16:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch.nn.functional, 'embedding')
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1)
        numerical_features = numerical_features.half()
        dummy_top_mlp_input = dummy_top_mlp_input.half()

    # Warm up to mitigate some overhead
    model_out = model(numerical_features, categorical_features)
    model.zero_grad()
    loss = loss_fn(model_out, click)
    loss.backward()

    if FLAGS.misc:
        optimizer_time = time_optimizer(optimizer)

    print("Operation   | fwd (ms) | bwd (ms)")
    print("---------------------------------")
    if FLAGS.interaction:
        bottom_mlp_output = model.bottom_mlp(numerical_features).clone()
        bottom_mlp_output.requires_grad_()
        dummy_embedding_outputs = []
        # Overwrite padding to aovid type mismatch
        model._interaction_padding = torch.zeros(
            FLAGS.batch_size, 1, dtype=bottom_mlp_output.dtype, device=bottom_mlp_output.device)
        for embedding in model.embeddings:
            dtype = torch.float32 if not FLAGS.fp16 else torch.half
            dummy_embedding_outputs.append(
                torch.rand(FLAGS.batch_size, embedding.embedding_dim, device=FLAGS.base_device, dtype=dtype))
        fwd_interaction_time, bwd_interaction_time = time_interaction(
            model._interaction, bottom_mlp_output, dummy_embedding_outputs, FLAGS.batch_size)
        print(F"interaction |  {fwd_interaction_time:.4f}  |  {bwd_interaction_time:.4f}")

    if FLAGS.bottom_mlp:
        fwd_bottom_mlp_time, bwd_bottom_mlp_time = time_mlp(model.bottom_mlp, numerical_features)
        print(F"bottom_mlp  |  {fwd_bottom_mlp_time:.4f}  |  {bwd_bottom_mlp_time:.4f}")

    if FLAGS.top_mlp:
        fwd_top_mlp_time, bwd_top_mlp_time = time_mlp(model.top_mlp, dummy_top_mlp_input)
        print(F"top_mlp     |  {fwd_top_mlp_time:.4f}  |  {bwd_top_mlp_time:.4f}")

    if FLAGS.embeddings:
        fwd_embedding_time, bwd_embedding_time = time_embeddings(model, categorical_features)
        print(F"embedding   |  {fwd_embedding_time:.4f}  |  {bwd_embedding_time:.4f}")

    if FLAGS.misc:
        print(F"bce loss    |  {bce_time:.4f}")
        print(F"Optimizer   |  {optimizer_time:.4f}")

if __name__ == '__main__':
    app.run(main)
